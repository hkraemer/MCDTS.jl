using Pkg
current_dir = pwd()
Pkg.activate(current_dir)

using DelayEmbeddings
using DynamicalSystemsBase
using Random

"""
    give_potential_delays(Ys::Dataset, τs, w::Int, τ_vals,
                    ts_vals, L_old [; kwargs...] → τ_pot, ts_pot, L_pot, flag
Compute the potential delay `τ_pot` and time series values `ts_pot`, which would
each result in a potential L-statistic value `L_pot`, by using the PECUZAL
embedding method and for a range of possible delay values `τs`. The input
dataset `Ys` can be multivariate. `w` is the Theiler window (neighbors in time
with index `w` close to the point, that are excluded from being true neighbors.
`w=0` means to exclude only the point itself, and no temporal neighbors. In case
of multivariate time series input choose `w` as the maximum of all `wᵢ's`.

## Keyword arguments

* `samplesize::Real = 1`: determine the fraction of all phase space points
  (=`length(s)`) to be considered (fiducial points v) to average ε★ to produce
  `⟨ε★⟩`.
* `K::Int = 13`: the amount of nearest neighbors in the δ-ball (read algorithm description).
   Must be at least 8 (in order to gurantee a valid statistic). `⟨ε★⟩` is computed
   taking the minimum result over all `k ∈ K` (read algorithm description).
* `KNN::Int = 3`: the amount of nearest neighbors considered, in order to compute
   σ_k^2 (read algorithm description [`uzal_cost`]@ref). If given a vector, minimum
   result over all `knn ∈ KNN` is returned.
* `Tw::Int = 4*w`: the maximal considered time horizon for obtaining σ_k^2 (read
   algorithm description [`uzal_cost`]@ref).
* `α::Real = 0.05`: The significance level for obtaining the continuity statistic
* `p::Real = 0.5`: The p-parameter for the binomial distribution used for the
  computation of the continuity statistic ⟨ε★⟩.
"""
function give_potential_delays(Ys::Dataset, τs, w::Int, τ_vals, ts_vals, L_old;
                samplesize::Real=1, K::Int = 13, α::Real = 0.05, p::Real = 0.5,
                Tw::Int = 4*w, KNN::Int = 3)
    metric = Euclidean()
    # compute Y_act
    Y_act = genembed(Ys, τ_vals, ts_vals)
    # compute potential delay values with corresponding time series values and
    # L-statistic-values
    τ_pots, ts_pots, L_pots = embedding_cycle_pecuzal(Y_act, Ys, τs, w, samplesize,
                            K, metric, α, p, Tw, KNN, τ_vals, ts_vals)

    # transform array of arrays to a single array
    τ_pot = τ_pots[1]
    ts_pot = ts_pots[1]
    L_pot = L_pots[1]

    if length(τ_pots) >1
        for i = 2:length(τ_pots)
            τ_pot = vcat(τ_pot, τ_pots[i])
            ts_pot = vcat(ts_pot, ts_pots[i])
            L_pot = vcat(L_pot, L_pots[i])
        end
    end

    if minimum(L_pot) ≥ L_old
        flag = true
    else
        flag = false
    end
    return τ_pot, ts_pot, L_pot, flag
end


"""
Perform a potential embedding cycle from the multi- or univariate Dataset `Ys`.
Return the possible delays `τ_pot`, the associated time series `ts_pot` and
the corresponding L-statistic-values, `L_pot` for each peak, i.e. for each
(`τ_pot`, `ts_pot`) pair.
"""
function embedding_cycle_pecuzal(Y_act, Ys, τs, w, samplesize,
                    K, metric, α, p, Tw, KNN, τ_vals, ts_vals)

    ε★, _ = pecora(Ys, Tuple(τ_vals), Tuple(ts_vals); delays = τs, w = w,
            samplesize = samplesize, K = K, metric = metric, α = α,
            p = p, undersampling = false)
    # update τ_vals, ts_vals, Ls, ε★s
    τ_pot, ts_pot, L_pot = pick_possible_embedding_params(ε★, Y_act, Ys, τs, Tw,
                                                    KNN, w, samplesize, metric)

    return τ_pot, ts_pot, L_pot
end


"""
    Compute all possible τ-values (and according time series numbers) and their
corresponding L-statistics for the input continuity statistic ε★.
"""
function pick_possible_embedding_params(ε★, Y_act, Ys, τs, Tw, KNN, w,
                                                            samplesize, metric)
    L_pots = []
    τ_pots = []
    ts_pots = []

    for ts = 1:size(Ys,2)
        # zero-padding of ⟨ε★⟩ in order to also cover τ=0 (important for the multivariate case)
        # get the L-statistic for each peak in ⟨ε★⟩
        L_trials, max_idx, _ = local_L_statistics(vec([0; ε★[:,ts]]), Y_act, Ys[:,ts],
                                        τs, Tw, KNN, w, samplesize, metric)
        push!(L_pots, L_trials)
        push!(τ_pots, τs[max_idx.-1])
        push!(ts_pots, fill(ts,length(L_trials)))
    end
    return τ_pots, ts_pots, L_pots
end

"""
Return the L-statistic `L` and indices `max_idx` and weighted peak height
`ξ = peak-height * L` for all local maxima in ε★
"""
function local_L_statistics(ε★, Y_act, s, τs, Tw, KNN, w, samplesize, metric)
    maxima, max_idx = get_maxima(ε★) # determine local maxima in ⟨ε★⟩
    L_trials = zeros(Float64, length(max_idx))
    ξ_trials = zeros(Float64, length(max_idx))
    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        Y_trial = DelayEmbeddings.hcat_lagged_values(Y_act,s,τs[τ_idx-1])
        # compute L-statistic
        L_trials[i] = uzal_cost(Y_trial; Tw = Tw, K = KNN, w = w,
                samplesize = samplesize, metric = metric)
        ξ_trials[i] = L_trials[i]*maxima[i]
    end
    return L_trials, max_idx, ξ_trials
end


"""
Return the local maxima of the given time series s and its indices
"""
function get_maxima(s::Vector{T}) where {T}
    maximas = T[]
    maximas_idx = Int[]
    N = length(s)
    flag = false
    first_point = 0
    for i = 2:N-1
        if s[i-1] < s[i] && s[i+1] < s[i]
            flag = false
            push!(maximas, s[i])
            push!(maximas_idx, i)
        end
        # handling constant values
        if flag
            if s[i+1] < s[first_point]
                flag = false
                push!(maximas, s[first_point])
                push!(maximas_idx, first_point)
            elseif s[i+1] > s[first_point]
                flag = false
            end
        end
        if s[i-1] < s[i] && s[i+1] == s[i]
            flag = true
            first_point = i
        end
    end
    # make sure there is no empty vector returned
    if isempty(maximas)
        maximas, maximas_idx = findmax(s)
    end
    return maximas, maximas_idx
end
