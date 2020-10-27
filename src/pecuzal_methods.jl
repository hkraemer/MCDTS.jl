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
* `FNN:Bool = false`: Determines whether the algorithm should minimize the L-statistic
  or the FNN-statistic
"""
function give_potential_delays(Yss::Dataset, τs, w::Int, τ_vals, ts_vals, L_old;
                samplesize::Real=1, K::Int = 13, α::Real = 0.05, p::Real = 0.5,
                Tw::Int = 4*w, KNN::Int = 3, FNN::Bool = false)
    metric = Euclidean()
    Ys = regularize(Yss)
    # compute Y_act
    Y_act = genembed(Ys, τ_vals, ts_vals)

    # compute potential delay values with corresponding time series values and
    # L-statistic-values (or FNN-statistic-values, these will be binded in
    # `L_pots` for simplicity, anyway)
    τ_pots, ts_pots, L_pots = embedding_cycle_pecuzal(Y_act, Ys, τs, w, samplesize,
                            K, metric, α, p, Tw, KNN, τ_vals, ts_vals, FNN)

    # transform array of arrays to a single array
    τ_pot = reduce(vcat, τ_pots)
    ts_pot = reduce(vcat, ts_pots)
    L_pot = reduce(vcat, L_pots)

    if minimum(L_pot) ≥ L_old
        flag = true
        return Int[],Int[],eltype(L_pot)[], flag
    else
        flag = false

        ind = L_pot .< L_old
        return τ_pot[ind],ts_pot[ind],L_pot[ind],flag
    end
end


"""
Perform a potential embedding cycle from the multi- or univariate Dataset `Ys`.
Return the possible delays `τ_pot`, the associated time series `ts_pot` and
the corresponding L-statistic-values, `L_pot` for each peak, i.e. for each
(`τ_pot`, `ts_pot`) pair. If `FNN=true`, `L_pot` stores the corresponding
fnn-statistic-values.
"""
function embedding_cycle_pecuzal(Y_act, Ys, τs, w, samplesize,
                    K, metric, α, p, Tw, KNN, τ_vals, ts_vals, FNN)

    ε★, _ = pecora(Ys, Tuple(τ_vals), Tuple(ts_vals); delays = τs, w = w,
            samplesize = samplesize, K = K, metric = metric, α = α,
            p = p, undersampling = false)
    # update τ_vals, ts_vals, Ls, ε★s
    τ_pot, ts_pot, L_pot = pick_possible_embedding_params(ε★, Y_act, Ys, τs, Tw,
                                                KNN, w, samplesize, metric, FNN)

    return τ_pot, ts_pot, L_pot
end


"""
    Compute all possible τ-values (and according time series numbers) and their
corresponding L-statistics (or FNN-statistics, if `FNN=true`) for the input
continuity statistic ε★.
"""
function pick_possible_embedding_params(ε★, Y_act, Ys, τs, Tw, KNN, w,
                                                        samplesize, metric, FNN)
    L_pots = []
    τ_pots = []
    ts_pots = []

    for ts = 1:size(Ys,2)
        # zero-padding of ⟨ε★⟩ in order to also cover τ=0 (important for the multivariate case)
        # get the L-statistic for each peak in ⟨ε★⟩
        if FNN
            L_trials, max_idx, _ = local_fnn_statistics(vec([0; ε★[:,ts]]), Y_act,
                                   Ys[:,ts], τs, w, metric)
        else
            L_trials, max_idx, _ = local_L_statistics(vec([0; ε★[:,ts]]), Y_act,
                                  Ys[:,ts], τs, Tw, KNN, w, samplesize, metric)
        end
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
Return the FNN-statistic `FNN` and indices `max_idx` and weighted peak height
`ξ = peak-height * L` for all local maxima in ε★
"""
function local_fnn_statistics(ε★, Y_act, s, τs, w, metric; r=2)

    maxima, max_idx = get_maxima(ε★) # determine local maxima in ⟨ε★⟩
    FNN_trials = zeros(Float64, length(max_idx))
    ξ_trials = zeros(Float64, length(max_idx))

    # compute nearest-neighbor-distances for actual trajectory
    Y_act2 = Y_act[1:end-τs[maximum(max_idx)-1],:]
    Y_act2 = regularize(Y_act2)
    vtree = KDTree(Y_act2, metric)
    _, NNdist_old = DelayEmbeddings.all_neighbors(vtree, Y_act2, 1:length(Y_act2), 1, w)

    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        Y_trial = DelayEmbeddings.hcat_lagged_values(Y_act,s,τs[τ_idx-1])
        Y_trial = regularize(Y_trial)
        vtree = KDTree(Y_trial, metric)
        _, NNdist_new = DelayEmbeddings.all_neighbors(vtree, Y_trial, 1:length(Y_trial), 1, w)
        # compute FNN-statistic
        FNN_trials[i] = fnn_embedding_cycle(NNdist_old, NNdist_new[1:length(NNdist_old)], r)
        ξ_trials[i] = FNN_trials[i]*maxima[i]
    end
    return FNN_trials, max_idx, ξ_trials
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

"""
    fnn_embedding_cycle(NNdist, NNdistnew, r=2) -> FNNs
Compute the amount of false nearest neighbors `FNNs`, when adding another component
to a given (vector-) time series. This new component is the `τ`-lagged version
of a univariate time series. `NNdist` is storing the distances of the nearest
neighbor for all considered fiducial points and `NNdistnew` is storing the
distances of the nearest neighbor for each fiducial point in one embedding
dimension higher using a given `τ`. The obligatory threshold `r` is by default
set to 2.
"""
function fnn_embedding_cycle(NNdist, NNdistnew, r::Real=2)
    @assert length(NNdist) == length(NNdistnew) "Both input vectors need to store the same number of distances."
    N = length(NNdist)
    fnns = 0
    fnns2= 0
    inverse_r = 1/r
    @inbounds for i = 1:N
        if NNdistnew[i][1]/NNdist[i][1] > r && NNdist[i][1] < inverse_r
            fnns +=1
        end
        if NNdist[i][1] < inverse_r
            fnns2 +=1
        end
    end
    if fnns==0
        return 1
    else
        return fnns/fnns2
    end
end

"""
    pecuzal_embedding(s; kwargs...) → Y, τ_vals, ts_vals, Ls ,⟨ε★⟩
A unified approach to properly embed a time series or a set of time series
(`Dataset`) based on the ideas of Pecora et al. [^Pecoral2007] and Uzal et al.
[^Uzal2011].

## Keyword arguments

* `τs= 0:50`: Possible delay values `τs` (in sampling time units). For each of
  the `τs`'s the continuity statistic ⟨ε★⟩ gets computed and further processed
  in order to find optimal delays `τᵢ` for each embedding cycle `i` (read
  algorithm description).
* `w::Int = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `samplesize::Real = 0.1`: determine the fraction of all phase space points
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
* `metric = Euclidean()`: metric with which to find nearest neigbhors
* `α::Real = 0.05`: The significance level for obtaining the continuity statistic
* `p::Real = 0.5`: The p-parameter for the binomial distribution used for the
  computation of the continuity statistic ⟨ε★⟩.
* `max_cycles = 50`: The algorithm will stop after that many cycles no matter what.


## Description
The method works iteratively and gradually builds the final embedding vectors
`Y`. Based on the `⟨ε★⟩`-statistic [`pecora`](@ref) the algorithm picks an
optimal delay value `τᵢ` for each embedding cycle i.
For achieving that, we take the inpute time series `s` and compute the continuity
statistic `⟨ε★⟩`, 1. each local maxima in `⟨ε★⟩` is used for constructing a
candidate embedding trajectory `Y_trial` with a delay corresponding to that
specific peak in `⟨ε★⟩`. 2. We then compute the `L`-statistic [`uzal_cost`](@ref)
for `Y_trial`. 3. We pick the peak/`τ`-value, for which `L` is minimal and
construct the actual embedding trajectory `Y_actual` (1.-3. corresponds to an
embedding cycle). 4. We repeat steps 1.-3. with `Y_actual` as input and stop the
algorithm when `L` can not be reduced anymore. `Y_actual` -> `Y`.

In case of multivariate embedding, i.e. when embedding a set of M time series
(`s::Dataset`), in each embedding cycle `⟨ε★⟩` gets computed for all M time series
available. The optimal delay value `τ` in each embedding cycle is chosen
as the peak/`τ`-value for which `L` is minimal under all available peaks and under
all M `⟨ε★⟩`'s. In the first embedding cycle there will be M! different `⟨ε★⟩`'s
to consider, since it is not clear a priori which time series of the input should
consitute the first component of the embedding vector and form `Y_actual`.

The range of considered delay values is determined in `τs` and for the
nearest neighbor search we respect the Theiler window `w`. The final embedding
vector is stored in `Y` (`Dataset`). The chosen delay values for each embedding
cycle are stored in `τ_vals` and the according time series number chosen for the
each delay value in `τ_vals` is stored in `ts_vals`. For univariate embedding
(`s::Vector`) `ts_vals` is a vector of ones of length `τ_vals`, because there is
simply just one time series to choose from. The function also returns the
`L`-statistic `Ls` for each embedding cycle and the continuity statistic `⟨ε★⟩`
as an `Array` of `Vector`s.

[^Pecora2007]: Pecora, L. M., Moniz, L., Nichols, J., & Carroll, T. L. (2007). [A unified approach to attractor reconstruction. Chaos 17(1)](https://doi.org/10.1063/1.2430294).
[^Uzal2011]: Uzal, L. C., Grinblat, G. L., Verdes, P. F. (2011). [Optimal reconstruction of dynamical systems: A noise amplification approach. Physical Review E 84, 016223](https://doi.org/10.1103/PhysRevE.84.016223).
"""
function pecuzal_embedding(s::Vector{T}; τs = 0:50 , w::Int = 1,
    samplesize::Real = 1, K::Int = 13, KNN::Int = 3, Tw::Int=4*w,
    metric = Euclidean(), α::Real = 0.05, p::Real = 0.5,
    max_cycles = 50) where {T<:Real}

    @assert 0 < samplesize ≤ 1 "Please select a valid `samplesize`, which denotes a fraction of considered fiducial points, i.e. `samplesize` ∈ (0 1]"
    @assert all(x -> x ≥ 0, τs)
    s_orig = s
    s = regularize(s) # especially important for comparative L-statistics
    # define actual phase space trajectory
    Y_act = Dataset(s)

    L_init = uzal_cost(Y_act; samplesize = samplesize, K = KNN, metric = metric,
                       w = w, Tw = Tw)

    # set a flag, in order to tell the while loop when to stop. Each loop
    # stands for encountering a new embedding dimension
    flag, counter = true, 1

    # preallocate output variables
    τ_vals = Int64[0]
    ts_vals = Int64[1]
    Ls = Float64[]
    ε★s = Array{T}(undef, length(τs), max_cycles)

    # loop over increasing embedding dimensions until some break criterion will
    # tell the loop to stop/break
    while flag
        Y_act = pecuzal_embedding_cycle!(
                Y_act, flag, s, τs, w, counter, ε★s, τ_vals, metric,
                Ls, ts_vals, samplesize, K, α, p, Tw, KNN)
        flag = pecuzal_break_criterion(Ls, counter, max_cycles, L_init)
        counter += 1
    end
    # construct final reconstruction vector
    NN = (length(s)-sum(τ_vals[1:counter-1]))
    Y_final = s_orig
    for i = 2:length(τ_vals[1:counter-1])
        Y_final = DelayEmbeddings.hcat_lagged_values(Y_final,s_orig,τ_vals[i])
    end
    return Y_final, τ_vals[1:end-1], ts_vals[1:end-1], Ls, ε★s[:,1:counter-1]

end

function pecuzal_embedding(Y::Dataset{D, T}; τs = 0:50 , w::Int = 1,
    samplesize::Real = 1, K::Int = 13, KNN::Int = 3, Tw::Int=4*w,
    metric = Euclidean(), α::Real = 0.05, p::Real = 0.5,
    max_cycles = 50) where {D, T<:Real}

    @assert 0 < samplesize ≤ 1 "Please select a valid `samplesize`, which denotes a fraction of considered fiducial points, i.e. `samplesize` ∈ (0 1]"
    @assert all(x -> x ≥ 0, τs)
    Y_orig = Y
    Y = regularize(Y) # especially important for comparative L-statistics
    # compute initial L values for each time series
    L_inits = zeros(size(Y,2))
    for i = 1:size(Y,2)
        L_inits[i] = uzal_cost(Dataset(Y[:,i]); samplesize = samplesize, K = KNN, metric = metric,
                           w = w, Tw = Tw)
    end
    L_init = minimum(L_inits)

    # define actual phase space trajectory
    Y_act = []

    # set a flag, in order to tell the while loop when to stop. Each loop
    # stands for encountering a new embedding dimension
    flag, counter = true, 1

    # preallocate output variables
    τ_vals = Int64[0]
    ts_vals = Int64[]
    Ls = Float64[]
    ε★s = fill(zeros(T, length(τs), size(Y,2)), 1, max_cycles)

    # loop over increasing embedding dimensions until some break criterion will
    # tell the loop to stop/break
    while flag
        Y_act = pecuzal_multivariate_embedding_cycle!(
                Y_act, flag, Y, τs, w, counter, ε★s, τ_vals, metric,
                Ls, ts_vals, samplesize, K, α, p, Tw, KNN)

        flag = pecuzal_break_criterion(Ls, counter, max_cycles, L_init)
        counter += 1
    end
    # construct final reconstruction vector
    Y_final = Y_orig[:,ts_vals[1]]
    for i = 2:length(τ_vals[1:counter-1])
        Y_final = DelayEmbeddings.hcat_lagged_values(Y_final,Y_orig[:,ts_vals[i]],τ_vals[i])
    end

    return Y_final, τ_vals[1:end-1], ts_vals[1:end-1], Ls, ε★s[:,1:counter-1]

end


"""
Perform one univariate embedding cycle on `Y_act`. Return the new `Y_act`
"""
function pecuzal_embedding_cycle!(
        Y_act, flag, s, τs, w, counter, ε★s, τ_vals, metric,
        Ls, ts_vals, samplesize, K, α, p, Tw, KNN)

    ε★, _ = pecora(s, Tuple(τ_vals), Tuple(ts_vals); delays = τs, w = w,
                samplesize = samplesize, K = K, metric = metric, α = α,
                p = p, undersampling = false)
    ε★s[:,counter] = ε★

    # zero-padding of ⟨ε★⟩ in order to also cover τ=0 (important for the multivariate case)
    ε★ = vec([0; ε★])
    # get the L-statistic for each peak in ⟨ε★⟩ and take the one according to L_min
    L_trials, max_idx, _ = local_L_statistics(ε★, Y_act, s, τs, Tw, KNN, w, samplesize, metric)
    L_min, min_idx = findmin(L_trials)

    push!(τ_vals, τs[max_idx[min_idx]-1])
    push!(ts_vals, 1)
    push!(Ls, L_min)

    # create phase space vector for this embedding cycle
    Y_act = DelayEmbeddings.hcat_lagged_values(Y_act,s,τ_vals[counter+1])

    return Y_act
end

"""
Perform one embedding cycle on `Y` with a multivariate set Ys
"""
function pecuzal_multivariate_embedding_cycle!(
        Y_act, flag, Ys, τs, w, counter, ε★s, τ_vals, metric,
        Ls, ts_vals, samplesize, K, α, p, Tw, KNN)

    M = size(Ys,2)
    # in the 1st cycle we have to check all (size(Y,2)^2 combinations and pick
    # the tau according to minimial ξ = (peak height * resulting L-statistic)
    if counter == 1
        Y_act = first_embedding_cycle_pecuzal!(Ys, M, τs, w, samplesize, K,
                                metric, α, p, Tw, KNN, τ_vals, ts_vals, Ls, ε★s)
    # in all other cycles we just have to check (size(Y,2)) combinations and pick
    # the tau according to minimal resulting L-statistic
    else
        Y_act = embedding_cycle_pecuzal!(Y_act, Ys, counter, M, τs, w, samplesize,
                            K, metric, α, p, Tw, KNN, τ_vals, ts_vals, Ls, ε★s)
    end
    return Y_act
end

"""
Perform the first embedding cycle of the multivariate embedding. Return the
actual reconstruction vector `Y_act`.
"""
function first_embedding_cycle_pecuzal!(Ys, M, τs, w, samplesize, K,
                        metric, α, p, Tw, KNN, τ_vals, ts_vals, Ls, ε★s)
    counter = 1
    L_min = zeros(M)
    L_min_idx = zeros(Int, M)
    ε★ = zeros(length(τs), M*M)
    idx = zeros(Int, M)
    ξ_min = zeros(M)
    for ts = 1:M
        ε★[:,1+(M*(ts-1)):M*ts], _ = pecora(Ys, (0,), (ts,); delays = τs,
                    w = w, samplesize = samplesize, K = K, metric = metric,
                    α = α, p = p, undersampling = false)
        L_min[ts], L_min_idx[ts], idx[ts], ξ_min[ts] = choose_right_embedding_params(
                                        ε★[:,1+(M*(ts-1)):M*ts], Ys[:,ts],
                                        Ys, τs, Tw, KNN, w, samplesize,
                                        metric)
    end
    ξ_mini, min_idx = findmin(ξ_min)
    L_mini = L_min[min_idx]
    # update τ_vals, ts_vals, Ls, ε★s
    push!(τ_vals, τs[L_min_idx[min_idx]])
    push!(ts_vals, min_idx)             # time series to start with
    push!(ts_vals, idx[min_idx])        # result of 1st embedding cycle
    push!(Ls, L_mini)
    ε★s[counter] = ε★[:,1+(M*(ts_vals[1]-1)):M*ts_vals[1]]

    # create phase space vector for this embedding cycle
    Y_act = DelayEmbeddings.hcat_lagged_values(Ys[:,ts_vals[counter]],
                                 Ys[:,ts_vals[counter+1]],τ_vals[counter+1])

    return Y_act
end

"""
Perform an embedding cycle of the multivariate embedding, but the first one.
Return the actual reconstruction vector `Y_act`.
"""
function embedding_cycle_pecuzal!(Y_act, Ys, counter, M, τs, w, samplesize,
                    K, metric, α, p, Tw, KNN, τ_vals, ts_vals, Ls, ε★s)

    ε★, _ = pecora(Ys, Tuple(τ_vals), Tuple(ts_vals); delays = τs, w = w,
            samplesize = samplesize, K = K, metric = metric, α = α,
            p = p, undersampling = false)
    # update τ_vals, ts_vals, Ls, ε★s
    choose_right_embedding_params!(ε★, Y_act, Ys, τ_vals, ts_vals, Ls, ε★s,
                                counter, τs, Tw, KNN, w, samplesize, metric)
    # create phase space vector for this embedding cycle
    Y_act = DelayEmbeddings.hcat_lagged_values(Y_act, Ys[:, ts_vals[counter+1]],
                                                        τ_vals[counter+1])
    return Y_act
end


"""
    Choose the minimum L and corresponding τ for each ε★-statistic, based on
picking the peak in ε★, which corresponds to the minimal `L`-statistic.
"""
function choose_right_embedding_params!(ε★, Y, Ys, τ_vals, ts_vals, Ls, ε★s,
                                 counter, τs, Tw, KNN, w, samplesize, metric)
    L_min_ = zeros(size(Ys,2))
    τ_idx = zeros(Int,size(Ys,2))
    for ts = 1:size(Ys,2)
        # zero-padding of ⟨ε★⟩ in order to also cover τ=0 (important for the multivariate case)
        # get the L-statistic for each peak in ⟨ε★⟩ and take the one according to L_min
        L_trials_, max_idx_, _ = local_L_statistics(vec([0; ε★[:,ts]]), Y, Ys[:,ts],
                                        τs, Tw, KNN, w, samplesize, metric)
        L_min_[ts], min_idx_ = findmin(L_trials_)
        τ_idx[ts] = max_idx_[min_idx_]-1
    end
    idx = sortperm(L_min_)
    L_mini, min_idx = findmin(L_min_)
    push!(τ_vals, τs[τ_idx[min_idx]])
    push!(ts_vals, min_idx)
    push!(Ls, L_mini)

    ε★s[counter] = ε★
end

"""
    Choose the right embedding parameters of the ε★-statistic in the first
embedding cycle. Return the `L`-value, the corresponding index value of the
chosen peak `τ_idx` and the number of the chosen time series to start with `idx`.
Here the peak is chosen not on the basis of minimal `L`, as in all consecutive
embedding cycles, but on the basis of minimal `ξ` = (peak height * resulting
`L`-statistic), which is the last output variable.
"""
function choose_right_embedding_params(ε★, Y, Ys, τs, Tw, KNN, w, samplesize, metric)
    ξ_min_ = zeros(size(Ys,2))
    L_min_ = zeros(size(Ys,2))
    τ_idx = zeros(Int,size(Ys,2))
    for ts = 1:size(Ys,2)
        # zero-padding of ⟨ε★⟩ in order to also cover τ=0 (important for the multivariate case)
        # get the L-statistic for each peak in ⟨ε★⟩ and take the one according to L_min
        L_trials_, max_idx_, ξ_trials_ = local_L_statistics(vec([0; ε★[:,ts]]), Y, Ys[:,ts],
                                        τs, Tw, KNN, w, samplesize, metric)
        ξ_min_[ts], min_idx_ = findmin(ξ_trials_)
        L_min_[ts] = L_trials_[min_idx_]
        τ_idx[ts] = max_idx_[min_idx_]-1
    end
    idx = sortperm(ξ_min_)
    return L_min_[idx[1]], τ_idx[idx[1]], idx[1], ξ_min_[idx[1]]
end


function pecuzal_break_criterion(Ls, counter, max_num_of_cycles, L_init)
    flag = true
    if counter == 1
        if Ls[end] > L_init
            println("Algorithm stopped due to increasing L-values. "*
                    "Valid embedding NOT achieved ⨉.")
            flag = false
        end
    end
    if counter > 1 && Ls[end]>Ls[end-1]
        println("Algorithm stopped due to minimum L-value reached. "*
                "VALID embedding achieved ✓.")
        flag = false
    end
    if max_num_of_cycles == counter
        println("Algorithm stopped due to hitting max cycle number. "*
                "Valid embedding NOT achieved ⨉.")
        flag = false
    end
    return flag
end
