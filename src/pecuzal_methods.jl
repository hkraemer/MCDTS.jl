using DelayEmbeddings
using DynamicalSystemsBase
using Random
using Neighborhood
using Revise

"""
    give_potential_delays(Ys::Dataset, τs, w::Int, τ_vals,
                    ts_vals [; kwargs...]) → τ_pot, ts_pot, L_pot/FNN_flag, flag
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
* `α::Real = 0.05`: The significance level for obtaining the continuity statistic
* `p::Real = 0.5`: The p-parameter for the binomial distribution used for the
  computation of the continuity statistic ⟨ε★⟩.
* `FNN:Bool = false`: Determines whether the algorithm should minimize the L-statistic
  or the FNN-statistic
* `PRED::Bool = false`: Determines whether the algorithm should minimize the
  L-statistic or a cost function based on minimizing the `Tw`-step-prediction error
* `Tw::Int = 1`: If `PRED = true`, this is the considered prediction horizon
* `linear::Bool=false`: If `PRED = true`, this determines whether the prediction shall
  be made on the zeroth or a linear predictor.
* `PRED_mean::Bool=false`: If `PRED = true`, this determines whether the prediction shall
  be optimized on the mean MSE of all components or only on the 1st-component (Default)
* `PRED_L::Bool=false`: If `PRED = true`, this determines whether the prediction shall
  be optimized on possible delay values gained from the continuity statistic or on
  delays = 0:25 (Default)
* `PRED_KL::Bool=false`: If `PRED = true`, this determines whether the prediction shall
  be optimized on the Kullback-Leibler-divergence of the in-sample prediction and
  the true in-sample values, or if the optimization shall be made on the MSE of them (Default)
* `threshold::Real = 0`: The algorithm does not pick a peak from the continuity
  statistic, when its corresponding `ΔL`/FNN-value exceeds this threshold. Please
  provide a positive number for both, `L` and `FNN`-statistic option (since the
  `ΔL`-values are negative numbers for meaningful embedding cycles, this threshold
  gets internally sign-switched).
* `tws::Range = 2:τs[end]`: Customization of the sampling of the different T's,
  when computing Uzal's L-statistics. Here any kind of integer ranges (starting
  at 2) are allowed, up to `τs[end]`.
"""
function give_potential_delays(Yss::Dataset{D, T}, τs, w::Int, τ_vals, ts_vals, L_old;
                samplesize::Real=1, K::Int = 13, α::Real = 0.05, p::Real = 0.5,
                KNN::Int = 3, FNN::Bool = false, PRED::Bool = false, Tw::Int = 1,
                threshold::Real = 0, tws::AbstractRange{Int} = 2:τs[end],
                linear::Bool=false, PRED_mean::Bool=false, PRED_L::Bool=false,
                PRED_KL::Bool=false) where {D, T}

    @assert (FNN || PRED) || (~FNN && ~PRED) "Select either FNN or PRED keyword (or none)."
    @assert 0 < samplesize ≤ 1 "Please select a valid `samplesize`, which denotes a fraction of considered fiducial points, i.e. `samplesize` ∈ (0 1]"
    @assert all(x -> x ≥ 0, τs)
    @assert threshold ≥ 0
    if ~FNN && ~PRED
        threshold = -threshold # due to the negativity of L-decrease
    end
    metric = Euclidean()
    Ys = regularize(Yss)
    # compute Y_act
    Y_act = genembed(Ys, τ_vals, ts_vals)

    # compute potential delay values with corresponding time series values and
    # L-statistic-values (or FNN-statistic-values or PRED-statistic-values,
    # these will be binded in `L_pots` for simplicity, anyway)
    τ_pots, ts_pots, L_pots = embedding_cycle_pecuzal(Y_act, Ys, τs, w, samplesize,
                            K, metric, α, p, KNN, τ_vals, ts_vals, FNN, tws, PRED,
                            Tw; linear = linear, PRED_mean = PRED_mean, PRED_L = PRED_L,
                            PRED_KL = PRED_KL)
    if isempty(τ_pots)
        flag = true
        return Int[],Int[],eltype(L_pots)[], flag
    end
    # transform array of arrays to a single array
    τ_pot = reduce(vcat, τ_pots)
    ts_pot = reduce(vcat, ts_pots)
    L_pot = reduce(vcat, L_pots)

    if FNN || PRED
        if (minimum(L_pot) ≥ L_old)
            flag = true
            return Int[],Int[],eltype(L_pot)[], flag
        elseif (minimum(L_pot) ≤ threshold)
            flag = true
            ind = L_pot .< L_old
            return τ_pot[ind],ts_pot[ind],L_pot[ind],flag
        else
            flag = false
            ind = L_pot .< L_old
            return τ_pot[ind],ts_pot[ind],L_pot[ind],flag
        end
    else
        if minimum(L_pot) > threshold
            flag = true
            ind = L_pot .≤ threshold
            return τ_pot[ind],ts_pot[ind],L_pot[ind],flag
        else
            flag = false
            ind = L_pot .≤ threshold
            return τ_pot[ind],ts_pot[ind],L_pot[ind],flag
        end
    end
end


"""
Perform a potential embedding cycle from the multi- or univariate Dataset `Ys`.
Return the possible delays `τ_pot`, the associated time series `ts_pot` and
the corresponding L-statistic-values, `L_pot` for each peak, i.e. for each
(`τ_pot`, `ts_pot`) pair. If `FNN=true`, `L_pot` stores the corresponding
fnn-statistic-values.
"""
function embedding_cycle_pecuzal(Y_act, Ys, τs, w, samplesize, K, metric, α, p,
                    KNN, τ_vals, ts_vals, FNN, tws, PRED, Tw; linear::Bool=false,
                    PRED_mean::Bool=false, PRED_L::Bool=false, PRED_KL::Bool=false)
    if PRED && ~PRED_L
        ε★ = zeros(length(τs), size(Ys,2))
    elseif PRED && PRED_L
        ε★, _ = pecora(Ys, Tuple(τ_vals.*(-1)), Tuple(ts_vals); delays = τs, w = w,
                samplesize = samplesize, K = K, metric = metric, α = α,
                p = p, undersampling = false)
    else
        ε★, _ = pecora(Ys, Tuple(τ_vals), Tuple(ts_vals); delays = τs, w = w,
                samplesize = samplesize, K = K, metric = metric, α = α,
                p = p, undersampling = false)
    end
    # update τ_vals, ts_vals, Ls, ε★s
    τ_pot, ts_pot, L_pot = pick_possible_embedding_params(ε★, Y_act, Ys, τs,
                                            KNN, w, samplesize, metric, FNN,
                                            tws, PRED, Tw; τ_vals = τ_vals,
                                            ts_vals = ts_vals, linear = linear,
                                            PRED_mean = PRED_mean, PRED_L = PRED_L,
                                            PRED_KL = PRED_KL)
    return τ_pot, ts_pot, L_pot
end


"""
    Compute all possible τ-values (and according time series numbers) and their
corresponding L-statistics (or FNN-statistics, if `FNN=true`, or MSE, if
`PRED_true`) for the input continuity statistic ε★.
"""
function pick_possible_embedding_params(ε★, Y_act, Ys, τs, KNN, w, samplesize,
            metric, FNN, tws, PRED, Tw; τ_vals = [0], ts_vals = [1],
            linear::Bool=false, PRED_mean::Bool=false, PRED_L::Bool=false,
            PRED_KL::Bool=false)

    L_pots = []
    τ_pots = []
    ts_pots = []

    for ts = 1:size(Ys,2)
        # zero-padding of ⟨ε★⟩ in order to also cover τ=0 (important for the multivariate case)
        # get the L-statistic for each peak in ⟨ε★⟩
        if FNN
            L_trials, max_idx, _ = local_fnn_statistics(vec([0; ε★[:,ts]]), Y_act,
                                   Ys[:,ts], τs, w, metric)
        elseif PRED
            L_trials, max_idx = local_PRED_statistics(vec([0; ε★[:,ts]]), Y_act,
                                   Ys, τs, w, metric, Tw; τ_vals = τ_vals,
                                   ts_vals = ts_vals, ts = ts, linear = linear,
                                   K = KNN, PRED_mean = PRED_mean,
                                   PRED_L = PRED_L, PRED_KL = PRED_KL)
        else
            L_trials, max_idx = MCDTS.local_L_statistics(vec([0; ε★[:,ts]]), Y_act,
                            Ys[:,ts], τs, KNN, w, samplesize, metric; tws = tws)
        end
        push!(L_pots, L_trials)
        push!(ts_pots, fill(ts,length(L_trials)))
        push!(τ_pots, τs[max_idx.-1])
    end
    return τ_pots, ts_pots, L_pots
end

"""
    Return the maximum decrease of the L-statistic `L_decrease` and corresponding
delay-indices `max_idx` for all local maxima in ε★
"""
function local_L_statistics(ε★::Vector{T}, Y_act::Dataset{D, T}, s::Vector{T},
        τs, KNN::Int, w::Int, samplesize::Real, metric;
        tws::AbstractRange{Int} = 2:τs[end]) where {D, T}

    maxima, max_idx = get_maxima(ε★) # determine local maxima in ⟨ε★⟩
    L_decrease = zeros(Float64, length(max_idx))
    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        Y_trial = DelayEmbeddings.hcat_lagged_values(Y_act, s, τs[τ_idx-1])
        # compute L-statistic for Y_act and Y_trial and get the maximum decrease
        L_decrease[i] = MCDTS.uzal_cost_pecuzal_mcdts(Y_act, Y_trial, τs[end]; K = KNN,
                                w = w, metric = metric, tws = tws)
    end
    return L_decrease, max_idx
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
Return costs (MSE) of a `Tw`-step-ahead local-prediction.
"""
function local_PRED_statistics(ε★, Y_act, Ys, τs, w, metric, Tw; τ_vals = [0],
                        ts_vals = [1], ts = 1, K::Int = 1, linear::Bool=false,
                        PRED_mean::Bool=false, PRED_L::Bool=false,
                        PRED_KL::Bool=false)
    s = Ys[:,ts]
    if PRED_L
        _, max_idx = get_maxima(ε★) # determine local maxima in ⟨ε★⟩
    else
        max_idx = Vector(τs.+2)
        ts_idx = findall(e->e==ts, ts_vals) # do not consider already taken delays
        filter!(e->e∉(τ_vals[ts_idx] .+ 2), max_idx) # do not consider already taken delays
    end
    PRED_mse = zeros(Float64, length(max_idx))
    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        tau_trials = ((τ_vals.*(-1))...,(τs[τ_idx-1]*(-1)),)
        ts_trials = (ts_vals...,ts,)
        Y_trial = genembed(Ys, tau_trials, ts_trials)
        # compute PRED-statistic for Y_trial
        if linear
            if PRED_mean
                if PRED_KL
                    PRED_mse[i] = mean(MCDTS.linear_prediction_cost_KL(Y_trial; w = w,
                            K = 2*(size(Y_trial,2)+1), Tw = Tw, metric = metric))
                else
                    PRED_mse[i] = mean(MCDTS.linear_prediction_cost(Y_trial; w = w,
                            K = 2*(size(Y_trial,2)+1), Tw = Tw, metric = metric))
                end

            else
                if PRED_KL
                    PRED_mse[i] = MCDTS.linear_prediction_cost_KL(Y_trial; w = w,
                            K = 2*(size(Y_trial,2)+1), Tw = Tw, metric = metric)[1]
                else
                    PRED_mse[i] = MCDTS.linear_prediction_cost(Y_trial; w = w,
                            K = 2*(size(Y_trial,2)+1), Tw = Tw, metric = metric)[1]
                end
            end
        else
            if PRED_mean
                if PRED_KL
                    PRED_mse[i] = mean(MCDTS.zeroth_prediction_cost_KL(Y_trial; w = w,
                            K = K, Tw = Tw,  metric = metric))
                else
                    PRED_mse[i] = mean(MCDTS.zeroth_prediction_cost(Y_trial; w = w,
                            K = K, Tw = Tw,  metric = metric))
                end
            else
                if PRED_KL
                    PRED_mse[i] = MCDTS.zeroth_prediction_cost_KL(Y_trial; w = w,
                                    K = K, Tw = Tw,  metric = metric)[1]
                else
                    PRED_mse[i] = MCDTS.zeroth_prediction_cost(Y_trial; w = w,
                                    K = K, Tw = Tw,  metric = metric)[1]
                end
            end
        end
    end
    return PRED_mse, max_idx
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
    uzal_cost_pecuzal_mcdts(Y1::Dataset, Y2::Dataset, Tw; kwargs...) → L_decrease
This function is based on the functionality of [`uzal_cost`](@ref), here
specifically tailored for the needs in the PECUZAL algorithm.
Compute the L-statistics `L1` and `L2` for the input datasets `Y1` and `Y2` for
increasing time horizons `T = 1:Tw`. For each `T`, compute `L1` and `L2` and
decrease `L_decrease = L2 - L1`. If `L_decrease` is a negative value, then `Y2`
can be regarded as a "better" reconstruction that `Y1`. Break, when `L_decrease`
reaches the 1st local minima, since this will typically also be the global
minimum. Return the according minimum `L_decrease`-value.

## Keyword arguments

* `K = 3`: the amount of nearest neighbors considered, in order to compute σ_k^2
  (read algorithm description).
  If given a vector, minimum result over all `k ∈ K` is returned.
* `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
  state space trajectory `Y.
* `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `econ::Bool = false`: Economy-mode for L-statistic computation. Instead of
  computing L-statistics for time horizons `2:Tw`, here we only compute them for
  `2:2:Tw`.
* `tws::Range = 2:Tw`: Further customization of the sampling of the different T's.
  While `econ=true` gives `tws = 2:2:Tw`, here any kind of interger ranges (starting at 2)
  are allowed, up to `Tw`.
"""
function uzal_cost_pecuzal_mcdts(Y::Dataset{D, ET}, Y_trial::Dataset{DT, ET}, Tw::Int;
        K::Int = 3, w::Int = 1, econ::Bool = false, tws::AbstractRange{Int} = 2:Tw,
        metric = Euclidean) where {D, DT, ET}

    @assert DT == D+1
    @assert Tw ≥ 0
    @assert tws[1]==2

    if econ
        tws = 2:2:Tw # start at 2 will eliminate bad results for noise
    end

    NNN = length(Y_trial)-1
    # preallocation for 1st dataset
    ϵ² = zeros(NNN)             # neighborhood size
    E² = zeros(NNN, length(tws))         # conditional variance
    ϵ_ball = zeros(ET, K+1, D)  # epsilon neighbourhood
    u_k = zeros(ET, D)          # center of mass
    # preallocation for 2nd dataset
    ϵ²_trial = zeros(NNN)             # neighborhood size
    E²_trial = zeros(NNN, length(tws))         # conditional variance
    ϵ_ball_trial = zeros(ET, K+1, DT) # epsilon neighbourhood
    u_k_trial = zeros(ET, DT)         # center of mass

    dist_former = 9999999 # intial L-decrease

    # loop over each time horizon
    cnt = 1
    for T in tws
        NN = length(Y_trial)-T
        if NN < 1
            error("Time series too short for given possible delays and Theiler window to find enough nearest neighbours")
        end
        ns = 1:NN

        vs = Y[ns] # the fiducial points in the data set
        vs_trial = Y_trial[ns] # the fiducial points in the data set

        vtree = KDTree(Y[1:NN], metric)
        allNNidxs, allNNdist = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, w)
        vtree_trial = KDTree(Y_trial[1:NN], metric)
        allNNidxs_trial, allNNdist_trial = DelayEmbeddings.all_neighbors(vtree_trial, vs_trial, ns, K, w)

        # compute conditional variances and neighborhood-sizes
        DelayEmbeddings.compute_conditional_variances!(ns, vs, vs_trial, allNNidxs,
            allNNidxs_trial, Y, Y_trial, ϵ_ball, ϵ_ball_trial, u_k, u_k_trial,
            T, K, metric, ϵ², ϵ²_trial, E², E²_trial, cnt)

        # compute distance of L-values and check whether that distance can be
        # increased
        dist = DelayEmbeddings.compute_L_decrease(E², E²_trial, ϵ², ϵ²_trial, cnt, NN)
        if dist > dist_former && dist_former<0
            break
        else
            dist_former = dist
        end
        cnt += 1
    end
    return dist_former
end
