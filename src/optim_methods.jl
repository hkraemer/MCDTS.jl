
## Methods for Loss stats

# Methods for altering arrays containing nodes (children) depending on the chosen Loss-function
# FNN, CCM & Prediction
function push!(children::Union{Array{Node,1},Nothing}, n, Γ::AbstractLoss, current_node::AbstractTreeElement)
    Base.push!(children, Node(n[1],n[2],n[3],[get_τs(current_node); n[1]], [get_ts(current_node); n[2]], nothing, n[4]))
end
# L-function
function push!(children::Union{Array{Node,1},Nothing}, n, Γ::L_statistic, current_node::AbstractTreeElement)
    if typeof(current_node) == MCDTS.Root
        Base.push!(children, Node(n[1],n[2],(current_node.Lmin+n[3]),[get_τs(current_node); n[1]], [get_ts(current_node); n[2]], nothing, n[4]))
    else
        Base.push!(children, Node(n[1],n[2],(current_node.L+n[3]),[get_τs(current_node); n[1]], [get_ts(current_node); n[2]], nothing, n[4]))
    end
end

# Methods for intitializing embedding parameters and loss function value depending on the chosen Loss-function

# force 1st component to be 1st time series in all cases
function init_embedding_params(Γ::AbstractLoss, N::Int)
    return [0], [1], 99999*ones(N), [nothing]
end
function init_embedding_params(Γ::FNN_statistic, N::Int)
    return [0], [1], ones(N), [nothing]
end
function init_embedding_params(Γ::L_statistic, N::Int)
    return [0], [1], zeros(N), [nothing]
end
function init_embedding_params(Γ::CCM_ρ, N::Int)
    return [0], [1], zeros(N), [nothing]
end


"""
    get_potential_delays(optimalg::AbstractMCDTSOptimGoal, Ys::Dataset, τs, w::Int, τ_vals,
                    ts_vals, L_old ; kwargs...]) → τ_pot, ts_pot, L_pot, flag, temps

    Compute the potential delay `τ_pot` and time series values `ts_pot`, which would
    each result in a potential Loss-statistic value `L_pot`, by using an
    embedding method specified in `optimalg` [^Kraemer2021b] (see [`MCDTSOptimGoal`](@ref))
    and for a range of possible delay values `τs`. The input dataset `Ys` can be
    multivariate. `w` is the Theiler window (neighbors in time with index `w` close
    to the point, that are excluded from being true neighbors. `w=0` means to
    exclude only the point itself, and no temporal neighbors. In case of multivariate
    time series input choose `w` as the maximum of all `wᵢ's`. `τ_vals` and `ts_vals`
    describe the embedding up to the current embedding cycle.

    ## Keyword arguments
    * See [`mcdts_embedding`](@ref) for a list of all keywords.
"""
function get_potential_delays(optimalg::AbstractMCDTSOptimGoal, Yss::Dataset{D, T},
                τs, w::Int, τ_vals, ts_vals, L_old; kwargs...) where {D, T}

    Ys = DelayEmbeddings.standardize(Yss)

    # compute actual embedding trajectory Y_act
    if typeof(optimalg.Γ) == Prediction_error
        Y_act = genembed(Ys, τ_vals .* (-1), ts_vals) # ensure causality for forecasts
    else
        Y_act = genembed(Ys, τ_vals, ts_vals)
    end

    # compute potential delay values with corresponding time series values and
    # Loss-values
    τ_pots, ts_pots, L_pots, temps = embedding_cycle(optimalg, Y_act, Ys, τs, w, τ_vals, ts_vals; kwargs...)

    # transform array of arrays to a single array
    τ_pot = reduce(vcat, τ_pots)
    ts_pot = reduce(vcat, ts_pots)
    L_pot = reduce(vcat, L_pots)
    temps = reduce(vcat, temps)

    if isempty(τ_pot)
        flag = true
        return Int[],Int[],eltype(L_pots)[], flag, temps
    end

    taus, ts, Ls, converge, temps = get_embedding_params_according_to_loss(optimalg.Γ,
                                            τ_pot, ts_pot, L_pot, temps, L_old)

    return taus, ts, Ls, converge, temps
end

"""
    get_embedding_params_according_to_loss(Γ::AbstractLoss, τ_pot, ts_popt, L_pot, L_old)

    Helper function for [`get_potential_delays`](@ref). Computes the potential
    delay-, time series- and according Loss-values with respect to the actual loss
    in the current embedding cycle.
"""
function get_embedding_params_according_to_loss(Γ::AbstractLoss, τ_pot, ts_pot, L_pot, temp, L_old)
    threshold = Γ.threshold
    if (minimum(L_pot) ≥ L_old)
        flag = true
        return Int[],Int[],eltype(L_pot)[], flag, nothing
    elseif (minimum(L_pot) ≤ threshold)
        flag = true
        ind = L_pot .< L_old
        return τ_pot[ind],ts_pot[ind],L_pot[ind],flag, temp[ind]
    else
        flag = false
        ind = L_pot .< L_old
        return τ_pot[ind],ts_pot[ind],L_pot[ind],flag, temp[ind]
    end
end

function get_embedding_params_according_to_loss(Γ::L_statistic, τ_pot, ts_pot, L_pot, temp, L_old)
    threshold = Γ.threshold
    if minimum(L_pot) > threshold
        flag = true
        ind = L_pot .≤ threshold
        return τ_pot[ind], ts_pot[ind], L_pot[ind], flag, temp[ind]
    else
        flag = false
        ind = L_pot .≤ threshold
        return τ_pot[ind], ts_pot[ind], L_pot[ind], flag, temp[ind]
    end
end


"""
    Perform a potential embedding cycle from the multi- or univariate Dataset `Ys`.
    Return the possible delays `τ_pot`, the associated time series `ts_pot` and
    the corresponding L-statistic-values, `L_pot` for each peak, i.e. for each
    (`τ_pot`, `ts_pot`) pair. If `FNN=true`, `L_pot` stores the corresponding
    fnn-statistic-values.
"""
function embedding_cycle(optimalg::AbstractMCDTSOptimGoal, Y_act, Ys, τs,
                                                    w, τ_vals, ts_vals; kwargs...)

    # Compute Delay-pre-selection method according to `optimalg.Λ`
    delay_pre_selection_statistic = get_delay_statistic(optimalg.Λ, Ys, τs, w, τ_vals, ts_vals; kwargs... )

    # update τ_vals, ts_vals, Ls, ε★s
    τ_pot, ts_pot, L_pot, temps = pick_possible_embedding_params(optimalg.Γ, optimalg.Λ, delay_pre_selection_statistic, Y_act, Ys, τs, w, τ_vals, ts_vals; kwargs...)

    return τ_pot, ts_pot, L_pot, temps
end


"""
    Compute all possible τ-values (and according time series numbers) and their
    corresponding Loss-statistics for the input delay_pre_selection_statistic `dps`.
"""
function pick_possible_embedding_params(Γ::AbstractLoss, Λ::AbstractDelayPreselection, dps, Y_act::Dataset{D, T}, Ys, τs, w::Int, τ_vals, ts_vals; kwargs...) where {D, T}
    L_pots = []
    τ_pots = []
    ts_pots = []
    temps = []

    for ts = 1:size(Ys,2)
        # compute loss and its corresponding index w.r.t `delay_pre_selection_statistic`

        # zero-padding of dps in order to also cover τ=0 (important for the multivariate case)
        L_trials, max_idx, temp = compute_loss(Γ, Λ, vec([0; dps[:,ts]]), Y_act, Ys, τs, w, ts, τ_vals, ts_vals; kwargs...)
        tt = τs[max_idx.-1]
        if typeof(tt)==Int
            tt = [tt]
        end
        Base.push!(L_pots, L_trials)
        Base.push!(ts_pots, fill(ts,length(L_trials)))
        Base.push!(τ_pots, tt)
        Base.push!(temps, temp)
    end
    return τ_pots, ts_pots, L_pots, temps
end

"""
    Compute the loss of a given delay-preselection statistic `dps` and the loss
    determined by `optimalg.Γ`.
"""
compute_loss

"""
    Return the loss based on the maximum decrease of the L-statistic `L_decrease` and corresponding
    delay-indices `max_idx` for all local maxima in ε★
"""
function compute_loss(Γ::L_statistic, Λ::AbstractDelayPreselection, dps::Vector{P}, Y_act::Dataset{D, T}, Ys, τs, w::Int, ts::Int, τ_vals, ts_vals; metric=Euclidean(), kwargs...) where {P, D, T}

    KNN = Γ.KNN
    tws = Γ.tws
    s = Ys[:,ts]

    max_idx = get_max_idx(Λ, dps, τ_vals, ts_vals, ts) # get the candidate delays

    L_decrease = zeros(Float64, length(max_idx))
    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        Y_trial = DelayEmbeddings.hcat_lagged_values(Y_act, s, τs[τ_idx-1])
        # compute L-statistic for Y_act and Y_trial and get the maximum decrease
        L_decrease[i] = MCDTS.uzal_cost_pecuzal_mcdts(Y_act, Y_trial, τs[end]; K = KNN,
                                w = w, metric = metric, tws = tws)
    end
    return L_decrease, max_idx, [nothing for i in max_idx]
end

"""
    Return the loss based on the FNN-statistic `FNN` and indices `max_idx`  for all local maxima in dps
"""
function compute_loss(Γ::FNN_statistic, Λ::AbstractDelayPreselection, dps::Vector{P}, Y_act::Dataset{D, T}, Ys, τs, w::Int, ts::Int, τ_vals, ts_vals; metric=Euclidean(), kwargs...) where {P, D, T}

    r = Γ.r
    s = Ys[:,ts]

    max_idx = get_max_idx(Λ, dps, τ_vals, ts_vals, ts) # get the candidate delays

    FNN_trials = zeros(Float64, length(max_idx))
    ξ_trials = zeros(Float64, length(max_idx))

    # compute nearest-neighbor-distances for actual trajectory
    Y_act2 = Y_act[1:end-τs[maximum(max_idx)-1],:]
    Y_act2 = DelayEmbeddings.standardize(Y_act2)
    vtree = KDTree(Y_act2, metric)
    _, NNdist_old = DelayEmbeddings.all_neighbors(vtree, Y_act2, 1:length(Y_act2), 1, w)

    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        Y_trial = DelayEmbeddings.hcat_lagged_values(Y_act,s,τs[τ_idx-1])
        Y_trial = DelayEmbeddings.standardize(Y_trial)
        vtree = KDTree(Y_trial, metric)
        _, NNdist_new = DelayEmbeddings.all_neighbors(vtree, Y_trial, 1:length(Y_trial), 1, w)
        # compute FNN-statistic
        FNN_trials[i] = fnn_embedding_cycle(NNdist_old, NNdist_new[1:length(NNdist_old)], r)
    end
    return FNN_trials, max_idx, [nothing for i in max_idx]
end

"""
    Return the loss based on the negative correlation coefficient for CCM.
"""
function compute_loss(Γ::CCM_ρ, Λ::AbstractDelayPreselection, dps::Vector{P}, Y_act::Dataset{D, T}, Ys, τs, w::Int, ts::Int, τ_vals, ts_vals; metric=Euclidean(), kwargs...) where {P, D, T}

    Y_other = DelayEmbeddings.standardize(Γ.timeseries)

    max_idx = get_max_idx(Λ, dps, τ_vals, ts_vals, ts) # get the candidate delays

    ρ_CCM = zeros(Float64, length(max_idx))

    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        tau_trials = ((τ_vals.*(-1))...,(τs[τ_idx-1]*(-1)),)
        ts_trials = (ts_vals...,ts,)
        Y_trial = genembed(Ys, tau_trials, ts_trials)
        # account for value-shift due to negative lags
        Ys_other = Y_other[1+maximum(tau_trials.*(-1)):length(Y_trial)+maximum(tau_trials.*(-1))]
        # compute ρ_CCM for Y_trial and Y_other
        ρ_CCM[i], _ = MCDTS.ccm(Y_trial, Ys_other; metric = metric, w = w)

    end
    return -ρ_CCM, max_idx, [nothing for i in max_idx]
end

"""
    Return the loss based on a `Tw`-step-ahead local-prediction.
"""
function compute_loss(Γ::Prediction_error, Λ::AbstractDelayPreselection, dps::Vector{P}, Y_act::Dataset{D, T}, Ys, τs, w::Int, ts::Int, τ_vals, ts_vals; metric=Euclidean(), kwargs...) where {P, D, T}

    PredictionLoss = Γ.PredictionType.loss
    PredictionMethod = Γ.PredictionType.method
    max_idx = get_max_idx(Λ, dps, τ_vals, ts_vals, ts) # get the candidate delays

    costs = zeros(Float64, length(max_idx))
    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        tau_trials = ((τ_vals.*(-1))...,(τs[τ_idx-1]*(-1)),)
        ts_trials = (ts_vals...,ts,)
        Y_trial = genembed(Ys, tau_trials, ts_trials)
        # make a in-sample prediction for Y_trial
        prediction = make_prediction(PredictionMethod, Y_trial; K = PredictionMethod.KNN, w = w,
            Tw = PredictionMethod.Tw, metric = metric, i_cycle=length(τ_vals))
        # compute loss/costs
        costs[i] = compute_costs_from_prediction(PredictionLoss, prediction, Y_trial, PredictionMethod.Tw)
    end
    return costs, max_idx, [nothing for i in max_idx]
end


"""
    get_max_idx(Λ::AbstractDelayPreselection, dps::Vector, τ_vals, ts_vals) → max_idx

    Compute the candidate delay values from the given delay pre-selection statistic
    `dps` with respect to `Λ`, which determined how `dps` was obtained and how
    to select the candidates (e.g. pick the maxima of `dps` in case of the
    `Λ` being the Continuity function). See [`Continuity_function`](@ref) and
    [`Range_function`](@ref).
"""
function get_max_idx(Λ::Range_function, dps::Vector{T}, τ_vals, ts_vals, ts) where {T}

    max_idx = Vector(dps[2:end].+1)
    ts_idx = findall(e->e==ts, ts_vals) # do not consider already taken delays
    filter!(e->e∉(τ_vals[ts_idx] .+ 2), max_idx) # do not consider already taken delays
    return max_idx
end
function get_max_idx(Λ::Continuity_function, dps::Vector{T}, τ_vals, ts_vals, ts) where {T}

    _, max_idx = get_maxima(dps) # determine local maxima in delay_pre_selection_statistic
    return max_idx
end


"""
    make_prediction(pred_meth::AbstractPredictionMethod, Y::AbstractDataset{D, ET};
            K::Int = 3, w::Int = 1, Tw::Int = 1, metric = Euclidean()) → prediction

    Compute a in-sample `Tw`-time-steps-ahead of the data `Y`, using the prediction
    method `pred_meth`. `w` is the Theiler window and `K` the nearest neighbors used.

* `Y`: Dataset (Nt x N_embedd)
* `K`: Nearest Neighbours
* `w`: Theiler window
* `Tw`: Prediction horizon
* `metric`: Metric for NN search
* `i_cycle`: Which embedding cycling we are predicting for
"""
function make_prediction(pred_meth::AbstractLocalPredictionMethod{:zeroth}, Y::AbstractDataset{D, ET}; K::Int = 3, w::Int = 1,
    Tw::Int = 1, metric = Euclidean(), i_cycle::Int=1) where {D, ET}

    NN = length(Y)-Tw;
    ns = 1:NN  # the fiducial point indices
    vs = Y[ns] # the fiducial points in the data set
    vtree = KDTree(Y[1:end-Tw], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, w)

    prediction = zeros(ET, NN, D)
    # loop over each fiducial point
    for (i,v) in enumerate(vs)
        NNidxs = allNNidxs[i] # indices of k nearest neighbors to v

        ϵ_ball = zeros(ET, K, D) # preallocation
        # determine neighborhood one time step ahead
        @inbounds for (k, j) in enumerate(NNidxs)
            ϵ_ball[k, :] .= Y[j + Tw]
        end
        # take the average as a prediction
        prediction[i,:] = mean(ϵ_ball; dims=1)
    end
    return Dataset(prediction)
end
function make_prediction(pred_meth::AbstractLocalPredictionMethod{:linear}, Y::AbstractDataset{D, ET}; K::Int = 3, w::Int = 1,
    Tw::Int = 1, metric = Euclidean(), i_cycle::Int=1) where {D, ET}

    K = 2*(size(Y,2)+1)
    NN = length(Y)-Tw;
    ns = 1:NN  # the fiducial point indices
    vs = Y[ns] # the fiducial points in the data set
    vtree = KDTree(Y[1:end-Tw], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, w)

    prediction = zeros(ET, NN, D)
    # loop over each fiducial point
    for (i,v) in enumerate(vs)
        NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
        A = ones(K,D) # datamatrix for later linear equation to solve for AR-process
        ϵ_ball = zeros(ET, K, D) # preallocation
        # determine neighborhood one time step ahead
        @inbounds for (k, j) in enumerate(NNidxs)
            ϵ_ball[k, :] .= Y[j + Tw]
            A[k,:] = Y[j]
        end
        b  = zeros(D)
        ar_coeffs = zeros(D, D)
        namess = ["X"*string(z) for z = 1:D]
        ee = Meta.parse.(namess)
        formula_expression = Term(:Y) ~ sum(term.(ee))

        for j = 1:D
            data = DataFrame()
            for (cnt,var) in enumerate(namess)
                data[!, var] = A[:,cnt]
            end
            data.Y = ϵ_ball[:,j]

            ols = lm(formula_expression, data)
            b[j] = coef(ols)[1]
            for k = 1:D
                ar_coeffs[j,k] = coef(ols)[k+1]
            end
            prediction[i,j] = Y[ns[i],:]'*ar_coeffs[j,:] + b[j]
        end
    end
    return Dataset(prediction)
end

"""
    Compute the in-sample prediction costs based on the loss-metric determined
    by PredictionLoss
"""
function compute_costs_from_prediction(PredictionLoss::AbstractPredictionLoss{1}, prediction::AbstractDataset{D, T},
                            Y::AbstractDataset{D, T}, Tw::Int) where {D, T}

    NN = length(Y)-Tw;
    ns = 1:NN  # the fiducial point indices
    costs = zeros(T, NN, D)
    @inbounds for i = 1:NN
        costs[i,:] = (Vector(prediction[i]) .- Vector(Y[ns[i]+Tw])).^2
    end
    c = sqrt.(mean(costs; dims=1))
    return c[1]
end
function compute_costs_from_prediction(PredictionLoss::AbstractPredictionLoss{2}, prediction::AbstractDataset{D, T},
                            Y::AbstractDataset{D, T}, Tw::Int) where {D, T}

    NN = length(Y)-Tw;
    ns = 1:NN  # the fiducial point indices
    costs = zeros(T, NN, D)
    @inbounds for i = 1:NN
        costs[i,:] = (Vector(prediction[i]) .- Vector(Y[ns[i]+Tw])).^2
    end
    c = sqrt.(mean(costs; dims=1))
    return mean(c)
end
function compute_costs_from_prediction(PredictionLoss::AbstractPredictionLoss{3}, prediction::AbstractDataset{D, T},
                            Y::AbstractDataset{D, T}, Tw::Int) where {D, T}

    NN = length(Y)-Tw;
    ns = 1:NN  # the fiducial point indices
    costs = zeros(D)
    for j = 1:D
        costs[j] = compute_KL_divergence(Vector(prediction[:,j]),Y[ns .+ Tw,j])
    end
    return costs[1]
end
function compute_costs_from_prediction(PredictionLoss::AbstractPredictionLoss{4}, prediction::AbstractDataset{D, T},
                            Y::AbstractDataset{D, T}, Tw::Int) where {D, T}

    NN = length(Y)-Tw;
    ns = 1:NN  # the fiducial point indices
    costs = zeros(D)
    for j = 1:D
        costs[j] = compute_KL_divergence(Vector(prediction[:,j]),Y[ns .+ Tw,j])
    end
    return mean(costs)
end


"""
    zeroth_prediction_cost(Y::Dataset; kwargs...) → Cost

    Compute the mean squared one-step prediction error `Cost` of the Dataset `Y`.
    The prediction is based on [`local_zeroth_prediction`](@ref).

    ## Keyword arguments

    * `samplesize = 1.0`: Number of considered fiducial points v as a fraction of
      input state space trajectory `Y`'s length, in order to average the conditional
      variances and neighborhood sizes (read algorithm description) to produce `L`.
    * `K = 3`: the amount of nearest neighbors considered, in order to compute the
      mean squared prediction error (read algorithm description).
    * `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
      state space trajectory `Y.
    * `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
      that are excluded from being true neighbors). `w=0` means to exclude only the
      point itself, and no temporal neighbors.
    * `Tw = 1`: The time horizon for predictions. The `Cost` is the average error
      over these timesteps.
"""
function zeroth_prediction_cost(Y::AbstractDataset{D, ET};
        K::Int = 3, w::Int = 1, Tw::Int = 1,samplesize::Real = 1.0,
        metric = Euclidean()) where {D, ET}

    # select a random state space vector sample according to input samplesize
    NN = length(Y)-Tw;
    NNN = floor(Int, samplesize*NN)
    if samplesize < 1
        ns = sample(1:NN, NNN; replace=false) # the fiducial point indices
    else
        ns = 1:NN  # the fiducial point indices
    end

    vs = Y[ns] # the fiducial points in the data set

    vtree = KDTree(Y[1:end-Tw], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, w)

    error = zeros(ET, NN, D)
    # loop over each fiducial point
    for (i,v) in enumerate(vs)
        NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
        errors = zeros(ET, Tw, D)
        for T = 1:Tw
            ϵ_ball = zeros(ET, K, D) # preallocation
            # determine neighborhood one time step ahead
            @inbounds for (k, j) in enumerate(NNidxs)
                ϵ_ball[k, :] .= Y[j + T]
            end
            # take the average as a prediction
            prediction = zeros(ET, D)
            prediction[:] = mean(ϵ_ball; dims=1)
            errors[T,:] = (Vector(prediction) .- Vector(Y[ns[i]+T])).^2
        end
        error[i,:] = mean(errors; dims=1)
    end
    return sqrt.(mean(error; dims=1))
end


"""
    zeroth_prediction_cost_KL(Y::Dataset; kwargs...) → Cost
Compute the KL-divergence `Cost` of the Dataset `Y`.
The prediction is based on [`local_zeroth_prediction`](@ref).

## Keyword arguments

* `samplesize = 1.0`: Number of considered fiducial points v as a fraction of
  input state space trajectory `Y`'s length, in order to average the conditional
  variances and neighborhood sizes (read algorithm description) to produce `L`.
* `K = 3`: the amount of nearest neighbors considered, in order to compute the
  mean squared prediction error (read algorithm description).
* `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
  state space trajectory `Y.
* `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `Tw = 1`: The time horizon for predictions. The `Cost` is the average error
  over these timesteps.

"""
function zeroth_prediction_cost_KL(Y::AbstractDataset{D, ET};
        K::Int = 3, w::Int = 1, Tw::Int = 1,samplesize::Real = 1.0,
        metric = Euclidean()) where {D, ET}

    # select a random state space vector sample according to input samplesize
    NN = length(Y)-Tw;
    NNN = floor(Int, samplesize*NN)
    if samplesize < 1
        ns = sample(1:NN, NNN; replace=false) # the fiducial point indices
    else
        ns = 1:NN  # the fiducial point indices
    end
    vs = Y[ns] # the fiducial points in the data set
    vtree = KDTree(Y[1:end-Tw], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, w)

    KL_distances = zeros(ET, Tw, D)

    for T = 1:Tw
        predictions = zeros(ET, NN, D)
        # loop over each fiducial point
        for (i,v) in enumerate(vs)
            NNidxs = allNNidxs[i] # indices of k nearest neighbors to v

            ϵ_ball = zeros(ET, K, D) # preallocation
            # determine neighborhood one time step ahead
            @inbounds for (k, j) in enumerate(NNidxs)
                ϵ_ball[k, :] .= Y[j + T]
            end
            # take the average as a prediction
            predictions[i,:] = mean(ϵ_ball; dims=1)

        end
        # compute KL-divergence for each component of the prediction
        for j = 1:D
            KL_distances[T,j] = compute_KL_divergence(Vector(predictions[:,j]),Y[ns .+ T,j])
        end
    end
    return mean(KL_distances; dims=1)
end

"""
    linear_prediction_cost_KL(Y::Dataset; kwargs...) → Cost
Compute the KL-divergence `Cost` of the Dataset `Y`.
The prediction is based on [`local_linear_prediction`](@ref).

## Keyword arguments

* `samplesize = 1.0`: Number of considered fiducial points v as a fraction of
  input state space trajectory `Y`'s length, in order to average the conditional
  variances and neighborhood sizes (read algorithm description) to produce `L`.
* `K = 3`: the amount of nearest neighbors considered, in order to compute the
  mean squared prediction error (read algorithm description).
* `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
  state space trajectory `Y.
* `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `Tw = 1`: The time horizon for predictions. The `Cost` is the average error
  over these timesteps.

"""
function linear_prediction_cost_KL(Y::AbstractDataset{D, ET};
        K::Int = 3, w::Int = 1, Tw::Int = 1, samplesize::Real = 1.0,
        metric = Euclidean()) where {D, ET}

    # select a random state space vector sample according to input samplesize
    NN = length(Y)-Tw;
    NNN = floor(Int, samplesize*NN)
    if samplesize < 1
        ns = sample(1:NN, NNN; replace=false) # the fiducial point indices
    else
        ns = 1:NN  # the fiducial point indices
    end
    vs = Y[ns] # the fiducial points in the data set

    vtree = KDTree(Y[1:end-Tw], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, w)

    KL_distances = zeros(ET, Tw, D)

    for T = 1:Tw
        predictions = zeros(ET, NN, D)
        # loop over each fiducial point
        for (i,v) in enumerate(vs)
            NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
            A = ones(K,D) # datamatrix for later linear equation to solve for AR-process
            ϵ_ball = zeros(ET, K, D) # preallocation
            # determine neighborhood one time step ahead
            @inbounds for (k, j) in enumerate(NNidxs)
                ϵ_ball[k, :] .= Y[j + T]
                A[k,:] = Y[j]
            end

            b  = zeros(D)
            ar_coeffs = zeros(D, D)
            namess = ["X"*string(z) for z = 1:D]
            ee = Meta.parse.(namess)
            formula_expression = Term(:Y) ~ sum(term.(ee))

            for j = 1:D
                data = DataFrame()
                for (cnt,var) in enumerate(namess)
                    data[!, var] = A[:,cnt]
                end
                data.Y = ϵ_ball[:,j]

                ols = lm(formula_expression, data)
                b[j] = coef(ols)[1]
                for k = 1:D
                    ar_coeffs[j,k] = coef(ols)[k+1]
                end
                predictions[i,j] = Y[ns[i],:]'*ar_coeffs[j,:] + b[j]
            end

        end
        # compute KL-divergence for each component of the prediction
        for j = 1:D
            KL_distances[T,j] = compute_KL_divergence(Vector(predictions[:,j]),Y[ns .+ T,j])
        end
    end
    return mean(KL_distances; dims=1)

end


"""
    linear_prediction_cost(Y::Dataset; kwargs...) → Cost
Compute the mean squared one-step prediction error `Cost` of the Dataset `Y`.
The prediction is based on [`local_linear_prediction`](@ref).

## Keyword arguments

* `samplesize = 1.0`: Number of considered fiducial points v as a fraction of
  input state space trajectory `Y`'s length, in order to average the conditional
  variances and neighborhood sizes (read algorithm description) to produce `L`.
* `K = 3`: the amount of nearest neighbors considered, in order to compute the
  mean squared prediction error (read algorithm description).
* `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
  state space trajectory `Y.
* `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `Tw = 1`: The time horizon for predictions. The `Cost` is the average error
  over these timesteps.

"""
function linear_prediction_cost(Y::AbstractDataset{D, ET};
        K::Int = 3, w::Int = 1, Tw::Int = 1, samplesize::Real = 1.0,
        metric = Euclidean()) where {D, ET}

    # select a random state space vector sample according to input samplesize
    NN = length(Y)-Tw;
    NNN = floor(Int, samplesize*NN)
    if samplesize < 1
        ns = sample(1:NN, NNN; replace=false) # the fiducial point indices
    else
        ns = 1:NN  # the fiducial point indices
    end

    vs = Y[ns] # the fiducial points in the data set

    vtree = KDTree(Y[1:end-Tw], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, w)

    error = zeros(ET, NN, D)
    # loop over each fiducial point
    for (i,v) in enumerate(vs)
        NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
        errors = zeros(ET, Tw, D)
        for T = 1:Tw
            A = ones(K,D) # datamatrix for later linear equation to solve for AR-process
            ϵ_ball = zeros(ET, K, D) # preallocation
            # determine neighborhood one time step ahead
            @inbounds for (k, j) in enumerate(NNidxs)
                ϵ_ball[k, :] .= Y[j + T]
                A[k,:] = Y[j]
            end
            # make local linear model of the last point of the trajectory
            prediction = zeros(D)
            b  = zeros(D)
            ar_coeffs = zeros(D, D)
            namess = ["X"*string(z) for z = 1:D]
            ee = Meta.parse.(namess)
            formula_expression = Term(:Y) ~ sum(term.(ee))

            for j = 1:D
                data = DataFrame()
                for (cnt,var) in enumerate(namess)
                    data[!, var] = A[:,cnt]
                end
                data.Y = ϵ_ball[:,j]

                ols = lm(formula_expression, data)
                b[j] = coef(ols)[1]
                for k = 1:D
                    ar_coeffs[j,k] = coef(ols)[k+1]
                end
                prediction[j] = Y[ns[i],:]'*ar_coeffs[j,:] + b[j]
            end
            errors[T,:] = (prediction .- Y[ns[i]+T]).^2
        end
        error[i,:] = mean(errors; dims=1)
    end
    return vec(sqrt.(mean(error; dims=1)))
end

"""
    Compute the Kullback-Leibler-Divergence of the two Vectors `a` and `b`.
"""
function compute_KL_divergence(a::Vector{T}, b::Vector{T}) where {T}
    # normalization
    a /= maximum(a)
    b /= maximum(b)
    # make histograms
    h1 = fit(Histogram, a)
    edges = h1.edges
    h2 = fit(Histogram, b, edges...)
    # get the probabilities
    pdf1 = h1.weights / sum(h1.weights)
    pdf2 = h2.weights / sum(h2.weights)
    # compute KL-divergence
    return kl_divergence(pdf2, pdf1)
end


"""
    ccm(X, y; kwargs...) → ρ, y_hat

    Compute the convergent crossmapping (CCM) (Sugihara et al. 2012) of a
    vector time series `X` (an embedded time series `x`) on the time series `y`
    NOTE: 'X' and 'y' must have the same length and you have to make sure that
    'y' starts at the same time index as 'X' does. - When using [`genembed`](@ref)
    with negative delays to construct `X` from `x`, which is mandatory here, then
    'y' needs to be shifted by the largest negative delay value, which has been
    used to construct `X`.

    Returns the correlation coefficient of `y` and its predicted values for `y_hat`,
    based on the nearest neighbour structure of `X`.
    It is said that 'y' causes 'x', if ρ increases with increasing time series
    length AND ρ is "quite high".

    Keyword arguments:
    *`metric = Euclidean()`: The metric for vector distance computation.
    *`w::Int = 1`: The Theiler window in sampling units.
    *`lags::Array = [0]`: The lag for the cross mapping, in order to detect time lagged
                          causal relationships. The output ρ is an array of size
                          `length(lags)`, the output Y_hat is the one corresponding
                          to a lag of zero.

"""
function ccm(X::Dataset{D,T},Y::Vector{T}; metric = Euclidean(), w::Int = 1,
    lags::AbstractArray = [0]) where {D,T<:Real}

    K = D+1
    @assert length(X)==length(Y)
    XX = Matrix(X)
    N = length(X)

    # for potential later extension to sampled batches of fraction samplesize
    samplesize = 1
    NN = floor(Int, samplesize*N)
    if samplesize == 1
        ns = 1:N # the fiducial point indices
    else
        ns = sample(1:N, NN; replace=false) # the fiducial point indices
    end

    vxs = X[ns] # the fiducial points in the data set
    vtree = KDTree(X, metric)
    allNNidxs, allNNdist = DelayEmbeddings.all_neighbors(vtree, vxs, ns, K, w)
    Y_hat = zeros(T, NN) # preallocation
    # loop over each fiducial point
    for (i,v) in enumerate(vxs)
        NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
        NNdist = allNNdist[i] # indices of k nearest neighbors to v
        # determine weights
        u = zeros(K)
        ws = zeros(K)
        @inbounds for (k, j) in enumerate(NNdist)
            u[k] = exp(-(j/NNdist[1]))
        end
        ws = u ./ sum(u)
        # compute Y_hat as a wheighted mean
        Y_hat[i] = sum(ws .* Y[NNidxs])
    end
    ρ = Statistics.cor(Y_hat, Y)
    return ρ, Dataset(Y_hat)
end



## Methods for delay preselection stats

"""
    get_delay_statistic(optimalg.Λ<: AbstractDelayPreselection, Ys, τs, w, τ_vals, ts_vals; kwargs... )

    Compute the delay statistic according to the chosen method in `optimalg.Λ` (see [`MCDTSOptimGoal`](@ref))
"""
function get_delay_statistic(Λ::Continuity_function, Ys, τs, w, τ_vals, ts_vals; metric = Euclidean(), kwargs... )
    ε★, _ = pecora(Ys, Tuple(τ_vals), Tuple(ts_vals); delays = τs, w = w,
            samplesize = Λ.samplesize, K = Λ.K, metric = metric, α = Λ.α,
            p = Λ.p, undersampling = false)
    return ε★
end

function get_delay_statistic(Λ::Range_function, Ys, τs, w, τ_vals, ts_vals; kwargs... )
    return repeat(Vector(1:length(τs)), outer = [1,size(Ys,2)])
end


## Others

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
            Base.push!(maximas, s[i])
            Base.push!(maximas_idx, i)
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
