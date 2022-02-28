
"""
    Perform the RecurrenceAnalysis of some reconstruction trajectories `Y₁`, `Y₂`,
    `Y₃`. Specifically, compute the fraction of recurrence rates from the
    "original"/reference trajectory `Y_ref` with the one from the JRP of the
    original `Y₁`, `Y₂`, `Y₃` together with the reconstructed trajectory. Also
    compute RQA quantifiers of the recurrence plots of `Y₁`, `Y₂`, `Y₃` and `Y_ref`.

    ## Keyword arguments:
    *`ε = 0.05`: The used threshold for constructing the recurrence plots
        The reconstruction method is fixed recurrence rate.
    *`w = 1`: Theiler window used for all Datasets
    *`lmin = 2`: Minimum used line length for digaonal line based RQA measures.
    *`kNN = 1`: The number of nearest neighbors used for obtaining the mutual
        nearest neighbors measure
"""
function perform_recurrence_analysis(Y_ref::Dataset, Y₁::Dataset,
                        Y₂::Dataset, Y₃::Dataset;
                        ε::Real = 0.05, w::Int = 1, lmin::Int = 2, kNN::Int = 1)

    N1 = length(Y₁)
    N2 = length(Y₂)
    N3 = length(Y₃)
    N = minimum(hcat(N1, N2, N3))

    R_ref = RecurrenceMatrix(Y_ref[1:N,:], ε; metric = "euclidean", fixedrate = true)
    R1 = RecurrenceMatrix(Y₁[1:N,:], ε; fixedrate = true)
    R2 = RecurrenceMatrix(Y₂[1:N,:], ε; fixedrate = true)
    R3 = RecurrenceMatrix(Y₃[1:N,:], ε; fixedrate = true)

    f1 = jrp_rr_frac(R_ref, R1)
    f2 = jrp_rr_frac(R_ref, R2)
    f3 = jrp_rr_frac(R_ref, R3)

    mfnn1 = mfnn(Y_ref[1:N,:], Y₁[1:N,:]; w = w, kNN = kNN)
    mfnn2 = mfnn(Y_ref[1:N,:], Y₂[1:N,:]; w = w, kNN = kNN)
    mfnn3 = mfnn(Y_ref[1:N,:], Y₃[1:N,:]; w = w, kNN = kNN)

    RQA_ref = rqa(R_ref; theiler = w, lmin = lmin)
    RQA1 = rqa(R1; theiler = w, lmin = lmin)
    RQA2 = rqa(R2; theiler = w, lmin = lmin)
    RQA3 = rqa(R3; theiler = w, lmin = lmin)

    trans_ref =

    return mfnn1, mfnn2, mfnn3, f1, f2, f3, RQA_ref, RQA1, RQA2,
                                            RQA3, R_ref, R1, R2, R3
end


"""
    Computes the similarity between recurrence plots `RP₁` and `RP₂`. Outputs the
    fraction of recurrences rates gained from RP₁ and of the joint recurrence
    plot `RP₁ .* RP₂`.
"""
function jrp_rr_frac(RP₁::RecurrenceMatrix, RP₂::RecurrenceMatrix)
    @assert size(RP₁) == size(RP₂)

    RR1 = sum(RP₁)/(size(RP₁,1)*size(RP₁,1))
    JRP = JointRecurrenceMatrix(RP₁, RP₂)
    RR2 = sum(JRP)/(size(JRP,1)*size(JRP,1))

    f = RR2 / RR1
    return f
end

"""
    Computes the mututal false nearest neighbours (mfnn) for a reference trajectory
    `Y_ref` and a reconstruction `Y_rec` after [^Rulkov1995].

    ## Keyword arguments
    *`w = 1`: Theiler window for the surpression of serially correlated neighbors in
        the nearest neighbor-search
    *`kNN = 1`: The number of considered nearest neighbours (in the paper always 1)

    [^Rulkov1995]: Rulkov, Nikolai F. and Sushchik, Mikhail M. and Tsimring, Lev S. and Abarbanel, Henry D.I. (1995). [Generalized synchronization of chaos in directionally coupled chaotic systems. Physical Review E 51, 980](https://doi.org/10.1103/PhysRevE.51.980).
"""
function mfnn(Y_ref::Dataset, Y_rec::Dataset; w::Int = 1, kNN::Int = 1)

    @assert length(Y_ref) == length(Y_rec)
    @assert kNN > 0
    N = length(Y_ref)
    metric = Euclidean()

    # compute nearest neighbor distances for both trajectories
    vtree = KDTree(Y_ref, metric)
    allNNidxs_ref, _ = DelayEmbeddings.all_neighbors(vtree, Y_ref,
                                                        1:length(Y_ref), kNN, w)
    vtree = KDTree(Y_rec, metric)
    allNNidxs_rec, _ = DelayEmbeddings.all_neighbors(vtree, Y_rec,
                                                        1:length(Y_rec), kNN, w)

    F = zeros(N)
    factor1_nom = zeros(kNN)
    factor1_denom = zeros(kNN)
    factor2_nom = zeros(kNN)
    factor2_denom = zeros(kNN)
    for i = 1:N
        for j = 1:kNN
            factor1_nom[j] = evaluate(Euclidean(), Y_rec[i], Y_rec[allNNidxs_ref[i][j]])
            factor1_denom[j] = evaluate(Euclidean(), Y_ref[i], Y_ref[allNNidxs_ref[i][j]])
            factor2_nom[j] = evaluate(Euclidean(), Y_ref[i], Y_ref[allNNidxs_rec[i][j]])
            factor2_denom[j] = evaluate(Euclidean(), Y_rec[i], Y_rec[allNNidxs_rec[i][j]])
        end
        factor1 = sum(factor1_nom)/sum(factor1_denom)
        factor2 = sum(factor2_nom)/sum(factor2_denom)
        F[i] = factor1*factor2                                         # Eq.(27)
    end
    return mean(F)
end


"""
    Generate data from a AR(1) process for a initial value `u0`, a AR-coefficient
    `α` and a white noise scaling parameter `p`. Return a time series of length `N`.
"""
function ar_process(u0::T, α::T, p::T, N::Int) where {T<:Real}
    x = zeros(T, N+10)
    x[1] = u0
    for i = 2:N+10
        x[i] = α*x[i-1] + p*randn()
    end
    return x[11:end]
end


"""
    compute_delta_L(s, τs, (js,) T_max; KNN = 3, w = 1, metric = Euclidean) → ΔL

    Compute the overall L-decrease `ΔL` of a given embedding of the time series
    `s::Vector` with the delay values `τs` up to a maximum `T`-value `T_max`. We
    respect the Theiler window `w`, the chosen `metric` and the number of considered
    nearest neighbors `KNN`. It is also possible to compute `ΔL` for a multivariate
    input `Y::Dataset`. Then one additionally needs to supply a vector `js`, which
    lists the chosen time series corresponding to the given delay values in `τs`.
    This is similar to the procedure in [`genembed`]@ref. The computations are based
    on z-standardized input for ensuring comparability.
"""
function compute_delta_L(s::Vector{T}, τs::Vector{Int}, T_max::Int;
        KNN::Int = 3, w::Int = 1, metric = Euclidean(), tws::AbstractRange{Int} = 2:T_max
    ) where {T}
    ts = DelayEmbeddings.standardize(Dataset(s))
    tss = vec(Matrix(Dataset(ts)))
    ΔL = 0
    Y_act = ts
    for i = 1 : (length(τs)-1)
        τ = τs[i+1]
        # embedding one cycle
        Y_next = DelayEmbeddings.hcat_lagged_values(Y_act, tss, τ)
        # compute ΔL for this cycle
        ΔL += uzal_cost_pecuzal_mcdts(Y_act, Y_next, T_max; K = KNN, w,
                                                    metric, tws)
        Y_act = Y_next
    end
    return ΔL
end

function compute_delta_L(Y::Dataset{D, T}, τs::Vector{Int}, js::Vector{Int},
        T_max::Int; KNN::Int = 3, w::Int = 1, metric = Euclidean(),
        tws::AbstractRange{Int} = 2:T_max) where {D, T}
    @assert length(τs) == length(js)
    ts = DelayEmbeddings.standardize(Y)
    ΔL = 0
    Y_act = Dataset(ts[:,js[1]])
    for i = 1: (length(τs)-1)
        τ = τs[i+1]
        # embedding one cycle
        Y_next = DelayEmbeddings.hcat_lagged_values(Y_act, ts[:,js[i+1]], τ)
        # compute ΔL for this cycle
        ΔL += uzal_cost_pecuzal_mcdts(Y_act, Y_next, T_max; K = KNN, w,
                                                    metric, tws)
        Y_act = Y_next
    end
    return ΔL
end

"""
    Simple nonlinear noise reduction algorithm by Schreiber 1993

    Params:
    *`m`           denotes the local embedding dimension
    *`epsilon`     denotes the local neighborhood size as number of neighbours

    Returns:
    the filtered signal

    Note that by applying this filter, there will be lost `m-1` datapoints.
    We therefore phase-shift each datapoint in the resulting signal by `(m-1)/2`.
"""
function nonlin_noise_reduction(x::Vector{T}, m::Int, epsilon::Int
    ) where {T}

    if m < 2
        error("Parameter m must be a positive integer value larger that 1.")
    end
    if epsilon <= 0
        error("Parameter epsilon must be a positive integer value larger than 0")
    end

    # Embed the data
    Y = DelayEmbeddings.embed(x, m, 1)
    # Estimate Theiler window
    theiler = DelayEmbeddings.estimate_delay(x, "mi_min")

    filtered_signal = zeros(length(Y))

    # Distances and nearest neighbors
    vtree = KDTree(Y, Chebyshev())
    NNdist_idx, _ = DelayEmbeddings.all_neighbors(vtree, Y, 1:length(Y),
                                                            epsilon, theiler)

    # iterate over each point in phase space
    for i = 1:length(Y)
        dd = 0
        for idx in NNdist_idx[i]
            dd += Y[idx,Int(ceil(m/2))]
        end
        filtered_signal[i] = dd / epsilon
    end
    # phase shift / correction of the output signal
    output = NaN*ones(length(x))
    output[1+Int(floor((m-1)/2)):end-Int(ceil((m-1)/2))] = filtered_signal

    return output
end

"""
    Moving average of a timeseries `vs` over a window `n`
"""
moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

"""
    local_random_analogue_prediction(Y::Dataset, K::Int; kwargs...) → Y_predict
Compute a one step ahead prediction `Y_predict` of the input `Y`, based on `K`
nearest neighbors. Here the prediction is a random pick from all `K`-nearest
neighbour images and, thus, invokes some kind of randomness.

Keywords:
* `metric = Euclidean()`: Metric used for distance computation
* `theiler::Int = 1`: Theiler window for excluding serially correlated points from
   the nearest neighbour search.
"""
function local_random_analogue_prediction(Y::Dataset{D,T}, K::Int = 5;
    metric = Euclidean(), theiler::Int = 1) where {D,T}

    Tw = 1
    NN = length(Y)
    ns = 1:NN
    vs = Y[ns]
    vtree = KDTree(Y, metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, theiler)

    ϵ_ball = zeros(T, K, D) # preallocation
    # loop over each fiducial point
    NNidxs = allNNidxs[end] # indices of k nearest neighbors to v
    # determine neighborhood `Tw` time steps ahead
    @inbounds for (i, j) in enumerate(NNidxs)
        ϵ_ball[i, :] .= Y[j + Tw]
    end
    # take a random sample as the prediction
    idx = sample(vec(1:K))
    prediction = ϵ_ball[idx,:]

    return vec(prediction)
end


"""
    iterated_local_zeroth_prediction_embed(Y::Dataset, τs::Vector, K::Int = 5, Tw::Int = 2; kwargs...) → Y_predict
Perform an iterated one step forecast over `Tw` time steps using the local linear
prediction algorithm. `Y_predict` is a Dataset of length `Tw` and dimension like
`Y`. In contrast to `iterated_local_linear_prediction()` we here use the time
delays `τs` to reconstruct all components of a predicted trajectory point from
the 1st component, which is obtained from the local model. This only works for
univariate embeddings.

Keywords:
* `metric = Euclidean()`: Metric used for distance computation
* `theiler::Int = 1`: Theiler window for excluding serially correlated points from
   the nearest neighbour search.
* `verbose::Bool = false`: When set to `true`, the function prints the actual time
  step, which it is computing.
"""
function iterated_local_zeroth_prediction_embed(Y::Dataset{D,T}, τs::Vector, K::Int = 5, Tw::Int = 2;
    metric = Euclidean(), theiler::Int = 1, verbose::Bool = false) where {D,T}

    @assert Tw > 1 "Time horizon must be a positive integer"
    N = length(Y)
    d = size(Y,2)
    @assert length(τs) == d "Vector storing the delays must have the same dimensionality as the Input trajectory."
    @assert τs[1] == 0 "Vector storing the delays must have 0 as its first entry."
    @assert sum(τs.<0) == 0 "Vector storing the delays must have 0 as its first entry."
    predicted_trajectory = deepcopy(Y)
    for Th = 1:Tw
        if verbose
            println("Compute prediction for time step $Th")
        end
        # iterated one step
        predicted, _ = MCDTS.local_zeroth_prediction(predicted_trajectory, K;
                                            theiler, metric)
        for i = 2:d
            predicted[i] = predicted_trajectory[end-τs[i],1]
        end
        push!(predicted_trajectory, predicted)
    end
    return Dataset(predicted_trajectory[N+1:end])
end

"""
    Compute the mean squared error between `prediction` and `reference`
"""
function compute_mse(prediction::Vector{T}, reference::Vector{T}) where {T}
    return sqrt(mean((prediction .- reference).^2))
end

"""
    Compute the total absolute error between `prediction` and `reference`
"""
function compute_abs_err(prediction::Vector{T}, reference::Vector{T}) where {T}
    return sum(abs.(prediction .- reference))
end

"""
    Compute the scaling term for the MASE measure. This is tge average in-sample
    forecast error for a random-walk prediction, which uses the previous value in
    the observed signal `x` as the forecast. `Tw` is the prediction time horizon.
"""
function rw_norm(x::Vector{T}, Tw::Int) where {T}
    N = length(x)
    xx = sum(abs.(diff(x)))
    xx *= (Tw/length(xx))
    return xx
end

"""
    local_zeroth_prediction(Y::Dataset, K::Int = 5; kwargs...) → x_pred, e_expect

    Perform a "zeroth" order prediction for the time horizon `Tw` (default = 1). Based
    on `K` nearest neighbours of the last point of the given trajectory `Y`, the
    `Tw`-step ahead prediction is simply the mean of the images of these `K`-nearest
    neighbours. The output `x_pred` is, thus, the `Tw`-step ahead prediction vector.
    The function also returns `e_expect`, the expected error on the prediction `x_pred`,
    computed as the mean of the RMS-errors of all `K`-neighbours-errors.

    Keywords:
    * `metric = Euclidean()`: Metric used for distance computation
    * `theiler::Int = 1`: Theiler window for excluding serially correlated points from
       the nearest neighbour search.
    * `Tw::Int = 1`: The prediction time in sampling units. If `Tw > 1`, a multi-step
      prediction is performed.

"""
function local_zeroth_prediction(Y::Dataset{D,T}, K::Int = 5;
    metric = Euclidean(), theiler::Int = 1, Tw::Int = 1) where {D,T}

    NN = length(Y)
    ns = 1:NN
    vs = Y[ns]
    vtree = KDTree(Y[1:length(Y)-Tw,:], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, theiler)

    ϵ_ball = zeros(T, K, D) # preallocation
    # consider NNs of the very last point of the trajectory
    NNidxs = allNNidxs[end] # indices of k nearest neighbors to v
    # determine neighborhood `Tw` time steps ahead
    @inbounds for (i, j) in enumerate(NNidxs)
        ϵ_ball[i, :] .= Y[j + Tw]
    end
    # take the average as a prediction
    prediction = mean(ϵ_ball; dims=1)
    # predicted prediction error
    error_predict = sum((ϵ_ball .- prediction).^2; dims=1)

    return vec(prediction), vec(error_predict)
end

"""
    iterated_local_zeroth_prediction(Y::Dataset, K::Int = 5, Tw::Int = 2; kwargs...) → Y_predict
    Perform an iterated one step forecast over `Tw` time steps using the local zeroth
    prediction algorithm. `Y_predict` is a Vector of dimension like `Y`.

    Keywords:
    * `metric = Euclidean()`: Metric used for distance computation
    * `theiler::Int = 1`: Theiler window for excluding serially correlated points from
       the nearest neighbour search.
    * `verbose::Bool = false`: When set to `true`, the function prints the actual time
      step, which it is computing.
"""
function iterated_local_zeroth_prediction(Y::Dataset{D,T}, K::Int = 5, Tw::Int = 2;
    metric = Euclidean(), theiler::Int = 1, verbose::Bool = false) where {D,T}

    @assert Tw > 0 "Time horizon must be a positive integer"
    N = length(Y)
    predicted_trajectory = deepcopy(Y)
    for Th = 1:Tw
        if verbose
            println("Compute prediction for time step $Th")
        end
        # iterated one step
        predicted, _ = MCDTS.local_zeroth_prediction(predicted_trajectory, K;
                                            theiler, metric)
        push!(predicted_trajectory, predicted)
    end
    return predicted_trajectory[N+1:end]
end

"""
    local_linear_prediction(Y::Dataset, K::Int = 5; kwargs...) → x_pred, e_expect

    Perform a prediction for the time horizon `Tw` (default = 1) by a locally linear
    fit. Based on `K` nearest neighbours of the last point of the given trajectory
    `Y`, we fit a linear model to these points and their `Tw`-step ahead images. The
    output `x_pred` is, thus, the `Tw`-step ahead prediction vector.
    The function also returns `e_expect`, the expected error on the prediction `x_pred`,
    computed as the mean of the RMS-errors of all `K`-neighbours-errors.

    Keywords:
    * `metric = Euclidean()`: Metric used for distance computation
    * `theiler::Int = 1`: Theiler window for excluding serially correlated points from
       the nearest neighbour search.
    * `Tw::Int = 1`: The prediction time in sampling units. If `Tw > 1`, a multi-step
      prediction is performed.
"""
function local_linear_prediction(Y::Dataset{D,T}, K::Int = 5;
    metric = Euclidean(), theiler::Int = 1, Tw::Int = 1) where {D,T}

    NN = length(Y)
    ns = 1:NN
    vs = Y[ns]
    vtree = KDTree(Y[1:length(Y)-Tw,:], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, theiler)

    ϵ_ball = zeros(T, K, D) # preallocation
    A = ones(K,D) # datamatrix for later linear equation to solve for AR-process
    # consider NNs of the very last point of the trajectory
    NNidxs = allNNidxs[end] # indices of k nearest neighbors to v

    # determine neighborhood `Tw` time steps ahead
    @inbounds for (i, j) in enumerate(NNidxs)
        ϵ_ball[i, :] .= Y[j + Tw]
        A[i,:] = Y[j]
    end

    # make local linear model of the last point of the trajectory
    prediction = zeros(D)
    b  = zeros(D)
    ar_coeffs = zeros(D, D)
    namess = ["X"*string(i) for i = 1:D]
    ee = Meta.parse.(namess)
    formula_expression = Term(:Y) ~ sum(term.(ee))

    for i = 1:D
        data = DataFrame()
        for (cnt,var) in enumerate(namess)
            data[!, var] = A[:,cnt]
        end
        data.Y = ϵ_ball[:,i]

        ols = lm(formula_expression, data)
        b[i] = coef(ols)[1]
        for j = 1:D
            ar_coeffs[i,j] = coef(ols)[j+1]
        end
        prediction[i] = Y[NN,:]'*ar_coeffs[i,:] + b[i]
    end

    # predicted prediction error
    error_predict = sum((ϵ_ball .- prediction').^2; dims=1)

    return vec(prediction), vec(error_predict)
end

"""
    iterated_local_linear_prediction(Y::Dataset, K::Int = 5, Tw::Int = 2; kwargs...) → Y_predict

    Perform an iterated one step forecast over `Tw` time steps using the local linear
    prediction algorithm. `Y_predict` is a Vector of dimension like `Y`.

    Keywords:
    * `metric = Euclidean()`: Metric used for distance computation
    * `theiler::Int = 1`: Theiler window for excluding serially correlated points from
       the nearest neighbour search.
    * `verbose::Bool = false`: When set to `true`, the function prints the actual time
      step, which it is computing.
"""
function iterated_local_linear_prediction(Y::Dataset{D,T}, K::Int = 5, Tw::Int = 2;
    metric = Euclidean(), theiler::Int = 1, verbose::Bool = false) where {D,T}

    @assert Tw > 0 "Time horizon must be a positive integer"
    N = length(Y)
    predicted_trajectory = deepcopy(Y)
    for Th = 1:Tw
        if verbose
            println("Compute prediction for time step $Th")
        end
        # iterated one step
        predicted, _ = MCDTS.local_linear_prediction(predicted_trajectory, K;
                                            theiler, metric)
        push!(predicted_trajectory, predicted)
    end
    return predicted_trajectory[N+1:end]
end



"""
    iterated_local_linear_prediction_embed(Y::Dataset, τs::Vector, K::Int = 5, Tw::Int = 2; kwargs...) → Y_predict
    Perform an iterated one step forecast over `Tw` time steps using the local linear
    prediction algorithm. `Y_predict` is a Dataset of length `Tw` and dimension like
    `Y`. In contrast to `iterated_local_linear_prediction()` we here use the time
    delays `τs` to reconstruct all components of a predicted trajectory point from
    the 1st component, which is obtained from the local model. This only works for
    univariate embeddings.

    Keywords:
    * `metric = Euclidean()`: Metric used for distance computation
    * `theiler::Int = 1`: Theiler window for excluding serially correlated points from
       the nearest neighbour search.
    * `verbose::Bool = false`: When set to `true`, the function prints the actual time
      step, which it is computing.
"""
function iterated_local_linear_prediction_embed(Y::Dataset{D,T}, τs::Vector, K::Int = 5, Tw::Int = 2;
    metric = Euclidean(), theiler::Int = 1, verbose::Bool = false) where {D,T}

    @assert Tw > 1 "Time horizon must be a positive integer"
    N = length(Y)
    d = size(Y,2)
    @assert length(τs) == d "Vector storing the delays must have the same dimensionality as the Input trajectory."
    @assert τs[1] == 0 "Vector storing the delays must have 0 as its first entry."
    @assert sum(τs.<0) == 0 "Vector storing the delays must have 0 as its first entry."
    predicted_trajectory = deepcopy(Y)
    for Th = 1:Tw
        if verbose
            println("Compute prediction for time step $Th")
        end
        # iterated one step
        predicted, _ = MCDTS.local_linear_prediction(predicted_trajectory, K;
                                            theiler, metric)
        for i = 2:d
            predicted[i] = predicted_trajectory[end-τs[i],1]
        end
        push!(predicted_trajectory, predicted)
    end
    return Dataset(predicted_trajectory[N+1:end])
end

"""
    get_ar_prediction(x::Vector, coeffs::Vector; kwargs...) → Y_predict

    Computes a prediction `Y_predict` of the AR-model determined by the coefficients
    in `coeffs`. The order of the AR-model equals the length of `coeffs`. `x` can be
    a long vector (the time series), but needs to contain at least `length(coeffs)`
    values, in order to initialize the model. If the time horizon `Tw` is larger than
    1, an iterated one-step prediction is peformed.

    Keywords:
    * `Tw::Int = 1`: Time horizon for the prediction
    * `c::Real = 0`: Offset-parameter for AR-model
    * `rng::AbstractRNG = Random.GLOBAL_RNG`: Random number generator
"""
function get_ar_prediction(x::Vector{T}, coeffs::Vector; Tw::Int = 1, σ::Real = 1,
    c::Real = 0, rng::AbstractRNG = Random.GLOBAL_RNG) where {T}

    N = length(x)
    @assert N == length(coeffs) "Priors must be as many as the order of the chosen AR-process."
    @assert Tw > 0  "Provide a valid forecast horizon in sampling units (positiv integer)."
    forecast = zeros(N+Tw)
    forecast[1:N] = x
    idx = N + 1
    for i = 1:Tw
        forecast[idx] = c + forecast[idx-N:idx-1]'*coeffs + σ*randn(rng)
        idx += 1
    end
    return forecast[N+1:end]
end