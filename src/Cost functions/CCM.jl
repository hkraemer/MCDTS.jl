# All functionality corresponding to the correlation coefficient of the 
# convergent cross mapping (CCM) as Loss function Γ.


## Constructors:

"""
    CCM_ρ <: AbstractLoss

    Constructor for the CCM_ρ loss function (correlation coefficient of the
    convergent cross mapping) based on Sugihara et al.[^Sugihara2012], see also
    [`ccm`](@ref). In this case MCDTS tries to maximize the correlation coefficient
    of the convergent cross mapping from the input `data` and `Y_CCM`, the time
    series CCM should cross map to from `data` (see [`mcdts_embedding`](@ref)).

    ## Fieldnames
    * `timeseries`: The time series CCM should cross map to.
    * `threshold:: = 1: A threshold for the sufficient correlation of the
      cross-mapped values and the true values from `Y_CMM` for the current embedding.
      When the correlation coefficient exeeds this threshold in an embedding cycle
      the embedding stops.
    * `samplesize::Real = 1.`: the fraction of all phase space points
     to be considered in the computation of CCM_ρ.

    ## Defaults
    * When calling `CCM_ρ(timeseries)`, a CCM_ρ-object is created, storing
      `timeseries`, which is considered to be causally depended and the
      `threshold=1` is used, i.e. no threshold, since the correlation coefficient
      can not exceed 1.

    [^Sugihara2012]: Sugihara et al., [Detecting Causality in Complex Ecosystems. Science 338, 6106, 496-500](https://doi.org/10.1126/science.1227079)
"""
struct CCM_ρ <: AbstractLoss
    timeseries::Vector
    threshold::AbstractFloat
    samplesize::Real

    # Constraints and Defaults
    CCM_ρ(x,y=1.0,z=1) = begin
        @assert 0 < y ≤ 1 "Threshold must be a ∈ (0 1]"
        @assert 0 < z ≤ 1. "The samplesize must be in the interval (0 1]."
        typeof(y) <: Int ? new(x,convert(AbstractFloat, -y),z) : new(x,-y,z)
    end
end


## Functions:

"""
    Return the loss based on the negative correlation coefficient for CCM.
"""
function compute_loss(Γ::CCM_ρ, Λ::AbstractDelayPreselection, dps::Vector{P}, Y_act::Dataset{D, T}, Ys, τs, w::Int, ts::Int, τ_vals, ts_vals; metric=Euclidean(), kwargs...) where {P, D, T}

    Y_other = DelayEmbeddings.standardize(Γ.timeseries)
    samplesize = Γ.samplesize

    max_idx = get_max_idx(Λ, dps, τ_vals, ts_vals, ts) # get the candidate delays
    isempty(max_idx) && return Float64[], Int64[], []

    ρ_CCM = zeros(Float64, length(max_idx))

    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        tau_trials = (τ_vals...,τs[τ_idx-1],)
        ts_trials = (ts_vals...,ts,)
        Y_trial = genembed(Ys, tau_trials.*(-1), ts_trials)
        # account for value-shift due to negative lags
        Ys_other = Y_other[1+maximum(tau_trials.*(-1)):length(Y_trial)+maximum(tau_trials.*(-1))]
        # compute ρ_CCM for Y_trial and Y_other
        ρ_CCM[i], _, _ = MCDTS.ccm(Y_trial, Ys_other; metric = metric, w = w, samplesize = samplesize)
    end
    return -ρ_CCM, max_idx, [nothing for i in max_idx]
end


"""
    ccm(X, y; kwargs...) → ρ, y_hat, y_idx

    Compute the convergent crossmapping (CCM) [^Sugihara2012](Sugihara et al. 2012) of a
    vector time series `X` (an embedded time series `x`) on the time series `y`
    NOTE: 'X' and 'y' must have the same length and you have to make sure that
    'y' starts at the same time index as 'X' does. - When using [`genembed`](@ref)
    with negative delays to construct `X` from `x`, which is mandatory here, then
    'y' needs to be shifted by the largest negative delay value, which has been
    used to construct `X`.

    Returns the correlation coefficient of `y` and its predicted values for `y_hat`,
    based on the nearest neighbour structure of `X`. `y_idx` are the corresponding
    indices, which have been used for computing `y_hat`.
    It is said that 'y' causes 'x', if ρ increases with increasing time series
    length AND ρ is "quite high".

    Keyword arguments:
    *`metric = Euclidean()`: The metric for vector distance computation.
    *`w::Int = 1`: The Theiler window in sampling units.
    *`lags::Array = [0]`: The lag for the cross mapping, in order to detect time lagged
                          causal relationships. The output ρ is an array of size
                          `length(lags)`, the output Y_hat is the one corresponding
                          to a lag of zero.
    * `samplesize::Real = 0.1`: fraction of all phase space points (=`length(X)`)
                                to be considered (fiducial points v)

"""
function ccm(X::Dataset{D,T},Y::Vector{T}; metric = Euclidean(), w::Int = 1,
    lags::AbstractArray = [0], samplesize::Real = 1.) where {D,T<:Real}

    K = D+1
    @assert length(X)==length(Y)
    @assert 0 < samplesize ≤ 1. "`samplesize` for computing ρ-CCM must be ∈ (0,1]"
    XX = Matrix(X)
    N = length(X)

    # consider subset of state space points
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
    ρ = Statistics.cor(Y_hat, Y[ns])
    return ρ, Dataset(Y_hat), ns
end