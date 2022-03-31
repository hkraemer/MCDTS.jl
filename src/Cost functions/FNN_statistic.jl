# All functionality corresponding to the FNN-statistic as Loss function Γ.

## Constructors:

"""
    FNN_statistic <: AbstractLoss

    Constructor for the FNN-statistic loss function (false nearest neighbor) based
    on Hegger & Kantz [^Hegger1999].

    ## Fieldnames
    * `threshold::Float`: A threshold for the tolerable cumulative FNN decrease
      for the current embedding. When the fraction of FNNs fall below this threshold
      in an embedding cycle the embedding stops.
    * `r::Float = 2`: The FNN-distance-expansion threshold (typically set to 2).
    * `samplesize::Real = 1.`: the fraction of all phase space points
      to be considered in the computation of the L-statistic(s).

    ## Defaults
    * When calling `FNN_statistic()`, a FNN_statistic-object is created, which uses no
      threshold and uses the FNN-inter threshold `r=2`.
    * When calling `FNN_statistic(threshold)`, a FNN_statistic-object is created, which uses
      the given `threshold` and uses the FNN-inter threshold `r=2`.

    [^Hegger1999]: Hegger & Kantz, [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
struct FNN_statistic <: AbstractLoss
    threshold::AbstractFloat
    r::AbstractFloat
    samplesize::Real
    # Constraints and Defaults
    FNN_statistic(x=0.,y=2.,z=1.) = begin
        @assert x >= 0 "Threshold for FNN-statistic must be ≥ 0"
        @assert y > 0 "FNN-distance-expansion threshold must be >0"
        @assert 0 < z ≤ 1. "The samplesize must be in the interval (0 1]."
        typeof(x) <: Int ? new(convert(AbstractFloat, x),y,z) : new(x,y,z)
        typeof(y) <: Int ? new(x,convert(AbstractFloat,y),z) : new(x,y,z)
    end
end



## Functions:

"""
    Return the loss based on the FNN-statistic `FNN` and indices `max_idx`  for all local maxima in dps
"""
function compute_loss(Γ::FNN_statistic, Λ::AbstractDelayPreselection, dps::Vector{P}, Y_act::Dataset{D, T}, Ys, τs, w::Int, ts::Int, τ_vals, ts_vals; metric=Euclidean(), kwargs...) where {P, D, T}

    r = Γ.r
    s = Ys[:,ts]
    samplesize = Γ.samplesize

    max_idx = get_max_idx(Λ, dps, τ_vals, ts_vals, ts) # get the candidate delays
    isempty(max_idx) && return Float64[], Int64[], []

    FNN_trials = zeros(Float64, length(max_idx))

    # compute nearest-neighbor-distances for actual trajectory
    Y_act2 = Y_act[1:end-τs[maximum(max_idx)-1],:]
    Y_act2 = DelayEmbeddings.standardize(Y_act2)
    vtree = KDTree(Y_act2, metric)

    NN = length(Y_act2)
    if samplesize==1
        ns = 1:NN
        Nfp = length(ns)
    else
        Nfp = Int(floor(samplesize*NN)) # number of considered fiducial points
        ns = sample(vec(1:NN), Nfp, replace = false)  # indices of fiducial points
    end

    _, NNdist_old = DelayEmbeddings.all_neighbors(vtree, Y_act2[ns], ns, 1, w)

    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        Y_trial = DelayEmbeddings.hcat_lagged_values(Y_act,s,τs[τ_idx-1])
        Y_trial = DelayEmbeddings.standardize(Y_trial)
        vtree = KDTree(Y_trial, metric)
        _, NNdist_new = DelayEmbeddings.all_neighbors(vtree, Y_trial[ns], ns, 1, w)
        # compute FNN-statistic
        FNN_trials[i] = fnn_embedding_cycle(NNdist_old, NNdist_new[1:length(NNdist_old)], r)
    end
    return FNN_trials, max_idx, [nothing for i in max_idx]
end


"""
    fnn_embedding_cycle(NNdist, NNdistnew, r=2) -> FNNs

    Compute the amount of false nearest neighbors `FNNs` [^Hegger1999], when adding another component
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