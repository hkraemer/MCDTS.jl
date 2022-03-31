# All functionality corresponding to the L-statistic as Loss function Γ.

## Constructors:

"""
    L_statistic <: AbstractLoss

    Constructor for the L-statistic loss function based on Uzal et al.[^Uzal2011]. Here
    we consider the decrease of the L-statistic `ΔL` in between embedding cycles,
    according to Kraemer et al.[^Kraemer2021] ([`pecuzal_embedding`](@ref)).

    ## Fieldnames
    * `threshold::Float`: A threshold for the tolerable `ΔL` decrease for the current
      embedding. When `ΔL` exceeds this threshold in an embedding cycle the embedding
      stops. Note that `ΔL` is a negative value therefore `threshold` must be a small
      negative number.
    * `KNN::Int`: the amount of nearest neighbors considered, in order to compute the
      L-statistic, in particular `σ_k^2` (read algorithm description [`uzal_cost`]@ref).
    * `tws::AbstractRange{Int}`: Customization of the sampling of the different time horizons
      (T's), when computing Uzal's L-statistics. Here any kind of integer ranges (starting at 2)
      are allowed.
    * `samplesize::Real = 1.`: the fraction of all phase space points
      to be considered in the computation of the L-statistic(s).

    ## Defaults
    * When calling `L_statistic()`, a L_statistic-object is created, which uses no
      threshold and consideres 3 nearest neighbors for time horizons `tws=2:100`.
    * When calling `L_statistic(threshold)`, a L_statistic-object is created, which uses
      the given `threshold` and consideres 3 nearest neighbors for time horizons `tws=2:100`.
    * When calling `L_statistic(threshold,KNN)`, a L_statistic-object is created, which uses
      the given `threshold`, consideres `KNN` nearest neighbors for time horizons `tws=2:100`.

    [^Kraemer2021]: Kraemer, K.H., Datseris, G., Kurths, J., Kiss, I.Z., Ocampo-Espindola, Marwan, N. (2021). [A unified and automated approach to attractor reconstruction. New Journal of Physics 23(3), 033017](https://iopscience.iop.org/article/10.1088/1367-2630/abe336).
    [^Uzal2011]: Uzal, L. C., Grinblat, G. L., Verdes, P. F. (2011). [Optimal reconstruction of dynamical systems: A noise amplification approach. Physical Review E 84, 016223](https://doi.org/10.1103/PhysRevE.84.016223).
"""
struct L_statistic <: AbstractLoss
    threshold::AbstractFloat
    KNN::Int
    tws::AbstractRange{Int}
    samplesize::Real
    # Constraints and Defaults
    L_statistic(x=0,y=3,z=2:100,s=1.) = begin
        @assert x <= 0 "Please provide a (small) negative number for the threshold of ΔL."
        @assert y > 0 "Number of considered nearest neighbors must be positive."
        @assert z[1] == 2 "The considered range for the time horizon of the L-function must start at 2."
        @assert 0 < s ≤ 1. "The samplesize must be in the interval (0 1]."
        typeof(x) <: Int ? new(convert(AbstractFloat, x), y, z, s) : new(x,y,z,s)
    end
end


## Functions:

function get_embedding_params_according_to_loss(Γ::L_statistic, embedding_pars::Vector{EmbeddingPars}, L_old)
    threshold = Γ.threshold
    L_pot = L.(embedding_pars)
    if minimum(L_pot) > threshold
        return EmbeddingPars[], true
    else
        ind = L_pot .≤ threshold
        return embedding_pars[ind], false
    end
end

"""
    Return the loss based on the maximum decrease of the L-statistic `L_decrease` and corresponding
    delay-indices `max_idx` for all local maxima in ε★
"""
function compute_loss(Γ::L_statistic, Λ::AbstractDelayPreselection, dps::Vector{P}, Y_act::Dataset{D, T}, Ys, τs, w::Int, ts::Int, τ_vals, ts_vals; metric=Euclidean(), kwargs...) where {P, D, T}

    KNN = Γ.KNN
    tws = Γ.tws
    samplesize = Γ.samplesize
    s = Ys[:,ts]

    max_idx = get_max_idx(Λ, dps, τ_vals, ts_vals, ts) # get the candidate delays
    isempty(max_idx) && return Float64[], Int64[], []

    L_decrease = zeros(Float64, length(max_idx))
    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        Y_trial = DelayEmbeddings.hcat_lagged_values(Y_act, s, τs[τ_idx-1])
        # compute L-statistic for Y_act and Y_trial and get the maximum decrease
        L_decrease[i] = MCDTS.uzal_cost_pecuzal_mcdts(Y_act, Y_trial, τs[end]; K = KNN,
                                w = w, metric = metric, tws = tws, samplesize = samplesize)
    end
    return L_decrease, max_idx, [nothing for i in max_idx]
end

"""
    uzal_cost_pecuzal_mcdts(Y1::Dataset, Y2::Dataset, Tw; kwargs...) → L_decrease

    This function is based on the functionality of [`uzal_cost`](@ref), here
    specifically tailored for the needs in the MCDTS (PECUZAL) algorithm.
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
    * `samplesize::Real = 1.0`: determine the fraction of all phase space points (=`length(Y)`)
      to be considered (fiducial points v)
"""
function uzal_cost_pecuzal_mcdts(Y::Dataset{D, ET}, Y_trial::Dataset{DT, ET}, Tw::Int;
        K::Int = 3, w::Int = 1, econ::Bool = false, tws::AbstractRange{Int} = 2:Tw,
        metric = Euclidean(), samplesize::Real = 1.) where {D, DT, ET}

    @assert DT == D+1
    @assert Tw ≥ 0
    @assert tws[1]==2
    @assert 0 < samplesize ≤ 1.

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

    NN = length(Y_trial)-tws[end]
    if NN < 1
        error("Time series too short for given possible delays and Theiler window to find enough nearest neighbours")
    end
    if samplesize==1
        ns = 1:NN
        Nfp = length(ns)
    else
        Nfp = Int(floor(samplesize*NN)) # number of considered fiducial points
        ns = sample(vec(1:NN), Nfp, replace = false)  # indices of fiducial points
    end

    vs = Y[ns] # the fiducial points in the data set
    vs_trial = Y_trial[ns] # the fiducial points in the data set

    vtree = KDTree(Y[1:NN], metric)
    allNNidxs, allNNdist = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, w)
    vtree_trial = KDTree(Y_trial[1:NN], metric)
    allNNidxs_trial, allNNdist_trial = DelayEmbeddings.all_neighbors(vtree_trial, vs_trial, ns, K, w)

    # loop over each time horizon
    cnt = 1
    for T in tws
        # compute conditional variances and neighborhood-sizes
        DelayEmbeddings.compute_conditional_variances!(ns, vs, vs_trial, allNNidxs,
            allNNidxs_trial, Y, Y_trial, ϵ_ball, ϵ_ball_trial, u_k, u_k_trial,
            T, K, metric, ϵ², ϵ²_trial, E², E²_trial, cnt)

        # compute distance of L-values and check whether that distance can be
        # increased
        dist = DelayEmbeddings.compute_L_decrease(E², E²_trial, ϵ², ϵ²_trial, cnt, Nfp)
        if isnan(dist)
            error("Computed 0-distances, due to duplicate datapoints in your data. Try to add minimal additive noise to the signal you wish to embed and try again.")
        end
        if dist > dist_former && dist_former<0
            break
        else
            dist_former = dist
        end
        cnt += 1
    end
    return dist_former
end
