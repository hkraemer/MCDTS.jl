"""
    Wrapper for the same function from DelayEmbeddings.jl, here with some added
    functionality w.r.t. to predictive modeling.
"""
function pecora(
        s, τs::NTuple{D, Int}, js::NTuple{D, Int} = Tuple(ones(Int, D));
        delays = 0:50 , J=maxdimspan(s), samplesize::Real = 1, K::Int = 13, w::Int = 1,
        metric = Chebyshev(), α::T = 0.05, p::T = 0.5, PRED::Bool = false) where {D, T<:Real}

    @assert K ≥ 8 "You must provide a δ-neighborhood size consisting of at least 8 neighbors."
    @assert all(x -> x ≥ 0, js) "τ's and j's for generalized embedding must be positive integers"
    @assert 0 < samplesize ≤ 1 "`samplesize` must be ∈ (0,1]"

    N = floor(Int,samplesize*length(s)) #number of fiducial points
    if PRED
        # causal embedding
        vspace = genembed(s, τs.*(-1), js) # takes positive τs's and converts internally
        vtree = KDTree(vspace.data[1+maximum(abs.(delays)):end], metric)
        # indices of random fiducial points (with valid time range w.r.t. T)
        ns = vec((1+maximum(abs.(delays))):length(vspace))
    else
        vspace = genembed(s, τs, js)
        vtree = KDTree(vspace.data[1:end-maximum(abs.(delays))], metric)
        # indices of random fiducial points (with valid time range w.r.t. T)
        ns = vec(1:(length(vspace)-maximum(abs.(delays))))
    end

    vs = vspace[ns]
    allNNidxs, allNNdist = all_neighbors(vtree, vs, ns, K, w)

    all_ε★ = zeros(length(delays), length(J))
    allts = columns(s)

    # Loop over potential timeseries to use in new embedding
    for i in 1:length(J)
        x = allts[J[i]]
        x = (x .- mean(x))./std(x) # so that different timeseries can be compared
        all_ε★[:, i] .= MCDTS.continuity_per_timeseries(x, ns, allNNidxs, delays, K, α, p; PRED = PRED)
    end
    return all_ε★
end

maxdimspan(s) = 1:size(s)[2]
maxdimspan(s::AbstractVector) = 1
columns(s::AbstractVector) = (s, )

function continuity_per_timeseries(x::AbstractVector, ns, allNNidxs, delays, K, α, p; PRED::Bool=false)
    avrg_ε★ = zeros(size(delays))
    Ks = [k for k in 8:K]
    δ_to_ε_amount = get_binomial_table(p, α; trial_range = length(Ks))
    if PRED
        delays = delays.*(-1)
    end
    for (ι, τ) in enumerate(delays) # Loop over the different delays
        c = 0
        for (i, n) in enumerate(ns) # Loop over fiducial points
            NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
            if PRED
                NNidxs = maximum(abs.(delays)) .+ NNidxs
            end
            # Check if any of the indices of the neighbors falls out of temporal range
            any(j -> (j+τ < 1), NNidxs) && continue
            # If not, calculate minimum ε
            avrg_ε★[ι] += ε★(x, n, τ, NNidxs, δ_to_ε_amount, Ks)
            c += 1
        end
        c == 0 && error("Encountered astronomically small chance of all neighbors having "*
                        "invalid temporal range... Run the function again or decrease `w`.")
        avrg_ε★[ι] /= c
    end
    return avrg_ε★
end


function ε★(x, n, τ, NNidxs, δ_to_ε_amount::Dict, Ks::AbstractVector)
    a = x[n+τ] # fiducial point in ε-space
    @inbounds dis = [abs(a - x[i+τ]) for i in NNidxs]
    ε = zeros(length(Ks))
    for (i, k) in enumerate(Ks)
        sortedds = sort!(dis[1:k]; alg = QuickSort)
        l = δ_to_ε_amount[k]
        ε[i] = sortedds[l]
    end
    return minimum(ε)
end


"""
    get_binomial_table(p, α; trial_range::Int=8) -> `δ_to_ε_amount`, Dict(δ_points => ϵ_points)
compute the numbers of points from the δ-neighborhood, which need to fall outside
the ϵ-neighborhood, in order to reject the Null Hypothesis at a significance
level `α`. One parameter of the binomial distribution is `p`, the other one would
be the number of trials, i.e. the considered number of points of the δ-neighborhood.
`trial_range` determines the number of considered δ-neighborhood-points, always
starting from 8. For instance, if `trial_range=8` (Default), then δ-neighborhood
sizes from 8 up to 15 are considered.
Return `δ_to_ε_amount`, a dictionary with `δ_points` as keys and the corresponding number of
points in order to reject the Null, `ϵ_points`, constitute the values.
"""
function get_binomial_table(p::T, α::T; trial_range::Int=8) where {T<:Real}
    @assert trial_range ≥ 1 "`trial_range` must be an integer ≥ 1"
    δ_to_ε_amount = Dict{Int, Int}()
    @inbounds for key = 8:(7+trial_range)
        δ_to_ε_amount[key] = quantile(Distributions.Binomial(key,p), 1-α)
    end
    return δ_to_ε_amount
end


function all_neighbors(vtree, vs, ns, K, w)
    w ≥ length(vtree.data)-1 && error("Theiler window larger than the entire data span!")
    k = maximum(K)
    tw = Theiler(w, ns)
    idxs, dists = bulksearch(vtree, vs, NeighborNumber(k), tw)
end

"""
    all_neighbors(A::Dataset, stype, w = 0) → idxs, dists
Find the neighbors of all points in `A` using search type `stype` (either
[`NeighborNumber`](@ref) or [`WithinRange`](@ref)) and `w` the [Theiler window](@ref).

This function is nothing more than a convinience call to `Neighborhood.bulksearch`.
"""
function all_neighbors(A::AbstractDataset, stype, w::Int = 0)
    theiler = Theiler(w)
    tree = KDTree(A)
    idxs, dists = bulksearch(tree, A, stype, theiler)
end

"""
    columns(dataset) -> x, y, z, ...
Return the individual columns of the dataset.
"""
function columns end
@generated function columns(data::AbstractDataset{D, T}) where {D, T}
    gens = [:(data[:, $k]) for k=1:D]
    quote tuple($(gens...)) end
end
