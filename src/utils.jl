# Some handy functions

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
                Base.push!(maximas, s[first_point])
                Base.push!(maximas_idx, first_point)
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
        maximas_s, maximas_idx_s = findmax(s)
        Base.push!(maximas, maximas_s)
        Base.push!(maximas_idx, maximas_idx_s)
    end
    return maximas, maximas_idx
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
