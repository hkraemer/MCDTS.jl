using DynamicalSystemsBase
using DelayEmbeddings
import Base.show

abstract type AbstractTreeElement end

"""
    mutable struct Root

The 'start'/root of Tree. Each node contains its children. The root contains the starting branches/nodes.

# Initialization:

`r = Root()`

# Fields:

* `children::Union{Array{Node,1},Nothing}`: The first nodes of the tree.
* `Lmin`; Is the global minimum of the cumulative ΔL statistic found so far.
"""
mutable struct Root <: AbstractTreeElement
    children
    Lmin
end
Root()=Root(nothing,0)
get_τs(n::Root) = Int[]
get_ts(n::Root) = Int[]
function Base.show(io::IO,n::Root)

    if n.children == nothing
        return print(io,string("Embedding tree, no tree search yet performed"))
    else
        best_node = best_embedding(n)

        return print(io,string("Embedding tree with current best embedding: L=",best_node.L," - full embd. τ=",best_node.τs," ,i_ts=",best_node.ts))
    end
end

"""
    mutable struct Node{T}

A node of the tree. Each node contains its children.

# Fields:

* `τ::Int`: The delay value of this node
* `L::T`: The value of the cumulative ΔL statistic at this node
* `τs::Array{Int,1}`: The complete vector with all τs chosen along this path up until this node
* `ts::Array{Int,1}`: The complex vector which of the possibly multivariate time series is used at each embedding step i
* `children::Union{Array{Node,1},Nothing}`: The children of this node
"""
mutable struct Node{T} <: AbstractTreeElement
    τ::Int
    t::Int
    L::T
    τs::Array{Int,1}
    ts::Array{Int,1}
    children::Union{Array{Node,1},Nothing}
end

N_children(n::AbstractTreeElement) = n.children == nothing ? 0 : length(n.children)
get_τs(n::Node) = n.τs
get_ts(n::Node) = n.ts
get_children_Ls(n::AbstractTreeElement) = [n.children[i].L for i in 1:N_children(n)]
get_children_τs(n::AbstractTreeElement) = [n.children[i].τ for i in 1:N_children(n)]
get_children_ts(n::AbstractTreeElement) = [n.children[i].t for i in 1:N_children(n)]
Base.show(io::IO,n::Node) = print(io,string("Node with τ=",n.τ,", i_t=",n.t," ,L=",n.L," - full embd. τ=",n.τs," ,i_ts=",n.ts))

"""
    choose_children(n::AbstractTreeElement, τ::Int, t:Int)

Pick one of the children of `n` with values `τ` and `t`. If there is none,
return `nothing`.
"""
function choose_children(n::AbstractTreeElement, τ::Int, t::Int)
    τs=get_children_τs(n)
    ts=get_children_ts(n)

    res = intersect(findall(τs .== τ), findall(ts .== t))
    if length(res)==0
        return nothing
    elseif length(res)==1
        return n.children[res[1]]
    else
        error("There's something wrong, there shouldn't be multiple children with the same values.")
    end
end


"""
    next_embedding(n::Node, Ys::Dataset{D, T}, w::Int, τs; KNN:Int = 3,
                        FNN::Bool = false, threshold::Real = 0,
                        tws::AbstractRange = 2:τs[end]) → τ_pot, ts_pot, L_pot, flag

Performs the next embedding step. For the actual embedding contained in `n`
compute as many conitnuity statistics as there are time series in the Dataset
`Ys` for a range of possible delays `τs`. Return the values for the best delay
`τ_pot`, its corresponding time series index `ts_pot` the according L-value
`L_pot` and `flag`, following the Pecuzal-logic.

# Keyword arguments
* `KNN = 3`: The number of nearest neighbors considered in the computation of
  the L-statistic.
* `FNN:Bool = false`: Determines whether the algorithm should minimize the
  L-statistic or the FNN-statistic.
* `threshold::Real = 0`: The algorithm does not pick a peak from the continuity
  statistic, when its corresponding `ΔL`/FNN-value exceeds this threshold. Please
  provide a positive number for both, `L` and `FNN`-statistic option (since the
  `ΔL`-values are negative numbers for meaningful embedding cycles, this threshold
  gets internally sign-switched).
* `tws::Range = 2:τs[end]`: Customization of the sampling of the different T's,
  when computing Uzal's L-statistics. Here any kind of integer ranges (starting
  at 2) are allowed, up to `τs[end]`.

# Returns

* `τ_pot`: Next delay
* `ts_pot`: Index of the time series used (in case of multivariate time series)
* `L_pot`: L statistic of next embedding step with delay `τ_pot` from `ts_pot`.
* `flag`: Did the embedding converge? i.e. L can not be further minimized anymore

"""
function next_embedding(n::Node, Ys::Dataset{D, T}, w::Int, τs; KNN::Int = 3,
                            FNN::Bool = false, tws::AbstractRange{Int} = 2:τs[end],
                            threshold::Real = 0) where {D, T<:Real}
    τs_old = get_τs(n)
    ts_old = get_ts(n)
    L_old = n.L
    # do the next embedding step
    τ_pot, ts_pot, L_pot, flag = give_potential_delays(Ys, τs, w, Tuple(τs_old),
                            Tuple(ts_old), L_old; KNN = KNN, FNN = FNN, tws = tws,
                            threshold = threshold)
    return τ_pot, ts_pot, L_pot, flag
end

"""
    next_embedding(n::Node, Ys::Dataset, w::Int, τs = 0:100, KNN = 3)

The first embedding step
"""
function next_embedding(n::Root, Ys::Dataset{D, T}, w::Int, τs; KNN::Int = 3,
                            FNN::Bool = false, tws::AbstractRange{Int} = 2:τs[end],
                            threshold::Real = 0) where {D, T<:Real}
    τ_pot = zeros(Int, size(Ys,2))
    ts_pot = Array(1:size(Ys,2))
    if FNN
        L_pot = ones(size(Ys,2))
    else
        L_pot = zeros(size(Ys,2))
    end

    return τ_pot, ts_pot, L_pot, false
end

"""
    choose_next_node(n::Union{Node,Root}, func, Lmin_global)

Returns one of the children of based on the function `func(Ls)->i_node`,
Lmin_global is the best L value so far in the optimization process, if any of
the input Ls to choose from is smaller than it, it is always chosen.
"""
function choose_next_node(n::Node,func, Lmin_global=-Inf)
    N = N_children(n)
    if N == 0
        return nothing
    elseif N == 1
        return n.children[1]
    else
        Ls = get_children_Ls(n)
        # check if any is smaller than the global min
        Lmins = findall(Ls .< Lmin_global)

        if isempty(Lmins)
            return n.children[func(get_children_Ls(n))]
        else
            return n.children[argmin(Ls)]
        end
    end
end

"""
    choose_next_node(n::Union{Node,Root}, func, Lmin_global)

Returns one of the children of based on the function `func(Ls)->i_node`
"""
function choose_next_node(n::Root,func,Lmin_global=-Inf)
    N = N_children(n)
    if N == 0
        return nothing
    elseif N == 1
        return n.children[1]
    else
        return n.children[rand(1:N)]
    end
end

minL(Ls) = argmin(Ls)

"""
    softmaxL(Ls; β=1.5)

Return an index with prob computed by a softmax of all Ls.
"""
function softmaxL(Ls; β=1.5)
    softmaxnorm = sum(exp.(-β*Ls))

    p_L = exp.(-β.*Ls) ./ softmaxnorm
    p_L = cumsum(p_L)
    rand_number = rand()

    for i=1:length(p_L)
        if p_L[i] > rand_number
            return i
        end
    end
    return length(p_L)
end

"""
    expand!(n::Union{Node,Root}, data::Dataset, w::Int, choose_func, delays;
                            max_depth=20, KNN=3, FNN=false, L_threshold=0,
                            tws=2:delays[end])

This is one single rollout and backprop of the tree.

* `n`: Starting node
* `data`: data
* `w`: Theiler Window
* `choose_func`: Function to choose next node with
* `delays = 0:100`: The possible time lags

# Keyword arguments
* `max_depth = 20`: Threshold, which determines the algorithm. It either breaks,
  when it converges, i.e. when there is no way to reduce the cost-function any
  further, or when this threshold is reached.
* `KNN = 3`: The number of nearest neighbors considered in the computation of
  the L-statistic.
* `FNN:Bool = false`: Determines whether the algorithm should minimize the
  L-statistic or the FNN-statistic.
* `threshold::Real = 0`: The algorithm does not pick a peak from the continuity
  statistic, when its corresponding `ΔL`/FNN-value exceeds this threshold. Please
  provide a positive number for both, `L` and `FNN`-statistic option (since the
  `ΔL`-values are negative numbers for meaningful embedding cycles, this threshold
  gets internally sign-switched).
* `tws::Range = 2:delays[end]`: Customization of the sampling of the different T's,
  when computing Uzal's L-statistics. Here any kind of integer ranges (starting
  at 2) are allowed, up to `delays[end]`.

"""
function expand!(n::Root, data::Dataset{D, T}, w::Int, choose_func,
            delays::AbstractRange{DT} = 0:100; max_depth::Int=20, KNN::Int=3,
            verbose=false, FNN::Bool = false, tws::AbstractRange{DT} = 2:delays[end],
            threshold::Real = 0) where {D, DT, T<:Real}

    @assert threshold ≥ 0
    @assert tws[1] == 2
    @assert w > 0
    current_node = n

    for i=1:max_depth # loops until converged or max_depth is reached
        # next embedding step

        # only if it was not already computed
        if current_node.children == nothing
            τs, ts, Ls, converged = next_embedding(current_node, data, w, delays;
                                                KNN = KNN, FNN = FNN, tws = tws,
                                                threshold = threshold)
            if converged
                break
            else
                # spawn children
                children = []
                for j = 1:length(τs)
                    if FNN
                        push!(children, Node(τs[j],ts[j],Ls[j],[get_τs(current_node); τs[j]], [get_ts(current_node); ts[j]], nothing))
                    else
                        if typeof(current_node) == MCDTS.Root
                            push!(children, Node(τs[j],ts[j],(current_node.Lmin+Ls[j]),[get_τs(current_node); τs[j]], [get_ts(current_node); ts[j]], nothing))
                        else
                            push!(children, Node(τs[j],ts[j],(current_node.L+Ls[j]),[get_τs(current_node); τs[j]], [get_ts(current_node); ts[j]], nothing))
                        end
                    end
                end
                current_node.children = children
            end
        end

        # choose next node
        current_node = choose_next_node(current_node, choose_func, n.Lmin)
        if verbose
            println(current_node)
        end
    end
    # now backprop the values (actually we go to top to bottom, but we know were to end because we got the correct τs and ts)
    backprop!(n, current_node.τs, current_node.ts, current_node.L)
end

"""
    Backpropagation of the tree spanned by all children in `n` (for this run).
All children-nodes L-values get set to the final value achieved in this run.
"""
function backprop!(n::Root,τs,ts,L_min)

    if n.Lmin > L_min
        n.Lmin = L_min
    end

    current_node = n
    for i=1:length(τs)
        # the initial embedding step is left out of the backprop
        current_node = choose_children(current_node,τs[i],ts[i])
        if current_node.L > L_min
            current_node.L = L_min
        end
    end
end


"""
    mc_delay(N::Int=100)

Do the monte carlo run with `N` trials, returns the tree.
"""
function mc_delay(data, w::Int, choose_func, delays::AbstractRange{D}, N::Int=40;
            max_depth::Int=20, KNN::Int = 3, FNN::Bool = false, verbose::Bool=false,
            tws::AbstractRange{D} = 2:delays[end], threshold::Real = 0) where {D}

    # initialize tree
    tree = Root()

    for i=1:N

        expand!(tree, data, w, choose_func, delays; KNN = KNN, FNN = FNN,
                    max_depth = max_depth, tws = tws, threshold = threshold)

        if verbose
            if (i%1)==0
                println(i,"/",N)
                println(best_embedding(tree))
                println("---")
            end
        end
    end

    return tree
end

"""
    best_embedding(r::Root)

Given the root `r` of a tree, return the best embedding in the form of the final node at the end of the best embedding
"""
function best_embedding(r::Root)

    not_finished = true
    current_node = r
    while not_finished

        if current_node.children == nothing
            not_finished = false
            return current_node
        else
            Ls = get_children_Ls(current_node)

            # traverse down the tree always with minimal L
            current_node = current_node.children[argmin(Ls)]
        end
    end
    return current_node
end
