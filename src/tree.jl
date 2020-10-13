using Pkg
current_dir = pwd()
Pkg.activate(current_dir)

using DynamicalSystemsBase
import Base.show

abstract type AbstractTreeElement end

"""
    mutable struct Root

The 'start'/root of Tree. Each node contains its children. The root contains the starting branches/nodes.

# Initialization:

`r = Root()`

# Fields:

* `children::Union{Array{Node,1},Nothing}`: The first nodes of the tree.
"""
mutable struct Root <: AbstractTreeElement
    children
end
Root()=Root(nothing)
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
* `L::T`: The value of the L statistic at this node
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
    next_embedding(n::Node, Ys::Dataset{D, T}, w::Int, τs; KNN:Int = 3)

Performs the next embedding step. For the actual embedding contained in `n`
compute as many conitnuity statistics as there are time series in the Dataset
`Ys` for a range of possible delays `τs`. Return the values for the best delay
`τ_pot`, its corresponding time series index `ts_pot` the according L-value
`L_pot` and `flag`, following the Pecuzal-logic.

# Keyword arguments
* `KNN = 3`: The number of nearest neighbors considered in the computation of
  the L-statistic.

# Returns

* `τ_pot`: Next delay
* `ts_pot`: Index of the time series used (in case of multivariate time series)
* `L_pot`: L statistic of next embedding step with delay `τ_pot` from `ts_pot`.
* `flag`: Did the embedding converge? i.e. L can not be further minimized anymore

"""
function next_embedding(n::Node, Ys::Dataset{D, T}, w::Int, τs; KNN::Int = 3) where {D, T<:Real}
    τs_old = get_τs(n)
    L_old = n.L
    ts_old = get_ts(n)
    # do the next embedding step
    τ_pot, ts_pot, L_pot, flag = give_potential_delays(Ys, τs, w, Tuple(τs_old),
                                                Tuple(ts_old), L_old; KNN = KNN)
    return τ_pot, ts_pot, L_pot, flag
end

"""
    next_embedding(n::Node, Ys::Dataset, w::Int, τs = 0:100, KNN = 3)

The first embedding step
"""
function next_embedding(n::Root, Ys::Dataset{D, T}, w::Int, τs; KNN::Int = 3) where {D, T<:Real}
    τ_pot = zeros(Int, size(Ys,2))
    ts_pot = Array(1:size(Ys,2))
    L_pot = zeros(size(Ys,2))
    flag = false
    for i = 1:size(Ys,2)
        L_pot[i] = uzal_cost(Dataset(Ys[:,i]); samplesize = 1, K = KNN, w = w, Tw = 4*w)
    end
    return τ_pot, ts_pot, L_pot, flag
end

"""
    choose_next_node(n::Union{Node,Root}, func)

Returns one of the children of based on the function `func(Ls)->i_node`
"""
function choose_next_node(n::Node,func)
    N = N_children(n)
    if N == 0
        return nothing
    elseif N == 1
        return n.children[1]
    else
        return n.children[func(get_children_Ls(n))]
    end
end

"""
    choose_next_node(n::Union{Node,Root}, func)

Returns one of the children of based on the function `func(Ls)->i_node`
"""
function choose_next_node(n::Root,func)
    N = N_children(n)
    if N == 0
        return nothing
    elseif N == 1
        return n.children[1]
    else
        return n.children[rand(1:N)]
    end
end

# some function, which choose the next children:
softmax(xi,X,β=1) = exp(-β*xi)/sum(exp.(-β*X))
minL(Ls) = argmin(Ls)

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
    expand!(n::Union{Node,Root}, data::Dataset, w::Int, choose_func, max_depth=20)

This is one single rollout and backprop of the tree.

* `n`: Starting node
* `data`: data
* `w`: Theiler Window
* `choose_func`: Function to choose next node with
"""
function expand!(n::Union{Node,Root}, data::Dataset{D, T}, w::Int, choose_func,
            delays = 0:100; max_depth::Int=20, KNN::Int=3, verbose=false) where {D, T<:Real}
    current_node = n

    for i=1:max_depth # loops until converged or max_depth is reached
        # next embedding step

        # only if it was not already computed
        if current_node.children == nothing
            τs, ts, Ls, converged = next_embedding(current_node, data, w, delays; KNN = KNN)
            if converged
                break
            else
                # spawn children
                children = []
                for j = 1:length(τs)
                    push!(children, Node(τs[j],ts[j],Ls[j],[get_τs(current_node); τs[j]], [get_ts(current_node); ts[j]], nothing))
                end
                current_node.children = children
            end
        end

        # choose next node
        current_node = choose_next_node(current_node, choose_func)
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
function backprop!(n::AbstractTreeElement,τs,ts,L_min)
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
function mc_delay(data, w, choose_func, delays, N::Int=40;  max_depth::Int=20, KNN::Int = 3, verbose::Bool=false)

    # initialize tree
    tree = Root()

    for i=1:N

        expand!(tree, data, w, choose_func, delays; KNN = KNN, max_depth = max_depth)

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

# given the tree, return the best embedding
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
