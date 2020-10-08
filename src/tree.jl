current_dir = pwd()
Pkg.activate(current_dir)

using DynamicalSystemsBase


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
    L::T
    τs::Array{Int,1}
    ts::Array{Int,1}
    children::Union{Array{Node,1},Nothing}
end

N_children(n::AbstractTreeElement) = n.children == nothing ? 0 : length(n.children)
get_τs(n::Node) = n.τs
get_ts(n::Node) = n.ts

softmax(xi,X,β=1) = exp(-β*xi)/sum(exp.(-β*X))

"""
    next_embedding(n::Node, Ys::Dataset{D, T}, w::Int, τs = 0:100) where {D::Int, T<:Real}

Performs the next embedding step.

???? What is τs=1:00

# Returns

* `τ_pot`: Next delay
* `ts_pot`: Index of the time series used (in case of multivariate time series)
* `L_pot`: L statistic of next embedding step
* `flag`: Did the embedding converge? i.e. L can not be further minimized anymore

"""
function next_embedding(n::Node, Ys::Dataset, w::Int, τs = 0:100)
    τs_old = get_τs(n)
    L_old = n.L
    ts_old = get_ts(n)
    # do the next embedding step
    τ_pot, ts_pot, L_pot, flag = give_potential_delays(Ys, τs, w, Tuple(τs_old),
                                                           Tuple(ts_old), L_old)


    return τ_pot, ts_pot, L_pot, flag
end

"""
    next_embedding(n::Node, Ys::Dataset, w::Int, τs = 0:100)

The first embedding step
"""
function next_embedding(n::Root, Ys::Dataset, w::Int, τs = 0:100)
    τ_pot = [0]
    ts_pot = [1]
    L_pot = [1e10]
    flag = [false]

    return τ_pot, ts_pot, L_pot, flag
end

"""
    choose_next_node(n::Union{Node,Root})

Returns one of the children of n based on softmax? probability of their L values
"""
function choose_next_node(n::Union{Node,Root})
    N = N_children(n)
    if N == 0
        return nothing
    elseif N == 1
        return n.children[1]
    else
        error("Not yet implemented.")
    end
end


"""
    expand!(n::Union{Node,Root}, max_depth=20)

This is one single rollout of the tree
"""
function expand!(n::Union{Node,Root}, max_depth=20)

    current_node = n

    for i=1:max_depth # loops until converged or max_depth is reached

        # next embedding step
        τs, ts, Ls, converged = next_embedding(current_node)

        if converged
            break
        end

        # spawn children
        children = []
        for iτ in τs, iL in Ls
            push!(children, Node(iτ,iL,[get_τs(current_node);iτ], [get_ts(current_node);ts], nothing))
        end
        current_node.children = children

        # choose next node
        current_node = choose_next_node(current_node)
    end
end

"""
    mc_delay(N::Int=100)

Do the monte carlo run with `N` trials, returns the tree.
"""
function mc_delay(N::Int=100)

    # initialize tree
    tree = Root()

    for i=1:N
        expand!(tree)
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
            return current_node.τs
        else
            Ls = []
            for ic in current_node.children
                push!(Ls, ic.L)
            end
            # traverse down the tree always with minimal L
            current_node = current_node.children[argmin(Ls)]
        end
    end
    return current_node
end
