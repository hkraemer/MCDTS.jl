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

Pick one of the children of `n` with values `τ` and `t`. If there is none, return `nothing`.
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
    flag = false

    return τ_pot, ts_pot, L_pot, flag
end

"""
    choose_next_node(n::Union{Node,Root}, func)

Returns one of the children of based on the function `func(Ls)->i_node`
"""
function choose_next_node(n::AbstractTreeElement,func)
    N = N_children(n)
    if N == 0
        return nothing
    elseif N == 1
        return n.children[1]
    else
        return n.children[func(get_children_Ls(n))]
        error("Not yet implemented.")
    end
end

softmax(xi,X,β=1) = exp(-β*xi)/sum(exp.(-β*X))
minL(Ls) = argmin(Ls)

function softmaxL(Ls; β=7)
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
    expand!(n::Union{Node,Root}, data, w, choose_func, max_depth=20)

This is one single rollout and backprop of the tree.

* `n`: Starting node
* `data`: data
* `w`: Theiler Window
* `choose_func`: Function to choose next node with
"""
function expand!(n::Union{Node,Root}, data, w, choose_func, max_depth=20)
    current_node = n

    for i=1:max_depth # loops until converged or max_depth is reached

        # next embedding step

        # only if it was not already computed
        if current_node.children == nothing
            τs, ts, Ls, converged = next_embedding(current_node, data, w)

            if converged
                break
            else

                # spawn children
                children = []
                for j in eachindex(τs)
                    push!(children, Node(τs[j],ts[j],Ls[j],[get_τs(current_node); τs[j]], [get_ts(current_node); ts[j]], nothing))
                end
                current_node.children = children
            end
        end

        # choose next node
        current_node = choose_next_node(current_node, choose_func)
    end

    # now backprop the values (actually we go to top to bottom, but we know were to end because we got the correct τs and ts)
    backprop!(n, current_node.τs, current_node.ts, current_node.L)

end



function backprop!(n::AbstractTreeElement,τs,ts,L_min)
    current_node = choose_children(n,τs[1],ts[1])
    for i=2:length(τs)
        # the initial embedding step is left out of the backprop
        current_node = choose_children(current_node,τs[i],ts[i])
        current_node.L = L_min
    end
end



"""
    mc_delay(N::Int=100)

Do the monte carlo run with `N` trials, returns the tree.
"""
function mc_delay(data, w, choose_func, N::Int=40)

    # initialize tree
    tree = Root()

    for i=1:N

        expand!(tree, data, w, choose_func)

        if (i%1)==0
            println(i,"/",N)
            #println(best_embedding(tree))
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
