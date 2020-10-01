module MCDTS

include("pecuzal_methods.jl")

mutable struct Root
    children::Union{Array{Node,1},Nothing}
end

mutable struct Node{T}
    τ::Int
    L::T
    τs::Array{Int,1} # the complete vector with all τs chosen along this path
    ts::Array{Int,1} # vector which of the possibly multivariate time series is used at emebdinning step i

    children::Union{Array{Node,1},Nothing}
end


softmax(xi,X,β=1) = exp(-β*xi)/sum(exp.(-β*X))

function next_embedding(n::Node, Ys::Dataset{D, T}, w::Int, τs = 0:100) where {D::Int, T<:Real}
    τs_old = n.τs
    L_old = n.L
    ts_old = n.ts
    # do the next embedding step
    τ_pot, ts_pot, L_pot, flag = give_potential_delays(Ys, w, τs, Tuple(τs_old),
                                                           Tuple(ts_old), L_old)

    # return all possible new τs, their corresponding ts, their Ls, and flag
    # if converged, i.e. L can not be minimized anymore
    return τ_pot, ts_pot, L_pot, flag
end

function choose_next_node(n::Union{Node,Root})
    # returns one of the children of n based on softmax? probability of their L values

end

# this is one single rollout of the tree
function expand(n::Union{Node,Root}, max_depth=20)

    current_node = n

    if i=1:max_depth # loops until converged or max_depth is reached

        # next embedding step
        τs, Ls, converged = next_embedding(current_node)

        if converged
            break
        end

        # spawn children
        children = []
        for iτ in τs, iL in Ls
            push!(children, Node(iτ,iL,[current_node.τs;iτ],nothing))
        end
        current_node.children = children

        # choose next node
        current_node = choose_next_node(current_node)
    end
end

# do the monte carlo run, returns the tree
function mc_delay(N=100)

    # initialize tree
    tree = Root(nothing)

    for i=1:N
        expand(tree)
    end

    return tree
end

# given the tree, return the best embedding
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
end

end # module
