module MCDTS




mutable struct Root
    children::Union{Array{Node,1},Nothing}
end

mutable struct Node{T}
    τ::Int
    L::T
    τs::Array{Int,1}

    children::Union{Array{Node,1},Nothing}
end


softmax(xi,X,β=1) = exp(-β*xi)/sum(exp.(-β*X))

function next_embedding(τs::AbstractArray{Int,1})
    # do the next embedding step. input is the delays so far

    # return all possible new τs, their Ls, and flag if converged
    return τs, Ls, converged
end

function choose_next_node(n::Union{Node,Root})
    # returns one of the children of n based on softmax? probability of their L values

end

# this is one single rollout of the tree
function expand(n::Union{Node,Root}, max_depth=20)

    current_node = n

    if i=1:max_depth # loops until converged or max_depth is reached

        # next embedding step
        τs, Ls, converged = next_embedding(current_node.τs)

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





end # module
