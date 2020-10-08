using Pkg
current_dir = pwd()
Pkg.activate(current_dir)

using DynamicalSystems
using MCDTS

ds = Systems.lorenz()
data = trajectory(ds,100)


# start testing everything in little parts here
# in the end there will be a more complete test
# this is more of a playing ground and debugging area here.



## this is the expand! function, deconstructed to debug it
r = MCDTS.Root()

# first embedding is a dummy function

# next embedding step
current_node = r

τs, ts, Ls, converged = MCDTS.next_embedding(current_node,data,0)
# spawn children

children = []
for i in eachindex(τs)
    push!(children, MCDTS.Node(τs[i],Ls[i],[MCDTS.get_τs(current_node);τs[i]], [MCDTS.get_ts(current_node);ts[i]], nothing))
end
current_node.children = children

current_node = MCDTS.choose_next_node(current_node, MCDTS.minL)

# this should be a proper embedding

τs, ts, Ls, converged = MCDTS.next_embedding(current_node,data,1)

children = []
for i in eachindex(τs)
    push!(children, MCDTS.Node(τs[i],Ls[i],[MCDTS.get_τs(current_node);τs[i]], [MCDTS.get_ts(current_node);ts[i]], nothing))
end
current_node.children = children

current_node = MCDTS.choose_next_node(current_node, MCDTS.minL)


true
