using Pkg
current_dir = pwd()
Pkg.activate(current_dir)

using MCDTS
using DynamicalSystems

ds = Systems.lorenz()
data = trajectory(ds,200)


# start testing everything in little parts here
# in the end there will be a more complete test
# this is more of a playing ground and debugging area here.

w = 20
delays = 0:100
FNN = false # true for FNN-statistic minimization and false for L-statistic
## this is the expand! function, deconstructed to debug it
r = MCDTS.Root()

# first embedding is a dummy function

# next embedding step
current_node = r

τs, ts, Ls, converged = MCDTS.next_embedding(current_node,data,w, delays; FNN = FNN, Tw = 100)
# spawn children
println(Ls)
children = []
for i in eachindex(τs)
    push!(children, MCDTS.Node(τs[i],ts[i],Ls[i],[MCDTS.get_τs(current_node);τs[i]], [MCDTS.get_ts(current_node);ts[i]], nothing))
end
current_node.children = children

current_node = MCDTS.choose_next_node(current_node, MCDTS.softmaxL)

# this should be a proper embedding

τs, ts, Ls, converged = MCDTS.next_embedding(current_node,data,w,delays; FNN = FNN)
println(Ls)
children = []
for i in eachindex(τs)
    push!(children, MCDTS.Node(τs[i],ts[i],Ls[i],[MCDTS.get_τs(current_node);τs[i]], [MCDTS.get_ts(current_node);ts[i]], nothing))
end
current_node.children = children

current_node = MCDTS.choose_next_node(current_node, MCDTS.softmaxL)


τs, ts, Ls, converged = MCDTS.next_embedding(current_node,data,w,delays; FNN = FNN)
println(Ls)
children = []
for i in eachindex(τs)
    push!(children, MCDTS.Node(τs[i],ts[i],Ls[i],[MCDTS.get_τs(current_node);τs[i]], [MCDTS.get_ts(current_node);ts[i]], nothing))
end
current_node.children = children

current_node = MCDTS.choose_next_node(current_node, MCDTS.softmaxL)

τs, ts, Ls, converged = MCDTS.next_embedding(current_node,data,w,delays; FNN = FNN)
println(Ls)
children = []
for i in eachindex(τs)
    push!(children, MCDTS.Node(τs[i],ts[i],Ls[i],[MCDTS.get_τs(current_node);τs[i]], [MCDTS.get_ts(current_node);ts[i]], nothing))
end
current_node.children = children

current_node = MCDTS.choose_next_node(current_node, MCDTS.softmaxL)

τs, ts, Ls, converged = MCDTS.next_embedding(current_node,data,w,delays; FNN = FNN)
println(Ls)
children = []
for i in eachindex(τs)
    push!(children, MCDTS.Node(τs[i],ts[i],Ls[i],[MCDTS.get_τs(current_node);τs[i]], [MCDTS.get_ts(current_node);ts[i]], nothing))
end
current_node.children = children

current_node = MCDTS.choose_next_node(current_node, MCDTS.softmaxL)

τs, ts, Ls, converged = MCDTS.next_embedding(current_node,data,w,delays; FNN = FNN)
println(Ls)
children = []
for i in eachindex(τs)
    push!(children, MCDTS.Node(τs[i],ts[i],Ls[i],[MCDTS.get_τs(current_node);τs[i]], [MCDTS.get_ts(current_node);ts[i]], nothing))
end
current_node.children = children

current_node = MCDTS.choose_next_node(current_node, MCDTS.softmaxL)

τs, ts, Ls, converged = MCDTS.next_embedding(current_node,data,w,delays; FNN = FNN)
println(Ls)
children = []
for i in eachindex(τs)
    push!(children, MCDTS.Node(τs[i],ts[i],Ls[i],[MCDTS.get_τs(current_node);τs[i]], [MCDTS.get_ts(current_node);ts[i]], nothing))
end
current_node.children = children

current_node = MCDTS.choose_next_node(current_node, MCDTS.softmaxL)

# this is converged

print(MCDTS.best_embedding(r))
# now start from the top again
#function backprop!(n::AbstractTreeElement,τs,ts,L_min)

MCDTS.backprop!(r,current_node.τs,current_node.ts,current_node.L)
true
