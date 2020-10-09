using Pkg
current_dir = pwd()
Pkg.activate(current_dir)

using DynamicalSystems
using MCDTS

ds = Systems.lorenz()
data = trajectory(ds,200)

w=10
# start testing everything in little parts here
# in the end there will be a more complete test
# this is more of a playing ground and debugging area here.

tree = MCDTS.Root()

MCDTS.expand!(tree, data, w, (L)->(MCDTS.softmaxL(L,β=2.)),0:100,3)

println(MCDTS.best_embedding(tree))

MCDTS.expand!(tree,data,w,MCDTS.softmaxL,0:100,3)

println(MCDTS.best_embedding(tree))

MCDTS.expand!(tree,data,w,MCDTS.softmaxL,0:100,3)

println(MCDTS.best_embedding(tree))

MCDTS.expand!(tree,data,w,MCDTS.softmaxL,0:100,3)

println(MCDTS.best_embedding(tree))
#println(best_node)
#L=MCDTS.get_children_Ls(tree.children[1])



#MCDTS.expand!(tree,data,w,MCDTS.softmaxL,3)

true
