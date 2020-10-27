using Pkg
current_dir = pwd()
Pkg.activate(current_dir)

using DynamicalSystems
using MCDTS

ds = Systems.lorenz()
data = trajectory(ds,200)

w=15
delays = 0:100
FNN = true
# start testing everything in little parts here
# in the end there will be a more complete test
# this is more of a playing ground and debugging area here.

tree = MCDTS.Root()

MCDTS.expand!(tree, Dataset(data[:,1]), w, (L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN)
println(MCDTS.best_embedding(tree))

tree = MCDTS.Root()
MCDTS.expand!(tree,data,w,(L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN)
println(MCDTS.best_embedding(tree))
MCDTS.expand!(tree,data,w,(L)->(MCDTS.softmaxL(L,β=1.)),delays; FNN = FNN)
println(MCDTS.best_embedding(tree))
MCDTS.expand!(tree,data,w,(L)->(MCDTS.softmaxL(L,β=8.)),delays; FNN = FNN)
println(MCDTS.best_embedding(tree))

tree = MCDTS.Root()
MCDTS.expand!(tree,Dataset(data[:,1]),w,(L)->(MCDTS.softmaxL(L,β=1.)),delays; FNN = FNN)
println(MCDTS.best_embedding(tree))

tree = MCDTS.Root()
MCDTS.expand!(tree,data,w,(L)->(MCDTS.softmaxL(L,β=1.)),delays; FNN = FNN)
println(MCDTS.best_embedding(tree))

true
