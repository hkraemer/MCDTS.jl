using DynamicalSystems
using MCDTS

ds = Systems.lorenz()
data = trajectory(ds,200)

w=15
delays = 0:100
FNN = false
# start testing everything in little parts here
# in the end there will be a more complete test
# this is more of a playing ground and debugging area here.

tree = MCDTS.Root()

MCDTS.expand!(tree, Dataset(data[:,1]), w, (L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN)
println(MCDTS.best_embedding(tree))

Delta_L = MCDTS.compute_delta_L(data[:,1], [0,18,9], delays[end];  w = w)

tree = MCDTS.Root()
MCDTS.expand!(tree,data,w,(L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN)
println(MCDTS.best_embedding(tree))

Delta_L = MCDTS.compute_delta_L(data, [0,0,0,100,60,85,44], [3,2,1,1,1,1,1], delays[end];  w = w)

Y_final, τ_vals, ts_vals, Ls, ε★s = MCDTS.pecuzal_embedding(data; τs = delays , w = w)


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
