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

tree = MCDTS.Root()

MCDTS.expand!(tree,data,1,MCDTS.minL,3)

best_node = MCDTS.best_embedding(tree)

true
