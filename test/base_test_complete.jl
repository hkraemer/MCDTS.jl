using Pkg
current_dir = pwd()
Pkg.activate(current_dir)

using DynamicalSystems
using MCDTS

ds = Systems.lorenz()
data = trajectory(ds,200)
w = 10
println("starting...")
tree = MCDTS.mc_delay(data,w,MCDTS.softmaxL,10)
println("..done")
best_node = MCDTS.best_embedding(tree)
println(best_node)

true
