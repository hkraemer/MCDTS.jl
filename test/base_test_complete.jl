using Pkg
current_dir = pwd()
Pkg.activate(current_dir)

using DynamicalSystems
using MCDTS

ds = Systems.lorenz()
data = trajectory(ds,200)
w = 15
println("starting...")
tree = MCDTS.mc_delay(Dataset(data[:,1]),w,MCDTS.softmaxL,0:100,100)
println("..done")
best_node = MCDTS.best_embedding(tree)
println(best_node)

true
