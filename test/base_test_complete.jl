using Pkg
current_dir = pwd()
Pkg.activate(current_dir)
import Random
Random.seed!(1234)

using DynamicalSystems
using MCDTS

ds = Systems.lorenz()
data = trajectory(ds,200)
w = 15
println("starting...")
tree = MCDTS.mc_delay(Dataset(data[:,1]),w,(L)->(MCDTS.softmaxL(L,β=2.)),0:100,40)
println("..done")
println(tree)

true
