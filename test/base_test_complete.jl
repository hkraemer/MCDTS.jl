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
#tree = MCDTS.mc_delay(Dataset(data[:,1]),w,(L)->(MCDTS.softmaxL(L,β=.5)),0:100,40)
#tree = MCDTS.mc_delay(data,w,(L)->(MCDTS.softmaxL(L,β=.5)),0:100,25)
tree = MCDTS.mc_delay(data,w,(L)->(MCDTS.softmaxL(L,β=1.)),0:100,300,verbose=true)
println("..done")
println(tree)

true
