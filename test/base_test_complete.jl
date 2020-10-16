using Pkg
current_dir = pwd()
Pkg.activate(current_dir)
import Random
Random.seed!(1234)

using DynamicalSystems
using MCDTS

ds = Systems.lorenz()
data = trajectory(ds,200)
data = data[1:10000,:]
w1 = estimate_delay(data[:,1],"mi_min")
w2 = estimate_delay(data[:,2],"mi_min")
w3 = estimate_delay(data[:,3],"mi_min")
w = maximum(hcat(w1,w2,w3))
println("starting MCDTS univariate...")
@time tree = MCDTS.mc_delay(Dataset(data[:,1]),w,(L)->(MCDTS.softmaxL(L,β=2.)),0:100,15)
println("..done")
println(tree)

println("starting PECUZAL univariate...")
@time Y_pec, τ_vals_pec, ts_vals_pec, Ls_pec , εs_pec = MCDTS.pecuzal_embedding(data[:,1];
                                                            τs = 0:100 , w = w)
L_pec = minimum(Ls_pec)
println("L=$L_pec, τs=$τ_vals_pec, ts=$ts_vals_pec")
println("..done")

println("starting MCDTS multivariate...")
@time tree = MCDTS.mc_delay(data,w,(L)->(MCDTS.softmaxL(L,β=2.)),0:100,15)
println("..done")
println(tree)

println("starting PECUZAL multivariate...")
@time Y_pec, τ_vals_pec, ts_vals_pec, Ls_pec , εs_pec = MCDTS.pecuzal_embedding(data;
                                                            τs = 0:100 , w = w)
L_pec = minimum(Ls_pec)
println("L=$L_pec, τs=$τ_vals_pec, ts=$ts_vals_pec")
println("..done")



true
