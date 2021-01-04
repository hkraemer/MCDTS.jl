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
@time tree = MCDTS.mc_delay(Dataset(data[:,1]),w1,(L)->(MCDTS.softmaxL(L,β=2.)),0:100,15)
println("..done")
best_node = MCDTS.best_embedding(tree)
println(best_node)

println("starting MCDTS univariate FNN...")
@time tree = MCDTS.mc_delay(Dataset(data[:,1]),w1,(L)->(MCDTS.softmaxL(L,β=2.)),0:100,15; FNN = true)
best_node = MCDTS.best_embedding(tree)
println(best_node)
YY = genembed(data[:,1],best_node.τs)
YY = regularize(YY)
L_YY = uzal_cost(YY; w = w1, Tw = 4*w1, samplesize=1)
println("..done")
println(tree)
println("L for the FNN best trial: $L_YY")

println("starting PECUZAL univariate...")
@time Y_pec, τ_vals_pec, ts_vals_pec, Ls_pec , εs_pec = MCDTS.pecuzal_embedding(data[:,1];
                                                            τs = 0:100 , w = w1)
L_pec = minimum(Ls_pec)
println("L=$L_pec, τs=$τ_vals_pec, ts=$ts_vals_pec")
println("..done")

println("starting MCDTS multivariate...")
@time tree = MCDTS.mc_delay(data,w,(L)->(MCDTS.softmaxL(L,β=2.)),0:100,15)
println("..done")
println(tree)

println("starting MCDTS multivariate FNN...")
@time tree = MCDTS.mc_delay(data,w,(L)->(MCDTS.softmaxL(L,β=2.)),0:100,15; FNN = true)
best_node = MCDTS.best_embedding(tree)
YY = genembed(data[:,1],best_node.τs)
YY = regularize(YY)
L_YY = uzal_cost(YY; w = w1, Tw = 4*w1, samplesize=1)
println("..done")
println(tree)
println("L for the FNN best trial: $L_YY")

println("starting PECUZAL multivariate...")
@time Y_pec, τ_vals_pec, ts_vals_pec, Ls_pec , εs_pec = MCDTS.pecuzal_embedding(data;
                                                            τs = 0:100 , w = w)
L_pec = minimum(Ls_pec)
println("L=$L_pec, τs=$τ_vals_pec, ts=$ts_vals_pec")
println("..done")



true
