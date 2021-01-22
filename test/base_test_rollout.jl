using DynamicalSystems
using MCDTS
using Random
using Test
using DelayEmbeddings


# Check Lorenz System
Random.seed!(1234)
ds = Systems.lorenz()
data = trajectory(ds,200)
data = data[10001:end,:]

w1 = DelayEmbeddings.estimate_delay(data[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(data[:,2],"mi_min")
w3 = DelayEmbeddings.estimate_delay(data[:,3],"mi_min")
w = maximum(hcat(w1,w2,w3))
delays = 0:100


println("\nTesting MCDTS single rollouts, Lorenz63 univariate:")

@testset "MCDTS single rollout on univariate data" begin

    # L
    tree = MCDTS.Root()
    FNN = false
    @time MCDTS.expand!(tree, Dataset(data[:,1]), w, (L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN)
    best = MCDTS.best_embedding(tree)
    println(best)
    Delta_L = MCDTS.compute_delta_L(data[:,1], best.τs, delays[end];  w = w)
    @test best.L == Delta_L

    # L with different tws
    tree = MCDTS.Root()
    @time MCDTS.expand!(tree, Dataset(data[:,1]), w, (L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN, tws = 2:2:delays[end])
    best = MCDTS.best_embedding(tree)
    println(best)
    Delta_L = MCDTS.compute_delta_L(data[:,1], best.τs, delays[end];  w = w, tws = 2:2:delays[end])
    @time Y, τ_vals, ts_vals, Ls , εs = DelayEmbeddings.pecuzal_embedding(data[:,1];
                                   τs = delays , w = w, econ = true)
    @test τ_vals == best.τs
    @test sum(Ls) == Delta_L == best.L

    # L with different tws and L_threshold set
    tree = MCDTS.Root()
    @time MCDTS.expand!(tree, Dataset(data[:,1]), w, (L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN, tws = 2:4:delays[end], threshold = 0.5)
    best = MCDTS.best_embedding(tree)
    println(best)
    Delta_L = MCDTS.compute_delta_L(data[:,1], best.τs, delays[end];  w = w, tws = 2:4:delays[end])
    @test Delta_L == best.L
    @test length(best.τs) == 2

    # FNN
    tree = MCDTS.Root()
    FNN = true
    Random.seed!(1234)
    @time MCDTS.expand!(tree, Dataset(data[:,1]), w, (L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN)
    best = MCDTS.best_embedding(tree)
    println(best)
    # FNN with threshold
    tree2 = MCDTS.Root()
    FNN = true
    Random.seed!(1234)
    @time MCDTS.expand!(tree2, Dataset(data[:,1]), w, (L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN, threshold = 0.05)
    best2 = MCDTS.best_embedding(tree2)
    println(best2)
    @test best2.L > best.L
    @test best2.L > 0.05
    @test length(best2.τs) < length(best.τs)
    @test best2.τs == [0, 57 ,71, 13]

end


println("\nTesting MCDTS single rollouts, Lorenz63 multivariate:")

@testset "MCDTS single rollout on multivariate data" begin

    # L
    Random.seed!(1234)
    tree = MCDTS.Root()
    FNN = false
    @time MCDTS.expand!(tree,data,w,(L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN)
    best = MCDTS.best_embedding(tree)
    println(best)
    Delta_L = MCDTS.compute_delta_L(data, best.τs, best.ts, delays[end];  w = w)
    @test Delta_L == best.L
    @test best.τs == [0, 8, 0]
    @test best.ts == [1, 2, 2]


    # L with threshold and tws
    tree = MCDTS.Root()
    @time MCDTS.expand!(tree,data,w,(L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN, tws = 2:4:delays[end], threshold = 0.05)
    best = MCDTS.best_embedding(tree)
    println(best)

    # FNN with threshold
    Random.seed!(1234)
    tree = MCDTS.Root()
    FNN = true
    @time MCDTS.expand!(tree,data,w,(L)->(MCDTS.softmaxL(L,β=2.)),delays; FNN = FNN, threshold = 0.01)
    best = MCDTS.best_embedding(tree)
    println(best)
    @test best.L > 0.01
    @test best.τs == [0, 8, 5]
    @test best.ts == [1, 2, 1]

end

true
