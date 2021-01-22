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
runs = 5
runs2 = 10


println("\nTesting MCDTS complete tree, Lorenz63 univariate:")
@testset "MCDTS single rollout on univariate data" begin

    # L
    Random.seed!(1234)
    @time tree = MCDTS.mc_delay(Dataset(data[:,1]),w1,(L)->(MCDTS.softmaxL(L,β=2.)),delays,runs)
    best_node = MCDTS.best_embedding(tree)
    println(best_node)

    # L with tws
    Random.seed!(1234)
    @time tree2 = MCDTS.mc_delay(Dataset(data[:,1]),w1,(L)->(MCDTS.softmaxL(L,β=2.)),delays,runs; tws = 2:4:delays[end])
    best_node2 = MCDTS.best_embedding(tree2)
    println(best_node2)
    @test best_node2.τs == best_node.τs
    @test best_node2.L > best_node.L

    # L with tws and threshold
    Random.seed!(1234)
    @time tree3 = MCDTS.mc_delay(Dataset(data[:,1]),w1,(L)->(MCDTS.softmaxL(L,β=2.)),delays,runs; tws = 2:4:delays[end], threshold = 0.5)
    best_node3 = MCDTS.best_embedding(tree3)
    println(best_node3)
    @test length(best_node3.τs) < length(best_node2.τs)
    @test best_node3.τs == [0, 18]
    @test best_node3.L > best_node2.L

    # FNN with threshold
    Random.seed!(1234)
    @time tree = MCDTS.mc_delay(Dataset(data[:,1]),w1,(L)->(MCDTS.softmaxL(L,β=2.)),delays,runs; FNN = true, threshold = 0.05)
    best_node = MCDTS.best_embedding(tree)
    println(best_node)
    L_YY = MCDTS.compute_delta_L(data[:,1], best_node.τs, delays[end];  w = w1)
    @test L_YY > best_node2.L
    @test length(best_node.τs) == 4
    @test best_node.τs == [0, 57, 71,13]

end


println("\nTesting MCDTS complete tree, Lorenz63 multivariate:")
@testset "MCDTS single rollout on multivariate data" begin

    Random.seed!(1234)
    @time tree = MCDTS.mc_delay(data,w,(L)->(MCDTS.softmaxL(L,β=2.)),delays,runs2; tws = 2:4:delays[end], threshold = 0.05)
    println(tree)
    best_node = MCDTS.best_embedding(tree)

    Random.seed!(1234)
    @time tree2 = MCDTS.mc_delay(data,w,(L)->(MCDTS.softmaxL(L,β=2.)),delays, runs2; FNN = true, threshold = 0.05)
    best_node2 = MCDTS.best_embedding(tree2)
    println(best_node2)
    L_YY = MCDTS.compute_delta_L(data, best_node2.τs, best_node2.ts, delays[end];  w = w, tws = 2:4:delays[end])

    @test best_node.L < L_YY
    @test length(best_node.τs) > length(best_node2.τs)

end

true
