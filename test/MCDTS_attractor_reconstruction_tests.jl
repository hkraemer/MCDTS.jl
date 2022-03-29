
delays = 0:50
runs = 10
runs2 = 10
T_steps = 50

println("\nTesting MCDTS complete tree, Lorenz63 univariate:")
@time begin
@testset "MCDTS single rollout on univariate data" begin

    # L
    Random.seed!(1234)
    pecuzal = MCDTS.PecuzalOptim()
    tree = mcdts_embedding(Dataset(data[:,1]), pecuzal, w1, delays, runs)
    best_node = MCDTS.best_embedding(tree)
    @test best_node.τs == [0, 9, 42, 20]
    @test -0.94 < L(best_node) < -0.929

    # L with tws
    Random.seed!(1234)
    tws = 2:4:delays[end]
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws), MCDTS.Continuity_function())
    tree2 = mcdts_embedding(Dataset(data[:,1]), optmodel2, w1, delays, runs)
    best_node2 = MCDTS.best_embedding(tree2)
    @test best_node2.τs == best_node.τs
    @test L(best_node2) > L(best_node)

    # L with tws and less fiducials for computation
    Random.seed!(1234)
    tws = 2:2:delays[end]
    samplesize = 0.5
    optmodel3 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws,samplesize), MCDTS.Continuity_function())
    tree3 = mcdts_embedding(Dataset(data[:,1]), optmodel3, w1, delays, runs)
    best_node3 = MCDTS.best_embedding(tree3)
    @test best_node3.τs == best_node.τs
    @test L(best_node) - 0.1 < L(best_node3) < L(best_node) + 0.1

    # L with tws and threshold
    Random.seed!(1234)
    optmodel4 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(-0.5,3,tws), MCDTS.Continuity_function())
    tree4 = mcdts_embedding(Dataset(data[:,1]), optmodel4, w1, delays, runs)
    best_node4 = MCDTS.best_embedding(tree4)
    @test length(best_node4.τs) < length(best_node2.τs)
    @test best_node4.τs == [0, 9]
    @test L(best_node4) > L(best_node2)

    # FNN with threshold
    Random.seed!(1234)
    optmodel4 = MCDTS.MCDTSOptimGoal(MCDTS.FNN_statistic(0.05), MCDTS.Continuity_function())
    tree = mcdts_embedding(Dataset(data[:,1]), optmodel4, w1, delays, runs)
    best_node = MCDTS.best_embedding(tree)
    @test best_node.τs == [0, 27, 21, 6]

    # FNN with threshold and less fid-points
    Random.seed!(1234)
    optmodel4 = MCDTS.MCDTSOptimGoal(MCDTS.FNN_statistic(0.05,2,0.8), MCDTS.Continuity_function())
    tree = mcdts_embedding(Dataset(data[:,1]), optmodel4, w1, delays, runs)
    best_node_less = MCDTS.best_embedding(tree)
    @test best_node_less.τs == [0, 27, 21, 6]

    L_YY = MCDTS.compute_delta_L(data[:,1], best_node.τs, delays[end];  w = w1)
    L_YY2 = MCDTS.compute_delta_L(data[:,1], best_node_less.τs, delays[end];  w = w1)
    @test L_YY == L_YY2

end


println("\nTesting MCDTS complete tree, Lorenz63 multivariate:")
@testset "MCDTS single rollout on multivariate data" begin

    Random.seed!(1234)
    tws = 2:4:delays[end]
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(-0.05,3,tws), MCDTS.Continuity_function())
    tree = MCDTS.mc_delay(data, optmodel, w, delays, runs2)
    best_node = MCDTS.best_embedding(tree)
    
    Random.seed!(1234)
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.FNN_statistic(0.05), MCDTS.Continuity_function())
    tree2 = MCDTS.mc_delay(data, optmodel2, w, delays, runs2)
    best_node2 = MCDTS.best_embedding(tree2)
    L_YY = MCDTS.compute_delta_L(data, best_node2.τs, best_node2.ts, delays[end];  w = w, tws = 2:4:delays[end])
    
    @test best_node.τs == [0, 9, 42, 20]
    @test best_node.ts == [1, 1, 1, 1]
    @test best_node2.τs == [0, 22, 16, 7]
    @test best_node2.ts == [1, 2, 2, 1]
    @test -0.92 < L(best_node) < -0.91
    @test -0.43 < L_YY < -0.42
    
    # less fid points
    Random.seed!(1234)
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.FNN_statistic(0.05,2,0.2), MCDTS.Continuity_function())
    tree3 = MCDTS.mc_delay(data, optmodel2, w, delays, runs2)
    best_node3 = MCDTS.best_embedding(tree3)
    
    @test best_node3.τs == [0, 48, 39, 0, 2]
    @test best_node3.ts == [1, 3, 2, 3, 2]

end
end
true