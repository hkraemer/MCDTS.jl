println("\nTesting MCDTS complete tree, coupled Logistic, CCM:")
@time begin
@testset "MCDTS single rollout on univariate Logistic map data" begin

    Lval = 3500
    x = zeros(Lval)
    y = zeros(Lval)
    r = 3.8
    r2 = 3.5
    βxy = 0.02
    βyx = 0.1
    x[1]=0.4
    y[1]=0.2

    for i = 2:Lval
        x[i]=x[i-1]*(r-r*x[i-1]-βxy*y[i-1])
        y[i]=y[i-1]*(r2-r2*y[i-1]-βyx*x[i-1])
    end

    w1 = DelayEmbeddings.estimate_delay(x, "mi_min")
    w2 = DelayEmbeddings.estimate_delay(y, "mi_min")

    test1 = DelayEmbeddings.standardize(x)
    test2 = DelayEmbeddings.standardize(y)
    # try MCDTS with CCM
    taus1 = 0:10 # the possible delay vals
    trials = 20 # the sampling of the tree

    Random.seed!(1234)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.CCM_ρ(test2), MCDTS.Range_function())
    tree = mcdts_embedding(Dataset(test1), optmodel, w1, taus1, trials)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    ts_mcdts = best_node.ts
    Lval = L(best_node)

    # less fid points
    Random.seed!(1234)
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.CCM_ρ(test2,1,0.5), MCDTS.Range_function())
    tree2 = mcdts_embedding(Dataset(test1), optmodel2, w1, taus1, trials)
    best_node2 = MCDTS.best_embedding(tree2)
    τ_mcdts2 = best_node2.τs
    ts_mcdts2 = best_node2.ts
    L2 = L(best_node2)

    @test Lval < -0.95
    @test Lval - .01 < L2 < Lval + .01
    @test length(τ_mcdts) ==  3 == length(τ_mcdts2)
    @test τ_mcdts[1] == 0 == τ_mcdts2[1]
    @test τ_mcdts[2] == 2 == τ_mcdts2[2]
    @test τ_mcdts[3] == 1 == τ_mcdts2[3]

    Random.seed!(1234)
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.CCM_ρ(test1), MCDTS.Range_function())
    tree2 = mcdts_embedding(Dataset(test2), optmodel2, w2, taus1, trials)
    best_node = MCDTS.best_embedding(tree2)
    τ_mcdts = best_node.τs
    ts_mcdts = best_node.ts
    Lval = L(best_node)

    @test Lval < -0.997
    @test length(τ_mcdts) ==  3
    @test τ_mcdts[1] == 0
    @test τ_mcdts[2] == 1
    @test τ_mcdts[3] == 2

end
end
true