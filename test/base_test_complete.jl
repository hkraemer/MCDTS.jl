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
runs = 10
runs2 = 10

##


# L = 3500
# x = zeros(L)
# y = zeros(L)
# r = 3.8
# r2 = 3.5
# βxy = 0.02
# βyx = 0.1
# x[1]=0.4
# y[1]=0.2
#
# for i = 2:L
#     x[i]=x[i-1]*(r-r*x[i-1]-βxy*y[i-1])
#     y[i]=y[i-1]*(r2-r2*y[i-1]-βyx*x[i-1])
# end
#
# w1 = DelayEmbeddings.estimate_delay(x, "mi_min")
# w2 = DelayEmbeddings.estimate_delay(y, "mi_min")
#
# test1 = DelayEmbeddings.standardize(x)
# test2 = DelayEmbeddings.standardize(y)
# # try MCDTS with CCM
# taus1 = 0:10 # the possible delay vals
# trials = 20 # the sampling of the tree
# Random.seed!(1234)
# optmodel = MCDTS.MCDTSOptimGoal(MCDTS.CCM_ρ(test2), MCDTS.Range_function())
# @time tree = mcdts_embedding(Dataset(test1), optmodel, w1, taus1, trials)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts = best_node.τs
# ts_mcdts = best_node.ts
# L = best_node.L
#
# @test L < -0.95601
# @test length(τ_mcdts) ==  3
# @test τ_mcdts[1] == 0
# @test τ_mcdts[2] == 2
# @test τ_mcdts[3] == 3
#
# Random.seed!(1234)
# optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.CCM_ρ(test1), MCDTS.Range_function())
# @time tree2 = mcdts_embedding(Dataset(test2), optmodel, w2, taus1, trials)
# best_node = MCDTS.best_embedding(tree2)
# τ_mcdts = best_node.τs
# ts_mcdts = best_node.ts
# L = best_node.L
#
# @test L < -0.99232
# @test length(τ_mcdts) ==  2
# @test τ_mcdts[1] == 0
# @test τ_mcdts[2] == 1
##


println("\nTesting MCDTS complete tree, Lorenz63 univariate:")
@testset "MCDTS single rollout on univariate data" begin

    # L
    Random.seed!(1234)
    pecuzal = MCDTS.PecuzalOptim()
    @time tree = mcdts_embedding(Dataset(data[:,1]), pecuzal, w1, delays, runs)
    best_node = MCDTS.best_embedding(tree)

    # L with tws
    Random.seed!(1234)
    tws = 2:4:delays[end]
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws), MCDTS.Continuity_function())
    @time tree2 = mcdts_embedding(Dataset(data[:,1]), optmodel2, w1, delays, runs)
    best_node2 = MCDTS.best_embedding(tree2)
    @test best_node2.τs == best_node.τs
    @test best_node2.L > best_node.L

    # L with tws and threshold
    Random.seed!(1234)
    optmodel3 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(-0.5,3,tws), MCDTS.Continuity_function())
    @time tree3 = mcdts_embedding(Dataset(data[:,1]), optmodel3, w1, delays, runs)
    best_node3 = MCDTS.best_embedding(tree3)
    @test length(best_node3.τs) < length(best_node2.τs)
    @test best_node3.τs == [0, 61, 14]
    @test best_node3.L > best_node2.L

    # FNN with threshold
    Random.seed!(1234)
    optmodel4 = MCDTS.MCDTSOptimGoal(MCDTS.FNN_statistic(0.05), MCDTS.Continuity_function())
    @time tree = mcdts_embedding(Dataset(data[:,1]), optmodel4, w1, delays, runs)
    best_node = MCDTS.best_embedding(tree)

    L_YY = MCDTS.compute_delta_L(data[:,1], best_node.τs, delays[end];  w = w1)
    @test L_YY > best_node2.L
    @test length(best_node.τs) == 4
    @test best_node.τs == [0, 61, 14, 51]

end


println("\nTesting MCDTS complete tree, Lorenz63 multivariate:")
@testset "MCDTS single rollout on multivariate data" begin

    Random.seed!(1234)
    tws = 2:4:delays[end]
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(-0.05,3,tws), MCDTS.Continuity_function())
    @time tree = MCDTS.mc_delay(data, optmodel, w, delays, runs2)
    best_node = MCDTS.best_embedding(tree)

    Random.seed!(1234)
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.FNN_statistic(0.05), MCDTS.Continuity_function())
    @time tree2 = MCDTS.mc_delay(data, optmodel2, w, delays, runs2)
    best_node2 = MCDTS.best_embedding(tree2)
    L_YY = MCDTS.compute_delta_L(data, best_node2.τs, best_node2.ts, delays[end];  w = w, tws = 2:4:delays[end])

    @test best_node.L < L_YY
    @test length(best_node.τs) > length(best_node2.τs)

end

println("\nTesting MCDTS complete tree, coupled Logistic, CCM:")
@testset "MCDTS single rollout on univariate Logistic map data" begin

    L = 3500
    x = zeros(L)
    y = zeros(L)
    r = 3.8
    r2 = 3.5
    βxy = 0.02
    βyx = 0.1
    x[1]=0.4
    y[1]=0.2

    for i = 2:L
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
    @time tree = mcdts_embedding(Dataset(test1), optmodel, w1, taus1, trials)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    ts_mcdts = best_node.ts
    L = best_node.L

    @test L < -0.95601
    @test length(τ_mcdts) ==  3
    @test τ_mcdts[1] == 0
    @test τ_mcdts[2] == 2
    @test τ_mcdts[3] == 3

    Random.seed!(1234)
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.CCM_ρ(test1), MCDTS.Range_function())
    @time tree2 = mcdts_embedding(Dataset(test2), optmodel, w2, taus1, trials)
    best_node = MCDTS.best_embedding(tree2)
    τ_mcdts = best_node.τs
    ts_mcdts = best_node.ts
    L = best_node.L

    @test L < -0.99232
    @test length(τ_mcdts) ==  2
    @test τ_mcdts[1] == 0
    @test τ_mcdts[2] == 1


    # L = 3500
    # x = zeros(L)
    # y = zeros(L)
    # r = 3.8
    # r2 = 3.5
    # βxy = 0.02
    # βyx = 0.1
    # x[1]=0.4
    # y[1]=0.2
    #
    # for i = 2:L
    #     x[i]=x[i-1]*(r-r*x[i-1]-βxy*y[i-1])
    #     y[i]=y[i-1]*(r2-r2*y[i-1]-βyx*x[i-1])
    # end
    #
    # w1 = DelayEmbeddings.estimate_delay(x, "mi_min")
    # w2 = DelayEmbeddings.estimate_delay(y, "mi_min")
    #
    # test1 = DelayEmbeddings.standardize(x)
    # test2 = DelayEmbeddings.standardize(y)
    # # try MCDTS with CCM
    # taus1 = 0:10 # the possible delay vals
    # trials = 20 # the sampling of the tree
    # Random.seed!(1234)
    # tree = MCDTS.mc_delay(Dataset(test1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials;
    #     verbose=false, CCM = true, Y_CCM = test2)
    # best_node = MCDTS.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # ts_mcdts = best_node.ts
    # L = best_node.L
    #
    # @test L < -0.95601
    # @test length(τ_mcdts) ==  3
    # @test τ_mcdts[1] == 0
    # @test τ_mcdts[2] == 2
    # @test τ_mcdts[3] == 3
    #
    # Random.seed!(1234)
    # tree = MCDTS.mc_delay(Dataset(test2), w2, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials;
    #     verbose=false, CCM = true, Y_CCM = test1)
    # best_node = MCDTS.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # ts_mcdts = best_node.ts
    # L = best_node.L
    #
    # @test L < -0.99232
    # @test length(τ_mcdts) ==  2
    # @test τ_mcdts[1] == 0
    # @test τ_mcdts[2] == 1

end

true
