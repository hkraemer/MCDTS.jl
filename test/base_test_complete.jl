# Check Lorenz System
Random.seed!(1234)
ds = Systems.lorenz()
data = trajectory(ds,200)
data = data[2001:2:3750,:]

w1 = DelayEmbeddings.estimate_delay(data[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(data[:,2],"mi_min")
w3 = DelayEmbeddings.estimate_delay(data[:,3],"mi_min")
w = maximum(hcat(w1,w2,w3))
delays = 0:50
runs = 10
runs2 = 10
T_steps = 50

println("\nTesting MCDTS complete tree, Lorenz63 univariate:")
@testset "MCDTS single rollout on univariate data" begin

    # L
    Random.seed!(1234)
    pecuzal = MCDTS.PecuzalOptim()
    tree = mcdts_embedding(Dataset(data[:,1]), pecuzal, w1, delays, runs)
    best_node = MCDTS.best_embedding(tree)
    @test best_node.τs == [0, 9, 42, 20]
    @test -0.94 < best_node.L < -0.929

    # L with tws
    Random.seed!(1234)
    tws = 2:4:delays[end]
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws), MCDTS.Continuity_function())
    tree2 = mcdts_embedding(Dataset(data[:,1]), optmodel2, w1, delays, runs)
    best_node2 = MCDTS.best_embedding(tree2)
    @test best_node2.τs == best_node.τs
    @test best_node2.L > best_node.L

    # L with tws and less fiducials for computation
    Random.seed!(1234)
    tws = 2:2:delays[end]
    samplesize = 0.5
    optmodel3 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws,samplesize), MCDTS.Continuity_function())
    tree3 = mcdts_embedding(Dataset(data[:,1]), optmodel3, w1, delays, runs)
    best_node3 = MCDTS.best_embedding(tree3)
    @test best_node3.τs == best_node.τs
    @test best_node.L - 0.1 < best_node3.L < best_node.L + 0.1

    # L with tws and threshold
    Random.seed!(1234)
    optmodel4 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(-0.5,3,tws), MCDTS.Continuity_function())
    tree4 = mcdts_embedding(Dataset(data[:,1]), optmodel4, w1, delays, runs)
    best_node4 = MCDTS.best_embedding(tree4)
    @test length(best_node4.τs) < length(best_node2.τs)
    @test best_node4.τs == [0, 9]
    @test best_node4.L > best_node2.L

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
    @test -0.92 < best_node.L < -0.91
    @test -0.43 < L_YY < -0.42
    
    # less fid points
    Random.seed!(1234)
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.FNN_statistic(0.05,2,0.2), MCDTS.Continuity_function())
    tree3 = MCDTS.mc_delay(data, optmodel2, w, delays, runs2)
    best_node3 = MCDTS.best_embedding(tree3)
    
    @test best_node3.τs == [0, 48, 39, 0, 2]
    @test best_node3.ts == [1, 3, 2, 3, 2]

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
    tree = mcdts_embedding(Dataset(test1), optmodel, w1, taus1, trials)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    ts_mcdts = best_node.ts
    L = best_node.L

    # less fid points
    Random.seed!(1234)
    optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.CCM_ρ(test2,1,0.5), MCDTS.Range_function())
    tree2 = mcdts_embedding(Dataset(test1), optmodel2, w1, taus1, trials)
    best_node2 = MCDTS.best_embedding(tree2)
    τ_mcdts2 = best_node2.τs
    ts_mcdts2 = best_node2.ts
    L2 = best_node2.L

    @test L < -0.95
    @test L - .01 < L2 < L + .01
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
    L = best_node.L

    @test L < -0.997
    @test length(τ_mcdts) ==  3
    @test τ_mcdts[1] == 0
    @test τ_mcdts[2] == 1
    @test τ_mcdts[3] == 2

end

println("\nTesting MCDTS complete tree, Lorenz Prediction:")
@testset "MCDTS single rollout on prediction of Lorenz" begin

    max_depth = 15
    x1 = data[1:end-T_steps,1]
    x2 = data[end-T_steps+1:end,1]
    y1 = data[1:end-T_steps,2]
    y2 = data[end-T_steps+1:end,2]
    
    # Prediction range-function, zeroth predictor first comp-MSE
    delays = 0:5
    runs = 10
    
    Random.seed!(1234)
    Tw = 1 #prediction horizon
    KNN = 3 # nearest neighbors for pred method
    
    PredMeth = MCDTS.local_model("zeroth", KNN, Tw)
    PredLoss = MCDTS.PredictionLoss(1)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType), MCDTS.Range_function())
    
    tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth = max_depth)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    L_mcdts = best_node.L
    @test length(τ_mcdts) == 2
    @test τ_mcdts[2] == 2
    @test 0.045 < L_mcdts < 0.046
    
    # Prediction range-function, linear predictor first comp-MSE
    delays = 0:5
    runs = 5
    Random.seed!(1234)
    Tw = 1 #prediction horizon
    KNN = 1 # nearest neighbors for pred method
    
    PredMeth = MCDTS.local_model("linear", KNN, Tw)
    PredLoss = MCDTS.PredictionLoss(1)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType), MCDTS.Range_function())
    
    tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth = max_depth)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts2 = best_node.τs
    L_mcdts2 = best_node.L
    @test length(τ_mcdts2) == 6
    @test τ_mcdts2[2] == 1
    @test τ_mcdts2[3] == 2
    @test τ_mcdts2[4] == 4
    @test τ_mcdts2[5] == 5
    @test τ_mcdts2[6] == 3
    @test 5.8e-5 < L_mcdts2 < 5.9e-5
    
    # Prediction range-function, zeroth predictor mean-KL
    delays = 0:10
    runs = 10
    
    Random.seed!(1234)
    Tw = 1 #prediction horizon
    KNN = 3 # nearest neighbors for pred method
    
    PredMeth = MCDTS.local_model("zeroth", KNN, Tw)
    PredLoss = MCDTS.PredictionLoss(4)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType), MCDTS.Range_function())
    
    tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth = max_depth)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts3 = best_node.τs
    L_mcdts3 = best_node.L
    @test length(τ_mcdts3) == 2
    @test τ_mcdts3[2] == 3
    @test 0.0016 < L_mcdts3 < 0.0017
    
    # Prediction range-function, linear predictor first-comp-KL
    delays = 0:5
    runs = 5
    Random.seed!(1234)
    Tw = 1 #prediction horizon
    KNN = 3 # nearest neighbors for pred method
    
    PredMeth = MCDTS.local_model("linear", KNN, Tw)
    PredLoss = MCDTS.PredictionLoss(3)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType), MCDTS.Range_function())
    
    tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth = max_depth)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts4 = best_node.τs
    L_mcdts4 = best_node.L
    @test τ_mcdts4 == [0, 3]
    @test 1.11e-5 < L_mcdts4 < 1.12e-5
    
    # multivariate prediction range-function, zeroth predictor first-comp-MSE
    delays = 0:5
    runs = 5
    data_sample = Dataset(hcat(x1,y1))
    
    Random.seed!(1234)
    Tw = 1 #prediction horizon
    KNN = 3 # nearest neighbors for pred method
    
    PredMeth = MCDTS.local_model("zeroth", KNN, Tw)
    PredLoss = MCDTS.PredictionLoss(1)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType), MCDTS.Range_function())
    
    tree = mcdts_embedding(data_sample, optmodel, w1, delays, runs; max_depth = max_depth)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts5 = best_node.τs
    ts_mcdts5 = best_node.ts
    L_mcdts5 = best_node.L
    @test τ_mcdts5 == [0, 2]
    @test ts_mcdts5 == [1, 2]
    
    # Prediction Continuity-function, zeroth predictor first comp-MSE
    delays = 0:100
    runs = 10
    Random.seed!(1234)
    Tw = 1 #prediction horizon
    KNN = 3 # nearest neighbors for pred method
    
    PredMeth = MCDTS.local_model("zeroth", KNN, Tw)
    PredLoss = MCDTS.PredictionLoss(1)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType), MCDTS.Continuity_function())
    
    tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth = max_depth)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    L_mcdts = best_node.L
    @test τ_mcdts == [0, 9]
    @test 0.05 < L_mcdts < 0.06
    
    # Prediction Continuity-function, zeroth predictor first comp-MSE, more neighbors
    Random.seed!(1234)
    Tw = 1 #prediction horizon
    KNN = 8 # nearest neighbors for pred method
    
    PredMeth = MCDTS.local_model("zeroth", KNN, Tw)
    PredLoss = MCDTS.PredictionLoss(1)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType), MCDTS.Continuity_function())
    
    tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth = max_depth)
    best_node2 = MCDTS.best_embedding(tree)
    τ_mcdts2 = best_node2.τs
    L_mcdts2 = best_node2.L
    @test τ_mcdts2 == τ_mcdts
    @test 0.065 < L_mcdts2 < 0.066
    
    # Prediction Range-function, zeroth predictor first comp-MSE, Tw = 60
    delays = 0:5
    runs = 10
    Random.seed!(1234)
    Tw = 60 #prediction horizon
    KNN = 3 # nearest neighbors for pred method
    
    PredMeth = MCDTS.local_model("zeroth", KNN, Tw)
    PredLoss = MCDTS.PredictionLoss(1)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType), MCDTS.Range_function())
    
    tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth = max_depth)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    L_mcdts = best_node.L
    @test τ_mcdts == [0, 3, 4]
    @test 1.02 < L_mcdts < 1.03
    
    # Prediction Continuity-function, zeroth predictor first all-comp-MSE, Tw = 50, more neighbors
    delays = 0:80
    runs = 10
    Random.seed!(1234)
    Tw = 50 #prediction horizon
    KNN = 6 # nearest neighbors for pred method
    
    PredMeth = MCDTS.local_model("zeroth", KNN, Tw)
    PredLoss = MCDTS.PredictionLoss(2)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType), MCDTS.Continuity_function())
    
    tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth = max_depth)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    L_mcdts = best_node.L
    @test τ_mcdts == [0, 58, 39, 25, 49, 46, 44, 15]
    @test 0.78 < L_mcdts < 0.79
    
    
    # Prediction Continuity-function, linear predictor first all-comp-MSE, Tw = 50, more neighbors
    delays = 0:80
    runs = 10
    Random.seed!(1234)
    Tw = 5 #prediction horizon
    KNN = 6 # nearest neighbors for pred method
    
    PredMeth = MCDTS.local_model("linear", KNN, Tw)
    PredLoss = MCDTS.PredictionLoss(2)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType), MCDTS.Continuity_function())
    
    tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth = max_depth)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    L_mcdts = best_node.L
    @test τ_mcdts == [0, 9, 5]
    @test 0.02 < L_mcdts < 0.03

end

true