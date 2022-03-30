begin
    import Pkg
    Pkg.activate(".")

    using MCDTS
    using DelayEmbeddings
    using DynamicalSystemsBase
    using DelimitedFiles
    using ChaosTools
    using Random

    using PyPlot
    pygui(true)

    lo = Systems.lorenz()
    tr = trajectory(lo, 500; dt = 0.01, Ttr = 10)

    # Lyapunov exponent and time
    λ = ChaosTools.lyapunov(lo, 100000, dt=0.01; Ttr=1000)
    lyap_time = Int(floor((1/λ) / 0.01))

    x = tr[:,1]
    y = tr[:,2]
    w1 = DelayEmbeddings.estimate_delay(tr[:,1],"mi_min")
    w2 = DelayEmbeddings.estimate_delay(tr[:,2],"mi_min")
    w3 = DelayEmbeddings.estimate_delay(tr[:,3],"mi_min")
    theiler = maximum([w1,w2,w3])
end


## Predictions based on embedding
begin 
    T_steps = 8*lyap_time
    x1 = tr[1:end-T_steps,1]
    x2 = tr[end-T_steps+1:end,1]
    y1 = tr[1:end-T_steps,2]
    y2 = tr[end-T_steps+1:end,2]

    dmax = 10
    # cao
    𝒟, τ_tde1, _ = optimal_traditional_de(x1, "afnn"; dmax = dmax, w = w1)
    optimal_d_tde1 = size(𝒟, 2)
    τ_cao = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
    Y_cao = genembed(x1, τ_cao.*(-1))

    # kennel's method
    𝒟, τ_tde2, _ = optimal_traditional_de(x1, "fnn"; dmax = dmax, w = w1)
    optimal_d_tde2 = size(𝒟, 2)
    τ_kennel = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
    Y_kennel = genembed(x1, τ_kennel.*(-1))

    # hegger's method
    𝒟, τ_tde3, _ = optimal_traditional_de(x1, "ifnn"; dmax = dmax, w = w1)
    optimal_d_tde3 = size(𝒟, 2)
    τ_hegger = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
    Y_hegger = genembed(x1, τ_hegger.*(-1))
end

# mcdts
begin
    using MCDTS
    Random.seed!(1234)
    taus = 0:150
    trials2 = 30
    trials = 10
    Tw_in = 50 # prediction horizon in
    max_depth = 10
    Tw_out = 150 # prediction horizon out
    KNN = 5 # nearest neighbors for pred method
    K = KNN
    samplesize = 0.2
    PredMeth = MCDTS.local_model("zeroth", KNN, Tw_out, Tw_in, trials2)
    PredLoss = MCDTS.PredictionLoss(4)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType,0,samplesize), MCDTS.Continuity_function(13,samplesize))
end
@time tree = mcdts_embedding(Dataset(x1), optmodel, w1, taus, trials; max_depth = max_depth, choose_func = (L->(MCDTS.softmaxL(L,β=1.))), verbose=true)
best_node = MCDTS.best_embedding(tree)
τ_mcdts = best_node.τs
# τ_mcdts = [0, 18, 123, 82, 66, 114, 29, 73]
Y_mcdts = genembed(x1, τ_mcdts.*(-1))

begin
    using MCDTS
    Random.seed!(1234)
    taus = 0:150
    trials2 = 30
    trials = 10
    max_depth = 10
    Tw = 100 # prediction horizon out
    KNN = 5 # nearest neighbors for pred method
    K = KNN
    samplesize = 0.2
    PredMeth = MCDTS.local_model("linear", KNN, Tw, 1, trials2)
    PredLoss = MCDTS.PredictionLoss(4)
    PredType = MCDTS.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = MCDTS.MCDTSOptimGoal(MCDTS.Prediction_error(PredType,0,samplesize), MCDTS.Continuity_function(13,samplesize))
end
@time tree = mcdts_embedding(Dataset(x1), optmodel, w1, taus, trials; max_depth = max_depth, verbose=true)
best_node = MCDTS.best_embedding(tree)
τ_mcdts2 = best_node.τs
#τ_mcdts2 = [0,18,9]
Y_mcdts2 = genembed(x1, τ_mcdts.*(-1))


## make predictions based on these embedding models
offset = 300
prediction_cao = deepcopy(Y_cao[1:end-offset])
prediction_kennel = deepcopy(Y_kennel[1:end-offset])
prediction_hegger = deepcopy(Y_hegger[1:end-offset])
prediction_mcdts = deepcopy(Y_mcdts[1:end-offset])


K1 = 5
#T_steps = 483
for T = 1:T_steps
    println(T)
    # iterated one step
    predicted_cao, _ = MCDTS.local_zeroth_prediction(prediction_cao, K1; theiler = w1)
    push!(prediction_cao, predicted_cao)
    predicted_kennel, _ = MCDTS.local_zeroth_prediction(prediction_kennel, K1; theiler = w1)
    push!(prediction_kennel, predicted_kennel)
    predicted_hegger, _ = MCDTS.local_zeroth_prediction(prediction_hegger, K1; theiler = w1)
    push!(prediction_hegger, predicted_hegger)
    predicted_mcdts, _ = MCDTS.local_zeroth_prediction(prediction_mcdts, K1; theiler = w1)
    push!(prediction_mcdts, predicted_mcdts)
    # linear
    # predicted_cao, _ = MCDTS.local_linear_prediction(prediction_cao, K1; theiler = w1)
    # push!(prediction_cao, predicted_cao)
    # predicted_kennel, _ = MCDTS.local_linear_prediction(prediction_kennel, K1; theiler = w1)
    # push!(prediction_kennel, predicted_kennel)
    # predicted_hegger, _ = MCDTS.local_linear_prediction(prediction_hegger, K1; theiler = w1)
    # push!(prediction_hegger, predicted_hegger)
    # predicted_mcdts, _ = MCDTS.local_linear_prediction(prediction_mcdts, K1; theiler = w1)
    # push!(prediction_mcdts, predicted_mcdts)
end

## Plot predictions
begin
    sp = length(tr[1:end-offset])-T_steps
    t2 = (-sp+1:T_steps) ./ lyap_time

    figure(figsize=(20,10))
    subplot(4,1,1)
    plot(t2[8000:end], tr[8000:end-offset,1], ".-", label="true data")
    plot(t2[end-T_steps:end], prediction_cao[end-T_steps:end,1], ".-", label="Cao")
    title("x-component (iterated one-step)")
    xlim(-5, 12)
    ylim(-20,20)
    xlabel("Lyapunov time units")
    legend()
    grid()

    subplot(4,1,2)
    plot(t2[8000:end], tr[8000:end-offset,1], ".-", label="true data")
    plot(t2[end-T_steps:end], prediction_kennel[end-T_steps:end,1], ".-", label="Kennel")
    title("x-component (iterated one-step)")
    xlim(-5, 12)
    ylim(-20,20)
    xlabel("Lyapunov time units")
    legend()
    grid()

    subplot(4,1,3)
    plot(t2[8000:end], tr[8000:end-offset,1], ".-", label="true data")
    plot(t2[end-T_steps:end], prediction_hegger[end-T_steps:end,1], ".-", label="Hegger")
    title("x-component (iterated one-step)")
    xlim(-5, 12)
    ylim(-20,20)
    xlabel("Lyapunov time units")
    legend()
    grid()

    subplot(4,1,4)
    plot(t2[8000:end], tr[8000:end-offset,1], ".-", label="true data")
    plot(t2[end-T_steps:end], prediction_mcdts[end-T_steps:end,1], ".-", label="MCDTS")
    title("x-component (iterated one-step)")
    xlim(-5, 12)
    ylim(-20,20)
    xlabel("Lyapunov time units")
    legend()
    grid()

end
