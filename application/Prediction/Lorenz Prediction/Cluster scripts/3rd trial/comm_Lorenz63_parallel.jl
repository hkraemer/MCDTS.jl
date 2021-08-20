## We apply the MCDTS-approach to the Lorenz system and compare the results of
# the reconstruction to the ones gained from PECUZAL and standard TDE. We look
# at many realizations of the system, perform the reconstructions and make predictions
# from them

using ClusterManagers
using Distributed
@everywhere N_tasks = parse(Int, ARGS[1])
@everywhere N_worker = N_tasks
addprocs(SlurmManager(N_worker))

@everywhere begin
    using ClusterManagers
    using Distributed
    using IterTools
    using MCDTS
    using DynamicalSystems
    using DelayEmbeddings
    using RecurrenceAnalysis
    using Statistics
    using DelimitedFiles
    using Random

    # System integration time step:
    dt = 0.01

    # noise level
    σ = .05

    # Parameters analysis:
    dmax = 12   # maximum dimension for traditional tde
    trials1 = 80 # trials for MCDTS univariate
    trials2 = 100 # trials for MCDTS multivariate
    taus1 = 0:100 # possible delays
    taus2 = 0:25 # possible delays for PRED optimization
    max_depth = 15 # depth of the tree
    Tw = 1  # time horizon for PRED
    KK = 1 # considered nearest neighbors for PRED

    # time series to pick
    t_idx_1 = 1         # univariate
    t_idx_2 = [1,3]     # multivariate

    # initial conditions
    number_of_ics = 100 # number of different initial conditions
    ics = [rand(3) for i in 1:number_of_ics]

end

@time begin
# loop over different F's
results = @distributed (vcat) for i in eachindex(ics)

    # set different initial condition and get trajectory
    ic = ics[i]
    lo = Systems.lorenz(ic)
    tr = trajectory(lo, 58.8; dt = dt, Ttr = 10)

    # normalize time series
    data = regularize(tr)

    Random.seed!(234)

    T_steps = 881 # 8*lyap_time

    x = data[:,1]
    x_n = data[:,1] .+ σ*randn(length(data))

    x1 = x[1:end-T_steps]       # training
    x2 = x[end-T_steps+1:end]   # prediction
    x1_n = x_n[1:end-T_steps]
    x2_n = x_n[end-T_steps+1:end]

    z1 = data[1:5000,t_idx_2[2]]
    z1_n = data[1:5000,t_idx_2[2]] .+ σ*randn(length(data[1:5000]))

    data_sample = Dataset(x1,z1)
    data_sample_n = Dataset(x1_n,z1_n)

    w1 = DelayEmbeddings.estimate_delay(x1, "mi_min")
    w1_n = DelayEmbeddings.estimate_delay(x1_n, "mi_min")

    σ₂ = sqrt(var(x2[1:T_steps]))   # rms deviation for normalization
    σ₂_n = sqrt(var(x2_n[1:T_steps]))

    # preallocation
    MSEs = zeros(13,T_steps)
    MSEs_n = zeros(13,T_steps)

    # make the reconstructions and then the predictions
    # cao
    𝒟, τ_tde1, _ = optimal_traditional_de(x1, "afnn"; dmax = dmax, w = w1)
    optimal_d_tde1 = size(𝒟, 2)
    τ_cao = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
    Y = genembed(x1, τ_cao .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[1,i] = MCDTS.compute_mse(prediction[1:i,1], x2[1:i]) / σ₂
    end

    𝒟, τ_tde1, _ = optimal_traditional_de(x1_n, "afnn"; dmax = dmax, w = w1_n)
    optimal_d_tde1 = size(𝒟, 2)
    τ_cao_n = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
    Y = genembed(x1_n, τ_cao_n .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[1,i] = MCDTS.compute_mse(prediction[1:i,1], x2_n[1:i]) / σ₂_n
    end

    # kennel's method
    𝒟, τ_tde2, _ = optimal_traditional_de(x1, "fnn"; dmax = dmax, w = w1)
    optimal_d_tde2 = size(𝒟, 2)
    τ_kennel = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
    Y = genembed(x1, τ_kennel .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[2,i] = MCDTS.compute_mse(prediction[1:i,1], x2[1:i]) / σ₂
    end

    𝒟, τ_tde2, _ = optimal_traditional_de(x1_n, "fnn"; dmax = dmax, w = w1_n)
    optimal_d_tde2 = size(𝒟, 2)
    τ_kennel_n = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
    Y = genembed(x1_n, τ_kennel_n .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[2,i] = MCDTS.compute_mse(prediction[1:i,1], x2_n[1:i]) / σ₂_n
    end

    # hegger's method
    𝒟, τ_tde3, _ = optimal_traditional_de(x1, "ifnn"; dmax = dmax, w = w1)
    optimal_d_tde3 = size(𝒟, 2)
    τ_hegger = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
    Y = genembed(x1, τ_hegger .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[3,i] = MCDTS.compute_mse(prediction[1:i,1], x2[1:i]) / σ₂
    end

    𝒟, τ_tde3, _ = optimal_traditional_de(x1_n, "ifnn"; dmax = dmax, w = w1_n)
    optimal_d_tde3 = size(𝒟, 2)
    τ_hegger_n = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
    Y = genembed(x1_n, τ_hegger_n .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[3,i] = MCDTS.compute_mse(prediction[1:i,1], x2_n[1:i]) / σ₂_n
    end

    # pecuzal
    𝒟, τ_pec, _, L, _ = pecuzal_embedding(x1; τs = taus1, w = w1)
    Y = genembed(x1, τ_pec .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[4,i] = MCDTS.compute_mse(prediction[1:i,1], x2[1:i]) / σ₂
    end

    𝒟, τ_pec_n, _, L, _ = pecuzal_embedding(x1_n; τs = taus1, w = w1_n)
    Y = genembed(x1_n, τ_pec_n .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[4,i] = MCDTS.compute_mse(prediction[1:i,1], x2_n[1:i]) / σ₂_n
    end

    𝒟, τ_pec2, ts_pec2, L, _ = pecuzal_embedding(data_sample; τs = taus1, w = w1)
    Y = genembed(data_sample, τ_pec2 .* (-1), ts_pec2)
    tts = findall(x -> x==1, ts_pec2)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[5,i] = MCDTS.compute_mse(prediction[1:i,tts], x2[1:i]) / σ₂
    end

    𝒟, τ_pec2_n, ts_pec2_n, L, _ = pecuzal_embedding(data_sample_n; τs = taus1, w = w1_n)
    Y = genembed(data_sample_n, τ_pec2_n .* (-1), ts_pec2_n)
    tts = findall(x -> x==1, ts_pec2_n)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[5,i] = MCDTS.compute_mse(prediction[1:i,tts], x2_n[1:i]) / σ₂_n
    end

    # mcdts L
    tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials1; tws = 2:taus[end])
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    Y = genembed(x1, τ_mcdts .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[6,i] = MCDTS.compute_mse(prediction[1:i,1], x2[1:i]) / σ₂
    end

    tree = MCDTS.mc_delay(Dataset(x1_n), w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials1; tws = 2:taus[end])
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_n = best_node.τs
    Y = genembed(x1_n, τ_mcdts_n .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[6,i] = MCDTS.compute_mse(prediction[1:i,1], x2_n[1:i]) / σ₂_n
    end

    tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials2; tws = 2:taus[end])
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts2 = best_node.τs
    ts_mcdts2 = best_node.ts
    Y = genembed(data_sample, τ_mcdts2 .* (-1), ts_mcdts2)
    tts = findall(x -> x==1, ts_mcdts2)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[7,i] = MCDTS.compute_mse(prediction[1:i,tts], x2[1:i]) / σ₂
    end

    tree = MCDTS.mc_delay(data_sample_n, w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials2; tws = 2:taus[end])
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts2_n = best_node.τs
    ts_mcdts2_n = best_node.ts
    Y = genembed(data_sample_n, τ_mcdts2_n .* (-1), ts_mcdts2_n)
    tts = findall(x -> x==1, ts_mcdts2_n)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[7,i] = MCDTS.compute_mse(prediction[1:i,tts], x2_n[1:i]) / σ₂_n
    end

    # mcdts FNN
    tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials1; FNN=true, threshold = 0.01)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_fnn = best_node.τs
    Y = genembed(x1, τ_mcdts_fnn .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[8,i] = MCDTS.compute_mse(prediction[1:i,1], x2[1:i]) / σ₂
    end

    tree = MCDTS.mc_delay(Dataset(x1_n), w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials1; FNN=true, threshold = 0.05)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_fnn_n = best_node.τs
    Y = genembed(x1_n, τ_mcdts_fnn_n .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[8,i] = MCDTS.compute_mse(prediction[1:i,1], x2_n[1:i]) / σ₂_n
    end

    tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials2; FNN=true, threshold = 0.01)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_fnn2 = best_node.τs
    ts_mcdts_fnn2 = best_node.ts
    Y = genembed(data_sample, τ_mcdts_fnn2 .* (-1), ts_mcdts_fnn2)
    tts = findall(x -> x==1, ts_mcdts_fnn2)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[9,i] = MCDTS.compute_mse(prediction[1:i,tts], x2[1:i]) / σ₂
    end

    tree = MCDTS.mc_delay(data_sample_n, w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials2; FNN=true, threshold = 0.05)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_fnn2_n = best_node.τs
    ts_mcdts_fnn2_n = best_node.ts
    Y = genembed(data_sample_n, τ_mcdts_fnn2_n .* (-1), ts_mcdts_fnn2_n)
    tts = findall(x -> x==1, ts_mcdts_fnn2_n)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[9,i] = MCDTS.compute_mse(prediction[1:i,tts], x2_n[1:i]) / σ₂_n
    end

    # mcdts PRED MSE
    tree = MCDTS.mc_delay(Dataset(x1),w1,(L)->(MCDTS.softmaxL(L,β=2.)),
        taus2, trials1; max_depth = max_depth, PRED = true, KNN = KK,
        threshold = 5e-6)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_PRED = best_node.τs
    Y = genembed(x1, τ_mcdts_PRED .*(-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[10,i] = MCDTS.compute_mse(prediction[1:i,1], x2[1:i]) / σ₂
    end

    tree = MCDTS.mc_delay(Dataset(x1_n),w1_n,(L)->(MCDTS.softmaxL(L,β=2.)),
        taus2, trials1; max_depth = max_depth, PRED = true, KNN = KK,
        threshold = 5e-6)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_PRED_n = best_node.τs
    Y = genembed(x1_n, τ_mcdts_PRED_n .*(-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[10,i] = MCDTS.compute_mse(prediction[1:i,1], x2_n[1:i]) / σ₂_n
    end

    tree = MCDTS.mc_delay(data_sample,w1,(L)->(MCDTS.softmaxL(L,β=2.)),
        taus2, trials2; max_depth = max_depth, PRED = true, KNN = KK,
        threshold = 5e-6)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_PRED_multi = best_node.τs
    ts_mcdts_PRED_multi = best_node.ts
    Y = genembed(data_sample, τ_mcdts_PRED_multi .*(-1), ts_mcdts_PRED_multi)
    tts = findall(x -> x==1, ts_mcdts_PRED_multi)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[11,i] = MCDTS.compute_mse(prediction[1:i,tts], x2[1:i]) / σ₂
    end

    tree = MCDTS.mc_delay(data_sample_n,w1_n,(L)->(MCDTS.softmaxL(L,β=2.)),
        taus2, trials2; max_depth = max_depth, PRED = true, KNN = KK,
        threshold = 5e-6)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_PRED_multi_n = best_node.τs
    ts_mcdts_PRED_multi_n = best_node.ts
    Y = genembed(data_sample_n, τ_mcdts_PRED_multi_n .*(-1), ts_mcdts_PRED_multi_n)
    tts = findall(x -> x==1, ts_mcdts_PRED_multi_n)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[11,i] = MCDTS.compute_mse(prediction[1:i,tts], x2_n[1:i]) / σ₂_n
    end


    # mcdts PRED KL
    tree = MCDTS.mc_delay(Dataset(x1),w1,(L)->(MCDTS.softmaxL(L,β=2.)),
        taus2, trials1; max_depth = max_depth, PRED = true, KNN = KK,
        PRED_KL = true)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_PRED = best_node.τs
    Y = genembed(x1, τ_mcdts_PRED .*(-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[12,i] = MCDTS.compute_mse(prediction[1:i,1], x2[1:i]) / σ₂
    end

    tree = MCDTS.mc_delay(Dataset(x1_n),w1_n,(L)->(MCDTS.softmaxL(L,β=2.)),
        taus2, trials1; max_depth = max_depth, PRED = true, KNN = KK,
        PRED_KL = true)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_PRED_n = best_node.τs
    Y = genembed(x1_n, τ_mcdts_PRED_n .*(-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[12,i] = MCDTS.compute_mse(prediction[1:i,1], x2_n[1:i]) / σ₂_n
    end

    tree = MCDTS.mc_delay(data_sample,w1,(L)->(MCDTS.softmaxL(L,β=2.)),
        taus2, trials2; max_depth = max_depth, PRED = true, KNN = KK,
        PRED_KL = true)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_PRED_multi = best_node.τs
    ts_mcdts_PRED_multi = best_node.ts
    Y = genembed(data_sample, τ_mcdts_PRED_multi .*(-1), ts_mcdts_PRED_multi)
    tts = findall(x -> x==1, ts_mcdts_PRED_multi)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs[13,i] = MCDTS.compute_mse(prediction[1:i,tts], x2[1:i]) / σ₂
    end

    tree = MCDTS.mc_delay(data_sample_n,w1_n,(L)->(MCDTS.softmaxL(L,β=2.)),
        taus2, trials2; max_depth = max_depth, PRED = true, KNN = KK,
        PRED_KL = true)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_PRED_multi_n = best_node.τs
    ts_mcdts_PRED_multi_n = best_node.ts
    Y = genembed(data_sample_n, τ_mcdts_PRED_multi_n .*(-1), ts_mcdts_PRED_multi_n)
    tts = findall(x -> x==1, ts_mcdts_PRED_multi_n)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_n[13,i] = MCDTS.compute_mse(prediction[1:i,tts], x2_n[1:i]) / σ₂_n
    end

    # TODO HERE

    # Output
    tuple(τ_tde, optimal_d_tde, RQA_tde, L_tde,
        τ_pec, ts_pec, optimal_d_pec, RQA_pec, L_pec,
        best_node.τs, best_node.ts, optimal_d_mcdts, RQA_mcdts, L_mcdts)

end

end

writedlm("results_Lorenz96_N_$(N)_1d_params.csv", params)
writedlm("results_Lorenz96_N_$(N)_1d_Fs.csv", Fs)

varnames = ["tau_tde", "optimal_d_tde", "RQA_tde", "L_tde",
    "tau_pec", "ts_pec", "optimal_d_pec", "RQA_pec", "L_pec",
    "tau_MCDTS", "ts_MCDTS", "optimal_d_mcdts", "RQA_mcdts", "L_mcdts"]

for i = 1:length(varnames)
    writestr = "results_Lorenz96_N_$(N)_1d_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
