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

    # noise level
    σ = .05

    # Parameters analysis:
    dmax = 12   # maximum dimension for traditional tde
    trials1 = 80 # trials for MCDTS univariate
    trials2 = 100 # trials for MCDTS multivariate
    taus1 = 0:100 # possible delays
    taus2 = 0:50 # possible delays for PRED optimization
    max_depth = 15 # depth of the tree
    Tw = 1  # time horizon for PRED
    KK = 1 # considered nearest neighbors for PRED

    # time series to pick
    t_idx_1 = 1         # univariate
    t_idx_2 = [1,2]     # multivariate

    # initial conditions
    Random.seed!(234)
    number_of_ics = 100 # number of different initial conditions
    ics = [rand(2) for i in 1:number_of_ics]

end

@time begin
# loop over different F's
results = @distributed (vcat) for i in eachindex(ics)

    # set different initial condition and get trajectory
    ic = ics[i]
    hen = Systems.henon(ic)
    tr = trajectory(hen, 10030; Ttr = 1000)

    # normalize time series
    data = regularize(tr)

    Random.seed!(234)

    T_steps = 31 # 15*lyap_time

    x = data[:,1]
    x_n = data[:,1] .+ σ*randn(length(data))

    x1 = x[1:end-T_steps]       # training
    x2 = x[end-T_steps+1:end]   # prediction
    x1_n = x_n[1:end-T_steps]
    x2_n = x_n[end-T_steps+1:end]

    z1 = data[1:10000,t_idx_2[2]]
    z1_n = data[1:10000,t_idx_2[2]] .+ σ*randn(length(data[1:10000]))

    data_sample = Dataset(x1,z1)
    data_sample_n = Dataset(x1_n,z1_n)

    w1 = DelayEmbeddings.estimate_delay(x1, "mi_min")
    w1_n = DelayEmbeddings.estimate_delay(x1_n, "mi_min")

    σ₂ = sqrt(var(x2[1:T_steps]))   # rms deviation for normalization
    σ₂_n = sqrt(var(x2_n[1:T_steps]))

    # make the reconstructions and then the predictions
    # cao
    MSEs_cao = zeros(T_steps)
    𝒟, τ_tde1, _ = optimal_traditional_de(x1, "afnn"; dmax = dmax, w = w1)
    optimal_d_tde1 = size(𝒟, 2)
    τ_cao = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
    Y = genembed(x1, τ_cao .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs_cao[j] = MCDTS.compute_mse(prediction[1:j,1], x2[1:j]) / σ₂
    end

    MSEs_cao_n = zeros(T_steps)
    𝒟, τ_tde1, _ = optimal_traditional_de(x1_n, "afnn"; dmax = dmax, w = w1_n)
    optimal_d_tde1 = size(𝒟, 2)
    τ_cao_n = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
    Y = genembed(x1_n, τ_cao_n .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_cao_n[j] = MCDTS.compute_mse(prediction[1:j,1], x2_n[1:j]) / σ₂_n
    end

    # kennel's method
    MSEs_kennel = zeros(T_steps)
    𝒟, τ_tde2, _ = optimal_traditional_de(x1, "fnn"; dmax = dmax, w = w1)
    optimal_d_tde2 = size(𝒟, 2)
    τ_kennel = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
    Y = genembed(x1, τ_kennel .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs_kennel[j] = MCDTS.compute_mse(prediction[1:j,1], x2[1:j]) / σ₂
    end

    MSEs_kennel_n = zeros(T_steps)
    𝒟, τ_tde2, _ = optimal_traditional_de(x1_n, "fnn"; dmax = dmax, w = w1_n)
    optimal_d_tde2 = size(𝒟, 2)
    τ_kennel_n = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
    Y = genembed(x1_n, τ_kennel_n .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_kennel_n[j] = MCDTS.compute_mse(prediction[1:j,1], x2_n[1:j]) / σ₂_n
    end

    # hegger's method
    MSEs_hegger = zeros(T_steps)
    𝒟, τ_tde3, _ = optimal_traditional_de(x1, "ifnn"; dmax = dmax, w = w1)
    optimal_d_tde3 = size(𝒟, 2)
    τ_hegger = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
    Y = genembed(x1, τ_hegger .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs_hegger[j] = MCDTS.compute_mse(prediction[1:j,1], x2[1:j]) / σ₂
    end

    MSEs_hegger_n = zeros(T_steps)
    𝒟, τ_tde3, _ = optimal_traditional_de(x1_n, "ifnn"; dmax = dmax, w = w1_n)
    optimal_d_tde3 = size(𝒟, 2)
    τ_hegger_n = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
    Y = genembed(x1_n, τ_hegger_n .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_hegger_n[j] = MCDTS.compute_mse(prediction[1:j,1], x2_n[1:j]) / σ₂_n
    end

    # pecuzal
    MSEs_pec = zeros(T_steps)
    𝒟, τ_pec, _, L, _ = pecuzal_embedding(x1; τs = taus1, w = w1)
    Y = genembed(x1, τ_pec .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs_pec[j] = MCDTS.compute_mse(prediction[1:j,1], x2[1:j]) / σ₂
    end

    MSEs_pec_n = zeros(T_steps)
    𝒟, τ_pec_n, _, L, _ = pecuzal_embedding(x1_n; τs = taus1, w = w1_n)
    Y = genembed(x1_n, τ_pec_n .* (-1))
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_pec_n[j] = MCDTS.compute_mse(prediction[1:j,1], x2_n[1:j]) / σ₂_n
    end

    MSEs_pec2 = zeros(T_steps)
    𝒟, τ_pec2, ts_pec2, L, _ = pecuzal_embedding(data_sample; τs = taus1, w = w1)
    Y = genembed(data_sample, τ_pec2 .* (-1), ts_pec2)
    if sum(ts_pec2 .== 1)>0
        tts = findall(x -> x==1, ts_pec2)[1]
    else
        tts = ts_pec2[1]
    end
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs_pec2[j] = MCDTS.compute_mse(prediction[1:j,tts], x2[1:j]) / σ₂
    end

    MSEs_pec2_n = zeros(T_steps)
    𝒟, τ_pec2_n, ts_pec2_n, L, _ = pecuzal_embedding(data_sample_n; τs = taus1, w = w1_n)
    Y = genembed(data_sample_n, τ_pec2_n .* (-1), ts_pec2_n)
    if sum(ts_pec2_n .== 1)>0
        tts = findall(x -> x==1, ts_pec2_n)[1]
    else
        tts = ts_pec2_n[1]
    end
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_pec2_n[j] = MCDTS.compute_mse(prediction[1:j,tts], x2_n[1:j]) / σ₂_n
    end


    # Output
    tuple(MSEs_cao, MSEs_cao_n ,MSEs_kennel, MSEs_kennel_n ,MSEs_hegger,
                MSEs_hegger_n ,MSEs_pec, MSEs_pec_n, MSEs_pec2, MSEs_pec2_n)

end

end


varnames = ["MSEs_cao", "MSEs_cao_n", "MSEs_kennel", "MSEs_kennel_n",
    "MSEs_hegger", "MSEs_hegger_n", "MSEs_pec", "MSEs_pec_n", "MSEs_pec2",
    "MSEs_pec2_n"]

for i = 1:length(varnames)
    writestr = "results_Henon_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
