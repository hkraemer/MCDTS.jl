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

    # mcdts FNN
    MSEs_mcdts2_FNN = zeros(T_steps)
    tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials2; FNN=true, threshold = 0.01, verbose=true)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_fnn2 = best_node.τs
    ts_mcdts_fnn2 = best_node.ts
    Y = genembed(data_sample, τ_mcdts_fnn2 .* (-1), ts_mcdts_fnn2)
    tts = findall(x -> x==1, ts_mcdts_fnn2)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
    for j = 1:T_steps
        MSEs_mcdts2_FNN[j] = MCDTS.compute_mse(prediction[1:j,tts], x2[1:j]) / σ₂
    end

    MSEs_mcdts2_FNN_n = zeros(T_steps)
    tree = MCDTS.mc_delay(data_sample_n, w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials2; FNN=true, threshold = 0.05, verbose=true)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts_fnn2_n = best_node.τs
    ts_mcdts_fnn2_n = best_node.ts
    Y = genembed(data_sample_n, τ_mcdts_fnn2_n .* (-1), ts_mcdts_fnn2_n)
    tts = findall(x -> x==1, ts_mcdts_fnn2_n)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_mcdts2_FNN_n[j] = MCDTS.compute_mse(prediction[1:j,tts], x2_n[1:j]) / σ₂_n
    end

    # Output
    tuple(MSEs_mcdts2_FNN, MSEs_mcdts2_FNN_n)

end

end


varnames = ["MSEs_mcdts2_FNN", "MSEs_mcdts2_FNN_n"]

for i = 1:length(varnames)
    writestr = "results_Henon_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
