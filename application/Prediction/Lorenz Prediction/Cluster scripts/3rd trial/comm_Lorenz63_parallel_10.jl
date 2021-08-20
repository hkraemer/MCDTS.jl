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
    Random.seed!(234)
    number_of_ics = 100 # number of different initial conditions
    ics = [rand(3) for i in 1:number_of_ics]

end

@time begin
# loop over different F's
results = @distributed (vcat) for i in eachindex(ics)

    # set different initial condition and get trajectory
    ic = ics[i]
    lo = Systems.lorenz(ic)
    tr = trajectory(lo, 108.8; dt = dt, Ttr = 10)

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

    z1 = data[1:10000,t_idx_2[2]]
    z1_n = data[1:10000,t_idx_2[2]] .+ σ*randn(length(data[1:10000]))

    data_sample = Dataset(x1,z1)
    data_sample_n = Dataset(x1_n,z1_n)

    w1 = DelayEmbeddings.estimate_delay(x1, "mi_min")
    w1_n = DelayEmbeddings.estimate_delay(x1_n, "mi_min")

    σ₂ = sqrt(var(x2[1:T_steps]))   # rms deviation for normalization
    σ₂_n = sqrt(var(x2_n[1:T_steps]))


    # make the reconstructions and then the predictions

    # mcdts L
    MSEs_mcdts2_L_n = zeros(T_steps)
    tree = MCDTS.mc_delay(data_sample_n, w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials2; tws = 2:taus1[end], verbose=true)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts2_n = best_node.τs
    ts_mcdts2_n = best_node.ts
    Y = genembed(data_sample_n, τ_mcdts2_n .* (-1), ts_mcdts2_n)
    tts = findall(x -> x==1, ts_mcdts2_n)[1]
    prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
    for j = 1:T_steps
        MSEs_mcdts2_L_n[j] = MCDTS.compute_mse(prediction[1:j,tts], x2_n[1:j]) / σ₂_n
    end


    # Output
    tuple(MSEs_mcdts2_L_n)

end

end


varnames = ["MSEs_mcdts2_L_n"]

for i = 1:length(varnames)
    writestr = "results_Lorenz63_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
