## We apply the MCDTS-approach to the Lorenz system and compare the results of
# the reconstruction to the ones gained from PECUZAL and standard TDE. Here we
# compute the Lyapunov spectra for a bunch of parameter-settings
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

    # Parameters data:
    N = 40 # number of oscillators
    Fs = 3.5:0.004:5 # parameter spectrum
    dt = 0.1 # sampling time
    total = 5000  # time series length

    # Parameters analysis:
    ε = 0.05  # recurrence threshold
    dmax = 10   # maximum dimension for traditional tde
    lmin = 2   # minimum line length for RQA
    trials = 80 # trials for MCDTS
    taus = 0:100 # possible delays
    L_threshold = 0 # threshold for minimum tolerable ΔL decrease per embedding cycle

    # pick 3 time series
    t_idx = [2,15,28]

    # init Lorenz96
    u0 = [0.590; 0.766; 0.566; 0.460; 0.794; 0.854; 0.200; 0.298]
    lo96 = Systems.lorenz96(N; F = 3.5)

    params = tuple(N,dt,total,ε,dmax,lmin,trials,taus,t_idx,L_threshold)
end

@time begin
# loop over different ic's
results = @distributed (vcat) for i in eachindex(Fs)

    F = Fs[i]
    set_parameter!(lo96, 1, F)
    data = trajectory(lo96, total*dt; dt = dt, Ttr = 2500*dt)
    data_sample = data[:,t_idx]

    # Traditional time delay embedding
    τ_tde = zeros(Int,3)
    optimal_d_tde = zeros(Int,3)
    RQA_tde = zeros(15,3)
    L_tde = zeros(3)
    for i = 1:3
        𝒟, τ_tde[i], _ = optimal_traditional_de(data_sample[:,i], "fnn"; dmax = dmax)
        optimal_d_tde[i] = size(𝒟, 2)
        R = RecurrenceMatrix(𝒟, ε; fixedrate = true)
        RQA = rqa(R; theiler = τ_tde[i], lmin = lmin)
        RQA_tde[:,i] = hcat(RQA...)
        τ_tdes = [(k-1)*τ_tde[i] for k = 1:optimal_d_tde[i]]
        L_tde[i] = MCDTS.compute_delta_L(data_sample[:,i], τ_tdes, taus[end]; w = τ_tde[i], tws = 2:2:taus[end])
    end

    # PECUZAL
    theiler = Int(floor(mean(τ_tde)))
    𝒟_pec, τ_pec, ts_pec, Ls_pec , _ = DelayEmbeddings.pecuzal_embedding(data_sample; τs = taus , w = theiler, econ = true, L_threshold = L_threshold)
    optimal_d_pec = size(𝒟_pec,2)
    R = RecurrenceMatrix(𝒟_pec, ε; fixedrate = true)
    RQA = rqa(R; theiler = theiler, lmin = lmin)
    RQA_pec = hcat(RQA...)
    L_pec = sum(Ls_pec)

    # MCDTS
    tree = MCDTS.mc_delay(Dataset(data_sample), theiler, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:2:taus[end], threshold = L_threshold, max_depth = 15)
    best_node = MCDTS.best_embedding(tree)
    𝒟_mcdts = genembed(data_sample, best_node.τs, best_node.ts)
    optimal_d_mcdts = size(𝒟_mcdts,2)
    R = RecurrenceMatrix(𝒟_mcdts, ε; fixedrate = true)
    RQA = rqa(R; theiler = theiler, lmin = lmin)
    RQA_mcdts = hcat(RQA...)
    L_mcdts = best_node.L

    # Output
    tuple(τ_tde, optimal_d_tde, RQA_tde, L_tde,
        τ_pec, ts_pec, optimal_d_pec, RQA_pec, L_pec,
        best_node.τs, best_node.ts, optimal_d_mcdts, RQA_mcdts, L_mcdts)

end

end

writedlm("results_Lorenz96_N_$(N)_3d_params.csv", params)
writedlm("results_Lorenz96_N_$(N)_3d_Fs.csv", Fs)

varnames = ["tau_tde", "optimal_d_tde", "RQA_tde", "L_tde",
    "tau_pec", "ts_pec", "optimal_d_pec", "RQA_pec", "L_pec",
    "tau_MCDTS", "ts_MCDTS", "optimal_d_mcdts", "RQA_mcdts", "L_mcdts"]

for i = 1:length(varnames)
    writestr = "results_Lorenz96_N_$(N)_3d_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
