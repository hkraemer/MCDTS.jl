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
    N = 8 # number of oscillators
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
    t_idx = [2,4,7]

    # init Lorenz96
    u0 = [0.590; 0.766; 0.566; 0.460; 0.794; 0.854; 0.200; 0.298]
    lo96 = Systems.lorenz96(N, u0; F = 3.5)

    params = tuple(N,dt,total,ε,dmax,lmin,trials,taus,t_idx,L_threshold)
end

@time begin
# loop over different ic's
results = @distributed (vcat) for i in eachindex(Fs)

    F = Fs[i]
    set_parameter!(lo96, 1, F)
    data = trajectory(lo96, total*dt; dt = dt, Ttr = 2500*dt)
    data_sample = data[:,t_idx]

    # PECUZAL
    w1 = DelayEmbeddings.estimate_delay(data_sample[:,1], "mi_min")
    w2 = DelayEmbeddings.estimate_delay(data_sample[:,2], "mi_min")
    w3 = DelayEmbeddings.estimate_delay(data_sample[:,3], "mi_min")
    theiler = maximum([w1,w2,w3])
    𝒟_pec, τ_pec, ts_pec, Ls_pec , _ = DelayEmbeddings.pecuzal_embedding(data_sample; τs = taus , w = theiler, econ = true, L_threshold = L_threshold)
    optimal_d_pec = size(𝒟_pec,2)
    R = RecurrenceMatrix(𝒟_pec, ε; fixedrate = true)
    RQA = rqa(R; theiler = theiler, lmin = lmin)
    RQA_pec = hcat(RQA...)
    L_pec = sum(Ls_pec)

    # Output
    tuple(τ_pec, ts_pec, optimal_d_pec, RQA_pec, L_pec)

end

end

writedlm("results_Lorenz96_N_$(N)_3d_params.csv", params)
writedlm("results_Lorenz96_N_$(N)_3d_Fs.csv", Fs)

varnames = ["tau_pec", "ts_pec", "optimal_d_pec", "RQA_pec", "L_pec"]

for i = 1:length(varnames)
    writestr = "results_Lorenz96_N_$(N)_3d_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
