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
    Fs = 3.7:0.002:4 # parameter spectrum
    dt = 0.1 # sampling time
    total = 5000  # time series length

    # Parameters analysis:
    ε = 0.05  # recurrence threshold
    dmax = 10   # maximum dimension for traditional tde
    lmin = 2   # minimum line length for RQA
    trials = 100 # trials for MCDTS
    taus = 0:100 # possible delays
    L_threshold = 0 # threshold for minimum tolerable ΔL decrease per embedding cycle

    # pick one time series
    t_idx = 2

    # init Lorenz96
    u0 = [0.590; 0.766; 0.566; 0.460; 0.794; 0.854; 0.200; 0.298]
    lo96 = Systems.lorenz96(N, u0; F = 3.5)

    params = tuple(N,dt,total,ε,dmax,lmin,trials,taus,t_idx,L_threshold)
end

@time begin
# loop over different F's
results = @distributed (vcat) for i in eachindex(Fs)

    F = Fs[i]
    set_parameter!(lo96, 1, F)
    data = trajectory(lo96, total*dt; dt = dt, Ttr = 2500 * dt)
    data_sample = data[:,t_idx]

    # Traditional time delay embedding
    # Cao's method
    𝒟, τ_tde1, E = optimal_traditional_de(data_sample, "afnn"; dmax = dmax)
    optimal_d_tde1 = size(𝒟, 2)
    R1 = RecurrenceMatrix(𝒟, ε; fixedrate = true)
    RQA = rqa(R1; theiler = τ_tde1, lmin = lmin)
    RQA_tde1 = hcat(RQA...)
    τ_tdes = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
    L_tde1 = MCDTS.compute_delta_L(data_sample, τ_tdes, taus[end]; w = τ_tde1, tws = 2:2:taus[end])
    FNN_tde1 = E[optimal_d_tde1]

    # kennel's method
    𝒟, τ_tde2, E = optimal_traditional_de(data_sample, "fnn"; dmax = dmax)
    optimal_d_tde2 = size(𝒟, 2)
    R2 = RecurrenceMatrix(𝒟, ε; fixedrate = true)
    RQA = rqa(R2; theiler = τ_tde2, lmin = lmin)
    RQA_tde2 = hcat(RQA...)
    τ_tdes = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
    L_tde2 = MCDTS.compute_delta_L(data_sample, τ_tdes, taus[end]; w = τ_tde2, tws = 2:2:taus[end])
    FNN_tde2 = E[optimal_d_tde2]

    # hegger's method
    𝒟, τ_tde3, E = optimal_traditional_de(data_sample, "ifnn"; dmax = dmax)
    optimal_d_tde3 = size(𝒟, 2)
    R3 = RecurrenceMatrix(𝒟, ε; fixedrate = true)
    RQA = rqa(R3; theiler = τ_tde3, lmin = lmin)
    RQA_tde3 = hcat(RQA...)
    τ_tdes = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
    L_tde3 = MCDTS.compute_delta_L(data_sample, τ_tdes, taus[end]; w = τ_tde3, tws = 2:2:taus[end])
    FNN_tde3 = E[optimal_d_tde3]

    # Reference
    R_ref = RecurrenceMatrix(data, ε; fixedrate = true)
    RQA_ref = rqa(R_ref; theiler = τ_tde3, lmin = lmin)

    # compute RP-accordance
    RP_frac_tde1 = MCDTS.jrp_rr_frac(RecurrenceMatrix(data[1:size(R1,1),:], ε; fixedrate = true),R1)
    RP_frac_tde2 = MCDTS.jrp_rr_frac(RecurrenceMatrix(data[1:size(R2,1),:], ε; fixedrate = true),R2)
    RP_frac_tde3 = MCDTS.jrp_rr_frac(RecurrenceMatrix(data[1:size(R3,1),:], ε; fixedrate = true),R3)

    # Output
    tuple(τ_tde1, optimal_d_tde1, RQA_tde1, L_tde1, FNN_tde1,
        τ_tde2, optimal_d_tde2, RQA_tde2, L_tde2, FNN_tde2,
        τ_tde3, optimal_d_tde3, RQA_tde3, L_tde3, FNN_tde3,
        RQA_ref, RP_frac_tde1, RP_frac_tde2, RP_frac_tde3)

end

end

writedlm("results_Lorenz96_N_$(N)_final_params.csv", params)
writedlm("results_Lorenz96_N_$(N)_final_Fs.csv", Fs)

varnames = ["tau_tde1", "optimal_d_tde1", "RQA_tde1", "L_tde1", "FNN_tde1",
    "tau_tde2", "optimal_d_tde2", "RQA_tde2", "L_tde2", "FNN_tde2",
    "tau_tde3", "optimal_d_tde3", "RQA_tde3", "L_tde3", "FNN_tde3",
    "RQA_ref", "RP_frac_tde1", "RP_frac_tde2", "RP_frac_tde3"]

for i = 1:length(varnames)
    writestr = "results_Lorenz96_N_$(N)_final_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
