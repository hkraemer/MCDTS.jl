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

    # pick one time series
    t_idx = 2

    # init Lorenz96
    Random.seed!(1234)
    lo96 = Systems.lorenz96(N; F = 3.5)

    params = tuple(N,dt,total,ε,dmax,lmin,trials,taus,t_idx,L_threshold)
end

@time begin
# loop over different F's
results = @distributed (vcat) for i in eachindex(Fs)

    F = Fs[i]
    set_parameter!(lo96, 1, F)
    data = trajectory(lo96, total*dt; dt = dt, Ttr = 2500 * dt)

    theiler = 0
    for i = 1:size(data,2)
        w = estimate_delay(data[:,i], "mi_min")
        if w > theiler
            theiler = w
        end
    end
    R = RecurrenceMatrix(data, ε; fixedrate = true)
    RQA = rqa(R; theiler = theiler, lmin = lmin)

    # Output
    tuple(RQA)

end

end

writedlm("results_Lorenz96_N_$(N)_ref_params.csv", params)
writedlm("results_Lorenz96_N_$(N)_ref_Fs.csv", Fs)

varnames = ["RQA_ref"]

for i = 1:length(varnames)
    writestr = "results_Lorenz96_N_$(N)_ref_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
