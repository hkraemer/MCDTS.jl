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
    using DelimitedFiles
    using DynamicalSystems

    # Parameters data:
    N = 40 # number of oscillators
    #Fs = 3.5:0.004:5 # parameter spectrum
    Fs = 3.5:0.004:3.54 # parameter spectrum
    # init Lorenz96
    lo96 = Systems.lorenz96(N; F = 3.5)
end

@time begin
# loop over different F's
results = @distributed (vcat) for i in eachindex(Fs)
    F = Fs[i]
    set_parameter!(lo96, 1, F)
    λs = lyapunovs(lo96, 100000; Ttr = 10000)
    # Output
    tuple(λs)
end
end

writedlm("results_Lorenz96_N_$(N)_lyapunovs_Fs.csv", Fs)
writedlm("results_Lorenz96_N_$(N)_lyapunovs.csv", results)
