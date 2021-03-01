## We apply the MCDTS-approach to the Lorenz system and compare the results of
# the reconstruction to the ones gained from PECUZAL and standard TDE. Here we
# compute the Lyapunov spectra for a bunch of parameter-settings
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
trials = 150 # trials for MCDTS
taus = 0:100 # possible delays
L_threshold = 0 # threshold for minimum tolerable ΔL decrease per embedding cycle

# pick one time series
t_idx = [2,4,7]

# init Lorenz96
u0 = [0.590; 0.766; 0.566; 0.460; 0.794; 0.854; 0.200; 0.298]
lo96 = Systems.lorenz96(N, u0; F = 3.5)

params = tuple(N,dt,total,ε,dmax,lmin,trials,taus,t_idx,L_threshold)

tau_MCDTS = []
ts_MCDTS = []
optimal_d_mcdtss = zeros(length(Fs))
RQA_mcdtss = []
L_mcdtss = zeros(length(Fs))
RP_frac_mcdtss = zeros(length(Fs))

@time begin
# loop over different F's
for i in eachindex(Fs)

    F = Fs[i]
    set_parameter!(lo96, 1, F)
    data = trajectory(lo96, total*dt; dt = dt, Ttr = 2500*dt)
    data_sample = data[:,t_idx]

    # theiler window estimation
    w1 = DelayEmbeddings.estimate_delay(data_sample[:,1], "mi_min")
    w2 = DelayEmbeddings.estimate_delay(data_sample[:,2], "mi_min")
    w3 = DelayEmbeddings.estimate_delay(data_sample[:,3], "mi_min")
    theiler = maximum([w1,w2,w3])

    # MCDTS
    tree = MCDTS.mc_delay(data_sample, theiler, (L)->(MCDTS.softmaxL(L,β=2.)),
            taus, trials; tws = 2:2:taus[end], threshold = L_threshold, max_depth = 30, verbose=true)
    best_node = MCDTS.best_embedding(tree)
    𝒟_mcdts = genembed(data_sample, best_node.τs, best_node.ts)
    optimal_d_mcdts = size(𝒟_mcdts,2)
    R = RecurrenceMatrix(𝒟_mcdts, ε; fixedrate = true)
    RQA = rqa(R; theiler = theiler, lmin = lmin)
    RQA_mcdts = hcat(RQA...)
    L_mcdts = best_node.L

    R_ref = RecurrenceMatrix(data[1:length(𝒟_mcdts),:], ε; fixedrate = true)

    RP_frac_mcdts = MCDTS.jrp_rr_frac(R_ref,R)

    # Output
    push!(tau_MCDTS,best_node.τs)
    push!(ts_MCDTS,best_node.ts)
    optimal_d_mcdtss[i] = optimal_d_mcdts
    push!(RQA_mcdtss,RQA_mcdts)
    L_mcdtss[i] = L_mcdts
    RP_frac_mcdtss[i] = RP_frac_mcdts
end

writedlm("results_Lorenz96_N_$(N)_mcdts_3d_params.csv", params)
writedlm("results_Lorenz96_N_$(N)_mcdts_3d_Fs.csv", Fs)

writedlm("results_Lorenz96_N_$(N)_mcdts_3d_tau_MCDTS.csv", tau_MCDTS)
writedlm("results_Lorenz96_N_$(N)_mcdts_3d_ts_MCDTS.csv", ts_MCDTS)
writedlm("results_Lorenz96_N_$(N)_mcdts_3d_optimal_d_mcdts.csv", optimal_d_mcdtss)
writedlm("results_Lorenz96_N_$(N)_mcdts_3d_RQA_mcdts.csv", RQA_mcdtss)
writedlm("results_Lorenz96_N_$(N)_mcdts_3d_L_mcdts.csv", L_mcdtss)
writedlm("results_Lorenz96_N_$(N)_mcdts_3d_RP_frac_mcdts.csv", RP_frac_mcdtss)
