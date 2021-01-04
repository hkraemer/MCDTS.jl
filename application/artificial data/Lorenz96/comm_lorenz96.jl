using DynamicalSystems
using DelayEmbeddings
using MCDTS
using RecurrenceAnalysis
using Statistics

## We apply the MCDTS-approach to the Lorenz system and compare the results of
# the reconstruction to the ones gained from PECUZAL and standard TDE. Here we
# compute the Lyapunov spectra for a bunch of parameter-settings

# Parameters data:
N = 8 # number of oscillators
Fs = 2.5:0.1:2.5 # parameter spectrum
dt = 0.01 # sampling time
total = 3000  # time series length

# Parameters analysis:
ε = 0.05  # recurrence threshold
dmax = 10   # maximum dimension for traditional tde
lmin = 2   # minimum line length for RQA
trials = 50 # trials for MCDTS
taus = 0:100 # possible delays

# randomly pick 3 time series
t_idx = rand(1:N, 3)

# init Lorenz96
lo96 = Systems.lorenz96(N; F = 2.5)
# loop over different ic's
for (i, F) in enumerate(Fs)
    set_parameter!(lo96, 1, F)
    data = trajectory(lo96, total*dt; dt = 0.01, Ttr = 40)
    data_sample = data[:,t_idx]

    # for ts perform classic TDE
    τ_tde = zeros(Int,3)
    optimal_d_tde = zeros(Int,3)
    RQA_tde = zeros(15,3)
    L_tde = zeros(3)
    for i = 1:3
        𝒟, τ_tde[i], _ = optimal_traditional_de(data_sample[:,i], "fnn"; dmax)
        optimal_d_tde[i] = size(𝒟, 2)
        R = RecurrenceMatrix(𝒟, ε; fixedrate = true)
        RQA = rqa(R; theiler = τ_tde[i], lmin = lmin)
        RQA_tde[:,i] = hcat(RQA...)
        L_tde[i] = uzal_cost(regularize(𝒟); w = τ_tde[i], samplesize=1)
    end

    # PECUZAL
    theiler = Int(floor(mean(τ_tde)))
    𝒟_pec, τ_pec, ts_pec, Ls_pec , _ = pecuzal_embedding(data_sample; τs = taus , w = theiler)
    optimal_d_pec = size(𝒟_pec,2)
    R = RecurrenceMatrix(𝒟_pec, ε; fixedrate = true)
    RQA = rqa(R; theiler, lmin)
    RQA_pec = hcat(RQA...)
    L_pec = minimum(Ls_pec)

    # MCDTS
    tree = MCDTS.mc_delay(data_sample, theiler, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials)
    best_node = MCDTS.best_embedding(tree)
    𝒟_mcdts = genembed(data_sample, best_node.τs, best_node.ts)
    optimal_d_mcdts = size(𝒟_mcdts,2)
    R = RecurrenceMatrix(𝒟_mcdts, ε; fixedrate = true)
    RQA = rqa(R; theiler, lmin)
    RQA_mcdts = hcat(RQA...)
    L_mcdts = best_node.L

    # Output
    tuple(τ_tde, optimal_d_tde, RQA_tde, L_tde,
        τ_pec, ts_pec, optimal_d_pec, RQA_pec, L_pec,
        best_node.τs, best_node.ts, optimal_d_mcdts, RQA_mcdts, L_mcdts)
end
