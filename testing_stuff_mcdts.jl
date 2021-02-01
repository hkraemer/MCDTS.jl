using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using DelimitedFiles

using PyPlot
pygui(true)

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

# pick one time series
t_idx = 2
#t_idx = [2,4,7]

# init Lorenz96
u0 = [0.590; 0.766; 0.566; 0.460; 0.794; 0.854; 0.200; 0.298]
lo96 = Systems.lorenz96(N, u0; F = 3.5)

i = 370
F = Fs[i]
F = 4.8
set_parameter!(lo96, 1, F)
data = trajectory(lo96, total*dt; dt = dt, Ttr = 2500 * dt)
data_sample = data[:,t_idx]


figure()
plot(data_sample)

# Traditional time delay embedding
𝒟, τ_tde, _ = optimal_traditional_de(data_sample, "fnn"; dmax = dmax)
optimal_d_tde = size(𝒟, 2)
R = RecurrenceMatrix(𝒟, ε; fixedrate = true)
RQA = rqa(R; theiler = τ_tde, lmin = lmin)
RQA_tde = hcat(RQA...)
τ_tdes = [(i-1)*τ_tde for i = 1:optimal_d_tde]
L_tde = MCDTS.compute_delta_L(data_sample, τ_tdes, taus[end]; w = τ_tde)

# PECUZAL
theiler = τ_tde
@time 𝒟_pec, τ_pec, ts_pec, Ls_pec , _ = DelayEmbeddings.pecuzal_embedding(data_sample; τs = taus , w = theiler, econ = true, threshold = 0.05)
optimal_d_pec = size(𝒟_pec,2)
R = RecurrenceMatrix(𝒟_pec, ε; fixedrate = true)
RQA = rqa(R; theiler = theiler, lmin = lmin)
RQA_pec = hcat(RQA...)
L_pec = sum(Ls_pec)

# MCDTS
@time tree = MCDTS.mc_delay(Dataset(data_sample), theiler, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:2:taus[end], threshold = 0.05)
best_node = MCDTS.best_embedding(tree)
𝒟_mcdts = genembed(data_sample, best_node.τs, best_node.ts)
optimal_d_mcdts = size(𝒟_mcdts,2)
R = RecurrenceMatrix(𝒟_mcdts, ε; fixedrate = true)
RQA = rqa(R; theiler = theiler, lmin = lmin)
RQA_mcdts = hcat(RQA...)
L_mcdts = best_node.L



pwd()
λs = readdlm("./application/artificial data/Lorenz96/Lyapunov spectrum/results_Lorenz96_N_40_lyapunovs.csv", ',', Any, '[')
Fs = readdlm("./application/artificial data/Lorenz96/Lyapunov spectrum/results_Lorenz96_N_40_lyapunovs_Fs.csv")
λs = λs[:,1:end-1]

pos_Lyap_idx = λs[:,1] .> 10^-3

l_width_vert = 0.1
figure(figsize=(20,10))
plot(Fs, λs)
ylims1 = axis1.get_ylim()
vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
title("Lyaps")
ylabel("embedding dimension")
grid()
