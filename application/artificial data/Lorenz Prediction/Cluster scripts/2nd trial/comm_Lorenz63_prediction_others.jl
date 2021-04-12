using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using DelimitedFiles
using ChaosTools
using Random

## We predict the Lorenz63-system based on different state space reconstruction methods

# Here we compute the different reconstructions based on the different reconstruction
# methods. The computationally more demanding reconstructions have been carried out
# on the cluster (scripts in folder /Cluster scripts)

Random.seed!(234)
lo = Systems.lorenz()
tr = trajectory(lo, 1000; dt = 0.01, Ttr = 500)

# noise level
σ = .05

# normalize time series
tr = regularize(tr)

Random.seed!(234)

T_steps = 1700 # 12*lyap_time

x = tr[:,1]
y = tr[:,2]
x_n = tr[:,1] .+ σ*randn(length(tr))
y_n = tr[:,2] .+ σ*randn(length(tr))

x1 = x[1:end-T_steps]
x2 = x[end-T_steps+1:end]
y1 = y[1:end-T_steps]
y2 = y[end-T_steps+1:end]
x1_n = x_n[1:end-T_steps]
x2_n = x_n[end-T_steps+1:end]
y1_n = y_n[1:end-T_steps]
y2_n = y_n[end-T_steps+1:end]

w1 = DelayEmbeddings.estimate_delay(x1, "mi_min")
w1_n = DelayEmbeddings.estimate_delay(x1_n, "mi_min")

# Lyapunov exponent and time
λ = ChaosTools.lyapunov(lo, 100000, dt=0.01; Ttr=1000)
lyap_time = Int(floor((1/λ) / 0.01))

data_sample = Dataset(hcat(x1,y1))
data_sample_n = Dataset(hcat(x1_n,y1_n))

## Predictions based on embedding
dmax = 12
# cao
𝒟, τ_tde1, _ = optimal_traditional_de(x1, "afnn"; dmax = dmax, w = w1)
optimal_d_tde1 = size(𝒟, 2)
τ_cao = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
Y_cao = genembed(x1, τ_cao .* (-1))

𝒟, τ_tde1, _ = optimal_traditional_de(x1_n, "afnn"; dmax = dmax, w = w1_n)
optimal_d_tde1 = size(𝒟, 2)
τ_cao_n = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
Y_cao_n = genembed(x1_n, τ_cao_n .* (-1))

# kennel's method
𝒟, τ_tde2, _ = optimal_traditional_de(x1, "fnn"; dmax = dmax, w = w1)
optimal_d_tde2 = size(𝒟, 2)
τ_kennel = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
Y_kennel = genembed(x1, τ_kennel .* (-1))

𝒟, τ_tde2, _ = optimal_traditional_de(x1_n, "fnn"; dmax = dmax, w = w1_n)
optimal_d_tde2 = size(𝒟, 2)
τ_kennel_n = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
Y_kennel_n = genembed(x1_n, τ_kennel_n .* (-1))

# hegger's method
𝒟, τ_tde3, _ = optimal_traditional_de(x1, "ifnn"; dmax = dmax, w = w1)
optimal_d_tde3 = size(𝒟, 2)
τ_hegger = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
Y_hegger = genembed(x1, τ_hegger .* (-1))

𝒟, τ_tde3, _ = optimal_traditional_de(x1_n, "ifnn"; dmax = dmax, w = w1_n)
optimal_d_tde3 = size(𝒟, 2)
τ_hegger_n = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
Y_hegger_n = genembed(x1_n, τ_hegger_n .* (-1))

# pecuzal
taus = 0:100
𝒟, τ_pec, _, L, _ = pecuzal_embedding(x1; τs = taus, w = w1)
optimal_d_tde4 = size(𝒟, 2)
Y_pec = genembed(x1, τ_pec .* (-1))

𝒟, τ_pec_n, _, L, _ = pecuzal_embedding(x1_n; τs = taus, w = w1_n)
optimal_d_tde4 = size(𝒟, 2)
Y_pec_n = genembed(x1_n, τ_pec_n .* (-1))

# mcdts
trials = 80
tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end])
best_node = MCDTS.best_embedding(tree)
τ_mcdts = best_node.τs
Y_mcdts = genembed(x1, τ_mcdts .* (-1))

# tree = MCDTS.mc_delay(Dataset(x1_n), w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end])
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_n = best_node.τs
# Y_mcdts_n = MCDTS.genembed_for_prediction(x1_n, τ_mcdts_n)

# pecuzal
# 𝒟, τ_pec2, ts_pec2, L, _ = pecuzal_embedding(data_sample; τs = taus, w = w1)
# Y_pec2 = MCDTS.genembed_for_prediction(data_sample, τ_pec2, ts_pec2)
#
# 𝒟, τ_pec2_n, ts_pec2_n, L, _ = pecuzal_embedding(data_sample_n; τs = taus, w = w1_n)
# Y_pec2_n = MCDTS.genembed_for_prediction(data_sample_n, τ_pec2_n, ts_pec2_n)

# mcdts
# trials = 120
# tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end])
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts2 = best_node.τs
# ts_mcdts2 = best_node.ts
# Y_mcdts2 = MCDTS.genembed_for_prediction(data_sample, τ_mcdts2, ts_mcdts2)


# tree = MCDTS.mc_delay(data_sample_n, w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end], verbose=true)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts2_n = best_node.τs
# ts_mcdts2_n = best_node.ts
# Y_mcdts2_n = MCDTS.genembed_for_prediction(data_sample_n, τ_mcdts2_n, ts_mcdts2_n)
#
# # mcdts FNN
# trials = 80
# tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN=true, threshold = 0.01)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_fnn = best_node.τs
# Y_mcdts_fnn = MCDTS.genembed_for_prediction(x1, τ_mcdts_fnn)
#
# tree = MCDTS.mc_delay(Dataset(x1_n), w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN=true, threshold = 0.05)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_fnn_n = best_node.τs
# Y_mcdts_fnn_n = MCDTS.genembed_for_prediction(x1_n, τ_mcdts_fnn_n)
#
# trials = 120
# tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN=true, threshold = 0.01)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_fnn2 = best_node.τs
# ts_mcdts_fnn2 = best_node.ts
# Y_mcdts_fnn2 = MCDTS.genembed_for_prediction(data_sample, τ_mcdts_fnn2, ts_mcdts_fnn2)
#
# tree = MCDTS.mc_delay(data_sample_n, w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN=true, threshold = 0.05)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_fnn2_n = best_node.τs
# ts_mcdts_fnn2_n = best_node.ts
# Y_mcdts_fnn2_n = MCDTS.genembed_for_prediction(data_sample_n, τ_mcdts_fnn2_n, ts_mcdts_fnn2_n)


writedlm("x1.csv", x1)
writedlm("x2.csv", x2)
writedlm("y1.csv", y1)
writedlm("y2.csv", y2)
writedlm("x1_n.csv", x1_n)
writedlm("x2_n.csv", x2_n)
writedlm("y1_n.csv", y1_n)
writedlm("y2_n.csv", y2_n)
writedlm("data_sample.csv", data_sample)
writedlm("data_sample_n.csv", data_sample_n)
writedlm("tr.csv", tr)
writedlm("Y_cao.csv", Y_cao)
writedlm("Y_cao_n.csv", Y_cao_n)
writedlm("tau_cao.csv", τ_cao)
writedlm("tau_cao_n.csv", τ_cao_n)
writedlm("Y_kennel.csv", Y_kennel)
writedlm("Y_kennel_n.csv", Y_kennel_n)
writedlm("tau_kennel.csv", τ_kennel)
writedlm("tau_kennel_n.csv", τ_kennel_n)
writedlm("Y_hegger.csv", Y_hegger)
writedlm("Y_hegger_n.csv", Y_hegger_n)
writedlm("tau_hegger.csv", τ_hegger)
writedlm("tau_hegger_n.csv", τ_hegger_n)
writedlm("Y_pec.csv", Y_pec)
writedlm("Y_pec_n.csv", Y_pec_n)
writedlm("tau_pec.csv", τ_pec)
writedlm("tau_pec_n.csv", τ_pec_n)
writedlm("Y_mcdts.csv", Y_mcdts)
writedlm("Y_mcdts_n.csv", Y_mcdts_n)
writedlm("tau_mcdts.csv", τ_mcdts)
writedlm("tau_mcdts_n.csv", τ_mcdts_n)
