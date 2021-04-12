using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using ChaosTools
using DelimitedFiles
using Random

## We predict the Lorenz63-system based on different state space reconstruction methods
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

## Predictions based on embedding

taus = 0:100
data_sample = Dataset(hcat(x1,y1))
data_sample_n = Dataset(hcat(x1_n,y1_n))


# mcdts
Tw = 1  # time horizon
KK = 1 # considered nearest neighbors
trials = 120
max_depth = 15
tree = MCDTS.mc_delay(data_sample,w1,(L)->(MCDTS.softmaxL(L,β=2.)),
    taus, trials; max_depth = max_depth, PRED = true, verbose = true, KNN = KK,
    threshold = 5e-6)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_PRED_multi = best_node.τs
ts_mcdts_PRED_multi = best_node.ts
Y_mcdts_PRED_multi = genembed(data_sample, τ_mcdts_PRED_multi .* (-1), ts_mcdts_PRED_multi)

# Save data
writedlm("Y_mcdts_PRED_multi.csv", Y_mcdts_PRED_multi)
writedlm("tau_mcdts_PRED_multi.csv", τ_mcdts_PRED_multi)
writedlm("ts_mcdts_PRED_multi.csv", ts_mcdts_PRED_multi)
