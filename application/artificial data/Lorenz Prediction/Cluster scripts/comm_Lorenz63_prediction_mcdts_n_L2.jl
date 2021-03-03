using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using ChaosTools
using DelimitedFiles
using Random

## We predict the Lorenz63-system based on different state space reconstruction methods
Random.seed!(1234)
lo = Systems.lorenz()
tr = trajectory(lo, 500; dt = 0.01, Ttr = 100)

# noise level
σ = .1

# normalize time series
tr = regularize(tr)

Random.seed!(1234)

T_steps = 900 # 8*lyap_time
x1 = tr[1:end-T_steps,1]
x2 = tr[end-T_steps+1:end,1]
y1 = tr[1:end-T_steps,2]
y2 = tr[end-T_steps+1:end,2]

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
trials = 120

tree = MCDTS.mc_delay(data_sample_n, w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end], verbose=true)
best_node = MCDTS.best_embedding(tree)
τ_mcdts2_n = best_node.τs
ts_mcdts2_n = best_node.ts
Y_mcdts2_n = MCDTS.genembed_for_prediction(data_sample_n, τ_mcdts2_n, ts_mcdts2_n)


# Save data
writedlm("Y_mcdts2_n.csv", Y_mcdts2_n)
writedlm("tau_mcdts2_n.csv", τ_mcdts2_n)
writedlm("ts_mcdts2_n.csv", ts_mcdts2_n)
