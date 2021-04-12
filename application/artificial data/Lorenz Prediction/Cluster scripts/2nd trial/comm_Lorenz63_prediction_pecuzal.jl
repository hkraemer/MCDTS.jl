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


# pecuzal
𝒟, τ_pec2, ts_pec2, L, _ = pecuzal_embedding(data_sample; τs = taus, w = w1)
Y_pec2 = genembed(data_sample, τ_pec2 .*(-1), ts_pec2)

𝒟, τ_pec2_n, ts_pec2_n, L, _ = pecuzal_embedding(data_sample_n; τs = taus, w = w1_n)
Y_pec2_n = genembed(data_sample_n, τ_pec2_n .*(-1), ts_pec2_n)


# Save data
writedlm("Y_pec2.csv", Y_pec2)
writedlm("Y_pec2_n.csv", Y_pec2_n)
writedlm("tau_pec2.csv", τ_pec2)
writedlm("tau_pec2_n.csv", τ_pec2_n)
writedlm("ts_pec2.csv", ts_pec2)
writedlm("ts_pec2_n.csv", ts_pec2_n)
