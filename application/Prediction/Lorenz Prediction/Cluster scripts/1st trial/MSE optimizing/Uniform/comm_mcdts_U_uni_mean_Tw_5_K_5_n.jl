using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using ChaosTools
using DelimitedFiles
using Random

## We predict the Lorenz63-system based on different state space reconstruction methods
Random.seed!(1234)
lo = Systems.lorenz()
tr = trajectory(lo, 1000; dt = 0.01, Ttr = 100) # results 3

# noise level
#σ = .1  # results 2
σ = .05 # results 3

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

data_sample = Dataset(hcat(x1,y1))
data_sample_n = Dataset(hcat(x1_n,y1_n))


# mcdts

# params
Tw = 5  # time horizon
KK = 5 # considered nearest neighbors
PRED_L = false
PRED_mean = true
trials = 80
max_depth = 25
taus = 0:25

@time tree = MCDTS.mc_delay(Dataset(x1_n),w1_n,(L)->(MCDTS.softmaxL(L,β=2.)),
    taus, trials; max_depth = max_depth, PRED = true, verbose = true, KNN = KK,
    Tw = Tw, threshold = 5e-6, PRED_L = PRED_L, PRED_mean = PRED_mean)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_PRED = best_node.τs
Y_mcdts_PRED = MCDTS.genembed_for_prediction(x1_n, τ_mcdts_PRED)

# Save data
if PRED_mean
    if PRED_L
        writedlm("Y_mcdts_PRED_L_K_$(KK)_Tw_$(Tw)_mean_uni_n.csv", Y_mcdts_PRED)
        writedlm("tau_mcdts_PRED_L_K_$(KK)_Tw_$(Tw)_mean_uni_n.csv", τ_mcdts_PRED)
    else
        writedlm("Y_mcdts_PRED_U_K_$(KK)_Tw_$(Tw)_mean_uni_n.csv", Y_mcdts_PRED)
        writedlm("tau_mcdts_PRED_U_K_$(KK)_Tw_$(Tw)_mean_uni_n.csv", τ_mcdts_PRED)
    end
else
    if PRED_L
        writedlm("Y_mcdts_PRED_L_K_$(KK)_Tw_$(Tw)_x_uni_n.csv", Y_mcdts_PRED)
        writedlm("tau_mcdts_PRED_L_K_$(KK)_Tw_$(Tw)_x_uni_n.csv", τ_mcdts_PRED)
    else
        writedlm("Y_mcdts_PRED_U_K_$(KK)_Tw_$(Tw)_x_uni_n.csv", Y_mcdts_PRED)
        writedlm("tau_mcdts_PRED_U_K_$(KK)_Tw_$(Tw)_x_uni_n.csv", τ_mcdts_PRED)
    end
end
