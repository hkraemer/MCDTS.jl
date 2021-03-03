using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using DelimitedFiles
using ChaosTools
using Random

using PyPlot
pygui(true)

## We predict the Lorenz63-system based on different state space reconstruction methods

lo = Systems.lorenz()
tr = trajectory(lo, 200; dt = 0.01, Ttr = 100)

# noise level
σ = .1

# Lyapunov exponent and time
λ = ChaosTools.lyapunov(lo, 100000, dt=0.01; Ttr=1000)
lyap_time = Int(floor((1/λ) / 0.01))

# normalize time series
tr = regularize(tr)

Random.seed!(1234)

T_steps = 8*lyap_time
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

figure()
subplot(211)
plot(x)
grid()
subplot(212)
plot(x_n)
grid()

## Predictions based on embedding

dmax = 12
# cao
𝒟, τ_tde1, _ = optimal_traditional_de(x1, "afnn"; dmax = dmax, w = w1)
optimal_d_tde1 = size(𝒟, 2)
τ_cao = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
Y_cao = MCDTS.genembed_for_prediction(x1, τ_cao)

𝒟, τ_tde1, _ = optimal_traditional_de(x1_n, "afnn"; dmax = dmax, w = w1_n)
optimal_d_tde1 = size(𝒟, 2)
τ_cao_n = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
Y_cao_n = MCDTS.genembed_for_prediction(x1_n, τ_cao_n)

# kennel's method
𝒟, τ_tde2, _ = optimal_traditional_de(x1, "fnn"; dmax = dmax, w = w1)
optimal_d_tde2 = size(𝒟, 2)
τ_kennel = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
Y_kennel = MCDTS.genembed_for_prediction(x1, τ_kennel)

𝒟, τ_tde2, _ = optimal_traditional_de(x1_n, "fnn"; dmax = dmax, w = w1_n)
optimal_d_tde2 = size(𝒟, 2)
τ_kennel_n = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
Y_kennel_n = MCDTS.genembed_for_prediction(x1_n, τ_kennel_n)

# hegger's method
𝒟, τ_tde3, _ = optimal_traditional_de(x1, "ifnn"; dmax = dmax, w = w1)
optimal_d_tde3 = size(𝒟, 2)
τ_hegger = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
Y_hegger = MCDTS.genembed_for_prediction(x1, τ_hegger)

𝒟, τ_tde3, _ = optimal_traditional_de(x1_n, "ifnn"; dmax = dmax, w = w1_n)
optimal_d_tde3 = size(𝒟, 2)
τ_hegger_n = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
Y_hegger_n = MCDTS.genembed_for_prediction(x1_n, τ_hegger_n)

# pecuzal
taus = 0:100
𝒟, τ_pec, _, L, _ = pecuzal_embedding(x1; τs = taus, w = w1)
optimal_d_tde4 = size(𝒟, 2)
Y_pec = MCDTS.genembed_for_prediction(x1, τ_pec)

𝒟, τ_pec_n, _, L, _ = pecuzal_embedding(x1_n; τs = taus, w = w1_n)
optimal_d_tde4 = size(𝒟, 2)
Y_pec_n = MCDTS.genembed_for_prediction(x1_n, τ_pec_n)

# mcdts
trials = 80
tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end])
best_node = MCDTS.best_embedding(tree)
τ_mcdts = best_node.τs
Y_mcdts = MCDTS.genembed_for_prediction(x1, τ_mcdts)

tree = MCDTS.mc_delay(Dataset(x1_n), w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end])
best_node = MCDTS.best_embedding(tree)
τ_mcdts_n = best_node.τs
Y_mcdts_n = MCDTS.genembed_for_prediction(x1_n, τ_mcdts_n)

data_sample = Dataset(hcat(x1,y1))
data_sample_n = Dataset(hcat(x1_n,y1_n))

# pecuzal
𝒟, τ_pec2, ts_pec2, L, _ = pecuzal_embedding(data_sample; τs = taus, w = w1)
Y_pec2 = MCDTS.genembed_for_prediction(data_sample, τ_pec2, ts_pec2)

𝒟, τ_pec2_n, ts_pec2_n, L, _ = pecuzal_embedding(data_sample_n; τs = taus, w = w1_n)
Y_pec2_n = MCDTS.genembed_for_prediction(data_sample_n, τ_pec2_n, ts_pec2_n)

# mcdts
trials = 120
tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end])
best_node = MCDTS.best_embedding(tree)
τ_mcdts2 = best_node.τs
ts_mcdts2 = best_node.ts
Y_mcdts2 = MCDTS.genembed_for_prediction(data_sample, τ_mcdts2, ts_mcdts2)

tree = MCDTS.mc_delay(data_sample_n, w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end], verbose=true)
best_node = MCDTS.best_embedding(tree)
τ_mcdts2_n = best_node.τs
ts_mcdts2_n = best_node.ts
Y_mcdts2_n = MCDTS.genembed_for_prediction(data_sample_n, τ_mcdts2_n, ts_mcdts2_n)

println("1")
# mcdts FNN
trials = 80
tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN=true, threshold = 0.01)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_fnn = best_node.τs
Y_mcdts_fnn = MCDTS.genembed_for_prediction(x1, τ_mcdts_fnn)
println("2")
tree = MCDTS.mc_delay(Dataset(x1_n), w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN=true, threshold = 0.05)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_fnn_n = best_node.τs
Y_mcdts_fnn_n = MCDTS.genembed_for_prediction(x1_n, τ_mcdts_fnn_n)
println("3")
trials = 120
tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN=true, threshold = 0.01)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_fnn2 = best_node.τs
ts_mcdts_fnn2 = best_node.ts
Y_mcdts_fnn2 = MCDTS.genembed_for_prediction(data_sample, τ_mcdts_fnn2, ts_mcdts_fnn2)
println("4")
tree = MCDTS.mc_delay(data_sample_n, w1_n, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN=true, threshold = 0.05)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_fnn2_n = best_node.τs
ts_mcdts_fnn2_n = best_node.ts
Y_mcdts_fnn2_n = MCDTS.genembed_for_prediction(data_sample_n, τ_mcdts_fnn2_n, ts_mcdts_fnn2_n)

# Save data
writedlm("./application/artificial data/Lorenz Prediction/Results 2/x1.csv", x1)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/x2.csv", x2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/y1.csv", y1)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/y2.csv", y2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/x1_n.csv", x1_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/x2_n.csv", x2_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/y1_n.csv", y1_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/y2_n.csv", y2_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/data_sample.csv", data_sample)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/data_sample_n.csv", data_sample_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tr.csv", tr)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_cao.csv", Y_cao)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_cao_n.csv", Y_cao_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_cao.csv", τ_cao)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_cao_n.csv", τ_cao_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_kennel.csv", Y_kennel)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_kennel_n.csv", Y_kennel_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_kennel.csv", τ_kennel)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_kennel_n.csv", τ_kennel_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_hegger.csv", Y_hegger)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_hegger_n.csv", Y_hegger_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_hegger.csv", τ_hegger)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_hegger_n.csv", τ_hegger_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_pec.csv", Y_pec)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_pec_n.csv", Y_pec_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_pec.csv", τ_pec)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_pec_n.csv", τ_pec_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_mcdts.csv", Y_mcdts)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_mcdts_n.csv", Y_mcdts_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_mcdts.csv", τ_mcdts)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_mcdts_n.csv", τ_mcdts_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_pec2.csv", Y_pec2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_pec2_n.csv", Y_pec2_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_pec2.csv", τ_pec2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_pec2_n.csv", τ_pec2_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/ts_pec2.csv", ts_pec2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/ts_pec2_n.csv", ts_pec2_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_mcdts2.csv", Y_mcdts2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_mcdts2_n.csv", Y_mcdts2_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_mcdts2.csv", τ_mcdts2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_mcdts2_n.csv", τ_mcdts2_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/ts_mcdts2.csv", ts_mcdts2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/ts_mcdts2_n.csv", ts_mcdts2_n)

writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_mcdts_fnn.csv", Y_mcdts_fnn)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_mcdts_fnn_n.csv", Y_mcdts_fnn_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_mcdts_fnn.csv", τ_mcdts_fnn)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_mcdts_fnn_n.csv", τ_mcdts_fnn_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_mcdts_fnn2.csv", Y_mcdts_fnn2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/Y_mcdts_fnn2_n.csv", Y_mcdts_fnn2_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_mcdts_fnn2.csv", τ_mcdts_fnn2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/tau_mcdts_fnn2_n.csv", τ_mcdts_fnn2_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/ts_mcdts_fnn2.csv", τ_mcdts_fnn2)
writedlm("./application/artificial data/Lorenz Prediction/Results 2/ts_mcdts_fnn2_n.csv", τ_mcdts_fnn2_n)

# Load data

# make predictions
prediction_cao = deepcopy(Y_cao)
prediction_kennel = deepcopy(Y_kennel)
prediction_hegger = deepcopy(Y_hegger)
prediction_pec = deepcopy(Y_pec)
prediction_mcdts = deepcopy(Y_mcdts)
prediction_mcdts_fnn = deepcopy(Y_mcdts_fnn)
prediction_pec2 = deepcopy(Y_pec2)
prediction_mcdts2 = deepcopy(Y_mcdts2)
prediction_mcdts_fnn2 = deepcopy(Y_mcdts_fnn2)

prediction_cao_n = deepcopy(Y_cao_n)
prediction_kennel_n = deepcopy(Y_kennel_n)
prediction_hegger_n = deepcopy(Y_hegger_n)
prediction_pec_n = deepcopy(Y_pec_n)
prediction_mcdts_n = deepcopy(Y_mcdts_n)
prediction_mcdts_fnn_n = deepcopy(Y_mcdts_fnn_n)
prediction_pec2_n = deepcopy(Y_pec2_n)
prediction_mcdts2_n = deepcopy(Y_mcdts2_n)
prediction_mcdts_fnn2_n = deepcopy(Y_mcdts_fnn2_n)

# Neighbourhoodsize
K = 20

for T = 1:T_steps
    println(T)
    # iterated one step
    predicted_cao, _ = MCDTS.local_linear_prediction(prediction_cao, K; theiler = w1)
    push!(prediction_cao, predicted_cao)
    predicted_kennel, _ = MCDTS.local_linear_prediction(prediction_kennel, K; theiler = w1)
    push!(prediction_kennel, predicted_kennel)
    predicted_hegger, _ = MCDTS.local_linear_prediction(prediction_hegger, K; theiler = w1)
    push!(prediction_hegger, predicted_hegger)
    predicted_pec, _ = MCDTS.local_linear_prediction(prediction_pec, K; theiler = w1)
    push!(prediction_pec, predicted_pec)
    predicted_mcdts, _ = MCDTS.local_linear_prediction(prediction_mcdts, K; theiler = w1)
    push!(prediction_mcdts, predicted_mcdts)
    predicted_mcdts_fnn, _ = MCDTS.local_linear_prediction(prediction_mcdts_fnn, K; theiler = w1)
    push!(prediction_mcdts_fnn, predicted_mcdts_fnn)
    predicted_pec2, _ = MCDTS.local_linear_prediction(prediction_pec2, K; theiler = w1)
    push!(prediction_pec2, predicted_pec2)
    predicted_mcdts2, _ = MCDTS.local_linear_prediction(prediction_mcdts2, K; theiler = w1)
    push!(prediction_mcdts2, predicted_mcdts2)
    predicted_mcdts_fnn2, _ = MCDTS.local_linear_prediction(prediction_mcdts_fnn2, K; theiler = w1)
    push!(prediction_mcdts_fnn2, predicted_mcdts_fnn2)

    predicted_cao_n, _ = MCDTS.local_linear_prediction(prediction_cao_n, K; theiler = w1_n)
    push!(prediction_cao_n, predicted_cao_n)
    predicted_kennel_n, _ = MCDTS.local_linear_prediction(prediction_kennel_n, K; theiler = w1_n)
    push!(prediction_kennel_n, predicted_kennel_n)
    predicted_hegger_n, _ = MCDTS.local_linear_prediction(prediction_hegger_n, K; theiler = w1_n)
    push!(prediction_hegger_n, predicted_hegger_n)
    predicted_pec_n, _ = MCDTS.local_linear_prediction(prediction_pec_n, K; theiler = w1_n)
    push!(prediction_pec_n, predicted_pec)_n
    predicted_mcdts_n, _ = MCDTS.local_linear_prediction(prediction_mcdts_n, K; theiler = w1_n)
    push!(prediction_mcdts_n, predicted_mcdts_n)
    predicted_mcdts_fnn_n, _ = MCDTS.local_linear_prediction(prediction_mcdts_fnn_n, K; theiler = w1_n)
    push!(prediction_mcdts_fnn_n, predicted_mcdts_fnn_n)
    predicted_pec2_n, _ = MCDTS.local_linear_prediction(prediction_pec2_n, K; theiler = w1_n)
    push!(prediction_pec2_n, predicted_pec2_n)
    predicted_mcdts2_n, _ = MCDTS.local_linear_prediction(prediction_mcdts2_n, K; theiler = w1_n)
    push!(prediction_mcdts2_n, predicted_mcdts2_n)
    predicted_mcdts_fnn2_n, _ = MCDTS.local_linear_prediction(prediction_mcdts_fnn2_n, K; theiler = w1_n)
    push!(prediction_mcdts_fnn2_n, predicted_mcdts_fnn2_n)

    # predicted_cao, _ = MCDTS.local_linear_prediction_ar(prediction_cao, K; theiler = w1)
    # push!(prediction_cao, predicted_cao)
    # predicted_kennel, _ = MCDTS.local_linear_prediction_ar(prediction_kennel, K; theiler = w1)
    # push!(prediction_kennel, predicted_kennel)
    # predicted_hegger, _ = MCDTS.local_linear_prediction_ar(prediction_hegger, K; theiler = w1)
    # push!(prediction_hegger, predicted_hegger)
    # predicted_pec, _ = MCDTS.local_linear_prediction_ar(prediction_pec, K; theiler = w1)
    # push!(prediction_pec, predicted_pec)
    # predicted_mcdts, _ = MCDTS.local_linear_prediction_ar(prediction_mcdts, K; theiler = w1)
    # push!(prediction_mcdts, predicted_mcdts)
    # predicted_mcdts_fnn, _ = MCDTS.local_linear_prediction_ar(prediction_mcdts_fnn, K; theiler = w1)
    # push!(prediction_mcdts_fnn, predicted_mcdts_fnn)
    # predicted_pec2, _ = MCDTS.local_linear_prediction_ar(prediction_pec2, K; theiler = w1)
    # push!(prediction_pec2, predicted_pec2)
    # predicted_mcdts2, _ = MCDTS.local_linear_prediction_ar(prediction_mcdts2, K; theiler = w1)
    # push!(prediction_mcdts2, predicted_mcdts2)
    # predicted_mcdts_fnn2, _ = MCDTS.local_linear_prediction_ar(prediction_mcdts_fnn2, K; theiler = w1)
    # push!(prediction_mcdts_fnn2, predicted_mcdts_fnn2)

    # predicted_cao_n, _ = MCDTS.local_linear_prediction_ar(prediction_cao_n, K; theiler = w1_n)
    # push!(prediction_cao_n, predicted_cao_n)
    # predicted_kennel_n, _ = MCDTS.local_linear_prediction_ar(prediction_kennel_n, K; theiler = w1_n)
    # push!(prediction_kennel_n, predicted_kennel_n)
    # predicted_hegger_n, _ = MCDTS.local_linear_prediction_ar(prediction_hegger_n, K; theiler = w1_n)
    # push!(prediction_hegger_n, predicted_hegger_n)
    # predicted_pec_n, _ = MCDTS.local_linear_prediction_ar(prediction_pec_n, K; theiler = w1_n)
    # push!(prediction_pec_n, predicted_pec)_n
    # predicted_mcdts_n, _ = MCDTS.local_linear_prediction_ar(prediction_mcdts_n, K; theiler = w1_n)
    # push!(prediction_mcdts_n, predicted_mcdts_n)
    # predicted_mcdts_fnn_n, _ = MCDTS.local_linear_prediction_ar(prediction_mcdts_fnn_n, K; theiler = w1_n)
    # push!(prediction_mcdts_fnn_n, predicted_mcdts_fnn_n)
    # predicted_pec2_n, _ = MCDTS.local_linear_prediction_ar(prediction_pec2_n, K; theiler = w1_n)
    # push!(prediction_pec2_n, predicted_pec2_n)
    # predicted_mcdts2_n, _ = MCDTS.local_linear_prediction_ar(prediction_mcdts2_n, K; theiler = w1_n)
    # push!(prediction_mcdts2_n, predicted_mcdts2_n)
    # predicted_mcdts_fnn2_n, _ = MCDTS.local_linear_prediction_ar(prediction_mcdts_fnn2_n, K; theiler = w1_n)
    # push!(prediction_mcdts_fnn2_n, predicted_mcdts_fnn2_n)
end

push!(prediction_hegger, predicted_hegger)
push!(prediction_pec, predicted_pec)
push!(prediction_mcdts, predicted_mcdts)
push!(prediction_mcdts_fnn, predicted_mcdts_fnn)
push!(prediction_pec2, predicted_pec2)
push!(prediction_mcdts2, predicted_mcdts2)
push!(prediction_mcdts_fnn2, predicted_mcdts_fnn2)

predicted_cao = prediction_cao[end,:]
predicted_kennel = prediction_kennel[end,:]
predicted_hegger = prediction_hegger[end,:]
predicted_pec = prediction_pec[end,:]
predicted_mcdts = prediction_mcdts[end,:]
predicted_pec2 = prediction_pec2[end,:]
predicted_mcdts2 = prediction_mcdts2[end,:]
predicted_mcdts_fnn2 = prediction_mcdts_fnn2[end,:]
predicted_mcdts_fnn = prediction_mcdts_fnn[end,:]


for i = 625:T_steps
    push!(prediction_cao, predicted_cao)
    push!(prediction_kennel, predicted_kennel)
    push!(prediction_hegger, predicted_hegger)
    push!(prediction_pec, predicted_pec)
    push!(prediction_mcdts, predicted_mcdts)
    push!(prediction_mcdts_fnn, predicted_mcdts_fnn)
    push!(prediction_pec2, predicted_pec2)
    push!(prediction_mcdts2, predicted_mcdts2)
    push!(prediction_mcdts_fnn2, predicted_mcdts_fnn2)
end


# time axis
t2 = (0:T_steps-1) ./lyap_time
t1 = (-length(x1):-1) ./lyap_time
NN = 1000
tt = vcat(t1[end-NN:end], t2)
M = length(tt)
true_data = vcat(x1[end-NN:end], x2)
true_data_n = vcat(x1_[end-NN:end], x2_)

# compute MSE of predictions
MSE_cao = zeros(T_steps+1)
MSE_kennel = zeros(T_steps+1)
MSE_hegger = zeros(T_steps+1)
MSE_pec = zeros(T_steps+1)
MSE_pec2 = zeros(T_steps+1)
MSE_mcdts = zeros(T_steps+1)
MSE_mcdts2 = zeros(T_steps+1)
MSE_mcdts_fnn = zeros(T_steps+1)
MSE_mcdts_fnn2 = zeros(T_steps+1)

MSE_cao_n = zeros(T_steps+1)
MSE_kennel_n = zeros(T_steps+1)
MSE_hegger_n = zeros(T_steps+1)
MSE_pec_n = zeros(T_steps+1)
MSE_pec2_n = zeros(T_steps+1)
MSE_mcdts_n = zeros(T_steps+1)
MSE_mcdts2_n = zeros(T_steps+1)
MSE_mcdts_fnn_n = zeros(T_steps+1)
MSE_mcdts_fnn2_n = zeros(T_steps+1)

for i = 1:T_steps+1
    MSE_cao[i] = MCDTS.compute_mse(prediction_cao[end-T_steps:end-T_steps+i-1,1], true_data[end-T_steps:end-T_steps+i-1])
    MSE_kennel[i] = MCDTS.compute_mse(prediction_kennel[end-T_steps:end-T_steps+i-1,1], true_data[end-T_steps:end-T_steps+i-1])
    MSE_hegger[i] = MCDTS.compute_mse(prediction_hegger[end-T_steps:end-T_steps+i-1,1], true_data[end-T_steps:end-T_steps+i-1])
    MSE_pec[i] = MCDTS.compute_mse(prediction_pec[end-T_steps:end-T_steps+i-1,1], true_data[end-T_steps:end-T_steps+i-1])
    MSE_pec2[i] = MCDTS.compute_mse(prediction_pec2[end-T_steps:end-T_steps+i-1,2], true_data[end-T_steps:end-T_steps+i-1])
    MSE_mcdts[i] = MCDTS.compute_mse(prediction_mcdts[end-T_steps:end-T_steps+i-1,1], true_data[end-T_steps:end-T_steps+i-1])
    MSE_mcdts2[i] = MCDTS.compute_mse(prediction_mcdts2[end-T_steps:end-T_steps+i-1,4], true_data[end-T_steps:end-T_steps+i-1])
    MSE_mcdts_fnn[i] = MCDTS.compute_mse(prediction_mcdts_fnn[end-T_steps:end-T_steps+i-1,1], true_data[end-T_steps:end-T_steps+i-1])
    MSE_mcdts_fnn2[i] = MCDTS.compute_mse(prediction_mcdts_fnn2[end-T_steps:end-T_steps+i-1,2], true_data[end-T_steps:end-T_steps+i-1])

    MSE_cao_n[i] = MCDTS.compute_mse(prediction_cao_n[end-T_steps:end-T_steps+i-1,1], true_data_n[end-T_steps:end-T_steps+i-1])
    MSE_kennel_n[i] = MCDTS.compute_mse(prediction_kennel_n[end-T_steps:end-T_steps+i-1,1], true_data_n[end-T_steps:end-T_steps+i-1])
    MSE_hegger_n[i] = MCDTS.compute_mse(prediction_hegger_n[end-T_steps:end-T_steps+i-1,1], true_data_n[end-T_steps:end-T_steps+i-1])
    MSE_pec_n[i] = MCDTS.compute_mse(prediction_pec_n[end-T_steps:end-T_steps+i-1,1], true_data_n[end-T_steps:end-T_steps+i-1])
    MSE_pec2_n[i] = MCDTS.compute_mse(prediction_pec2_n[end-T_steps:end-T_steps+i-1,2], true_data_n[end-T_steps:end-T_steps+i-1])
    MSE_mcdts_n[i] = MCDTS.compute_mse(prediction_mcdts_n[end-T_steps:end-T_steps+i-1,1], true_data_n[end-T_steps:end-T_steps+i-1])
    MSE_mcdts2_n[i] = MCDTS.compute_mse(prediction_mcdts2_n[end-T_steps:end-T_steps+i-1,4], true_data_n[end-T_steps:end-T_steps+i-1])
    MSE_mcdts_fnn_n[i] = MCDTS.compute_mse(prediction_mcdts_fnn_n[end-T_steps:end-T_steps+i-1,1], true_data_n[end-T_steps:end-T_steps+i-1])
    MSE_mcdts_fnn2_n[i] = MCDTS.compute_mse(prediction_mcdts_fnn2_n[end-T_steps:end-T_steps+i-1,2], true_data_n[end-T_steps:end-T_steps+i-1])
end


## Plot predictions
figure(figsize=(20,10))
subplot(5,1,1)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_cao[end-T_steps:end,1], ".-", label="Cao")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,2)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_kennel[end-T_steps:end,1], ".-", label="Kennel")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,3)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_hegger[end-T_steps:end,1], ".-", label="Hegger")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,4)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_pec[end-T_steps:end,1], ".-", label="PECUZAL")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,5)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts[end-T_steps:end,1], ".-", label="MCDTS")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()
subplots_adjust(hspace=.8)


figure(figsize=(20,10))
subplot(5,1,1)
plot(tt, true_data_n, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_cao_n[end-T_steps:end,1], ".-", label="Cao")
title("NOISE x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,2)
plot(tt, true_data_n, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_kennel_n[end-T_steps:end,1], ".-", label="Kennel")
title("NOISE x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,3)
plot(tt, true_data_n, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_hegger_n[end-T_steps:end,1], ".-", label="Hegger")
title("NOISE x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,4)
plot(tt, true_data_n, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_pec_n[end-T_steps:end,1], ".-", label="PECUZAL")
title("NOISE x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,5)
plot(tt, true_data_n, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts_n[end-T_steps:end,1], ".-", label="MCDTS")
title("NOISE x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()
subplots_adjust(hspace=.8)


##

figure(figsize=(20,10))
subplot(5,1,1)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_cao[end-T_steps:end,1], ".-", label="Cao")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,2)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts_fnn[end-T_steps:end,1], ".-", label="MCDTS FNN")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,3)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts_fnn2[end-T_steps:end,2], ".-", label="MCDTS FNN 2")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,4)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_pec2[end-T_steps:end,2], ".-", label="PECUZAL2")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,5)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts2[end-T_steps:end,4], ".-", label="MCDTS2")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()

subplots_adjust(hspace=.8)


figure(figsize=(20,10))
subplot(5,1,1)
plot(tt, true_data_n, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_cao_n[end-T_steps:end,1], ".-", label="Cao")
title("NOISE x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,2)
plot(tt, true_data_n, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts_fnn_n[end-T_steps:end,1], ".-", label="MCDTS FNN")
title("NOISE x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,3)
plot(tt, true_data_n, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts_fnn2_n[end-T_steps:end,2], ".-", label="MCDTS FNN 2")
title("NOISE x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,4)
plot(tt, true_data_n, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_pec2_n[end-T_steps:end,2], ".-", label="PECUZAL2")
title("NOISE x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,1,5)
plot(tt, true_data_n, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts2_n[end-T_steps:end,4], ".-", label="MCDTS2")
title("NOISE x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()

subplots_adjust(hspace=.8)


## MSEs
figure(figsize=(20,10))
subplot(121)
plot(tt[end-T_steps:end], MSE_mcdts2, ".-", label="MCDTS L 2")
plot(tt[end-T_steps:end], MSE_mcdts, "--", label="MCDTS L")
plot(tt[end-T_steps:end], MSE_mcdts_fnn2, "-", label="MCDTS FNN 2")
plot(tt[end-T_steps:end], MSE_mcdts_fnn, "-.", label="MCDTS FNN")
plot(tt[end-T_steps:end], MSE_pec2, "r.-", label="PECUZAL 2")
plot(tt[end-T_steps:end], MSE_pec, "r.-.", label="PECUZAL")
plot(tt[end-T_steps:end], MSE_cao, "k--", label="CAO")
plot(tt[end-T_steps:end], MSE_kennel, "k-", label="Kennel")
plot(tt[end-T_steps:end], MSE_hegger, "k.-.", label="Hegger")
legend()
title("Forecast Error")
yscale("log")
xlim(-0, 8)
ylabel("MSE")
xlabel("Lyapunov time units")
grid()

subplot(122)
plot(tt[end-T_steps:end], MSE_mcdts2_n, ".-", label="MCDTS L 2")
plot(tt[end-T_steps:end], MSE_mcdts_n, "--", label="MCDTS L")
plot(tt[end-T_steps:end], MSE_mcdts_fnn2_n, "-", label="MCDTS FNN 2")
plot(tt[end-T_steps:end], MSE_mcdts_fnn_n, "-.", label="MCDTS FNN")
plot(tt[end-T_steps:end], MSE_pec2_n, "r.-", label="PECUZAL 2")
plot(tt[end-T_steps:end], MSE_pec_n, "r.-.", label="PECUZAL")
plot(tt[end-T_steps:end], MSE_cao_n, "k--", label="CAO")
plot(tt[end-T_steps:end], MSE_kennel_n, "k-", label="Kennel")
plot(tt[end-T_steps:end], MSE_hegger_n, "k.-.", label="Hegger")
legend()
title("Forecast Error of noisy time series")
yscale("log")
xlim(-0, 8)
ylabel("MSE")
xlabel("Lyapunov time units")
grid()
