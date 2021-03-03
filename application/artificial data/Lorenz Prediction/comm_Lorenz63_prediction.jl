using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using DelimitedFiles
using ChaosTools

using PyPlot
pygui(true)

## We predict the Lorenz63-system based on different state space reconstruction methods

lo = Systems.lorenz()
tr = trajectory(lo, 1000; dt = 0.01, Ttr = 100)

# Lyapunov exponent and time
λ = ChaosTools.lyapunov(lo, 100000, dt=0.01; Ttr=1000)
lyap_time = Int(floor((1/λ) / 0.01))

x = tr[:,1]
y = tr[:,2]
w1 = DelayEmbeddings.estimate_delay(tr[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(tr[:,2],"mi_min")

w3 = DelayEmbeddings.estimate_delay(tr[:,3],"mi_min")
theiler = maximum([w1,w2,w3])

## Predictions based on embedding
T_steps = 8*lyap_time
x1 = tr[1:end-T_steps,1]
x2 = tr[end-T_steps+1:end,1]
y1 = tr[1:end-T_steps,2]
y2 = tr[end-T_steps+1:end,2]

dmax = 10
# cao
𝒟, τ_tde1, _ = optimal_traditional_de(x1, "afnn"; dmax = dmax, w = w1)
optimal_d_tde1 = size(𝒟, 2)
τ_cao = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
Y_cao = MCDTS.genembed_for_prediction(x1, τ_cao)

# kennel's method
𝒟, τ_tde2, _ = optimal_traditional_de(x1, "fnn"; dmax = dmax, w = w1)
optimal_d_tde2 = size(𝒟, 2)
τ_kennel = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
Y_kennel = MCDTS.genembed_for_prediction(x1, τ_kennel)

# hegger's method
𝒟, τ_tde3, _ = optimal_traditional_de(x1, "ifnn"; dmax = dmax, w = w1)
optimal_d_tde3 = size(𝒟, 2)
τ_hegger = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
Y_hegger = MCDTS.genembed_for_prediction(x1, τ_hegger)

# pecuzal
taus = 0:100
𝒟, τ_pec, _, L, _ = pecuzal_embedding(x1; τs = taus, w = w1)
optimal_d_tde4 = size(𝒟, 2)
Y_pec = MCDTS.genembed_for_prediction(x1, τ_pec)

# mcdts
trials = 80
tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end], max_depth = 15)
best_node = MCDTS.best_embedding(tree)
τ_mcdts = best_node.τs
Y_mcdts = MCDTS.genembed_for_prediction(x1, τ_mcdts)

data_sample = Dataset(hcat(x1,y1))

# pecuzal
𝒟, τ_pec2, ts_pec2, L, _ = pecuzal_embedding(data_sample; τs = taus, w = w1)
Y_pec2 = MCDTS.genembed_for_prediction(data_sample, τ_pec2, ts_pec2)

# mcdts
trials = 120
tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end], max_depth = 15)
best_node = MCDTS.best_embedding(tree)
τ_mcdts2 = best_node.τs
ts_mcdts2 = best_node.ts
Y_mcdts2 = MCDTS.genembed_for_prediction(data_sample, τ_mcdts2, ts_mcdts2)

# mcdts FNN
trials = 80
tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN=true, max_depth = 15)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_fnn = best_node.τs
Y_mcdts_fnn = MCDTS.genembed_for_prediction(x1, τ_mcdts_fnn)
trials = 120
tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN=true, max_depth = 15)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_fnn2 = best_node.τs
ts_mcdts_fnn2 = best_node.ts
Y_mcdts_fnn2 = MCDTS.genembed_for_prediction(data_sample, τ_mcdts_fnn2, ts_mcdts_fnn2)

# Save data
writedlm("./application/artificial data/Lorenz Prediction/Results/x1.csv", x1)
writedlm("./application/artificial data/Lorenz Prediction/Results/x2.csv", x2)
writedlm("./application/artificial data/Lorenz Prediction/Results/y1.csv", y1)
writedlm("./application/artificial data/Lorenz Prediction/Results/y2.csv", y2)
writedlm("./application/artificial data/Lorenz Prediction/Results/data_sample.csv", data_sample)
writedlm("./application/artificial data/Lorenz Prediction/Results/tr.csv", tr)
writedlm("./application/artificial data/Lorenz Prediction/Results/Y_cao.csv", Y_cao)
writedlm("./application/artificial data/Lorenz Prediction/Results/tau_cao.csv", τ_cao)
writedlm("./application/artificial data/Lorenz Prediction/Results/Y_kennel.csv", Y_kennel)
writedlm("./application/artificial data/Lorenz Prediction/Results/tau_kennel.csv", τ_kennel)
writedlm("./application/artificial data/Lorenz Prediction/Results/Y_hegger.csv", Y_hegger)
writedlm("./application/artificial data/Lorenz Prediction/Results/tau_hegger.csv", τ_hegger)
writedlm("./application/artificial data/Lorenz Prediction/Results/Y_pec.csv", Y_pec)
writedlm("./application/artificial data/Lorenz Prediction/Results/tau_pec.csv", τ_pec)
writedlm("./application/artificial data/Lorenz Prediction/Results/Y_mcdts.csv", Y_mcdts)
writedlm("./application/artificial data/Lorenz Prediction/Results/tau_mcdts.csv", τ_mcdts)
writedlm("./application/artificial data/Lorenz Prediction/Results/Y_pec2.csv", Y_pec2)
writedlm("./application/artificial data/Lorenz Prediction/Results/tau_pec2.csv", τ_pec2)
writedlm("./application/artificial data/Lorenz Prediction/Results/ts_pec2.csv", ts_pec2)
writedlm("./application/artificial data/Lorenz Prediction/Results/Y_mcdts2.csv", Y_mcdts2)
writedlm("./application/artificial data/Lorenz Prediction/Results/tau_mcdts2.csv", τ_mcdts2)
writedlm("./application/artificial data/Lorenz Prediction/Results/ts_mcdts2.csv", ts_mcdts2)

writedlm("./application/artificial data/Lorenz Prediction/Results/Y_mcdts_fnn.csv", Y_mcdts_fnn)
writedlm("./application/artificial data/Lorenz Prediction/Results/tau_mcdts_fnn.csv", τ_mcdts_fnn)
writedlm("./application/artificial data/Lorenz Prediction/Results/Y_mcdts_fnn2.csv", Y_mcdts_fnn2)
writedlm("./application/artificial data/Lorenz Prediction/Results/tau_mcdts_fnn2.csv", τ_mcdts_fnn2)
writedlm("./application/artificial data/Lorenz Prediction/Results/ts_mcdts_fnn2.csv", ts_mcdts_fnn2)

# Load data
####

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
    # predicted_mcdts_fnn2, _ = MCDTS.local_linear_prediction(prediction_mcdts_fnn2, K; theiler = w1)
    # push!(prediction_mcdts_fnn2, predicted_mcdts_fnn2)
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
end

# Plot predictions
figure(figsize=(20,10))
subplot(5,2,1)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_cao[end-T_steps:end,1], ".-", label="Cao")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,2,2)
plot(tt[end-T_steps:end], MSE_cao, ".-")
title("Forecast Error")
yscale("log")
xlim(-5, 12)
ylabel("MSE")
grid()

subplot(5,2,3)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_kennel[end-T_steps:end,1], ".-", label="Kennel")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,2,4)
plot(tt[end-T_steps:end], MSE_kennel, ".-")
title("Forecast Error")
yscale("log")
xlim(-5, 12)
ylabel("MSE")
grid()

subplot(5,2,5)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_hegger[end-T_steps:end,1], ".-", label="Hegger")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,2,6)
plot(tt[end-T_steps:end], MSE_hegger, ".-")
title("Forecast Error")
yscale("log")
xlim(-5, 12)
ylabel("MSE")
grid()

subplot(5,2,7)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_pec[end-T_steps:end,1], ".-", label="PECUZAL")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,2,8)
plot(tt[end-T_steps:end], MSE_pec, ".-")
title("Forecast Error")
yscale("log")
xlim(-5, 12)
ylabel("MSE")
grid()

subplot(5,2,9)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts[end-T_steps:end,1], ".-", label="MCDTS")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()

subplot(5,2,10)
plot(tt[end-T_steps:end], MSE_mcdts, ".-")
title("Forecast Error")
yscale("log")
xlim(-5, 12)
ylabel("MSE")
xlabel("Lyapunov time units")
grid()

subplots_adjust(hspace=.8)


figure(figsize=(20,10))
subplot(5,2,1)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_cao[end-T_steps:end,1], ".-", label="Cao")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,2,2)
plot(tt[end-T_steps:end], MSE_cao, ".-")
title("Forecast Error")
yscale("log")
xlim(-5, 12)
ylabel("MSE")
grid()

subplot(5,2,3)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts_fnn[end-T_steps:end,1], ".-", label="MCDTS FNN")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,2,4)
plot(tt[end-T_steps:end], MSE_mcdts_fnn, ".-")
title("Forecast Error")
yscale("log")
xlim(-5, 12)
ylabel("MSE")
grid()

subplot(5,2,5)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts_fnn2[end-T_steps:end,2], ".-", label="MCDTS FNN 2")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,2,6)
plot(tt[end-T_steps:end], MSE_mcdts_fnn2, ".-")
title("Forecast Error")
yscale("log")
xlim(-5, 12)
ylabel("MSE")
grid()

subplot(5,2,7)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_pec2[end-T_steps:end,2], ".-", label="PECUZAL2")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
legend()
grid()

subplot(5,2,8)
plot(tt[end-T_steps:end], MSE_pec2, ".-")
title("Forecast Error")
yscale("log")
xlim(-5, 12)
ylabel("MSE")
grid()

subplot(5,2,9)
plot(tt, true_data, ".-", label="true data")
plot(tt[end-T_steps:end], prediction_mcdts2[end-T_steps:end,4], ".-", label="MCDTS2")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()

subplot(5,2,10)
plot(tt[end-T_steps:end], MSE_mcdts2, ".-")
title("Forecast Error")
yscale("log")
xlim(-5, 12)
ylabel("MSE")
xlabel("Lyapunov time units")
grid()
subplots_adjust(hspace=.8)


figure(figsize=(15,10))
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
