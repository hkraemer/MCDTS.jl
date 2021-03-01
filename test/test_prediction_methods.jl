using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using DelimitedFiles
using ChaosTools

using PyPlot
pygui(true)

lo = Systems.lorenz()
tr = trajectory(lo, 100; dt = 0.01, Ttr = 10)

# Lyapunov exponent and time
λ = ChaosTools.lyapunov(lo, 100000, dt=0.01; Ttr=1000)
lyap_time = Int(floor((1/λ) / 0.01))

x = tr[:,1]
y = tr[:,2]
w1 = DelayEmbeddings.estimate_delay(tr[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(tr[:,2],"mi_min")
w3 = DelayEmbeddings.estimate_delay(tr[:,3],"mi_min")
theiler = maximum([w1,w2,w3])

## Predictions based on true data

D = size(tr,2)
# make predictions with different neighborhoodsizes and optimize w.r.t. this parameter
max_neighbours = 30
MSEs1 = zeros(max_neighbours-(D))
MSEs2 = zeros(max_neighbours-(D))
cnt = 1
for K = (D+1):max_neighbours
    global cnt
    prediction1, _ = MCDTS.local_linear_prediction(tr[1:end-1,:], K; theiler = theiler)
    MSEs1[cnt] = MCDTS.compute_mse(prediction1, Vector(tr[end,:]))
    prediction2 = MCDTS.local_linear_prediction_ar(tr[1:end-1,:], K; theiler = theiler)
    MSEs2[cnt] = MCDTS.compute_mse(prediction2, Vector(tr[end,:]))
    cnt += 1
end

xs = (D+1):max_neighbours
figure()
plot(xs,MSEs1, label="loc-lin")
plot(xs,MSEs2, label="loc-lin-ar")
legend()
grid()
xlabel("Neighbourhoodsize [no. of neighbours]")
ylabel("root mean squared error")

K1 = xs[findmin(MSEs1)[2]]
K2 = xs[findmin(MSEs2)[2]]

K1 = 20
K2 = 20

# make predictions; multi-step `ms` and iterated one-step `ios`
T_steps = 12*lyap_time
prediction1_ms = deepcopy(tr[1:end-T_steps,:])
error_prediction1_ms = Dataset(zeros(size(tr[1:end-T_steps,:])))
prediction1_ios = deepcopy(tr[1:end-T_steps,:])
error_prediction1_ios = Dataset(zeros(size(tr[1:end-T_steps,:])))
prediction2_ms = deepcopy(tr[1:end-T_steps,:])
error_prediction2_ms = Dataset(zeros(size(tr[1:end-T_steps,:])))
prediction2_ios = deepcopy(tr[1:end-T_steps,:])
error_prediction2_ios = Dataset(zeros(size(tr[1:end-T_steps,:])))
for T = 1:T_steps
    println(T)
    # iterated one step
    predicted1_ios, error_predicted1_ios = MCDTS.local_linear_prediction(prediction1_ios, K1; theiler = theiler)
    push!(prediction1_ios,predicted1_ios)
    push!(error_prediction1_ios, error_predicted1_ios)
    predicted2_ios, error_predicted2_ios = MCDTS.local_linear_prediction_ar(prediction2_ios, K2; theiler = theiler)
    push!(prediction2_ios,predicted2_ios)
    push!(error_prediction2_ios,error_predicted2_ios)
    # multistep
    predicted1_ms, error_predicted1_ms = MCDTS.local_linear_prediction(tr[1:end-T_steps,:], K1; Tw = T, theiler = theiler)
    push!(prediction1_ms,predicted1_ms)
    push!(error_prediction1_ms,error_predicted1_ms)
    predicted2_ms, error_predicted2_ms = MCDTS.local_linear_prediction_ar(tr[1:end-T_steps,:], K2; Tw = T, theiler = theiler)
    push!(prediction2_ms,predicted2_ms)
    push!(error_prediction2_ms,error_predicted2_ms)
end

# Compute real error
actual_error_prediction1_ios = zeros(size(tr))
actual_error_prediction1_ms = zeros(size(tr))
actual_error_prediction2_ios = zeros(size(tr))
actual_error_prediction2_ms = zeros(size(tr))
for i = 1:length(tr)
    actual_error_prediction1_ios[i,:] = abs.(tr[i,:] .- prediction1_ios[i,:])
    actual_error_prediction1_ms[i,:] = abs.(tr[i,:] .- prediction1_ms[i,:])
    actual_error_prediction2_ios[i,:] = abs.(tr[i,:] .- prediction2_ios[i,:])
    actual_error_prediction2_ms[i,:] = abs.(tr[i,:] .- prediction2_ms[i,:])
end

comp = ["x-component", "y-component", "z-component"]
# Plot predicted vs actual errors
figure(figsize=(20,10))
cnt = 1
for i = 1:2:5
    global cnt
    subplot(3,2,i)
    scatter(error_prediction1_ios[end-T_steps:end,cnt], actual_error_prediction1_ios[end-T_steps:end,cnt])
    plot(0:110, 0:110, "k--")
    title("Reliability "*comp[cnt]*" (IOS), loc-lin")
    xlabel("predicted error")
    ylabel("actual error")
    yscale("log")
    xscale("log")
    grid()

    subplot(3,2,i+1)
    scatter(error_prediction1_ms[end-T_steps:end,cnt], actual_error_prediction1_ms[end-T_steps:end,cnt])
    plot(0:10000, 0:10000, "k--")
    title("Reliability "*comp[cnt]*" (MS), loc-lin")
    xlabel("predicted error")
    ylabel("actual error")
    yscale("log")
    xscale("log")
    grid()
    subplots_adjust(hspace=.6)

    cnt += 1
end

figure(figsize=(20,10))
cnt = 1
for i = 1:2:5
    global cnt
    subplot(3,2,i)
    scatter(error_prediction2_ios[:,cnt], actual_error_prediction2_ios[:,cnt])
    plot(0:110, 0:110, "k--")
    title("Reliability "*comp[cnt]*" (IOS), loc-lin-ar")
    xlabel("predicted error")
    ylabel("actual error")
    yscale("log")
    xscale("log")
    grid()

    subplot(3,2,i+1)
    scatter(error_prediction2_ms[:,cnt], actual_error_prediction2_ms[:,cnt])
    plot(0:10000, 0:10000, "k--")
    title("Reliability "*comp[cnt]*" (MS), loc-lin-ar")
    xlabel("predicted error")
    ylabel("actual error")
    yscale("log")
    xscale("log")
    grid()

    subplots_adjust(hspace=.6)

    cnt += 1
end

# Plot predictions
time_axis = 1:length(tr)
sp = length(tr)-T_steps
t2 = (-sp+1:T_steps) ./ lyap_time

figure(figsize=(20,10))
subplot(3,1,1)
plot(t2, tr[:,1], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction1_ios[length(tr)-T_steps+1:length(tr),1], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps+1:end], prediction2_ios[length(tr)-T_steps+1:length(tr),1], ".-", label="prediction [loc-lin-ar]")
title("x-component (iterated one-step)")
xlim(-10, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()
subplot(3,1,2)
plot(t2, tr[:,2], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction1_ios[length(tr)-T_steps+1:length(tr),2], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps+1:end], prediction2_ios[length(tr)-T_steps+1:length(tr),2], ".-", label="prediction [loc-lin-ar]")
title("y-component (iterated one-step)")
xlim(-10, 12)
ylim(-25,25)
xlabel("Lyapunov time units")
legend()
grid()
subplot(3,1,3)
plot(t2, tr[:,3], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction1_ios[length(tr)-T_steps+1:length(tr),3], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps+1:end], prediction2_ios[length(tr)-T_steps+1:length(tr),3], ".-", label="prediction [loc-lin-ar]")
title("z-component (iterated one-step)")
xlim(-10, 12)
ylim(0,45)
xlabel("Lyapunov time units")
legend()
grid()
subplots_adjust(hspace=.6)

figure(figsize=(20,10))
subplot(3,1,1)
plot(t2, tr[:,1], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction1_ms[length(tr)-T_steps+1:length(tr),1], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps+1:end], prediction2_ms[length(tr)-T_steps+1:length(tr),1], ".-", label="prediction [loc-lin-ar]")
title("x-component (multi-step)")
xlim(-10, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()
subplot(3,1,2)
plot(t2, tr[:,2], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction1_ms[length(tr)-T_steps+1:length(tr),2], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps+1:end], prediction2_ms[length(tr)-T_steps+1:length(tr),2], ".-", label="prediction [loc-lin-ar]")
title("y-component (multi-step)")
xlim(-10, 12)
ylim(-25,25)
xlabel("Lyapunov time units")
legend()
grid()
subplot(3,1,3)
plot(t2, tr[:,3], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction1_ms[length(tr)-T_steps+1:length(tr),3], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps+1:end], prediction2_ms[length(tr)-T_steps+1:length(tr),3], ".-", label="prediction [loc-lin-ar]")
title("z-component (multi-step)")
xlim(-10, 12)
ylim(0,45)
xlabel("Lyapunov time units")
legend()
grid()
subplots_adjust(hspace=.6)


## Predictions based on embedding
T_steps = 12*lyap_time
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


# make predictions
prediction_cao = deepcopy(Y_cao)
prediction_kennel = deepcopy(Y_kennel)
prediction_hegger = deepcopy(Y_hegger)
prediction_pec = deepcopy(Y_pec)
prediction_mcdts = deepcopy(Y_mcdts)
prediction_pec2 = deepcopy(Y_pec2)
prediction_mcdts2 = deepcopy(Y_mcdts2)

# Neighbourhoodsize
K = 10

for T = 1:T_steps
    println(T)
    # iterated one step
    predicted_cao, _ = MCDTS.local_linear_prediction_ar(prediction_cao, K; theiler = w1)
    push!(prediction_cao, predicted_cao)
    predicted_kennel, _ = MCDTS.local_linear_prediction_ar(prediction_kennel, K; theiler = w1)
    push!(prediction_cao, predicted_cao)
    predicted_hegger, _ = MCDTS.local_linear_prediction_ar(prediction_hegger, K; theiler = w1)
    push!(prediction_hegger, predicted_hegger)
    predicted_pec, _ = MCDTS.local_linear_prediction_ar(prediction_pec, K; theiler = w1)
    push!(prediction_pec, predicted_pec)
    predicted_mcdts, _ = MCDTS.local_linear_prediction_ar(prediction_mcdts, K; theiler = w1)
    push!(prediction_mcdts, predicted_mcdts)
    predicted_pec2, _ = MCDTS.local_linear_prediction_ar(prediction_pec2, K; theiler = w1)
    push!(prediction_pec2, predicted_pec2)
    predicted_mcdts2, _ = MCDTS.local_linear_prediction_ar(prediction_mcdts2, K; theiler = w1)
    push!(prediction_mcdts2, predicted_mcdts2)
end

# Plot predictions
sp = length(tr)-T_steps
t2 = (-sp+1:T_steps)

figure(figsize=(20,10))
subplot(5,1,1)
plot(t2[8000:end], tr[8000:end,1], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction_cao[end-T_steps+1:end,1], ".-", label="Cao")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()

subplot(5,1,2)
plot(t2[8000:end], tr[8000:end,1], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction_kennel[end-T_steps+1:end,1], ".-", label="Kennel")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()

subplot(5,1,3)
plot(t2[8000:end], tr[8000:end,1], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction_hegger[end-T_steps+1:end,1], ".-", label="Hegger")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()

subplot(5,1,4)
plot(t2[8000:end], tr[8000:end,1], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction_pec[end-T_steps+1:end,1], ".-", label="PECUZAL")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()

subplot(5,1,5)
plot(t2[8000:end], tr[8000:end,1], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction_mcdts[end-T_steps+1:end,1], ".-", label="MCDTS")
title("x-component (iterated one-step)")
xlim(-5, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()
