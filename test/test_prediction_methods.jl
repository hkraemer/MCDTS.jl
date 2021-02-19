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

w1 = DelayEmbeddings.estimate_delay(tr[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(tr[:,2],"mi_min")
w3 = DelayEmbeddings.estimate_delay(tr[:,3],"mi_min")
theiler = maximum([w1,w2,w3])

D = size(tr,2)
# make predictions with different neighborhoodsizes and optimize w.r.t. this parameter
max_neighbours = 30
MSEs1 = zeros(max_neighbours-(D))
MSEs2 = zeros(max_neighbours-(D))
cnt = 1
for K = (D+1):max_neighbours
    global cnt
    prediction1 = MCDTS.local_linear_prediction(tr[1:end-1,:], K; theiler = theiler)
    MSEs1[cnt] = MCDTS.compute_mse(prediction1, Vector(tr[end,:]))
    prediction2 = MCDTS.local_linear_prediction_ar(tr[1:end-1,:], K; theiler = theiler)
    MSEs2[cnt] = MCDTS.compute_mse(prediction2, Vector(tr[end,:]))
    cnt += 1
end

xs = (D+1):max_neighbours
figure()
plot(xs,MSEs1, label="loc-lin")
plot(xs, label="loc-lin-ar")
legend()
grid()
xlabel("Neighbourhoodsize [no. of neighbours]")
ylabel("root mean squared error")

K1 = xs[findmin(MSEs1)[2]]
K2 = xs[findmin(MSEs2)[2]]

# make predictions
T_steps = 12*lyap_time
prediction1 = deepcopy(tr[1:end-T_steps,:])
prediction2 = deepcopy(tr[1:end-T_steps,:])
for T = 1:T_steps
    println(T)
    predicted1 = MCDTS.local_linear_prediction(prediction1, K1; theiler = theiler)
    push!(prediction1,predicted1)
    predicted2 = MCDTS.local_linear_prediction_ar(prediction2, K2; theiler = theiler)
    push!(prediction2,predicted2)
end


time_axis = 1:length(tr)
sp = length(tr)-T_steps
t2 = (-sp+1:T_steps) ./ lyap_time

figure(figsize=(20,10))
subplot(3,1,1)
plot(t2, tr[:,1], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction1[length(tr)-T_steps+1:length(tr),1], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps+1:end], prediction2[length(tr)-T_steps+1:length(tr),1], ".-", label="prediction [loc-lin-ar]")
title("x-component")
xlim(-10, 12)
ylim(-20,20)
xlabel("Lyapunov time units")
legend()
grid()
subplot(3,1,2)
plot(t2, tr[:,2], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction1[length(tr)-T_steps+1:length(tr),2], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps+1:end], prediction2[length(tr)-T_steps+1:length(tr),2], ".-", label="prediction [loc-lin-ar]")
title("y-component")
xlim(-10, 12)
ylim(-25,25)
xlabel("Lyapunov time units")
legend()
grid()
subplot(3,1,3)
plot(t2, tr[:,3], ".-", label="true data")
plot(t2[end-T_steps+1:end], prediction1[length(tr)-T_steps+1:length(tr),3], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps+1:end], prediction2[length(tr)-T_steps+1:length(tr),3], ".-", label="prediction [loc-lin-ar]")
title("z-component")
xlim(-10, 12)
ylim(0,45)
xlabel("Lyapunov time units")
legend()
grid()
subplots_adjust(hspace=.6)


##
