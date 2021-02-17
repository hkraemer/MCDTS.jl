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

# make predictions with different neighborhoodsizes and optimize w.r.t. this parameter
MSEs = zeros(30)
for K = 1:30
    prediction = MCDTS.local_linear_prediction(tr[1:end-1,:], K; theiler = 11)
    MSEs[K] = MCDTS.compute_mse(prediction, Vector(tr[end,:]))
end
figure()
plot(1:30,MSEs)
grid()


# make predictions
T_steps = 12*lyap_time
K = 15
predict_lorenz = zeros(T_steps)
prediction = deepcopy(tr[1:end-T_steps,:])
for T = 1:T_steps
    global prediction
    predicted = MCDTS.local_linear_prediction(prediction, K; theiler = 11)
    push!(prediction,predicted)
end

time_axis = 1:length(tr)
sp = length(tr)-T_steps
t2 = (-sp+1:T_steps) ./ lyap_time

figure(figsize=(20,10))
subplot(3,1,1)
plot(t2, tr[:,1], ".-", label="training")
plot(t2[end-T_steps+1:end], prediction[length(tr)-T_steps+1:length(tr),1], ".-", label="prediction")
title("x-component")
xlim(-10, 12)
xlabel("Lyapunov time units")
legend()
grid()
subplot(3,1,2)
plot(t2, tr[:,2], ".-", label="training")
plot(t2[end-T_steps+1:end], prediction[length(tr)-T_steps+1:length(tr),2], ".-", label="prediction")
title("y-component")
xlim(-10, 12)
xlabel("Lyapunov time units")
legend()
grid()
subplot(3,1,3)
plot(t2, tr[:,3], ".-", label="training")
plot(t2[end-T_steps+1:end], prediction[length(tr)-T_steps+1:length(tr),3], ".-", label="prediction")
title("z-component")
xlim(-10, 12)
xlabel("Lyapunov time units")
legend()
grid()
subplots_adjust(hspace=.6)


##
