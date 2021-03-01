using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using DelimitedFiles
using ChaosTools
using DSP
using Statistics

using PyPlot
pygui(true)


data = readdlm("AR3_example.csv")
data = vec(data)

trajectories = zeros(100,T_steps)
coeffs = vec([0.4, 0.2, 0.3])
for i = 1:100
    trajectories[i,:] = MCDTS.get_ar_prediction(data[end-T_steps-2:end-T_steps], coeffs; Tw = T_steps, c=-4, σ=0.4)
end


figure()
plot(1:length(data),data, "k.-", linewidth = 2)
plot(length(data)-T_steps+1:length(data), vec(mean(trajectories;dims=1)))
grid()


tt = vec(mean(trajectories;dims=1))
length(tt)

theiler = DelayEmbeddings.estimate_delay(data, "mi_min")

T_steps = 50
Y_pec, τ_pec, _, L, _ = pecuzal_embedding(data[1:end-T_steps]; τs = 0:100, w = theiler)

YY = MCDTS.genembed_for_prediction(data[1:end-T_steps], τ_pec)

K1 = K2 = 6

tr = deepcopy(YY)

prediction1_ms = deepcopy(tr)
error_prediction1_ms = Dataset(zeros(size(tr)))
prediction1_ios = deepcopy(tr)
error_prediction1_ios = Dataset(zeros(size(tr)))
prediction2_ms = deepcopy(tr)
error_prediction2_ms = Dataset(zeros(size(tr)))
prediction2_ios = deepcopy(tr)
error_prediction2_ios = Dataset(zeros(size(tr)))
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
    predicted1_ms, error_predicted1_ms = MCDTS.local_linear_prediction(tr, K1; Tw = T, theiler = theiler)
    push!(prediction1_ms,predicted1_ms)
    push!(error_prediction1_ms,error_predicted1_ms)
    predicted2_ms, error_predicted2_ms = MCDTS.local_linear_prediction_ar(tr, K2; Tw = T, theiler = theiler)
    push!(prediction2_ms,predicted2_ms)
    push!(error_prediction2_ms,error_predicted2_ms)
end

# Compute real error
actual_error_prediction1_ios = zeros(size(data))
actual_error_prediction1_ms = zeros(size(data))
actual_error_prediction2_ios = zeros(size(data))
actual_error_prediction2_ms = zeros(size(data))
diff = abs(length(tr)-length(data)+T_steps)
for i = 1+diff:length(data)
    actual_error_prediction1_ios[i] = abs.(data[i] .- prediction1_ios[i-diff,1])
    actual_error_prediction1_ms[i] = abs.(data[i] .- prediction1_ms[i-diff,1])
    actual_error_prediction2_ios[i] = abs.(data[i] .- prediction2_ios[i-diff,1])
    actual_error_prediction2_ms[i] = abs.(data[i] .- prediction2_ms[i-diff,1])
end

# Plot predicted vs actual errors
figure(figsize=(20,10))

subplot(1,2,1)
scatter(error_prediction1_ios[end-T_steps:end,1], actual_error_prediction1_ios[end-T_steps:end])
plot(0:110, 0:110, "k--")
title("Reliability (IOS), loc-lin")
xlabel("predicted error")
ylabel("actual error")
yscale("log")
xscale("log")
grid()

subplot(1,2,2)
scatter(error_prediction1_ms[end-T_steps:end,1], actual_error_prediction1_ms[end-T_steps:end])
plot(0:100, 0:100, "k--")
title("Reliability (MS), loc-lin")
xlabel("predicted error")
ylabel("actual error")
yscale("log")
xscale("log")
grid()
subplots_adjust(hspace=.6)


figure(figsize=(20,10))
subplot(1,2,1)
scatter(error_prediction2_ios[end-T_steps:end,1], actual_error_prediction2_ios[end-T_steps:end])
plot(0:110, 0:110, "k--")
title("Reliability (IOS), loc-lin-ar")
xlabel("predicted error")
ylabel("actual error")
yscale("log")
xscale("log")
grid()

subplot(1,2,2)
scatter(error_prediction2_ms[end-T_steps:end,1], actual_error_prediction2_ms[end-T_steps:end])
plot(0:100, 0:100, "k--")
title("Reliability (MS), loc-lin-ar")
xlabel("predicted error")
ylabel("actual error")
yscale("log")
xscale("log")
grid()

subplots_adjust(hspace=.6)

# Plot predictions
time_axis = 1:length(data)
sp = length(data)-T_steps
t2 = (-sp+1:T_steps)

figure(figsize=(20,10))
subplot(2,1,1)
plot(t2[1+diff:end], data[1+diff:end], ".-", label="true data")
plot(t2[end-T_steps:end], prediction1_ios[end-T_steps:end,1], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps:end], prediction2_ios[end-T_steps:end,1], ".-", label="prediction [loc-lin-ar]")
title("iterated one-step forecast ($K2 neighbours)")
xlim(-10, T_steps)
# ylim(-20,20)
xlabel("time units")
legend()
grid()

subplot(2,1,2)
plot(t2, data, ".-", label="true data")
plot(t2[end-T_steps:end], prediction1_ms[end-T_steps:end,1], ".-", label="prediction [loc-lin]")
plot(t2[end-T_steps:end], prediction2_ms[end-T_steps:end,1], ".-", label="prediction [loc-lin-ar]")
title("multi-step forecast ($K2 neighbours)")
xlim(-10, T_steps)
# ylim(-20,20)
xlabel("time units")
legend()
grid()


## Many different Neighbourhoodsizes
MSEs1_ios = zeros(length(4:20))
MSEs2_ios = zeros(length(4:20))
MSEs1_ms = zeros(length(4:20))
MSEs2_ms = zeros(length(4:20))
cnt = 1
for K = 4:20
    global cnt
    println(K)
    tr = deepcopy(YY)
    K1 = K2 = K
    prediction1_ms = deepcopy(tr)
    error_prediction1_ms = Dataset(zeros(size(tr)))
    prediction1_ios = deepcopy(tr)
    error_prediction1_ios = Dataset(zeros(size(tr)))
    prediction2_ms = deepcopy(tr)
    error_prediction2_ms = Dataset(zeros(size(tr)))
    prediction2_ios = deepcopy(tr)
    error_prediction2_ios = Dataset(zeros(size(tr)))
    for T = 1:T_steps
        # iterated one step
        predicted1_ios, error_predicted1_ios = MCDTS.local_linear_prediction(prediction1_ios, K1; theiler = theiler)
        push!(prediction1_ios,predicted1_ios)
        push!(error_prediction1_ios, error_predicted1_ios)
        predicted2_ios, error_predicted2_ios = MCDTS.local_linear_prediction_ar(prediction2_ios, K2; theiler = theiler)
        push!(prediction2_ios,predicted2_ios)
        push!(error_prediction2_ios,error_predicted2_ios)
        # multistep
        predicted1_ms, error_predicted1_ms = MCDTS.local_linear_prediction(tr, K1; Tw = T, theiler = theiler)
        push!(prediction1_ms,predicted1_ms)
        push!(error_prediction1_ms,error_predicted1_ms)
        predicted2_ms, error_predicted2_ms = MCDTS.local_linear_prediction_ar(tr, K2; Tw = T, theiler = theiler)
        push!(prediction2_ms,predicted2_ms)
        push!(error_prediction2_ms,error_predicted2_ms)
    end

    MSEs1_ios[cnt] = MCDTS.compute_mse(data[1+diff:end], prediction1_ios[:,1])
    MSEs2_ios[cnt] = MCDTS.compute_mse(data[1+diff:end], prediction2_ios[:,1])
    MSEs1_ms[cnt] = MCDTS.compute_mse(data[1+diff:end], prediction1_ms[:,1])
    MSEs2_ms[cnt] = MCDTS.compute_mse(data[1+diff:end], prediction2_ms[:,1])

    cnt += 1
end

figure()
plot(4:20,MSEs1_ios, label="IOS loc-lin")
plot(4:20,MSEs1_ms, label="MS loc-lin")
plot(4:20,MSEs2_ios, label="IOS loc-lin-ar")
plot(4:20,MSEs2_ms, label="MS loc-lin-ar")
xlabel("Neighbourhoodsize")
legend()
grid()
ylabel("MSE")


##

lo = Systems.lorenz()
tr = trajectory(lo, 100; dt = 0.01, Ttr = 10)

λ = ChaosTools.lyapunov(lo, 100000, dt=0.01; Ttr=1000)

lyap_time = Int(floor((1/λ) / 0.01))

metric = Euclidean()
K = 5
theiler = 17
Y = deepcopy(tr[1:end-1,:])
Tw = 1
NN = length(Y)
ns = 1:NN
vs = Y[ns]
vtree = KDTree(Y, metric)
allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, theiler)
T = eltype(Y)
D = size(Y,2)

ϵ_ball = zeros(T, K, D) # preallocation
A = ones(K,D) # datamatrix for later linear equation to solve for AR-process
# loop over each fiducial point
NNidxs = allNNidxs[end] # indices of k nearest neighbors to v
# determine neighborhood `Tw` time steps ahead
@inbounds for (i, j) in enumerate(NNidxs)
    ϵ_ball[i, :] .= Y[j + Tw]
    A[i,:] = Y[j]
end


data = DataFrame()
X = A[:,1]

append!(data,DataFrame(X1=X))


data = DataFrame(X1=A[:,1], X2=A[:,2], X3=A[:,3], Y=ϵ_ball[:,1])
ols = lm(@formula(Y ~ X1 + X2 + X3), data)

b  = zeros(D)
ar_coeffs = zeros(D,D)
# make local linear model of the last point of the trajectory
for i = 1:D
    data = DataFrame(X1=A[:,1], X2=A[:,2], X3=A[:,3], Y=ϵ_ball[:,i])
    ols = lm(@formula(Y ~ X1 + X2 + X3), data)
    b[i] = coef(ols)[1]
    ar_coeffs[i,1] = coef(ols)[2]
    ar_coeffs[i,2] = coef(ols)[3]
    ar_coeffs[i,3] = coef(ols)[4]
end

prediction = zeros(D)
prediction[1] = Y[NN,:]'*ar_coeffs[1,:] + b[1]
prediction[2] = Y[NN,:]'*ar_coeffs[2,:] + b[2]
prediction[3] = Y[NN,:]'*ar_coeffs[3,:] + b[3]






# make predictions
prediction = MCDTS.local_linear_prediction(tr[1:end,:], 5; theiler = 11)
pre2 = MCDTS.local_linear_prediction_ar(tr[1:end,:], 5; theiler = 11)
pre3 = MCDTS.local_linear_prediction_ar2(tr[1:end,:], 5; theiler = 11)


MSEs = zeros(30)
for K = 1:30
    prediction = MCDTS.local_linear_prediction(tr[1:end-1,:], K; theiler = 11)
    MSEs[K] = MCDTS.compute_mse(prediction, Vector(tr[end,:]))
end

figure()
plot(1:30,MSEs)
grid()

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



figure(figsize=(14., 8.))
subplot(1,2,1, projection="3d")
plot3D(prediction[:,1], prediction[:,2], prediction[:,3],"gray")
title("Predicted Lorenz System")
xlabel("x(t+1))")
ylabel("y(t+1)")
zlabel("z(t+1)")
grid()

subplot(1,2,2, projection="3d")
plot3D(tr[:,1], tr[:,2], tr[:,3],"gray")
title("Original Lorenz System")
xlabel("x(t)")
ylabel("y(t)")
zlabel("z(t)")
grid()


# Parameters data:
N = 8 # number of oscillators
Fs = 3.5:0.004:5 # parameter spectrum
dt = 0.1 # sampling time
total = 5000  # time series length

# Parameters analysis:
ε = 0.05  # recurrence threshold
dmax = 10   # maximum dimension for traditional tde
lmin = 2   # minimum line length for RQA
trials = 80 # trials for MCDTS
taus = 0:100 # possible delays

# pick one time series
t_idx = 2
#t_idx = [2,4,7]

# init Lorenz96
u0 = [0.590; 0.766; 0.566; 0.460; 0.794; 0.854; 0.200; 0.298]
lo96 = Systems.lorenz96(N, u0; F = 3.5)

i = 370
F = Fs[i]
F = 4.8
set_parameter!(lo96, 1, F)
data = trajectory(lo96, total*dt; dt = dt, Ttr = 2500 * dt)
data_sample = data[:,t_idx]


figure()
plot(data_sample)

# Traditional time delay embedding
𝒟, τ_tde, _ = optimal_traditional_de(data_sample, "fnn"; dmax = dmax)
optimal_d_tde = size(𝒟, 2)
R = RecurrenceMatrix(𝒟, ε; fixedrate = true)
RQA = rqa(R; theiler = τ_tde, lmin = lmin)
RQA_tde = hcat(RQA...)
τ_tdes = [(i-1)*τ_tde for i = 1:optimal_d_tde]
L_tde = MCDTS.compute_delta_L(data_sample, τ_tdes, taus[end]; w = τ_tde)

# PECUZAL
theiler = τ_tde
@time 𝒟_pec, τ_pec, ts_pec, Ls_pec , _ = DelayEmbeddings.pecuzal_embedding(data_sample; τs = taus , w = theiler, econ = true, threshold = 0.05)
optimal_d_pec = size(𝒟_pec,2)
R = RecurrenceMatrix(𝒟_pec, ε; fixedrate = true)
RQA = rqa(R; theiler = theiler, lmin = lmin)
RQA_pec = hcat(RQA...)
L_pec = sum(Ls_pec)

# MCDTS
@time tree = MCDTS.mc_delay(Dataset(data_sample), theiler, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:2:taus[end], threshold = 0.05)
best_node = MCDTS.best_embedding(tree)
𝒟_mcdts = genembed(data_sample, best_node.τs, best_node.ts)
optimal_d_mcdts = size(𝒟_mcdts,2)
R = RecurrenceMatrix(𝒟_mcdts, ε; fixedrate = true)
RQA = rqa(R; theiler = theiler, lmin = lmin)
RQA_mcdts = hcat(RQA...)
L_mcdts = best_node.L



pwd()
λs = readdlm("./application/artificial data/Lorenz96/Lyapunov spectrum/results_Lorenz96_N_40_lyapunovs.csv", ',', Any, '[')
Fs = readdlm("./application/artificial data/Lorenz96/Lyapunov spectrum/results_Lorenz96_N_40_lyapunovs_Fs.csv")
λs = λs[:,1:end-1]

pos_Lyap_idx = λs[:,1] .> 10^-3

l_width_vert = 0.1
figure(figsize=(20,10))
plot(Fs, λs)
ylims1 = axis1.get_ylim()
vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
title("Lyaps")
ylabel("embedding dimension")
grid()
