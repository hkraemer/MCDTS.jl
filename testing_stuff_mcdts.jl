using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using DelimitedFiles
using ChaosTools
using DSP
using Statistics
using Revise
using Random

using PyPlot
pygui(true)


##

lo = Systems.lorenz(rand(3))
dt = 0.01
tr = trajectory(lo, 5000; dt = dt, Ttr = 10)

data = regularize(tr)

ro = Systems.roessler(a=0.25, b=0.28, c=5.8)
tr = trajectory(ro, 544; dt = 0.05, Ttr = 50)

λ = ChaosTools.lyapunov(ro, 1000000, dt=0.05; Ttr=1000)
lyap_time = Int(floor((1/λ) / 0.05))

dmax = 12   # maximum dimension for traditional tde
trials1 = 80 # trials for MCDTS univariate
trials2 = 100 # trials for MCDTS multivariate
taus1 = 0:100 # possible delays
taus2 = 0:25 # possible delays for PRED optimization
max_depth = 15 # depth of the tree
Tw = 1  # time horizon for PRED
KK = 1 # considered nearest neighbors for PRED

# time series to pick
t_idx_1 = 1         # univariate
t_idx_2 = [1,2]     # multivariate

dt = 0.01

# initial conditions
Random.seed!(234)
number_of_ics = 100 # number of different initial conditions
ics = [rand(3) for i in 1:number_of_ics]

i = 1
# set different initial condition and get trajectory
ic = ics[i]
lo = Systems.lorenz(ic)
tr = trajectory(lo, 11.1; dt = dt, Ttr = 10)

figure()
plot3D(tr[:,1], tr[:,2], tr[:,3])

# normalize time series
data = regularize(tr)

Random.seed!(234)
σ = 0.05
T_steps = 110 # 1*lyap_time

x = data[:,1]
x_n = data[:,1] .+ σ*randn(length(data))

x1 = x[1:end-T_steps]       # training
x2 = x[end-T_steps+1:end]   # prediction
x1_n = x_n[1:end-T_steps]
x2_n = x_n[end-T_steps+1:end]

z1 = data[1:end-T_steps,t_idx_2[2]]
z1_n = data[1:end-T_steps,t_idx_2[2]] .+ σ*randn(length(data[1:end-T_steps]))

data_sample = Dataset(x1,z1)
data_sample_n = Dataset(x1_n,z1_n)

w1 = DelayEmbeddings.estimate_delay(x1, "mi_min")
w1_n = DelayEmbeddings.estimate_delay(x1_n, "mi_min")

σ₂ = sqrt(var(x2[1:T_steps]))   # rms deviation for normalization
σ₂_n = sqrt(var(x2_n[1:T_steps]))

# make the reconstructions and then the predictions
# cao
MSEs_cao = zeros(T_steps)
MSEs_cao2 = zeros(T_steps)
𝒟, τ_tde1, _ = optimal_traditional_de(x1, "ifnn"; dmax = dmax, w = w1)
optimal_d_tde1 = size(𝒟, 2)
τ_cao = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
Y = genembed(x1, τ_cao .* (-1))
prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
prediction2 = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)
for j = 1:T_steps
    MSEs_cao[j] = MCDTS.compute_mse(prediction[1:j,1], x2[1:j]) / σ₂
    MSEs_cao2[j] = MCDTS.compute_mse(prediction2[1:j,1], x2[1:j]) / σ₂
end

MSEs_cao_n = zeros(T_steps)
MSEs_cao2_n = zeros(T_steps)
𝒟, τ_tde1, _ = optimal_traditional_de(x1_n, "afnn"; dmax = dmax, w = w1_n)
optimal_d_tde1 = size(𝒟, 2)
τ_cao_n = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
Y = genembed(x1_n, τ_cao_n .* (-1))
prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1_n)
prediction2 = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1_n)
for j = 1:T_steps
    MSEs_cao_n[j] = MCDTS.compute_mse(prediction[1:j,1], x2_n[1:j]) / σ₂_n
    MSEs_cao2_n[j] = MCDTS.compute_mse(prediction2[1:j,1], x2_n[1:j]) / σ₂_n
end

figure()
plot(1:110, MSEs_cao, label="zeroth")
plot(1:110, MSEs_cao2, label="linear")
legend()
yscale("log")
grid()

figure()
plot(1:110, MSEs_pec, label="zeroth")
plot(1:110, MSEs_pec2, label="linear")
legend()
yscale("log")
grid()

figure()
plot(1:110, x2)
plot(1:110, prediction[:,1], label="zeroth")
plot(1:110, prediction2[:,1], label="linear")
grid()
legend()

i = 3
# set different initial condition and get trajectory
ic = ics[i]
lo = Systems.lorenz(ic)
log = Systems.logistic()
hen = Systems.henon(rand(2))
tr = trajectory(lo, 11.1; dt = dt, Ttr = 10)
tr = trajectory(log, 10110)
tr = trajectory(hen, 10030; Ttr = 1000)

λ = ChaosTools.lyapunov(hen, 10000000; Ttr=10000)
lyap_time = Int(floor((1/λ) / 1))

T_steps = 51

figure()
plot3D(tr[:,1], tr[:,2], tr[:,3])

# normalize time series
data = regularize(tr)

Random.seed!(234)
σ = 0.05
T_steps = 51 # 1*lyap_time

x = data[:,1]
x_n = data[:,1] .+ σ*randn(length(data))

x1 = x[1:end-T_steps]       # training
x2 = x[end-T_steps+1:end]   # prediction
x1_n = x_n[1:end-T_steps]
x2_n = x_n[end-T_steps+1:end]

z1 = data[1:end-T_steps,t_idx_2[2]]
z1_n = data[1:end-T_steps,t_idx_2[2]] .+ σ*randn(length(data[1:end-T_steps]))

data_sample = Dataset(x1,z1)
data_sample_n = Dataset(x1_n,z1_n)

w1 = DelayEmbeddings.estimate_delay(x1, "mi_min")
w1_n = DelayEmbeddings.estimate_delay(x1_n, "mi_min")

figure()
plot(1:110, x2)
plot(1:110, prediction[:,1], label="zeroth")
plot(1:110, prediction2[:,1], label="linear")
grid()
legend()

_,taus_pec,_,_,_ = pecuzal_embedding(data_sample)
Y = genembed(x1, taus_pec .* (-1), [1,2])

using MCDTS
MSEs_pec = zeros(T_steps)
MSEs_pec2 = zeros(T_steps)
KK=1
tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), 0:100, 80; tws = 2:taus1[end],
    PRED=true, linear=false, PRED_L=false, PRED_KL = false, Tw = 1, verbose=true)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_n = best_node.τs
ts_mcdts_n = best_node.ts

Y = genembed(x1, τ_mcdts_n .* (-1), ts_mcdts_n)
Y = genembed(x1, [0,18,3,4] .* (-1), [1,1,2,1])
prediction = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)
prediction2 = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)


for j = 1:T_steps
    MSEs_pec[j] = MCDTS.compute_mse(prediction[1:j,1], x2[1:j]) / σ₂
    MSEs_pec2[j] = MCDTS.compute_mse(prediction2[1:j,1], x2[1:j]) / σ₂
end
figure()
plot(1:110, MSEs_pec, label="zeroth")
plot(1:110, MSEs_pec2, label="linear")
legend()
ylim([0.01, 3])
yscale("log")
grid()

figure()
plot(1:1111,vcat(x1,x2))
