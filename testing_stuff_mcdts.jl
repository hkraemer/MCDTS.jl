using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using Statistics
using Revise
using Random
using DSP
using GLM
using DataFrames
using StatsBase
using LinearAlgebra
using StatsModels
using DelimitedFiles

using PyPlot
pygui(true)


## Lorenz

lo = Systems.lorenz(rand(3))
dt = 0.01
tr = trajectory(lo, 500; dt = dt, Ttr = 10)

data = DelayEmbeddings.standardize(tr)

w = DelayEmbeddings.estimate_delay(tr[:,1], "mi_min")
Yx, taus, _, _, _ = pecuzal_embedding(tr[:,1]; w=w)
Yy = genembed(tr[:,2], taus)

figure()
plot3D(Yx[:,1], Yx[:,2], Yx[:,3])

figure()
plot3D(Yy[:,1], Yy[:,2], Yy[:,3])


lengths = 1000:1000:length(Yx)
ρ_xy = zeros(length(lengths))
ρ_yx = zeros(length(lengths))
for (i,l) in enumerate(lengths)
    println(i)
    ρ_xy[i], _ = MCDTS.ccm(Yx[1:l,:], Yy[1:l,:])
    ρ_yx[i], _ = MCDTS.ccm(Yy[1:l,:], Yx[1:l,:])
end

figure()
plot(lengths,ρ_xy, label="X→Y")
plot(lengths,ρ_xy, label="Y→X")
grid()
legend()


# fiktive Zeitreihe
timeseries = 1:100
Ys = Dataset(timeseries) # Wenn man mehrere Zeitreihen hätte, dann würde Ys mehrere Dimensionen haben

# First embedding cycle:

# fiktive potentielle Delays aufgrund von irgendeiner Statistik oder loss oderso
potential_delays = [5, 8 , 19]

ts_number = (1, 1) # hier immer 1, weil wir haben nur eine Zeitreihe
# Nehmen wir an wir benutzen als erste Komponente die ungelaggte Zeitreihe:
delays1 = (0, potential_delays[1])
delays2= (0, potential_delays[2])
delays3 = (0, potential_delays[3])


Y_trial_1 = genembed(Ys, delays1, ts_number)
Y_trial_2 = genembed(Ys, delays2, ts_number)
Y_trial_3 = genembed(Ys, delays3, ts_number)

# So, jetzt sagen wir, dass in diesem Embedding Cycle der delay 8 der tollste war,
# dann sähe der nächste embedding cycle so aus:

# fiktive potentielle Delays aufgrund von irgendeiner Statistik oder loss oderso
potential_delays = [1, 5]

ts_number = (1, 1, 1) # hier immer 1, weil wir haben nur eine Zeitreihe

delays1 = (delays2..., potential_delays[1])
delays2 = (delays2..., potential_delays[2])

Y_trial_1 = genembed(Ys, delays1, ts_number)
Y_trial_2 = genembed(Ys, delays2, ts_number)

## coupled Logistic
L = 3500
x = zeros(L)
y = zeros(L)
r = 3.8
r2 = 3.5
βxy = 0.02
βyx = 0.1
x[1]=0.4
y[1]=0.2

for i = 2:L
    x[i]=x[i-1]*(r-r*x[i-1]-βxy*y[i-1])
    y[i]=y[i-1]*(r2-r2*y[i-1]-βyx*x[i-1])
end

figure()
plot(x)
plot(y)

using MCDTS
using Random
Yx = embed(x,2,1)
Yy = embed(y,2,1)

w1 = DelayEmbeddings.estimate_delay(x, "mi_min")
ρ, Y_hat = MCDTS.ccm(Yx, y[1:length(Yx)]; w = w1)

yyy, _ = optimal_traditional_de(x, "fnn"; w = w1)

Random.seed!(1234)
tree = MCDTS.mc_delay(Dataset(x), w1, (L)->(MCDTS.softmaxL(L,β=2.)), 0:10, 10;
    verbose=true, CCM = true, Y_CCM = y)
best_node = MCDTS.best_embedding(tree)
τ_mcdts = best_node.τs .*(-1)
ts_mcdts2_n = best_node.ts
L = best_node.L

##
d1 = 2
d2 = 2
τ1 = 1
τ2 = 1
cnt = 1
rho_x = zeros(length(100:100:3500))
rho_y = zeros(length(100:100:3500))
rho_mcdts_x = zeros(length(100:100:3500))
rho_mcdts_y = zeros(length(100:100:3500))
for i = 100:100:3500
    xx = x[1:i]
    yy = y[1:i]
    Yx = genembed(xx,[0,-τ1])
    Yy = genembed(yy,[0,-τ2])

    Yx2 = genembed(xx, τ_mcdts)
    Yy2 = genembed(yy, τ_mcdts)

    if i == 100
        w = 5
    else
        w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
        w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")
        w = maximum([w1,w2])
    end

    rho_y[cnt], _ = MCDTS.ccm(Yx,yy[1+τ1:length(Yx)+τ1]; w = w)
    rho_x[cnt], _ = MCDTS.ccm(Yy,xx[1+τ2:length(Yy)+τ2]; w = w)

    rho_mcdts_y[cnt], _ = MCDTS.ccm(Yx2,yy[1+maximum(τ_mcdts.*(-1)):length(Yx2)+maximum(τ_mcdts.*(-1))]; w = w)
    rho_mcdts_x[cnt], _ = MCDTS.ccm(Yy2,xx[1+maximum(τ_mcdts.*(-1)):length(Yy2)+maximum(τ_mcdts.*(-1))]; w = w)

    cnt +=1
end


figure()
plot(100:100:3500,rho_x)
plot(100:100:3500,rho_y)
grid()

figure()
plot(100:100:3500,rho_mcdts_x)
plot(100:100:3500,rho_mcdts_y)
grid()

using MCDTS

test1 = regularize(Dataset(x))
test2 = regularize(Dataset(y))
# try MCDTS with CCM
taus1 = 0:10 # the possible delay vals
trials = 20 # the sampling of the tree
Random.seed!(1234)
tree = MCDTS.mc_delay(test1, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials;
    verbose=false, CCM = true, Y_CCM = test2)
best_node = MCDTS.best_embedding(tree)
τ_mcdts2_n = best_node.τs
ts_mcdts2_n = best_node.ts
L = best_node.L


Random.seed!(1234)
tree = MCDTS.mc_delay(test2, w2, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials;
    verbose=false, CCM = true, Y_CCM = test1)
best_node = MCDTS.best_embedding(tree)
τ_mcdts2_n = best_node.τs
ts_mcdts2_n = best_node.ts
L = best_node.L



test = Dataset(x)

println(test[1:5])

## coupled logistic with lag

## coupled Logistic
using MCDTS
using DelayEmbeddings
using PyPlot
pygui(true)

L = 3000
x = zeros(L)
y = zeros(L)
r = 3.78
r2 = 3.77
βxy = 0.07
βyx = 0.08
x[1]=0.2
y[1]=0.4
x[2]=0.6
y[2]=0.9
x[3]=0.8
y[3]=0.3
x[4]=0.41
y[4]=0.7
x[4]=0.89
y[4]=0.63
lag = 0
for i = 2+lag:L
    x[i]=x[i-1]*(r-r*x[i-1]-βxy*y[i-1])
    y[i]=y[i-1]*(r2-r2*y[i-1]-βyx*x[i-1-lag])
end

xx = x[100:end]
yy = y[100:end]

figure()
plot(xx)

d1 = 2
d2 = 2
τ1 = 1
τ2 = 1
taus1 = [-i*τ1 for i = 0:d1-1]
taus2 = [-i*τ2 for i = 0:d2-1]
cnt = 1
# rho_x = zeros(length(-18:16))
# rho_y = zeros(length(-18:16))

Yx = genembed(xx,taus1)
Yy = genembed(yy,taus2)

w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")

rho_y, _ = MCDTS.ccm(Yx,Yy; w = w1)
rho_x, _ = MCDTS.ccm(Yy,Yx; w = w2)

# for l = -18:16
#
#     Yx = genembed(xx,taus1)
#     Yy = genembed(yy,taus2)
#
#     # shift vals for lagged ccm
#     if l < 0
#         YX = Yx[-l+1:end,:]
#         YY = Yx[1:end+l,:]
#     else
#         YX = Yx[1:end-l,:]
#         YY = Yx[l+1:end,:]
#     end
#
#     w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
#     w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")
#
#     rho_y[cnt], _ = MCDTS.ccm(YX,YY; w = w1)
#     rho_x[cnt], _ = MCDTS.ccm(YY,YX; w = w2)
#
#     cnt +=1
# end

figure()
plot(Vector(-10:10),rho_x, label="x xmaps y")
plot(Vector(-10:10),rho_y, label="y xmaps x")
grid()
legend()


## Combustion data

data1 = readdlm("application/Causality/Combustion/data/pressure_downsampled_same_sampling.txt")
data2 = readdlm("application/Causality/Combustion/data/heat_release_downsampled_same_sampling.txt")

## Generate subset

Random.seed!(111)
N = 5000
s = rand(1:length(data1)-N)
s1 = data1[s:s+N]
s2 = data2[s:s+N]
s1 = s1 .+ 0.0000000001.*randn(length(s1))
s2 = s2 .+ 0.0000000001.*randn(length(s2))

s1 = (s1 .- mean(s1)) ./ std(s1)
s2 = (s2 .- mean(s2)) ./ std(s2)

# Embedding
w1 = DelayEmbeddings.estimate_delay(s1, "mi_min")
w2 = DelayEmbeddings.estimate_delay(s2, "mi_min")

ρp = Statistics.cor(s1,s2)

figure()
plot(s1,".-")
plot(s2,".-")


Y, delay, _ = optimal_traditional_de(s1, "afnn"; w = w1)
taus_cao1 = [i*delay for i = 0:size(Y,2)-1]
Y, delay, _ = optimal_traditional_de(s2, "afnn"; w = w2)
taus_cao2 = [i*delay for i = 0:size(Y,2)-1]
_, taus_pec1,_,_,_ = pecuzal_embedding(s1; w = w1, econ = true)
_, taus_pec2,_,_,_ = pecuzal_embedding(s2; w = w2, KNN=2, econ = true)

Random.seed!(1234)
tree = MCDTS.mc_delay(Dataset(s1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), 0:30, 10;
    verbose=true, CCM = true, Y_CCM = s2)
best_node = MCDTS.best_embedding(tree)
τ_mcdts1 = best_node.τs
L = best_node.L


tree = MCDTS.mc_delay(Dataset(s2), w2, (L)->(MCDTS.softmaxL(L,β=2.)), 0:30, 5;
    verbose=true, CCM = true, Y_CCM = s1)
best_node = MCDTS.best_embedding(tree)
τ_mcdts2 = best_node.τs
L = best_node.L



τs = 0:60
trials = 50

cnt = 1
rho_x = zeros(length(100:100:5000))
rho_y = zeros(length(100:100:5000))
rho_x2 = zeros(length(100:100:5000))
rho_y2 = zeros(length(100:100:5000))
rho_pec_x = zeros(length(100:100:5000))
rho_pec_y = zeros(length(100:100:5000))
rho_pec_x2 = zeros(length(100:100:5000))
rho_pec_y2 = zeros(length(100:100:5000))
rho_mcdts_x = zeros(length(100:100:5000))
rho_mcdts_y = zeros(length(100:100:5000))
rho_mcdts_x2 = zeros(length(100:100:5000))
rho_mcdts_y2 = zeros(length(100:100:5000))
for i = 100:100:5000
    println(i)
    xx = s1[1:i]
    yy = s2[1:i]

    if i == 100
        w = 5
    else
        w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
        w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")
        w = maximum([w1,w2])
    end

    # embedding
    #classic
    # Y, delay, _ = optimal_traditional_de(s1, "afnn"; w = w1)
    # taus_cao1 = [i*delay for i = 0:size(Y,2)-1]
    # Y, delay, _ = optimal_traditional_de(s2, "afnn"; w = w2)
    # taus_cao2 = [i*delay for i = 0:size(Y,2)-1]
    #
    # # pecuzal
    # _, taus_pec1,_,_,_ = pecuzal_embedding(s1; w = w1, econ = true)
    # _, taus_pec2,_,_,_ = pecuzal_embedding(s2; w = w2, KNN=2, econ = true)
    #
    # # mcdts
    # Random.seed!(1234)
    # tree = MCDTS.mc_delay(Dataset(s1), w, (L)->(MCDTS.softmaxL(L,β=2.)), τs, trials;
    #     verbose=true, CCM = true, Y_CCM = s2)
    # best_node = MCDTS.best_embedding(tree)
    # τ_mcdts1 = best_node.τs
    # L = best_node.L
    #
    # tree = MCDTS.mc_delay(Dataset(s2), w, (L)->(MCDTS.softmaxL(L,β=2.)), τs, trials;
    #     verbose=true, CCM = true, Y_CCM = s1)
    # best_node = MCDTS.best_embedding(tree)
    # τ_mcdts2 = best_node.τs
    # L = best_node.L


    Yx_cao = genembed(xx,-taus_cao1)
    Yy_cao = genembed(yy,-taus_cao1)

    Yx_cao2 = genembed(xx,-taus_cao2)
    Yy_cao2 = genembed(yy,-taus_cao2)

    Yx_pec = genembed(xx,-taus_pec1)
    Yy_pec = genembed(yy,-taus_pec1)

    Yx_pec2 = genembed(xx,-taus_pec2)
    Yy_pec2 = genembed(yy,-taus_pec2)

    Yx_mcdts = genembed(xx,-τ_mcdts1)
    Yy_mcdts = genembed(yy,-τ_mcdts1)

    Yx_mcdts2 = genembed(xx,-τ_mcdts2)
    Yy_mcdts2 = genembed(yy,-τ_mcdts2)

    # compute CCM
    rho_y[cnt], _ = MCDTS.ccm(Yx_cao, Yy_cao; w = w)
    rho_x[cnt], _ = MCDTS.ccm(Yy_cao, Yx_cao; w = w)

    rho_y2[cnt], _ = MCDTS.ccm(Yx_cao2, Yy_cao2; w = w)
    rho_x2[cnt], _ = MCDTS.ccm(Yy_cao2, Yx_cao2; w = w)

    rho_pec_y[cnt], _ = MCDTS.ccm(Yx_pec, Yy_pec; w = w)
    rho_pec_x[cnt], _ = MCDTS.ccm(Yy_pec, Yx_pec; w = w)

    rho_pec_y2[cnt], _ = MCDTS.ccm(Yx_pec2, Yy_pec2; w = w)
    rho_pec_x2[cnt], _ = MCDTS.ccm(Yy_pec2, Yx_pec2; w = w)

    rho_mcdts_y[cnt], _ = MCDTS.ccm(Yx_mcdts, Yy_mcdts; w = w)
    rho_mcdts_x[cnt], _ = MCDTS.ccm(Yy_mcdts, Yx_mcdts; w = w)

    rho_mcdts_y2[cnt], _ = MCDTS.ccm(Yx_mcdts2, Yy_mcdts2; w = w)
    rho_mcdts_x2[cnt], _ = MCDTS.ccm(Yy_mcdts2, Yx_mcdts2; w = w)

    cnt +=1
end

tt = Vector(100:100:5000)

figure(figsize=(10,5))
subplot(121)
plot(tt, rho_y, label="ρ_y")
plot(tt, rho_x, label="ρ_x")
title("x → y & y → x (based on x-Cao-embedding)")
ylabel("CCM-correlation")
xlabel("time series length")
ylim([0, 0.9])
legend()
grid()

subplot(122)
plot(tt, rho_y2, label="ρ_y2")
plot(tt, rho_x2, label="ρ_x2")
title("x → y & y → x (based on y-Cao-embedding)")
ylabel("CCM-correlation")
xlabel("time series length")
ylim([0, 0.9])
legend()
grid()

figure(figsize=(10,5))
subplot(121)
plot(tt, rho_pec_y, label="ρ_y PEC")
plot(tt, rho_pec_x, label="ρ_x PEC")
title("x → y & y → x (based on x-PECUZAL-embedding)")
ylabel("CCM-correlation")
xlabel("time series length")
ylim([0, 0.9])
legend()
grid()

subplot(122)
plot(tt, rho_pec_y2, label="ρ_y2 PEC")
plot(tt, rho_pec_x2, label="ρ_x2 PEC")
title("x → y & y → x (based on y-PECUZAL-embedding)")
ylabel("CCM-correlation")
xlabel("time series length")
ylim([0, 0.9])
legend()
grid()

figure(figsize=(10,5))
subplot(121)
plot(tt, rho_mcdts_y, label="ρ_y MCDTS")
plot(tt, rho_mcdts_x, label="ρ_x MCDTS")
title("x → y & y → x (based on x-MCDTS-embedding)")
ylabel("CCM-correlation")
xlabel("time series length")
ylim([0, 0.9])
legend()
grid()

subplot(122)
plot(tt, rho_mcdts_y2, label="ρ_y2 MCDTS")
plot(tt, rho_mcdts_x2, label="ρ_x2 MCDTS")
title("x → y & y → x (based on y-MCDTS-embedding)")
ylabel("CCM-correlation")
xlabel("time series length")
ylim([0, 0.9])
legend()
grid()

i = 100


xx = s1[1:i]
yy = s2[1:i]
Yx = genembed(xx,[0,6,12,18,24,30])
Yy = genembed(yy,[0,6,12,18,24,30])

Yx2 = genembed(xx,[0,1,30,2])
Yy2 = genembed(yy,[0,1,30,2])

if i == 100
    w = 5
else
    w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
    w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")
    w = maximum([w1,w2])
end
println(w)
rho_y[cnt], _ = MCDTS.ccm(Yx,Yy; w = w)
rho_x[cnt], _ = MCDTS.ccm(Yy,Yx; w = w)

rho_y2[cnt], _ = MCDTS.ccm(Yx2,Yy2; w = w)
rho_x2[cnt], _ = MCDTS.ccm(Yy2,Yx2; w = w)

using MCDTS


trial1 = genembed(s1, -τ_mcdts)
trial2 = genembed(s2, -τ_mcdts)
rho_mcdts, _ = MCDTS.ccm(trial1, trial2; w = w1)


f9790df7707bca163e43a2488ece4255d66787fc3ca967aeb5bf2c466f555613


d1 = readdlm("results_analysis_CCM_full_combustion_1_Pearson.csv")
d1_old = readdlm("./application/Causality/Combustion/results/results_analysis_CCM_full_combustion_1_Pearson.csv")

dx_cao = readdlm("results_analysis_CCM_full_combustion_1_x1_cao.csv")
dy_cao = readdlm("results_analysis_CCM_full_combustion_1_y1_cao.csv")
dx_cao_old = readdlm("./application/Causality/Combustion/results/results_analysis_CCM_full_combustion_1_x1_cao.csv")
dy_cao_old = readdlm("./application/Causality/Combustion/results/results_analysis_CCM_full_combustion_1_y2_cao.csv")

dx_pec = readdlm("results_analysis_CCM_full_combustion_1_x1_pec.csv")
dy_pec = readdlm("results_analysis_CCM_full_combustion_1_y1_pec.csv")
dx_pec_old = readdlm("./application/Causality/Combustion/results/results_analysis_CCM_full_combustion_1_x1_pec.csv")
dy_pec_old = readdlm("./application/Causality/Combustion/results/results_analysis_CCM_full_combustion_1_y2_pec.csv")

dx_mcdts = readdlm("results_analysis_CCM_full_combustion_1_x1_mcdts.csv")
dy_mcdts = readdlm("results_analysis_CCM_full_combustion_1_y1_mcdts.csv")
dx_mcdts_old = readdlm("./application/Causality/Combustion/results/results_analysis_CCM_full_combustion_1_x1_mcdts.csv")
dy_mcdts_old = readdlm("./application/Causality/Combustion/results/results_analysis_CCM_full_combustion_1_y2_mcdts.csv")

figure()
subplot(231)
plot(dx_cao)
plot(dy_cao)
plot(d1, "r-")
ylim([0, 0.8])
grid()
legend(["X cao", "Y_cao", "Pearson"])

subplot(232)
plot(dx_pec)
plot(dy_pec)
plot(d1, "r-")
ylim([0, 0.8])
grid()
legend(["X pec", "Y_pec", "Pearson"])

subplot(233)
plot(dx_mcdts)
plot(dy_mcdts)
plot(d1, "r-")
ylim([0, 0.8])
grid()
legend(["X mcdts", "Y_mcdts", "Pearson"])

subplot(234)
plot(dx_cao_old)
plot(dy_cao_old)
plot(d1_old, "r-")
ylim([0, 0.8])
grid()
legend(["X cao_old", "Y_cao_old", "Pearson"])

subplot(235)
plot(dx_pec_old)
plot(dy_pec_old)
plot(d1_old, "r-")
ylim([0, 0.8])
grid()
legend(["X pec_old", "Y_pec_old", "Pearson"])

subplot(236)
plot(dx_mcdts_old)
plot(dy_mcdts_old)
plot(d1_old, "r-")
ylim([0, 0.8])
grid()
legend(["X mcdts_old", "Y_mcdts_old", "Pearson"])
