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

using PyPlot
pygui(true)


## Lorenz

lo = Systems.lorenz(rand(3))
dt = 0.01
tr = trajectory(lo, 500; dt = dt, Ttr = 10)

data = regularize(tr)

w = DelayEmbeddings.estimate_delay(tr[:,1], "mi_min")
Yx, taus, _, _, _ = pecuzal_embedding(tr[:,1]; w=w)
Yy = genembed(tr[:,2], taus)

figure()
plot3D(Yx[:,1], Yx[:,2], Yx[:,3])

figure()
plot3D(Yy[:,1], Yy[:,2], Yy[:,3])

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
Yx = embed(x,2,1)
Yy = embed(y,2,1)

w1 = DelayEmbeddings.estimate_delay(x, "mi_min")
ρ, Y_hat = MCDTS.ccm(Yx; w = w1)

yyy, _ = optimal_traditional_de(x, "fnn"; w = w1)

##
d1 = 5
d2 = 5
τ1 = 15
τ2 = 15
cnt = 1
rho_x = zeros(length(100:100:3500))
rho_y = zeros(length(100:100:3500))
for i = 100:100:3500
    xx = x[1:i]
    yy = y[1:i]
    Yx = embed(xx,d1,τ1)
    Yy = embed(yy,d2,τ2)

    if i == 100
        w = 5
    else
        w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
        w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")
        w = maximum([w1,w2])
    end

    rho_y[cnt], _ = MCDTS.ccm(Yx,Yy; w = w)
    rho_x[cnt], _ = MCDTS.ccm(Yy,Yx; w = w)

    cnt +=1
end


figure()
plot(100:100:3500,rho_x)
plot(100:100:3500,rho_y)
grid()


lag = 1
figure()
scatter(Yy[1+lag:end,3], Y_hat[1:end-lag,3])

figure()
plot(Yy[1:end-lag,1],"ro",linewidth=2)
plot(Y_hat[1+lag:end,1],"kx", linewidth=2)
grid()

lags = 0:500
corrs = zeros(length(lags))
corrs2 = zeros(length(lags)-1)
for (i,l) in enumerate(lags)
    corrs[i] = Statistics.cor(Yy[1+l:end,1], Y_hat[1:end-l,1])
    if i>1
        corrs2[i-1] = Statistics.cor(Yy[1:end-l,1], Y_hat[1+l:end,1])
    end
end
coss = vcat(corrs,corrs2)

figure()
plot(-500:500,coss)

N = 4000
x = zeros(N)
y = zeros(N)
x[1] = 0.2
y[1] = 0.12
for n = 1:N-1
    x[n+1]= x[n]*(3.8 - 3.8*x[n] - 0.02*y[n])
    y[n+1]= y[n]*(3.5 - 3.5*y[n] - 0.1*x[n])
end

figure()
plot(x)
plot(y)
grid()


w = DelayEmbeddings.estimate_delay(x, "mi_min")
Yx, taus, _, _, _ = pecuzal_embedding(x; w=w)

Yx = embed(x,3,1)
Yy = embed(y,3,1)

Y_hat, idx = MCDTS.ccm(Yx,Yy; w = w)

lag = 0

figure()
plot(Yy[1:end-lag,1],"r",linewidth=2)
plot(Y_hat[1+lag:end,1],"k", linewidth=2)
grid()

corrs = Statistics.cor(Yy[1+l:end,1], Y_hat[1:end-l,1])

x = zeros(10000)
x[1]=0.2
for i = 2:10000
    x[i]=4*x[i-1] * (1-x[i-1])
end

figure()
plot(x)

#x = 0.0000000001 .* randn(10000)

Y_,tau,_,_,_ = pecuzal_embedding(x)

Y = optimal_traditional_de(x)
