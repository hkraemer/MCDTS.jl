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
ρ, Y_hat = MCDTS.ccm(Yx, Yy; w = w1)

yyy, _ = optimal_traditional_de(x, "fnn"; w = w1)

##
d1 = 2
d2 = 2
τ1 = 1
τ2 = 1
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

using MCDTS

test1 = regularize(Dataset(x))
test2 = regularize(Dataset(y))
# try MCDTS with CCM
taus1 = 0:10 # the possible delay vals
trials = 20 # the sampling of the tree
tree = MCDTS.mc_delay(test1, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials;
    verbose=false, CCM = true, Y_CCM = test2)
best_node = MCDTS.best_embedding(tree)
τ_mcdts2_n = best_node.τs
ts_mcdts2_n = best_node.ts


test = Dataset(x)

println(test[1:5])
