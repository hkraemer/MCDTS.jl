using Pkg
Pkg.activate(".")
using MCDTS
using DynamicalSystemsBase
using PyPlot
pygui(true)

lo = Systems.lorenz()
tr = trajectory(lo, 500; dt = 0.01, Ttr = 10)

s = tr[1:4000,1]

dds = 0:100
ee = MCDTS.pecora(s, (0,18), (1,1); w = 17, delays = dds)

figure()
plot(dds, ee)
grid()

ee2 = MCDTS.pecora(s, (0,18), (1,1); w = 17, delays = dds, PRED=true)

figure()
plot(dds, ee2)
grid()


τs = (0,2)
js = (1,1)

using DelayEmbeddings
using Distances
using StatsBase
using Neighborhood

metric = Euclidean()
delays = dds
vspace = DelayEmbeddings.genembed(s, τs, js)
vtree = KDTree(vspace.data[1:end-maximum(abs.(delays))], metric)

vspace = MCDTS.genembed_for_prediction(s, τs, js)
# indices of random fiducial points (with valid time range w.r.t. T)
ns = vec(1:(length(vspace)-maximum(abs.(delays))))


s = collect(1:100)
τs = [0, 18, 4]
js = [1, 1, 1]
vspace1 = genembed(s, τs.*(-1), js) # takes positive τs's and converts internally
vspace2 = MCDTS.genembed_for_prediction(s, τs, js)



τ_vals = [0, 18, 9]
ts_vals = [1,1,1]
τs = delays
w = w1
metric = Euclidean()


ε★, _ = DelayEmbeddings.pecora(Dataset(data[:,1]), Tuple(τ_vals), Tuple(ts_vals); delays = τs, w = w,
        metric = metric)

using PyPlot
pygui(true)

figure()
plot(delays,ε★)
