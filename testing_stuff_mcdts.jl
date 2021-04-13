using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using DelimitedFiles
using ChaosTools
using DSP
using Statistics
using Revise

using PyPlot
pygui(true)


##

lo = Systems.lorenz()
tr = trajectory(lo, 100; dt = 0.01, Ttr = 10)


tr = trajectory(lo, 1000; dt = 0.01, Ttr = 100) # results 3

s = tr[:,1]
mi = DelayEmbeddings.estimate_delay(s, "mi_min")


using MCDTS
delays = 0:30
trials = 5
max_depth = 15
@time tree = MCDTS.mc_delay(tr[1:7500,1:2],mi,(L)->(MCDTS.softmaxL(L,β=2.)),
    delays, trials; KNN = 1, max_depth = max_depth, PRED = true, verbose = true,
    linear = false, Tw = 1, PRED_KL = true)
best_node = MCDTS.best_embedding(tree)
ttaus = best_node.τs
tts = best_node.ts
println(best_node)


Y = genembed(s[1:7500], ttaus .*(-1))

KK=1
T_steps = 1000
Y_forecast = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = mi, verbose=true)

_,taus_pec,_,_,_ = pecuzal_embedding(s[1:7500]; w=mi, τs=0:200)
Y_pec = genembed(s[1:7500], taus_pec .*(-1))
Y2 = genembed(s[1:7500], [0, -2])
Y_forecast2 = MCDTS.iterated_local_linear_prediction(Y2, KK, T_steps; theiler = mi, verbose=true)
Y_forecast3 = MCDTS.iterated_local_linear_prediction(Y_pec, KK, T_steps; theiler = mi, verbose=true)

figure()
plot(1:1200, s[7301:8500], label="true")
plot(201:1200, Y_forecast[:,1], label="1st")
plot(201:1200, Y_forecast2[:,1], label="2nd")
plot(201:1200, Y_forecast3[:,1], label="3rd")
#plot(1:200, Y[end-199:end,1], label="1st")
#plot(1:200, Y2[end-199:end,1], label="2nd")
grid()
legend()

##
Y = genembed(s[1:7500], [0, -2])
ET = eltype(s)
D = size(Y,2)
Tw = 1
samplesize = 1
metric = Euclidean()
K = 1
NN = length(Y)-Tw
NNN = floor(Int, samplesize*NN)
if samplesize < 1
    ns = sample(1:NN, NNN; replace=false) # the fiducial point indices
else
    ns = 1:NN  # the fiducial point indices
end
vs = Y[ns] # the fiducial points in the data set
vtree = KDTree(Y[1:end-Tw], metric)
allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, mi)

KL_distances = zeros(ET, Tw, D)

T = 1
predictions = zeros(ET, NN, D)
# loop over each fiducial point
for (i,v) in enumerate(vs)
    NNidxs = allNNidxs[i] # indices of k nearest neighbors to v

    ϵ_ball = zeros(ET, K, D) # preallocation
    # determine neighborhood one time step ahead
    @inbounds for (k, j) in enumerate(NNidxs)
        ϵ_ball[k, :] .= Y[j + T]
    end
    # take the average as a prediction
    predictions[i,:] = mean(ϵ_ball; dims=1)

end
# compute KL-divergence for each component of the prediction
for j = 1:D
    KL_distances[T,j] = compute_KL_divergence(Vector(predictions[:,j]), Vector(view(Y, ns .+ T)[j]))
end

mean(KL_distances; dims=1)

figure()
plot(predictions[:,1], label="pred")
plot(Y[ns .+ T, 3], label="true")
legend()
grid()
