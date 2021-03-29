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


delays = 0:100
trials = 10
max_depth = 10
@time tree = MCDTS.mc_delay(Dataset(s[1:10000]),mi,(L)->(MCDTS.softmaxL(L,β=2.)),
    delays, trials; max_depth = max_depth, PRED = true, verbose = true, threshold = 5e-6, linear = false, Tw = 10)
best_node = MCDTS.best_embedding(tree)
FNNS = best_node.L
println(best_node)


Y = DelayEmbeddings.hcat_lagged_values(s,s,mi)

cost_K = zeros(10)
for K = 1:10
    println(K)
    cost_K[K],_ = MCDTS.linear_prediction_cost(Y; K = 6, Tw = K, w = mi)
end
figure()
plot(1:10, cost_K)
grid()
