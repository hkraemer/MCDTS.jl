using Pkg
Pkg.activate(".")
using Revise
using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using PyPlot
pygui(true)
using Random

Random.seed!(1234)
ds = Systems.lorenz()
data = trajectory(ds,200)
data = data[10001:end,:]

w1 = DelayEmbeddings.estimate_delay(data[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(data[:,2],"mi_min")
w3 = DelayEmbeddings.estimate_delay(data[:,3],"mi_min")
w = maximum(hcat(w1,w2,w3))
delays = 0:100
runs = 10
runs2 = 10
T_steps = 100

##


##
Y = data[:,1:2]
Y_trial = data[:,1:3]
Tw = 50
L_decrease = MCDTS.uzal_cost_pecuzal_mcdts(Y[1:length(Y_trial)], Y_trial, Tw; samplesize = 1.)
L_decrease2 = MCDTS.uzal_cost_pecuzal_mcdts(Y[1:length(Y_trial)], Y_trial, Tw; samplesize = .1)

dist, L1, L2 = uzal_cost_pecuzal_mcdts2(Y[1:length(Y_trial)], Y_trial, Tw; samplesize = .1)

figure()
plot(2:50, L1, label="1")
plot(2:50, L2, label="2")
legend()
grid()


##



Random.seed!(1234)
pecuzal = MCDTS.PecuzalOptim()
@time tree = mcdts_embedding(Dataset(data[:,1]), pecuzal, w1, delays, runs)
best_node = MCDTS.best_embedding(tree)

# L with tws
Random.seed!(1234)
tws = 2:4:delays[end]

begin
        runs = 1
        println("Continuity full / L - full")
        optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws), MCDTS.Continuity_function())
        @time tree2 = mcdts_embedding(Dataset(data[:,1]), optmodel2, w1, delays, runs)

        println("Continuity full / L - full")
        optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws), MCDTS.Continuity_function())
        @time tree2 = mcdts_embedding(Dataset(data[:,1]), optmodel2, w1, delays, runs)

        println("Continuity half / L - full")
        optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws), MCDTS.Continuity_function(13,.5,0.05,0.5))
        @time tree2 = mcdts_embedding(Dataset(data[:,1]), optmodel2, w1, delays, runs)

        println("Continuity full / L - half")
        optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws,0.5), MCDTS.Continuity_function())
        @time tree2 = mcdts_embedding(Dataset(data[:,1]), optmodel2, w1, delays, runs)

        println("Continuity half / L - half")
        optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws,0.5), MCDTS.Continuity_function(13,.5,0.05,0.5))
        @time tree2 = mcdts_embedding(Dataset(data[:,1]), optmodel2, w1, delays, runs)

end

dd = MCDTS.L_statistic(0,3,tws,0.5)
optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws,.9), MCDTS.Continuity_function())
@time tree2 = mcdts_embedding(Dataset(data[:,1]), optmodel2, w1, delays, runs)

optmodel2 = MCDTS.MCDTSOptimGoal(MCDTS.L_statistic(0,3,tws), MCDTS.Continuity_function())
@time tree2 = mcdts_embedding(Dataset(data[:,1]), optmodel2, w1, delays, runs)
best_node2 = MCDTS.best_embedding(tree2)
@test best_node2.τs == best_node.τs
@test best_node2.L > best_node.L
