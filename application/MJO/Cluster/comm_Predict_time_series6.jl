# for use on the Cluster

using MCDTS
using DynamicalSystems
using DelayEmbeddings
using Statistics
using DelimitedFiles

RMM1 = readdlm("RMM1_cut.csv")
RMM1 = [i for i in RMM1]
RMM2 = readdlm("RMM2_interpolated.csv")
RMM2 = [i for i in RMM2]


# Make Reconstructions & Predictions:

# Parameters analysis:
T_steps = 200
dmax = 12   # maximum dimension for traditional tde
trials1 = 80 # trials for MCDTS univariate
trials2 = 200 # trials for MCDTS multivariate
taus1 = 0:100 # possible delays
taus2 = 0:50 # possible delays for PRED optimization
max_depth = 15 # depth of the tree
Tw = 1  # time horizon for PRED
KK = 1 # considered nearest neighbors for PRED


# time series binding
x1 = RMM1[1:end-T_steps]       # training
x2 = RMM1[end-T_steps+1:end]   # prediction
y1 = RMM2[1:end-T_steps]       # training
y2 = RMM2[end-T_steps+1:end]   # prediction

data_sample = Dataset(x1,y1) # mutlivariate set

w1 = DelayEmbeddings.estimate_delay(x1, "mi_min")

σ₂ = sqrt(var(x2[1:T_steps]))   # rms deviation for normalization

# The different reconstruction/prediction methods
methods = ["Cao", "Kennel", "Hegger", "PECUZAL", "PECUZAL mult.", "MCDTS L",
            "MCDTS L mult.", "MCDTS FNN", "MCDTS FNN mult.", "MCDTS PRED",
            "MCDTS PRED mult.", "MCDTS PRED KL", "MCDTS PRED KL mult.",
            "MCDTS PRED-L KL", "MCDTS PRED-L KL mult."]


# preallocation
prediction_zeroth = zeros(T_steps,length(methods))
prediction_linear = zeros(T_steps,length(methods))
MSEs_zeroth = zeros(T_steps,length(methods))
MSEs_linear = zeros(T_steps,length(methods))

# cao
# println("Cao")
# 𝒟, τ_tde1, _ = optimal_traditional_de(x1, "afnn"; dmax = dmax, w = w1)
# optimal_d_tde1 = size(𝒟, 2)
# τ_cao = [(i-1)*τ_tde1 for i = 1:optimal_d_tde1]
# Y = genembed(x1, τ_cao .* (-1))
# prediction_zeroth[:,1] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# prediction_linear[:,1] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# for j = 1:T_steps
#     MSEs_zeroth[j,1] = MCDTS.compute_mse(prediction_zeroth[1:j,1], x2[1:j]) / σ₂
#     MSEs_linear[j,1] = MCDTS.compute_mse(prediction_linear[1:j,1], x2[1:j]) / σ₂
# end
#
# # kennel's method
# println("Kennel")
# 𝒟, τ_tde2, _ = optimal_traditional_de(x1, "fnn"; dmax = dmax, w = w1)
# optimal_d_tde2 = size(𝒟, 2)
# τ_kennel = [(i-1)*τ_tde2 for i = 1:optimal_d_tde2]
# Y = genembed(x1, τ_kennel .* (-1))
# prediction_zeroth[:,2] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# prediction_linear[:,2] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# for j = 1:T_steps
#     MSEs_zeroth[j,2] = MCDTS.compute_mse(prediction_zeroth[1:j,2], x2[1:j]) / σ₂
#     MSEs_linear[j,2] = MCDTS.compute_mse(prediction_linear[1:j,2], x2[1:j]) / σ₂
# end
#
# # hegger's method
# println("Hegger")
# 𝒟, τ_tde3, _ = optimal_traditional_de(x1, "ifnn"; dmax = dmax, w = w1)
# optimal_d_tde3 = size(𝒟, 2)
# τ_hegger = [(i-1)*τ_tde3 for i = 1:optimal_d_tde3]
# Y = genembed(x1, τ_hegger .* (-1))
# prediction_zeroth[:,3] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# prediction_linear[:,3] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# for j = 1:T_steps
#     MSEs_zeroth[j,3] = MCDTS.compute_mse(prediction_zeroth[1:j,3], x2[1:j]) / σ₂
#     MSEs_linear[j,3] = MCDTS.compute_mse(prediction_linear[1:j,3], x2[1:j]) / σ₂
# end
#
# writedlm("results_MJO_taus_cao.csv", τ_cao)
# writedlm("results_MJO_taus_kennel.csv", τ_kennel)
# writedlm("results_MJO_taus_hegger.csv", τ_hegger)
#
# # pecuzal
# println("PECUZAL")
# 𝒟, τ_pec, _, L, _ = pecuzal_embedding(x1; τs = taus1, w = w1)
# Y = genembed(x1, τ_pec .* (-1))
# prediction_zeroth[:,4] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# prediction_linear[:,4] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# for j = 1:T_steps
#     MSEs_zeroth[j,4] = MCDTS.compute_mse(prediction_zeroth[1:j,4], x2[1:j]) / σ₂
#     MSEs_linear[j,4] = MCDTS.compute_mse(prediction_linear[1:j,4], x2[1:j]) / σ₂
# end
#
# writedlm("results_MJO_taus_pec.csv", τ_pec)
#
# 𝒟, τ_pec2, ts_pec2, L, _ = pecuzal_embedding(data_sample; τs = taus1, w = w1)
# Y = genembed(data_sample, τ_pec2 .* (-1), ts_pec2)
# if sum(ts_pec2 .== 1)>0
#     tts = findall(x -> x==1, ts_pec2)[1]
# else
#     tts = ts_pec2[1]
# end
# prediction_zeroth[:,5] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
# prediction_linear[:,5] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
# for j = 1:T_steps
#     MSEs_zeroth[j,5] = MCDTS.compute_mse(prediction_zeroth[1:j,5], x2[1:j]) / σ₂
#     MSEs_linear[j,5] = MCDTS.compute_mse(prediction_linear[1:j,5], x2[1:j]) / σ₂
# end
#
# writedlm("results_MJO_taus_pec_multi.csv", τ_pec2)
# writedlm("results_MJO_ts_pec_multi.csv", ts_pec2)


# # mcdts L
# println("MCDTS L")
# tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials1; tws = 2:taus1[end], verbose=true)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts = best_node.τs
# Y = genembed(x1, τ_mcdts .* (-1))
# prediction_zeroth[:,6] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# prediction_linear[:,6] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# for j = 1:T_steps
#     MSEs_zeroth[j,6] = MCDTS.compute_mse(prediction_zeroth[1:j,6], x2[1:j]) / σ₂
#     MSEs_linear[j,6] = MCDTS.compute_mse(prediction_linear[1:j,6], x2[1:j]) / σ₂
# end
#
# writedlm("results_MJO_taus_mcdts_L.csv", τ_mcdts)


# tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials2; tws = 2:taus1[end], verbose=true)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts2 = best_node.τs
# ts_mcdts2 = best_node.ts
# Y = genembed(data_sample, τ_mcdts2 .* (-1), ts_mcdts2)
# if sum(ts_mcdts2 .== 1)>0
#     tts = findall(x -> x==1, ts_mcdts2)[1]
# else
#     tts = ts_mcdts2[1]
# end
# prediction_zeroth[:,7] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
# prediction_linear[:,7] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
# for j = 1:T_steps
#     MSEs_zeroth[j,7] = MCDTS.compute_mse(prediction_zeroth[1:j,7], x2[1:j]) / σ₂
#     MSEs_linear[j,7] = MCDTS.compute_mse(prediction_linear[1:j,7], x2[1:j]) / σ₂
# end
#
# writedlm("results_MJO_taus_mcdts_L_multi.csv", τ_mcdts2)
# writedlm("results_MJO_ts_mcdts_L_multi.csv", ts_mcdts2)

#
# # mcdts FNN
# println("MCDTS FNN")
# tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials1; FNN=true, threshold = 0.01, verbose=true)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_fnn = best_node.τs
# Y = genembed(x1, τ_mcdts_fnn .* (-1))
# prediction_zeroth[:,8] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# prediction_linear[:,8] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# for j = 1:T_steps
#     MSEs_zeroth[j,8] = MCDTS.compute_mse(prediction_zeroth[1:j,8], x2[1:j]) / σ₂
#     MSEs_linear[j,8] = MCDTS.compute_mse(prediction_linear[1:j,8], x2[1:j]) / σ₂
# end
#
# writedlm("results_MJO_taus_mcdts_fnn.csv", τ_mcdts_fnn)
#
# tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials2; FNN=true, threshold = 0.01, verbose=true)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_fnn2 = best_node.τs
# ts_mcdts_fnn2 = best_node.ts
# Y = genembed(data_sample, τ_mcdts_fnn2 .* (-1), ts_mcdts_fnn2)
# if sum(ts_mcdts_fnn2 .== 1)>0
#     tts = findall(x -> x==1, ts_mcdts_fnn2)[1]
# else
#     tts = ts_mcdts_fnn2[1]
# end
# prediction_zeroth[:,9] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
# prediction_linear[:,9] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
# for j = 1:T_steps
#     MSEs_zeroth[j,9] = MCDTS.compute_mse(prediction_zeroth[1:j,9], x2[1:j]) / σ₂
#     MSEs_linear[j,9] = MCDTS.compute_mse(prediction_linear[1:j,9], x2[1:j]) / σ₂
# end
#
#
# writedlm("results_MJO_taus_mcdts_fnn_multi.csv", τ_mcdts_fnn2)
# writedlm("results_MJO_ts_mcdts_fnn_multi.csv", ts_mcdts_fnn2)



# # mcdts PRED MSE
# println("MCDTS PRED MSE")
# tree = MCDTS.mc_delay(Dataset(x1),w1,(L)->(MCDTS.softmaxL(L,β=2.)),
#     taus2, trials1; max_depth = max_depth, PRED = true, KNN = KK,
#     threshold = 5e-6, verbose=true)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_PRED = best_node.τs
# Y = genembed(x1, τ_mcdts_PRED .*(-1))
# prediction_zeroth[:,10] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# prediction_linear[:,10] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# for j = 1:T_steps
#     MSEs_zeroth[j,10] = MCDTS.compute_mse(prediction_zeroth[1:j,10], x2[1:j]) / σ₂
#     MSEs_linear[j,10] = MCDTS.compute_mse(prediction_linear[1:j,10], x2[1:j]) / σ₂
# end
#
# writedlm("results_MJO_taus_mcdts_PRED_MSE.csv", τ_mcdts_PRED)
#
# tree = MCDTS.mc_delay(data_sample,w1,(L)->(MCDTS.softmaxL(L,β=2.)),
#     taus2, trials2; max_depth = max_depth, PRED = true, KNN = KK,
#     threshold = 5e-6, verbose=true)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_PRED_multi = best_node.τs
# ts_mcdts_PRED_multi = best_node.ts
# Y = genembed(data_sample, τ_mcdts_PRED_multi .*(-1), ts_mcdts_PRED_multi)
# if sum(ts_mcdts_PRED_multi .== 1)>0
#     tts = findall(x -> x==1, ts_mcdts_PRED_multi)[1]
# else
#     tts = ts_mcdts_PRED_multi[1]
# end
# prediction_zeroth[:,11] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
# prediction_linear[:,11] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
# for j = 1:T_steps
#     MSEs_zeroth[j,11] = MCDTS.compute_mse(prediction_zeroth[1:j,11], x2[1:j]) / σ₂
#     MSEs_linear[j,11] = MCDTS.compute_mse(prediction_linear[1:j,11], x2[1:j]) / σ₂
# end
#
# writedlm("results_MJO_taus_mcdts_PRED_MSE_multi.csv", τ_mcdts_PRED_multi)
# writedlm("results_MJO_ts_mcdts_PRED_MSE_multi.csv", ts_mcdts_PRED_multi)



# mcdts PRED KL
println("MCDTS PRED KL")
tree = MCDTS.mc_delay(Dataset(x1),w1,(L)->(MCDTS.softmaxL(L,β=2.)),
    taus2, trials1; max_depth = max_depth, PRED = true, KNN = KK,
    PRED_KL = true, verbose=true)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_PRED_KL = best_node.τs
Y = genembed(x1, τ_mcdts_PRED_KL .*(-1))
prediction_zeroth[:,12] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,1]
prediction_linear[:,12] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,1]
for j = 1:T_steps
    MSEs_zeroth[j,12] = MCDTS.compute_mse(prediction_zeroth[1:j,12], x2[1:j]) / σ₂
    MSEs_linear[j,12] = MCDTS.compute_mse(prediction_linear[1:j,12], x2[1:j]) / σ₂
end

writedlm("results_MJO_taus_mcdts_PRED_KL.csv", τ_mcdts_PRED_KL)

tree = MCDTS.mc_delay(data_sample,w1,(L)->(MCDTS.softmaxL(L,β=2.)),
    taus2, trials2; max_depth = max_depth, PRED = true, KNN = KK,
    PRED_KL = true, verbose=true)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_PRED_KL_multi = best_node.τs
ts_mcdts_PRED_KL_multi = best_node.ts
Y = genembed(data_sample, τ_mcdts_PRED_KL_multi .*(-1), ts_mcdts_PRED_KL_multi)
if sum(ts_mcdts_PRED_KL_multi .== 1)>0
    tts = findall(x -> x==1, ts_mcdts_PRED_KL_multi)[1]
else
    tts = ts_mcdts_PRED_KL_multi[1]
end
prediction_zeroth[:,13] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
prediction_linear[:,13] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
for j = 1:T_steps
    MSEs_zeroth[j,13] = MCDTS.compute_mse(prediction_zeroth[1:j,13], x2[1:j]) / σ₂
    MSEs_linear[j,13] = MCDTS.compute_mse(prediction_linear[1:j,13], x2[1:j]) / σ₂
end

writedlm("results_MJO_taus_mcdts_PRED_KL_multi.csv", τ_mcdts_PRED_KL_multi)
writedlm("results_MJO_ts_mcdts_PRED_KL_multi.csv", ts_mcdts_PRED_KL_multi)
writedlm("results_MJO_prediction_zeroth6.csv", prediction_zeroth)
writedlm("results_MJO_prediction_linear6.csv", prediction_linear)
writedlm("results_MJO_MSEs_zeroth6.csv", MSEs_zeroth)
writedlm("results_MJO_MSEs_linear6.csv", MSEs_linear)
#
#
# # mcdts PRED-L KL
# println("MCDTS PRED-L KL")
# MSEs_mcdts_PRED_L_KL = zeros(T_steps)
# tree = MCDTS.mc_delay(Dataset(x1), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials1; tws = 2:taus1[end], verbose=true,
#             PRED_L = true, PRED = true, PRED_KL = true)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_PRED_L_KL = best_node.τs
# Y = genembed(x1, τ_mcdts_PRED_L_KL .* (-1))
# prediction_zeroth[:,14] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# prediction_linear[:,14] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,1]
# for j = 1:T_steps
#     MSEs_zeroth[j,14] = MCDTS.compute_mse(prediction_zeroth[1:j,14], x2[1:j]) / σ₂
#     MSEs_linear[j,14] = MCDTS.compute_mse(prediction_linear[1:j,14], x2[1:j]) / σ₂
# end
#
# writedlm("results_MJO_taus_mcdts_PRED_L_KL.csv", τ_mcdts_PRED_L_KL)
#
# MSEs_mcdts2_PRED_L_KL = zeros(T_steps)
# tree = MCDTS.mc_delay(data_sample, w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials2; tws = 2:taus1[end], verbose=true,
#             PRED_L = true, PRED = true, PRED_KL = true)
# best_node = MCDTS.best_embedding(tree)
# τ_mcdts_PRED_L_KL_multi = best_node.τs
# ts_mcdts_PRED_L_KL_multi = best_node.ts
# Y = genembed(x1, τ_mcdts_PRED_L_KL_multi .* (-1), ts_mcdts_PRED_L_KL_multi)
# if sum(ts_mcdts_PRED_L_KL_multi .== 1)>0
#     tts = findall(x -> x==1, ts_mcdts_PRED_L_KL_multi)[1]
# else
#     tts = ts_mcdts_PRED_L_KL_multi[1]
# end
# prediction_zeroth[:,15] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
# prediction_linear[:,15] = MCDTS.iterated_local_linear_prediction(Y, KK, T_steps; theiler = w1)[:,tts]
# for j = 1:T_steps
#     MSEs_zeroth[j,15] = MCDTS.compute_mse(prediction_zeroth[1:j,15], x2[1:j]) / σ₂
#     MSEs_linear[j,15] = MCDTS.compute_mse(prediction_linear[1:j,15], x2[1:j]) / σ₂
# end
#
# writedlm("results_MJO_taus_mcdts_PRED_L_KL_multi.csv", τ_mcdts_PRED_L_KL_multi)
# writedlm("results_MJO_ts_mcdts_PRED_L_KL_multi.csv", ts_mcdts_PRED_L_KL_multi)
#
#
# # save data
# writedlm("results_MJO_prediction_zeroth.csv", prediction_zeroth)
# writedlm("results_MJO_prediction_linear.csv", prediction_linear)
# writedlm("results_MJO_MSEs_zeroth.csv", MSEs_zeroth)
# writedlm("results_MJO_MSEs_linear.csv", MSEs_linear)
