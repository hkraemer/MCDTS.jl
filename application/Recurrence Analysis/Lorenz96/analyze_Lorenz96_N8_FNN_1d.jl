## Here we look at the results gained from the computations made in the scripts
# stored in the `cluster scripts`-folder:
# For a range of `F`-parameter values (F=2.5:0.1:6) we analyzed embeddings
# obtained from traditional time delay embedding, from PECUZAL and from MCDTS, by
# computing RQA-quantifiers.

using PyPlot
pygui(true)
using DelimitedFiles
using Statistics

# determine from which trial you want to process the data
# trial 1: N=8; Fs = 3.5:0.004:5, MCDTS (80 trials), ts-length = 5000, dt = 0.1, fixed ic's, L_thres = 0

N = 8
trial = 1
# determine the tde method; #1 Cao, #2 Kennel, #3 Hegger
tde = 1

# bind variables
params = readdlm("./application/Recurrence Analysis/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_$(N)_FNN_1d_params.csv")
N,dt,total,ε,dmax,lmin,trials,taus,Tw,t_idx = params
Fs = readdlm("./application/Recurrence Analysis/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_$(N)_FNN_1d_Fs.csv")

tau_tde = Int.(readdlm("./application/Recurrence Analysis/Lorenz96/Results/time delay embedding/trial 1/results_Lorenz96_N_8_1d_tau_tde$tde.csv"))
tau_MCDTSs = readdlm("./application/Recurrence Analysis/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_$(N)_FNN_1d_tau_MCDTS.csv")
# convert taus into right shape in case of PECUZAL and MCDTS
tau_MCDTS = []
for i = 1:size(tau_MCDTS,1)
    index = isa.(tau_MCDTSs[i,:],Number)
    push!(tau_MCDTS, tau_MCDTSs[i,index])
end

ts_MCDTSs = readdlm("./application/Recurrence Analysis/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_$(N)_FNN_1d_ts_MCDTS.csv")
ts_MCDTS = []
for i = 1:size(ts_MCDTS,1)
    index = isa.(ts_MCDTSs[i,:],Number)
    push!(ts_MCDTS, ts_MCDTSs[i,index])
end

optimal_d_tde = Int.(readdlm("./application/Recurrence Analysis/Lorenz96/Results/time delay embedding/trial 1/results_Lorenz96_N_8_1d_optimal_d_tde$tde.csv"))
optimal_d_mcdts = Int.(readdlm("./application/Recurrence Analysis/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_$(N)_FNN_1d_optimal_d_mcdts.csv"))

L_tde = readdlm("./application/Recurrence Analysis/Lorenz96/Results/time delay embedding/trial 1/results_Lorenz96_N_8_1d_L_tde$tde.csv")
L_mcdts = readdlm("./application/Recurrence Analysis/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_$(N)_FNN_1d_FNN_mcdts.csv")

RQA_tde = readdlm("./application/Recurrence Analysis/Lorenz96/Results/time delay embedding/trial 1/results_Lorenz96_N_8_1d_RQA_tde$tde.csv")
RQA_mcdts = readdlm("./application/Recurrence Analysis/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_$(N)_FNN_1d_RQA_mcdts.csv")
RQA_ref = readdlm("./application/Recurrence Analysis/Lorenz96/Results/Reference/results_Lorenz96_N_$(N)_ref_RQA_ref.csv")

##
λs = readdlm("./application/Recurrence Analysis/Lorenz96/Lyapunov spectrum/Lyaps_Lo96_N_8_4.csv")
λs = λs[1:2:end,:]
pos_Lyap_idx = λs[:,1] .> 10^-3

non_embedding_idx = findall(optimal_d_mcdts.==1)
n_e_i = [non_embedding_idx[i][1] for i in eachindex(non_embedding_idx)]

l_width_vert = 0.1

# figure(figsize=(20,10))
# axis1 = subplot(321)
# plot(Fs, λs)
# ylims1 = axis1.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
# title("Lyaps")
# ylabel("embedding dimension")
# grid()
#
# axis1 = subplot(322)
# plot(Fs, λs)
# ylims1 = axis1.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
# title("Lyaps")
# ylabel("embedding dimension")
# grid()
#
# axis2 = subplot(323)
# plot(Fs, (L_tde[:,1]-L_mcdts))
# ylims2 = axis2.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
# title("TDE (FNN)")
# ylabel("L_tde - L_mcdts")
# grid()
#
# axis2 = subplot(324)
# plot(Fs, optimal_d_tde[:,1])
# ylims2 = axis2.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
# title("TDE (FNN)")
# ylabel("embedding dimension")
# grid()
#
# axis4 = subplot(325)
# plot(Fs, (L_mcdts-L_mcdts))
# ylims4 = axis4.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
# title("MCDTS (FNN)")
# ylabel("L_mcdts - L_mcdts")
# grid()
#
# axis4 = subplot(326)
# plot(Fs, optimal_d_mcdts)
# ylims4 = axis4.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
# title("MCDTS (FNN)")
# ylabel("embedding dimension")
# grid()

RQA_names = ["RR", "TRANS", "DET", "L_mean", "L_max", "DIV", "ENTR", "TREND",
    "LAM", "TT", "V_max", "V_ENTR", "MRT", "RTE", "NMPRT"]


# for RQA_val = 1:2:14
#
#     figure(figsize=(20,10))
#     axis1 = subplot(321)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis1 = subplot(322)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis2 = subplot(323)
#     plot(Fs, RQA_tde[:,RQA_val])
#     ylims2 = axis2.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE (FNN)")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis2 = subplot(324)
#     plot(Fs, RQA_tde[:,RQA_val+1])
#     ylims2 = axis2.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE (FNN)")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
#
#     axis4 = subplot(325)
#     plot(Fs, RQA_mcdts[:,RQA_val])
#     ylims4 = axis4.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
#     title("MCDTS (FNN)")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis4 = subplot(326)
#     plot(Fs, RQA_mcdts[:,RQA_val+1])
#     ylims4 = axis4.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
#     title("MCDTS (FNN)")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
# end
#
#
# for RQA_val = 1:2:14
#
#     figure(figsize=(20,10))
#     axis1 = subplot(321)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis1 = subplot(322)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis2 = subplot(323)
#     plot(Fs, (abs.(RQA_tde[:,RQA_val] .- RQA_ref[:,RQA_val]) ./ RQA_ref[:,RQA_val]))
#     ylims2 = axis2.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE (rel. DIFFS) (FNN)")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis2 = subplot(324)
#     plot(Fs, (abs.(RQA_tde[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]) ./ RQA_ref[:,RQA_val+1]))
#     ylims3 = axis2.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE (rel. DIFFS) (FNN)")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
#
#     axis4 = subplot(325)
#     plot(Fs, (abs.(RQA_mcdts[:,RQA_val] .- RQA_ref[:,RQA_val]) ./ RQA_ref[:,RQA_val]))
#     axis4.set_ylim(ylims2)
#     #ylims4 = axis4.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("MCDTS (rel. DIFFS) (FNN)")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis4 = subplot(326)
#     plot(Fs, (abs.(RQA_mcdts[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]) ./ RQA_ref[:,RQA_val+1]))
#     axis4.set_ylim(ylims3)
#     #ylims4 = axis4.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
#     title("MCDTS (rel. DIFFS) (FNN)")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
# end

# for RQA_val = 1:2:14
#
#     figure(figsize=(20,10))
#     axis1 = subplot(321)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis1 = subplot(322)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis2 = subplot(323)
#     plot(Fs, RQA_ref[:,RQA_val], label = "ref")
#     plot(Fs, RQA_tde[:,RQA_val], label = "exp")
#     plot(Fs, RQA_tde[:,RQA_val] .- RQA_ref[:,RQA_val], color = "red",label = "diff")
#     ylims2 = axis2.get_ylim()
#     legend()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE (MSE: $(mean((RQA_tde[:,RQA_val] .- RQA_ref[:,RQA_val]).^2 ./RQA_ref[:,RQA_val]))")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis2 = subplot(324)
#     plot(Fs, RQA_ref[:,RQA_val+1], label = "ref")
#     plot(Fs, RQA_tde[:,RQA_val+1], label = "exp")
#     plot(Fs, RQA_tde[:,RQA_val+1] .- RQA_ref[:,RQA_val+1], color = "red",label = "diff")
#     ylims3 = axis2.get_ylim()
#     legend()
#     vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE (MSE: $(mean((RQA_tde[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]).^2 ./ RQA_ref[:,RQA_val+1]))")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
#
#     axis4 = subplot(325)
#     var = RQA_mcdts[:,RQA_val] .- RQA_ref[:,RQA_val]
#     plot(Fs, RQA_ref[:,RQA_val], label = "ref")
#     plot(Fs, RQA_mcdts[:,RQA_val], label = "exp")
#     plot(Fs, var, color = "red",label = "diff")
#     legend()
#     axis4.set_ylim(ylims2)
#     #ylims4 = axis4.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("MCDTS (MSE: $(mean((RQA_mcdts[:,RQA_val] .- RQA_ref[:,RQA_val]).^2 ./ RQA_ref[:,RQA_val]))")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis4 = subplot(326)
#     var2 = RQA_mcdts[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]
#     plot(Fs, RQA_ref[:,RQA_val+1], label = "ref")
#     plot(Fs, RQA_mcdts[:,RQA_val+1], label = "exp")
#     plot(Fs, var2, color = "red", label = "diff")
#     legend()
#     axis4.set_ylim(ylims3)
#     #ylims4 = axis4.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
#     title("MCDTS (MSE: $(mean((RQA_mcdts[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]).^2 ./ RQA_ref[:,RQA_val+1]))")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
# end


# compute mean squared errors
cnt = 0
for i = 1:size(RQA_ref,2)
    global cnt
    println("MSE TDE $(RQA_names[i]):  $(mean((RQA_tde[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]))")
    println("MSE MCDTS $(RQA_names[i]):  $(mean((RQA_mcdts[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]))")

    if i>1 && mean((RQA_mcdts[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]) < mean((RQA_tde[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i])
        cnt += 1
    end
    println("*******************")
end
println("$cnt / $(size(RQA_ref,2)-1)")
