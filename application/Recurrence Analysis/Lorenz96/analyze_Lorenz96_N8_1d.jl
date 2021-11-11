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
# trial 1: N=8; Fs = 3.5:0.004:5, MCDTS (80 trials), ts-length = 5000, dt = 0.1, fixed ic's, L_thres = 0.05
# trial 2: N=8; Fs = 3.5:0.004:5, MCDTS (80 trials), ts-length = 5000, dt = 0.1, fixed ic's, L_thres = 0

trial = 2
# determine the tde method; #1 Cao, #2 Kennel, #3 Hegger
tde = 1

# bind variables
params = readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_params.csv")
N,dt,total,ε,dmax,lmin,trials,taus,Tw,t_idx = params
Fs = readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_Fs.csv")

tau_tde = Int.(readdlm("./application/artificial data/Lorenz96/Results/time delay embedding/trial 1/results_Lorenz96_N_8_1d_tau_tde$tde.csv"))
tau_pecs = readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_tau_pec.csv")
tau_MCDTSs = readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_tau_MCDTS.csv")
# convert taus into right shape in case of PECUZAL and MCDTS
tau_pec = []
tau_MCDTS = []
for i = 1:size(tau_pecs,1)
    index = isa.(tau_pecs[i,:],Number)
    push!(tau_pec, tau_pecs[i,index])
    index = isa.(tau_MCDTSs[i,:],Number)
    push!(tau_MCDTS, tau_MCDTSs[i,index])
end

ts_pecs = readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_ts_pec.csv")
ts_MCDTSs = readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_ts_MCDTS.csv")
ts_pec = []
ts_MCDTS = []
for i = 1:size(ts_pecs,1)
    index = isa.(ts_pecs[i,:],Number)
    push!(ts_pec, ts_pecs[i,index])
    index = isa.(ts_MCDTSs[i,:],Number)
    push!(ts_MCDTS, ts_MCDTSs[i,index])
end

optimal_d_tde = Int.(readdlm("./application/artificial data/Lorenz96/Results/time delay embedding/trial 1/results_Lorenz96_N_8_1d_optimal_d_tde$tde.csv"))
optimal_d_pec = Int.(readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_optimal_d_pec.csv"))
optimal_d_mcdts = Int.(readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_optimal_d_mcdts.csv"))

L_tde = readdlm("./application/artificial data/Lorenz96/Results/time delay embedding/trial 1/results_Lorenz96_N_8_1d_L_tde$tde.csv")
L_pec = readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_L_pec.csv")
L_mcdts = readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_L_mcdts.csv")

RQA_tde = readdlm("./application/artificial data/Lorenz96/Results/time delay embedding/trial 1/results_Lorenz96_N_8_1d_RQA_tde$tde.csv")
RQA_pec = readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_RQA_pec.csv")
RQA_mcdts = readdlm("./application/artificial data/Lorenz96/Results/1 dimensional input/trial $(trial)/results_Lorenz96_N_8_1d_RQA_mcdts.csv")
RQA_ref = readdlm("./application/artificial data/Lorenz96/Results/Reference/results_Lorenz96_N_8_ref_RQA_ref.csv")

##
λs = readdlm("./application/artificial data/Lorenz96/Lyapunov spectrum/Lyaps_Lo96_N_8_4.csv")
λs = λs[1:2:end,:]
pos_Lyap_idx = λs[:,1] .> 10^-3

# find indices where PECUZAL does not perform an embedding
non_embedding_idx = findall(optimal_d_pec.==1)
n_e_i = [non_embedding_idx[i][1] for i in eachindex(non_embedding_idx)]

l_width_vert = 0.1

# figure(figsize=(20,10))
# axis1 = subplot(421)
# plot(Fs, λs)
# ylims1 = axis1.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
# title("Lyaps")
# ylabel("embedding dimension")
# grid()
#
# axis1 = subplot(422)
# plot(Fs, λs)
# ylims1 = axis1.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
# title("Lyaps")
# ylabel("embedding dimension")
# grid()
#
# axis2 = subplot(423)
# plot(Fs, (L_tde[:,1]-L_mcdts))
# ylims2 = axis2.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
# title("TDE")
# ylabel("L_tde - L_mcdts")
# grid()
#
# axis2 = subplot(424)
# plot(Fs, optimal_d_tde[:,1])
# ylims2 = axis2.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
# title("TDE")
# ylabel("embedding dimension")
# grid()
#
# axis3 = subplot(425)
# plot(Fs, (L_pec-L_mcdts))
# plot(Fs[n_e_i], (L_pec[n_e_i]-L_mcdts[n_e_i]),"*")
# ylims3 = axis3.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
# title("PECUZAL")
# ylabel("L_pec - L_mcdts")
# grid()
#
# axis3 = subplot(426)
# plot(Fs, optimal_d_pec)
# plot(Fs[n_e_i], optimal_d_pec[n_e_i],"*")
# ylims3 = axis3.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
# title("PECUZAL")
# ylabel("embedding dimension")
# grid()
#
# axis4 = subplot(427)
# plot(Fs, (L_mcdts-L_mcdts))
# ylims4 = axis4.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
# title("MCDTS")
# ylabel("L_mcdts - L_mcdts")
# grid()
#
# axis4 = subplot(428)
# plot(Fs, optimal_d_mcdts)
# ylims4 = axis4.get_ylim()
# vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
# title("MCDTS")
# ylabel("embedding dimension")
# grid()

RQA_names = ["RR", "TRANS", "DET", "L_mean", "L_max", "DIV", "ENTR", "TREND",
    "LAM", "TT", "V_max", "V_ENTR", "MRT", "RTE", "NMPRT"]


# for RQA_val = 1:2:14
#
#     figure(figsize=(20,10))
#     axis1 = subplot(421)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis1 = subplot(422)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis2 = subplot(423)
#     plot(Fs, RQA_tde[:,RQA_val])
#     ylims2 = axis2.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis2 = subplot(424)
#     plot(Fs, RQA_tde[:,RQA_val+1])
#     ylims2 = axis2.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
#
#     axis3 = subplot(425)
#     plot(Fs, RQA_pec[:,RQA_val])
#     plot(Fs[n_e_i], RQA_pec[n_e_i,RQA_val],"*")
#     ylims3 = axis3.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
#     title("PECUZAL")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis3 = subplot(426)
#     plot(Fs, RQA_pec[:,RQA_val+1])
#     plot(Fs[n_e_i], RQA_pec[n_e_i,RQA_val+1],"*")
#     ylims3 = axis3.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
#     title("PECUZAL")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
#
#     axis4 = subplot(427)
#     plot(Fs, RQA_mcdts[:,RQA_val])
#     ylims4 = axis4.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
#     title("MCDTS")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis4 = subplot(428)
#     plot(Fs, RQA_mcdts[:,RQA_val+1])
#     ylims4 = axis4.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
#     title("MCDTS")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
# end


# for RQA_val = 1:2:14
#
#     figure(figsize=(20,10))
#     axis1 = subplot(421)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis1 = subplot(422)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis2 = subplot(423)
#     plot(Fs, (abs.(RQA_tde[:,RQA_val] .- RQA_ref[:,RQA_val]) ./ RQA_ref[:,RQA_val]))
#     ylims2 = axis2.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE (rel. DIFFS)")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis2 = subplot(424)
#     plot(Fs, (abs.(RQA_tde[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]) ./ RQA_ref[:,RQA_val+1]))
#     ylims3 = axis2.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE (rel. DIFFS)")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
#
#     axis3 = subplot(425)
#     var = (abs.(RQA_pec[:,RQA_val] .- RQA_ref[:,RQA_val]) ./ RQA_ref[:,RQA_val])
#     plot(Fs, var)
#     plot(Fs[n_e_i], var[n_e_i],"*")
#     axis3.set_ylim(ylims2)
#     #ylims3 = axis3.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("PECUZAL (rel. DIFFS)")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis3 = subplot(426)
#     var2 = (abs.(RQA_pec[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]) ./ RQA_ref[:,RQA_val+1])
#     plot(Fs, var2)
#     plot(Fs[n_e_i], var2[n_e_i],"*")
#     axis3.set_ylim(ylims3)
#     #ylims3 = axis3.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
#     title("PECUZAL (rel. DIFFS)")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
#
#     axis4 = subplot(427)
#     plot(Fs, (abs.(RQA_mcdts[:,RQA_val] .- RQA_ref[:,RQA_val]) ./ RQA_ref[:,RQA_val]))
#     axis4.set_ylim(ylims2)
#     #ylims4 = axis4.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("MCDTS (rel. DIFFS)")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis4 = subplot(428)
#     plot(Fs, (abs.(RQA_mcdts[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]) ./ RQA_ref[:,RQA_val+1]))
#     axis4.set_ylim(ylims3)
#     #ylims4 = axis4.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
#     title("MCDTS (rel. DIFFS)")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
# end

# for RQA_val = 1:2:14
#
#     figure(figsize=(20,10))
#     subplots_adjust(hspace=0.4)

#     axis1 = subplot(421)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis1 = subplot(422)
#     plot(Fs, λs)
#     ylims1 = axis1.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
#     title("Lyaps")
#     grid()
#
#     axis2 = subplot(423)
#     plot(Fs, RQA_ref[:,RQA_val], label = "ref")
#     plot(Fs, RQA_tde[:,RQA_val], label = "exp")
#     plot(Fs, RQA_tde[:,RQA_val] .- RQA_ref[:,RQA_val], color = "red",label = "diff")
#     ylims2 = axis2.get_ylim()
#     legend()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("TDE (MSE: $(mean((RQA_tde[:,RQA_val] .- RQA_ref[:,RQA_val]).^2 ./ RQA_ref[:,RQA_val]))")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis2 = subplot(424)
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
#     axis3 = subplot(425)
#     var = RQA_pec[:,RQA_val] .- RQA_ref[:,RQA_val]
#     plot(Fs, RQA_ref[:,RQA_val], label = "ref")
#     plot(Fs, RQA_pec[:,RQA_val], label = "exp")
#     plot(Fs, var, color = "red",label = "diff")
#     plot(Fs[n_e_i], var[n_e_i],"*")
#     legend()
#     axis3.set_ylim(ylims2)
#     #ylims3 = axis3.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
#     title("PECUZAL (MSE: $(mean((RQA_pec[:,RQA_val] .- RQA_ref[:,RQA_val]).^2 ./ RQA_ref[:,RQA_val]))")
#     ylabel(RQA_names[RQA_val])
#     grid()
#
#     axis3 = subplot(426)
#     var2 = RQA_pec[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]
#     plot(Fs, RQA_ref[:,RQA_val+1], label = "ref")
#     plot(Fs, RQA_pec[:,RQA_val+1], label = "exp")
#     plot(Fs, var2, color = "red",label = "diff")
#     plot(Fs[n_e_i], var2[n_e_i],"*")
#     legend()
#     axis3.set_ylim(ylims3)
#     #ylims3 = axis3.get_ylim()
#     vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
#     title("PECUZAL (MSE: $(mean((RQA_pec[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]).^2 ./ RQA_ref[:,RQA_val+1]))")
#     ylabel(RQA_names[RQA_val+1])
#     grid()
#
#     axis4 = subplot(427)
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
#     axis4 = subplot(428)
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

# compute (relative) mean squared errors
cnt = 0
cnt2 = 0
for i = 1:size(RQA_ref,2)
    global cnt
    global cnt2

    RQA_ref_ = deleteat!(RQA_ref[:,i], n_e_i)
    RQA_tde_ = deleteat!(RQA_tde[:,i], n_e_i)
    RQA_pec_ = deleteat!(RQA_pec[:,i], n_e_i)
    RQA_mcdts_ = deleteat!(RQA_mcdts[:,i], n_e_i)

    println("MSE TDE $(RQA_names[i]):  $(mean((RQA_tde_ .- RQA_ref_).^2 ./ RQA_ref_))")
    println("MSE PEC $(RQA_names[i]):  $(mean((RQA_pec_ .- RQA_ref_).^2 ./ RQA_ref_))")
    println("MSE MCDTS $(RQA_names[i]):  $(mean((RQA_mcdts_ .- RQA_ref_).^2 ./ RQA_ref_))")

    if i>1 && mean((RQA_mcdts_ .- RQA_ref_).^2 ./ RQA_ref_) < mean((RQA_tde_ .- RQA_ref_).^2 ./ RQA_ref_)
        cnt2 += 1
    end

    if i>1 && mean((RQA_pec_ .- RQA_ref_).^2 ./ RQA_ref_) < mean((RQA_tde_ .- RQA_ref_).^2 ./ RQA_ref_)
        cnt += 1
    end
    println("*******************")
end
println("PEC: $cnt / $(size(RQA_ref,2)-1)")
println("MCDTS: $cnt2 / $(size(RQA_ref,2)-1)")