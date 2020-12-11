## Here we look at the results gained from the computations made in the scripts
# stored in the `cluster scripts`-folder:
# For a range of `F`-parameter values (F=2.5:0.1:6) we analyzed embeddings
# obtained from traditional time delay embedding, from PECUZAL and from MCDTS, by
# computing RQA-quantifiers.

using PyPlot
pygui(true)
using DelimitedFiles

# determine from which trial you want to process the data
# trial 1: N=8; Fs = 3.5:0.002:5, MCDTS (80 trials), ts-length = 12500, dt = 0.01, Tw = 100
# trial 2: N=8; Fs = 3.5:0.002:5, MCDTS (80 trials), ts-length = 2500, dt = 0.1, Tw = 100
# trial 3: N=8; Fs = 3.5:0.002:5, MCDTS (80 trials), ts-length = 5000, dt = 0.1, Tw = 100
# trial 4: N=8; Fs = 3.5:0.002:5, MCDTS (80 trials), ts-length = 5000, dt = 0.1, Tw = 4*tau_tde
# trial 5: N=8; Fs = 3.5:0.002:5, MCDTS (80 trials), ts-length = 5000, dt = 0.1, Tw = 4*tau_tde, fixed ic's
# trial 6: N=8; Fs = 3.5:0.002:5, MCDTS (80 trials), ts-length = 5000, dt = 0.1, Tw = tau_tde, fixed ic's

trial = 4
ts_trial = 1

# bind variables
params = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_params.csv")
N,dt,total,ε,dmax,lmin,trials,taus,Tw,t_idx = params
Fs = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_Fs.csv")

tau_tde = Int.(readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_tau_tde.csv"))
tau_pecs = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_tau_pec.csv")
tau_MCDTSs = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_tau_MCDTS.csv")
# convert taus into right shape in case of PECUZAL and MCDTS
tau_pec = []
tau_MCDTS = []
for i = 1:size(tau_pecs,1)
    index = isa.(tau_pecs[i,:],Number)
    push!(tau_pec, tau_pecs[i,index])
    index = isa.(tau_MCDTSs[i,:],Number)
    push!(tau_MCDTS, tau_MCDTSs[i,index])
end

ts_pecs = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_ts_pec.csv")
ts_MCDTSs = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_ts_MCDTS.csv")
ts_pec = []
ts_MCDTS = []
for i = 1:size(ts_pecs,1)
    index = isa.(ts_pecs[i,:],Number)
    push!(ts_pec, ts_pecs[i,index])
    index = isa.(ts_MCDTSs[i,:],Number)
    push!(ts_MCDTS, ts_MCDTSs[i,index])
end

optimal_d_tde = Int.(readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_optimal_d_tde.csv"))
optimal_d_pec = Int.(readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_optimal_d_pec.csv"))
optimal_d_mcdts = Int.(readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_optimal_d_mcdts.csv"))

L_tde = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_L_tde.csv")
L_pec = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_L_pec.csv")
L_mcdts = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_L_mcdts.csv")

RQA_tde = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_RQA_tde.csv")
RQA_pec = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_RQA_pec.csv")
RQA_mcdts = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/results_Lorenz96_N_8_3d_RQA_mcdts.csv")

##
λs = readdlm("./application/artificial data/Lorenz96/Results/3 dimensional input/trial $(trial)/Lyaps_Lo96_N_8.csv")

pos_Lyap_idx = λs[:,1] .> 10^-3

non_embedding_idx = findall(optimal_d_pec.==1)
n_e_i = [non_embedding_idx[i][1] for i in eachindex(non_embedding_idx)]

l_width_vert = 0.1

figure(figsize=(20,10))
axis1 = subplot(421)
plot(Fs, λs)
ylims1 = axis1.get_ylim()
vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
title("Lyaps")
ylabel("embedding dimension")
grid()

axis1 = subplot(422)
plot(Fs, λs)
ylims1 = axis1.get_ylim()
vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
title("Lyaps")
ylabel("embedding dimension")
grid()

axis2 = subplot(423)
plot(Fs, (L_tde[:,ts_trial]-L_mcdts))
ylims2 = axis2.get_ylim()
vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
title("TDE")
ylabel("L_tde - L_mcdts")
grid()

axis2 = subplot(424)
plot(Fs, optimal_d_tde[:,ts_trial])
ylims2 = axis2.get_ylim()
vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
title("TDE")
ylabel("embedding dimension")
grid()

axis3 = subplot(425)
plot(Fs, (L_pec-L_mcdts))
plot(Fs[n_e_i], (L_pec[n_e_i]-L_mcdts[n_e_i]),"*")
ylims3 = axis3.get_ylim()
vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
title("PECUZAL")
ylabel("L_pec - L_mcdts")
grid()

axis3 = subplot(426)
plot(Fs, optimal_d_pec)
plot(Fs[n_e_i], optimal_d_pec[n_e_i],"*")
ylims3 = axis3.get_ylim()
vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
title("PECUZAL")
ylabel("embedding dimension")
grid()

axis4 = subplot(427)
plot(Fs, (L_mcdts-L_mcdts))
ylims4 = axis4.get_ylim()
vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
title("MCDTS")
ylabel("L_mcdts - L_mcdts")
grid()

axis4 = subplot(428)
plot(Fs, optimal_d_mcdts)
ylims4 = axis4.get_ylim()
vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
title("MCDTS")
ylabel("embedding dimension")
grid()

RQA_names = ["RR", "TRANS", "DET", "L_mean", "L_max", "DIV", "ENTR", "TREND",
    "LAM", "TT", "V_max", "V_ENTR", "MRT", "RTE", "NMPRT"]


for RQA_val = 1:2:14

    figure(figsize=(20,10))
    axis1 = subplot(421)
    plot(Fs, λs)
    ylims1 = axis1.get_ylim()
    vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
    title("Lyaps")
    grid()

    axis1 = subplot(422)
    plot(Fs, λs)
    ylims1 = axis1.get_ylim()
    vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
    title("Lyaps")
    grid()

    axis2 = subplot(423)
    plot(Fs, RQA_tde[:,RQA_val])
    ylims2 = axis2.get_ylim()
    vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
    title("TDE")
    ylabel(RQA_names[RQA_val])
    grid()

    axis2 = subplot(424)
    plot(Fs, RQA_tde[:,RQA_val+1])
    ylims2 = axis2.get_ylim()
    vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
    title("TDE")
    ylabel(RQA_names[RQA_val+1])
    grid()

    axis3 = subplot(425)
    plot(Fs, RQA_pec[:,RQA_val])
    plot(Fs[n_e_i], RQA_pec[n_e_i,RQA_val],"*")
    ylims3 = axis3.get_ylim()
    vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
    title("PECUZAL")
    ylabel(RQA_names[RQA_val])
    grid()

    axis3 = subplot(426)
    plot(Fs, RQA_pec[:,RQA_val+1])
    plot(Fs[n_e_i], RQA_pec[n_e_i,RQA_val+1],"*")
    ylims3 = axis3.get_ylim()
    vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
    title("PECUZAL")
    ylabel(RQA_names[RQA_val+1])
    grid()

    axis4 = subplot(427)
    plot(Fs, RQA_mcdts[:,RQA_val])
    ylims4 = axis4.get_ylim()
    vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
    title("MCDTS")
    ylabel(RQA_names[RQA_val])
    grid()

    axis4 = subplot(428)
    plot(Fs, RQA_mcdts[:,RQA_val+1])
    ylims4 = axis4.get_ylim()
    vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
    title("MCDTS")
    ylabel(RQA_names[RQA_val+1])
    grid()
end
