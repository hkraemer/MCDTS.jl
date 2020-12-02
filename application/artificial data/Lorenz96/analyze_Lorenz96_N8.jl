## Here we look at the results gained from the computations made in the scripts
# stored in the `cluster scripts`-folder:
# For a range of `F`-parameter values (F=2.5:0.1:6) we analyzed embeddings
# obtained from traditional time delay embedding, from PECUZAL and from MCDTS, by
# computing RQA-quantifiers.

using PyPlot
pygui(true)
using DelimitedFiles

# determine from which trial you want to process the data
trial = 2

# bind variables
t_idx = vec(Int.(readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_chosen_time_series.csv")))

tau_tde = Int.(readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_tau_tde.csv"))
tau_pecs = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_tau_pec.csv")
tau_MCDTSs = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_tau_MCDTS.csv")
# convert taus into right shape in case of PECUZAL and MCDTS
tau_pec = []
tau_MCDTS = []
for i = 1:size(tau_pecs,1)
    index = isa.(tau_pecs[i,:],Number)
    push!(tau_pec, tau_pecs[i,index])
    index = isa.(tau_MCDTSs[i,:],Number)
    push!(tau_MCDTS, tau_MCDTSs[i,index])
end

ts_pecs = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_ts_pec.csv")
ts_MCDTSs = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_ts_MCDTS.csv")
ts_pec = []
ts_MCDTS = []
for i = 1:size(ts_pecs,1)
    index = isa.(ts_pecs[i,:],Number)
    push!(ts_pec, ts_pecs[i,index])
    index = isa.(ts_MCDTSs[i,:],Number)
    push!(ts_MCDTS, ts_MCDTSs[i,index])
end

optimal_d_tde = Int.(readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_optimal_d_tde.csv"))
optimal_d_pec = Int.(readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_optimal_d_pec.csv"))
optimal_d_mcdts = Int.(readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_optimal_d_mcdts.csv"))

L_tde = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_L_tde.csv")
L_pec = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_L_pec.csv")
L_mcdts = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_L_mcdts.csv")

RQA_tde = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_RQA_tde.csv")
RQA_pec = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_RQA_pec.csv")
RQA_mcdts = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_RQA_mcdts.csv")

Fs = readdlm("./application/artificial data/Lorenz96/Results/trial $(trial)/results_Lorenz96_N_8_Fs.csv")

##
λs = readdlm("./application/artificial data/Lorenz96/Lyapunov spectrum/Lyaps_Lo96_N_8_4.csv")

pos_Lyap_idx = λs[:,1] .> 10^-3


figure(figsize=(10,10))
axis1 = subplot(421)
plot(F, λs)
ylims1 = axis1.get_ylim()
vlines(F[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=0.5)
title("Lyaps")
ylabel("embedding dimension")
grid()

axis1 = subplot(422)
plot(F, λs)
ylims1 = axis1.get_ylim()
vlines(F[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=0.5)
title("Lyaps")
ylabel("embedding dimension")
grid()

axis2 = subplot(423)
#plot(F, optimal_d_tde[:,1])
plot(F, (L_tde[:,1]-L_mcdts))
ylims2 = axis2.get_ylim()
vlines(F[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=0.5)
title("TDE")
ylabel("L_tde - L_mcdts")
grid()

axis2 = subplot(424)
plot(F, optimal_d_tde[:,1])
#plot(F, (L_tde[:,1]-L_mcdts))
ylims2 = axis2.get_ylim()
vlines(F[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=0.5)
title("TDE")
ylabel("embedding dimension")
grid()

axis3 = subplot(425)
#plot(F, optimal_d_pec)
plot(F, (L_pec-L_mcdts))
ylims3 = axis3.get_ylim()
vlines(F[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=0.5)
title("PECUZAL")
ylabel("L_pec - L_mcdts")
grid()

axis3 = subplot(426)
plot(F, optimal_d_pec)
#plot(F, (L_pec-L_mcdts))
ylims3 = axis3.get_ylim()
vlines(F[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=0.5)
title("PECUZAL")
ylabel("embedding dimension")
grid()

axis4 = subplot(427)
#plot(F, optimal_d_mcdts)
plot(F, (L_mcdts-L_mcdts))
ylims4 = axis4.get_ylim()
vlines(F[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=0.5)
title("MCDTS")
ylabel("L_mcdts - L_mcdts")
grid()

axis4 = subplot(428)
plot(F, optimal_d_mcdts)
#plot(F, (L_mcdts-L_mcdts))
ylims4 = axis4.get_ylim()
vlines(F[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=0.5)
title("MCDTS")
ylabel("embedding dimension")
grid()

RQA_names = ["RR", "TRANS", "DET", "L_mean", "L_max", "DIV", "ENTR", "TREND",
    "LAM", "TT", "V_max", "V_ENTR", "MRT", "RTE", "NMPRT"]

for RQA_val = 1:15

    figure(figsize=(10,10))
    axis1 = subplot(411)
    plot(F, λs)
    ylims1 = axis1.get_ylim()
    vlines(F[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=0.5)
    title("Lyaps")
    grid()

    axis2 = subplot(412)
    plot(F, RQA_tde[:,RQA_val])
    ylims2 = axis2.get_ylim()
    vlines(F[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=0.5)
    title("TDE")
    ylabel(RQA_names[RQA_val])
    grid()

    axis3 = subplot(413)
    plot(F, RQA_pec[:,RQA_val])
    ylims3 = axis3.get_ylim()
    vlines(F[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=0.5)
    title("PECUZAL")
    ylabel(RQA_names[RQA_val])
    grid()

    axis4 = subplot(414)
    plot(F, RQA_mcdts[:,RQA_val])
    ylims4 = axis4.get_ylim()
    vlines(F[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=0.5)
    title("MCDTS")
    ylabel(RQA_names[RQA_val])
    grid()
end
