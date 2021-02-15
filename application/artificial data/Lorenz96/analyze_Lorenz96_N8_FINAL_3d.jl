## Here we look at the results gained from the computations made in the scripts
# stored in the `cluster scripts`-folder:
# For a range of `F`-parameter values (F=2.5:0.1:6) we analyzed embeddings
# obtained from traditional time delay embedding, from PECUZAL and from MCDTS, by
# computing RQA-quantifiers.

using PyPlot
pygui(true)
using DelimitedFiles
using Statistics
N = 8
#cd("./application/artificial data/Lorenz96/Results/Final N8")

# determine the tde method; #1 Cao, #2 Kennel, #3 Hegger
tde = 2

# FNN or L-statistic based?
FNN = true

# display results
show1 = false # embedding dimension and FNN/L-statistic
show2 = true  # RQA statistics

# bind variables
methodss = ["Cao", "Kennel", "Hegger"]
method = methodss[tde]
params = readdlm("./time delay and reference/results_Lorenz96_N_$(N)_final_params.csv")
N,dt,total,ε,dmax,lmin,trials,taus,t_idx,L_threshold = params
Fs = readdlm("./time delay and reference/results_Lorenz96_N_$(N)_final_Fs.csv")
tau_tde = Int.(readdlm("./time delay and reference/results_Lorenz96_N_8_final_tau_tde$tde.csv"))
optimal_d_tde = Int.(readdlm("./time delay and reference/results_Lorenz96_N_8_final_optimal_d_tde$tde.csv"))
L_tde = readdlm("./time delay and reference/results_Lorenz96_N_8_final_L_tde$tde.csv")
FNN_tde = readdlm("./time delay and reference/results_Lorenz96_N_8_final_FNN_tde$tde.csv")
RQA_tde = readdlm("./time delay and reference/results_Lorenz96_N_8_final_RQA_tde$tde.csv")
RP_frac_tde = readdlm("./time delay and reference/results_Lorenz96_N_8_final_RP_frac_tde$tde.csv")

RQA_ref = readdlm("./time delay and reference/results_Lorenz96_N_$(N)_final_RQA_ref.csv")

if FNN
    tau_MCDTSs = readdlm("./MCDTS FNN/results_Lorenz96_N_$(N)_FNN_3d_tau_MCDTS.csv")
    ts_MCDTSs = readdlm("./MCDTS FNN/results_Lorenz96_N_$(N)_FNN_3d_ts_MCDTS.csv")
    optimal_d_mcdts = Int.(readdlm("./MCDTS FNN/results_Lorenz96_N_$(N)_FNN_3d_optimal_d_mcdts.csv"))
    FNN_mcdts = readdlm("./MCDTS FNN/results_Lorenz96_N_$(N)_FNN_3d_FNN_mcdts.csv")
    RQA_mcdts = readdlm("./MCDTS FNN/results_Lorenz96_N_$(N)_FNN_3d_RQA_mcdts.csv")
    RP_frac_mcdts = readdlm("./MCDTS FNN/results_Lorenz96_N_$(N)_FNN_3d_RP_frac_mcdts.csv")

else
    tau_MCDTSs = readdlm("./MCDTS L/results_Lorenz96_N_$(N)_mcdts_3d_tau_MCDTS.csv")
    tau_pec = readdlm("./PECUZAL/results_Lorenz96_N_$(N)_pec_3d_tau_pec.csv")
    ts_MCDTSs = readdlm("./MCDTS L/results_Lorenz96_N_$(N)_mcdts_3d_ts_MCDTS.csv")
    ts_pec = readdlm("./PECUZAL/results_Lorenz96_N_$(N)_pec_3d_ts_pec.csv")
    optimal_d_mcdts = Int.(readdlm("./MCDTS L/results_Lorenz96_N_$(N)_mcdts_3d_optimal_d_mcdts.csv"))
    optimal_d_pec = Int.(readdlm("./PECUZAL/results_Lorenz96_N_$(N)_pec_3d_optimal_d_pec.csv"))
    L_mcdts = readdlm("./MCDTS L/results_Lorenz96_N_$(N)_mcdts_3d_L_mcdts.csv")
    L_pec = readdlm("./PECUZAL/results_Lorenz96_N_$(N)_pec_3d_L_pec.csv")

    RQA_pec = readdlm("./PECUZAL/results_Lorenz96_N_$(N)_pec_3d_RQA_pec.csv")
    RP_frac_pec = readdlm("./PECUZAL/results_Lorenz96_N_$(N)_pec_3d_RP_frac_pec.csv")

    RQA_mcdts = readdlm("./MCDTS L/results_Lorenz96_N_$(N)_mcdts_3d_RQA_mcdts.csv")
    RP_frac_mcdts = readdlm("./MCDTS L/results_Lorenz96_N_$(N)_mcdts_3d_RP_frac_mcdts.csv")
end

# convert taus into right shape in case of MCDTS
tau_MCDTS = []
for i = 1:size(tau_MCDTS,1)
    index = isa.(tau_MCDTSs[i,:],Number)
    push!(tau_MCDTS, tau_MCDTSs[i,index])
end
ts_MCDTS = []
for i = 1:size(ts_MCDTS,1)
    index = isa.(ts_MCDTSs[i,:],Number)
    push!(ts_MCDTS, ts_MCDTSs[i,index])
end

##
λs = readdlm("../../Lyapunov spectrum/Lyaps_Lo96_N_8_3_7_to_4.csv")
pos_Lyap_idx = λs[:,1] .> 10^-3

if FNN
    non_embedding_idx = findall(optimal_d_mcdts.==1)
    statistic = "FNN"
else
    non_embedding_idx = findall(optimal_d_pec.==1)
    statistic = "L"
end
n_e_i = [non_embedding_idx[i][1] for i in eachindex(non_embedding_idx)]

l_width_vert = 0.1

if show1
    if FNN
        figure(figsize=(20,10))
        axis1 = subplot(321)
        plot(Fs, λs)
        ylims1 = axis1.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
        title("Lyaps")
        ylabel("embedding dimension")
        grid()

        axis1 = subplot(322)
        plot(Fs, λs)
        ylims1 = axis1.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
        title("Lyaps")
        ylabel("embedding dimension")
        grid()

        axis2 = subplot(323)
        plot(Fs, (FNN_tde .- FNN_mcdts))
        ylims2 = axis2.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
        title("TDE ($statistic)")
        ylabel("FNN_tde - FNN_mcdts")
        grid()

        axis2 = subplot(324)
        plot(Fs, optimal_d_tde)
        ylims2 = axis2.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
        title("TDE $method ($statistic)")
        ylabel("embedding dimension")
        grid()

        axis4 = subplot(325)
        plot(Fs, FNN_mcdts)
        ylims4 = axis4.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
        title("MCDTS ($statistic)")
        ylabel("FNN_mcdts")
        grid()

        axis4 = subplot(326)
        plot(Fs, optimal_d_mcdts)
        ylims4 = axis4.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
        title("MCDTS ($statistic)")
        ylabel("embedding dimension")
        grid()

    else

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
        plot(Fs, (L_tde .- L_mcdts))
        ylims2 = axis2.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
        title("TDE ($statistic)")
        ylabel("L_tde - L_mcdts")
        grid()

        axis2 = subplot(424)
        plot(Fs, optimal_d_tde)
        ylims2 = axis2.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
        title("TDE $method ($statistic)")
        ylabel("embedding dimension")
        grid()

        axis4 = subplot(425)
        plot(Fs, L_pec .- L_mcdts)
        ylims4 = axis4.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
        title("PECUZAL")
        ylabel("L_pec - L_mcdts")
        grid()

        axis4 = subplot(426)
        plot(Fs, optimal_d_pec)
        ylims4 = axis4.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
        title("PECUZAL")
        ylabel("embedding dimension")
        grid()

        axis4 = subplot(427)
        plot(Fs, L_mcdts)
        ylims4 = axis4.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
        title("MCDTS ($statistic)")
        ylabel("L_mcdts")
        grid()

        axis4 = subplot(428)
        plot(Fs, optimal_d_mcdts)
        ylims4 = axis4.get_ylim()
        vlines(Fs[pos_Lyap_idx], ylims4[1], ylims4[2], linestyle="dashed", linewidth=l_width_vert)
        title("MCDTS ($statistic)")
        ylabel("embedding dimension")
        grid()

    end

end

RQA_names = ["RR", "TRANS", "DET", "L_mean", "L_max", "DIV", "ENTR", "TREND",
    "LAM", "TT", "V_max", "V_ENTR", "MRT", "RTE", "NMPRT"]

if show2
    for RQA_val = 1:2:14

        if FNN

            figure(figsize=(20,10))
            axis1 = subplot(321)
            plot(Fs, λs)
            ylims1 = axis1.get_ylim()
            vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
            title("Lyaps")
            grid()

            axis1 = subplot(322)
            plot(Fs, λs)
            ylims1 = axis1.get_ylim()
            vlines(Fs[pos_Lyap_idx], ylims1[1], ylims1[2], linestyle="dashed", linewidth=l_width_vert)
            title("Lyaps")
            grid()

            axis2 = subplot(323)
            plot(Fs, RQA_ref[:,RQA_val], label = "ref")
            plot(Fs, RQA_tde[:,RQA_val], label = "exp")
            plot(Fs, RQA_tde[:,RQA_val] .- RQA_ref[:,RQA_val], color = "red",label = "diff")
            ylims2 = axis2.get_ylim()
            legend()
            vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
            title("TDE $method (MSE: $(mean((RQA_tde[:,RQA_val] .- RQA_ref[:,RQA_val]).^2 ./RQA_ref[:,RQA_val]))")
            ylabel(RQA_names[RQA_val])
            grid()

            axis2 = subplot(324)
            plot(Fs, RQA_ref[:,RQA_val+1], label = "ref")
            plot(Fs, RQA_tde[:,RQA_val+1], label = "exp")
            plot(Fs, RQA_tde[:,RQA_val+1] .- RQA_ref[:,RQA_val+1], color = "red",label = "diff")
            ylims3 = axis2.get_ylim()
            legend()
            vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
            title("TDE $method (MSE: $(mean((RQA_tde[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]).^2 ./ RQA_ref[:,RQA_val+1]))")
            ylabel(RQA_names[RQA_val+1])
            grid()

            axis4 = subplot(325)
            var = RQA_mcdts[:,RQA_val] .- RQA_ref[:,RQA_val]
            plot(Fs, RQA_ref[:,RQA_val], label = "ref")
            plot(Fs, RQA_mcdts[:,RQA_val], label = "exp")
            plot(Fs, var, color = "red",label = "diff")
            legend()
            axis4.set_ylim(ylims2)
            #ylims4 = axis4.get_ylim()
            vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
            title("MCDTS $statistic (MSE: $(mean((RQA_mcdts[:,RQA_val] .- RQA_ref[:,RQA_val]).^2 ./ RQA_ref[:,RQA_val]))")
            ylabel(RQA_names[RQA_val])
            grid()

            axis4 = subplot(326)
            var2 = RQA_mcdts[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]
            plot(Fs, RQA_ref[:,RQA_val+1], label = "ref")
            plot(Fs, RQA_mcdts[:,RQA_val+1], label = "exp")
            plot(Fs, var2, color = "red", label = "diff")
            legend()
            axis4.set_ylim(ylims3)
            #ylims4 = axis4.get_ylim()
            vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
            title("MCDTS $statistic (MSE: $(mean((RQA_mcdts[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]).^2 ./ RQA_ref[:,RQA_val+1]))")
            ylabel(RQA_names[RQA_val+1])
            grid()

        else

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
            plot(Fs, RQA_ref[:,RQA_val], label = "ref")
            plot(Fs, RQA_tde[:,RQA_val], label = "exp")
            plot(Fs, RQA_tde[:,RQA_val] .- RQA_ref[:,RQA_val], color = "red",label = "diff")
            ylims2 = axis2.get_ylim()
            legend()
            vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
            title("TDE $method (MSE: $(mean((RQA_tde[:,RQA_val] .- RQA_ref[:,RQA_val]).^2 ./RQA_ref[:,RQA_val]))")
            ylabel(RQA_names[RQA_val])
            grid()

            axis2 = subplot(424)
            plot(Fs, RQA_ref[:,RQA_val+1], label = "ref")
            plot(Fs, RQA_tde[:,RQA_val+1], label = "exp")
            plot(Fs, RQA_tde[:,RQA_val+1] .- RQA_ref[:,RQA_val+1], color = "red",label = "diff")
            ylims3 = axis2.get_ylim()
            legend()
            vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
            title("TDE $method (MSE: $(mean((RQA_tde[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]).^2 ./ RQA_ref[:,RQA_val+1]))")
            ylabel(RQA_names[RQA_val+1])
            grid()

            axis4 = subplot(425)
            var = RQA_pec[:,RQA_val] .- RQA_ref[:,RQA_val]
            plot(Fs, RQA_ref[:,RQA_val], label = "ref")
            plot(Fs, RQA_pec[:,RQA_val], label = "exp")
            plot(Fs, var, color = "red",label = "diff")
            legend()
            axis4.set_ylim(ylims2)
            #ylims4 = axis4.get_ylim()
            vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
            title("PECUZAL (MSE: $(mean((RQA_pec[:,RQA_val] .- RQA_ref[:,RQA_val]).^2 ./ RQA_ref[:,RQA_val]))")
            ylabel(RQA_names[RQA_val])
            grid()

            axis4 = subplot(426)
            var2 = RQA_pec[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]
            plot(Fs, RQA_ref[:,RQA_val+1], label = "ref")
            plot(Fs, RQA_pec[:,RQA_val+1], label = "exp")
            plot(Fs, var2, color = "red", label = "diff")
            legend()
            axis4.set_ylim(ylims3)
            #ylims4 = axis4.get_ylim()
            vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
            title("PECUZAL (MSE: $(mean((RQA_pec[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]).^2 ./ RQA_ref[:,RQA_val+1]))")
            ylabel(RQA_names[RQA_val+1])
            grid()

            axis4 = subplot(427)
            var = RQA_mcdts[:,RQA_val] .- RQA_ref[:,RQA_val]
            plot(Fs, RQA_ref[:,RQA_val], label = "ref")
            plot(Fs, RQA_mcdts[:,RQA_val], label = "exp")
            plot(Fs, var, color = "red",label = "diff")
            legend()
            axis4.set_ylim(ylims2)
            #ylims4 = axis4.get_ylim()
            vlines(Fs[pos_Lyap_idx], ylims2[1], ylims2[2], linestyle="dashed", linewidth=l_width_vert)
            title("MCDTS $statistic (MSE: $(mean((RQA_mcdts[:,RQA_val] .- RQA_ref[:,RQA_val]).^2 ./ RQA_ref[:,RQA_val]))")
            ylabel(RQA_names[RQA_val])
            grid()

            axis4 = subplot(428)
            var2 = RQA_mcdts[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]
            plot(Fs, RQA_ref[:,RQA_val+1], label = "ref")
            plot(Fs, RQA_mcdts[:,RQA_val+1], label = "exp")
            plot(Fs, var2, color = "red", label = "diff")
            legend()
            axis4.set_ylim(ylims3)
            #ylims4 = axis4.get_ylim()
            vlines(Fs[pos_Lyap_idx], ylims3[1], ylims3[2], linestyle="dashed", linewidth=l_width_vert)
            title("MCDTS $statistic (MSE: $(mean((RQA_mcdts[:,RQA_val+1] .- RQA_ref[:,RQA_val+1]).^2 ./ RQA_ref[:,RQA_val+1]))")
            ylabel(RQA_names[RQA_val+1])
            grid()
        end
    end
end

# compute mean squared errors
cnt = 0
cnt2 = 0
for i = 2:size(RQA_ref,2)
    global cnt
    global cnt2
    if FNN
        println("MSE TDE $method $(RQA_names[i]):  $(mean((RQA_tde[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]))")
        println("MSE MCDTS $statistic $(RQA_names[i]):  $(mean((RQA_mcdts[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]))")

        if i>1 && mean((RQA_mcdts[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]) < mean((RQA_tde[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i])
            cnt += 1
        end
        println("*******************")
    else
        println("MSE TDE $method $(RQA_names[i]):  $(mean((RQA_tde[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]))")
        println("MSE PECUZAL $(RQA_names[i]):  $(mean((RQA_pec[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]))")
        println("MSE MCDTS $statistic $(RQA_names[i]):  $(mean((RQA_mcdts[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]))")

        if i>1 && mean((RQA_mcdts[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]) < mean((RQA_tde[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i])
            cnt += 1
        end
        if i>1 && mean((RQA_pec[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i]) < mean((RQA_tde[:,i] .- RQA_ref[:,i]).^2 ./ RQA_ref[:,i])
            cnt2 += 1
        end
        println("*******************")

    end
end
println("$cnt2 / $(size(RQA_ref,2)-1)")
println("$cnt / $(size(RQA_ref,2)-1)")
