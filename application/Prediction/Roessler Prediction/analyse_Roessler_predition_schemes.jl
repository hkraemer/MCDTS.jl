using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using DelimitedFiles
using ChaosTools
using Random
using Statistics

using PyPlot
pygui(true)


# load data (Reconstructions)
# The reconstructions have been computed in the script `comm_Lorenz63_reconstructions.jl`
# and in the scripts stored in the folder `/Cluster scripts`.
begin
    x1 = vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/x1.csv"))
    x2 = vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/x2.csv"))
    y1 = vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/y1.csv"))
    y2 = vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/y2.csv"))
    x1_n = vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/x1_n.csv"))
    x2_n = vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/x2_n.csv"))
    y1_n = vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/y1_n.csv"))
    y2_n = vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/y2_n.csv"))

    data_sample = Dataset(readdlm("./application/artificial data/Roessler Prediction/Results 1/data_sample.csv"))
    data_sample_n = Dataset(readdlm("./application/artificial data/Roessler Prediction/Results 1/data_sample_n.csv"))
    tr = Dataset(readdlm("./application/artificial data/Roessler Prediction/Results 1/tr.csv"))
    τ_cao = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_cao.csv")))
    τ_cao_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_cao_n.csv")))
    τ_kennel = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_kennel.csv")))
    τ_kennel_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_kennel_n.csv")))
    τ_hegger = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_hegger.csv")))
    τ_hegger_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_hegger_n.csv")))
    τ_pec = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_pec.csv")))
    τ_pec_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_pec_n.csv")))
    τ_mcdts = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts.csv")))
    τ_mcdts_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts_n.csv")))
    τ_pec2 = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_pec2.csv")))
    τ_pec2_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_pec2_n.csv")))
    ts_pec2 = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/ts_pec2.csv")))
    ts_pec2_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/ts_pec2_n.csv")))
    τ_mcdts2 = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts2.csv")))
    ts_mcdts2 = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/ts_mcdts2.csv")))
    τ_mcdts_fnn = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts_fnn.csv")))
    τ_mcdts_fnn2 = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts_fnn2.csv")))
    ts_mcdts_fnn2 = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/ts_mcdts_fnn2.csv")))
    τ_mcdts_fnn_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts_fnn_n.csv")))
    τ_mcdts_fnn2_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts_fnn2_n.csv")))
    ts_mcdts_fnn2_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/ts_mcdts_fnn2_n.csv")))
    τ_mcdts2_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts2_n.csv")))
    ts_mcdts2_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/ts_mcdts2_n.csv")))

    τ_mcdts_PRED = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts_PRED.csv")))
    τ_mcdts_PRED2 = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts_PRED_multi.csv")))
    ts_mcdts_PRED2 = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/ts_mcdts_PRED_multi.csv")))

    τ_mcdts_PRED_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts_PRED_n.csv")))
    τ_mcdts_PRED2_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/tau_mcdts_PRED_multi_n.csv")))
    ts_mcdts_PRED2_n = Int.(vec(readdlm("./application/artificial data/Roessler Prediction/Results 1/ts_mcdts_PRED_multi_n.csv")))

    # make the right reconstructions
    # uni-variate
    Y_cao = genembed(x1, τ_cao .* (-1))
    Y_cao_n = genembed(x1_n, τ_cao_n .* (-1))
    Y_kennel = genembed(x1, τ_kennel .* (-1))
    Y_kennel_n = genembed(x1_n, τ_kennel_n .* (-1))
    Y_hegger = genembed(x1, τ_hegger .* (-1))
    Y_hegger_n = genembed(x1_n, τ_hegger_n .* (-1))
    Y_pec = genembed(x1, τ_pec .* (-1))
    Y_pec_n = genembed(x1_n, τ_pec_n .* (-1))
    Y_mcdts = genembed(x1, τ_mcdts .* (-1))
    Y_mcdts_n = genembed(x1_n, τ_mcdts_n .* (-1))
    Y_mcdts_fnn = genembed(x1, τ_mcdts_fnn .* (-1))
    Y_mcdts_fnn_n = genembed(x1_n, τ_mcdts_fnn_n .* (-1))
    Y_mcdts_PRED = genembed(x1, τ_mcdts_PRED .* (-1))
    Y_mcdts_PRED_n = genembed(x1_n, τ_mcdts_PRED_n .* (-1))

    # multi-variate
    Y_pec2 = genembed(data_sample, τ_pec2 .* (-1), ts_pec2)
    Y_pec2_n = genembed(data_sample_n, τ_pec2_n .* (-1), ts_pec2_n)
    Y_mcdts2 = genembed(data_sample, τ_mcdts2 .* (-1), ts_mcdts2)
    Y_mcdts2_n = genembed(data_sample_n, τ_mcdts2_n .* (-1), ts_mcdts2_n)
    Y_mcdts_fnn2 = genembed(data_sample, τ_mcdts_fnn2 .* (-1), ts_mcdts_fnn2)
    Y_mcdts_fnn2_n = genembed(data_sample_n, τ_mcdts_fnn2_n .* (-1), ts_mcdts_fnn2_n)
    Y_mcdts_PRED2 = genembed(data_sample, τ_mcdts_PRED2 .* (-1), ts_mcdts_PRED2)
    Y_mcdts_PRED2_n = genembed(data_sample_n, τ_mcdts_PRED2_n .* (-1), ts_mcdts_PRED2_n)
end

# Theiler window:
w1 = DelayEmbeddings.estimate_delay(x1, "mi_min")
w1_n = DelayEmbeddings.estimate_delay(x1_n, "mi_min")

# Lyapunov time
lyap_time = Int.(readdlm("./application/artificial data/Roessler Prediction/Results 1/lyap_time.csv"))[1]
T_steps = 3000

## make predictions

# Results 2: zeroth T_steps = 3000, KK = 1

KK = 1 # number of nearest neighbours for zeroth predictor

begin
    # iterated one step
    # Zeroth
    # println("Cao:")
    # println("*****")
    # Cao_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_cao, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/Cao_zeroth.csv",Cao_zeroth)
    # Cao_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_cao_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/Cao_zeroth_n.csv",Cao_zeroth_n)
    # println("Kennel")
    # println("*****")
    # Kennel_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_kennel, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/Kennel_zeroth.csv",Kennel_zeroth)
    # Kennel_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_kennel_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/Kennel_zeroth_n.csv",Kennel_zeroth_n)
    # println("Hegger")
    # println("*****")
    # Hegger_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_hegger, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/Hegger_zeroth.csv",Hegger_zeroth)
    # Hegger_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_hegger_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/Hegger_zeroth_n.csv",Hegger_zeroth_n)
    # println("Pec")
    # println("*****")
    # Pec_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_pec, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/Pec_zeroth.csv",Pec_zeroth)
    # Pec_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_pec2, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/Pec_zeroth2.csv",Pec_zeroth2)
    # Pec_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_pec_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/Pec_zeroth_n.csv",Pec_zeroth_n)
    # Pec_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_pec2_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/Pec_zeroth2_n.csv", Pec_zeroth2_n)
    # println("mcdts:")
    # println("*****")
    # mcdts_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_mcdts, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_zeroth.csv",mcdts_zeroth)
    # mcdts_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_mcdts2, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_zeroth2.csv",mcdts_zeroth2)
    # mcdts_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_zeroth_n.csv",mcdts_zeroth_n)
    mcdts_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts2_n, KK, T_steps; theiler = w1_n, verbose=true)
    writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_zeroth2_n.csv",mcdts_zeroth2_n)
    # println("mcdts FNN:")
    # println("*****")
    # mcdts_fnn_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_fnn_zeroth.csv",mcdts_fnn_zeroth)
    # mcdts_fnn_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn2, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_fnn_zeroth2.csv",mcdts_fnn_zeroth2)
    # mcdts_fnn_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn_n,  KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_fnn_zeroth_n.csv",mcdts_fnn_zeroth_n)
    # mcdts_fnn_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn2_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_fnn_zeroth.csv",mcdts_fnn_zeroth2_n)
    # println("mcdts PRED:")
    # println("*****")
    # mcdts_PRED_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_PRED_zeroth.csv",mcdts_PRED_zeroth)
    # mcdts_PRED_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED2, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_PRED_zeroth2.csv",mcdts_PRED_zeroth2)
    # mcdts_PRED_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_n,  KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_PRED_zeroth_n.csv",mcdts_PRED_zeroth_n)
    # mcdts_PRED_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED2_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_PRED_zeroth2_n.csv",mcdts_PRED_zeroth2_n)

end

# pool all predictions
PREDICTIONS = zeros(11,T_steps)
PREDICTIONS_n = zeros(11,T_steps)

methods = ["Cao", "Kennel", "Hegger", "PECUZAL uni", "PECUZAL multi", "MCDTS L uni", "MCDTS L multi",
            "MCDTS FNN uni", "MCDTS FNN multi", "MCDTS PRED uni", "MCDTS PRED multi"]

# load data
load = begin
    PREDICTIONS[1,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/Cao_zeroth.csv"))[:,1]
    PREDICTIONS_n[1,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/Cao_zeroth_n.csv"))[:,1]
    PREDICTIONS[2,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/Kennel_zeroth.csv"))[:,1]
    PREDICTIONS_n[2,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/Kennel_zeroth_n.csv"))[:,1]
    PREDICTIONS[3,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/Hegger_zeroth.csv"))[:,1]
    PREDICTIONS_n[3,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/Hegger_zeroth_n.csv"))[:,1]
    PREDICTIONS[4,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/Pec_zeroth.csv"))[:,1]
    PREDICTIONS[5,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/Pec_zeroth2.csv"))[:,2]
    PREDICTIONS_n[4,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/Pec_zeroth_n.csv"))[:,1]
    PREDICTIONS_n[5,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/Pec_zeroth2_n.csv"))[:,2]
    PREDICTIONS[6,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_zeroth.csv"))[:,1]
    PREDICTIONS[7,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_zeroth2.csv"))[:,1]
    PREDICTIONS_n[6,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_zeroth_n.csv"))[:,1]
    PREDICTIONS_n[7,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_zeroth2_n.csv"))[:,1]
    PREDICTIONS[8,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_fnn_zeroth.csv"))[:,1]
    PREDICTIONS[9,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_fnn_zeroth2.csv"))[:,1]
    PREDICTIONS_n[8,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_fnn_zeroth_n.csv"))[:,1]
    PREDICTIONS_n[9,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_fnn_zeroth.csv"))[:,1]
    PREDICTIONS[10,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_PRED_zeroth.csv"))[:,1]
    PREDICTIONS[11,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_PRED_zeroth2.csv"))[:,1]
    PREDICTIONS_n[10,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_PRED_zeroth_n.csv"))[:,1]
    PREDICTIONS_n[11,:] = Dataset(readdlm("./application/artificial data/Roessler Prediction/Prediction Results/mcdts_PRED_zeroth2_n.csv"))[:,1]
end


# time axis
t2 = (0:T_steps-1) ./lyap_time
t1 = (-length(x1):-1) ./lyap_time
NN = 1000
tt = vcat(t1[end-NN:end], t2)
M = length(tt)
true_data = vcat(x1[end-NN:end], x2)
true_data_n = vcat(x1_n[end-NN:end], x2_n)

## MSEs
# compute MSE of predictions
begin
    MSEs = zeros(11,T_steps)
    MSEs_n = zeros(11,T_steps)

    σ₂ = sqrt(var(x2[1:T_steps]))   # rms deviation for normalization
    σ₂_n = sqrt(var(x2_n[1:T_steps]))

    for i = 1:T_steps
        for j = 1:11
            MSEs[j,i] = MCDTS.compute_mse(PREDICTIONS[j,1:i], x2[1:i]) / σ₂
            MSEs_n[j,i] = MCDTS.compute_mse(PREDICTIONS_n[j,1:i], x2_n[1:i]) / σ₂_n
        end
    end

    markers = ["s", "P", "*", "+", "X", "D", "o", "v", "1", "<", "|"]

    colorss = ["b", "m", "g", "k", "r", "y", "c", "darkgreen", "navy", "brown", "pink"]

    sc = 10 # markersize

    # # Plot MSEs
    # figure(figsize=(20,10))
    # subplot(221)
    # for i = 1:5
    #     plot(t2[1:T_steps], MSEs[i,:], colorss[i])
    #     scatter(t2[1:T_steps], MSEs[i,:], s = sc, color=colorss[i], marker = markers[i], label=methods[i])
    # end
    # legend()
    # title("Forecast Error")
    # yscale("log")
    # xlim(-0, T_steps/lyap_time)
    # ylim(0.001, 1.5)
    # ylabel("MSE")
    # xlabel("Lyapunov time units")
    # grid()
    #
    # subplot(222)
    # for i = 1:5
    #     plot(t2[1:T_steps], MSEs_n[i,:], colorss[i])
    #     scatter(t2[1:T_steps], MSEs_n[i,:], s = sc, color=colorss[i], marker = markers[i], label=methods[i])
    # end
    # legend()
    # title("Forecast Error (noisy)")
    # yscale("log")
    # xlim(-0, T_steps/lyap_time)
    # ylim(0.001, 1.5)
    # ylabel("MSE")
    # xlabel("Lyapunov time units")
    # grid()
    #
    # subplot(223)
    # for i = 6:11
    #     plot(t2[1:T_steps], MSEs[i,:], colorss[i-5])
    #     scatter(t2[1:T_steps], MSEs[i,:], s = sc, color=colorss[i-5], marker = markers[i-5], label=methods[i])
    # end
    # legend()
    # title("Forecast Error")
    # yscale("log")
    # xlim(-0, T_steps/lyap_time)
    # ylim(0.001, 1.5)
    # ylabel("MSE")
    # xlabel("Lyapunov time units")
    # grid()
    #
    # subplot(224)
    # for i = 6:11
    #     plot(t2[1:T_steps], MSEs_n[i,:], colorss[i-5])
    #     scatter(t2[1:T_steps], MSEs_n[i,:], s = sc, color=colorss[i-5], marker = markers[i-5], label=methods[i])
    # end
    # legend()
    # title("Forecast Error (noisy)")
    # yscale("log")
    # xlim(-0, T_steps/lyap_time)
    # ylim(0.001, 1.5)
    # ylabel("MSE")
    # xlabel("Lyapunov time units")
    # grid()

    # Plot MSEs
    figure(figsize=(20,10))
    for i = 1:11
        plot(t2[1:T_steps], MSEs[i,:], colorss[i])
        scatter(t2[1:T_steps], MSEs[i,:], s = sc, color=colorss[i], marker = markers[i], label=methods[i])
    end
    legend()
    title("Forecast Error")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.5)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()

    figure(figsize=(20,10))
    for i = 1:11
        plot(t2[1:T_steps], MSEs_n[i,:], colorss[i])
        scatter(t2[1:T_steps], MSEs_n[i,:], s = sc, color=colorss[i], marker = markers[i], label=methods[i])
    end
    legend()
    title("Forecast Error (noisy)")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.5)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()
end

begin

end


##

## Plot predictions
prints = begin
    # y-lims
    ylim1 = -3
    ylim2 = 3

    figure(figsize=(20,10))
    for i = 1:5
        subplot(5,1,i)
        if i == 1
            plot(tt, true_data, ".-", label="true data")
            title("x-component (Rössler)")
        else
            plot(tt, true_data, ".-")
        end
        plot(t2, PREDICTIONS[i,:], ".-", label=methods[i])
        xlim(-2, T_steps/lyap_time)
        ylim(ylim1,ylim2)
        legend(loc=2)
        grid()
    end
    subplots_adjust(hspace=.5)

    figure(figsize=(20,10))
    cnt = 1
    for i = 6:11
        subplot(6,1,cnt)
        if cnt == 1
            plot(tt, true_data, ".-", label="true data")
            title("x-component (Rössler)")
        else
            plot(tt, true_data, ".-")
        end
        plot(t2, PREDICTIONS[i,:], ".-", label=methods[i])
        xlim(-2, T_steps/lyap_time)
        ylim(ylim1,ylim2)
        legend(loc=2)
        grid()
        cnt += 1
    end
    subplots_adjust(hspace=.5)

    figure(figsize=(20,10))
    for i = 1:5
        subplot(5,1,i)
        if i == 1
            plot(tt, true_data_n, ".-", label="true data")
            title("x-component (Rössler noisy)")
        else
            plot(tt, true_data_n, ".-")
        end
        plot(t2, PREDICTIONS_n[i,:], ".-", label=methods[i])
        xlim(-2, T_steps/lyap_time)
        ylim(ylim1,ylim2)
        legend(loc=2)
        grid()
    end
    subplots_adjust(hspace=.5)

    figure(figsize=(20,10))
    cnt = 1
    for i = 6:11
        subplot(6,1,cnt)
        if cnt == 1
            plot(tt, true_data_n, ".-", label="true data")
            title("x-component (Rössler noisy)")
        else
            plot(tt, true_data_n, ".-")
        end
        plot(t2, PREDICTIONS_n[i,:], ".-", label=methods[i])
        xlim(-2, T_steps/lyap_time)
        ylim(ylim1,ylim2)
        legend(loc=2)
        grid()
        cnt += 1
    end
    subplots_adjust(hspace=.5)
end

# ## Save variables in order to plot nicely in Matlab
#
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/t1.csv",t1)
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/t2.csv",t2)
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/NN.csv",NN)
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/tt.csv",tt)
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/M.csv",M)
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/true_data.csv",true_data)
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/true_data_n.csv",true_data_n)
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/T_steps.csv",T_steps)
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/lyap_time.csv",lyap_time)
#
# recons = ["Cao", "Hegger", "Kennel", "MCDTS", "MCDTS mult.","MCDTS PRED","MCDTS PRED mult.",
#         "MCDTS PRED mult. 5", "MCDTS PRED 5", "MCDTS FNN", "MCDTS FNN mult.", "PECUZAL", "PECUZAL mult."]
#
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/recons.csv",recons)
#
# # Pool all MSE values and save the matrix
# forecast_errors = zeros(13,T_steps)
# forecast_errors_n = zeros(13,T_steps)
#
# forecast_errors[1,:] = MSE_zeroth_cao
# forecast_errors[2,:] = MSE_zeroth_hegger
# forecast_errors[3,:] = MSE_zeroth_kennel
# forecast_errors[4,:] = MSE_zeroth_mcdts
# forecast_errors[5,:] = MSE_zeroth_mcdts2
# forecast_errors[6,:] = MSE_zeroth_mcdts_PRED
# forecast_errors[7,:] = MSE_zeroth_mcdts_PRED2
# forecast_errors[8,:] = MSE_zeroth_mcdts_PRED2_5
# forecast_errors[9,:] = MSE_zeroth_mcdts_PRED_5
# forecast_errors[10,:] = MSE_zeroth_mcdts_fnn
# forecast_errors[11,:] = MSE_zeroth_mcdts_fnn2
# forecast_errors[12,:] = MSE_zeroth_pec
# forecast_errors[13,:] = MSE_zeroth_pec2
#
# forecast_errors_n[1,:] = MSE_zeroth_cao_n
# forecast_errors_n[2,:] = MSE_zeroth_hegger_n
# forecast_errors_n[3,:] = MSE_zeroth_kennel_n
# forecast_errors_n[4,:] = MSE_zeroth_mcdts_n
# forecast_errors_n[5,:] = MSE_zeroth_mcdts2_n
# forecast_errors_n[6,:] = MSE_zeroth_mcdts_PRED_n
# forecast_errors_n[7,:] = MSE_zeroth_mcdts_PRED2_n
# forecast_errors_n[8,:] = MSE_zeroth_mcdts_PRED2_5_n
# forecast_errors_n[9,:] = MSE_zeroth_mcdts_PRED_5_n
# forecast_errors_n[10,:] = MSE_zeroth_mcdts_fnn_n
# forecast_errors_n[11,:] = MSE_zeroth_mcdts_fnn2_n
# forecast_errors_n[12,:] = MSE_zeroth_pec_n
# forecast_errors_n[13,:] = MSE_zeroth_pec2_n
#
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/forecast_errors.csv",forecast_errors)
# writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/forecast_errors_n.csv",forecast_errors_n)
