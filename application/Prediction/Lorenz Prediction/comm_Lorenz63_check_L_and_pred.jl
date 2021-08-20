# We check the overall L-statistic and in-sample prediction error for the different
# reconstructions of the Lorenz63-system

using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using DelimitedFiles
using ChaosTools
using Random
using Statistics

using PyPlot
pygui(true)


# load data
begin
    x1 = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/x1.csv"))
    x2 = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/x2.csv"))
    y1 = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/y1.csv"))
    y2 = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/y2.csv"))
    x1_n = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/x1_n.csv"))
    x2_n = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/x2_n.csv"))
    y1_n = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/y1_n.csv"))
    y2_n = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/y2_n.csv"))

    x1_ = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/x1_long.csv"))
    x2_ = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/x2_long.csv"))
    y1_ = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/y1_long.csv"))
    y2_ = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/y2_long.csv"))
    x1_n_ = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/x1_n_long.csv"))
    x2_n_ = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/x2_n_long.csv"))
    y1_n_ = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/y1_n_long.csv"))
    y2_n_ = vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/y2_n_long.csv"))

    data_sample = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/data_sample.csv"))
    data_sample_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/data_sample_n.csv"))
    tr = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tr.csv"))
    Y_cao = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_cao.csv"))
    Y_cao_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_cao_n.csv"))
    τ_cao = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_cao.csv")))
    τ_cao_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_cao_n.csv")))
    Y_kennel = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_kennel.csv"))
    Y_kennel_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_kennel_n.csv"))
    τ_kennel = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_kennel.csv")))
    τ_kennel_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_kennel_n.csv")))
    Y_hegger = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_hegger.csv"))
    Y_hegger_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_hegger_n.csv"))
    τ_hegger = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_hegger.csv")))
    τ_hegger_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_hegger_n.csv")))
    Y_pec = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_pec.csv"))
    Y_pec_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_pec_n.csv"))
    τ_pec = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_pec.csv")))
    τ_pec_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_pec_n.csv")))
    Y_mcdts = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts.csv"))
    Y_mcdts_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_n.csv"))
    τ_mcdts = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts.csv")))
    τ_mcdts_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_n.csv")))
    Y_pec2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_pec2.csv"))
    Y_pec2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_pec2_n.csv"))
    τ_pec2 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_pec2.csv")))
    τ_pec2_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_pec2_n.csv")))
    ts_pec2 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_pec2.csv")))
    ts_pec2_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_pec2_n.csv")))
    Y_mcdts2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts2.csv"))
    τ_mcdts2 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts2.csv")))
    ts_mcdts2 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_mcdts2.csv")))
    Y_mcdts_fnn = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_fnn.csv"))
    τ_mcdts_fnn = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_fnn.csv")))
    Y_mcdts_fnn2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_fnn2.csv"))
    τ_mcdts_fnn2 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_fnn2.csv")))
    ts_mcdts_fnn2 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_mcdts_fnn2.csv")))
    Y_mcdts_fnn_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_fnn_n.csv"))
    τ_mcdts_fnn_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_fnn_n.csv")))
    Y_mcdts_fnn2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_fnn2_n.csv"))
    τ_mcdts_fnn2_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_fnn2_n.csv")))
    ts_mcdts_fnn2_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_mcdts_fnn2_n.csv")))
    Y_mcdts2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts2_n.csv"))
    τ_mcdts2_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts2_n.csv")))
    ts_mcdts2_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_mcdts2_n.csv")))

    Y_mcdts_PRED = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_PRED.csv"))
    τ_mcdts_PRED = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_PRED.csv")))
    Y_mcdts_PRED2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_PRED_multi.csv"))
    τ_mcdts_PRED2 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_PRED_multi.csv")))
    ts_mcdts_PRED2 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_mcdts_PRED_multi.csv")))

    Y_mcdts_PRED_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_PRED_n.csv"))
    τ_mcdts_PRED_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_PRED_n.csv")))
    Y_mcdts_PRED2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_PRED_multi_n.csv"))
    τ_mcdts_PRED2_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_PRED_multi_n.csv")))
    ts_mcdts_PRED2_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_mcdts_PRED_multi_n.csv")))

    Y_mcdts_PRED_5 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_PRED_5.csv"))
    τ_mcdts_PRED_5 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_PRED_5.csv")))
    Y_mcdts_PRED2_5 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_PRED_multi_5.csv"))
    τ_mcdts_PRED2_5 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_PRED_multi_5.csv")))
    ts_mcdts_PRED2_5 = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_mcdts_PRED_multi_5.csv")))

    Y_mcdts_PRED_5_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_PRED_n_5.csv"))
    τ_mcdts_PRED_5_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_PRED_n_5.csv")))
    Y_mcdts_PRED2_5_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts_PRED_multi_5_n.csv"))
    τ_mcdts_PRED2_5_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts_PRED_multi_5_n.csv")))
    ts_mcdts_PRED2_5_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_mcdts_PRED_multi_5_n.csv")))

    # error correction:
    Y_mcdts_PRED_5_n = MCDTS.genembed_for_prediction(x1_n, τ_mcdts_PRED_5_n)
end

T_max = 100
w1 = DelayEmbeddings.estimate_delay(x1, "mi_min")
w1_n = DelayEmbeddings.estimate_delay(x1_n, "mi_min")
# total number of reconstructions:
recons = ["Cao", "Hegger", "Kennel", "MCDTS", "MCDTS mult.","MCDTS PRED","MCDTS PRED mult.",
        "MCDTS PRED mult. 5", "MCDTS PRED 5", "MCDTS FNN", "MCDTS FNN mult.", "PECUZAL", "PECUZAL mult."]
Ls = zeros(13)
Ls_n = zeros(13)

Ls[1] = MCDTS.compute_delta_L(Y_cao, τ_cao, ones(Int,length(τ_cao)),T_max; w = w1)
Ls[2] = MCDTS.compute_delta_L(Y_hegger, τ_hegger, ones(Int,length(τ_hegger)),T_max; w = w1)
Ls[3] = MCDTS.compute_delta_L(Y_kennel, τ_kennel, ones(Int,length(τ_kennel)),T_max; w = w1)
Ls[4] = MCDTS.compute_delta_L(Y_mcdts, τ_mcdts, ones(Int,length(τ_mcdts)),T_max; w = w1)
Ls[5] = MCDTS.compute_delta_L(Y_mcdts2, τ_mcdts2, ts_mcdts2,T_max; w = w1)
Ls[6] = MCDTS.compute_delta_L(Y_mcdts_PRED, τ_mcdts_PRED, ones(Int,length(τ_mcdts_PRED)),T_max; w = w1)
Ls[7] = MCDTS.compute_delta_L(Y_mcdts_PRED2, τ_mcdts_PRED2, ts_mcdts_PRED2,T_max; w = w1)
Ls[8] = MCDTS.compute_delta_L(Y_mcdts_PRED2_5, τ_mcdts_PRED2_5, ts_mcdts_PRED2_5,T_max; w = w1)
Ls[9] = MCDTS.compute_delta_L(Y_mcdts_PRED_5, τ_mcdts_PRED_5, ones(Int,length(τ_mcdts_PRED_5)),T_max; w = w1)
Ls[10] = MCDTS.compute_delta_L(Y_mcdts_fnn, τ_mcdts_fnn, ones(Int,length(τ_mcdts_fnn)),T_max; w = w1)
Ls[11] = MCDTS.compute_delta_L(Y_mcdts_fnn2, τ_mcdts_fnn2, ts_mcdts_fnn2, T_max; w = w1)
Ls[12] = MCDTS.compute_delta_L(Y_pec, τ_pec, ones(Int,length(τ_pec)),T_max; w = w1)
Ls[13] = MCDTS.compute_delta_L(Y_pec2, τ_pec2, ts_pec2, T_max; w = w1)

Ls_n[1] = MCDTS.compute_delta_L(Y_cao_n, τ_cao_n, ones(Int,length(τ_cao_n)),T_max; w = w1_n)
Ls_n[2] = MCDTS.compute_delta_L(Y_hegger_n, τ_hegger_n, ones(Int,length(τ_hegger_n)),T_max; w = w1_n)
Ls_n[3] = MCDTS.compute_delta_L(Y_kennel_n, τ_kennel_n, ones(Int,length(τ_kennel_n)),T_max; w = w1_n)
Ls_n[4] = MCDTS.compute_delta_L(Y_mcdts_n, τ_mcdts_n, ones(Int,length(τ_mcdts_n)),T_max; w = w1_n)
Ls_n[5] = MCDTS.compute_delta_L(Y_mcdts2_n, τ_mcdts2_n, ts_mcdts2_n,T_max; w = w1_n)
Ls_n[6] = MCDTS.compute_delta_L(Y_mcdts_PRED_n, τ_mcdts_PRED_n, ones(Int,length(τ_mcdts_PRED_n)),T_max; w = w1_n)
Ls_n[7] = MCDTS.compute_delta_L(Y_mcdts_PRED2_n, τ_mcdts_PRED2_n, ts_mcdts_PRED2_n,T_max; w = w1_n)
Ls_n[8] = MCDTS.compute_delta_L(Y_mcdts_PRED2_5_n, τ_mcdts_PRED2_5_n, ts_mcdts_PRED2_5_n,T_max; w = w1_n)
Ls_n[9] = MCDTS.compute_delta_L(Y_mcdts_PRED_5_n, τ_mcdts_PRED_5_n, ones(Int,length(τ_mcdts_PRED_5_n)),T_max; w = w1_n)
Ls_n[10] = MCDTS.compute_delta_L(Y_mcdts_fnn_n, τ_mcdts_fnn_n, ones(Int,length(τ_mcdts_fnn_n)),T_max; w = w1_n)
Ls_n[11] = MCDTS.compute_delta_L(Y_mcdts_fnn2_n, τ_mcdts_fnn2_n, ts_mcdts_fnn2_n, T_max; w = w1_n)
Ls_n[12] = MCDTS.compute_delta_L(Y_pec_n, τ_pec_n, ones(Int,length(τ_pec_n)),T_max; w = w1_n)
Ls_n[13] = MCDTS.compute_delta_L(Y_pec2_n, τ_pec2_n, ts_pec2_n, T_max; w = w1_n)


Ls_tot = zeros(13,50)
Ls_tot_n = zeros(13,50)

for i = 1:50
    println("run: $i")
    Ls_tot[1,i] = DelayEmbeddings.uzal_cost(Y_cao; w = w1, Tw=i)
    Ls_tot[2,i] = DelayEmbeddings.uzal_cost(Y_hegger; w = w1, Tw=i)
    Ls_tot[3,i] = DelayEmbeddings.uzal_cost(Y_kennel; w = w1, Tw=i)
    Ls_tot[4,i] = DelayEmbeddings.uzal_cost(Y_mcdts; w = w1, Tw=i)
    Ls_tot[5,i] = DelayEmbeddings.uzal_cost(Y_mcdts2; w = w1, Tw=i)
    Ls_tot[6,i] = DelayEmbeddings.uzal_cost(Y_mcdts_PRED; w = w1, Tw=i)
    Ls_tot[7,i] = DelayEmbeddings.uzal_cost(Y_mcdts_PRED2; w = w1, Tw=i)
    Ls_tot[8,i] = DelayEmbeddings.uzal_cost(Y_mcdts_PRED2_5; w = w1, Tw=i)
    Ls_tot[9,i] = DelayEmbeddings.uzal_cost(Y_mcdts_PRED_5; w = w1, Tw=i)
    Ls_tot[10,i] = DelayEmbeddings.uzal_cost(Y_mcdts_fnn; w = w1, Tw=i)
    Ls_tot[11,i] = DelayEmbeddings.uzal_cost(Y_mcdts_fnn2; w = w1, Tw=i)
    Ls_tot[12,i] = DelayEmbeddings.uzal_cost(Y_pec; w = w1, Tw=i)
    Ls_tot[13,i] = DelayEmbeddings.uzal_cost(Y_pec2; w = w1, Tw=i)

    Ls_tot_n[1,i] = DelayEmbeddings.uzal_cost(Y_cao_n; w = w1_n, Tw=i)
    Ls_tot_n[2,i] = DelayEmbeddings.uzal_cost(Y_hegger_n; w = w1_n, Tw=i)
    Ls_tot_n[3,i] = DelayEmbeddings.uzal_cost(Y_kennel_n; w = w1_n, Tw=i)
    Ls_tot_n[4,i] = DelayEmbeddings.uzal_cost(Y_mcdts_n; w = w1_n, Tw=i)
    Ls_tot_n[5,i] = DelayEmbeddings.uzal_cost(Y_mcdts2_n; w = w1_n, Tw=i)
    Ls_tot_n[6,i] = DelayEmbeddings.uzal_cost(Y_mcdts_PRED_n; w = w1_n, Tw=i)
    Ls_tot_n[7,i] = DelayEmbeddings.uzal_cost(Y_mcdts_PRED2_n; w = w1_n, Tw=i)
    Ls_tot_n[8,i] = DelayEmbeddings.uzal_cost(Y_mcdts_PRED2_5_n; w = w1_n, Tw=i)
    Ls_tot_n[9,i] = DelayEmbeddings.uzal_cost(Y_mcdts_PRED_5_n; w = w1_n, Tw=i)
    Ls_tot_n[10,i] = DelayEmbeddings.uzal_cost(Y_mcdts_fnn_n; w = w1_n, Tw=i)
    Ls_tot_n[11,i] = DelayEmbeddings.uzal_cost(Y_mcdts_fnn2_n; w = w1_n, Tw=i)
    Ls_tot_n[12,i] = DelayEmbeddings.uzal_cost(Y_pec_n; w = w1_n, Tw=i)
    Ls_tot_n[13,i] = DelayEmbeddings.uzal_cost(Y_pec2_n; w = w1_n, Tw=i)
end


# mean Forecast error
Preds = zeros(13)
Preds_n = zeros(13)
K = 1

Preds[1] = mean(MCDTS.zeroth_prediction_cost(Y_cao; K = K, w = w1))
Preds[2] = mean(MCDTS.zeroth_prediction_cost(Y_hegger; K = K, w = w1))
Preds[3] = mean(MCDTS.zeroth_prediction_cost(Y_kennel; K = K, w = w1))
Preds[4] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts; K = K, w = w1))
Preds[5] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts2; K = K, w = w1))
Preds[6] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_PRED; K = K, w = w1))
Preds[7] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2; K = K, w = w1))
Preds[8] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2_5; K = K, w = w1))
Preds[9] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_PRED_5; K = K, w = w1))
Preds[10] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_fnn; K = K, w = w1))
Preds[11] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_fnn2; K = K, w = w1))
Preds[12] = mean(MCDTS.zeroth_prediction_cost(Y_pec; K = K, w = w1))
Preds[13] = mean(MCDTS.zeroth_prediction_cost(Y_pec2; K = K, w = w1))

Preds_n[1] = mean(MCDTS.zeroth_prediction_cost(Y_cao_n; K = K, w = w1_n))
Preds_n[2] = mean(MCDTS.zeroth_prediction_cost(Y_hegger_n; K = K, w = w1_n))
Preds_n[3] = mean(MCDTS.zeroth_prediction_cost(Y_kennel_n; K = K, w = w1_n))
Preds_n[4] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_n; K = K, w = w1_n))
Preds_n[5] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts2_n; K = K, w = w1_n))
Preds_n[6] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_PRED_n; K = K, w = w1_n))
Preds_n[7] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2_n; K = K, w = w1_n))
Preds_n[8] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2_5_n; K = K, w = w1_n))
Preds_n[9] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_PRED_5_n; K = K, w = w1_n))
Preds_n[10] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_fnn_n; K = K, w = w1_n))
Preds_n[11] = mean(MCDTS.zeroth_prediction_cost(Y_mcdts_fnn2_n; K = K, w = w1_n))
Preds_n[12] = mean(MCDTS.zeroth_prediction_cost(Y_pec_n; K = K, w = w1_n))
Preds_n[13] = mean(MCDTS.zeroth_prediction_cost(Y_pec2_n; K = K, w = w1_n))

# Forecast error on x-component
Preds_x = zeros(13)
Preds_x_n = zeros(13)
K = 1

Preds_x[1] = MCDTS.zeroth_prediction_cost(Y_cao; K = K, w = w1)[1]
Preds_x[2] = MCDTS.zeroth_prediction_cost(Y_hegger; K = K, w = w1)[1]
Preds_x[3] = MCDTS.zeroth_prediction_cost(Y_kennel; K = K, w = w1)[1]
Preds_x[4] = MCDTS.zeroth_prediction_cost(Y_mcdts; K = K, w = w1)[1]
Preds_x[5] = MCDTS.zeroth_prediction_cost(Y_mcdts2; K = K, w = w1)[2]
Preds_x[6] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED; K = K, w = w1)[1]
Preds_x[7] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2; K = K, w = w1)[1]
Preds_x[8] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2_5; K = K, w = w1)[1]
Preds_x[9] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED_5; K = K, w = w1)[1]
Preds_x[10] = MCDTS.zeroth_prediction_cost(Y_mcdts_fnn; K = K, w = w1)[1]
Preds_x[11] = MCDTS.zeroth_prediction_cost(Y_mcdts_fnn2; K = K, w = w1)[1]
Preds_x[12] = MCDTS.zeroth_prediction_cost(Y_pec; K = K, w = w1)[1]
Preds_x[13] = MCDTS.zeroth_prediction_cost(Y_pec2; K = K, w = w1)[2]

Preds_x_n[1] = MCDTS.zeroth_prediction_cost(Y_cao_n; K = K, w = w1_n)[1]
Preds_x_n[2] = MCDTS.zeroth_prediction_cost(Y_hegger_n; K = K, w = w1_n)[1]
Preds_x_n[3] = MCDTS.zeroth_prediction_cost(Y_kennel_n; K = K, w = w1_n)[1]
Preds_x_n[4] = MCDTS.zeroth_prediction_cost(Y_mcdts_n; K = K, w = w1_n)[1]
Preds_x_n[5] = MCDTS.zeroth_prediction_cost(Y_mcdts2_n; K = K, w = w1_n)[1]
Preds_x_n[6] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED_n; K = K, w = w1_n)[1]
Preds_x_n[7] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2_n; K = K, w = w1_n)[1]
Preds_x_n[8] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2_5_n; K = K, w = w1_n)[1]
Preds_x_n[9] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED_5_n; K = K, w = w1_n)[1]
Preds_x_n[10] = MCDTS.zeroth_prediction_cost(Y_mcdts_fnn_n; K = K, w = w1_n)[1]
Preds_x_n[11] = MCDTS.zeroth_prediction_cost(Y_mcdts_fnn2_n; K = K, w = w1_n)[1]
Preds_x_n[12] = MCDTS.zeroth_prediction_cost(Y_pec_n; K = K, w = w1_n)[1]
Preds_x_n[13] = MCDTS.zeroth_prediction_cost(Y_pec2_n; K = K, w = w1_n)[1]



## Plot results

recons = ["Cao", "Hegger", "Kennel", "MCDTS", "MCDTS mult.","MCDTS PRED","MCDTS PRED mult.",
        "MCDTS PRED mult. 5", "MCDTS PRED 5", "MCDTS FNN", "MCDTS FNN mult.", "PECUZAL", "PECUZAL mult."]
marks = [".", "o","v","1","8","s","p","*","H","+","x","D","X"]
colorss = ["b","b","b","r","r","g","g","g","g","m","m","y","y"]

# L statistics
figure(figsize=(15,10))
bar(1:2:26,Ls, color=colorss)
xticks(1:2:26, recons, rotation=45)
title("Noisefree")
ylabel("L-statistic")
grid()

figure(figsize=(15,10))
bar(1:2:26,Ls_n, color=colorss)
xticks(1:2:26, recons, rotation=45)
title("5 % additive noise")
ylabel("L-statistic")
grid()

# forecats errors
figure(figsize=(15,10))
bar(1:2:26,Preds, color=colorss)
xticks(1:2:26, recons, rotation=45)
title("Noisefree")
ylabel("Mean Forecast-error")
grid()

figure(figsize=(15,10))
bar(1:2:26,Preds_n, color=colorss)
xticks(1:2:26, recons, rotation=45)
title("5 % additive noise")
ylabel("Mean Forecast-error")
grid()

figure(figsize=(15,10))
bar(1:2:26,Preds_x, color=colorss)
xticks(1:2:26, recons, rotation=45)
title("Noisefree")
ylabel("Forecast-error on x-component")
grid()

figure(figsize=(15,10))
bar(1:2:26,Preds_x_n, color=colorss)
xticks(1:2:26, recons, rotation=45)
title("5 % additive noise")
ylabel("Forecast-error on x-component")
grid()


figure(figsize=(15,10))
for i = 1:length(recons)
    plot(1:50, Ls_tot[i,:], label=recons[i], color=colorss[i], marker=marks[i])
end
grid()
legend()
xlabel("Horizon")
ylabel("L-statistic value")
title("L-Statistic for different time horizons (noisefree)")

figure(figsize=(15,10))
for i = 1:length(recons)
    plot(1:50, Ls_tot_n[i,:], label=recons[i], color=colorss[i], marker=marks[i])
end
grid()
legend()
title("L-Statistic for different time horizons (noisy)")
xlabel("Horizon")
ylabel("L-statistic value")


## Double check for noisy case:

# example Cao and mcdts_PRED
Y_example_cao = MCDTS.genembed_for_prediction(x1_n, [0, 17])
Y_example_PRED = MCDTS.genembed_for_prediction(x1_n, [0, 2])

Error_cao = MCDTS.zeroth_prediction_cost(Y_example_cao; K = 1, w = w1_n)[1]
Error_PRED = MCDTS.zeroth_prediction_cost(Y_example_PRED; K = 1, w = w1_n)[1]

Error = zeros(40)
for i = 1:40
    Y_example = MCDTS.genembed_for_prediction(Vector(xx[:,1]), [0, i])
    Y_example = DelayEmbeddings.hcat_lagged_values(xx, Vector(xx[:,1]), i)
    Error[i] = MCDTS.zeroth_prediction_cost(Y_example; K = 1, w = w1_n, Tw = 1)[1]
end

figure()
plot(Error)

Y_example1 = MCDTS.genembed_for_prediction(Vector(xx[:,1]), [0, 1])
Y_example2 = DelayEmbeddings.hcat_lagged_values(xx, Vector(xx[:,1]), 1)
Y_example3 = genembed(xx, (0,-1))

m1 = MCDTS.zeroth_prediction_cost(Y_example1; K = 1, w = w1_n, Tw = 1)[1]
m2 = MCDTS.zeroth_prediction_cost(Y_example2; K = 1, w = w1_n, Tw = 1)[1]
m3 = MCDTS.zeroth_prediction_cost(Y_example3; K = 1, w = w1_n, Tw = 1)[1]

using Revise
using MCDTS

Tw = 1  # time horizon
KK = 1 # considered nearest neighbors
trials = 1
max_depth = 15
PRED_mean = false
PRED_L = false
tree = MCDTS.mc_delay(Dataset(x1_n) ,w1_n,(L)->(MCDTS.softmaxL(L,β=2.)),
    0:5, trials; max_depth = max_depth, PRED = true, verbose = true, KNN = KK,
    threshold = 5e-6, PRED_mean = PRED_mean, PRED_L = PRED_L)
best_node = MCDTS.best_embedding(tree)
