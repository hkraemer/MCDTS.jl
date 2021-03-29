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


Preds = zeros(13)
Preds_n = zeros(13)
K = 1

Preds[1] = MCDTS.zeroth_prediction_cost(Y_cao; K = K, w = w1)
Preds[2] = MCDTS.zeroth_prediction_cost(Y_hegger; K = K, w = w1)
Preds[3] = MCDTS.zeroth_prediction_cost(Y_kennel; K = K, w = w1)
Preds[4] = MCDTS.zeroth_prediction_cost(Y_mcdts; K = K, w = w1)
Preds[5] = MCDTS.zeroth_prediction_cost(Y_mcdts2; K = K, w = w1)
Preds[6] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED; K = K, w = w1)
Preds[7] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2; K = K, w = w1)
Preds[8] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2_5; K = K, w = w1)
Preds[9] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED_5; K = K, w = w1)
Preds[10] = MCDTS.zeroth_prediction_cost(Y_mcdts_fnn; K = K, w = w1)
Preds[11] = MCDTS.zeroth_prediction_cost(Y_mcdts_fnn2; K = K, w = w1)
Preds[12] = MCDTS.zeroth_prediction_cost(Y_pec; K = K, w = w1)
Preds[13] = MCDTS.zeroth_prediction_cost(Y_pec2; K = K, w = w1)

Preds_n[1] = MCDTS.zeroth_prediction_cost(Y_cao_n; K = K, w = w1_n)
Preds_n[2] = MCDTS.zeroth_prediction_cost(Y_hegger_n; K = K, w = w1_n)
Preds_n[3] = MCDTS.zeroth_prediction_cost(Y_kennel_n; K = K, w = w1_n)
Preds_n[4] = MCDTS.zeroth_prediction_cost(Y_mcdts_n; K = K, w = w1_n)
Preds_n[5] = MCDTS.zeroth_prediction_cost(Y_mcdts2_n; K = K, w = w1_n)
Preds_n[6] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED_n; K = K, w = w1_n)
Preds_n[7] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2_n; K = K, w = w1_n)
Preds_n[8] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED2_5_n; K = K, w = w1_n)
Preds_n[9] = MCDTS.zeroth_prediction_cost(Y_mcdts_PRED_5_n; K = K, w = w1_n)
Preds_n[10] = MCDTS.zeroth_prediction_cost(Y_mcdts_fnn_n; K = K, w = w1_n)
Preds_n[11] = MCDTS.zeroth_prediction_cost(Y_mcdts_fnn2_n; K = K, w = w1_n)
Preds_n[12] = MCDTS.zeroth_prediction_cost(Y_pec_n; K = K, w = w1_n)
Preds_n[13] = MCDTS.zeroth_prediction_cost(Y_pec2_n; K = K, w = w1_n)
