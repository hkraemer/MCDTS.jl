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
# and in the scripts stored in the folder `/Cluster scripts/MSE optimizing`. Here with
# a focus on the MCDTS method combined with optimizing the MSE

# choose from "L" (L-Statistic used for τ-preselection) or "U" (uniform τ-range)
method = "U"

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

    # Kennel method for comparison of results
    Y_kennel = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_kennel.csv"))
    Y_kennel_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_kennel_n.csv"))
    τ_kennel = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_kennel.csv")))
    τ_kennel_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_kennel_n.csv")))

    # MCDTS methods (noisefree)
    # -> univariate
    Y_mcdts_PRED_K_1_Tw_1_x_uni = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_1_x_uni.csv"))
    τ_mcdts_PRED_K_1_Tw_1_x_uni = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_1_x_uni.csv")))
    Y_mcdts_PRED_K_1_Tw_1_mean_uni = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_1_mean_uni.csv"))
    τ_mcdts_PRED_K_1_Tw_1_mean_uni = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_1_mean_uni.csv")))
    #
    Y_mcdts_PRED_K_5_Tw_1_x_uni = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_1_x_uni.csv"))
    τ_mcdts_PRED_K_5_Tw_1_x_uni = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_1_x_uni.csv")))
    Y_mcdts_PRED_K_5_Tw_1_mean_uni = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_1_mean_uni.csv"))
    τ_mcdts_PRED_K_5_Tw_1_mean_uni = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_1_mean_uni.csv")))
    #
    Y_mcdts_PRED_K_1_Tw_5_x_uni = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_5_x_uni.csv"))
    τ_mcdts_PRED_K_1_Tw_5_x_uni = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_5_x_uni.csv")))
    Y_mcdts_PRED_K_1_Tw_5_mean_uni = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_5_mean_uni.csv"))
    τ_mcdts_PRED_K_1_Tw_5_mean_uni = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_5_mean_uni.csv")))
    #
    Y_mcdts_PRED_K_5_Tw_5_x_uni = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_5_x_uni.csv"))
    τ_mcdts_PRED_K_5_Tw_5_x_uni = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_5_x_uni.csv")))
    Y_mcdts_PRED_K_5_Tw_5_mean_uni = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_5_mean_uni.csv"))
    τ_mcdts_PRED_K_5_Tw_5_mean_uni = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_5_mean_uni.csv")))
    #
    # # -> multivariate
    Y_mcdts_PRED_K_1_Tw_1_x_multi = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_1_x_multi.csv"))
    τ_mcdts_PRED_K_1_Tw_1_x_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_1_x_multi.csv")))
    ts_mcdts_PRED_K_1_Tw_1_x_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_1_Tw_1_x_multi.csv")))
    Y_mcdts_PRED_K_1_Tw_1_mean_multi = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_1_mean_multi.csv"))
    τ_mcdts_PRED_K_1_Tw_1_mean_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_1_mean_multi.csv")))
    ts_mcdts_PRED_K_1_Tw_1_mean_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_1_Tw_1_mean_multi.csv")))
    #
    Y_mcdts_PRED_K_5_Tw_1_x_multi = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_1_x_multi.csv"))
    τ_mcdts_PRED_K_5_Tw_1_x_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_1_x_multi.csv")))
    ts_mcdts_PRED_K_5_Tw_1_x_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_5_Tw_1_x_multi.csv")))
    Y_mcdts_PRED_K_5_Tw_1_mean_multi = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_1_mean_multi.csv"))
    τ_mcdts_PRED_K_5_Tw_1_mean_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_1_mean_multi.csv")))
    ts_mcdts_PRED_K_5_Tw_1_mean_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_5_Tw_1_mean_multi.csv")))
    #
    Y_mcdts_PRED_K_1_Tw_5_x_multi = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_5_x_multi.csv"))
    τ_mcdts_PRED_K_1_Tw_5_x_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_5_x_multi.csv")))
    ts_mcdts_PRED_K_1_Tw_5_x_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_1_Tw_5_x_multi.csv")))
    Y_mcdts_PRED_K_1_Tw_5_mean_multi = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_5_mean_multi.csv"))
    τ_mcdts_PRED_K_1_Tw_5_mean_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_5_mean_multi.csv")))
    ts_mcdts_PRED_K_1_Tw_5_mean_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_1_Tw_5_mean_multi.csv")))
    #
    Y_mcdts_PRED_K_5_Tw_5_x_multi = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_5_x_multi.csv"))
    τ_mcdts_PRED_K_5_Tw_5_x_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_5_x_multi.csv")))
    ts_mcdts_PRED_K_5_Tw_5_x_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_5_Tw_5_x_multi.csv")))
    # Y_mcdts_PRED_K_5_Tw_5_mean_multi = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_5_mean_multi.csv"))
    # τ_mcdts_PRED_K_5_Tw_5_mean_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_5_mean_multi.csv")))
    # ts_mcdts_PRED_K_5_Tw_5_mean_multi = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_5_Tw_5_mean_multi.csv")))
    #
    # # MCDTS methods (noisy)
    # # -> univariate
    Y_mcdts_PRED_K_1_Tw_1_x_uni_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_1_x_uni_n.csv"))
    τ_mcdts_PRED_K_1_Tw_1_x_uni_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_1_x_uni_n.csv")))
    Y_mcdts_PRED_K_1_Tw_1_mean_uni_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_1_mean_uni_n.csv"))
    τ_mcdts_PRED_K_1_Tw_1_mean_uni_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_1_mean_uni_n.csv")))

    Y_mcdts_PRED_K_5_Tw_1_x_uni_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_1_x_uni_n.csv"))
    τ_mcdts_PRED_K_5_Tw_1_x_uni_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_1_x_uni_n.csv")))
    Y_mcdts_PRED_K_5_Tw_1_mean_uni_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_1_mean_uni_n.csv"))
    τ_mcdts_PRED_K_5_Tw_1_mean_uni_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_1_mean_uni_n.csv")))

    Y_mcdts_PRED_K_1_Tw_5_x_uni_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_5_x_uni_n.csv"))
    τ_mcdts_PRED_K_1_Tw_5_x_uni_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_5_x_uni_n.csv")))
    Y_mcdts_PRED_K_1_Tw_5_mean_uni_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_5_mean_uni_n.csv"))
    τ_mcdts_PRED_K_1_Tw_5_mean_uni_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_5_mean_uni_n.csv")))

    Y_mcdts_PRED_K_5_Tw_5_x_uni_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_5_x_uni_n.csv"))
    τ_mcdts_PRED_K_5_Tw_5_x_uni_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_5_x_uni_n.csv")))
    Y_mcdts_PRED_K_5_Tw_5_mean_uni_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_5_mean_uni_n.csv"))
    τ_mcdts_PRED_K_5_Tw_5_mean_uni_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_5_mean_uni_n.csv")))
    #
    # # -> multivariate
    # Y_mcdts_PRED_K_1_Tw_1_x_multi_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_1_x_multi_n.csv"))
    # τ_mcdts_PRED_K_1_Tw_1_x_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_1_x_multi_n.csv")))
    # ts_mcdts_PRED_K_1_Tw_1_x_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_1_Tw_1_x_multi_n.csv")))
    Y_mcdts_PRED_K_1_Tw_1_mean_multi_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_1_mean_multi_n.csv"))
    τ_mcdts_PRED_K_1_Tw_1_mean_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_1_mean_multi_n.csv")))
    ts_mcdts_PRED_K_1_Tw_1_mean_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_1_Tw_1_mean_multi_n.csv")))
    #
    # Y_mcdts_PRED_K_5_Tw_1_x_multi_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_1_x_multi_n.csv"))
    # τ_mcdts_PRED_K_5_Tw_1_x_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_1_x_multi_n.csv")))
    # ts_mcdts_PRED_K_5_Tw_1_x_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_5_Tw_1_x_multi_n.csv")))
    Y_mcdts_PRED_K_5_Tw_1_mean_multi_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_1_mean_multi_n.csv"))
    τ_mcdts_PRED_K_5_Tw_1_mean_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_1_mean_multi_n.csv")))
    ts_mcdts_PRED_K_5_Tw_1_mean_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_5_Tw_1_mean_multi_n.csv")))

    # Y_mcdts_PRED_K_1_Tw_5_x_multi_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_5_x_multi_n.csv"))
    # τ_mcdts_PRED_K_1_Tw_5_x_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_5_x_multi_n.csv")))
    # ts_mcdts_PRED_K_1_Tw_5_x_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_1_Tw_5_x_multi_n.csv")))
    Y_mcdts_PRED_K_1_Tw_5_mean_multi_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_1_Tw_5_mean_multi_n.csv"))
    τ_mcdts_PRED_K_1_Tw_5_mean_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_1_Tw_5_mean_multi_n.csv")))
    ts_mcdts_PRED_K_1_Tw_5_mean_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_1_Tw_5_mean_multi_n.csv")))

    # Y_mcdts_PRED_K_5_Tw_5_x_multi_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_5_x_multi_n.csv"))
    # τ_mcdts_PRED_K_5_Tw_5_x_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_5_x_multi_n.csv")))
    # ts_mcdts_PRED_K_5_Tw_5_x_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_5_Tw_5_x_multi_n.csv")))
    # Y_mcdts_PRED_K_5_Tw_5_mean_multi_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/Y_mcdts_PRED_$(method)_K_5_Tw_5_mean_multi_n.csv"))
    # τ_mcdts_PRED_K_5_Tw_5_mean_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/tau_mcdts_PRED_$(method)_K_5_Tw_5_mean_multi_n.csv")))
    # ts_mcdts_PRED_K_5_Tw_5_mean_multi_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results MSE optimizing/ts_mcdts_PRED_$(method)_K_5_Tw_5_mean_multi_n.csv")))

end

# Theiler window:
w1 = 17
w1_n = 20

# Lyapunov time
lyap_time = 110
T_steps = 1700

## make predictions
# Different computations for different number of considered nearest neighbours
# the prediction (KK). Results for KK=1 look the best for all reconstruction
# methods.

# Results 3: zeroth and linear T_steps = 900, KK = 1
# Results 4: zeroth T_steps = 1700, KK = 1
# Results 5: zeroth T_steps = 1700, KK = 10
# Results 6: zeroth T_steps = 1700, KK = 5

# Predictions MSE Optimizing/Results L-U/K 1: zeroth T_steps = 1700, KK = 1
# Predictions MSE Optimizing/Results L-U/K 5: zeroth T_steps = 1700, KK = 5

KK1 = 1 # number of nearest neighbours for zeroth predictor
KK2 = 5

begin
    # iterated one step
    # Zeroth

    println("Noise free Uni T1, K1:")
    println("*****")
    # -> univariate
    ## mcdts_PRED_K_1_Tw_1_x_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_x_uni, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_1_x_uni.csv", mcdts_PRED_K_1_Tw_1_x_uni)
    ## mcdts_PRED_K_1_Tw_1_x_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_x_uni, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_1_x_uni.csv", mcdts_PRED_K_1_Tw_1_x_uni)
    ## mcdts_PRED_K_1_Tw_1_mean_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_mean_uni, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_1_mean_uni.csv", mcdts_PRED_K_1_Tw_1_mean_uni)
    ## mcdts_PRED_K_1_Tw_1_mean_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_mean_uni, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_1_mean_uni.csv", mcdts_PRED_K_1_Tw_1_mean_uni)

    println("Noise free Uni T1, K5:")
    println("*****")
    ## mcdts_PRED_K_5_Tw_1_x_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_x_uni, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_1_x_uni.csv", mcdts_PRED_K_5_Tw_1_x_uni)
    ## mcdts_PRED_K_5_Tw_1_x_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_x_uni, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_1_x_uni.csv", mcdts_PRED_K_5_Tw_1_x_uni)
    ## mcdts_PRED_K_5_Tw_1_mean_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_mean_uni, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_1_mean_uni.csv", mcdts_PRED_K_5_Tw_1_mean_uni)
    ## mcdts_PRED_K_5_Tw_1_mean_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_mean_uni, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_1_mean_uni.csv", mcdts_PRED_K_5_Tw_1_mean_uni)

    println("Noise free Uni T5, K1:")
    println("*****")
    ## mcdts_PRED_K_1_Tw_5_x_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_x_uni, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_5_x_uni.csv", mcdts_PRED_K_1_Tw_5_x_uni)
    ## mcdts_PRED_K_1_Tw_5_x_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_x_uni, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_5_x_uni.csv", mcdts_PRED_K_1_Tw_5_x_uni)
    ## mcdts_PRED_K_1_Tw_5_mean_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_mean_uni, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_5_mean_uni.csv", mcdts_PRED_K_1_Tw_5_mean_uni)
    ## mcdts_PRED_K_1_Tw_5_mean_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_mean_uni, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_5_mean_uni.csv", mcdts_PRED_K_1_Tw_5_mean_uni)

    println("Noise free Uni T5, K5:")
    println("*****")
    ## mcdts_PRED_K_5_Tw_5_x_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_x_uni, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_5_x_uni.csv", mcdts_PRED_K_5_Tw_5_x_uni)
    ## mcdts_PRED_K_5_Tw_5_x_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_x_uni, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_5_x_uni.csv", mcdts_PRED_K_5_Tw_5_x_uni)
    ## mcdts_PRED_K_5_Tw_5_mean_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_x_uni, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_5_mean_uni.csv", mcdts_PRED_K_5_Tw_5_mean_uni)
    ## mcdts_PRED_K_5_Tw_5_mean_uni = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_mean_uni, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_5_mean_uni.csv", mcdts_PRED_K_5_Tw_5_mean_uni)



    # -> multivariate
    println("Noise free Multi T1, K1:")
    println("*****")
    ## mcdts_PRED_K_1_Tw_1_x_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_x_multi, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_1_x_multi.csv", mcdts_PRED_K_1_Tw_1_x_multi)
    ## mcdts_PRED_K_1_Tw_1_x_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_x_multi, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_1_x_multi.csv", mcdts_PRED_K_1_Tw_1_x_multi)
    ## mcdts_PRED_K_1_Tw_1_mean_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_mean_multi, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_1_mean_multi.csv", mcdts_PRED_K_1_Tw_1_mean_multi)
    ## mcdts_PRED_K_1_Tw_1_mean_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_mean_multi, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_1_mean_multi.csv", mcdts_PRED_K_1_Tw_1_mean_multi)

    println("Noise free Multi T1, K5:")
    println("*****")
    ## mcdts_PRED_K_5_Tw_1_x_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_x_multi, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_1_x_multi.csv", mcdts_PRED_K_5_Tw_1_x_multi)
    ## mcdts_PRED_K_5_Tw_1_x_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_x_multi, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_1_x_multi.csv", mcdts_PRED_K_5_Tw_1_x_multi)
    ## mcdts_PRED_K_5_Tw_1_mean_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_mean_multi, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_1_mean_multi.csv", mcdts_PRED_K_5_Tw_1_mean_multi)
    ## mcdts_PRED_K_5_Tw_1_mean_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_mean_multi, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_1_mean_multi.csv", mcdts_PRED_K_5_Tw_1_mean_multi)



    println("Noise free Multi T5, K1:")
    println("*****")
    ## mcdts_PRED_K_1_Tw_5_x_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_x_multi, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_5_x_multi.csv", mcdts_PRED_K_1_Tw_5_x_multi)
    ## mcdts_PRED_K_1_Tw_5_x_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_x_multi, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_5_x_multi.csv", mcdts_PRED_K_1_Tw_5_x_multi)
    ## mcdts_PRED_K_1_Tw_5_mean_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_mean_multi, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_5_mean_multi.csv", mcdts_PRED_K_1_Tw_5_mean_multi)
    ## mcdts_PRED_K_1_Tw_5_mean_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_mean_multi, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_5_mean_multi.csv", mcdts_PRED_K_1_Tw_5_mean_multi)

    println("Noise free Multi T5, K5:")
    println("*****")
    ## mcdts_PRED_K_5_Tw_5_x_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_x_multi, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_5_x_multi.csv", mcdts_PRED_K_5_Tw_5_x_multi)
    ## mcdts_PRED_K_5_Tw_5_x_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_x_multi, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_5_x_multi.csv", mcdts_PRED_K_5_Tw_5_x_multi)
    # mcdts_PRED_K_5_Tw_5_mean_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_mean_multi, KK1, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_5_mean_multi.csv", mcdts_PRED_K_5_Tw_5_mean_multi)
    # mcdts_PRED_K_5_Tw_5_mean_multi = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_mean_multi, KK2, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_5_mean_multi.csv", mcdts_PRED_K_5_Tw_5_mean_multi)



    # MCDTS methods (noisy)
    println("Noisy Uni T1, K1:")
    println("*****")
    # -> univariate
    ## mcdts_PRED_K_1_Tw_1_x_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_x_uni_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_1_x_uni_n.csv", mcdts_PRED_K_1_Tw_1_x_uni_n)
    ## mcdts_PRED_K_1_Tw_1_x_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_x_uni_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_1_x_uni_n.csv", mcdts_PRED_K_1_Tw_1_x_uni_n)
    ## mcdts_PRED_K_1_Tw_1_mean_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_mean_uni_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_1_mean_uni_n.csv", mcdts_PRED_K_1_Tw_1_mean_uni_n)
    ## mcdts_PRED_K_1_Tw_1_mean_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_mean_uni_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_1_mean_uni_n.csv", mcdts_PRED_K_1_Tw_1_mean_uni_n)

    println("Noisy Uni T1, K5:")
    println("*****")
    ## mcdts_PRED_K_5_Tw_1_x_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_x_uni_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_1_x_uni_n.csv", mcdts_PRED_K_5_Tw_1_x_uni_n)
    ## mcdts_PRED_K_5_Tw_1_x_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_x_uni_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_1_x_uni_n.csv", mcdts_PRED_K_5_Tw_1_x_uni_n)
    ## mcdts_PRED_K_5_Tw_1_mean_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_mean_uni_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_1_mean_uni_n.csv", mcdts_PRED_K_5_Tw_1_mean_uni_n)
    ## mcdts_PRED_K_5_Tw_1_mean_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_mean_uni_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_1_mean_uni_n.csv", mcdts_PRED_K_5_Tw_1_mean_uni_n)

    println("Noisy Uni T5, K1:")
    println("*****")
    ## mcdts_PRED_K_1_Tw_5_x_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_x_uni_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_5_x_uni_n.csv", mcdts_PRED_K_1_Tw_5_x_uni_n)
    ## mcdts_PRED_K_1_Tw_5_x_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_x_uni_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_5_x_uni_n.csv", mcdts_PRED_K_1_Tw_5_x_uni_n)
    ## mcdts_PRED_K_1_Tw_5_mean_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_mean_uni_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_5_mean_uni_n.csv", mcdts_PRED_K_1_Tw_5_mean_uni_n)
    ## mcdts_PRED_K_1_Tw_5_mean_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_mean_uni_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_5_mean_uni_n.csv", mcdts_PRED_K_1_Tw_5_mean_uni_n)


    println("Noisy Uni T5, K5:")
    println("*****")
    ## mcdts_PRED_K_5_Tw_5_x_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_x_uni_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_5_x_uni_n.csv", mcdts_PRED_K_1_Tw_5_x_uni_n) # same embedding as with K1,Tw5
    ## mcdts_PRED_K_5_Tw_5_x_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_x_uni_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_5_x_uni_n.csv", mcdts_PRED_K_1_Tw_5_x_uni_n) # same embedding as with K1,Tw5
    ## mcdts_PRED_K_5_Tw_5_mean_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_mean_uni_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_5_mean_uni_n.csv", mcdts_PRED_K_5_Tw_5_mean_uni_n)
    ## mcdts_PRED_K_5_Tw_5_mean_uni_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_mean_uni_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_5_mean_uni_n.csv", mcdts_PRED_K_5_Tw_5_mean_uni_n)



    # -> multivariate
    println("Noisy Multi T1, K1:")
    println("*****")
    # mcdts_PRED_K_1_Tw_1_x_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_x_multi_n, KK1, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_1_x_multi_n.csv", mcdts_PRED_K_1_Tw_1_x_multi_n)
    # mcdts_PRED_K_1_Tw_1_x_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_x_multi_n, KK2, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_1_x_multi_n.csv", mcdts_PRED_K_1_Tw_1_x_multi_n)
    ## mcdts_PRED_K_1_Tw_1_mean_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_mean_multi_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_1_mean_multi_n.csv", mcdts_PRED_K_1_Tw_1_mean_multi_n)
    ## mcdts_PRED_K_1_Tw_1_mean_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_1_mean_multi_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_1_mean_multi_n.csv", mcdts_PRED_K_1_Tw_1_mean_multi_n)

    println("Noisy Multi T1, K5:")
    println("*****")
    # mcdts_PRED_K_5_Tw_1_x_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_x_multi_n, KK1, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_1_x_multi_n.csv", mcdts_PRED_K_5_Tw_1_x_multi_n)
    # mcdts_PRED_K_5_Tw_1_x_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_x_multi_n, KK2, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_1_x_multi_n.csv", mcdts_PRED_K_5_Tw_1_x_multi_n)
    ## mcdts_PRED_K_5_Tw_1_mean_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_mean_multi_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_1_mean_multi_n.csv", mcdts_PRED_K_5_Tw_1_mean_multi_n)
    ## mcdts_PRED_K_5_Tw_1_mean_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_1_mean_multi_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_1_mean_multi_n.csv", mcdts_PRED_K_5_Tw_1_mean_multi_n)

    println("Noisy Multi T5, K1:")
    println("*****")
    # mcdts_PRED_K_1_Tw_5_x_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_x_multi_n, KK1, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_5_x_multi_n.csv", mcdts_PRED_K_1_Tw_5_x_multi_n)
    # mcdts_PRED_K_1_Tw_5_x_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_x_multi_n, KK2, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_5_x_multi_n.csv", mcdts_PRED_K_1_Tw_5_x_multi_n)
    ## mcdts_PRED_K_1_Tw_5_mean_multi_n= MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_mean_multi_n, KK1, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_1_Tw_5_mean_multi_n.csv", mcdts_PRED_K_1_Tw_5_mean_multi_n)
    ## mcdts_PRED_K_1_Tw_5_mean_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_1_Tw_5_mean_multi_n, KK2, T_steps; theiler = w1, verbose=true)
    ## writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_1_Tw_5_mean_multi_n.csv", mcdts_PRED_K_1_Tw_5_mean_multi_n)

    println("Noisy Multi T5, K5:")
    println("*****")
    # mcdts_PRED_K_5_Tw_5_x_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_x_multi_n, KK1, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_5_x_multi_n.csv", mcdts_PRED_K_5_Tw_5_x_multi_n)
    # mcdts_PRED_K_5_Tw_5_x_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_x_multi_n, KK2, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_5_x_multi_n.csv", mcdts_PRED_K_5_Tw_5_x_multi_n)
    # mcdts_PRED_K_5_Tw_5_mean_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_mean_multi_n, KK1, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/mcdts_PRED_K_5_Tw_5_mean_multi_n.csv", mcdts_PRED_K_5_Tw_5_mean_multi_n)
    # mcdts_PRED_K_5_Tw_5_mean_multi_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_K_5_Tw_5_mean_multi_n, KK2, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 5/mcdts_PRED_K_5_Tw_5_mean_multi_n.csv", mcdts_PRED_K_5_Tw_5_mean_multi_n)

end

## Fake noise forecasts in order to understand goodnees of forecasts...
Y_fake = genembed(x1_n, [0,-2])
fake_1 = MCDTS.iterated_local_zeroth_prediction(Y_fake, 1, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/fake_1.csv", fake_1)
Y_fake2 = genembed(x1_n, [0,-1])
fake_2 = MCDTS.iterated_local_zeroth_prediction(Y_fake2, 1, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/fake_2.csv", fake_2)
Y_fake3 = genembed(x1_n, [0,-3])
fake_3 = MCDTS.iterated_local_zeroth_prediction(Y_fake3, 1, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/fake_3.csv", fake_3)
Y_fake4 = genembed(x1_n, [0,-4])
fake_4 = MCDTS.iterated_local_zeroth_prediction(Y_fake4, 1, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/fake_4.csv", fake_4)
Y_fake5 = genembed(x1_n, [0,-5])
fake_5 = MCDTS.iterated_local_zeroth_prediction(Y_fake5, 1, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/fake_5.csv", fake_5)
Y_fake6 = genembed(x1_n, [0,-6])
fake_6 = MCDTS.iterated_local_zeroth_prediction(Y_fake6, 1, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/fake_6.csv", fake_6)
Y_fake7 = genembed(x1_n, [0,-17])
fake_7 = MCDTS.iterated_local_zeroth_prediction(Y_fake7, 1, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/fake_7.csv", fake_7)
Y_fake8 = genembed(x1_n, [0,-11])
fake_8 = MCDTS.iterated_local_zeroth_prediction(Y_fake8, 1, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K 1/fake_8.csv", fake_8)

fakes = ["-2","-1","-3","-4","-5","-6","-17","-11"]
# Different computations for different number of considered nearest neighbours
# the prediction (KK). Results for KK=1 look the best for all reconstruction
# methods.

# Results 3: zeroth and linear T_steps = 900, KK = 1    (all reconstructions are stored here)
# Results 4: zeroth T_steps = 1700, KK = 1
# Results 5: zeroth T_steps = 1700, KK = 10
# Results 6: zeroth T_steps = 1700, KK = 5

# Predictions MSE Optimizing/Results L-U/K 1: zeroth T_steps = 1700, KK = 1
# Predictions MSE Optimizing/Results L-U/K 5: zeroth T_steps = 1700, KK = 5

# load data
KK = 1  # number of neighbours for the prediction (set to 1 or 5)
PREDICTIONS = zeros(32,T_steps)
FAKES = zeros(8,T_steps)

load = begin
    Kennel_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/Kennel_zeroth.csv"))
    Kennel_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/Kennel_zeroth_n.csv"))

    # --> Noisefree uni
    PREDICTIONS[1,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_1_x_uni.csv"))[:,1]
    PREDICTIONS[2,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_1_mean_uni.csv"))[:,1]
    PREDICTIONS[3,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_1_x_uni.csv"))[:,1]
    PREDICTIONS[4,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_1_mean_uni.csv"))[:,1]
    PREDICTIONS[5,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_5_x_uni.csv"))[:,1]
    PREDICTIONS[6,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_5_mean_uni.csv"))[:,1]
    PREDICTIONS[7,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_5_x_uni.csv"))[:,1]
    PREDICTIONS[8,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_5_mean_uni.csv"))[:,1]

    # --> Noisefree multi
    PREDICTIONS[9,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_1_x_multi.csv"))[:,1]
    PREDICTIONS[10,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_1_mean_multi.csv"))[:,1]
    PREDICTIONS[11,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_1_x_multi.csv"))[:,1]
    PREDICTIONS[12,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_1_mean_multi.csv"))[:,1]
    PREDICTIONS[13,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_5_x_multi.csv"))[:,1]
    PREDICTIONS[14,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_5_mean_multi.csv"))[:,1]
    PREDICTIONS[15,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_5_x_multi.csv"))[:,1]
    # PREDICTIONS[16,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_5_mean_multi.csv"))[:,1]

    # --> Noisy uni
    PREDICTIONS[17,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_1_x_uni_n.csv"))[:,1]
    PREDICTIONS[18,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_1_mean_uni_n.csv"))[:,1]
    PREDICTIONS[19,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_1_x_uni_n.csv"))[:,1]
    PREDICTIONS[20,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_1_mean_uni_n.csv"))[:,1]
    PREDICTIONS[21,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_5_x_uni_n.csv"))[:,1]
    PREDICTIONS[22,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_5_mean_uni_n.csv"))[:,1]
    PREDICTIONS[23,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_5_x_uni_n.csv"))[:,1]
    PREDICTIONS[24,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_5_mean_uni_n.csv"))[:,1]

    # --> Noisy multi
    # PREDICTIONS[25,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_1_x_multi_n.csv"))[:,1]
    PREDICTIONS[26,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_1_mean_multi_n.csv"))[:,1]
    # PREDICTIONS[27,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_1_x_multi_n.csv"))[:,1]
    PREDICTIONS[28,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_1_mean_multi_n.csv"))[:,1]
    # PREDICTIONS[29,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_5_x_multi_n.csv"))[:,1]
    PREDICTIONS[30,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_1_Tw_5_mean_multi_n.csv"))[:,1]
    # PREDICTIONS[31,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_5_x_multi_n.csv"))[:,1]
    # PREDICTIONS[32,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/mcdts_PRED_K_5_Tw_5_mean_multi_n.csv"))[:,1]


    # fake set
    FAKES[1,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/fake_1.csv"))[:,1]
    FAKES[2,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/fake_2.csv"))[:,1]
    FAKES[3,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/fake_3.csv"))[:,1]
    FAKES[4,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/fake_4.csv"))[:,1]
    FAKES[5,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/fake_5.csv"))[:,1]
    FAKES[6,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/fake_6.csv"))[:,1]
    FAKES[7,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/fake_7.csv"))[:,1]
    FAKES[8,:] = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Predictions MSE optimizing/Results $(method)/K $(KK)/fake_8.csv"))[:,1]

end


# time axis
t2 = (0:T_steps-1) ./lyap_time
t1 = (-length(x1):-1) ./lyap_time
NN = 1000
tt = vcat(t1[end-NN:end], t2)
M = length(tt)
true_data = vcat(x1_[end-NN:end], x2_)
true_data_n = vcat(x1_n_[end-NN:end], x2_n_)

## MSEs
# compute MSE of predictions

types = ["K=1, Tw=1, x-var, UNI", "K=1, Tw=1, mean, UNI", "K=5, Tw=1, x-var, UNI", "K=5, Tw=1, mean, UNI",
        "K=1, Tw=5, x-var, UNI", "K=1, Tw=5, mean, UNI", "K=5, Tw=5, x-var, UNI", "K=5, Tw=5, mean, UNI",
        "K=1, Tw=1, x-var, MULTI", "K=1, Tw=1, mean, MULTI", "K=5, Tw=1, x-var, MULTI", "K=5, Tw=1, mean, MULTI",
        "K=1, Tw=5, x-var, MULTI", "K=1, Tw=5, mean, MULTI", "K=5, Tw=5, x-var, MULTI", "K=5, Tw=5, mean, MULTI",
        "K=1, Tw=1, x-var, UNI noise", "K=1, Tw=1, mean, UNI noise", "K=5, Tw=1, x-var, UNI noise", "K=5, Tw=1, mean, UNI noise",
        "K=1, Tw=5, x-var, UNI noise", "K=1, Tw=5, mean, UNI noise", "K=5, Tw=5, x-var, UNI noise", "K=5, Tw=5, mean, UNI noise",
        "K=1, Tw=1, x-var, MULTI noise", "K=1, Tw=1, mean, MULTI noise", "K=5, Tw=1, x-var, MULTI noise", "K=5, Tw=1, mean, MULTI noise",
        "K=1, Tw=5, x-var, MULTI noise", "K=1, Tw=5, mean, MULTI noise", "K=5, Tw=5, x-var, MULTI noise", "K=5, Tw=5, mean, MULTI noise"]

begin
    MSE_zeroth_kennel = zeros(T_steps)
    MSE_zeroth_kennel_n = zeros(T_steps)
    MSEs = zeros(32,T_steps)
    MSE_fakes = zeros(8,T_steps)

    σ₂ = sqrt(var(x2_[1:T_steps]))   # rms deviation for normalization
    σ₂_n = sqrt(var(x2_n_[1:T_steps]))

    for i = 1:T_steps
        # normalized MSE error
        MSE_zeroth_kennel[i] = MCDTS.compute_mse(Kennel_zeroth[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_kennel_n[i] = MCDTS.compute_mse(Kennel_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂_n

        for j = 1:32
            # --> Noisefree uni
            if j < 17
                MSEs[j,i] = MCDTS.compute_mse(PREDICTIONS[j,1:i], x2_[1:i]) / σ₂
            else
                MSEs[j,i] = MCDTS.compute_mse(PREDICTIONS[j,1:i], x2_n_[1:i]) / σ₂_n
            end
        end

        for j = 1:8
            MSE_fakes[j,i] = MCDTS.compute_mse(FAKES[j,1:i], x2_n_[1:i]) / σ₂_n
        end
    end

    markers = ["v", "s", "P", "p", "*", "+", "X", "D",
            "v", "s", "P", "p", "*", "+", "X", "D",
            "v", "s", "P", "p", "*", "+", "X", "D",
            "v", "s", "P", "p", "*", "+", "X", "D"]

    colorss = ["b", "b", "m", "m", "g", "g", "k", "k",
            "b", "b", "m", "m", "g", "g", "k", "k",
            "b", "b", "m", "m", "g", "g", "k", "k",
            "b", "b", "m", "m", "g", "g", "k", "k"]

    sc = 10 # markersize

    # Plot MSEs
    figure(figsize=(20,10))
    subplot(221)
    plot(t2[1:T_steps], MSE_zeroth_kennel, "r")
    scatter(t2[1:T_steps], MSE_zeroth_kennel, color="r", marker = "o", label="Kennel")
    for i = 1:8
        plot(t2[1:T_steps], MSEs[i,:], colorss[i])
        scatter(t2[1:T_steps], MSEs[i,:], s = sc, color=colorss[i], marker = markers[i], label=types[i])
    end
    legend()
    title("Forecast Error (ZEROTH predictor) univariate")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.5)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()

    subplot(222)
    plot(t2[1:T_steps], MSE_zeroth_kennel, "r")
    scatter(t2[1:T_steps], MSE_zeroth_kennel, color="r", marker = "o", label="Kennel")
    for i = 9:16
        plot(t2[1:T_steps], MSEs[i,:], colorss[i])
        scatter(t2[1:T_steps], MSEs[i,:], s = sc, color=colorss[i], marker = markers[i], label=types[i])
    end
    legend()
    title("Forecast Error (ZEROTH predictor) multivariate")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.5)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()

    subplot(223)
    plot(t2[1:T_steps], MSE_zeroth_kennel_n, "r")
    scatter(t2[1:T_steps], MSE_zeroth_kennel_n, color="r", marker = "o", label="Kennel")
    for i = 17:24
        plot(t2[1:T_steps], MSEs[i,:], colorss[i])
        scatter(t2[1:T_steps], MSEs[i,:], s = sc, color=colorss[i], marker = markers[i], label=types[i])
    end
    legend()
    title("Forecast Error (ZEROTH predictor) univariate noise")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.5)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()

    subplot(224)
    plot(t2[1:T_steps], MSE_zeroth_kennel_n, "r")
    scatter(t2[1:T_steps], MSE_zeroth_kennel_n, color="r", marker = "o", label="Kennel")
    for i = 25:32
        plot(t2[1:T_steps], MSEs[i,:], colorss[i])
        scatter(t2[1:T_steps], MSEs[i,:], s = sc, color=colorss[i], marker = markers[i], label=types[i])
    end
    legend()
    title("Forecast Error (ZEROTH predictor) multivariate noise")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.5)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()

    figure(figsize=(20,10))
    plot(t2[1:T_steps], MSE_zeroth_kennel_n, "r")
    scatter(t2[1:T_steps], MSE_zeroth_kennel_n, color="r", marker = "o", label="Kennel")
    for i = 1:8
        plot(t2[1:T_steps], MSE_fakes[i,:], colorss[i])
        scatter(t2[1:T_steps], MSE_fakes[i,:], s = sc, color=colorss[i], marker = markers[i], label=fakes[i])
    end
    legend()
    title("Forecast Error (ZEROTH predictor) univariate noise")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.5)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()

end

begin
    figure()
    plot(t2[1:T_steps], MSE_zeroth_kennel_n, "r")
    scatter(t2[1:T_steps], MSE_zeroth_kennel_n, color="r", marker = "o", label="Kennel")

    plot(t2[1:T_steps], MSE_fakes[1,:], colorss[1])
    scatter(t2[1:T_steps], MSE_fakes[1,:], s = sc, color=colorss[1], marker = markers[1], label=fakes[1])
    plot(t2[1:T_steps], MSE_fakes[2,:], colorss[2])
    scatter(t2[1:T_steps], MSE_fakes[2,:], s = sc, color=colorss[2], marker = markers[2], label=fakes[2])


    plot(t2[1:T_steps], MSE_fakes[3,:], colorss[3])
    scatter(t2[1:T_steps], MSE_fakes[3,:], s = sc, color=colorss[3], marker = markers[3], label=fakes[3])
    plot(t2[1:T_steps], MSE_fakes[4,:], colorss[4])
    scatter(t2[1:T_steps], MSE_fakes[4,:], s = sc, color=colorss[4], marker = markers[4], label=fakes[4])

    plot(t2[1:T_steps], MSE_fakes[5,:], colorss[5])
    scatter(t2[1:T_steps], MSE_fakes[5,:], s = sc, color=colorss[5], marker = markers[5], label=fakes[5])
    plot(t2[1:T_steps], MSE_fakes[6,:], colorss[6])
    scatter(t2[1:T_steps], MSE_fakes[6,:], s = sc, color=colorss[6], marker = markers[6], label=fakes[6])

    legend()
    title("Forecast Error (ZEROTH predictor) multivariate noise")
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
    title("x-component (zeroth - iterated one-step) univariate")
    for i = 1:8
        subplot(8,1,i)
        if i == 1
            plot(tt, true_data, ".-", label="true data")
        else
            plot(tt, true_data, ".-")
        end
        plot(t2, PREDICTIONS[i,:], ".-", label=types[i])
        xlim(-.5, T_steps/lyap_time)
        ylim(ylim1,ylim2)
        legend(loc=2)
        grid()
    end

    figure(figsize=(20,10))
    cnt = 1
    title("x-component (zeroth - iterated one-step) multivariate")
    for i = 9:16
        subplot(8,1,cnt)
        if cnt == 1
            plot(tt, true_data, ".-", label="true data")
        else
            plot(tt, true_data, ".-")
        end
        plot(t2, PREDICTIONS[i,:], ".-", label=types[i])
        xlim(-4, T_steps/lyap_time)
        ylim(ylim1,ylim2)
        legend(loc=2)
        grid()
        cnt += 1
    end


    figure(figsize=(20,10))
    cnt = 1
    title("x-component (zeroth - iterated one-step) univariate noise")
    for i = 17:24
        subplot(8,1,cnt)
        if cnt == 1
            plot(tt, true_data, ".-", label="true data")
        else
            plot(tt, true_data, ".-")
        end
        plot(t2, PREDICTIONS[i,:], ".-", label=types[i])
        xlim(-4, T_steps/lyap_time)
        ylim(ylim1,ylim2)
        legend(loc=2)
        grid()
        cnt += 1
    end


    figure(figsize=(20,10))
    cnt = 1
    title("x-component (zeroth - iterated one-step) multivariate noise")
    for i = 25:32
        subplot(8,1,cnt)
        if cnt == 1
            plot(tt, true_data, ".-", label="true data")
        else
            plot(tt, true_data, ".-")
        end
        plot(t2, PREDICTIONS[i,:], ".-", label=types[i])
        xlim(-4, T_steps/lyap_time)
        ylim(ylim1,ylim2)
        legend(loc=2)
        grid()
        cnt += 1
    end


end

## Save variables in order to plot nicely in Matlab

## TODO here updating

writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/t1.csv",t1)
writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/t2.csv",t2)
writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/NN.csv",NN)
writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/tt.csv",tt)
writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/M.csv",M)
writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/true_data.csv",true_data)
writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/true_data_n.csv",true_data_n)
writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/T_steps.csv",T_steps)
writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/lyap_time.csv",lyap_time)

recons = ["Cao", "Hegger", "Kennel", "MCDTS", "MCDTS mult.","MCDTS PRED","MCDTS PRED mult.",
        "MCDTS PRED mult. 5", "MCDTS PRED 5", "MCDTS FNN", "MCDTS FNN mult.", "PECUZAL", "PECUZAL mult."]

writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/recons.csv",recons)

# Pool all MSE values and save the matrix
forecast_errors = zeros(13,T_steps)
forecast_errors_n = zeros(13,T_steps)

forecast_errors[1,:] = MSE_zeroth_cao
forecast_errors[2,:] = MSE_zeroth_hegger
forecast_errors[3,:] = MSE_zeroth_kennel
forecast_errors[4,:] = MSE_zeroth_mcdts
forecast_errors[5,:] = MSE_zeroth_mcdts2
forecast_errors[6,:] = MSE_zeroth_mcdts_PRED
forecast_errors[7,:] = MSE_zeroth_mcdts_PRED2
forecast_errors[8,:] = MSE_zeroth_mcdts_PRED2_5
forecast_errors[9,:] = MSE_zeroth_mcdts_PRED_5
forecast_errors[10,:] = MSE_zeroth_mcdts_fnn
forecast_errors[11,:] = MSE_zeroth_mcdts_fnn2
forecast_errors[12,:] = MSE_zeroth_pec
forecast_errors[13,:] = MSE_zeroth_pec2

forecast_errors_n[1,:] = MSE_zeroth_cao_n
forecast_errors_n[2,:] = MSE_zeroth_hegger_n
forecast_errors_n[3,:] = MSE_zeroth_kennel_n
forecast_errors_n[4,:] = MSE_zeroth_mcdts_n
forecast_errors_n[5,:] = MSE_zeroth_mcdts2_n
forecast_errors_n[6,:] = MSE_zeroth_mcdts_PRED_n
forecast_errors_n[7,:] = MSE_zeroth_mcdts_PRED2_n
forecast_errors_n[8,:] = MSE_zeroth_mcdts_PRED2_5_n
forecast_errors_n[9,:] = MSE_zeroth_mcdts_PRED_5_n
forecast_errors_n[10,:] = MSE_zeroth_mcdts_fnn_n
forecast_errors_n[11,:] = MSE_zeroth_mcdts_fnn2_n
forecast_errors_n[12,:] = MSE_zeroth_pec_n
forecast_errors_n[13,:] = MSE_zeroth_pec2_n

writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/forecast_errors.csv",forecast_errors)
writedlm("./application/artificial data/Lorenz Prediction/Results for plotting/forecast_errors_n.csv",forecast_errors_n)




### TESTING AND DEBUGGING
using MCDTS
using DelayEmbeddings
Tw = 1  # time horizon
KK = 5 # considered nearest neighbors
PRED_L = false
PRED_mean = false
trials = 1
max_depth = 30
taus = 0:25

@time tree = MCDTS.mc_delay(data_sample_n,w1_n,(L)->(MCDTS.softmaxL(L,β=2.)),
    taus, trials; max_depth = max_depth, PRED = true, verbose = true, KNN = KK,
    Tw = Tw, threshold = 5e-6, PRED_L = PRED_L, PRED_mean = PRED_mean)
best_node = MCDTS.best_embedding(tree)
τ_mcdts_PRED = best_node.τs
Y_mcdts_PRED = MCDTS.genembed_for_prediction(x1, τ_mcdts_PRED)
