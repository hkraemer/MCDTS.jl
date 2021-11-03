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
KK = 1 # number of nearest neighbours for zeroth predictor

begin
    # iterated one step
    # Zeroth
    # println("Cao:")
    # println("*****")
    # Cao_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_cao, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/Cao_zeroth.csv",Cao_zeroth)
    # Cao_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_cao_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/Cao_zeroth_n.csv",Cao_zeroth_n)
    # println("Kennel")
    # println("*****")
    # Kennel_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_kennel, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/Kennel_zeroth.csv",Kennel_zeroth)
    # Kennel_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_kennel_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/Kennel_zeroth_n.csv",Kennel_zeroth_n)
    # println("Hegger")
    # println("*****")
    # Hegger_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_hegger, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/Hegger_zeroth.csv",Hegger_zeroth)
    # Hegger_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_hegger_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/Hegger_zeroth_n.csv",Hegger_zeroth_n)
    # println("Pec")
    # println("*****")
    # Pec_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_pec, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/Pec_zeroth.csv",Pec_zeroth)
    # Pec_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_pec2, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/Pec_zeroth2.csv",Pec_zeroth2)
    # Pec_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_pec_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/Pec_zeroth_n.csv",Pec_zeroth_n)
    # Pec_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_pec2_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/Pec_zeroth2_n.csv", Pec_zeroth2_n)
    # println("mcdts:")
    # println("*****")
    # mcdts_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_mcdts, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_zeroth.csv",mcdts_zeroth)
    # mcdts_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_mcdts2, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_zeroth2.csv",mcdts_zeroth2)
    # mcdts_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_zeroth_n.csv",mcdts_zeroth_n)
    # mcdts_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts2_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_zeroth2_n.csv",mcdts_zeroth2_n)
    # println("mcdts FNN:")
    # println("*****")
    # mcdts_fnn_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_fnn_zeroth.csv",mcdts_fnn_zeroth)
    # mcdts_fnn_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn2, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_fnn_zeroth2.csv",mcdts_fnn_zeroth2)
    # mcdts_fnn_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn_n,  KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_fnn_zeroth_n.csv",mcdts_fnn_zeroth_n)
    # mcdts_fnn_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn2_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_fnn_zeroth.csv",mcdts_fnn_zeroth2_n)
    # println("mcdts PRED:")
    # println("*****")
    # mcdts_PRED_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_PRED_zeroth.csv",mcdts_PRED_zeroth)
    # mcdts_PRED_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED2, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_PRED_zeroth2.csv",mcdts_PRED_zeroth2)
    # mcdts_PRED_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_n,  KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_PRED_zeroth_n.csv",mcdts_PRED_zeroth_n)
    # mcdts_PRED_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED2_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_PRED_zeroth2_n.csv",mcdts_PRED_zeroth2_n)
    # println("mcdts PRED 5:")
    # println("*****")
    # mcdts_PRED_5_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_5, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_PRED_5_zeroth.csv",mcdts_PRED_5_zeroth)
    # mcdts_PRED_5_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED2_5, KK, T_steps; theiler = w1, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_PRED_5_zeroth2.csv",mcdts_PRED_5_zeroth2)
    # mcdts_PRED_5_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED_5_n,  KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_PRED_5_zeroth_n.csv",mcdts_PRED_5_zeroth_n)
    # mcdts_PRED_5_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_PRED2_5_n, KK, T_steps; theiler = w1_n, verbose=true)
    # writedlm("./application/artificial data/Lorenz Prediction/Results 4/mcdts_PRED_5_zeroth2_n.csv",mcdts_PRED_5_zeroth2_n)

end

# Different computations for different number of considered nearest neighbours
# the prediction (KK). Results for KK=1 look the best for all reconstruction
# methods.

# Results 3: zeroth and linear T_steps = 900, KK = 1    (all reconstructions are stored here)
# Results 4: zeroth T_steps = 1700, KK = 1
# Results 5: zeroth T_steps = 1700, KK = 10
# Results 6: zeroth T_steps = 1700, KK = 5

# load data
Numbers = 4     # We only consider the case KK=1
load = begin
    Cao_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/Cao_zeroth.csv"))
    Cao_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/Cao_zeroth_n.csv"))
    Kennel_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/Kennel_zeroth.csv"))
    Kennel_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/Kennel_zeroth_n.csv"))
    Hegger_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/Hegger_zeroth.csv"))
    Hegger_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/Hegger_zeroth_n.csv"))
    Pec_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/Pec_zeroth.csv"))
    Pec_zeroth2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/Pec_zeroth2.csv"))
    Pec_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/Pec_zeroth_n.csv"))
    Pec_zeroth2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/Pec_zeroth2_n.csv"))
    mcdts_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_zeroth.csv"))
    mcdts_zeroth2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_zeroth2.csv"))
    mcdts_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_zeroth_n.csv"))
    mcdts_zeroth2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_zeroth2_n.csv"))
    mcdts_fnn_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_fnn_zeroth.csv"))
    mcdts_fnn_zeroth2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_fnn_zeroth2.csv"))
    mcdts_fnn_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_fnn_zeroth_n.csv"))
    mcdts_fnn_zeroth2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_fnn_zeroth.csv"))
    mcdts_PRED_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_PRED_zeroth.csv"))
    mcdts_PRED_zeroth2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_PRED_zeroth2.csv"))
    mcdts_PRED_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_PRED_zeroth_n.csv"))
    mcdts_PRED_zeroth2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_PRED_zeroth2_n.csv"))
    mcdts_PRED_5_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_PRED_5_zeroth.csv"))
    mcdts_PRED_5_zeroth2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_PRED_5_zeroth2.csv"))
    mcdts_PRED_5_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_PRED_5_zeroth_n.csv"))
    mcdts_PRED_5_zeroth2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Numbers/mcdts_PRED_5_zeroth2_n.csv"))
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
begin
    MSE_zeroth_cao = zeros(T_steps)
    MSE_zeroth_cao_n = zeros(T_steps)
    MSE_zeroth_kennel = zeros(T_steps)
    MSE_zeroth_kennel_n = zeros(T_steps)
    MSE_zeroth_hegger = zeros(T_steps)
    MSE_zeroth_hegger_n = zeros(T_steps)
    MSE_zeroth_pec = zeros(T_steps)
    MSE_zeroth_pec_n = zeros(T_steps)
    MSE_zeroth_pec2 = zeros(T_steps)
    MSE_zeroth_pec2_n = zeros(T_steps)
    MSE_zeroth_mcdts = zeros(T_steps)
    MSE_zeroth_mcdts_n = zeros(T_steps)
    MSE_zeroth_mcdts2 = zeros(T_steps)
    MSE_zeroth_mcdts2_n = zeros(T_steps)
    MSE_zeroth_mcdts_fnn = zeros(T_steps)
    MSE_zeroth_mcdts_fnn_n = zeros(T_steps)
    MSE_zeroth_mcdts_fnn2 = zeros(T_steps)
    MSE_zeroth_mcdts_fnn2_n = zeros(T_steps)

    MSE_zeroth_mcdts_PRED = zeros(T_steps)
    MSE_zeroth_mcdts_PRED_n = zeros(T_steps)
    MSE_zeroth_mcdts_PRED2 = zeros(T_steps)
    MSE_zeroth_mcdts_PRED2_n = zeros(T_steps)

    MSE_zeroth_mcdts_PRED_5 = zeros(T_steps)
    MSE_zeroth_mcdts_PRED_5_n = zeros(T_steps)
    MSE_zeroth_mcdts_PRED2_5 = zeros(T_steps)
    MSE_zeroth_mcdts_PRED2_5_n = zeros(T_steps)

    σ₂ = sqrt(var(x2_[1:T_steps]))   # rms deviation for normalization
    σ₂_n = sqrt(var(x2_n_[1:T_steps]))

    for i = 1:T_steps
        # normalized MSE error
        MSE_zeroth_cao[i] = MCDTS.compute_mse(Cao_zeroth[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_kennel[i] = MCDTS.compute_mse(Kennel_zeroth[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_hegger[i] = MCDTS.compute_mse(Hegger_zeroth[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_pec[i] = MCDTS.compute_mse(Pec_zeroth[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_pec2[i] = MCDTS.compute_mse(Pec_zeroth2[1:i,2], x2_[1:i]) / σ₂
        MSE_zeroth_mcdts[i] = MCDTS.compute_mse(mcdts_zeroth[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_mcdts2[i] = MCDTS.compute_mse(mcdts_zeroth2[1:i,2], x2_[1:i]) / σ₂
        MSE_zeroth_mcdts_fnn[i] = MCDTS.compute_mse(mcdts_fnn_zeroth[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_mcdts_fnn2[i] = MCDTS.compute_mse(mcdts_fnn_zeroth2[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_cao_n[i] = MCDTS.compute_mse(Cao_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂_n
        MSE_zeroth_kennel_n[i] = MCDTS.compute_mse(Kennel_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂_n
        MSE_zeroth_hegger_n[i] = MCDTS.compute_mse(Hegger_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂_n
        MSE_zeroth_pec_n[i] = MCDTS.compute_mse(Pec_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂_n
        MSE_zeroth_pec2_n[i] = MCDTS.compute_mse(Pec_zeroth2_n[1:i,1], x2_n_[1:i]) / σ₂_n
        MSE_zeroth_mcdts_n[i] = MCDTS.compute_mse(mcdts_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂_n
        MSE_zeroth_mcdts2_n[i] = MCDTS.compute_mse(mcdts_zeroth2_n[1:i,1], x2_n_[1:i]) / σ₂_n
        MSE_zeroth_mcdts_fnn_n[i] = MCDTS.compute_mse(mcdts_fnn_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂_n
        MSE_zeroth_mcdts_fnn2_n[i] = MCDTS.compute_mse(mcdts_fnn_zeroth2_n[1:i,1], x2_n_[1:i]) / σ₂_n

        MSE_zeroth_mcdts_PRED[i] = MCDTS.compute_mse(mcdts_PRED_zeroth[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_mcdts_PRED2[i] = MCDTS.compute_mse(mcdts_PRED_zeroth2[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_mcdts_PRED_n[i] = MCDTS.compute_mse(mcdts_PRED_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂_n
        MSE_zeroth_mcdts_PRED2_n[i] = MCDTS.compute_mse(mcdts_PRED_zeroth2_n[1:i,1], x2_n_[1:i]) / σ₂_n

        MSE_zeroth_mcdts_PRED_5[i] = MCDTS.compute_mse(mcdts_PRED_5_zeroth[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_mcdts_PRED2_5[i] = MCDTS.compute_mse(mcdts_PRED_5_zeroth2[1:i,1], x2_[1:i]) / σ₂
        MSE_zeroth_mcdts_PRED_5_n[i] = MCDTS.compute_mse(mcdts_PRED_5_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂_n
        MSE_zeroth_mcdts_PRED2_5_n[i] = MCDTS.compute_mse(mcdts_PRED_5_zeroth2_n[1:i,1], x2_n_[1:i]) / σ₂_n

    end

    # Plot MSEs
    figure(figsize=(20,10))
    subplot(121)
    plot(t2[1:T_steps], MSE_zeroth_mcdts2, "r")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts2, color="r", marker = "o", label="MCDTS L 2")
    plot(t2[1:T_steps], MSE_zeroth_mcdts, "r")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts, color="r", marker = "d", label="MCDTS L")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn2, "m")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_fnn2, color="m", marker = "*", label="MCDTS FNN 2")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn, "m")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_fnn, color="m", marker = "<", label="MCDTS FNN")
    plot(t2[1:T_steps], MSE_zeroth_pec2, "y")
    scatter(t2[1:T_steps], MSE_zeroth_pec2, color="y", marker = "v", label="PECUZAL 2")
    plot(t2[1:T_steps], MSE_zeroth_pec, "y")
    scatter(t2[1:T_steps], MSE_zeroth_pec, color="y", marker = "1", label="PECUZAL")
    plot(t2[1:T_steps], MSE_zeroth_cao, "b")
    scatter(t2[1:T_steps], MSE_zeroth_cao, color="b", marker = "3", label="CAO")
    plot(t2[1:T_steps], MSE_zeroth_kennel, "b")
    scatter(t2[1:T_steps], MSE_zeroth_kennel, color="b", marker = "s", label="Kennel")
    plot(t2[1:T_steps], MSE_zeroth_hegger, "b")
    scatter(t2[1:T_steps], MSE_zeroth_hegger, color="b", marker = "p", label="Hegger")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED2, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED2, color="g", marker = "+", label="MCDTS PRED 2")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED, color="g", marker = "p", label="MCDTS PRED")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_5, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_5, color="g", marker = ".", label="MCDTS PRED 2 5 Tw")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED_5, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED_5, color="g", marker = "D", label="MCDTS PRED 5 Tw")
    legend()
    title("Forecast Error (ZEROTH predictor)")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.1)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()

    subplot(122)
    plot(t2[1:T_steps], MSE_zeroth_mcdts2_n, "r")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts2_n, color="r", marker = "o", label="MCDTS L 2")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_n, "r")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_n, color="r", marker = "d", label="MCDTS L")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn2_n, "m")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_fnn2_n, color="m", marker = "*", label="MCDTS FNN 2")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn_n, "m")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_fnn_n, color="m", marker = "<", label="MCDTS FNN")
    plot(t2[1:T_steps], MSE_zeroth_pec2_n, "y")
    scatter(t2[1:T_steps], MSE_zeroth_pec2_n, color="y", marker = "v", label="PECUZAL 2")
    plot(t2[1:T_steps], MSE_zeroth_pec_n, "y")
    scatter(t2[1:T_steps], MSE_zeroth_pec_n, color="y", marker = "1", label="PECUZAL")
    plot(t2[1:T_steps], MSE_zeroth_cao_n, "b")
    scatter(t2[1:T_steps], MSE_zeroth_cao_n, color="b", marker = "3", label="CAO")
    plot(t2[1:T_steps], MSE_zeroth_kennel_n, "b")
    scatter(t2[1:T_steps], MSE_zeroth_kennel_n, color="b", marker = "s", label="Kennel")
    plot(t2[1:T_steps], MSE_zeroth_hegger_n, "b")
    scatter(t2[1:T_steps], MSE_zeroth_hegger_n, color="b", marker = "p", label="Hegger")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_n, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_n, color="g", marker = "+", label="MCDTS PRED 2")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED_n, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED_n, color="g", marker = "p", label="MCDTS PRED")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_5_n, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_5_n, color="g", marker = ".", label="MCDTS PRED 2 5 Tw")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED_5_n, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED_5_n, color="g", marker = "D", label="MCDTS PRED 5 Tw")
    legend()
    title("Forecast Error of noisy time series (ZEROTH predictor)")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.1)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()

end

begin
    figure(figsize=(20,10))
    subplot(121)
    # plot(t2[1:T_steps], MSE_zeroth_mcdts2, "r")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts2, color="r", marker = "o", label="MCDTS L 2")
    # plot(t2[1:T_steps], MSE_zeroth_mcdts, "r")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts, color="r", marker = "d", label="MCDTS L")
    # plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn2, "m")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts_fnn2, color="m", marker = "*", label="MCDTS FNN 2")
    # plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn, "m")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts_fnn, color="m", marker = "<", label="MCDTS FNN")
    # plot(t2[1:T_steps], MSE_zeroth_pec2, "y")
    # scatter(t2[1:T_steps], MSE_zeroth_pec2, color="y", marker = "v", label="PECUZAL 2")
    # plot(t2[1:T_steps], MSE_zeroth_pec, "y")
    # scatter(t2[1:T_steps], MSE_zeroth_pec, color="y", marker = "1", label="PECUZAL")
    # plot(t2[1:T_steps], MSE_zeroth_cao, "b")
    # scatter(t2[1:T_steps], MSE_zeroth_cao, color="b", marker = "3", label="CAO")
    plot(t2[1:T_steps], MSE_zeroth_kennel, "b")
    scatter(t2[1:T_steps], MSE_zeroth_kennel, color="b", marker = "s", label="Kennel")
    # plot(t2[1:T_steps], MSE_zeroth_hegger, "b")
    # scatter(t2[1:T_steps], MSE_zeroth_hegger, color="b", marker = "p", label="Hegger")
    # plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED2, "g")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED2, color="g", marker = "+", label="MCDTS PRED 2")
    # plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED, "g")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED, color="g", marker = "p", label="MCDTS PRED")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_5, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_5, color="g", marker = ".", label="MCDTS PRED 2 5 Tw")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED_5, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED_5, color="g", marker = "D", label="MCDTS PRED 5 Tw")
    legend()
    title("Forecast Error (ZEROTH predictor)")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.1)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()

    subplot(122)
    # plot(t2[1:T_steps], MSE_zeroth_mcdts2_n, "r")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts2_n, color="r", marker = "o", label="MCDTS L 2")
    # plot(t2[1:T_steps], MSE_zeroth_mcdts_n, "r")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts_n, color="r", marker = "d", label="MCDTS L")
    # plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn2_n, "m")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts_fnn2_n, color="m", marker = "*", label="MCDTS FNN 2")
    # plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn_n, "m")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts_fnn_n, color="m", marker = "<", label="MCDTS FNN")
    # plot(t2[1:T_steps], MSE_zeroth_pec2_n, "y")
    # scatter(t2[1:T_steps], MSE_zeroth_pec2_n, color="y", marker = "v", label="PECUZAL 2")
    # plot(t2[1:T_steps], MSE_zeroth_pec_n, "y")
    # scatter(t2[1:T_steps], MSE_zeroth_pec_n, color="y", marker = "1", label="PECUZAL")
    # plot(t2[1:T_steps], MSE_zeroth_cao_n, "b")
    # scatter(t2[1:T_steps], MSE_zeroth_cao_n, color="b", marker = "3", label="CAO")
    plot(t2[1:T_steps], MSE_zeroth_kennel_n, "b")
    scatter(t2[1:T_steps], MSE_zeroth_kennel_n, color="b", marker = "s", label="Kennel")
    # plot(t2[1:T_steps], MSE_zeroth_hegger_n, "b")
    # scatter(t2[1:T_steps], MSE_zeroth_hegger_n, color="b", marker = "p", label="Hegger")
    # plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_n, "g")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_n, color="g", marker = "+", label="MCDTS PRED 2")
    # plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED_n, "g")
    # scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED_n, color="g", marker = "p", label="MCDTS PRED")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_5_n, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED2_5_n, color="g", marker = ".", label="MCDTS PRED 2 5 Tw")
    plot(t2[1:T_steps], MSE_zeroth_mcdts_PRED_5_n, "g")
    scatter(t2[1:T_steps], MSE_zeroth_mcdts_PRED_5_n, color="g", marker = "D", label="MCDTS PRED 5 Tw")
    legend()
    title("Forecast Error of noisy time series (ZEROTH predictor)")
    yscale("log")
    xlim(-0, T_steps/lyap_time)
    ylim(0.001, 1.1)
    ylabel("MSE")
    xlabel("Lyapunov time units")
    grid()
end


##

## Plot predictions
prints = begin
    # y-lims
    ylim1 = -3
    ylim2 = 3

    figure(figsize=(20,10))
    subplot(7,1,1)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, Cao_zeroth[:,1], ".-", label="Cao")
    title("x-component (zeroth - iterated one-step) ")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,2)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, Kennel_zeroth[:,1], ".-", label="Kennel")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,3)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, Hegger_zeroth[:,1], ".-", label="Hegger")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,4)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, Pec_zeroth[:,1], ".-", label="PECUZAL")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,5)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, Pec_zeroth2[:,2], ".-", label="PECUZAL 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,6)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_zeroth[:,1], ".-", label="MCDTS")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,7)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_zeroth2[:,2], ".-", label="MCDTS 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()
    xlabel("Lyapunov times")
    subplots_adjust(hspace=.5)

    figure(figsize=(20,10))
    subplot(6,1,1)
    title("x-component (zeroth - iterated one-step) ")
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_fnn_zeroth[:,1], ".-", label="MCDTS FNN")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(6,1,2)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_fnn_zeroth2[:,1], ".-", label="MCDTS FNN 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(6,1,3)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_PRED_zeroth[:,1], ".-", label="MCDTS PRED")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(6,1,4)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_PRED_5_zeroth[:,1], ".-", label="MCDTS PRED 5 Tw")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(6,1,5)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_PRED_zeroth2[:,1], ".-", label="MCDTS PRED 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(6,1,6)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_PRED_5_zeroth2[:,1], ".-", label="MCDTS PRED 2 5 Tw")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    xlabel("Lyapunov times")
    subplots_adjust(hspace=.5)



    figure(figsize=(20,10))
    subplot(7,1,1)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, Cao_zeroth_n[:,1], ".-", label="Cao")
    title("NOISY x-component (zeroth - iterated one-step) ")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,2)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, Kennel_zeroth_n[:,1], ".-", label="Kennel")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,3)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, Hegger_zeroth_n[:,1], ".-", label="Hegger")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,4)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, Pec_zeroth_n[:,1], ".-", label="PECUZAL")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,5)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, Pec_zeroth2_n[:,1], ".-", label="PECUZAL 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,6)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, mcdts_zeroth_n[:,1], ".-", label="MCDTS")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(7,1,7)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, mcdts_zeroth2_n[:,1], ".-", label="MCDTS 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    xlabel("Lyapunov times")
    subplots_adjust(hspace=.5)


    figure(figsize=(20,10))
    subplot(6,1,1)
    title("NOISY x-component (zeroth - iterated one-step) ")
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, mcdts_fnn_zeroth_n[:,1], ".-", label="MCDTS FNN")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(6,1,2)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, mcdts_fnn_zeroth2_n[:,1], ".-", label="MCDTS FNN 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(6,1,3)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, mcdts_PRED_zeroth_n[:,1], ".-", label="MCDTS PRED")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(6,1,4)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, mcdts_PRED_5_zeroth_n[:,1], ".-", label="MCDTS PRED 5 Tw")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(6,1,5)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_PRED_zeroth2_n[:,1], ".-", label="MCDTS PRED 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(6,1,6)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_PRED_5_zeroth2_n[:,1], ".-", label="MCDTS PRED 2 5 Tw")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    xlabel("Lyapunov times")
    subplots_adjust(hspace=.5)

end

## Save variables in order to plot nicely in Matlab

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
