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
#Y_mcdts2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results 3/Y_mcdts2_n.csv"))
Y_mcdts2_n = Y_mcdts2
#τ_mcdts2_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/tau_mcdts2_n.csv")))
τ_mcdts2_n = τ_mcdts2
#ts_mcdts2_n = Int.(vec(readdlm("./application/artificial data/Lorenz Prediction/Results 3/ts_mcdts2_n.csv")))
ts_mcdts2_n = ts_mcdts2


# Theiler window:
w1 = 17
w1_n = 20

# Lyapunov time
lyap_time = 110
T_steps = 1700

## make predictions

# params = 1
# # Choose right Neighbourhoodsize
# K_cao = Int(ceil(factorial(params+size(Y_cao,2))/(factorial(params)+factorial(size(Y_cao,2)))))
# K_cao_n = Int(ceil(factorial(params+size(Y_cao_n,2))/(factorial(params)+factorial(size(Y_cao_n,2)))))
# K_kennel = Int(ceil(factorial(params+size(Y_kennel,2))/(factorial(params)+factorial(size(Y_kennel,2)))))
# K_kennel_n = Int(ceil(factorial(params+size(Y_kennel_n,2))/(factorial(params)+factorial(size(Y_kennel_n,2)))))
# K_hegger = Int(ceil(factorial(params+size(Y_hegger,2))/(factorial(params)+factorial(size(Y_hegger,2)))))
# K_hegger_n = Int(ceil(factorial(params+size(Y_hegger_n,2))/(factorial(params)+factorial(size(Y_hegger_n,2)))))
# K_pec = Int(ceil(factorial(params+size(Y_pec,2))/(factorial(params)+factorial(size(Y_pec,2)))))
# K_pec2 = Int(ceil(factorial(params+size(Y_pec2,2))/(factorial(params)+factorial(size(Y_pec2,2)))))
# K_pec_n = Int(ceil(factorial(params+size(Y_pec_n,2))/(factorial(params)+factorial(size(Y_pec_n,2)))))
# K_pec2_n = Int(ceil(factorial(params+size(Y_pec2_n,2))/(factorial(params)+factorial(size(Y_pec2_n,2)))))
# K_mcdts = Int(ceil(factorial(params+size(Y_mcdts,2))/(factorial(params)+factorial(size(Y_mcdts,2)))))
# K_mcdts2 = Int(ceil(factorial(params+size(Y_mcdts2,2))/(factorial(params)+factorial(size(Y_mcdts2,2)))))
# K_mcdts_n = Int(ceil(factorial(params+size(Y_mcdts_n,2))/(factorial(params)+factorial(size(Y_mcdts_n,2)))))
# K_mcdts2_n = Int(ceil(factorial(params+size(Y_mcdts2_n,2))/(factorial(params)+factorial(size(Y_mcdts2_n,2)))))
# K_mcdts_fnn = Int(ceil(factorial(params+size(Y_mcdts_fnn,2))/(factorial(params)+factorial(size(Y_mcdts_fnn,2)))))
# K_mcdts_fnn2 = Int(ceil(factorial(params+size(Y_mcdts_fnn2,2))/(factorial(params)+factorial(size(Y_mcdts_fnn2,2)))))
# K_mcdts_fnn_n = Int(ceil(factorial(params+size(Y_mcdts_fnn_n,2))/(factorial(params)+factorial(size(Y_mcdts_fnn_n,2)))))
# K_mcdts_fnn2_n = Int(ceil(factorial(params+size(Y_mcdts_fnn2_n,2))/(factorial(params)+factorial(size(Y_mcdts_fnn2_n,2)))))


#T_steps = 100 # prediction horizon

# factor = 6 # factor for the neúmber of nearest neighbours, to get a better statistic

KK = 10 # number of nearest neighbours for zeroth predictor

# iterated one step
# Zeroth
println("Cao:")
println("*****")
Cao_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_cao, KK, T_steps; theiler = w1, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/Cao_zeroth.csv",Cao_zeroth)
Cao_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_cao_n, KK, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/Cao_zeroth_n.csv",Cao_zeroth_n)
println("Kennel")
println("*****")
Kennel_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_kennel, KK, T_steps; theiler = w1, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/Kennel_zeroth.csv",Kennel_zeroth)
Kennel_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_kennel_n, KK, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/Kennel_zeroth_n.csv",Kennel_zeroth_n)
println("Hegger")
println("*****")
Hegger_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_hegger, KK, T_steps; theiler = w1, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/Hegger_zeroth.csv",Hegger_zeroth)
Hegger_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_hegger_n, KK, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/Hegger_zeroth_n.csv",Hegger_zeroth_n)
println("Pec")
println("*****")
Pec_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_pec, KK, T_steps; theiler = w1, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/Pec_zeroth.csv",Pec_zeroth)
Pec_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_pec2, KK, T_steps; theiler = w1, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/Pec_zeroth2.csv",Pec_zeroth2)
Pec_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_pec_n, KK, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/Pec_zeroth_n.csv",Pec_zeroth_n)
Pec_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_pec2_n, KK, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/Pec_zeroth2_n.csv", Pec_zeroth2_n)
println("mcdts:")
println("*****")
mcdts_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_mcdts, KK, T_steps; theiler = w1, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/mcdts_zeroth.csv",mcdts_zeroth)
mcdts_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_mcdts2, KK, T_steps; theiler = w1, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/mcdts_zeroth2.csv",mcdts_zeroth2)
mcdts_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_n, KK, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/mcdts_zeroth_n.csv",mcdts_zeroth_n)
mcdts_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts2_n, KK, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/mcdts_zeroth.csv",mcdts_zeroth2_n)
println("mcdts FNN:")
println("*****")
mcdts_fnn_zeroth = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn, KK, T_steps; theiler = w1, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/mcdts_fnn_zeroth.csv",mcdts_fnn_zeroth)
mcdts_fnn_zeroth2 = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn2, KK, T_steps; theiler = w1, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/mcdts_fnn_zeroth2.csv",mcdts_fnn_zeroth2)
mcdts_fnn_zeroth_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn_n,  KK, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/mcdts_fnn_zeroth_n.csv",mcdts_fnn_zeroth_n)
mcdts_fnn_zeroth2_n = MCDTS.iterated_local_zeroth_prediction(Y_mcdts_fnn2_n, KK, T_steps; theiler = w1_n, verbose=true)
writedlm("./application/artificial data/Lorenz Prediction/Results 5/mcdts_fnn_zeroth.csv",mcdts_fnn_zeroth2_n)

# println("*****")
# println("LOCAL LINEAR")
# println("*****")
#
# # Local linear
# println("Cao:")
# println("*****")
# Cao_linear = MCDTS.iterated_local_linear_prediction(Y_cao, factor*K_cao, T_steps; theiler = w1, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/Cao_linear.csv",Cao_linear)
# Cao_linear_n = MCDTS.iterated_local_linear_prediction(Y_cao_n, factor*K_cao_n, T_steps; theiler = w1_n, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/Cao_linear_n.csv",Cao_linear_n)
# println("Kennel")
# println("*****")
# Kennel_linear = MCDTS.iterated_local_linear_prediction(Y_kennel, factor*K_kennel, T_steps; theiler = w1, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/Kennel_linear.csv",Kennel_linear)
# Kennel_linear_n = MCDTS.iterated_local_linear_prediction(Y_kennel_n, factor*K_kennel_n, T_steps; theiler = w1_n, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/Kennel_linear_n.csv",Kennel_linear_n)
# println("Hegger")
# println("*****")
# # Hegger_linear = MCDTS.iterated_local_linear_prediction(Y_hegger, factor*K_hegger, T_steps; theiler = w1, verbose=true)
# Hegger_linear = MCDTS.iterated_local_linear_prediction(Y_hegger, 75, T_steps; theiler = w1, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/Hegger_linear.csv",Hegger_linear)
# # Hegger_linear_n = MCDTS.iterated_local_linear_prediction(Y_hegger_n, factor*K_hegger_n, T_steps; theiler = w1_n, verbose=true)
# Hegger_linear_n = MCDTS.iterated_local_linear_prediction(Y_hegger_n, 75, T_steps; theiler = w1_n, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/Hegger_linear_n.csv",Hegger_linear_n)
# println("Pec")
# println("*****")
# Pec_linear = MCDTS.iterated_local_linear_prediction(Y_pec, factor*K_pec, T_steps; theiler = w1, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/Pec_linear.csv",Pec_linear)
# Pec_linear2 = MCDTS.iterated_local_linear_prediction(Y_pec2, factor*K_pec2, T_steps; theiler = w1, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/Pec_linear2.csv",Pec_linear2)
# Pec_linear_n = MCDTS.iterated_local_linear_prediction(Y_pec_n, factor*K_pec_n, T_steps; theiler = w1_n, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/Pec_linear_n.csv",Pec_linear_n)
# Pec_linear2_n = MCDTS.iterated_local_linear_prediction(Y_pec2_n, factor*K_pec2_n, T_steps; theiler = w1_n, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/Pec_linear2_n.csv",Pec_linear2_n)
# println("mcdts:")
# println("*****")
# mcdts_linear = MCDTS.iterated_local_linear_prediction(Y_mcdts, factor*K_mcdts, T_steps; theiler = w1, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/mcdts_linear.csv",mcdts_linear)
# mcdts_linear2 = MCDTS.iterated_local_linear_prediction(Y_mcdts2, factor*K_mcdts2, 2; theiler = w1, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/mcdts_linear2.csv",mcdts_linear2)
# mcdts_linear2 = mcdts_linear
# mcdts_linear_n = MCDTS.iterated_local_linear_prediction(Y_mcdts_n, factor*K_mcdts_n, T_steps; theiler = w1_n, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/mcdts_linear_n.csv",mcdts_linear_n)
# mcdts_linear2_n = MCDTS.iterated_local_linear_prediction(Y_mcdts2_n, factor*K_mcdts2_n, T_steps; theiler = w1_n, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/mcdts_linear2_n.csv",mcdts_linear2_n)
# println("mcdts FNN:")
# println("*****")
# mcdts_fnn_linear = MCDTS.iterated_local_linear_prediction(Y_mcdts_fnn, factor*K_mcdts_fnn, T_steps; theiler = w1, verbose=true)
# #mcdts_fnn_linear = MCDTS.iterated_local_linear_prediction_embed(Y_mcdts_fnn, τ_mcdts_fnn, 50, T_steps; theiler = w1, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/mcdts_fnn_linear.csv",mcdts_fnn_linear)
# mcdts_fnn_linear2 = MCDTS.iterated_local_linear_prediction(Y_mcdts_fnn2, factor*K_mcdts_fnn2, T_steps; theiler = w1, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/mcdts_fnn_linear2.csv",mcdts_fnn_linear2)
# mcdts_fnn_linear_n = MCDTS.iterated_local_linear_prediction(Y_mcdts_fnn_n, factor*K_mcdts_fnn_n, T_steps; theiler = w1_n, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/mcdts_fnn_linear_n.csv",mcdts_fnn_linear_n)
# mcdts_fnn_linear2_n = MCDTS.iterated_local_linear_prediction(Y_mcdts_fnn2_n, factor*K_mcdts_fnn2_n, T_steps; theiler = w1_n, verbose=true)
# writedlm("./application/artificial data/Lorenz Prediction/Results 3/mcdts_fnn_linear2_n.csv",mcdts_fnn_linear2_n)



# Results 3: zeroth and linear T_steps = 900, KK = 1
# Results 4: zeroth T_steps = 1300, KK = 1
# Results 5: zeroth T_steps = 1300, KK = 10

# load data
Number = 4
load = begin
    Cao_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/Cao_zeroth.csv"))
    Cao_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/Cao_zeroth_n.csv"))
    Kennel_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/Kennel_zeroth.csv"))
    Kennel_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/Kennel_zeroth_n.csv"))
    Hegger_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/Hegger_zeroth.csv"))
    Hegger_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/Hegger_zeroth_n.csv"))
    Pec_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/Pec_zeroth.csv"))
    Pec_zeroth2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/Pec_zeroth2.csv"))
    Pec_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/Pec_zeroth_n.csv"))
    Pec_zeroth2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/Pec_zeroth2_n.csv"))
    mcdts_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/mcdts_zeroth.csv"))
    mcdts_zeroth2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/mcdts_zeroth2.csv"))
    mcdts_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/mcdts_zeroth_n.csv"))
    mcdts_zeroth2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/mcdts_zeroth.csv"))
    mcdts_fnn_zeroth = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/mcdts_fnn_zeroth.csv"))
    mcdts_fnn_zeroth2 = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/mcdts_fnn_zeroth2.csv"))
    mcdts_fnn_zeroth_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/mcdts_fnn_zeroth_n.csv"))
    mcdts_fnn_zeroth2_n = Dataset(readdlm("./application/artificial data/Lorenz Prediction/Results $Number/mcdts_fnn_zeroth.csv"))
end



# time axis
t2 = (0:T_steps-1) ./lyap_time
t1 = (-length(x1):-1) ./lyap_time
NN = 1000
tt = vcat(t1[end-NN:end], t2)
M = length(tt)
#true_data = vcat(x1[end-NN:end], x2)
#true_data_n = vcat(x1_n[end-NN:end], x2_n)
true_data = vcat(x1_[end-NN:end], x2_)
true_data_n = vcat(x1_n_[end-NN:end], x2_n_)

## MSEs
# compute MSE of predictions
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

# MSE_linear_cao = zeros(T_steps)
# MSE_linear_cao_n = zeros(T_steps)
# MSE_linear_kennel = zeros(T_steps)
# MSE_linear_kennel_n = zeros(T_steps)
# MSE_linear_hegger = zeros(T_steps)
# MSE_linear_hegger_n = zeros(T_steps)
# MSE_linear_pec = zeros(T_steps)
# MSE_linear_pec_n = zeros(T_steps)
# MSE_linear_pec2 = zeros(T_steps)
# MSE_linear_pec2_n = zeros(T_steps)
# MSE_linear_mcdts = zeros(T_steps)
# MSE_linear_mcdts_n = zeros(T_steps)
# MSE_linear_mcdts2 = zeros(T_steps)
# MSE_linear_mcdts2_n = zeros(T_steps)
# MSE_linear_mcdts_fnn = zeros(T_steps)
# MSE_linear_mcdts_fnn_n = zeros(T_steps)
# MSE_linear_mcdts_fnn2 = zeros(T_steps)
# MSE_linear_mcdts_fnn2_n = zeros(T_steps)

σ₂ = sqrt(var(x2_[1:T_steps]))   # rms deviation for normalization
MASE_norm = MCDTS.rw_norm(x1_, T_steps)
MASE_norm_n = MCDTS.rw_norm(x1_n_, T_steps)
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
    MSE_zeroth_cao_n[i] = MCDTS.compute_mse(Cao_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂
    MSE_zeroth_kennel_n[i] = MCDTS.compute_mse(Kennel_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂
    MSE_zeroth_hegger_n[i] = MCDTS.compute_mse(Hegger_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂
    MSE_zeroth_pec_n[i] = MCDTS.compute_mse(Pec_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂
    MSE_zeroth_pec2_n[i] = MCDTS.compute_mse(Pec_zeroth2_n[1:i,1], x2_n_[1:i]) / σ₂
    MSE_zeroth_mcdts_n[i] = MCDTS.compute_mse(mcdts_zeroth_n[1:i,2], x2_n_[1:i]) / σ₂
    MSE_zeroth_mcdts2_n[i] = MCDTS.compute_mse(mcdts_zeroth2_n[1:i,2], x2_n_[1:i]) / σ₂
    MSE_zeroth_mcdts_fnn_n[i] = MCDTS.compute_mse(mcdts_fnn_zeroth_n[1:i,1], x2_n_[1:i]) / σ₂
    MSE_zeroth_mcdts_fnn2_n[i] = MCDTS.compute_mse(mcdts_fnn_zeroth2_n[1:i,1], x2_n_[1:i]) / σ₂
    #
    # MSE_linear_cao[i] = MCDTS.compute_mse(Cao_linear[1:i,1], x2_[1:i]) / σ₂
    # MSE_linear_kennel[i] = MCDTS.compute_mse(Kennel_linear[1:i,1], x2_[1:i]) / σ₂
    # MSE_linear_hegger[i] = MCDTS.compute_mse(Hegger_linear[1:i,1], x2_[1:i]) / σ₂
    # MSE_linear_pec[i] = MCDTS.compute_mse(Pec_linear[1:i,1], x2_[1:i]) / σ₂
    # MSE_linear_pec2[i] = MCDTS.compute_mse(Pec_linear2[1:i,2], x2_[1:i]) / σ₂
    # MSE_linear_mcdts[i] = MCDTS.compute_mse(mcdts_linear[1:i,1], x2_[1:i]) / σ₂
    # MSE_linear_mcdts2[i] = MCDTS.compute_mse(mcdts_linear2[1:i,1], x2_[1:i]) / σ₂
    # MSE_linear_mcdts_fnn[i] = MCDTS.compute_mse(mcdts_fnn_linear[1:i,1], x2_[1:i]) / σ₂
    # MSE_linear_mcdts_fnn2[i] = MCDTS.compute_mse(mcdts_fnn_linear2[1:i,1], x2_[1:i]) / σ₂
    # MSE_linear_cao_n[i] = MCDTS.compute_mse(Cao_linear[1:i,1], x2_n_[1:i]) / σ₂
    # MSE_linear_kennel_n[i] = MCDTS.compute_mse(Kennel_linear_n[1:i,1], x2_n_[1:i]) / σ₂
    # MSE_linear_hegger_n[i] = MCDTS.compute_mse(Hegger_linear_n[1:i,1], x2_n_[1:i]) / σ₂
    # MSE_linear_pec_n[i] = MCDTS.compute_mse(Pec_linear_n[1:i,1], x2_n_[1:i]) / σ₂
    # MSE_linear_pec2_n[i] = MCDTS.compute_mse(Pec_linear2_n[1:i,1], x2_n_[1:i]) / σ₂
    # MSE_linear_mcdts_n[i] = MCDTS.compute_mse(mcdts_linear_n[1:i,1], x2_n_[1:i]) / σ₂
    # MSE_linear_mcdts2_n[i] = MCDTS.compute_mse(mcdts_linear2_n[1:i,4], x2_n_[1:i]) / σ₂
    # MSE_linear_mcdts_fnn_n[i] = MCDTS.compute_mse(mcdts_fnn_linear_n[1:i,1], x2_n_[1:i]) / σ₂
    # MSE_linear_mcdts_fnn2_n[i] = MCDTS.compute_mse(mcdts_fnn_linear2_n[1:i,1], x2_n_[1:i]) / σ₂

    # MASE error
    # MSE_zeroth_cao[i] = MCDTS.compute_abs_err(Cao_zeroth[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_zeroth_kennel[i] = MCDTS.compute_abs_err(Kennel_zeroth[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_zeroth_hegger[i] = MCDTS.compute_abs_err(Hegger_zeroth[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_zeroth_pec[i] = MCDTS.compute_abs_err(Pec_zeroth[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_zeroth_pec2[i] = MCDTS.compute_abs_err(Pec_zeroth2[1:i,2], x2_[1:i]) / MASE_norm
    # MSE_zeroth_mcdts[i] = MCDTS.compute_abs_err(mcdts_zeroth[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_zeroth_mcdts2[i] = MCDTS.compute_abs_err(mcdts_zeroth2[1:i,2], x2_[1:i]) / MASE_norm
    # MSE_zeroth_mcdts_fnn[i] = MCDTS.compute_abs_err(mcdts_fnn_zeroth[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_zeroth_mcdts_fnn2[i] = MCDTS.compute_abs_err(mcdts_fnn_zeroth2[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_zeroth_cao_n[i] = MCDTS.compute_abs_err(Cao_zeroth[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_zeroth_kennel_n[i] = MCDTS.compute_abs_err(Kennel_zeroth_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_zeroth_hegger_n[i] = MCDTS.compute_abs_err(Hegger_zeroth_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_zeroth_pec_n[i] = MCDTS.compute_abs_err(Pec_zeroth_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_zeroth_pec2_n[i] = MCDTS.compute_abs_err(Pec_zeroth2_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_zeroth_mcdts_n[i] = MCDTS.compute_abs_err(mcdts_zeroth_n[1:i,2], x2_n_[1:i]) / MASE_norm_n
    # MSE_zeroth_mcdts2_n[i] = MCDTS.compute_abs_err(mcdts_zeroth2_n[1:i,2], x2_n_[1:i]) / MASE_norm_n
    # MSE_zeroth_mcdts_fnn_n[i] = MCDTS.compute_abs_err(mcdts_fnn_zeroth_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_zeroth_mcdts_fnn2_n[i] = MCDTS.compute_abs_err(mcdts_fnn_zeroth2_n[1:i,1], x2_n_[1:i]) / MASE_norm_n

    # MSE_linear_cao[i] = MCDTS.compute_abs_err(Cao_linear[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_linear_kennel[i] = MCDTS.compute_abs_err(Kennel_linear[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_linear_hegger[i] = MCDTS.compute_abs_err(Hegger_linear[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_linear_pec[i] = MCDTS.compute_abs_err(Pec_linear[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_linear_pec2[i] = MCDTS.compute_abs_err(Pec_linear2[1:i,2], x2_[1:i]) / MASE_norm
    # MSE_linear_mcdts[i] = MCDTS.compute_abs_err(mcdts_linear[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_linear_mcdts2[i] = MCDTS.compute_abs_err(mcdts_linear2[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_linear_mcdts_fnn[i] = MCDTS.compute_abs_err(mcdts_fnn_linear[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_linear_mcdts_fnn2[i] = MCDTS.compute_abs_err(mcdts_fnn_linear2[1:i,1], x2_[1:i]) / MASE_norm
    # MSE_linear_cao_n[i] = MCDTS.compute_abs_err(Cao_linear[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_linear_kennel_n[i] = MCDTS.compute_abs_err(Kennel_linear_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_linear_hegger_n[i] = MCDTS.compute_abs_err(Hegger_linear_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_linear_pec_n[i] = MCDTS.compute_abs_err(Pec_linear_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_linear_pec2_n[i] = MCDTS.compute_abs_err(Pec_linear2_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_linear_mcdts_n[i] = MCDTS.compute_abs_err(mcdts_linear_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_linear_mcdts2_n[i] = MCDTS.compute_abs_err(mcdts_linear2_n[1:i,4], x2_n_[1:i]) / MASE_norm_n
    # MSE_linear_mcdts_fnn_n[i] = MCDTS.compute_abs_err(mcdts_fnn_linear_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
    # MSE_linear_mcdts_fnn2_n[i] = MCDTS.compute_abs_err(mcdts_fnn_linear2_n[1:i,1], x2_n_[1:i]) / MASE_norm_n
end

# Plot MSEs
figure(figsize=(20,10))
subplot(121)
plot(t2[1:T_steps], MSE_zeroth_mcdts2, ".-", label="MCDTS L 2")
plot(t2[1:T_steps], MSE_zeroth_mcdts, "--", label="MCDTS L")
plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn2, "-", label="MCDTS FNN 2")
plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn, "-.", label="MCDTS FNN")
plot(t2[1:T_steps], MSE_zeroth_pec2, "r.-", label="PECUZAL 2")
plot(t2[1:T_steps], MSE_zeroth_pec, "r.-.", label="PECUZAL")
plot(t2[1:T_steps], MSE_zeroth_cao, "k--", label="CAO")
plot(t2[1:T_steps], MSE_zeroth_kennel, "k-", label="Kennel")
plot(t2[1:T_steps], MSE_zeroth_hegger, "k.-.", label="Hegger")
legend()
title("Forecast Error (ZEROTH predictor)")
yscale("log")
xlim(-0, T_steps/lyap_time)
ylabel("MSE")
xlabel("Lyapunov time units")
grid()

subplot(122)
plot(t2[1:T_steps], MSE_zeroth_mcdts2_n, ".-", label="MCDTS L 2")
plot(t2[1:T_steps], MSE_zeroth_mcdts_n, "--", label="MCDTS L")
plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn2_n, "-", label="MCDTS FNN 2")
plot(t2[1:T_steps], MSE_zeroth_mcdts_fnn_n, "-.", label="MCDTS FNN")
plot(t2[1:T_steps], MSE_zeroth_pec2_n, "r.-", label="PECUZAL 2")
plot(t2[1:T_steps], MSE_zeroth_pec_n, "r.-.", label="PECUZAL")
plot(t2[1:T_steps], MSE_zeroth_cao_n, "k--", label="CAO")
plot(t2[1:T_steps], MSE_zeroth_kennel_n, "k-", label="Kennel")
plot(t2[1:T_steps], MSE_zeroth_hegger_n, "k.-.", label="Hegger")
legend()
title("Forecast Error of noisy time series (ZEROTH predictor)")
yscale("log")
xlim(-0, T_steps/lyap_time)
ylabel("MSE")
xlabel("Lyapunov time units")
grid()


# figure(figsize=(20,10))
# subplot(121)
# plot(t2[1:T_steps], MSE_linear_mcdts2, ".-", label="MCDTS L 2")
# plot(t2[1:T_steps], MSE_linear_mcdts, "--", label="MCDTS L")
# plot(t2[1:T_steps], MSE_linear_mcdts_fnn2, "-", label="MCDTS FNN 2")
# plot(t2[1:T_steps], MSE_linear_mcdts_fnn, "-.", label="MCDTS FNN")
# plot(t2[1:T_steps], MSE_linear_pec2, "r.-", label="PECUZAL 2")
# plot(t2[1:T_steps], MSE_linear_pec, "r.-.", label="PECUZAL")
# plot(t2[1:T_steps], MSE_linear_cao, "k--", label="CAO")
# plot(t2[1:T_steps], MSE_linear_kennel, "k-", label="Kennel")
# plot(t2[1:T_steps], MSE_linear_hegger, "k.-.", label="Hegger")
# legend()
# title("Forecast Error (LINEAR predictor)")
# yscale("log")
# xlim(-0, T_steps/lyap_time)
# ylabel("MSE")
# xlabel("Lyapunov time units")
# grid()
#
# subplot(122)
# plot(t2[1:T_steps], MSE_linear_mcdts2_n, ".-", label="MCDTS L 2")
# plot(t2[1:T_steps], MSE_linear_mcdts_n, "--", label="MCDTS L")
# plot(t2[1:T_steps], MSE_linear_mcdts_fnn2_n, "-", label="MCDTS FNN 2")
# plot(t2[1:T_steps], MSE_linear_mcdts_fnn_n, "-.", label="MCDTS FNN")
# plot(t2[1:T_steps], MSE_linear_pec2_n, "r.-", label="PECUZAL 2")
# plot(t2[1:T_steps], MSE_linear_pec_n, "r.-.", label="PECUZAL")
# plot(t2[1:T_steps], MSE_linear_cao_n, "k--", label="CAO")
# plot(t2[1:T_steps], MSE_linear_kennel_n, "k-", label="Kennel")
# plot(t2[1:T_steps], MSE_linear_hegger_n, "k.-.", label="Hegger")
# legend()
# title("Forecast Error of noisy time series (LINEAR predictor)")
# yscale("log")
# xlim(-0, T_steps/lyap_time)
# ylabel("MSE")
# xlabel("Lyapunov time units")
# grid()


## Plot predictions
prints = begin
    # y-lims
    ylim1 = -3
    ylim2 = 3

    figure(figsize=(20,10))
    subplot(9,1,1)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, Cao_zeroth[:,1], ".-", label="Cao")
    title("x-component (zeroth - iterated one-step) ")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,2)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, Kennel_zeroth[:,1], ".-", label="Kennel")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,3)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, Hegger_zeroth[:,1], ".-", label="Hegger")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,4)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, Pec_zeroth[:,1], ".-", label="PECUZAL")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,5)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, Pec_zeroth2[:,2], ".-", label="PECUZAL 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,6)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_zeroth[:,1], ".-", label="MCDTS")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,7)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_zeroth2[:,2], ".-", label="MCDTS 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,8)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_fnn_zeroth[:,1], ".-", label="MCDTS FNN")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,9)
    plot(tt, true_data, ".-", label="true data")
    plot(t2, mcdts_fnn_zeroth2[:,1], ".-", label="MCDTS FNN 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()
    xlabel("Lyapunov times")
    subplots_adjust(hspace=.8)



    figure(figsize=(20,10))
    subplot(9,1,1)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, Cao_zeroth_n[:,1], ".-", label="Cao")
    title("NOISY x-component (zeroth - iterated one-step) ")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,2)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, Kennel_zeroth_n[:,1], ".-", label="Kennel")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,3)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, Hegger_zeroth_n[:,1], ".-", label="Hegger")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,4)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, Pec_zeroth_n[:,1], ".-", label="PECUZAL")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,5)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, Pec_zeroth2_n[:,1], ".-", label="PECUZAL 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,6)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, mcdts_zeroth_n[:,1], ".-", label="MCDTS")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,7)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, mcdts_zeroth2_n[:,2], ".-", label="MCDTS 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,8)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, mcdts_fnn_zeroth_n[:,1], ".-", label="MCDTS FNN")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()

    subplot(9,1,9)
    plot(tt, true_data_n, ".-", label="true data")
    plot(t2, mcdts_fnn_zeroth2_n[:,1], ".-", label="MCDTS FNN 2")
    xlim(-.5, T_steps/lyap_time)
    ylim(ylim1,ylim2)
    legend()
    grid()
    xlabel("Lyapunov times")
    subplots_adjust(hspace=.8)


    ##
    # figure(figsize=(20,10))
    # subplot(5,1,1)
    # plot(tt, true_data, ".-", label="true data")
    # plot(t2, Cao_linear[:,1], ".-", label="Cao")
    # title("x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,2)
    # plot(tt, true_data, ".-", label="true data")
    # plot(t2, Kennel_linear[:,1], ".-", label="Kennel")
    # title("x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,3)
    # plot(tt, true_data, ".-", label="true data")
    # plot(t2, Hegger_linear[:,1], ".-", label="Hegger")
    # title("x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,4)
    # plot(tt, true_data, ".-", label="true data")
    # plot(t2, Pec_linear[:,1], ".-", label="PECUZAL")
    # title("x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,5)
    # plot(tt, true_data, ".-", label="true data")
    # plot(t2, mcdts_linear[:,1], ".-", label="MCDTS")
    # title("x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    # subplots_adjust(hspace=.8)
    #
    # figure(figsize=(20,10))
    # subplot(5,1,1)
    # plot(tt, true_data, ".-", label="true data")
    # plot(t2, Pec_linear2[:,1], ".-", label="PECUZAL 2")
    # title("x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,2)
    # plot(tt, true_data, ".-", label="true data")
    # plot(t2, mcdts_linear[:,1], ".-", label="MCDTS")
    # title("x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,3)
    # plot(tt, true_data, ".-", label="true data")
    # plot(t2, mcdts_fnn_linear[:,1], ".-", label="MCDTS FNN")
    # title("x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,4)
    # plot(tt, true_data, ".-", label="true data")
    # plot(t2, mcdts_fnn_linear2[:,1], ".-", label="MCDTS FNN 2")
    # title("x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,5)
    # plot(tt, true_data, ".-", label="true data")
    # plot(t2, mcdts_linear2[:,1], ".-", label="MCDTS 2")
    # title("x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    # subplots_adjust(hspace=.8)
    #
    #
    #
    # figure(figsize=(20,10))
    # subplot(5,1,1)
    # plot(tt, true_data_n, ".-", label="true data")
    # plot(t2, Cao_linear_n[:,1], ".-", label="Cao")
    # title("NOISY x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,2)
    # plot(tt, true_data_n, ".-", label="true data")
    # plot(t2, Kennel_linear_n[:,1], ".-", label="Kennel")
    # title("NOISY x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,3)
    # plot(tt, true_data_n, ".-", label="true data")
    # plot(t2, Hegger_linear_n[:,1], ".-", label="Hegger")
    # title("NOISY x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,4)
    # plot(tt, true_data_n, ".-", label="true data")
    # plot(t2, Pec_linear_n[:,1], ".-", label="PECUZAL")
    # title("NOISY x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,5)
    # plot(tt, true_data_n, ".-", label="true data")
    # plot(t2, mcdts_linear_n[:,1], ".-", label="MCDTS")
    # title("NOISY x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    # subplots_adjust(hspace=.8)
    #
    # figure(figsize=(20,10))
    # subplot(5,1,1)
    # plot(tt, true_data_n, ".-", label="true data")
    # plot(t2, Pec_linear2_n[:,1], ".-", label="PECUZAL 2")
    # title("NOISY x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,2)
    # plot(tt, true_data_n, ".-", label="true data")
    # plot(t2, mcdts_linear_n[:,1], ".-", label="MCDTS")
    # title("NOISY x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,3)
    # plot(tt, true_data_n, ".-", label="true data")
    # plot(t2, mcdts_fnn_linear_n[:,1], ".-", label="MCDTS FNN")
    # title("NOISY x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,4)
    # plot(tt, true_data_n, ".-", label="true data")
    # plot(t2, mcdts_fnn_linear2_n[:,1], ".-", label="MCDTS FNN 2")
    # title("NOISY x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    #
    # subplot(5,1,5)
    # plot(tt, true_data_n, ".-", label="true data")
    # plot(t2, mcdts_linear2_n[:,1], ".-", label="MCDTS 2")
    # title("NOISY x-component (linear - iterated one-step) ")
    # xlim(-.5, T_steps/lyap_time)
    # ylim(ylim1,ylim2)
    # legend()
    # grid()
    # subplots_adjust(hspace=.8)

end
