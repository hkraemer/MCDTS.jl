using MCDTS
using DelayEmbeddings
using DynamicalSystemsBase
using ChaosTools
using DelimitedFiles
using Random

# In order to extend the prediction horizon up to 1700 steps in total (former 800),
# we need to extend the trajectory by 900 sampels. Setting random seed ensured
# reproducibility already in the first place. So the new trajectory shares exactly
# the same values with the "old" trajectory, where the reconstructions were based on.

## We predict the Lorenz63-system based on different state space reconstruction methods
Random.seed!(1234)
lo = Systems.lorenz()
# tr = trajectory(lo, 500; dt = 0.01, Ttr = 100) # results 2
tr = trajectory(lo, 1008; dt = 0.01, Ttr = 100) # results 3

end_old = 100001
# noise level
#σ = .1  # results 2
σ = .05 # results 3
# normalize time series
tr = regularize(tr)

Random.seed!(1234)

T_steps = 900 # 8*lyap_time
x1 = tr[1:end_old-T_steps,1]
x2 = tr[end_old-T_steps+1:end,1]
y1 = tr[1:end_old-T_steps,2]
y2 = tr[end_old-T_steps+1:end,2]

x = tr[:,1]
y = tr[:,2]
x_n = tr[:,1] .+ σ*randn(length(tr))
y_n = tr[:,2] .+ σ*randn(length(tr))

x1 = x[1:end_old-T_steps]
x2 = x[end_old-T_steps+1:end]
y1 = y[1:end_old-T_steps]
y2 = y[end_old-T_steps+1:end]
x1_n = x_n[1:end_old-T_steps]
x2_n = x_n[end_old-T_steps+1:end]
y1_n = y_n[1:end_old-T_steps]
y2_n = y_n[end_old-T_steps+1:end]

writedlm("./application/artificial data/Lorenz Prediction/Results 3/x1_long.csv", x1)
writedlm("./application/artificial data/Lorenz Prediction/Results 3/x1_n_long.csv", x1_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 3/x2_long.csv", x2)
writedlm("./application/artificial data/Lorenz Prediction/Results 3/x2_n_long.csv", x2_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 3/y1_long.csv", y1)
writedlm("./application/artificial data/Lorenz Prediction/Results 3/y1_n_long.csv", y1_n)
writedlm("./application/artificial data/Lorenz Prediction/Results 3/y2_long.csv", y2)
writedlm("./application/artificial data/Lorenz Prediction/Results 3/y2_n_long.csv", y2_n)
