using DelayEmbeddings
using DelimitedFiles
using Random
using Statistics

using PyPlot
pygui(true)


# noise level
σ = .05

# time series to pick
t_idx_1 = 1         # univariate
t_idx_2 = [1,3]     # multivariate

number_of_ics = 100 # number of different initial conditions
dt = 0.01
T_steps = 881
lyap_time = 110

methods = ["Cao", "Kennel", "Hegger", "PECUZAL", "PECUZAL mult.", "MCDTS L",
            "MCDTS L mult.", "MCDTS FNN", "MCDTS FNN mult.", "MCDTS PRED",
            "MCDTS PRED mult.", "MCDTS PRED KL", "MCDTS PRED KL mult."]

# all MSEs
MSEs = ones(13,number_of_ics,T_steps)
MSEs_n = ones(13,number_of_ics,T_steps)
# mean of all MSEs
MEANs = zeros(13,T_steps)
MEANs_n = zeros(13,T_steps)
# std of all MSEs
STDs = zeros(13,T_steps)
STDs_n = zeros(13,T_steps)

# load results
begin
    MSEs[1,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_cao.csv")
    MSEs[2,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_kennel.csv")
    MSEs[3,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_hegger.csv")
    MSEs[4,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_pec.csv")
    MSEs[5,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_pec2.csv")
    MSEs[6,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts_L.csv")
    #MSEs[7,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts2_L.csv")
    MSEs[8,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts_FNN.csv")
    MSEs[9,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts2_FNN.csv")
    MSEs[10,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts_PRED.csv")
    MSEs[11,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts2_PRED.csv")
    MSEs[12,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts_PRED_KL.csv")
    MSEs[13,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts2_PRED_KL.csv")

    MSEs_n[1,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_cao_n.csv")
    MSEs_n[2,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_kennel_n.csv")
    MSEs_n[3,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_hegger_n.csv")
    MSEs_n[4,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_pec_n.csv")
    MSEs_n[5,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_pec2_n.csv")
    #MSEs_n[6,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts_L_n.csv")
    #MSEs_n[7,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts2_L_n.csv")
    MSEs_n[8,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts_FNN_n.csv")
    MSEs_n[9,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts2_FNN_n.csv")
    MSEs_n[10,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts_PRED_n.csv")
    MSEs_n[11,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts2_PRED_n.csv")
    MSEs_n[12,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts_PRED_KL_n.csv")
    MSEs_n[13,:,:] = readdlm("./application/artificial data/Lorenz Prediction/3rd trial/results_Lorenz63_MSEs_mcdts2_PRED_KL_n.csv")

    for i = 1:13
        MEANs[i,:] = mean(MSEs[i,:,:], dims =1)
        MEANs_n[i,:] = mean(MSEs_n[i,:,:], dims =1)
        STDs[i,:] = std(MSEs[i,:,:], dims =1)
        STDs_n[i,:] = std(MSEs_n[i,:,:], dims =1)
    end
end

## Plot results

t = 1:T_steps
t = t ./ lyap_time
figure(figsize=(20,10))
for i = 1:13
    plot(t, MEANs_n[i,:], label = methods[i])
    if i == 1 || i == 2 || i == 3
        scatter(t, MEANs[i,:], marker = "*")
    end
end
yscale("log")
grid()
legend()

num = [12,13]
begin
    sc = 10
    figure(figsize=(20,10))
    subplot(121)
    plot(t, MEANs[1,:], color = "r", label = methods[1])
    scatter(t, MEANs[1,:], s = sc, color = "r", marker = "o")
    plot(t, MEANs[2,:], color = "r", label = methods[2])
    scatter(t, MEANs[2,:], s = sc, color = "r", marker = "<")
    plot(t, MEANs[3,:], color = "r", label = methods[3])
    scatter(t, MEANs[3,:], s = sc, color = "r", marker = "*")
    plot(t, MEANs[num[1],:], color = "b", label = methods[num[1]])
    scatter(t, MEANs[num[1],:], s = sc, color = "b", marker = "s")
    plot(t, MEANs[num[2],:], color = "k", label = methods[num[2]])
    scatter(t, MEANs[num[2],:], s = sc, color = "k", marker = "d")
    yscale("log")
    grid()
    legend()
    title("Mean MSE (100 runs, NO NOISE)")
    subplot(122)
    plot(t, MEANs_n[1,:], color = "r", label = methods[1])
    scatter(t, MEANs_n[1,:], s = sc, color = "r", marker = "o")
    plot(t, MEANs_n[2,:], color = "r", label = methods[2])
    scatter(t, MEANs_n[2,:], s = sc, color = "r", marker = "<")
    plot(t, MEANs_n[3,:], color = "r", label = methods[3])
    scatter(t, MEANs_n[3,:], s = sc, color = "r", marker = "*")
    plot(t, MEANs_n[num[1],:], color = "b", label = methods[num[1]])
    scatter(t, MEANs_n[num[1],:], s = sc, color = "b", marker = "s")
    plot(t, MEANs_n[num[2],:], color = "k", label = methods[num[2]])
    scatter(t, MEANs_n[num[2],:], s = sc, color = "k", marker = "d")
    yscale("log")
    grid()
    legend()
    title("Mean MSE (100 runs, 5% add.NOISE)")

    figure(figsize=(20,10))
    subplot(121)
    plot(t, STDs[1,:], color = "r", label = methods[1])
    scatter(t, STDs[1,:], s = sc, color = "r", marker = "o")
    plot(t, STDs[2,:], color = "r", label = methods[2])
    scatter(t, STDs[2,:], s = sc, color = "r", marker = "<")
    plot(t, STDs[3,:], color = "r", label = methods[3])
    scatter(t, STDs[3,:], s = sc, color = "r", marker = "*")
    plot(t, STDs[num[1],:], color = "b", label = methods[num[1]])
    scatter(t, STDs[num[1],:], s = sc, color = "b", marker = "s")
    plot(t, STDs[num[2],:], color = "k", label = methods[num[2]])
    scatter(t, STDs[num[2],:], s = sc, color = "k", marker = "d")
    yscale("log")
    grid()
    legend()
    title("Std of MSE (100 runs, NO NOISE)")
    subplot(122)
    plot(t, STDs_n[1,:], color = "r", label = methods[1])
    scatter(t, STDs_n[1,:], s = sc, color = "r", marker = "o")
    plot(t, STDs_n[2,:], color = "r", label = methods[2])
    scatter(t, STDs_n[2,:], s = sc, color = "r", marker = "<")
    plot(t, STDs_n[3,:], color = "r", label = methods[3])
    scatter(t, STDs_n[3,:], s = sc, color = "r", marker = "*")
    plot(t, STDs_n[num[1],:], color = "b", label = methods[num[1]])
    scatter(t, STDs_n[num[1],:], s = sc, color = "b", marker = "s")
    plot(t, STDs_n[num[2],:], color = "k", label = methods[num[2]])
    scatter(t, STDs_n[num[2],:], s = sc, color = "k", marker = "d")
    yscale("log")
    grid()
    legend()
    title("Std of MSE (100 runs, 5% add.NOISE)")
end


threshold = 0.11
times = zeros(13)
times_n = zeros(13)
for i = 1:13
    times[i] = t[findall(x -> x .> threshold, MEANs[i,:])[1]]
    times_n[i] = t[findall(x -> x .> threshold, MEANs_n[i,:])[1]]
end
figure(figsize=(20,10))
subplot(121)
bar(1:13, times)
xticks(ticks=1:13, labels=methods, rotation=90)
grid()
ylabel("Lyapunov time")
title("Normalized mean squared Prediction error larger than $threshold (NO NOISE)")

subplot(122)
bar(1:13, times_n)
xticks(ticks=1:13, labels=methods, rotation=90)
grid()
ylabel("Lyapunov time")
title("Normalized mean squared Prediction error larger than $threshold (NOISY)")
