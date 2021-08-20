using MCDTS
using DelayEmbeddings
using DelimitedFiles
using Random
using Statistics

using PyPlot
pygui(true)

data = readdlm("./application/CENOGRID/data/detrended.txt")

t = data[:,1]
O18 = data[:,2]
O13 = data[:,3]

reverse!(t)
reverse!(O18)
reverse!(O13)

# Make Predictions:
T_steps = 110
T_steps2 = 10

# time series binding
x1 = O18[1:end-T_steps]       # training
x2 = O18[end-T_steps+1:end]   # prediction
y1 = O13[1:end-T_steps]       # training
y2 = O13[end-T_steps+1:end]   # prediction
t1 = t[1:end-T_steps]
t2 = t[end-T_steps+1:end]
w1 = DelayEmbeddings.estimate_delay(x1, "mi_min")

# load the embedding params and make the reconstructions
method_strings = ["cao", "kennel", "hegger", "pec", "pec_multi", "mcdts_L",
                "mcdts_L_multi", "mcdts_fnn", "mcdts_fnn_multi", "mcdts_PRED_MSE",
                "mcdts_PRED_MSE_multi", "mcdts_PRED_KL", "mcdts_PRED_KL_multi",
                "mcdts_PRED_L_KL", "mcdts_PRED_L_KL_multi"]

KK = 1 # number of considered nearest neighbours for prediction
Tw = 1 # forward prediction-step

# preallocation
MSEs_zeroth = zeros(length(method_strings),(T_steps-T_steps2),T_steps2)
MSEs_linear = zeros(length(method_strings),(T_steps-T_steps2),T_steps2)
predictions_zeroth = zeros(length(method_strings),(T_steps-T_steps2),T_steps2)
predictions_linear = zeros(length(method_strings),(T_steps-T_steps2),T_steps2)


for i = 1:length(methods)
    println(i)
    if i == 6 || i == 7
        continue
    end
    aa = readdlm("./application/CenoGrid/Prediction results/results_CENOGRID_taus_"*method_strings[i]*".csv")
    τs = [Int(aa[g]) for g = 1:length(aa)]
    if i == 5 || i == 7 || i == 9 || i == 11 || i == 13 || i == 15
        bb = readdlm("./application/CENOGRID/Prediction results/results_CENOGRID_ts_"*method_strings[i]*".csv")
        ts = [Int(bb[g]) for g = 1:length(bb)]
    else
        ts = ones(Int,length(τs))
    end
    # reconstruction
    for j = 1:(T_steps-T_steps2)

        if i == 5 || i == 7 || i == 9 || i == 11 || i == 13 || i == 15
            if sum(ts .== 1)>0
                tts = findall(x -> x==1, ts)[1]
            else
                tts = ts[1]
            end
            xx = vcat(x1, x2[1:j])
            yy = vcat(y1, y2[1:j])
            data_sample = Dataset(xx,yy) # mutlivariate set
            Y = genembed(data_sample, τs, ts)

            predictions_zeroth[i,j,:] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps2; theiler = w1)[:,tts]
            predictions_linear[i,j,:] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps2; theiler = w1)[:,tts]
        else
            Y = genembed(vcat(x1, x2[1:j]), τs, ts)
            predictions_zeroth[i,j,:] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps2; theiler = w1)[:,1]
            predictions_linear[i,j,:] = MCDTS.iterated_local_zeroth_prediction(Y, KK, T_steps2; theiler = w1)[:,1]
        end
        for k = 1:T_steps2
            MSEs_zeroth[i,j,k] = MCDTS.compute_mse(predictions_zeroth[i,j,1:k], x2[j:j+k-1])
            MSEs_linear[i,j,k] = MCDTS.compute_mse(predictions_linear[i,j,1:k], x2[j:j+k-1])
        end
    end
end



methods = ["Cao", "Kennel", "Hegger", "PECUZAL", "PECUZAL mult.", "MCDTS L",
            "MCDTS L mult.", "MCDTS FNN", "MCDTS FNN mult.", "MCDTS PRED",
            "MCDTS PRED mult.", "MCDTS PRED KL", "MCDTS PRED KL mult.",
            "MCDTS PRED-L KL", "MCDTS PRED-L KL mult."]

MEANs_zeroth = zeros(length(methods), T_steps2)
MEANs_linear = zeros(length(methods), T_steps2)

for i = 1:length(methods)
    MEANs_zeroth[i,:] = mean(MSEs_zeroth[i,:,:], dims =1)
    MEANs_linear[i,:] = mean(MSEs_linear[i,:,:], dims =1)
end


## Save the data
writedlm("./application/CenoGrid/Prediction results/MEANs_zeroth.csv", MEANs_zeroth)
writedlm("./application/CenoGrid/Prediction results/MEANs_linear.csv", MEANs_linear)

for i = 1:length(methods)
    writedlm("./application/CenoGrid/Prediction results/predictions_zeroth_$i.csv", predictions_zeroth[i,:,:])
    writedlm("./application/CenoGrid/Prediction results/predictions_linear_$i.csv", predictions_linear[i,:,:])
    writedlm("./application/CenoGrid/Prediction results/MSEs_zeroth_$i.csv", MSEs_zeroth[i,:,:])
    writedlm("./application/CenoGrid/Prediction results/MSEs_linear_$i.csv", MSEs_linear[i,:,:])
end

## Load Data
MEANs_zeroth = readdlm("./application/CenoGrid/Prediction results/MEANs_zeroth.csv")
MEANs_linear = readdlm("./application/CenoGrid/Prediction results/MEANs_linear.csv")

MSEs_zeroth = zeros(length(method_strings),(T_steps-T_steps2),T_steps2)
MSEs_linear = zeros(length(method_strings),(T_steps-T_steps2),T_steps2)
predictions_zeroth = zeros(length(method_strings),(T_steps-T_steps2),T_steps2)
predictions_linear = zeros(length(method_strings),(T_steps-T_steps2),T_steps2)
for i = 1:length(methods)
    predictions_zeroth[i,:,:] = readdlm("./application/CenoGrid/Prediction results/predictions_zeroth_$i.csv")
    predictions_linear[i,:,:] = readdlm("./application/CenoGrid/Prediction results/predictions_linear_$i.csv")
    MSEs_zeroth[i,:,:] = readdlm("./application/CenoGrid/Prediction results/MSEs_zeroth_$i.csv")
    MSEs_linear[i,:,:] = readdlm("./application/CenoGrid/Prediction results/MSEs_linear_$i.csv")
end



##
begin
    i=11
    figure()
    plot(1:T_steps2, MEANs_zeroth[1,:], label = methods[1])
    plot(1:T_steps2, MEANs_zeroth[i,:], label = methods[i])
    yscale("log")
    grid()
    legend()
    ylim([0.001, 2])
end

##


## plot all results at once
begin
    figure(figsize=(20,10))
    subplot(121)
    for i = 1:length(methods)
        plot(1:T_steps2, MEANs_zeroth[i,:], label = methods[i])
        if i == 1 || i == 2 || i == 3
            scatter(1:T_steps2, MEANs_zeroth[i,:], marker = "*")
        end
    end
    yscale("log")
    title("Iterated 1-step prediction based on zeroth predictor")
    grid()
    legend()
    ylim([0.001, 2])
    subplot(122)
    for i = 1:length(methods)
        plot(1:T_steps2, MEANs_linear[i,:], label = methods[i])
        if i == 1 || i == 2 || i == 3
            scatter(1:T_steps2, MEANs_linear[i,:], marker = "*")
        end
    end
    yscale("log")
    title("Iterated 1-step prediction based on linear predictor")
    grid()
    legend()
    ylim([0.001, 2])
end


# only plot the traditional ones and one additional
num = [8,15]
begin
    figure(figsize=(20,10))
    for i = 1:3
        plot(1:T_steps2, MEANs_zeroth[i,:], label = methods[i])
        if i == 1 || i == 2 || i == 3
            scatter(1:T_steps2, MEANs_zeroth[i,:], marker = "*")
        end
    end
    plot(1:T_steps2, MEANs_zeroth[num[1],:], label = methods[num[1]])
    plot(1:T_steps2, MEANs_zeroth[num[2],:], label = methods[num[2]])
    yscale("log")
    title("Iterated 1-step prediction based on zeroth predictor")
    grid()
    legend()
end


# Plot all 100 trials for two selected methods
num = [2,10]
begin
    figure(figsize=(20,10))
    subplot(121)
    for i = 1:100
        plot(1:T_steps2, MSEs_zeroth[num[1],i,:])
    end
    plot(1:T_steps2,MEANs_zeroth[num[1],:], "k--", linewidth=2.5, label="mean")
    legend()
    yscale("log")
    ylim([0.001, 2])
    #xlabel("Lyapunov time")
    grid()
    title("MSE (100 runs), $(methods[num[1]])")

    subplot(122)
    for i = 1:100
        plot(1:T_steps2, MSEs_zeroth[num[2],i,:])
    end
    plot(1:T_steps2,MEANs_zeroth[num[2],:], "k--", linewidth=2.5, label="mean")
    legend()
    yscale("log")
    ylim([0.001, 2])
    #xlabel("Lyapunov time")
    grid()
    title("MSE (100 runs), $(methods[num[2]])")

end


# # Plot predictions for two selected methods and a selected trial
# trial = 3
# num = [2,5]
#
# begin
#     figure(figsize=(20,10))
#     subplot(211)
#     plot(t2[trial:trial+T_steps2-1], x2[trial:trial+T_steps2-1], label="true")
#     plot(t2[trial:trial+T_steps2-1], predictions_zeroth[num[1],trial,1:T_steps2], label=methods[num[1]])
#     legend()
#     #ylim([0.001, 2])
#     #xlabel("Lyapunov time")
#     grid()
#     title("Predictions for run $trial")
#
#     subplot(212)
#     plot(t2[trial:trial+T_steps2-1], x2[trial:trial+T_steps2-1], label="true")
#     plot(t2[trial:trial+T_steps2-1], predictions_zeroth[num[2],trial,1:T_steps2], label=methods[num[2]])
#     legend()
#     #ylim([0.001, 2])
#     #xlabel("Lyapunov time")
#     grid()
#     title("Predictions for run $trial")
#
# end
