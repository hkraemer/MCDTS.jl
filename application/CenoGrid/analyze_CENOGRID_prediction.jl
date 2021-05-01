using DelayEmbeddings
using DelimitedFiles
using Random
using Statistics

using PyPlot
pygui(true)


# noise level
σ = .05

# time series to pick
data = readdlm("./application/CENOGRID/data/detrended.txt")

t = data[:,1]
O18 = data[:,2]
O13 = data[:,3]

reverse!(t)
reverse!(O18)
reverse!(O13)

T_steps = 110

# time series binding
x1 = O18[1:end-T_steps]       # training
x2 = O18[end-T_steps+1:end]   # prediction
y1 = O13[1:end-T_steps]       # training
y2 = O13[end-T_steps+1:end]   # prediction
t1 = t[1:end-T_steps]
t2 = t[end-T_steps+1:end]

data_sample = Dataset(x1,y1) # mutlivariate set

methods = ["Cao", "Kennel", "Hegger", "PECUZAL", "PECUZAL mult.", "MCDTS L",
            "MCDTS L mult.", "MCDTS FNN", "MCDTS FNN mult.", "MCDTS PRED",
            "MCDTS PRED mult.", "MCDTS PRED KL", "MCDTS PRED KL mult.",
            "MCDTS PRED-L KL", "MCDTS PRED-L KL mult."]


# load results
begin
    MSEs_zeroth = readdlm("./application/CenoGrid/Prediction results/results_CENOGRID_MSEs_zeroth.csv")
    MSEs_linear = readdlm("./application/CenoGrid/Prediction results/results_CENOGRID_MSEs_linear.csv")
    predictions_zeroth = readdlm("./application/CenoGrid/Prediction results/results_CENOGRID_prediction_zeroth.csv")
    predictions_linear = readdlm("./application/CenoGrid/Prediction results/results_CENOGRID_prediction_zeroth.csv")
end

## Plot results


# plot all results at once
begin

    figure(figsize=(20,10))
    subplot(121)
    for i = 1:length(methods)
        plot(t2, MSEs_zeroth[:,i], label = methods[i])
        if i == 1 || i == 2 || i == 3
            scatter(t2, MSEs_zeroth[:,i], marker = "*")
        end
    end
    yscale("log")
    xlim([t2[1], 0.8])
    title(L"CENOGRID prediction of ^{18}O (zeroth)")
    grid()
    legend()
    ylim([0.1, 2])
    subplot(122)
    for i = 1:length(methods)
        plot(t2, MSEs_linear[:,i], label = methods[i])
        if i == 1 || i == 2 || i == 3
            scatter(t2, MSEs_linear[:,i], marker = "*")
        end
    end
    yscale("log")
    xlim([t2[1], 0.8])
    title(L"CENOGRID prediction of ^{18}O (linear)")
    grid()
    legend()
    ylim([0.1, 2])
end


# Plot all predictions
true_data = vcat(x1,x2)
begin
    t3 = vcat(t1[end],t2)
    ylim1 = -1.5
    ylim2 = 0.6
    is1 = [2,8,9,10,11]
    is2 = [1,12,13,14,15]

    figure(figsize=(20,10))
    cnt = 1
    for i in is1
        subplot(5,1,cnt)
        plot(t, true_data, ".-", label="true data")
        plot(t3, pushfirst!(predictions_zeroth[:,i],x1[end]), ".-", label=methods[i])
        if cnt == 1
            title(L"^{18}O~prediction~(zeroth,~iterated~one-step) ")
        end
        if cnt == 5
            xlabel("time BP [Mio. yrs]")
        end
        xlim(t[end-(T_steps+20)], t[end])
        ylim(ylim1,ylim2)
        legend()
        grid()
        cnt += 1
    end
    subplots_adjust(hspace=.5)

    figure(figsize=(20,10))
    cnt = 1
    for i in is2
        subplot(5,1,cnt)
        plot(t, true_data, ".-", label="true data")
        plot(t3, pushfirst!(predictions_zeroth[:,i],x1[end]), ".-", label=methods[i])
        if cnt == 1
            title(L"^{18}O~prediction~(zeroth,~iterated~one-step) ")
        end
        if cnt == 5
            xlabel("time BP [Mio. yrs]")
        end
        xlim(t[end-(T_steps+20)], t[end])
        ylim(ylim1,ylim2)
        legend()
        grid()
        cnt += 1
    end
    subplots_adjust(hspace=.5)


    figure(figsize=(20,10))
    cnt = 1
    for i in is1
        subplot(5,1,cnt)
        plot(t, true_data, ".-", label="true data")
        plot(t3, pushfirst!(predictions_linear[:,i],x1[end]), ".-", label=methods[i])
        if cnt == 1
            title(L"^{18}O~prediction~(linear,~iterated~one-step) ")
        end
        if cnt == 5
            xlabel("time BP [Mio. yrs]")
        end
        xlim(t[end-(T_steps+20)], t[end])
        ylim(ylim1,ylim2)
        legend()
        grid()
        cnt += 1
    end
    subplots_adjust(hspace=.5)

    figure(figsize=(20,10))
    cnt = 1
    for i in is2
        subplot(5,1,cnt)
        plot(t, true_data, ".-", label="true data")
        plot(t3, pushfirst!(predictions_linear[:,i],x1[end]), ".-", label=methods[i])
        if cnt == 1
            title(L"^{18}O~prediction~(linear,~iterated~one-step) ")
        end
        if cnt == 5
            xlabel("time BP [Mio. yrs]")
        end
        xlim(t[end-(T_steps+20)], t[end])
        ylim(ylim1,ylim2)
        legend()
        grid()
        cnt += 1
    end
    subplots_adjust(hspace=.5)
end
