# we analyze the multivariate DO-time series data with some simple statistics
using MCDTS
using Statistics
using DelimitedFiles
using DelayEmbeddings
using StatsBase
using PyPlot
pygui(true)

## bind data
data_ = readdlm("./application/do-data/raw data/DO.csv", ',', header=true)
variables = data_[2]
data = Dataset(data_[1])

## plot data
t = 1:length(data)
time_int = 5000:5500

figure(figsize=(20,10))
subplots_adjust(hspace=0.6)
for i = 1:size(data,2)
    subplot(5,2,i)
    plot(t[time_int],data[time_int,i],".-")
    ylabel(variables[i])
    title(variables[i])
    grid()
    if i == 9
        xlabel("time [au]")
    end
end

## simple nonlinear filter and simple moving average
m = 20
neighborhoodsize = 15
filtered = zeros(size(data))
filtered_ma = zeros(size(data,1)-m+1,size(data,2))
for i = 1:size(filtered,2)
    filtered[:,i] = MCDTS.nonlin_noise_reduction(data[:,i], m, neighborhoodsize)
    filtered_ma[:,i] = MCDTS.moving_average(data[:,i],m)
end
filtered = Dataset(filtered[10:end-10,:])
filtered_ma = Dataset(filtered_ma)


## pearson correlation
corr_coeff = cor(Matrix(data))
corr_coeff_f = cor(Matrix(filtered))
corr_coeff_ma = cor(Matrix(filtered_ma))

## time delay and correlation times
ws = zeros(9)
ws_f = zeros(9)
ws_f_ma = zeros(9)
method = "mi_min"
for i = 1:9
    ws[i] = DelayEmbeddings.estimate_delay(data[:,i], method)
    ws_f[i] = DelayEmbeddings.estimate_delay(filtered[:,i], method)
    ws_f_ma[i] = DelayEmbeddings.estimate_delay(filtered_ma[:,i], method)
end
theiler = Int(maximum(ws))
theiler_f = Int(maximum(ws_f))
theiler_f_ma = Int(maximum(ws_f_ma))

taus = 0:100
figure(figsize=(20,10))
subplots_adjust(hspace=0.6)
for i = 1:size(data,2)
    subplot(5,2,i)
    plot(taus,DelayEmbeddings.selfmutualinfo(data[:,i], taus),".-")
    ylabel("MI")
    title("MI of $(variables[i])")
    grid()
    if i == 9
        xlabel("lag τ")
    end
end

figure(figsize=(20,10))
subplots_adjust(hspace=0.6)
for i = 1:size(data,2)
    subplot(5,2,i)
    plot(taus,StatsBase.autocor(data[:,i], taus),".-")
    ylabel("AC")
    title("AC of $(variables[i])")
    grid()
    if i == 9
        xlabel("lag τ")
    end
end

figure(figsize=(20,10))
subplots_adjust(hspace=0.6)
for i = 1:size(data,2)
    subplot(5,2,i)
    plot(taus,DelayEmbeddings.selfmutualinfo(filtered[:,i], taus),".-")
    ylabel("MI")
    title("MI of FILTERED $(variables[i])")
    grid()
    if i == 9
        xlabel("lag τ")
    end
end

figure(figsize=(20,10))
subplots_adjust(hspace=0.6)
for i = 1:size(data,2)
    subplot(5,2,i)
    plot(taus,StatsBase.autocor(filtered[:,i], taus),".-")
    ylabel("AC")
    title("AC of FILTERED $(variables[i])")
    grid()
    if i == 9
        xlabel("lag τ")
    end
end

figure(figsize=(20,10))
subplots_adjust(hspace=0.6)
for i = 1:size(data,2)
    subplot(5,2,i)
    plot(taus,DelayEmbeddings.selfmutualinfo(filtered_ma[:,i], taus),".-")
    ylabel("MI")
    title("MI of FILTERED MA $(variables[i])")
    grid()
    if i == 9
        xlabel("lag τ")
    end
end

figure(figsize=(20,10))
subplots_adjust(hspace=0.6)
for i = 1:size(data,2)
    subplot(5,2,i)
    plot(taus,StatsBase.autocor(filtered_ma[:,i], taus),".-")
    ylabel("AC")
    title("AC of FILTERED MA $(variables[i])")
    grid()
    if i == 9
        xlabel("lag τ")
    end
end


## phase shift unfiltered data
data = data[10:end-10,:]
t = vec(t[10:end-10,:])

## plot data and filtered data

time_int = 5000:6000 # time interval to plot
ts = 1 # time series to plot

figure(figsize=(20,10))
plot(t[time_int], data[time_int,ts], ".-", label=variables[ts])
plot(t[time_int], filtered[time_int,ts], ".-", label="filtered $(variables[ts])")
plot(t[time_int], filtered_ma[time_int,ts], ".-", label="filtered MA $(variables[ts])")
legend()
grid()


## compute and plot the conituity statistics

eps,_ = pecora(filtered, (0,), (1,); delays=taus, w = theiler)

figure(figsize=(20,10))
subplots_adjust(hspace=0.6)
for i = 1:size(data,2)
    subplot(5,2,i)
    plot(taus,eps[:,i],".-")
    ylabel("ε⋆")
    title("ε⋆ of FILTERED $(variables[i])")
    grid()
    if i == 9
        xlabel("lag τ")
    end
end
