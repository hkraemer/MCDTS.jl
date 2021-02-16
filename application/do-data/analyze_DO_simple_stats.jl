# we analyze the multivariate DO-time series data with some simple statistics
using MCDTS
using Statistics
using DelimitedFiles
using DelayEmbeddings
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
filtered = zeros(size(data))
filtered_ma = zeros(size(data))
m = 20
neighborhoodsize = 15
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
for i = 1:9
    ws[i] = DelayEmbeddings.estimate_delay(data[:,i],"mi_min")
    ws_f[i] = DelayEmbeddings.estimate_delay(filtered[:,i],"mi_min")
end
theiler = Int(maximum(ws))
theiler_f = Int(maximum(ws_f))

taus = 0:20
figure(figsize=(20,10))
subplots_adjust(hspace=0.6)
for i = 1:size(data,2)
    subplot(5,2,i)
    plot(taus,DelayEmbeddings.selfmutualinfo(data[:,i], taus),".-")
    ylabel(variables[i])
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
    plot(taus,DelayEmbeddings.selfmutualinfo(filtered[:,i], taus),".-")
    ylabel(variables[i])
    title("MI of FILTERED $(variables[i])")
    grid()
    if i == 9
        xlabel("lag τ")
    end
end



eps,_ = pecora(filtered, (0,), (1,); delays=0:200, w = theiler)

figure()
plot(eps)
grid()

Y_pec, τ_pec, ts_pec, Ls, _ = pecuzal_embedding(data[:,1]; τs=taus, w = theiler, econ=true)
