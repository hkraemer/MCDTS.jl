%% We plot the time series and a random sample of length 5,000 for 
% visualizing the procedure in the paper

clear, clc

data1 = load('./data/Nonstationary 2/pressure_downsampled.txt');
data2 = load('./data/Nonstationary 2/heat_release_downsampled.txt');
time = load('./data/Nonstationary 2/time_downsampled.txt');

%% 
lw = 1;
fs = 18;

idx = 28301;
% idx = 333487;
N = 5000;

dt = mean(diff(time));
t = 0:dt:(dt*(length(data1)-1));

figure('Units','normalized','Position',[.001 .001 .99 .99])

subplot(221)
plot(t,data1,'LineWidth',lw)
xlabel('time')
xticklabels([])
title('Turbine pressure time series')
set(gca,'LineWidth',2, 'FontSize', fs)
xline(t(idx))
xline(t(idx+N))
ylabel("Pressure [V]")
grid on
xlim([t(1) t(end)])

subplot(222)
plot(t,data2,'LineWidth',lw)
xlabel('time')
xticklabels([])
ylabel("Heat release [mV]")
title('Turbine heat release time series')
set(gca,'LineWidth',2, 'FontSize', fs)
xline(t(idx))
xline(t(idx+N))
grid on
xlim([t(1) t(end)])


dd1 = data1(idx:idx+N);
dd2 = data2(idx:idx+N);
dd1 = (dd1 - mean(dd1)) / std(dd1);
dd2 = (dd2 - mean(dd2)) / std(dd2);

subplot(223)
plot(t(idx:idx+N),dd1,'LineWidth',lw)
xlabel('time [s]')
set(gca,'LineWidth',2, 'FontSize', fs)
grid on
ylabel("normalized pressure")
xlim([t(idx) t(idx+N)])

subplot(224)
plot(t(idx:idx+N),dd2,'LineWidth',lw)
xlabel('time [s]')
set(gca,'LineWidth',2, 'FontSize', fs)
grid on
ylabel("normalized Heat release")
xlim([t(idx) t(idx+N)])
