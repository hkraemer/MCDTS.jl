% We compare the short-time prediction power of the different methods under
% study by using a Welch-test

clear, clc

data = load("./data/detrended.txt");

t = data(:,1);
O18 = data(:,2);
O13 = data(:,3);

t = fliplr(t);
O18 = fliplr(O18);
O13 = fliplr(O13);

T_steps = 110;
T_steps2 = 10;
number_of_ics = T_steps-T_steps2;

% time series binding
x1 = O18(1:end-T_steps);       % training
x2 = O18(end-T_steps+1:end);   % prediction
y1 = O13(1:end-T_steps);       % training
y2 = O13(end-T_steps+1:end);   % prediction

w1 = 23;

methods = ["Cao", "Kennel", "Hegger", "PECUZAL", "PECUZAL (m)", "MCDTS-C-L",...
            "MCDTS-C-L (m)", "MCDTS-C-FNN", "MCDTS-C-FNN (m)", "MCDTS-R-MSE",...
            "MCDTS-R-MSE (m)", "MCDTS-R-MSE-KL", "MCDTS-R-MSE-KL (m)",...
            "MCDTS-C-MSE-KL", "MCDTS-C-MSE-KL (m)"];
method_strings = ["cao", "kennel", "hegger", "pec", "pec_multi", "mcdts_L",...
                "mcdts_L_multi", "mcdts_fnn", "mcdts_fnn_multi", "mcdts_PRED_MSE",...
                "mcdts_PRED_MSE_multi", "mcdts_PRED_KL", "mcdts_PRED_KL_multi",...
                "mcdts_PRED_L_KL", "mcdts_PRED_L_KL_multi"];

MEANs_zeroth = load("./Prediction results/MEANs_zeroth.csv");
MEANs_linear = load("./Prediction results/MEANs_linear.csv");

MSEs_zeroth = zeros(length(method_strings),(T_steps-T_steps2),T_steps2);
MSEs_linear = zeros(length(method_strings),(T_steps-T_steps2),T_steps2);
predictions_zeroth = zeros(length(method_strings),(T_steps-T_steps2),T_steps2);
predictions_linear = zeros(length(method_strings),(T_steps-T_steps2),T_steps2);
for i = 1:length(methods)
    loadstr = strcat("./Prediction results/predictions_zeroth_",num2str(i),".csv");
    predictions_zeroth(i,:,:) = load(loadstr);
    loadstr = strcat("./Prediction results/predictions_linear_",num2str(i),".csv");
    predictions_linear(i,:,:) = load(loadstr);
    loadstr = strcat("./Prediction results/MSEs_zeroth_",num2str(i),".csv");
    MSEs_zeroth(i,:,:) = load(loadstr);
    loadstr = strcat("./Prediction results/MSEs_linear_",num2str(i),".csv");
    MSEs_linear(i,:,:) = load(loadstr);
end

%% Statistical Test of better prediction performance

thres = 1;  % time step of the prediction
alpha = 0.01;

hs = zeros(length(methods),length(methods));
ps = ones(length(methods),length(methods));
hs_n = zeros(length(methods),length(methods));
ps_n = ones(length(methods),length(methods));
for i = 1:length(methods)
    for j = 1:length(methods)
        d1 = squeeze(MSEs_zeroth(i,:,thres));
        d2 = squeeze(MSEs_zeroth(j,:,thres));
        if median(d1)<median(d2)
            [ps(i,j),hs(i,j)] = ranksum(d1,d2,'alpha',alpha);
        end
        d1 = squeeze(MSEs_linear(i,:,thres));
        d2 = squeeze(MSEs_linear(j,:,thres));
        if median(d1)<median(d2)
            [ps_n(i,j),hs_n(i,j)] = ranksum(d1,d2,'alpha',alpha);
        end
    end
end

hs
ps
hs_n
ps_n

%% Plot data

%% Main results figure

sigma = sqrt(var(x1));

lw = 3.5; % linewidth in the plot
lw2 = 2; % linewidth of the axis
fs = 24; % fontsize
ms = 10; % marker size
lfs = 28; % legend fontsize
plot_idx = [1,2,3,11];


figure('Units','normalized','Position',[.001 .001 .99 .7])

subplot(131)
for i = 1:length(methods)
    if i == 1 || i == 2 || i == 3
        plot(1:T_steps2, MEANs_zeroth(i,:)./sigma, '.-', 'LineWidth',lw ,'Marker','*', 'MarkerSize', ms), hold on
    elseif i == 11
        plot(1:T_steps2, MEANs_zeroth(i,:)./sigma, '.-', 'LineWidth',lw ,'Marker','+', 'MarkerSize', ms), hold on
    end
end
lgd = legend(methods(plot_idx),'Location','southeast');
lgd.FontSize = lfs;
set(gca, 'YScale', 'log')
set(gca, 'Linewidth', lw2)
set(gca, 'FontSize', fs)
title("Forecast error on CENOGRID")
xlabel("prediction time steps")
ylabel("normalized RMS")
grid()
ylim([0.001, 4])

subplot(132)
for i = 1:100
    plot(1:T_steps2, squeeze(MSEs_zeroth(3,i,:)./sigma), 'HandleVisibility','off'), hold on
end
plot(1:T_steps2,MEANs_zeroth(3,:)./sigma, 'k--', 'linewidth', lw)
%legend("mean")
set(gca, 'YScale', 'log')
set(gca, 'Linewidth', lw2)
set(gca, 'FontSize', fs)
title("Forecast error for all 100 trials (Hegger)")
xlabel("prediction time steps")
ylabel("normalized RMS")
grid()
ylim([0.001, 4])

subplot(133)
for i = 1:100
    plot(1:T_steps2, squeeze(MSEs_zeroth(11,i,:)./sigma), 'HandleVisibility','off'), hold on
end
plot(1:T_steps2,MEANs_zeroth(11,:)./sigma, 'k--', 'linewidth', lw)
%legend("mean")
set(gca, 'YScale', 'log')
set(gca, 'Linewidth', lw2)
set(gca, 'FontSize', fs)
title("Forecast error for all 100 trials (MCDTS-R-MSE (m))")
xlabel("prediction time steps")
ylabel("normalized RMS")
grid()
ylim([0.001, 4])

%% Plot data

%% Appendix

%%% CAUTION ! C13 AND O18 SWAPPED %%%%

lw = 2;
fs = 20;

figure('Units','normalized','Position',[.001 .001 .99 .7])
subplot(211)
plot(t,O18, 'Color', [142,144,143]/255, 'LineWidth',lw)
title('detrended Cenozoic Global Reference benthic foraminifer carbon and oxygen Isotope Dataset (CENOGRID)')
xlim([0 t(end)])
ylabel('\delta^{13}C [‰]')
set(gca, 'Xdir','reverse')
set(gca, 'LineWidth',2)
set(gca, 'FontSize',fs)
grid on

subplot(212)
plot(t,O13, 'Color',  [142,144,143]/255, 'LineWidth',lw)
xlim([0 t(end)])
ylabel('\delta^{18}O [‰]')
xlabel('time [Mio yrs BP]')
set(gca, 'Xdir','reverse')
set(gca, 'Ydir','reverse')
set(gca, 'LineWidth',2)
set(gca, 'FontSize',fs)
grid on
