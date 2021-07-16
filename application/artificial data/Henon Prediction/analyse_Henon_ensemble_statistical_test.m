% We compare the short-time prediction power of the different methods under
% study by using a Welch-test

clear, clc

number_of_ics = 100; % number of different initial conditions
T_steps = 31;
lyap_time = 1/0.419;

t = 1:T_steps;
t = t ./ lyap_time;

methods = ["Cao", "Kennel", "Hegger", "PECUZAL", "PECUZAL (m)", "MCDTS-C-L",...
            "MCDTS-C-L (m)", "MCDTS-C-FNN", "MCDTS-C-FNN (m)", "MCDTS-R-MSE",...
            "MCDTS-R-MSE (m)", "MCDTS-R-MSE-KL", "MCDTS-R-MSE-KL (m)",...
            "MCDTS-C-MSE-KL", "MCDTS-C-MSE-KL (m)"];

MSEs = ones(15,number_of_ics,T_steps);
MSEs_n = ones(15,number_of_ics,T_steps);

MSEs(1,:,:) = load("./Results 1/results_Henon_MSEs_cao.csv");
MSEs(2,:,:) = load("./Results 1/results_Henon_MSEs_kennel.csv");
MSEs(3,:,:) = load("./Results 1/results_Henon_MSEs_hegger.csv");
MSEs(4,:,:) = load("./Results 1/results_Henon_MSEs_pec.csv");
MSEs(5,:,:) = load("./Results 1/results_Henon_MSEs_pec2.csv");
MSEs(6,:,:) = load("./Results 1/results_Henon_MSEs_mcdts_L.csv");
MSEs(7,:,:) = load("./Results 1/results_Henon_MSEs_mcdts2_L.csv");
MSEs(8,:,:) = load("./Results 1/results_Henon_MSEs_mcdts_FNN.csv");
MSEs(9,:,:) = load("./Results 1/results_Henon_MSEs_mcdts2_FNN.csv");
MSEs(10,:,:) = load("./Results 1/results_Henon_MSEs_mcdts_PRED.csv");
MSEs(11,:,:) = load("./Results 1/results_Henon_MSEs_mcdts2_PRED.csv");
MSEs(12,:,:) = load("./Results 1/results_Henon_MSEs_mcdts_PRED_KL.csv");
MSEs(13,:,:) = load("./Results 1/results_Henon_MSEs_mcdts2_PRED_KL.csv");
MSEs(14,:,:) = load("./Results 1/results_Henon_MSEs_mcdts_PRED_L_KL.csv");
MSEs(15,:,:) = load("./Results 1/results_Henon_MSEs_mcdts2_PRED_L_KL.csv");

MSEs_n(1,:,:) = load("./Results 2/results_Henon_MSEs_cao_n.csv");
MSEs_n(2,:,:) = load("./Results 2/results_Henon_MSEs_kennel_n.csv");
MSEs_n(3,:,:) = load("./Results 2/results_Henon_MSEs_hegger_n.csv");
MSEs_n(4,:,:) = load("./Results 2/results_Henon_MSEs_pec_n.csv");
MSEs_n(5,:,:) = load("./Results 2/results_Henon_MSEs_pec2_n.csv");
MSEs_n(6,:,:) = load("./Results 2/results_Henon_MSEs_mcdts_L_n.csv");
MSEs_n(7,:,:) = load("./Results 2/results_Henon_MSEs_mcdts2_L_n.csv");
MSEs_n(8,:,:) = load("./Results 2/results_Henon_MSEs_mcdts_FNN_n.csv");
MSEs_n(9,:,:) = load("./Results 2/results_Henon_MSEs_mcdts2_FNN_n.csv");
MSEs_n(10,:,:) = load("./Results 2/results_Henon_MSEs_mcdts_PRED_n.csv");
MSEs_n(11,:,:) = load("./Results 2/results_Henon_MSEs_mcdts2_PRED_n.csv");
MSEs_n(12,:,:) = load("./Results 2/results_Henon_MSEs_mcdts_PRED_KL_n.csv");
MSEs_n(13,:,:) = load("./Results 2/results_Henon_MSEs_mcdts2_PRED_KL_n.csv");
MSEs_n(14,:,:) = load("./Results 2/results_Henon_MSEs_mcdts_PRED_L_KL_n.csv");
MSEs_n(15,:,:) = load("./Results 2/results_Henon_MSEs_mcdts2_PRED_L_KL_n.csv");

% compute median values of the distributions
% means of all MSEs
MEANs = zeros(length(methods),T_steps);
MEANs_n = zeros(length(methods),T_steps);
for i = 1:length(methods)
    MEANs(i,:) = mean(MSEs(i,:,:));
    MEANs_n(i,:) = mean(MSEs_n(i,:,:));
end

%% Statistical Test of better prediction performance

% compute distribution of prediction times lower than a given threshold (aacuracy)
% for each method
threshold1 = 0.1;
threshold2 = 0.1;
times = zeros(15,number_of_ics);
times_n = zeros(15,number_of_ics);

% significance level
alphaval = 0.01;

for i = 1:15
    for j = 1:number_of_ics
        if isempty(find(MSEs(i,j,:)>threshold1,1))
            times(i,j) = 0;
        else
            times(i,j) = t(find(MSEs(i,j,:)>threshold1,1));
        end
        if isempty(find(MSEs_n(i,j,:)>threshold2,1))
            times_n(i,j) = 0;
        else
            times_n(i,j) = t(find(MSEs_n(i,j,:)>threshold2,1));
        end
    end
end

hs = zeros(15,15);
ps = ones(15,15);
hs_n = zeros(15,15);
ps_n = ones(15,15);
for i = 1:15
    for j = 1:15
        if median(times(i,:))>median(times(j,:))
            [ps(i,j),hs(i,j)] = ranksum(times(i,:),times(j,:),'alpha',alphaval);
        end
        if median(times_n(i,:))>median(times_n(j,:))
            [ps_n(i,j),hs_n(i,j)] = ranksum(times_n(i,:),times_n(j,:),'alpha',alphaval);
        end
    end
end

hs
ps
hs_n
ps_n

%% Plot data

%% Main results figure

lw = 3.5; % linewidth in the plot
lw2 = 2; % linewidth of the axis
fs = 24; % fontsize
ms = 10; % marker size
lfs = 28; % legend fontsize
plot_idx = [1,2,3,6,7,12,13,14,15];


figure('Units','normalized','Position',[.001 .001 .99 .7])

subplot(121)
for i = 1:length(methods)
    if i == 1 || i == 2 || i == 3
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','*', 'MarkerSize', ms), hold on
    elseif i == 4 || i == 5
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','d', 'MarkerSize', ms), hold on
    elseif i == 12 || i == 13
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','x', 'MarkerSize', ms), hold on
    elseif i == 14 || i == 15
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','^', 'MarkerSize', ms), hold on
    end
end
lgd = legend(methods(plot_idx),'Location','southeast');
lgd.FontSize = lfs;
set(gca, 'YScale', 'log')
set(gca, 'Linewidth', lw2)
set(gca, 'FontSize', fs)
title("Forecast error on Hénon x-time series")
xlabel("Lypunov time")
ylabel("normalized RMS")
grid()
ylim([0.002, 2])

subplot(122)
for i = 1:length(methods)
    if i == 1 || i == 2 || i == 3
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','*', 'MarkerSize', ms), hold on
    elseif i == 6 || i == 7
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','d', 'MarkerSize', ms), hold on
    elseif i == 12 || i == 13
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','x', 'MarkerSize', ms), hold on
    elseif i == 14 || i == 15
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','^', 'MarkerSize', ms), hold on
    end
end
lgd = legend(methods(plot_idx),'Location','southeast');
lgd.FontSize = lfs;
set(gca, 'YScale', 'log')
set(gca, 'Linewidth', lw2)
set(gca, 'FontSize', fs)
title("Forecast error on Hénon x-time series (3% additive noise)")
xlabel("Lypunov time")
ylabel("normalized RMS")
grid()
ylim([0.002, 2])

%% Appendix figure

lw = 3.5; % linewidth in the plot
lw2 = 2; % linewidth of the axis
fs = 22; % fontsize
ms = 10; % marker size
lfs = 26; % legend fontsize


figure('Units','normalized','Position',[.001 .001 .99 .7])

subplot(121)
for i = 1:length(methods)
    if i == 1 || i == 2 || i == 3
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','*', 'MarkerSize', ms), hold on
    elseif i == 4 || i == 5
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','o', 'MarkerSize', ms), hold on
    elseif i == 6 || i == 7
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','d', 'MarkerSize', ms), hold on
    elseif i == 8 || i == 9
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','s', 'MarkerSize', ms), hold on
    elseif i == 10 || i == 11
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','+', 'MarkerSize', ms), hold on
    elseif i == 12 || i == 13
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','x', 'MarkerSize', ms), hold on
    elseif i == 14 || i == 15
        plot(t, MEANs(i,:), '.-', 'LineWidth',lw ,'Marker','^', 'MarkerSize', ms), hold on
    end
end
lgd = legend(methods,'Location','southeast');
lgd.FontSize = lfs;
set(gca, 'YScale', 'log')
set(gca, 'Linewidth', lw2)
set(gca, 'FontSize', fs)
title("Forecast error on Hénon x-time series")
xlabel("Lypunov time")
ylabel("normalized RMS")
grid()
ylim([0.002, 2])

subplot(122)
for i = 1:length(methods)
    if i == 1 || i == 2 || i == 3
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','*', 'MarkerSize', ms), hold on
    elseif i == 4 || i == 5
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','o', 'MarkerSize', ms), hold on
    elseif i == 6 || i == 7
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','d', 'MarkerSize', ms), hold on
    elseif i == 8 || i == 9
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','s', 'MarkerSize', ms), hold on
    elseif i == 10 || i == 11
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','+', 'MarkerSize', ms), hold on
    elseif i == 12 || i == 13
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','x', 'MarkerSize', ms), hold on
    elseif i == 14 || i == 15
        plot(t, MEANs_n(i,:), '.-', 'LineWidth',lw ,'Marker','^', 'MarkerSize', ms), hold on
    end
end
lgd = legend(methods,'Location','southeast');
lgd.FontSize = lfs;
set(gca, 'YScale', 'log')
set(gca, 'Linewidth', lw2)
set(gca, 'FontSize', fs)
title("Forecast error on Hénon x-time series (3% additive noise)")
xlabel("Lypunov time")
ylabel("normalized RMS")
grid()
ylim([0.002, 2])

