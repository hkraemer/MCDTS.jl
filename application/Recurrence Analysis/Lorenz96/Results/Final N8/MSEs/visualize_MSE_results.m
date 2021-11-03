%% Heatmap - Main results for RQA comparison
% We visualize the normalized mean squared errors from the reconstructed
% RQA's by comparing them with each other
clear, clc

name_str = ["Cao", "Kennel", "Hegger", "PECUZAL", "MCDTS-L", "MCDTS-FNN", ...
                        "PECUZAL (m)", "MCDTS-L (m)", "MCDTS-FNN (m)"];
% load MSE's
MSEs(1,:) = load("MSE_Cao.csv");
MSEs(2,:) = load("MSE_Kennel.csv");
MSEs(3,:) = load("MSE_Hegger.csv");
MSEs(4,:) = load("MSE_Pecuzal.csv");
MSEs(5,:) = load("MSE_MCDTS_L.csv");
MSEs(6,:) = load("MSE_MCDTS_FNN.csv");
MSEs(7,:) = load("MSE_Pecuzal_2.csv");
MSEs(8,:) = load("MSE_MCDTS_L_2.csv");
MSEs(9,:) = load("MSE_MCDTS_FNN_2.csv");

% Since JRRP is a percentage measure, we flip signs
MSEs(:,end) = MSEs(:,end)*(-1);

results = zeros(size(MSEs,1),size(MSEs,1));
for i = 1:size(MSEs,1)
    for j = 1:size(MSEs,1)
        results(i,j) = sum(MSEs(i,:)<MSEs(j,:));
    end
end

%Display Heatmap

fs  = 18; % fontsize
title_str = "Comparison of RQA-results with ground truth (Lorenz96-system, 8 nodes)";

figure('Units','normalized','Position',[.3 .3 .6 .6])
h  = heatmap(name_str, name_str, results*10, 'Title', title_str, ...
    'ColorLimits',[10,90],'CellLabelFormat', '%.4g %%');
colormap(flipud(parula))
h.FontSize = fs;
set(h.YLabel, 'horizontalAlignment', 'left')

%% Exemplary time series from Lorenz96-N8 (for producing grafic)
% from `compute_time_series_for_plotting.jl`
 
clear, clc

dt = 0.01;
N = 8;
data = load("example_trajectory.csv");

subset = 1:1000;
t = dt * 1:size(data,1);

lw = 4; % linewidth
fs = 20; % fontsize

figure('Units','normalized','Position',[.1 .1 .4 .9])
for i = 1:N
    subplot(8,1,i)
    plot(t(subset), data(subset,i), 'LineWidth', lw)
    grid on
    set(gca, 'LineWidth', 2)
    set(gca, 'FontSize', fs)
    yticklabels([])
    if i == 1
        title("Exemplary time series (subset) of L96-system, 8 nodes")
    end
    if i == N
        xlabel('time [a.u.]')
    else
        xticklabels([])
    end
end

%% Exemplary reconstruction from Lorenz96-N8 (for producing grafic)
% from `compute_time_series_for_plotting.jl`

Y = load("example_reconstruction.csv");

figure('Units','normalized','Position',[.2 .2 .5 .5])
plot3(Y(:,1), Y(:,2), Y(:,3), 'LineWidth', 2)
title("Reconstructed attractor (subset)")
set(gca, 'LineWidth', 2)
set(gca, 'FontSize', 20)
xticklabels([])
yticklabels([])
zticklabels([])
grid on

figure('Units','normalized','Position',[.2 .2 .5 .5])
plot3(data(:,1), data(:,2), data(:,3), 'LineWidth', 2)
title("True attractor (subset)")
set(gca, 'LineWidth', 2)
set(gca, 'FontSize', 20)
xticklabels([])
yticklabels([])
zticklabels([])
grid on

%% Exemplary RP from reconstruction from Lorenz96-N8 (for producing grafic)

RP = rp(Y(1:5000,:),0.05,'var');
RP2= rp(data(1:5000,:),0.05,'var');
dt = 0.01;
t = dt * 1:size(RP,1);
figure
imagesc(t,t,RP), colormap([1 1 1; 0 0 0]), axis xy square
xticklabels([])
yticklabels([])
grid on
title("Recurrence Plot of reconstruction (subset)")
xlabel("time [a.u.]")
ylabel("time [a.u.]")
set(gca, 'LineWidth', 2)
set(gca, 'FontSize', 20)

figure
imagesc(t,t,RP2), colormap([1 1 1; 0 0 0]), axis xy square
xticklabels([])
yticklabels([])
grid on
title("Recurrence Plot of true trajectory (subset)")
xlabel("time [a.u.]")
ylabel("time [a.u.]")
set(gca, 'LineWidth', 2)
set(gca, 'FontSize', 20)

%% Exemplary RQA-time series

rqa_ref = load("results_Lorenz96_N_8_final_RQA_ref.csv");
rqa_rec = load("results_Lorenz96_N_8_pec_1d_RQA_pec.csv");
Fs = 3.7:0.002:4;

i = 8; % RQA-value to display

figure
plot(Fs,rqa_ref(:,i), 'LineWidth',3)
yticklabels([])
set(gca, 'LineWidth', 2)
set(gca, 'FontSize', 20)
% title("True RQA")
% xlabel("control parameter F")
% ylabel("RQA value")
xticklabels([])
grid on


figure
plot(Fs,rqa_rec(:,i), 'LineWidth',3)
yticklabels([])
set(gca, 'LineWidth', 2)
set(gca, 'FontSize', 20)
% title("Reconstructed RQA")
% xlabel("control parameter F")
% ylabel("RQA value")
xticklabels([])
grid on

%% Display single RQA-Table
jrrp_pec = load("results_Lorenz96_N_8_pec_1d_RP_frac_pec.csv");

RQA_name = ["RR", "TRANS", "DET", "L_mean", "L_max", "DIV", "ENTR", "TREND","LAM", "TT", "V_max", "V_ENTR", "MRT", "RTE", "NMPRT","JRRP"];
idx_keep = [2,3,4,5,6,7,8,13,14,16]; 
idx_keep2 = [2,3,4,5,6,7,8,13,14,15];
RQA_names = RQA_name(idx_keep);

rqa_ref_example = rqa_ref(100,idx_keep2);
rqa_rec_example = rqa_rec(100,idx_keep2);
rqa_rec_example(10) = jrrp_pec(100);
rqa_ref_example(10) = 1;

T_ref = table(rqa_ref_example', 'RowNames',RQA_names, 'VariableNames',"True");
T_rec = table(rqa_rec_example', 'RowNames',RQA_names, 'VariableNames',"Reconstructed");

% uitable('Data',T_ref{:,:},'ColumnName',T_ref.Properties.VariableNames,...
%     'RowName',T_ref.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
uitable('Data',T_rec{:,:},'ColumnName',T_rec.Properties.VariableNames,...
    'RowName',T_rec.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
