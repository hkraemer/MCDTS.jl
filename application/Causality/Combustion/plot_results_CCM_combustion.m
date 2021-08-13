% We plot the results for the CCM-estimation on two random sampels from 
% the combustion data (pressure & heat release), computed in the scripts 
% in the folder `./Cluster scripts/`

clear, clc

method1 = 1; % set 1: full (one embedding for all time series), 
             % set 0: for each time series a new embedding
             
sample = 13;  % set 1: sample 1, set 2: sample 2 ... until sample 20.


% load the results
if method1 == 0
    lstr1 = strcat('./results/results_analysis_CCM_combustion_',num2str(sample),'_');
else
    lstr1 = strcat('./results/results_analysis_CCM_full_combustion_',num2str(sample),'_');
end

x1(1,:) = load(strcat(lstr1,'x1_cao.csv'));
x1(2,:) = load(strcat(lstr1,'x1_pec.csv'));
x1(3,:) = load(strcat(lstr1,'x1_mcdts.csv'));

x2(1,:) = load(strcat(lstr1,'x2_cao.csv'));
x2(2,:) = load(strcat(lstr1,'x2_pec.csv'));
x2(3,:) = load(strcat(lstr1,'x2_mcdts.csv'));

y1(1,:) = load(strcat(lstr1,'y1_cao.csv'));
y1(2,:) = load(strcat(lstr1,'y1_pec.csv'));
y1(3,:) = load(strcat(lstr1,'y1_mcdts.csv'));

y2(1,:) = load(strcat(lstr1,'y2_cao.csv'));
y2(2,:) = load(strcat(lstr1,'y2_pec.csv'));
y2(3,:) = load(strcat(lstr1,'y2_mcdts.csv'));

rho_p = load(strcat(lstr1,'Pearson.csv'));

% Visualization

ts_lengths = 500:100:5000;

fs = 16;
lw = 3;

figure('Units','normalized','Position',[.001 .001 .6 .9])
plot(ts_lengths,x1(1,:), 'LineWidth', lw), hold on
plot(ts_lengths,x1(2,:), 'LineWidth', lw), hold on
plot(ts_lengths,x1(3,:), 'LineWidth', lw), hold on
plot(ts_lengths,rho_p, 'r--', 'LineWidth', lw)
legend('CCM Cao', 'CCM PECUZAL', 'CCM MCDTS', 'Pearson corr.coeff.')
grid on
xlabel('time series length')
ylabel('correlation')
title(strcat('CCM for sample ',num2str(sample),'{ }','{ }','{ }','{ }','{ }','Heat -> Pressure (pressure embedding)'))
set(gca, 'LineWidth',2, 'Fontsize',fs)
ylim([-0.3 1])

figure('Units','normalized','Position',[.001 .001 .6 .9])
plot(ts_lengths,x2(1,:), 'LineWidth', lw), hold on
plot(ts_lengths,x2(2,:), 'LineWidth', lw), hold on
plot(ts_lengths,x2(3,:), 'LineWidth', lw), hold on
plot(ts_lengths,rho_p, 'r--', 'LineWidth', lw)
legend('CCM Cao', 'CCM PECUZAL', 'CCM MCDTS', 'Pearson corr.coeff.')
grid on
xlabel('time series length')
ylabel('correlation')
title(strcat('CCM for sample ',num2str(sample),'{ }','{ }','{ }','{ }','{ }','Heat -> Pressure (heat embedding)'))
set(gca, 'LineWidth',2, 'Fontsize',fs)
ylim([-0.3 1])

figure('Units','normalized','Position',[.001 .001 .6 .9])
plot(ts_lengths,y1(1,:), 'LineWidth', lw), hold on
plot(ts_lengths,y1(2,:), 'LineWidth', lw), hold on
plot(ts_lengths,y1(3,:), 'LineWidth', lw), hold on
plot(ts_lengths,rho_p, 'r--', 'LineWidth', lw)
legend('CCM Cao', 'CCM PECUZAL', 'CCM MCDTS', 'Pearson corr.coeff.')
grid on
xlabel('time series length')
ylabel('correlation')
title(strcat('CCM for sample ',num2str(sample),'{ }','{ }','{ }','{ }','{ }','Pressure -> Heat (pressure embedding)'))
set(gca, 'LineWidth',2, 'Fontsize',fs)
ylim([-0.3 1])

figure('Units','normalized','Position',[.001 .001 .6 .9])
plot(ts_lengths,y2(1,:), 'LineWidth', lw), hold on
plot(ts_lengths,y2(2,:), 'LineWidth', lw), hold on
plot(ts_lengths,y2(3,:), 'LineWidth', lw), hold on
plot(ts_lengths,rho_p, 'r--', 'LineWidth', lw)
legend('CCM Cao', 'CCM PECUZAL', 'CCM MCDTS', 'Pearson corr.coeff.')
grid on
xlabel('time series length')
ylabel('correlation')
title(strcat('CCM for sample ',num2str(sample),'{ }','{ }','{ }','{ }','{ }','Pressure -> Heat (heat embedding)'))
set(gca, 'LineWidth',2, 'Fontsize',fs)
ylim([-0.3 1])

