% We plot the results for the CCM-estimation on two random sampels from 
% the combustion data (pressure & heat release), computed in the scripts 
% in the folder `./Cluster scripts/`

% Decoding: y1 = Heat -> Pressure, x1 = Pressure -> Heat

clear, clc
         
sample = 8;  % set 1: sample 1, set 2: sample 2 ... until sample 50.


% load the results
lstr1 = strcat('./results 3/results_analysis_CCM_full_combustion_',num2str(sample),'_');


% set colors for bars
c1 = [142/256 144/256 143/256]; % PIK gray
% c2 = [227/256 114/256 34/256]; % PIK orange
c2 = [0/256 159/256 218/256]; % PIK blue

x1(1,:) = load(strcat(lstr1,'x1_cao.csv'));
x1(2,:) = load(strcat(lstr1,'x1_pec.csv'));
x1(3,:) = load(strcat(lstr1,'x1_mcdts.csv'));

y1(1,:) = load(strcat(lstr1,'y1_cao.csv'));
y1(2,:) = load(strcat(lstr1,'y1_pec.csv'));
y1(3,:) = load(strcat(lstr1,'y1_mcdts.csv'));


rho_p = load(strcat(lstr1,'Pearson.csv'));

% Visualization

ts_lengths = 500:100:5000;

fs = 35;
lw = 5;

% compute slope of linear trend
p = polyfit(1:46, x1(1,:), 1);
f1 = polyval(p,1:46);

p = polyfit(1:46, x1(3,:), 1);
f3 = polyval(p,1:46);

figure('Units','normalized','Position',[.001 .001 .6 .9])
h = plot(ts_lengths,x1(1,:), 'LineWidth', lw); hold on
set(h,'Color',c1)
lh = plot(ts_lengths, f1, 'k--', 'LineWidth', 2, 'HandleVisibility', 'off');
lh.Color=[0,0,0,0.5];
% plot(ts_lengths,x1(2,:), 'LineWidth', lw), hold on
h = plot(ts_lengths,x1(3,:), 'LineWidth', lw); hold on
set(h,'Color',c2)
lh = plot(ts_lengths, f3, 'k--', 'LineWidth', 2, 'HandleVisibility', 'off');
lh.Color=[0,0,0,0.5];
plot(ts_lengths,rho_p, 'r--', 'LineWidth', lw)
% legend('CCM Cao', 'CCM PECUZAL', 'CCM MCDTS', 'Pearson corr.coeff.')
legend('CCM Cao', 'CCM MCDTS-C-CCM', '<-> time series')
legend('Location','northwest')
grid on
xlabel('time series length')
ylabel('\rho')
title(strcat('CCM for sample ',num2str(sample),'{ }','{ }','{ }','{ }','{ }','Pressure -> heat'))
set(gca, 'LineWidth',2, 'Fontsize',fs)
ylim([-0.3 1])
xlim([500 5000])

% compute slope of linear trend
p = polyfit(1:46, y1(1,:), 1);
f1 = polyval(p,1:46);

p = polyfit(1:46, y1(3,:), 1);
f3 = polyval(p,1:46);

figure('Units','normalized','Position',[.001 .001 .6 .9])
h = plot(ts_lengths,y1(1,:), 'LineWidth', lw); hold on
set(h,'Color',c1)
lh = plot(ts_lengths, f1, 'k--', 'LineWidth', 2, 'HandleVisibility', 'off');
lh.Color=[0,0,0,0.5];
% plot(ts_lengths,y1(2,:), 'LineWidth', lw), hold on
h = plot(ts_lengths,y1(3,:), 'LineWidth', lw); hold on
set(h,'Color',c2)
lh = plot(ts_lengths, f3, 'k--', 'LineWidth', 2, 'HandleVisibility', 'off');
lh.Color=[0,0,0,0.5];
plot(ts_lengths,rho_p, 'r--', 'LineWidth', lw)
% legend('CCM Cao', 'CCM PECUZAL', 'CCM MCDTS', 'Pearson corr.coeff.')
legend('CCM Cao', 'CCM MCDTS-C-CCM', '<-> time series')
legend('Location','northwest')
grid on
xlabel('time series length')
ylabel('\rho')
title(strcat('CCM for sample ',num2str(sample),'{ }','{ }','{ }','{ }','{ }','Heat -> Pressure'))
set(gca, 'LineWidth',2, 'Fontsize',fs)
ylim([-0.3 1])
xlim([500 5000])

