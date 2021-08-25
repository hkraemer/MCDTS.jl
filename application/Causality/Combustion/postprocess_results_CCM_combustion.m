%% We postprocess the CCM-estimate data for all trials and for all 
% reconstruction methods. Specifically, we want to investigate to which
% extend the reconstruction methods are able to support the
% expert-hypothesis that the heat-release variable is causally related to
% the pressure variable and vice verca, but heat -> pressure is supposed to
% have a stronger influence:
%
% "We discover a possible asymmetric bidi- rectional coupling between q ̇′ 
% and p′, wherein q ̇′ is observed to exert a stronger influence on p′ than 
% vice versa. Published by AIP Publishing. https://doi.org/10.1063/1.5052210"
%
% To check this we here analyse the CCM estimates (coefficient) as a function 
% of timeseries length for all considered methods (Cao's, PECUZAL, MCDTS) and 
% for 20 random samples from the whole, non-stationary time series for the 
% heat release and the pressure.
% In order to state a causal relationship the CCM-coefficient needs to
% increase with considered time series length and needs to be "sufficiently
% high". In order to test for this we 
% 
% 1) look for a minimum within the interval [0 2,000] and from this minimum we 
% 2) fit a linear model. If this gives a positiv slope AND 
% 3) the CCM-coefficient for the longest considered time series length 
% (last point of each CCM-coefficient time series) is larger than 0.3, then 
% we consider a causal relationship.
% IF such a relationship is found in both directions, then we 
% 4) compare the strengths of both relations by looking at the average 
% point-wise difference (average residuals), in order to see whether the 
% causal relation from heat -> pressure is higher than vice versa 
% (Expectation value)  

% Decoding of the synomyms:
% x1: Pressure causally affects Heat release 
% y1: Heat release causally affects Pressure
%
% We also plot the results for further grafical postprocessing in AI

clear, clc

sample_size = 50;  % considered batch size


sample = 1;
lstr1 = strcat('./results/results_analysis_CCM_full_combustion_',num2str(sample),'_');
rho_p = load(strcat(lstr1,'Pearson.csv'));

y1_cao = zeros(sample_size,length(rho_p));
y1_pec = zeros(sample_size,length(rho_p));
y1_mcdts = zeros(sample_size,length(rho_p));

x1_cao = zeros(sample_size,length(rho_p));
x1_pec = zeros(sample_size,length(rho_p));
x1_mcdts = zeros(sample_size,length(rho_p));

rho_ps = zeros(sample_size,length(rho_p));

for sample = 1:sample_size
    
    lstr1 = strcat('./results/results_analysis_CCM_full_combustion_',num2str(sample),'_');
    
    x1_cao(sample,:) = load(strcat(lstr1,'x1_cao.csv'));
    x1_pec(sample,:) = load(strcat(lstr1,'x1_pec.csv'));
    x1_mcdts(sample,:) = load(strcat(lstr1,'x1_mcdts.csv'));

    y1_cao(sample,:) = load(strcat(lstr1,'y1_cao.csv'));
    y1_pec(sample,:) = load(strcat(lstr1,'y1_pec.csv'));
    y1_mcdts(sample,:) = load(strcat(lstr1,'y1_mcdts.csv'));

    rho_ps(sample,:) = load(strcat(lstr1,'Pearson.csv'));
    
end

%% 1) Minimum detection

min_idx = 1; % index corresponding to time series length 500
% min_idx = 6; % index corresponding to time series length 1,000
% min_idx = 16; % index corresponding to time series length 2,000

[x1_cao_min, x1_cao_min_idx]  = min(x1_cao(:,1:min_idx),[],2);
[x1_pec_min, x1_pec_min_idx]  = min(x1_pec(:,1:min_idx),[],2);
[x1_mcdts_min, x1_mcdts_min_idx]  = min(x1_mcdts(:,1:min_idx),[],2);

[y1_cao_min, y1_cao_min_idx]  = min(y1_cao(:,1:min_idx),[],2);
[y1_pec_min, y1_pec_min_idx]  = min(y1_pec(:,1:min_idx),[],2);
[y1_mcdts_min, y1_mcdts_min_idx]  = min(y1_mcdts(:,1:min_idx),[],2);

% x1_cao_min_idx  = ones(1,sample_size);
% x1_pec_min_idx  = ones(1,sample_size);
% x1_mcdts_min_idx  = ones(1,sample_size);
% 
% y1_cao_min_idx  = ones(1,sample_size);
% y1_pec_min_idx  = ones(1,sample_size);
% y1_mcdts_min_idx  = ones(1,sample_size);


%% 2) Fit linear model

for sample = 1:sample_size
    p = polyfit(x1_cao_min_idx(sample):46, x1_cao(sample,x1_cao_min_idx(sample):end), 1);
    p_x1_cao(sample) = p(1);
    p = polyfit(x1_pec_min_idx(sample):46, x1_pec(sample,x1_pec_min_idx(sample):end), 1);
    p_x1_pec(sample) = p(1); 
    p = polyfit(x1_mcdts_min_idx(sample):46, x1_mcdts(sample,x1_mcdts_min_idx(sample):end), 1);
    p_x1_mcdts(sample) = p(1); 
        
    p = polyfit(y1_cao_min_idx(sample):46, y1_cao(sample,y1_cao_min_idx(sample):end), 1);
    p_y1_cao(sample) = p(1);
    p= polyfit(y1_pec_min_idx(sample):46, y1_pec(sample,y1_pec_min_idx(sample):end), 1);
    p_y1_pec(sample) = p(1);
    p = polyfit(y1_mcdts_min_idx(sample):46, y1_mcdts(sample,y1_mcdts_min_idx(sample):end), 1);
    p_y1_mcdts(sample) = p(1);
    
end

%% 3) Check for positive slope AND sufficiently high CCM-corrcoeff

threshold = 0.2; % minimum acceptable value of last CCM-rho
max_threshold = 0.9; % if CCM-rho exceeds this, then it does not matter if the slope is negative

x1_causal_cao = false(1,sample_size);
x1_causal_pec = false(1,sample_size);
x1_causal_mcdts = false(1,sample_size);

y1_causal_cao = false(1,sample_size);
y1_causal_pec = false(1,sample_size);
y1_causal_mcdts = false(1,sample_size);

for sample = 1:sample_size
    
    if p_x1_cao(sample) > 0 && x1_cao(sample,end) > threshold, x1_causal_cao(sample) = true; elseif x1_cao(sample,end) > max_threshold, x1_causal_cao(sample) = true; end
    if p_x1_pec(sample) > 0 && x1_pec(sample,end) > threshold, x1_causal_pec(sample) = true; elseif x1_pec(sample,end) > max_threshold, x1_causal_pec(sample) = true;end
    if p_x1_mcdts(sample) > 0 && x1_mcdts(sample,end) > threshold, x1_causal_mcdts(sample) = true; elseif x1_mcdts(sample,end) > max_threshold, x1_causal_mcdts(sample) = true;end
     
    if p_y1_cao(sample) > 0 && y1_cao(sample,end) > threshold, y1_causal_cao(sample) = true; elseif y1_cao(sample,end) > max_threshold, y1_causal_cao(sample) = true; end
    if p_y1_pec(sample) > 0 && y1_pec(sample,end) > threshold, y1_causal_pec(sample) = true; elseif y1_pec(sample,end) > max_threshold, y1_causal_pec(sample) = true; end
    if p_y1_mcdts(sample) > 0 && y1_mcdts(sample,end) > threshold, y1_causal_mcdts(sample) = true; elseif y1_mcdts(sample,end) > max_threshold, y1_causal_mcdts(sample) = true; end
    
end

%% Count number of found causal relationships

% simultaneous causal detection:
sim_cao = (x1_causal_cao .* y1_causal_cao);
sim_pec = (x1_causal_pec .* y1_causal_pec);
sim_mcdts = (x1_causal_mcdts .* y1_causal_mcdts);

clc

display("Pressure causally affects heat release:")
x1_cao_true = sum(x1_causal_cao);
x1_pec_true = sum(x1_causal_pec);
x1_mcdts_true = sum(x1_causal_mcdts);
display(strcat("Cao: ",num2str(x1_cao_true),"/50 = ",num2str(x1_cao_true/sample_size)))
display(strcat("Pecuzal: ",num2str(x1_pec_true),"/50 = ",num2str(x1_pec_true/sample_size)))
display(strcat("MCDTS: ",num2str(x1_mcdts_true),"/50 = ",num2str(x1_mcdts_true/sample_size)))
display("%%%%%%%%%%%%%%%%")

display("Heat release causally affects pressure:")
y1_cao_true = sum(y1_causal_cao);
y1_pec_true = sum(y1_causal_pec);
y1_mcdts_true = sum(y1_causal_mcdts);
display(strcat("Cao: ",num2str(y1_cao_true),"/50 = ",num2str(y1_cao_true/sample_size)))
display(strcat("Pecuzal: ",num2str(y1_pec_true),"/50 = ",num2str(y1_pec_true/sample_size)))
display(strcat("MCDTS: ",num2str(y1_mcdts_true),"/50 = ",num2str(y1_mcdts_true/sample_size)))
display("%%%%%%%%%%%%%%%%")

display("Detection of both causal effects at the same time:")
y1_cao_true = sum(sim_cao);
y1_pec_true = sum(sim_pec);
y1_mcdts_true = sum(sim_mcdts);
display(strcat("Cao: ",num2str(y1_cao_true),"/50 = ",num2str(y1_cao_true/sample_size)))
display(strcat("Pecuzal: ",num2str(y1_pec_true),"/50 = ",num2str(y1_pec_true/sample_size)))
display(strcat("MCDTS: ",num2str(y1_mcdts_true),"/50 = ",num2str(y1_mcdts_true/sample_size)))
display("%%%%%%%%%%%%%%%%")

%% Plot simple bar chart, which will get postprocessed in AI
clc

% set colors for bars
c1 = [142/256 144/256 143/256]; % PIK gray
% c2 = [227/256 114/256 34/256]; % PIK orange
c2 = [0/256 159/256 218/256]; % PIK blue


yy(1,1) = x1_cao_true/sample_size
yy(1,2) = 1 - x1_cao_true/sample_size;

yy(2,1) = x1_mcdts_true/sample_size
yy(2,2) = 1 - x1_mcdts_true/sample_size;

figure('Units','normalized','Position',[.001 .001 .99 .99])
subplot(121)
h = barh(1:2,yy,'stacked'); hold on
set(h(1),'FaceColor',c1);
set(h(2),'FaceColor',c2);
set(gca,'LineWidth',2)
grid on
box off
yticklabels(["CAO", "MCDTS-CCM"])
title("Pressure causally affects heat:")


yy(1,1) = y1_cao_true/sample_size
yy(1,2) = 1 - y1_cao_true/sample_size;

yy(2,1) = y1_mcdts_true/sample_size
yy(2,2) = 1 - y1_mcdts_true/sample_size;

subplot(122)
h = barh(1:2,yy,'stacked'); hold on
set(h(1),'FaceColor',c1);
set(h(2),'FaceColor',c2);
set(gca,'LineWidth',2)
grid on
yticklabels(["CAO", "MCDTS-CCM"])
set(gca, 'XDir', 'reverse')
box off
title("Heat causally affects pressure:")


%% 4) Check the strength of the interaction
clc
diff_cao = NaN*ones(1,sample_size);
diff_pec = NaN*ones(1,sample_size);
diff_mcdts = NaN*ones(1,sample_size);

for sample = 1:sample_size
    
    if x1_causal_cao(sample) && y1_causal_cao(sample), diff_cao(sample) = mean(y1_cao(sample,:) - x1_cao(sample,:)); end
    if x1_causal_pec(sample) && y1_causal_pec(sample), diff_pec(sample) = mean(y1_pec(sample,:) - x1_pec(sample,:)); end
    if x1_causal_mcdts(sample) && y1_causal_mcdts(sample), diff_mcdts(sample) = mean(y1_mcdts(sample,:) - x1_mcdts(sample,:)); end

end

%% Plot results (will become a figure in the paper)
clc

fs = 35;

% set colors for bars
c1 = [142/256 144/256 143/256]; % PIK gray
% c2 = [227/256 114/256 34/256]; % PIK orange
c2 = [0/256 159/256 218/256]; % PIK blue

c = 200;

yy = vertcat(diff_cao,diff_mcdts);
xxx = isnan(yy);
xx = false(2,50);
xx(xxx) = true;

figure('Units','normalized','Position',[.001 .001 .99 .99])
h = bar(1:sample_size,yy); hold on
set(h(1),'FaceColor',c1);
set(h(2),'FaceColor',c2);
% Get x centers; XOffset is undocumented
xCnt = (get(h(1),'XData') + cell2mat(get(h,'XOffset'))).';
xx1 = xCnt(xx(1,:),1);
scatter(xx1, zeros(1,length(xx1)), c, c1, 'd', 'filled'), hold on
xx1 = xCnt(xx(2,:),2);
scatter(xx1, zeros(1,length(xx1)), c, c2, 'd', 'filled'), hold on
grid on
title("CCM based on embedding of the pressure time series")
legend("Cao", "MCDTS-C-CCM")
ylabel("avrg. residuals")
xlabel("samples")
set(gca, 'FontSize',fs, 'LineWidth',2);
xticks(1:sample_size)
xticklabels([])
ylim([-0.5 0.6])




yy = vertcat(diff_cao,diff_mcdts);
xx = isnan(yy);
yy(xx) = -9999999;
yy = sort(yy,2,'descend');
xxx = find(yy == -9999999);
yy(xxx) = nan;
xx = false(2,50);
xx(xxx) = true;

figure('Units','normalized','Position',[.001 .001 .99 .99])
h = bar(1:sample_size,yy); hold on
set(h(1),'FaceColor',c1);
set(h(2),'FaceColor',c2);
% Get x centers; XOffset is undocumented
xCnt = (get(h(1),'XData') + cell2mat(get(h,'XOffset'))).';
xx1 = xCnt(xx(1,:),1);
scatter(xx1, zeros(1,length(xx1)), c, c1, 'd', 'filled'), hold on
xx1 = xCnt(xx(2,:),2);
scatter(xx1, zeros(1,length(xx1)), c, c2, 'd', 'filled'), hold on
grid on
title("CCM based on embedding of the pressure time series")
legend("Cao", "MCDTS-C-CCM")
ylabel("avrg. residuals")
xlabel("samples")
set(gca, 'FontSize',fs, 'LineWidth',2);
xticks(1:sample_size)
xticklabels([])
ylim([-0.5 0.6])
