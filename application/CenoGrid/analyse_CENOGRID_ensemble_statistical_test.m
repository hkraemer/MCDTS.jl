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

methods = ["Cao", "Kennel", "Hegger", "PECUZAL", "PECUZAL mult.", "MCDTS L",...
            "MCDTS L mult.", "MCDTS FNN", "MCDTS FNN mult.", "MCDTS PRED",...
            "MCDTS PRED mult.", "MCDTS PRED KL", "MCDTS PRED KL mult.",...
            "MCDTS PRED-L KL", "MCDTS PRED-L KL mult."];
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
alpha = 0.05;

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

%% Plot selected histograms of MSEs
% 
% i=11;
% figure
% subplot(121)
% for j = 1:100
%     plot(1:T_steps2, squeeze(MSEs_zeroth(2,j,:))),hold on
% end
% set(gca, 'YScale', 'log')
% grid on
% ylim([0.001 2])
% 
% subplot(122)
% for j = 1:100
%     plot(1:T_steps2,squeeze(MSEs_zeroth(i,j,:))),hold on
% end
% set(gca, 'YScale', 'log')
% grid on
% ylim([0.001 2])

