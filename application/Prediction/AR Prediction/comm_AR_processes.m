clear, clc

N = 50000; % length of ts and
sigma = 0.4; % std of Gaussian numbers

%% AR(3)
coeffs = [0.4, 0.2, 0.3]; % AR-coeffs
const = -4; % constant offset
t_steps = 2;
true_model = arima('Constant', const, 'AR', coeffs, 'Variance', sigma);

% get time series
y_true = simulate(true_model, N);

% figure
% plot(y_true)
% grid on

% Setup model
est_model = arima('ARLags',1:3);

% estimate params
[EstMdl,estParams,EstParamCov,logL(1)] = estimate(est_model,y_true(1:end-t_steps));

%%
% [E,V] = infer(EstMdl,y_true(1:end-50));
% 
% figure
% plot(y_true(1:end-50)), hold on
% plot(y_true(1:end-50)-E, '.-')
% legend("True", "Fitted")
% grid on

%%

[Y,YMSE] = forecast(EstMdl,t_steps,y_true(end-t_steps-4:end-t_steps));
YY = [y_true(1:end-t_steps); Y];
MSE = [zeros(length(y_true(1:end-t_steps)),1); YMSE];

figure
subplot(211)
plot(y_true), hold on 
plot(YY, '.-')
xline(length(y_true)-t_steps)
legend("True", "Forecasted")
title("Time series forecast")
xlim([49940, 50000])
grid on

subplot(212)
plot(MSE)
ylabel("MSE")
title("Error")
xline(length(y_true)-t_steps)
xlim([49940, 50000])
grid on



%% AR(4)
coeffs = [0.4, 0, 0.3, 0.1]; % AR-coeffs
const = 3; % constant offset

true_model = arima('Constant', const, 'AR', coeffs, 'Variance', sigma);

% get time series
y_true = simulate(true_model, N);

% figure
% plot(y_true)
% grid on

% Setup model
est_model1 = arima('ARLags',1:3);
est_model2 = arima('ARLags',1:4);
est_model3 = arima('ARLags',1:5);

% estimate params
[EstMdl,EstParamCov,logL1,info] = estimate(est_model1,y_true,'display','full');
[EstMdl,EstParamCov,logL2,info] = estimate(est_model2,y_true,'display','full');
[EstMdl,EstParamCov,logL3,info] = estimate(est_model3,y_true,'display','full');

[aic1,bic1] = aicbic(logL1, 5, N);
[aic2,bic2] = aicbic(logL2, 6, N);
[aic3,bic3] = aicbic(logL3, 7, N);

%% AR(5)
coeffs = [0.4, 0, 0.15, 0.1, 0.25]; % AR-coeffs
const = 2; % constant offset

true_model = arima('Constant', const, 'AR', coeffs, 'Variance', sigma);

% get time series
y_true = simulate(true_model, N);

figure
plot(y_true)
grid on

% Setup model
est_model = arima('ARLags',1:5);

% estimate params
[EstMdl,estParams,EstParamCov,logL] = estimate(est_model,y_true);

%% AR(6)
coeffs = [0.3, 0.1, 0, 0.2, 0.25, 0.1]; % AR-coeffs
const = 1; % constant offset

true_model = arima('Constant', const, 'AR', coeffs, 'Variance', sigma);

% get time series
y_true = simulate(true_model, N);

figure
plot(y_true)
grid on

% Setup model
est_model = arima('ARLags',1:6);

% estimate params
[EstMdl,estParams,EstParamCov,logL] = estimate(est_model,y_true);

%% Load time series and try to estimate params
clear, clc

data = load('ar_6_trial.csv');

%%
% Setup model
N = length(data);
est_model1 = arima('ARLags',1:3);
est_model2 = arima('ARLags',1:4);
est_model3 = arima('ARLags',1:5);
est_model4 = arima('ARLags',1:6);
est_model5 = arima('ARLags',1:7);
est_model6 = arima('ARLags',1:8);

% estimate params
[EstMdl1,EstParamCov,logL1(1),info] = estimate(est_model1,data);
[EstMdl2,EstParamCov,logL1(2),info] = estimate(est_model2,data);
[EstMdl3,EstParamCov,logL1(3),info] = estimate(est_model3,data);
[EstMdl4,EstParamCov,logL1(4),info] = estimate(est_model4,data);
[EstMdl5,EstParamCov,logL1(5),info] = estimate(est_model5,data);
[EstMdl6,EstParamCov,logL1(6),info] = estimate(est_model6,data);

[aic,bic] = aicbic(logL1, [5,6,7,8,9,10], N*ones(6,1))

%%
figure
plot(aic), hold on
plot(bic)
legend("AIC", "BIC")
grid on

%%
y1 = simulate(EstMdl1, N);
y2 = simulate(EstMdl2, N);
y3 = simulate(EstMdl3, N);
y4 = simulate(EstMdl4, N);
y5 = simulate(EstMdl5, N);
y6 = simulate(EstMdl6, N);

figure('Units','normalized','Position',[.01 .01 .99 .99])
subplot(321)
plot(data, 'r.-', 'LineWidth', 1.5), hold on
plot(y1, 'LineWidth', 1.5)
legend("TRUE AR(6)","AR(3)")
title(strcat("MSE: ",num2str(sqrt(mean((data-y1).^2)))))
xlim([1, 70])
grid on
set(gca,'LineWidth',2)

subplot(322)
plot(data, 'r.-', 'LineWidth', 1.5), hold on
plot(y2, 'LineWidth', 1.5)
legend("TRUE AR(6)","AR(4)")
title(strcat("MSE: ",num2str(sqrt(mean((data-y2).^2)))))
xlim([1, 70])
grid on
set(gca,'LineWidth',2)

subplot(323)
plot(data, 'r.-', 'LineWidth', 1.5), hold on
plot(y3, 'LineWidth', 1.5)
legend("TRUE AR(6)","AR(5)")
title(strcat("MSE: ",num2str(sqrt(mean((data-y3).^2)))))
xlim([1, 70])
grid on
set(gca,'LineWidth',2)

subplot(324)
plot(data, 'r.-', 'LineWidth', 1.5), hold on
plot(y4, 'LineWidth', 1.5)
legend("TRUE AR(6)","AR(6)")
title(strcat("MSE: ",num2str(sqrt(mean((data-y4).^2)))))
xlim([1, 70])
grid on
set(gca,'LineWidth',2)

subplot(325)
plot(data, 'r.-', 'LineWidth', 1.5), hold on
plot(y5, 'LineWidth', 1.5)
legend("TRUE AR(6)","AR(7)")
title(strcat("MSE: ",num2str(sqrt(mean((data-y5).^2)))))
xlim([1, 70])
grid on
set(gca,'LineWidth',2)

subplot(326)
plot(data, 'r.-', 'LineWidth', 1.5), hold on
plot(y6, 'LineWidth', 1.5)
legend("TRUE AR(6)","AR(8)")
xlim([1, 70])
title(strcat("MSE: ",num2str(sqrt(mean((data-y6).^2)))))
grid on
set(gca,'LineWidth',2)



