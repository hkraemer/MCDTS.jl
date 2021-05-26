clear, clc

rmm = load("rmm.mat");
rmm = rmm.rmm;
RMM1 = load("RMM1.csv");
RMM2 = load("RMM2.csv");

t = datetime(rmm(:,1:3));

%% Only use dataset after the big NaN-Gap

start_idx = 1676;
t = t(start_idx:end);
RMM1 = RMM1(start_idx:end);
RMM2 = RMM2(start_idx:end);

%% interpolate RMM2

[RMM22,tf] = fillmissing(RMM2,'pchip');

plot(t,RMM2), hold on
plot(t(tf),RMM22(tf),'r.')
grid on


