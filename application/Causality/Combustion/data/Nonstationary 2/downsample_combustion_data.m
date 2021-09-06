% Here we downsample the combustion data, in order to better work with them
% for the causality-based-on-state-space-reconstruction approach and MCDTS


clear, clc

% pick a small rate of linear control parameter increase

pressure_raw = load("./09Aug2021/Pressure/02. 100sec_14slpm_fuel/1.txt");
heat_raw = load("./09Aug2021/PMT/02. 100sec_14slpm_fuel/1.txt");

%% Visual inspection

figure()
subplot(211)
plot(pressure_raw(1:5000,1),pressure_raw(1:5000,2),'.-')
grid on
subplot(212)
plot(heat_raw(1:5000,1),heat_raw(1:5000,2),'.-')
grid on


%% Downsampling

pressure = pressure_raw(:,2);
heat = heat_raw(:,2);
t = heat_raw(:,1);

%% Save data

save('heat_release_downsampled.txt', 'heat', '-ascii')
save('pressure_downsampled.txt', 'pressure', '-ascii')
save('time_downsampled.txt', 't', '-ascii')