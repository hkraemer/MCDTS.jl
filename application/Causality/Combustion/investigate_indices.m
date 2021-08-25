% We investigate the intervals in the combustion time series for which the 
% first CCM-analysis revealed a stronger causal influence from Heat release
% on pressure. We save this interval and run the analysis again.

clear, clc

pressure = load("./data/Nonstationary/pressure_downsampled.txt");
heat_release = load("./data/Nonstationary/heat_release_downsampled.txt");

%%
% indices with higher heat --> pressure CCM
heats = [1148448, 1023952, 1255263, 1143245];
heats2 = [67596, 1630849, 89087, 34856, 1317389];

% indices with higher pressure --> heat CCM
pressures = [540288, 998265, 814447, 218198, 653379, 978971, 632031, 143725, 785506, 554196, 836660];

t = 1:length(pressure);

lw = 3;
fs = 20;

figure('Units','normalized','Position',[.01 .01 .99 .99])
subplot(211)

plot(t, pressure, '.-', 'LineWidth', 0.5), hold on
for i = 1:length(heats)
    xline(heats(i), 'r-', 'LineWidth',lw), hold on
end

for i = 1:length(heats)
    xline(heats2(i), 'm-', 'LineWidth',lw), hold on
end

for i = 1:length(heats)
    xline(pressures(i), 'g-', 'LineWidth',lw), hold on
end
title("Nonstationary pressure time series (data 2 (Rate = 5; no blowout) from 21.10.2020)")
set(gca,'LineWidth',2, 'FontSize', fs)
xlabel("sample no.")
grid on

subplot(212)
plot(t, heat_release, '.-', 'LineWidth', 0.5), hold on
for i = 1:length(heats)
    xline(heats(i), 'r-', 'LineWidth',lw), hold on
end

for i = 1:length(heats)
    xline(heats2(i), 'm-', 'LineWidth',lw), hold on
end

for i = 1:length(heats)
    xline(pressures(i), 'g-', 'LineWidth',lw), hold on
end
title("Nonstationary Heat release time series data 2 (Rate = 5; no blowout) from 21.10.2020")
set(gca,'LineWidth',2, 'FontSize', fs)
xlabel("sample no.")
grid on

%% Save time series interval, which includes the red lines

pp = pressure(1023952:1255263);
hh = heat_release(1023952:1255263);
save("pressure_downsampled.txt", "pp", "-ascii")
save("heat_release_downsampled.txt", "hh", "-ascii")
