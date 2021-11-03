% Here we pre-process the combustion data, in order to better work with them
% for the causality-based-on-state-space-reconstruction approach and MCDTS

clear ,clc
t = [];
heat_release = [];
pressure = [];
for i = 5:5
    data = load(strcat("p",num2str(i),".txt"));
    t = vertcat(t,data(:,1));
    heat_release = vertcat(heat_release,data(:,2));
    pressure = vertcat(pressure,data(:,3));
end

dt = mean(diff(t));
time = 0:dt:(dt*(length(pressure)-1));
%%
save("heat_release_stationary.txt","heat_release","-ascii")
save("pressure_stationary.txt","pressure","-ascii")
save("t_stationary.txt","time","-ascii")