% We compare the short-time prediction power of the different methods under
% study by using a Welch-test

clear, clc

number_of_ics = 100; % number of different initial conditions
T_steps = 881;
lyap_time = 271;

t = 1:T_steps;
t = t ./ lyap_time;

methods = ["Cao", "Kennel", "Hegger", "PECUZAL", "PECUZAL mult.", "MCDTS L",...
            "MCDTS L mult.", "MCDTS FNN", "MCDTS FNN mult.", "MCDTS PRED",...
            "MCDTS PRED mult.", "MCDTS PRED KL", "MCDTS PRED KL mult."];

MSEs = ones(13,number_of_ics,T_steps);
MSEs_n = ones(13,number_of_ics,T_steps);

MSEs(1,:,:) = load("./Results 2/results_Roessler_MSEs_cao.csv");
MSEs(2,:,:) = load("./Results 2/results_Roessler_MSEs_kennel.csv");
MSEs(3,:,:) = load("./Results 2/results_Roessler_MSEs_hegger.csv");
MSEs(4,:,:) = load("./Results 2/results_Roessler_MSEs_pec.csv");
MSEs(5,:,:) = load("./Results 2/results_Roessler_MSEs_pec2.csv");
MSEs(6,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts_L.csv");
MSEs(7,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts2_L.csv");
MSEs(8,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts_FNN.csv");
MSEs(9,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts2_FNN.csv");
MSEs(10,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts_PRED.csv");
MSEs(11,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts2_PRED.csv");
MSEs(12,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts_PRED_KL.csv");
MSEs(13,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts2_PRED_KL.csv");

MSEs_n(1,:,:) = load("./Results 2/results_Roessler_MSEs_cao_n.csv");
MSEs_n(2,:,:) = load("./Results 2/results_Roessler_MSEs_kennel_n.csv");
MSEs_n(3,:,:) = load("./Results 2/results_Roessler_MSEs_hegger_n.csv");
MSEs_n(4,:,:) = load("./Results 2/results_Roessler_MSEs_pec_n.csv");
MSEs_n(5,:,:) = load("./Results 2/results_Roessler_MSEs_pec2_n.csv");
MSEs_n(6,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts_L_n.csv");
MSEs_n(7,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts2_L_n.csv");
MSEs_n(8,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts_FNN_n.csv");
MSEs_n(9,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts2_FNN_n.csv");
MSEs_n(10,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts_PRED_n.csv");
MSEs_n(11,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts2_PRED_n.csv");
MSEs_n(12,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts_PRED_KL_n.csv");
MSEs_n(13,:,:) = load("./Results 2/results_Roessler_MSEs_mcdts2_PRED_KL_n.csv");

%% Statistical Test of better prediction performance

% compute distribution of prediction times lower than a given threshold (aacuracy)
% for each method
threshold = 0.05;
times = zeros(13,number_of_ics);
times_n = zeros(13,number_of_ics);

for i = 1:13
    for j = 1:number_of_ics
        if isempty(find(MSEs(i,j,:)>threshold,1))
            times(i,j) = 0;
        else
            times(i,j) = t(find(MSEs(i,j,:)>threshold,1));
        end
        if isempty(find(MSEs_n(i,j,:)>threshold,1))
            times_n(i,j) = 0;
        else
            times_n(i,j) = t(find(MSEs_n(i,j,:)>threshold,1));
        end
    end
end

hs = zeros(13,13);
ps = ones(13,13);
hs_n = zeros(13,13);
ps_n = ones(13,13);
for i = 1:13
    for j = 1:13
        if median(times(i,:))>median(times(j,:))
            [ps(i,j),hs(i,j)] = ranksum(times(i,:),times(j,:));
        end
        if median(times_n(i,:))>median(times_n(j,:))
            [ps_n(i,j),hs_n(i,j)] = ranksum(times_n(i,:),times_n(j,:));
        end

%         [ps(i,j),hs(i,j)] = ranksum(times(i,:),times(j,:));
%         [ps_n(i,j),hs_n(i,j)] = ranksum(times_n(i,:),times_n(j,:));

    end
end

hs
ps
hs_n
ps_n

%%



