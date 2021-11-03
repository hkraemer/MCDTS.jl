# We investigate the prediction power based on phase space reconstructions
# for time series stemming from auto-regressive processes (AR).
using MCDTS
using ARFIMA
using PyPlot
using Random
using DelayEmbeddings
using DelimitedFiles
using StructuredOptimization
pygui(true)

N, σ = 10_000, 0.5 # length of ts and std of Gaussian numbers
a, b, c = 0.4, 0.2, 0.3 # Parameters of AR-3
X = arfima(MersenneTwister(1234),N, σ, nothing, SVector(a, b, c))           # ARFIMA(1,0,0) ≡ AR(1)

figure()
plot(X)
xlabel("time [au]")
title("AR(3)-Process")
grid()

# Prediction horizon
T_steps = 30
## Embedding
w1 = DelayEmbeddings.estimate_delay(X, "mi_min")
w2 = DelayEmbeddings.estimate_delay(X, "ac_zero")

dmax = 15
D = zeros(Int,8)

Y_cao_mi, τ_tde_cao_mi, _ = optimal_traditional_de(X[1:end-T_steps], "afnn", "mi_min"; dmax = dmax)
D[1] = size(Y_cao_mi,2)
τ_tdes = [(i-1)*τ_tde_cao_mi for i = 1:D[1]]
Y_cao_mi = MCDTS.genembed_for_prediction(X[1:end-T_steps], τ_tdes)

Y_cao_ac, τ_tde_cao_ac, _ = optimal_traditional_de(X[1:end-T_steps], "afnn", "ac_zero"; dmax = dmax)
D[2] = size(Y_cao_ac,2)
τ_tdes = [(i-1)*τ_tde_cao_ac for i = 1:D[2]]
Y_cao_ac = MCDTS.genembed_for_prediction(X[1:end-T_steps], τ_tdes)

Y_kennel_mi, τ_tde_kennel_mi, _ = optimal_traditional_de(X[1:end-T_steps], "fnn", "mi_min"; dmax = dmax)
D[3] = size(Y_kennel_mi,2)
τ_tdes = [(i-1)*τ_tde_kennel_mi for i = 1:D[3]]
Y_kennel_mi = MCDTS.genembed_for_prediction(X[1:end-T_steps], τ_tdes)

Y_kennel_ac, τ_tde_kennel_ac, _ = optimal_traditional_de(X[1:end-T_steps], "fnn", "ac_zero"; dmax = dmax)
D[4] = size(Y_kennel_ac,2)
τ_tdes = [(i-1)*τ_tde_kennel_ac for i = 1:D[4]]
Y_kennel_ac = MCDTS.genembed_for_prediction(X[1:end-T_steps], τ_tdes)

Y_hegger_mi, τ_tde_hegger_mi, _ = optimal_traditional_de(X[1:end-T_steps], "ifnn", "mi_min"; dmax = dmax)
D[5] = size(Y_hegger_mi,2)
τ_tdes = [(i-1)*τ_tde_hegger_mi for i = 1:D[5]]
Y_hegger_mi = MCDTS.genembed_for_prediction(X[1:end-T_steps], τ_tdes)

Y_hegger_ac, τ_tde_hegger_ac, _ = optimal_traditional_de(X[1:end-T_steps], "ifnn", "ac_zero"; dmax = dmax)
D[6] = size(Y_hegger_ac,2)
τ_tdes = [(i-1)*τ_tde_hegger_ac for i = 1:D[6]]
Y_hegger_ac = MCDTS.genembed_for_prediction(X[1:end-T_steps], τ_tdes)

# Pecuzal
taus = 0:100
_, τ_pec, ts_pec, Ls_pec , _ = DelayEmbeddings.pecuzal_embedding(X[1:end-T_steps]; τs = taus , w = w1, econ = true)
Y_pec = MCDTS.genembed_for_prediction(X[1:end-T_steps], τ_pec)
D[7] = size(Y_pec,2)

# MCDTS
trials = 10
tree = MCDTS.mc_delay(Dataset(X[1:end-T_steps]), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:2:taus[end], max_depth = 15)
best_node = MCDTS.best_embedding(tree)
τ_mcdts = best_node.τs
Y_mcdts = MCDTS.genembed_for_prediction(X[1:end-T_steps], τ_mcdts)
D[8] = size(Y_mcdts,2)


# save results
writedlm("./application/artificial data/AR Prediction/Results 1st trial/tau_cao_mi.csv", τ_tde_cao_mi)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/tau_cao_ac.csv", τ_tde_cao_ac)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/Y_cao_mi.csv", Y_cao_mi)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/Y_cao_mi.csv", Y_cao_ac)

writedlm("./application/artificial data/AR Prediction/Results 1st trial/tau_kennel_mi.csv", τ_tde_kennel_mi)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/tau_kennel_ac.csv", τ_tde_kennel_ac)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/Y_kennel_mi.csv", Y_kennel_mi)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/Y_kennel_mi.csv", Y_kennel_ac)

writedlm("./application/artificial data/AR Prediction/Results 1st trial/tau_hegger_mi.csv", τ_tde_hegger_mi)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/tau_hegger_ac.csv", τ_tde_hegger_ac)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/Y_hegger_mi.csv", Y_hegger_mi)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/Y_hegger_mi.csv", Y_hegger_ac)

writedlm("./application/artificial data/AR Prediction/Results 1st trial/tau_pec.csv", τ_pec)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/tau_mcdts.csv", τ_mcdts)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/Y_pec.csv", Y_pec)
writedlm("./application/artificial data/AR Prediction/Results 1st trial/Y_mcdts.csv", Y_mcdts)

# Model the data using least-squares cost function and Akaike's or Rissanen's
# penalty term
order = 3
coeffs_akaike = Variable(order)
coeffs_rissanen = Variable(order)
# construct Delay-1 Matrix:
N_eff = N-order
A = zeros(N_eff, order)
A[:,1] = X[1:N_eff]
A[:,2] = X[2:N_eff+1]
A[:,3] = X[3:N_eff+2]
@minimize ls( A*coeffs_akaike - X[order+1:end] ) #* exp(order/N) # Akaike

~coeffs_akaike






## Make predictions based on reconstructions
Y = []
push!(Y, Y_cao_mi)
push!(Y, Y_cao_ac)
push!(Y, Y_kennel_mi)
push!(Y, Y_kennel_ac)
push!(Y, Y_hegger_mi)
push!(Y, Y_hegger_ac)
push!(Y, Y_pec)
push!(Y, Y_mcdts)
# make predictions with different neighborhoodsizes and optimize w.r.t. this parameter
max_neighbours = 30
theiler = w1
Ks = zeros(Int,8)
methodstr = ["Cao-Mi", "Cao-AC", "Kennel-Mi", "Kennel-Ac", "Hegger-Mi",  "Hegger-Ac", "PECUZAL", "MCDTS"]
for j = 1:8
    YY = Y[j]
    global max_neighbours
    global theiler
    MSEs = zeros(max_neighbours-(D[j]))
    cnt = 1
    for K = (D[j]+1):max_neighbours
        prediction = MCDTS.local_linear_prediction(YY[1:end-1,:], K; theiler = theiler)
        MSEs[cnt] = MCDTS.compute_mse(prediction, Vector(YY[end,:]))
        cnt += 1
    end
    xs = (D[j]+1):max_neighbours
    Ks[j] = xs[findmin(MSEs)[2]]
    figure()
    plot(xs,MSEs, label="loc-lin")
    legend()
    grid()
    xlabel("Neighbourhoodsize [no. of neighbours]")
    ylabel("root mean squared error")
    title(methodstr[j])
end

# make model
predictions = []
for j = 1:8
    println(j)
    global T_steps
    global predictions
    YY = Y[j]
    prediction = deepcopy(YY[1:end-T_steps,:])
    for T = 1:T_steps
        predicted = MCDTS.local_linear_prediction(prediction, Ks[j]; theiler = w1)
        #predicted = MCDTS.local_random_analogue_prediction(prediction, Ks[j]; theiler = w1)
        push!(prediction,predicted)
    end
    push!(predictions, prediction)
end

figure(figsize=(20,10))
for j = 1:8
    YY = Y[j]
    time_axis = 1:length(YY)
    sp = length(YY)-T_steps
    t2 = (-sp+1:T_steps)
    prediction = predictions[j]

    subplot(4,2,j)
    plot(t2, YY[:,1], ".-", label="true data")
    #plot(t2[end-T_steps+1:end], prediction[length(YY)-T_steps+1:length(YY),1], ".-", label="prediction [loc-lin]")
    plot(t2[end-T_steps:end], prediction[length(YY)-T_steps:length(YY),1], ".-", label="prediction [loc-lin]")
    title(methodstr[j])
    xlabel("time units")
    xlim(-30,T_steps)
    legend()
    grid()
    subplots_adjust(hspace=.6)
end


# make 1-step ahead prediction
MSEs = zeros(8)
cnt = 1
for j = 1:8
    global cnt
    global MSEs
    YY = Y[j]
    prediction = deepcopy(YY)
    predicted = MCDTS.local_linear_prediction(prediction, Ks[j]; theiler = w1)
    MSEs[cnt] = MCDTS.compute_mse(predicted, Vector(X[length(YY),:]))
    println("MSE: $(MSEs[cnt]) for "*methodstr[j])
    cnt += 1
end
