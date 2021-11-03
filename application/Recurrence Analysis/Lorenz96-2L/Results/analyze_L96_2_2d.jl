using DifferentialEquations
using RecurrenceAnalysis
using LinearAlgebra
using StatsBase
using Random
using DelayEmbeddings
using MCDTS
using DelimitedFiles
using PyPlot
pygui(true)


## Load computed data
fast = false
cd("./application/artificial data/Lorenz96-2L/Results")
if fast
    ts_sample = [3,19]
    tau_pec = Int.(vec(readdlm("./slow and fast variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_tau_pec.csv")))
    tau_mcdts = Int.(vec(readdlm("./slow and fast variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_tau_mcdts.csv")))
    ts_pec = Int.(vec(readdlm("./slow and fast variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_ts_pec.csv")))
    ts_mcdts = Int.(vec(readdlm("./slow and fast variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_ts_mcdts.csv")))
    optimal_d_pec = Int.(vec(readdlm("./slow and fast variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_optimal_d_pec.csv")))
    optimal_d_mcdts = Int.(vec(readdlm("./slow and fast variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_optimal_d_mcdts.csv")))
    L_pec = vec(readdlm("./slow and fast variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_L_pec.csv"))
    L_mcdts = vec(readdlm("./slow and fast variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_L_mcdts.csv"))
else
    ts_sample = [3,13]
    tau_pec = Int.(vec(readdlm("./slow and slow variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_tau_pec.csv")))
    tau_mcdts = Int.(vec(readdlm("./slow and slow variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_tau_mcdts.csv")))
    ts_pec = Int.(vec(readdlm("./slow and slow variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_ts_pec.csv")))
    ts_mcdts = Int.(vec(readdlm("./slow and slow variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_ts_mcdts.csv")))
    optimal_d_pec = Int.(vec(readdlm("./slow and slow variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_optimal_d_pec.csv")))
    optimal_d_mcdts = Int.(vec(readdlm("./slow and slow variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_optimal_d_mcdts.csv")))
    L_pec = vec(readdlm("./slow and slow variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_L_pec.csv"))
    L_mcdts = vec(readdlm("./slow and slow variable/results_Lorenz96_L2_N$(N)_M$(M)_2d_L_mcdts.csv"))
end

## Compute time series

const N = 15
const M = 5

function twolevel_l96!(du,u,p,t)
    F,h,c,b = p
    X = @view u[:,1]
    Y = @view u[:,2:end]
    hcb = h*c/b

    # boundary conditions solved by circular shift
    du[:,1] .= (-1 .*(circshift(X,-1) .- circshift(X,2)).*circshift(X,1) .- X .+ F .- hcb*sum(Y,dims=2))[:]

    # boundary conditions are complicaded for the fast variable....
    du[1,2] = -c*b*(Y[1,2] - Y[N,M-1])*Y[N,M] - c*Y[1,1] + hcb*X[1]
    du[1,3] = -c*b*(Y[1,3] - Y[N,M])*Y[1,1] - c*Y[1,2] + hcb*X[1]
    for j=3:M-1
        du[1,j+1] = -c*b*(Y[1,j+1] - Y[1,j-2])*Y[1,j-1] - c*Y[1,j] + hcb*X[1]
    end
    du[1,M+1] = -c*b*(Y[2,1] - Y[1,M-2])*Y[1,M-1] - c*Y[1,M] + hcb*X[1]

    for i=2:N-1
        du[i,2] = -c*b*(Y[i,2] - Y[i-1,M-1])*Y[i-1,M] - c*Y[i,1] + hcb*X[i]
        du[i,3] = -c*b*(Y[i,3] - Y[i-1,M])*Y[i,1] - c*Y[i,2] + hcb*X[i]
        for j=3:M-1
            du[i,j+1] = -c*b*(Y[i,j+1] - Y[i,j-2])*Y[i,j-1] - c*Y[i,j] + hcb*X[i]
        end
        du[i,M+1] = -c*b*(Y[i+1,1] - Y[i,M-2])*Y[i,M-1] - c*Y[i,M] + hcb*X[i]
    end

    du[N,2] = -c*b*(Y[N,2] - Y[N-1,M-1])*Y[N-1,M] - c*Y[N,1] + hcb*X[N]
    du[N,3] = -c*b*(Y[N,3] - Y[N-1,M])*Y[N,1] - c*Y[N,2] + hcb*X[N]
    for j=3:M-1
        du[N,j+1] = -c*b*(Y[N,j+1] - Y[N,j-2])*Y[N,j-1] - c*Y[N,j] + hcb*X[N]
    end
    du[N,M+1] = -c*b*(Y[1,1] - Y[N,M-2])*Y[N,M-1] - c*Y[N,M] + hcb*X[N]

end

begin
    F = 10. # forcing, 10 -> chaotic
    h = 2. # coupling
    c = 10. # time scale separation
    b = 10. # amplitude scale seperation

    pars = [F,h,c,b]
    Random.seed!(1234)
    u0 = rand(N,M+1) .- 0.5
end

t_end = 500.
prob = ODEProblem(twolevel_l96!, u0, (0.,t_end), pars)
sol = solve(prob)

data = Dataset(sol.u)
trajectory = Dataset(data[5001:20000])

data_sample = trajectory[:,ts_sample]
taus = 0:200
dmax = 20


## Time Delay Embedding seperately for fast and slow system input
Y_tde1_s, τ_tde1_s, _ = optimal_traditional_de(data_sample[:,1], "afnn"; dmax = dmax)
Y_tde1_f, τ_tde1_f, _ = optimal_traditional_de(data_sample[:,2], "afnn"; dmax = dmax)
Y_tde2_s, τ_tde2_s, _ = optimal_traditional_de(data_sample[:,1], "fnn"; dmax = dmax)
Y_tde2_f, τ_tde2_f, _ = optimal_traditional_de(data_sample[:,2], "fnn"; dmax = dmax)
Y_tde3_s, τ_tde3_s, _ = optimal_traditional_de(data_sample[:,1], "ifnn"; dmax = dmax)
Y_tde3_f, τ_tde3_f, _ = optimal_traditional_de(data_sample[:,2], "ifnn"; dmax = dmax)

optimal_d_tde1_s = size(Y_tde1_s,2)
optimal_d_tde2_s = size(Y_tde2_s,2)
optimal_d_tde3_s = size(Y_tde3_s,2)
optimal_d_tde1_f = size(Y_tde1_f,2)
optimal_d_tde2_f = size(Y_tde2_f,2)
optimal_d_tde3_f = size(Y_tde3_f,2)

# compute corresponding L's
τ_tdes1_s = [(i-1)*τ_tde1_s for i = 1:optimal_d_tde1_s]
L_tde1_s = MCDTS.compute_delta_L(data_sample, τ_tdes1_s, taus[end]; w = τ_tde1_s, tws = 2:2:taus[end])
τ_tdes2_s = [(i-1)*τ_tde2_s for i = 1:optimal_d_tde2_s]
L_tde2_s = MCDTS.compute_delta_L(data_sample, τ_tdes2_s, taus[end]; w = τ_tde2_s, tws = 2:2:taus[end])
τ_tdes3_s = [(i-1)*τ_tde3_s for i = 1:optimal_d_tde3_s]
L_tde3_s = MCDTS.compute_delta_L(data_sample, τ_tdes3_s, taus[end]; w = τ_tde3_s, tws = 2:2:taus[end])

τ_tdes1_f = [(i-1)*τ_tde1_f for i = 1:optimal_d_tde1_f]
L_tde1_f = MCDTS.compute_delta_L(data_fample, τ_tdes1_f, taus[end]; w = τ_tde1_f, tws = 2:2:taus[end])
τ_tdes2_f = [(i-1)*τ_tde2_f for i = 1:optimal_d_tde2_f]
L_tde2_f = MCDTS.compute_delta_L(data_fample, τ_tdes2_f, taus[end]; w = τ_tde2_f, tws = 2:2:taus[end])
τ_tdes3_f = [(i-1)*τ_tde3_f for i = 1:optimal_d_tde3_f]
L_tde3_f = MCDTS.compute_delta_L(data_fample, τ_tdes3_f, taus[end]; w = τ_tde3_f, tws = 2:2:taus[end])

# Reconstructions from PECUZAL and MCDTS
Y_pec = DelayEmbeddings.genembed(data_sample,tau_pec,ts_pec)
Y_mcdts = DelayEmbeddings.genembed(data_sample,tau_mcdts,ts_mcdts)

## compute RPs

ϵ = 0.08

# Reference
RP_ref = RecurrenceMatrix(trajectory, ϵ; fixedrate=true)
Rg_ref = grayscale(RP_ref)

# TDEs
RP_tde1_s = RecurrenceMatrix(Y_tde1_s, ϵ; fixedrate = true)
Rg_tde1_s = grayscale(RP_tde1_s)
RP_tde2_s = RecurrenceMatrix(Y_tde2_s, ϵ; fixedrate = true)
Rg_tde2_s = grayscale(RP_tde2_s)
RP_tde3_s = RecurrenceMatrix(Y_tde3_s, ϵ; fixedrate = true)
Rg_tde3_s = grayscale(RP_tde3_s)

RP_tde1_f = RecurrenceMatrix(Y_tde1_f, ϵ; fixedrate = true)
Rg_tde1_f = grayscale(RP_tde1_f)
RP_tde2_f = RecurrenceMatrix(Y_tde2_f, ϵ; fixedrate = true)
Rg_tde2_f = grayscale(RP_tde2_f)
RP_tde3_f = RecurrenceMatrix(Y_tde3_f, ϵ; fixedrate = true)
Rg_tde3_f = grayscale(RP_tde3_f)
# PECUZAL
RP_pec = RecurrenceMatrix(Y_pec, ϵ; fixedrate = true)
Rg_pec = grayscale(RP_pec)
# MCDTS
RP_mcdts = RecurrenceMatrix(Y_mcdts, ϵ; fixedrate = true)
Rg_mcdts = grayscale(RP_mcdts)

## compute similarity to reference RP
R_frac_tde1_s = MCDTS.jrp_rr_frac(RecurrenceMatrix(trajectory[1:length(Y_tde1_s),:], ϵ; fixedrate = true), RP_tde1_s)
R_frac_tde2_s = MCDTS.jrp_rr_frac(RecurrenceMatrix(trajectory[1:length(Y_tde2_s),:], ϵ; fixedrate = true), RP_tde2_s)
R_frac_tde3_s = MCDTS.jrp_rr_frac(RecurrenceMatrix(trajectory[1:length(Y_tde3_s),:], ϵ; fixedrate = true), RP_tde3_s)

R_frac_tde1_f = MCDTS.jrp_rr_frac(RecurrenceMatrix(trajectory[1:length(Y_tde1_f),:], ϵ; fixedrate = true), RP_tde1_f)
R_frac_tde2_f = MCDTS.jrp_rr_frac(RecurrenceMatrix(trajectory[1:length(Y_tde2_f),:], ϵ; fixedrate = true), RP_tde2_f)
R_frac_tde3_f = MCDTS.jrp_rr_frac(RecurrenceMatrix(trajectory[1:length(Y_tde3_f),:], ϵ; fixedrate = true), RP_tde3_f)

R_frac_pec = MCDTS.jrp_rr_frac(RecurrenceMatrix(trajectory[1:length(Y_pec),:], ϵ; fixedrate = true), RP_pec)
R_frac_mcdts = MCDTS.jrp_rr_frac(RecurrenceMatrix(trajectory[1:length(Y_mcdts),:], ϵ; fixedrate = true), RP_mcdts)

## Plot results

figure(figsize=(20,10))
subplot(231)
imshow(Rg_ref, cmap = "binary_r", extent = (1, size(RP_ref)[1], 1, size(RP_ref)[2]))
title("Reference")
grid()

subplot(232)
imshow(Rg_tde1_s, cmap = "binary_r", extent = (1, size(RP_tde1_s)[1], 1, size(RP_tde1_s)[2]))
title("TDE SLOW CAO ($(R_frac_tde1*100)% acc.)")
grid()

subplot(233)
imshow(Rg_tde2_s, cmap = "binary_r", extent = (1, size(RP_tde2_s)[1], 1, size(RP_tde2_s)[2]))
title("TDE SLOW Kennel ($(R_frac_tde2*100)% acc.)")
grid()

subplot(234)
imshow(Rg_tde3_s, cmap = "binary_r", extent = (1, size(RP_tde3_s)[1], 1, size(RP_tde3_s)[2]))
title("TDE SLOW Hegger ($(R_frac_tde3*100)% acc.)")
grid()

subplot(235)
imshow(Rg_pec, cmap = "binary_r", extent = (1, size(RP_pec)[1], 1, size(RP_pec)[2]))
title("PECUZAL ($(R_frac_pec*100)% acc.)")
grid()

subplot(236)
imshow(Rg_mcdts, cmap = "binary_r", extent = (1, size(RP_mcdts)[1], 1, size(RP_mcdts)[2]))
title("MCDTS ($(R_frac_mcdts*100)% acc.)")
grid()


##

figure(figsize=(20,10))
subplot(231)
imshow(Rg_ref, cmap = "binary_r", extent = (1, size(RP_ref)[1], 1, size(RP_ref)[2]))
title("Reference")
grid()

subplot(232)
imshow(Rg_tde1_f, cmap = "binary_r", extent = (1, size(RP_tde1_f)[1], 1, size(RP_tde1_f)[2]))
title("TDE FAST CAO ($(R_frac_tde1*100)% acc.)")
grid()

subplot(233)
imshow(Rg_tde2_f, cmap = "binary_r", extent = (1, size(RP_tde2_f)[1], 1, size(RP_tde2_f)[2]))
title("TDE FAST Kennel ($(R_frac_tde2*100)% acc.)")
grid()

subplot(234)
imshow(Rg_tde3_f, cmap = "binary_r", extent = (1, size(RP_tde3_f)[1], 1, size(RP_tde3_f)[2]))
title("TDE FAST Hegger ($(R_frac_tde3*100)% acc.)")
grid()

subplot(235)
imshow(Rg_pec, cmap = "binary_r", extent = (1, size(RP_pec)[1], 1, size(RP_pec)[2]))
title("PECUZAL ($(R_frac_pec*100)% acc.)")
grid()

subplot(236)
imshow(Rg_mcdts, cmap = "binary_r", extent = (1, size(RP_mcdts)[1], 1, size(RP_mcdts)[2]))
title("MCDTS ($(R_frac_mcdts*100)% acc.)")
grid()
