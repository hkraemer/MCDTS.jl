# two-level L96 model
# each of the N slow variables is coupled to M fast variables,
# so in total the model is N + M*N dimensional
# parameter values directly from Lorenz96 paper

using DifferentialEquations
using RecurrenceAnalysis
using LinearAlgebra
using StatsBase
using DelayEmbeddings

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
    h = 1. # coupling
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
trajectory = Dataset(data[5001:15000])


ϵ = 0.08
dmax = 20

RP_ref = RecurrenceMatrix(trajectory, ϵ; fixedrate=true)
Rg_ref = grayscale(RP_ref)

using PyPlot
pygui(true)


# 1D reconstruction
Y_tde, τ_tde1, _ = optimal_traditional_de(trajectory[:,1], "afnn"; dmax = dmax)
RP_tde = RecurrenceMatrix(Y_tde, ϵ; fixedrate = true)
Rg_tde = grayscale(RP_tde)

theiler = DelayEmbeddings.estimate_delay(trajectory[:,1], "mi_min")
Y_pec, τ_pec, ts_pec, Ls_pec , _ = DelayEmbeddings.pecuzal_embedding(trajectory; τs = 0:200 , w = theiler, econ = true)
RP_pec = RecurrenceMatrix(Y_pec, ϵ; fixedrate = true)
Rg_pec = grayscale(RP_pec)

tree = MCDTS.mc_delay(Dataset(trajectory[:,1]), theiler, (L)->(MCDTS.softmaxL(L,β=2.)), 0:200, 50; tws = 2:2:taus[end], threshold = 0, max_depth = 25)
best_node = MCDTS.best_embedding(tree)
Y_mcdts = genembed(data_sample, best_node.τs, best_node.ts)
RP_mcdts = RecurrenceMatrix(Y_mcdts, ϵ; fixedrate = true)
Rg_mcdts = grayscale(RP_mcdts)

figure()
subplot(121)
imshow(Rg_ref, cmap = "binary_r", extent = (1, size(RP_ref)[1], 1, size(RP_ref)[2]))
title("Reference")
grid()
subplot(121)
imshow(Rg_tde, cmap = "binary_r", extent = (1, size(RP_tde)[1], 1, size(RP_tde)[2]))
title("TDE")
grid()

figure()
subplot(121)
imshow(Rg_ref, cmap = "binary_r", extent = (1, size(RP_ref)[1], 1, size(RP_ref)[2]))
title("Reference")
grid()
subplot(121)
imshow(Rg_pec, cmap = "binary_r", extent = (1, size(RP_pec)[1], 1, size(RP_pec)[2]))
title("PECUZAL")
grid()

figure()
subplot(121)
imshow(Rg_ref, cmap = "binary_r", extent = (1, size(RP_ref)[1], 1, size(RP_ref)[2]))
title("Reference")
grid()
subplot(121)
imshow(Rg_mcdts, cmap = "binary_r", extent = (1, size(RP_mcdts)[1], 1, size(RP_mcdts)[2]))
title("MCDTS")
grid()
