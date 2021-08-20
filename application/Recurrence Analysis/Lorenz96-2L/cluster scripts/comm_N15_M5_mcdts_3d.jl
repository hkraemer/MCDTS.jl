using DifferentialEquations
using LinearAlgebra
using StatsBase
using DelayEmbeddings
using DelimitedFiles
using Random
using MCDTS

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

# integration and data-binding
t_end = 500.
prob = ODEProblem(twolevel_l96!, u0, (0.,t_end), pars)
sol = solve(prob)

data = Dataset(sol.u)
trajectory = Dataset(data[5001:20000])

# take a slow variable for reconstruction
ts_sample = [3,8,10]

# params for reconstruction
taus = 0:200
trials = 80

data_sample = trajectory[:,ts_sample]
w1 = DelayEmbeddings.estimate_delay(data_sample[:,1], "mi_min")
w2 = DelayEmbeddings.estimate_delay(data_sample[:,2], "mi_min")
w3 = DelayEmbeddings.estimate_delay(data_sample[:,3], "mi_min")
theiler = maximum([w1,w2,w3])
tree = MCDTS.mc_delay(Dataset(data_sample), theiler, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:2:taus[end], threshold = 0, max_depth = 90)
best_node = MCDTS.best_embedding(tree)
Y_mcdts = genembed(data_sample, best_node.τs, best_node.ts)
optimal_d_mcdts = size(Y_mcdts,2)

writestr = "./slow 3d/results_Lorenz96_L2_N$(N)_M$(M)_3d_tau_mcdts.csv"
writedlm(writestr, best_node.τs)
writestr = "./slow 3d/results_Lorenz96_L2_N$(N)_M$(M)_3d_ts_mcdts.csv"
writedlm(writestr, best_node.ts)
writestr = "./slow 3d/results_Lorenz96_L2_N$(N)_M$(M)_3d_optimal_d_mcdts.csv"
writedlm(writestr, optimal_d_mcdts)
writestr = "./slow 3d/results_Lorenz96_L2_N$(N)_M$(M)_3d_L_mcdts.csv"
writedlm(writestr, best_node.L)
