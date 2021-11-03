using ClusterManagers
using Distributed
@everywhere N_tasks = parse(Int, ARGS[1])
@everywhere N_worker = N_tasks
addprocs(SlurmManager(N_worker))

@everywhere begin
    using ClusterManagers
    using Distributed
    using IterTools
    using MCDTS
    using ARFIMA
    using Random
    using DelayEmbeddings
    using DelimitedFiles

    ar_coeffs = []
    push!(ar_coeffs, [0.53, 0.15, 0.3])
    push!(ar_coeffs, [0.4, 0, 0.3, 0.1])
    push!(ar_coeffs, [0.4, 0, 0.15, 0.1, 0.25])
    push!(ar_coeffs, [0.3, 0.1, 0, 0.2, 0.25, 0.1])

    σ = 0.4 # std of Gaussian numbers

    Random.seed!(1234)
    Ns = 10000:10000:60000
end

@time begin

# loop over different N's
results = @distributed (vcat) for i in eachindex(Ns)
    global ar_coeff
    global σ
    N = Ns[i]
    D = zeros(Int,length(ar_coeffs),3)
    τ_pec = []
    τ_mcdts_L = []
    τ_mcdts_FNN = []
    X = zeros(N,length(ar_coeffs))

    for j in eachindex(ar_coeffs)
        T_steps = 100
        coeffs = ar_coeffs[i]
        X[:,j] = arfima(N, σ, nothing, SVector(coeffs...))

        w1 = DelayEmbeddings.estimate_delay(X[:,j], "mi_min")
        dmax = 15

        # Pecuzal
        taus = 0:100
        _, τ_pecs, ts_pec, Ls_pec , _ = DelayEmbeddings.pecuzal_embedding(X[1:end-T_steps,j]; τs = taus , w = w1)
        D[j,1] = length(τ_pecs)
        push!(τ_pec, τ_pecs)

        # MCDTS
        trials = 80
        tree = MCDTS.mc_delay(Dataset(X[1:end-T_steps,j]), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 2:taus[end])
        best_node = MCDTS.best_embedding(tree)
        τ_mcdts_Ls = best_node.τs
        D[j,2] = length(τ_mcdts_Ls)
        push!(τ_mcdts_L, τ_mcdts_Ls)

        tree2 = MCDTS.mc_delay(Dataset(X[1:end-T_steps,j]), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; FNN = true, threshold = 0.05)
        best_node2 = MCDTS.best_embedding(tree2)
        τ_mcdts_FNNs = best_node2.τs
        D[j,3] = length(τ_mcdts_FNNs)
        push!(τ_mcdts_FNN, τ_mcdts_FNNs)

    end

    # Output
    tuple(X, D, τ_pec, τ_mcdts_L, τ_mcdts_FNN)

end

end

varnames = ["ts", "dimension", "taus_pec", "taus_mcdts_L", "taus_mcdts_FNN"]

for i = 1:length(varnames)
    writestr = "N_$(N)_AR$(i+2)_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
