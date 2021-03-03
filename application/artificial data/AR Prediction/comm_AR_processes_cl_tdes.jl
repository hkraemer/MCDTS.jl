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
    τ_cao = []
    τ_kennel = []
    τ_hegger = []

    X = zeros(N,length(ar_coeffs))

    for j in eachindex(ar_coeffs)
        T_steps = 100
        coeffs = ar_coeffs[i]
        X[:,j] = arfima(N, σ, nothing, SVector(coeffs...))

        w1 = DelayEmbeddings.estimate_delay(X[:,j], "mi_min")
        dmax = 15

        Y_cao, τ_tde_cao_, _ = optimal_traditional_de(X[1:end-T_steps,j], "afnn", "mi_min"; dmax = dmax, w = w1)
        D[j,1] = size(Y_cao,2)
        τ_caos = [(i-1)*τ_tde_cao_ for i = 1:D[j,1]]
        push!(τ_cao, τ_caos)

        Y_kennel, τ_tde_kennel_, _ = optimal_traditional_de(X[1:end-T_steps,j], "fnn", "mi_min"; dmax = dmax, w = w1)
        D[j,2] = size(Y_kennel,2)
        τ_kennels = [(i-1)*τ_tde_kennel_ for i = 1:D[j,2]]
        push!(τ_kennel, τ_kennels)

        Y_hegger, τ_tde_hegger_, _ = optimal_traditional_de(X[1:end-T_steps,j], "ifnn", "mi_min"; dmax = dmax, w = w1)
        D[j,3] = size(Y_hegger,2)
        τ_heggers = [(i-1)*τ_tde_hegger_ for i = 1:D[j,3]]
        push!(τ_hegger, τ_heggers)

    end

    # Output
    tuple(X, D, τ_cao, τ_kennel, τ_hegger)

end

end

varnames = ["ts_tdes", "dimension_tdes", "taus_cao", "taus_kennel", "taus_hegger"]

for i = 1:length(varnames)
    writestr = "N_$(N)_AR$(i+2)_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
