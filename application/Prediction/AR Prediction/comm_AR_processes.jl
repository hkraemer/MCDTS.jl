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

##
# array of arrays containing the ar-coefficients for the different orders
ar_coeff = []
push!(ar_coeff, [0.53, 0.15, 0.3])
push!(ar_coeff, [0.4, 0, 0.3, 0.1])
push!(ar_coeff, [0.4, 0, 0.15, 0.1, 0.25])
push!(ar_coeff, [0.3, 0.1, 0, 0.2, 0.25, 0.1])

σ = 0.4 # std of Gaussian numbers

Random.seed(1234)
for N = 10000:10000:60000
    global ar_coeff
    global σ
    X = zeros(N, length(ar_coeff))
    for i = 1:length(ar_coeff)
        T_step = 100
        coeffs = ar_coeff[i]
        X[:,i] = arfima(N, σ, nothing, SVector(coeffs...))

        w1 = DelayEmbeddings.estimate_delay(X, "ac_zero")
        dmax = 15
        D = zeros(Int,5)

        Y_cao, τ_tde_cao_, _ = optimal_traditional_de(X[1:end-T_steps,i], "afnn", "ac_zero"; dmax = dmax)
        D[1] = size(Y_cao,2)
        τ_cao = [(i-1)*τ_tde_cao_ for i = 1:D[1]]
        Y_cao = MCDTS.genembed_for_prediction(X[1:end-T_steps,i], τ_cao)

        Y_kennel, τ_tde_kennel_, _ = optimal_traditional_de(X[1:end-T_steps,i], "fnn", "ac_zero"; dmax = dmax)
        D[2] = size(Y_kennel,2)
        τ_kennel = [(i-1)*τ_tde_kennel_ for i = 1:D[2]]
        Y_kennel = MCDTS.genembed_for_prediction(X[1:end-T_steps,i], τ_kennel)

        Y_hegger, τ_tde_hegger_, _ = optimal_traditional_de(X[1:end-T_steps,i], "afnn", "ac_zero"; dmax = dmax)
        D[3] = size(Y_hegger,2)
        τ_hegger = [(i-1)*τ_tde_hegger_ for i = 1:D[3]]
        Y_hegger = MCDTS.genembed_for_prediction(X[1:end-T_steps,i], τ_hegger)

        # Pecuzal
        taus = 0:100
        _, τ_pec, ts_pec, Ls_pec , _ = DelayEmbeddings.pecuzal_embedding(X[1:end-T_steps,i]; τs = taus , w = w1)
        Y_pec = MCDTS.genembed_for_prediction(X[1:end-T_steps,i], τ_pec)
        D[4] = length(τ_pec)

        # MCDTS
        trials = 80
        tree = MCDTS.mc_delay(Dataset(X[1:end-T_steps,i]), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus, trials; tws = 1:taus[end], max_depth = 15)
        best_node = MCDTS.best_embedding(tree)
        τ_mcdts = best_node.τs
        Y_mcdts = MCDTS.genembed_for_prediction(X[1:end-T_steps], τ_mcdts)
        D[5] = size(Y_mcdts,2)

        # save variables
        writedlm("N_$(N)_AR$(i+2)_ts.csv", X[:,i])
        writedlm("N_$(N)_AR$(i+2)_dimension.csv", D)
        writedlm("N_$(N)_AR$(i+2)_taus_cao.csv", τ_cao)
        writedlm("N_$(N)_AR$(i+2)_taus_kennel.csv", τ_kennel)
        writedlm("N_$(N)_AR$(i+2)_taus_hegger.csv", τ_hegger)
        writedlm("N_$(N)_AR$(i+2)_taus_pec.csv", τ_pec)
        writedlm("N_$(N)_AR$(i+2)_taus_mcdts.csv", τ_mcdts)
        writedml("N_$(N)_AR$(i+2)_true_coeffs.csv", coeffs)
    end
end
