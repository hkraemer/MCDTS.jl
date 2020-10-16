using Pkg
current_dir = pwd()
Pkg.activate(current_dir)

using DynamicalSystems
using MCDTS

## We apply the MCDTS-approach to the Lorenz system and compare the results of
# the reconstruction to the ones gained from PECUZAL and standard TDE.

# generate data
ds = Systems.lorenz()
data = trajectory(ds,200; dt = 0.01, Ttr = 40)

# set parameters
softmax_beta = 2.       # β-value for softmax function
τs = 0:200              # possible delay values
Trials = 50             # Trials for MCDTS
rec_threshold = 0.08    # recurrence threshold for RQA etc.
σ = 0.01                 # noise level
Ns = 5000:5000:5000  # encountered time series lengths

# preallocation
τs_mcdts = []
ts_mcdts = []
τs_mcdts_multi = []
ts_mcdts_multi = []

τs_pec = []
ts_pec = []
τs_pec_multi = []
ts_pec_multi = []

τs_tde = []
ts_tde = []

dims = zeros(5,length(Ns))
L = zeros(5,length(Ns))
mfnn = zeros(5,length(Ns))
jrrf = zeros(5,length(Ns))
RTE = zeros(6,length(Ns))
ENTR = zeros(6,length(Ns))
LAM = zeros(6,length(Ns))
TRANS = zeros(6,length(Ns))

for (i,N) in enumerate(Ns)

    println("Run no: $i")

    tr = data[1:N,:]
    tr = regularize(tr)
    tr = Dataset(Matrix(tr) .+ σ.*randn(size(tr,1),size(tr,2)))
    w1 = estimate_delay(tr[:,1],"mi_min")
    w2 = estimate_delay(tr[:,2],"mi_min")
    w3 = estimate_delay(tr[:,3],"mi_min")
    w = maximum(hcat(w1,w2,w3))

    # MCDTS
    tree = MCDTS.mc_delay(Dataset(tr[:,2]),w,(L)->(MCDTS.softmaxL(L,β=softmax_beta)),τs,Trials)
    best_node = MCDTS.best_embedding(tree)
    push!(τs_mcdts, best_node.τs)
    push!(ts_mcdts, best_node.ts)
    Y_mcdts = genembed(tr, Tuple(best_node.τs), Tuple(best_node.ts))
    L[1,i] = best_node.L
    dims[1,i] = size(Y_mcdts,2)
    # multivariate
    tree_m = MCDTS.mc_delay(tr,w,(L)->(MCDTS.softmaxL(L,β=softmax_beta)),τs,Trials)
    best_node_m = MCDTS.best_embedding(tree_m)
    push!(τs_mcdts_multi, best_node_m.τs)
    push!(ts_mcdts_multi, best_node_m.ts)
    Y_mcdts_m = genembed(tr, Tuple(best_node_m.τs), Tuple(best_node_m.ts))
    L[4,i] = best_node_m.L
    dims[4,i] = size(Y_mcdts_m,2)


    # PECUZAL
    Y_pec, τspec, tspec, Ls_pec, _ = MCDTS.pecuzal_embedding(tr[:,2]; τs = τs, w = w)
    push!(τs_pec, τspec)
    push!(ts_pec, tspec)
    L[2,i] = minimum(Ls_pec)
    dims[2,i] = size(Y_pec,2)
    # multivariate
    Y_pec_m, τspec_m, tspec_m, Ls_pec_m, _ = MCDTS.pecuzal_embedding(tr; τs = τs, w = w)
    push!(τs_pec_multi, τspec_m)
    push!(ts_pec_multi, tspec_m)
    L[5,i] = minimum(Ls_pec_m)
    dims[5,i] = size(Y_pec_m,2)

    # TDE
    Y_tde, tau_TDE, _ = MCDTS.standard_embedding_cao(tr[:,2])
    L[3,i] = uzal_cost(Y_tde, Tw = (4*w), w = w, samplesize=1)
    push!(τs_tde, [(i-1)*tau_TDE for i = 1:size(Y_tde,2)])
    push!(ts_tde, fill(1, size(Y_tde,2)))
    dims[3,i] = size(Y_tde,2)

    # Peform analysis on the reconstructions
    mfnn[1,i], mfnn[2,i], mfnn[3,i], jrrf[1,i], jrrf[2,i],
    jrrf[3,i], RQA_ref, RQA1, RQA2, RQA3, _, _, _, _ =
        MCDTS.perform_recurrence_analysis(tr, Dataset(Y_mcdts), Dataset(Y_pec),
                    Dataset(Y_tde); ε = rec_threshold, w = w, kNN = 1)
    # multivariate
    mfnn[4,i], mfnn[5,i], _, jrrf[4,i], jrrf[5,i],
    _, _, RQA1m, RQA2m, _, _, _, _, _ =
        MCDTS.perform_recurrence_analysis(tr, Dataset(Y_mcdts_m), Dataset(Y_pec_m),
                    Dataset(Y_tde); ε = rec_threshold, w = w, kNN = 1)

    RTE[1,i] = RQA_ref.RTE
    ENTR[1,i] = RQA_ref.ENTR
    LAM[1,i] = RQA_ref.LAM
    TRANS[1,i] = RQA_ref.TRANS

    RTE[2,i] = RQA1.RTE
    ENTR[2,i] = RQA1.ENTR
    LAM[2,i] = RQA1.LAM
    TRANS[2,i] = RQA1.TRANS

    RTE[3,i] = RQA2.RTE
    ENTR[3,i] = RQA2.ENTR
    LAM[3,i] = RQA2.LAM
    TRANS[3,i] = RQA2.TRANS

    RTE[4,i] = RQA3.RTE
    ENTR[4,i] = RQA3.ENTR
    LAM[4,i] = RQA3.LAM
    TRANS[4,i] = RQA3.TRANS

    RTE[5,i] = RQA1m.RTE
    ENTR[5,i] = RQA1m.ENTR
    LAM[5,i] = RQA1m.LAM
    TRANS[5,i] = RQA1m.TRANS

    RTE[6,i] = RQA2m.RTE
    ENTR[6,i] = RQA2m.ENTR
    LAM[6,i] = RQA2m.LAM
    TRANS[6,i] = RQA2m.TRANS
end

println("Results for σ=$σ on ts lengths N=$Ns")

println("*******")
println("L_tde: $(L[3,:])")
println("L_mcdts uni: $(L[1,:])")
println("L_pec uni: $(L[2,:])")
println("L_mcdts multi: $(L[4,:])")
println("L_pec multi: $(L[5,:])")

println("*******")

println("mfnn_tde: $(mfnn[3,:])")
println("mfnn_mcdts uni: $(mfnn[1,:])")
println("mfnn_pec uni: $(mfnn[2,:])")
println("mfnn_mcdts multi: $(mfnn[4,:])")
println("mfnn_pec multi: $(mfnn[5,:])")

println("*******")

println("jrrf_tde: $(jrrf[3,:])")
println("jrrf_mcdts uni: $(jrrf[1,:])")
println("jrrf_pec uni: $(jrrf[2,:])")
println("jrrf_mcdts multi: $(jrrf[4,:])")
println("jrrf_pec multi: $(jrrf[5,:])")

println("*******")

println("LAM_tde: $(abs.(LAM[4,:].-LAM[1,:]))")
println("LAM_mcdts uni: $(abs.(LAM[2,:].-LAM[1,:]))")
println("LAM_pec uni: $(abs.(LAM[3,:].-LAM[1,:]))")
println("LAM_mcdts multi: $(abs.(LAM[5,:].-LAM[1,:]))")
println("LAM_pec multi: $(abs.(LAM[6,:].-LAM[1,:]))")

println("*******")

println("RTE_tde: $(abs.(RTE[4,:].-RTE[1,:]))")
println("RTE_mcdts uni: $(abs.(RTE[2,:].-RTE[1,:]))")
println("RTE_pec uni: $(abs.(RTE[3,:].-RTE[1,:]))")
println("RTE_mcdts multi: $(abs.(RTE[5,:].-RTE[1,:]))")
println("RTE_pec multi: $(abs.(RTE[6,:].-RTE[1,:]))")

println("*******")

println("ENTR_tde: $(abs.(ENTR[4,:].-ENTR[1,:]))")
println("ENTR_mcdts uni: $(abs.(ENTR[2,:].-ENTR[1,:]))")
println("ENTR_pec uni: $(abs.(ENTR[3,:].-ENTR[1,:]))")
println("ENTR_mcdts multi: $(abs.(ENTR[5,:].-ENTR[1,:]))")
println("ENTR_pec multi: $(abs.(ENTR[6,:].-ENTR[1,:]))")

println("*******")

println("Transitivity_tde: $(abs.(TRANS[4,:].-TRANS[1,:]))")
println("Transitivity_mcdts uni: $(abs.(TRANS[2,:].-TRANS[1,:]))")
println("Transitivity_pec uni: $(abs.(TRANS[3,:].-TRANS[1,:]))")
println("Transitivity_mcdts multi: $(abs.(TRANS[5,:].-TRANS[1,:]))")
println("Transitivity_pec multi: $(abs.(TRANS[6,:].-TRANS[1,:]))")

println("*******")
