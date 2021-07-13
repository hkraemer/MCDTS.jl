# Cluster script for computing the CCM for different time series length and
# different embeddings. Here the second sample

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
    using DelayEmbeddings
    using Statistics
    using Random
    using StatsBase
    using LinearAlgebra
    using DelimitedFiles

    ## Combustion data
    data1 = readdlm("pressure_downsampled_same_sampling.txt")
    data2 = readdlm("heat_release_downsampled_same_sampling.txt")

    ## Generate subset
    Random.seed!(111)
    N = 5000
    s = rand(1:length(data1)-N)
    s1 = data1[s:s+N]
    s2 = data2[s:s+N]
    s1 = s1 .+ 0.0000000001.*randn(length(s1))
    s2 = s2 .+ 0.0000000001.*randn(length(s2))

    s1 = (s1 .- mean(s1)) ./ std(s1)
    s2 = (s2 .- mean(s2)) ./ std(s2)

    # Parameters analysis:
    τs = 0:60
    trials = 50

    # starts and step
    starts = [i for i in 100:100:N]

end

@time begin
# loop over different F's
results = @distributed (hcat) for i in eachindex(starts)

    lim = starts[i]
    # bind time series window
    xx = s1[1:lim]
    yy = s2[1:lim]
    # normalization
    xx = (xx .- mean(xx)) ./ std(xx)
    yy = (yy .- mean(yy)) ./ std(yy)

    # standard Pearson
    ρp = Statistics.cor(xx,yy)

    if i == 100
        w = 5
    else
        w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
        w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")
        w = maximum([w1,w2])
    end

    # embedding
    #classic
    Y, delay, _ = optimal_traditional_de(xx, "afnn"; w = w1)
    taus_cao1 = [i*delay for i = 0:size(Y,2)-1]
    Y, delay, _ = optimal_traditional_de(yy, "afnn"; w = w2)
    taus_cao2 = [i*delay for i = 0:size(Y,2)-1]

    # pecuzal
    _, taus_pec1,_,_,_ = pecuzal_embedding(xx; τs = τs, w = w1, econ = true)
    _, taus_pec2,_,_,_ = pecuzal_embedding(yy; τs = τs, w = w2, KNN=2, econ = true)

    # mcdts
    Random.seed!(1234)
    tree = MCDTS.mc_delay(Dataset(xx), w, (L)->(MCDTS.softmaxL(L,β=2.)), τs, trials;
        verbose=false, CCM = true, Y_CCM = yy)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts1 = best_node.τs
    L = best_node.L

    tree = MCDTS.mc_delay(Dataset(yy), w, (L)->(MCDTS.softmaxL(L,β=2.)), τs, trials;
        verbose=false, CCM = true, Y_CCM = xx)
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts2 = best_node.τs
    L = best_node.L


    Yx_cao = genembed(xx,-taus_cao1)
    Yy_cao = genembed(yy,-taus_cao1)

    Yx_cao2 = genembed(xx,-taus_cao2)
    Yy_cao2 = genembed(yy,-taus_cao2)

    Yx_pec = genembed(xx,-taus_pec1)
    Yy_pec = genembed(yy,-taus_pec1)

    Yx_pec2 = genembed(xx,-taus_pec2)
    Yy_pec2 = genembed(yy,-taus_pec2)

    Yx_mcdts = genembed(xx,-τ_mcdts1)
    Yy_mcdts = genembed(yy,-τ_mcdts1)

    Yx_mcdts2 = genembed(xx,-τ_mcdts2)
    Yy_mcdts2 = genembed(yy,-τ_mcdts2)

    # compute CCM
    rho_y1_cao, _ = MCDTS.ccm(Yx_cao, Yy_cao; w = w)
    rho_x1_cao, _ = MCDTS.ccm(Yy_cao, Yx_cao; w = w)

    rho_y2_cao, _ = MCDTS.ccm(Yx_cao2, Yy_cao2; w = w)
    rho_x2_cao, _ = MCDTS.ccm(Yy_cao2, Yx_cao2; w = w)

    rho_y1_pec, _ = MCDTS.ccm(Yx_pec, Yy_pec; w = w)
    rho_x1_pec, _ = MCDTS.ccm(Yy_pec, Yx_pec; w = w)

    rho_y2_pec, _ = MCDTS.ccm(Yx_pec2, Yy_pec2; w = w)
    rho_x2_pec, _ = MCDTS.ccm(Yy_pec2, Yx_pec2; w = w)

    rho_y1_mcdts, _ = MCDTS.ccm(Yx_mcdts, Yy_mcdts; w = w)
    rho_x1_mcdts, _ = MCDTS.ccm(Yy_mcdts, Yx_mcdts; w = w)

    rho_y2_mcdts, _ = MCDTS.ccm(Yx_mcdts2, Yy_mcdts2; w = w)
    rho_x2_mcdts, _ = MCDTS.ccm(Yy_mcdts2, Yx_mcdts2; w = w)

    # Output
    tuple(rho_y1_cao, rho_x1_cao, rho_y1_pec, rho_x1_pec, rho_y1_mcdts, rho_x1_mcdts,
    rho_y2_cao, rho_x2_cao, rho_y2_pec, rho_x2_pec, rho_y2_mcdts, rho_x2_mcdts, ρp)

end

end


varnames = ["y1_cao", "x1_cao", "y1_pec", "x1_pec", "y1_mecdts", "x1_mcdts",
 "y2_cao", "x2_cao", "y2_pec", "x2_pec", "y2_mecdts", "x2_mcdts", "Pearson"]

for i = 1:length(varnames)
    writestr = "results_analysis_CCM_combustion_2_"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
