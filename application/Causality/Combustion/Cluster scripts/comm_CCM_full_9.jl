# Cluster script for computing the CCM for different time series length and
# different embeddings. Here the first sample

# One single embedding is computed for the full time series with N=5000

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
Random.seed!(121)
N = 5000
N_min = 500
step = 100
s = rand(1:length(data1)-N)
s1 = data1[s:s+N]
s2 = data2[s:s+N]
s1 = s1 .+ 0.0000000001.*randn(length(s1))
s2 = s2 .+ 0.0000000001.*randn(length(s2))

s1 = (s1 .- mean(s1)) ./ std(s1)
s2 = (s2 .- mean(s2)) ./ std(s2)

# Parameters analysis:
τs = 0:60
trials = 100


# bind time series window
xx = s1
yy = s2


# standard Pearson
ρp = Statistics.cor(xx,yy)


w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")
w = maximum([w1,w2])


# embedding
#classic
Y, delay, _ = optimal_traditional_de(xx, "afnn"; w = w1)
taus_cao1 = [j*delay for j = 0:size(Y,2)-1]
Y, delay, _ = optimal_traditional_de(yy, "afnn"; w = w2)
taus_cao2 = [j*delay for j = 0:size(Y,2)-1]

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

τs = 0:60
trials = 50

cnt = 0
rho_ccm = zeros(12,length(N_min:step:N))

for i = N_min:step:N

    global cnt += 1

    println(i)
    xx = s1[1:i]
    yy = s2[1:i]

    w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
    w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")
    w = maximum([w1,w2])


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
    rho_ccm[1,cnt], _ = MCDTS.ccm(Yx_cao, Yy_cao; w = w)
    rho_ccm[2,cnt], _ = MCDTS.ccm(Yy_cao, Yx_cao; w = w)

    rho_ccm[3,cnt], _ = MCDTS.ccm(Yx_cao2, Yy_cao2; w = w)
    rho_ccm[4,cnt], _ = MCDTS.ccm(Yy_cao2, Yx_cao2; w = w)

    rho_ccm[5,cnt], _ = MCDTS.ccm(Yx_pec, Yy_pec; w = w)
    rho_ccm[6,cnt], _ = MCDTS.ccm(Yy_pec, Yx_pec; w = w)

    rho_ccm[7,cnt], _ = MCDTS.ccm(Yx_pec2, Yy_pec2; w = w)
    rho_ccm[8,cnt], _ = MCDTS.ccm(Yy_pec2, Yx_pec2; w = w)

    rho_ccm[9,cnt], _ = MCDTS.ccm(Yx_mcdts, Yy_mcdts; w = w)
    rho_ccm[10,cnt], _ = MCDTS.ccm(Yy_mcdts, Yx_mcdts; w = w)

    rho_ccm[11,cnt], _ = MCDTS.ccm(Yx_mcdts2, Yy_mcdts2; w = w)
    rho_ccm[12,cnt], _ = MCDTS.ccm(Yy_mcdts2, Yx_mcdts2; w = w)

end


varnames = ["y1_cao", "x1_cao", "y1_pec", "x1_pec", "y1_mecdts", "x1_mcdts",
 "y2_cao", "x2_cao", "y2_pec", "x2_pec", "y2_mecdts", "x2_mcdts", "Pearson"]

for i = 1:length(varnames)
    writestr = "results_analysis_CCM_full_combustion_9_"*varnames[i]*".csv"
    if i == 13
        data = ρp
    else
        data = rho_ccm[i,:]
    end
    writedlm(writestr, data)
end
