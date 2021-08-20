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
data1 = readdlm("pressure_downsampled.txt")
data2 = readdlm("heat_release_downsampled.txt")

## Generate subset

run = 6

runs = [121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
    151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170]

Random.seed!(runs[run])
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
τs = 0:80
trials = 100

# bind time series window
xx = s1
yy = s2

w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")
w = maximum([w1,w2])

# embedding
#classic
Y, delay, _ = optimal_traditional_de(xx, "afnn"; w = w1)
taus_cao1 = [j*delay for j = 0:size(Y,2)-1]
Y, delay, _ = optimal_traditional_de(yy, "afnn"; w = w2)
taus_cao2 = [j*delay for j = 0:size(Y,2)-1]

println("taus_Cao 1: $taus_cao1")
println("taus_Cao 2: $taus_cao2")

# pecuzal
_, taus_pec1,_,_,_ = pecuzal_embedding(xx; τs = τs, w = w1, econ = true)
_, taus_pec2,_,_,_ = pecuzal_embedding(yy; τs = τs, w = w2, econ = true)

println("taus_pec 1: $taus_pec1")
println("taus_pec 2: $taus_pec2")

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

println("taus_mcdts 1: $τ_mcdts1")
println("taus_mcdts 2: $τ_mcdts2")

cnt = 0
rho_ccm = zeros(6,length(N_min:step:N))
ρp = zeros(length(N_min:step:N))

for i = N_min:step:N

    global cnt += 1

    println(i)
    xx = s1[1:i]
    yy = s2[1:i]

    w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
    w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")
    w = maximum([w1,w2])

    # standard Pearson
    ρp[cnt] = Statistics.cor(xx,yy)

    Yx_cao = genembed(xx,-taus_cao1)
    Yy_cao = genembed(yy,-taus_cao2)

    Yx_pec = genembed(xx,-taus_pec1)
    Yy_pec = genembed(yy,-taus_pec2)

    Yx_mcdts = genembed(xx,-τ_mcdts1)
    Yy_mcdts = genembed(yy,-τ_mcdts2)

    # compute CCM
    rho_ccm[1,cnt], _ = MCDTS.ccm(Yx_cao, yy[1+maximum(taus_cao1):length(Yx_cao)+maximum(taus_cao1)]; w = w1)
    rho_ccm[2,cnt], _ = MCDTS.ccm(Yy_cao, xx[1+maximum(taus_cao2):length(Yy_cao)+maximum(taus_cao2)]; w = w2)

    rho_ccm[3,cnt], _ = MCDTS.ccm(Yx_pec, yy[1+maximum(taus_pec1):length(Yx_pec)+maximum(taus_pec1)]; w = w1)
    rho_ccm[4,cnt], _ = MCDTS.ccm(Yy_pec, xx[1+maximum(taus_pec2):length(Yy_pec)+maximum(taus_pec2)]; w = w2)

    rho_ccm[5,cnt], _ = MCDTS.ccm(Yx_mcdts, yy[1+maximum(τ_mcdts1):length(Yx_mcdts)+maximum(τ_mcdts1)]; w = w1)
    rho_ccm[6,cnt], _ = MCDTS.ccm(Yy_mcdts, xx[1+maximum(τ_mcdts1):length(Yx_mcdts)+maximum(τ_mcdts1)]; w = w2)

end


varnames = ["y1_cao", "x1_cao", "y1_pec", "x1_pec", "y1_mcdts", "x1_mcdts", "Pearson"]

for i = 1:length(varnames)
    writestr = "results_analysis_CCM_full_combustion_$(run)_"*varnames[i]*".csv"
    if i == 7
        data = ρp
    else
        data = rho_ccm[i,:]
    end
    writedlm(writestr, data)
end
