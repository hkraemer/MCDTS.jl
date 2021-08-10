using MCDTS
using DelayEmbeddings
using Statistics
using Random
using StatsBase
using LinearAlgebra
using DelimitedFiles

## Combustion data
data1 = readdlm("./application/Causality/Combustion/data/pressure_downsampled_same_sampling.txt")
data2 = readdlm("./application/Causality/Combustion/data/heat_release_downsampled_same_sampling.txt")

## Generate subset

Random.seed!(122)
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


cnt = 1
ρp = zeros(length(N_min:step:N))

for i = N_min:step:N

    println(i)
    xx = s1[1:i]
    yy = s2[1:i]

    # Pearson rho
    ρp[cnt] = Statistics.cor(xx,yy)

    cnt +=1

end

writedlm("./application/Causality/Combustion/results/results_analysis_CCM_full_combustion_10_Pearson.csv",ρp)
