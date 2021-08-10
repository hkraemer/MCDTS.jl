using MCDTS
using DelayEmbeddings
using Statistics
using Random
using StatsBase
using LinearAlgebra
using DelimitedFiles

## Combustion data
data = readdlm("./application/Causality/Electrochemical oscillators/data/I0_08.txt")
#data = readdlm("./application/Causality/Electrochemical oscillators/data/Izero.txt")

# downsampling
data = data[1:5:end,:]

data1 = data[:,2]
data2 = data[:,3]

N = length(data1)
N_min = 500
step = 100

s1 = data1[:]
s2 = data2[:]
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

    # standard Pearson
    ρp[cnt] = Statistics.cor(xx,yy)

    cnt += 1
end


writedlm("./application/Causality/Electrochemical oscillators//results/results_analysis_CCM_full_chemosc_ps_Pearson.csv",ρp)
