using DelayEmbeddings
using DynamicalSystemsBase

idxs = findall(optimal_d_pec .== 1)
idx_uni = zeros(Int, length(idxs))
for i = 1:length(idxs)
    idx_uni[i] = idxs[i][1]
end

idxs = findall(optimal_d_pec .== 1)
idx_multi = zeros(Int, length(idxs))
for i = 1:length(idxs)
    idx_multi[i] = idxs[i][1]
end

idx = idx_uni[1]


F = Fs[idx]

N = 8 # number of oscillators
Fs = 3.5:0.002:5 # parameter spectrum
dt = 0.1 # sampling time
total = 5000  # time series length

# Parameters analysis:
ε = 0.05  # recurrence threshold
dmax = 10   # maximum dimension for traditional tde
lmin = 2   # minimum line length for RQA
trials = 80 # trials for MCDTS
taus = 0:100 # possible delays
Tw = 0  # time window for obtaining the L-value

# randomly pick one time series
t_idx = 2
t_idx = [2,4,7]

# init Lorenz96
u0 = [0.590; 0.766; 0.566; 0.460; 0.794; 0.854; 0.200; 0.298]
lo96 = Systems.lorenz96(N, u0; F = 3.5)

set_parameter!(lo96, 1, F)
data = trajectory(lo96, total*dt; dt = dt, Ttr = 2500 * dt)
data_sample = data[:,t_idx]

# Traditional time delay embedding
𝒟d, τ_tdess, _ = optimal_traditional_de(data_sample[:,1], "fnn"; dmax = dmax)
optimal_d_tdes = size(𝒟d, 2)
R = RecurrenceMatrix(𝒟, ε; fixedrate = true)
RQA = rqa(R; theiler = τ_tde, lmin = lmin)
RQA_tde = hcat(RQA...)
Tw = τ_tdess
L_tdes = uzal_cost(regularize(𝒟d); w = τ_tdess, samplesize=1, Tw=Tw)
tau_tde[idx]
L_tde[idx]

# PECUZAL
# theiler = Int(floor(mean(τ_tde)))
theiler = τ_tdess
𝒟_pecs, τ_pecs, ts_pecs, Ls_pecs , epss = pecuzal_embedding(data_sample; τs = taus , w = theiler, Tw = 32)
optimal_d_pecs = size(𝒟_pecs,2)
R = RecurrenceMatrix(𝒟_pec, ε; fixedrate = true)
RQA = rqa(R; theiler = theiler, lmin = lmin)
RQA_pec = hcat(RQA...)
L_pecs = minimum(Ls_pecs)
L_pec[idx]
tau_pec[idx]

maxis, max_idx = DelayEmbeddings.get_maxima(vec(epss))

figure()
plot(epss)
scatter(taus[max_idx], maxis, marker="*")
grid()

Lss = zeros(100)
Lsss = zeros(100)
ss = Dataset(data_sample)
for Tw = 1:100
    Lsss[Tw] = uzal_cost(ss; Tw = Tw, w=theiler,samplesize=1)
    Lss[Tw] = uzal_cost(DelayEmbeddings.hcat_lagged_values(ss, vec(Matrix(ss)), taus[max_idx[1]])); Tw = Tw, w=theiler,samplesize=1)
end
figure()
plot(1:100, Lss, label="multi")
plot(1:100, Lsss, label="single")
legend()
grid()

t = 1:1000
data = sin.(2*π*t/60)

figure()
plot(data)


𝒟_pecs, τ_pecs, ts_pecs, Ls_pecs , epss = pecuzal_embedding(data; τs = 0:200 , w = 15, Tw = 15)

𝒟d, τ_tdess, _ = optimal_traditional_de(data, "fnn"; dmax = dmax)
estimate_delay(data, "ac_zero")

theiler = 1
Lss = zeros(100)
Lsss = zeros(100)
ss = Dataset(data.+0.000001*randn(1000))
ss = Dataset(randn(1000))
for Tw = 1:100
    Lsss[Tw] = uzal_cost(ss; Tw = Tw, w=theiler,samplesize=1)
    Lss[Tw] = uzal_cost(DelayEmbeddings.hcat_lagged_values(ss, vec(Matrix(ss)), 1); Tw = Tw, w=theiler,samplesize=1)
end
figure()
plot(1:100, Lss, label="multi")
plot(1:100, Lsss, label="single")
legend()
grid()



using DelimitedFiles
data = readdlm("milankovitch_data.txt")

milo_inso = data[:,5]

w = estimate_delay(milo_inso, "mi_min")

𝒟d, τ_tdess, _ = optimal_traditional_de(milo_inso, "fnn"; dmax = dmax)
