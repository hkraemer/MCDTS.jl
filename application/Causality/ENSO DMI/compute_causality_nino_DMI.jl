using MCDTS
using DelayEmbeddings
using DelimitedFiles
using Random
using PyPlot
pygui(true)

# load data and regularize it
nino = Matrix(regularize(Dataset(readdlm("./application/Causality/ENSO DMI/raw data/nino34.txt"))))
DMI = Matrix(regularize(Dataset(readdlm("./application/Causality/ENSO DMI/raw data/DMI.txt"))))
# transform into vectors and add tiny noise
ninos = zeros(length(nino))
DMIs = zeros(length(nino))
Random.seed!(1234)
for i = 1:length(nino)
    ninos[i] = nino[i]+0.00000000001*randn()
    DMIs[i] = DMI[i]+0.00000000001*randn()
end

# figure()
# subplot(211)
# plot(ninos)
# subplot(212)
# plot(DMIs)

## make computations for only one time series length

xx = ninos[1:100]
yy = DMIs[1:100]
taus1 = 0:50
trials = 30
# compute decorrelation time
w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")

tree = MCDTS.mc_delay(Dataset(xx), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials;
    verbose=true, CCM = false, PRED = true, PRED_KL = true, Y_CCM = Dataset(yy))
best_node = MCDTS.best_embedding(tree)
τ_mcdts = best_node.τs
L = best_node.L
println("MCDTS taus: $τ_mcdts")
println("MCDTS ρ: $L")
Yx = genembed(xx, τ_mcdts)
Yy = genembed(yy, τ_mcdts)
ρ_nino_DMI_pecuzal[cnt], _ = MCDTS.ccm(Dataset(Yx),Dataset(Yy); w = w1)

tree = MCDTS.mc_delay(Dataset(yy), w2, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials;
    verbose=true, CCM = true, Y_CCM = Dataset(xx))
best_node = MCDTS.best_embedding(tree)
τ_mcdts = best_node.τs
L = best_node.L



## vary time series length for showing convergence properties



# define the different length scales
lengths = 200:100:length(nino)

cnt = 1
taus1 = 0:40 # possible delay for MCDTS
trials = 40 # sampling trials for MCDTS
ρ_DMI_nino_kennel = zeros(length(lengths))
ρ_nino_DMI_kennel = zeros(length(lengths))
ρ_DMI_nino_pecuzal = zeros(length(lengths))
ρ_nino_DMI_pecuzal = zeros(length(lengths))
for i in lengths
    # slice time series
    println(i)
    xx = ninos[1:i]
    yy = DMIs[1:i]
    # compute decorrelation time
    w1 = DelayEmbeddings.estimate_delay(xx, "mi_min")
    w2 = DelayEmbeddings.estimate_delay(yy, "mi_min")

    # Kennel embedding
    Yx, tau = optimal_traditional_de(xx,"fnn"; w = w1)
    taus = zeros(Int,size(Yx,2))
    for j = 2:size(Yx,2)
        taus[j] = (j-1)*tau
    end
    Yy = genembed(yy,taus)
    ρ_nino_DMI_kennel[cnt], _ = MCDTS.ccm(Yx,Yy; w = w1)

    Yx, tau = optimal_traditional_de(yy,"fnn"; w = w2)
    taus = zeros(Int,size(Yx,2))
    for j = 2:size(Yx,2)
        taus[j] = (j-1)*tau
    end
    Yy = genembed(xx,taus)

    ρ_DMI_nino_kennel[cnt], _ = MCDTS.ccm(Yx,Yy; w = w2)

    # MCDTS embedding
    tree = MCDTS.mc_delay(Dataset(xx), w1, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials;
        verbose=false, CCM = true, Y_CCM = Dataset(yy))
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    L = best_node.L
    println("MCDTS taus: $τ_mcdts")
    println("MCDTS ρ: $L")
    Yx = genembed(xx, τ_mcdts)
    Yy = genembed(yy, τ_mcdts)
    ρ_nino_DMI_pecuzal[cnt], _ = MCDTS.ccm(Dataset(Yx),Dataset(Yy); w = w1)

    tree = MCDTS.mc_delay(Dataset(yy), w2, (L)->(MCDTS.softmaxL(L,β=2.)), taus1, trials;
        verbose=false, CCM = true, Y_CCM = Dataset(xx))
    best_node = MCDTS.best_embedding(tree)
    τ_mcdts = best_node.τs
    L = best_node.L
    println("MCDTS taus: $τ_mcdts")
    println("MCDTS ρ: $L")
    Yx = genembed(xx, τ_mcdts)
    Yy = genembed(yy, τ_mcdts)
    ρ_DMI_nino_pecuzal[cnt], _ = MCDTS.ccm(Dataset(Yx),Dataset(Yy); w = w2)


    cnt +=1
end

figure()
subplot(121)
plot(lengths,ρ_nino_DMI_pecuzal,label="Nino → DMI")
plot(lengths,ρ_DMI_nino_pecuzal,label="DMI → Nino")
title("PECUZAL")
grid()
legend()

subplot(122)
plot(lengths,ρ_nino_DMI_kennel,label="Nino → DMI")
plot(lengths,ρ_DMI_nino_kennel,label="DMI → Nino")
title("Kennel")
grid()
legend()
