using Pkg
Pkg.activate(".")
using MCDTS
using DynamicalSystemsBase
using PyPlot
pygui(true)

lo = Systems.lorenz()
tr = trajectory(lo, 500; dt = 0.01, Ttr = 10)

s = tr[1:20000,:]

figure()
plot3D(s[:,1],s[:,2],s[:,3])


figure()
plot(s[:,1])

Y = s
Tw = 1
NN = length(Y)-Tw
prediction = Y  # intitial trajectory
prediction1 = MCDTS.insample_zeroth_prediction(prediction1; w = 15, K = 3)
prediction2 = MCDTS.insample_linear_prediction(prediction2; w = 15, K = 1)

prediction1, _ = MCDTS.make_insample_prediction(MCDTS.local_model("zeroth", 3, 6), Y; w = 15)
prediction2, _ = MCDTS.make_insample_prediction(MCDTS.local_model("linear", 3, 6), Y; w = 15)


N = length(s)
NN = length(prediction1)

figure()
plot(1:N,s[:,1])
plot(7:N,prediction1[:,1], label="zeroth")
plot(7:N,prediction2[:,1], label="linear")
legend()
grid()

predicted_hegger, _ = MCDTS.local_linear_prediction(prediction_hegger, K1; theiler = w1)
push!(prediction_hegger, predicted_hegger)

metric = Euclidean()

Y = prediction_hegger
NN = length(Y)
ns = 1:NN
vs = Y[ns]
vtree = KDTree(Y[1:length(Y)-1,:], metric)
allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, theiler)

ϵ_ball = zeros(T, K, D) # preallocation
A = ones(K,D) # datamatrix for later linear equation to solve for AR-process
# consider NNs of the very last point of the trajectory
NNidxs = allNNidxs[end-1]

figure()
plot(Y[end-700:end,1])
ylim([-20, 20])
