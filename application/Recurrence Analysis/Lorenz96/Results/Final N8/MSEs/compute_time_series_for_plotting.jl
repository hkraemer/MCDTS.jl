using DynamicalSystemsBase
using DelayEmbeddings
using DelimitedFiles

N = 8
dt = 0.01 # sampling time
total = 10000  # time series length
lo96 = Systems.lorenz96(N; F = 4.5)
data = trajectory(lo96, total*dt; dt = 0.01, Ttr = 40)

writedlm("./application/artificial data/Lorenz96/Results/Final N8/MSEs/example_trajectory.csv", data)


Y, ta_, ts, L, _ = pecuzal_embedding(data[:,2]; τs = 0:100, w = 17, econ=true)

writedlm("./application/artificial data/Lorenz96/Results/Final N8/MSEs/example_reconstruction.csv", Y)
