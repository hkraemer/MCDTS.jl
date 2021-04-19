using PyPlot
pygui(true)
using DelimitedFiles

data = readdlm("./application/CENOGRID/data/detrended.txt")

time = data[:,1]
O18 = data[:,2]
O13 = data[:,3]

figure()
plot(time,O18, label="O18")
plot(time,O13, label="O13")
grid()
legend()
