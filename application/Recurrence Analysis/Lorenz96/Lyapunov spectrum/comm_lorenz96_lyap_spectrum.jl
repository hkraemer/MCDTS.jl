using DynamicalSystems
using MCDTS
using DelimitedFiles

## We apply the MCDTS-approach to the Lorenz system and compare the results of
# the reconstruction to the ones gained from PECUZAL and standard TDE. Here we
# compute the Lyapunov spectra for a bunch of parameter-settings

N = 40 # number of oscillators
# init lorenz96
lo96 = Systems.lorenz96(N; F = 3.5)
# check Lyapunox spectrum
Fs = 3.5:0.004:5
λs = zeros(length(Fs),N)
for (i,F) in enumerate(Fs)
  println(i)
  set_parameter!(lo96, 1, F)
  λs[i,:] = lyapunovs(lo96, 100000; Ttr = 10000)
end

writedlm("./application/artificial data/Lorenz96/Lyapunov spectrum/Lyaps_Lo96_N_$(N).csv", λs)

##
using PyPlot
pygui(true)

figure()
for i = 1:N
  plot(Fs, λs[:,i], label="λ$i")
end
xlabel("F")
title("Lyapunov spectrum for Lorenz96 (N = $N)")
legend()
grid()
