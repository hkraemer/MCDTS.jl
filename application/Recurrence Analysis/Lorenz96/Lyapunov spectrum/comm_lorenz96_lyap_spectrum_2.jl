using DynamicalSystems
using MCDTS
using DelimitedFiles

## We apply the MCDTS-approach to the Lorenz system and compare the results of
# the reconstruction to the ones gained from PECUZAL and standard TDE. Here we
# compute the Lyapunov spectra for a bunch of parameter-settings

N = 8 # number of oscillators
# init Lorenz96
u0 = [0.590; 0.766; 0.566; 0.460; 0.794; 0.854; 0.200; 0.298]
lo96 = Systems.lorenz96(N, u0; F = 3.7)
# check Lyapunox spectrum
Fs = 3.7:0.002:4
λs = zeros(length(Fs),N)
for (i,F) in enumerate(Fs)
  println(i)
  set_parameter!(lo96, 1, F)
  λs[i,:] = lyapunovs(lo96, 100000; Ttr = 10000)
end

writedlm("./application/artificial data/Lorenz96/Lyapunov spectrum/Lyaps_Lo96_N_$(N)_3_7_to_4.csv", λs)

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
