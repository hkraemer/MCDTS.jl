# Monte Carlo Decision Tree Search for optimal embedding

This project implements the MCDTS algorithm outlined in the paper ____. It aims to provide an optimal time delay state space reconstruction from time series data with the help of decisions trees and suitable statistics that guide the decisions done during the rollout of these trees. For all details of the algorithm the reader is referred to the accompanying paper. Here we provide an implementation of all the variants described in the paper. All major functions have docstrings that describe their use. In what follows basic use examples are outlined.

## Usage

As a first example, we embedd a Lorenz63 system. This is meant as a toy example: we generate data from a Lorenz63 system and then try to reconstructe the full state space from only one of the three observables.

For this we also make use of DynamicalSystems.jl.

First we import the needed modules and generate a long trajectory of the Lorenz system (and discard any transient dynamics)

```julia
using DynamicalSystems, MCDTS, Random, Test, DelayEmbeddings

# Check Lorenz System
Random.seed!(1234)
ds = Systems.lorenz()
data = trajectory(ds,200)
data = data[10001:end,:]
```

Then we get an estimate for the Theiler window used with the help of DelayEmbeddings.jl (part of DynamicalSystems.jl) and set the range of delays that we want to consider.

```julia
w1 = DelayEmbeddings.estimate_delay(data[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(data[:,2],"mi_min")
w3 = DelayEmbeddings.estimate_delay(data[:,3],"mi_min")
w = maximum(hcat(w1,w2,w3))
delays = 0:100
N_trials = 100
```

The actual MCDTS embedding then gets done by calling 'mcdts_embedding' with the parameters of your choise. The full doc string specifies all options. Here, the most important ones are presented:
