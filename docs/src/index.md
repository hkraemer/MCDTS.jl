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

The easiest way to use MCDTS for embedding is to use it with its default options just as

```julia
tree = mcdts_embedding(data, 100)
```
here with `N=100` trials. This will perform the MCDTS algorithm and return the full decision tree. The best embedding can then just be printed as

```julia
best_node = MCDTS.best_embedding(tree)
println(best_node)
```
e.g.
```
Node with τ=12, i_t=1 ,L=-1.5795030971438209 - full embd. τ=[0, 61, 48, 12] ,i_ts=[1, 1, 1, 1]
```

This version of `mcdts_embedding` uses default options to estimate all parameters of the algorithm. It is of course also possible to choose these individually and thus also use all different versions that are presented in the paper.

The most important paremters are the Theiler window, the minimum temporal distance for points that are considered neighours in phase space. We can get an estimate with the help of DelayEmbeddings.jl (part of DynamicalSystems.jl)

```julia
w1 = DelayEmbeddings.estimate_delay(data[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(data[:,2],"mi_min")
w3 = DelayEmbeddings.estimate_delay(data[:,3],"mi_min")
w = maximum(hcat(w1,w2,w3))
delays = 0:100
N_trials = 100
```

Also, we set the range of delays that we want to consider, here `0:100`. Next, we speficy the wanted optimization. The default optimziation goal is the Pecuzal algorithm, as outlined in the paper, with
```julia
pecuzal = MCDTS.PecuzalOptim()
```

Then we can call the proper `mcdts_embedding` function with

```julia
MCDTS.mcdts_embedding(Dataset(data[:,1]), pecuzal, w1, delays, runs)
best_node = MCDTS.best_embedding(tree)
println(best_node)
```

Further optimisation goals can be set by combining an [`AbstractDelayPreselection`](@ref) with an [`AbstractLoss`](@ref), e.g. via
```
optimgoal = MCDTS.MCDTSOptimGoal(MCDTS.FNN_statistic(0.05), MCDTS.Continuity_function())
```
which uses the FNN statistic with a threshold of 0.05 as a loss function and the continuity as a delay preselection method. 
