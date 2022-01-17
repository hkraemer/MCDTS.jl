# Monte Carlo Decision Tree Search for optimal embedding

This project implements the MCDTS algorithm outlined in the paper "Optimal state space reconstruction via Monte Carlo Decision Tree Search", accepted for publication in Nonlinear Dynamics. It aims to provide an optimal time delay state space reconstruction from time series data with the help of decisions trees and suitable statistics that guide the decisions done during the rollout of these trees. For all details of the algorithm the reader is referred to the accompanying paper. Here we provide an implementation of all the variants described in the paper. All major functions have docstrings that describe their use. In what follows basic use examples are outlined.

## Usage

*This repository serves the purpose of reproducibility only*. The proposed method will soon be part of the [DynamicalSystems.jl framework](https://juliadynamics.github.io/DynamicalSystems.jl/latest/) which will also contain sufficient documentation of the functionality. In order to reproduce the data, which has been used in the paper you have to
set back this repository to commit 842037c6 (MCDTS version 0.9.10). We are still further developing this repo in order to incorporate more functionality and preparing the
code for a migration to [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/latest/).

If you do not want to wait that long, we give an example of the basic usage (for that old version we mentioned).
We embed a Lorenz63 system. This is meant as a toy example: we generate
data from a Lorenz63 system and then try to reconstruct the full state space from only one of the three observables.

For this we also make use of `DynamicalSystems.jl`.

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
This means we have a 4-dimensional embedding with delays [0, 61, 48, 12], which decreased
the chosen cost/Loss-function (in this case the L-statistic) by a total amount of
L=-1.5795030971438209. Further, customized embedding options covering an ensemble
of different cost-functions can be looked up in the documentation.
