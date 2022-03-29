import Pkg
Pkg.activate(".")

using MCDTS
using Test
using Random
using DynamicalSystemsBase
using DelayEmbeddings
import MCDTS.L

# Check Lorenz System
Random.seed!(1234)
ds = Systems.lorenz()
data = trajectory(ds,200)
data = data[2001:2:3750,:]

w1 = DelayEmbeddings.estimate_delay(data[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(data[:,2],"mi_min")
w3 = DelayEmbeddings.estimate_delay(data[:,3],"mi_min")

w = maximum(hcat(w1,w2,w3))

@testset "MCDTS embedding tests" begin

    include("MCDTS_attractor_reconstruction_tests.jl")
    include("MCDTS_convergent_cross_mapping_tests.jl")
    include("MCDTS_nearest_neighbor_prediction_test.jl")

end
