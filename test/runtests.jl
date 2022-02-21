import Pkg
Pkg.activate(".")

using MCDTS
using Test
using Random
using DynamicalSystemsBase
using DelayEmbeddings

println("testing basic functionality")
#@time @test include("base_test_rollout.jl")
#@time @test include("base_test_expand.jl")
@time @test include("base_test_complete.jl")
