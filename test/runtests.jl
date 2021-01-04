using MCDTS
using Test

println("testing basic functionality")
@time @test include("base_test_rollout.jl")
@time @test include("base_test_expand.jl")
@time @test include("base_test_complete.jl")
