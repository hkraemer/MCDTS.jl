using MCDTS
using Test

println("testing basic functionality")
@time @test include("base_test.jl")
