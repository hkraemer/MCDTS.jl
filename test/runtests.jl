current_dir = pwd()
Pkg.activate(current_dir)

using MCDTS
using Test

println("testing basic functionality")
@time @test include("base_test.jl")
