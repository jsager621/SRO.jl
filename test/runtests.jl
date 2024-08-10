using Test
using SRO
using Random
using Distributions

@testset "SRO Tests" begin
    include("test_models.jl")
    include("test_solvers.jl")
end