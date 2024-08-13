using Test
using SRO
using Random
using Distributions
using Mango
using Sockets: InetAddr, @ip_str

@testset "SRO Tests" begin
    include("test_models.jl")
    include("test_solvers.jl")
end