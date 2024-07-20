using SRO
using OffsetArrays

@testset "DiscreteModel" begin
    # constructor tests
    # type consistency
    p = ["a", "b", "c"]
    c = [1.0, 2.0, 3.0]
    @test_throws MethodError DiscreteResource(p, c)
    @test_throws MethodError DiscreteResource(c, p)

    p = Float64[1.0, 2.0, 3.0]
    c = BigFloat[1.0, 2.0, 3.0]
    @test_throws MethodError DiscreteResource(p, c)

    # p sum 1
    p = [0.3, 0.4, 0.5]
    c = [1.0, 2.0, 3.0]
    @test_throws ArgumentError DiscreteResource(p, c)

    # equal length
    p = [0.3, 0.3, 0.4]
    c = [1.0, 2.0, 3.0, 4.0]
    @test_throws ArgumentError DiscreteResource(p, c)

    # correct instantiation
    p = [0.3, 0.3, 0.4]
    c = [1.0, 2.0, 3.0]
    r = DiscreteResource(p, c)

    @test r.v == OffsetVector([0,1,2], 0:2)
    @test r.max_value == 2
    @test r[1] == (0.3, 2.0, 1)
    @test length(r) == 3
end