using SRO
using OffsetArrays

@testset "DiscreteResource" begin
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

    # cdf
    @test isapprox(cdf(r), OffsetVector([0.3, 0.6, 1.0], 0:2))

    # ccdf
    @test isapprox(ccdf(r), OffsetVector([0.7, 0.4, 0.0], 0:2))

    # adding resources
    c1 = DiscreteResource([0.2, 0.4, 0.4], [0.0, -1.0, -2.0])
    c2 = DiscreteResource([0.2, 0.4, 0.4], [0.0, -2.0, -4.0])
    c3 = add(c1, c2)

    expected_cost = OffsetVector([0.0, -1.5, -3.0, -4.5, -6.0], 0:4)
    @test isapprox(c3.c, expected_cost; rtol = 0.0000001)

    res_v = [c1, c2, c1, c2]
    c3 = add(res_v)
    c3_man = add(add(add(c1, c2), c1), c2)
    @test isapprox(c3.p , c3_man.p; rtol = 0.0000001)
    @test isapprox(c3.c , c3_man.c; rtol = 0.0000001)

end

@testset "DiscreteProblem" begin

end