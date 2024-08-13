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

    @test r.v == [0, 1, 2]
    @test r.max_value == 2
    @test r[2] == (0.3, 2.0, 1)
    @test length(r) == 3

    # cdf
    @test isapprox(cdf(r), [0.3, 0.6, 1.0])

    # ccdf
    @test isapprox(ccdf(r), [0.7, 0.4, 0.0])

    # adding resources
    c1 = DiscreteResource([0.2, 0.4, 0.4], [0.0, -1.0, -2.0])
    c2 = DiscreteResource([0.2, 0.4, 0.4], [0.0, -2.0, -4.0])
    c3 = combine(c1, c2)

    expected_cost = [0.0, -1.5, -3.0, -4.5, -6.0]
    @test isapprox(c3.c, expected_cost; rtol = 0.0000001)

    res_v = [c1, c2, c1, c2]
    c3 = combine(res_v)
    c3_man = combine(combine(combine(c1, c2), c1), c2)
    @test isapprox(c3.p, c3_man.p; rtol = 0.0000001)
    @test isapprox(c3.c, c3_man.c; rtol = 0.0000001)


    # target function
    p_target = 0.5
    v_target = 2

    c1 = DiscreteResource([0.2, 0.4, 0.4], [0.0, -1.0, -2.0])
    c2 = DiscreteResource([0.2, 0.4, 0.4], [0.0, -2.0, -4.0])

    # p of at least 1 is 96%
    @test isapprox(sro_target_function([c1, c2], 0.95, 1), -1.5)
    # p of at least 4 is 16%
    @test isapprox(sro_target_function([c1, c2], 0.15, 4), -6.0)

    # infeasible probabilities
    @test sro_target_function([c1, c2], 0.97, 1) == Inf
    @test sro_target_function([c1, c2], 0.17, 4) == Inf

    # infeasible values
    @test sro_target_function([c1, c2], 0.0, 5) == Inf
    
    # empty set handling
    @test sro_target_function(DiscreteResource{Float64}[], 0.0, 0) == Inf
end

@testset "DiscreteInstances" begin
    # constructor tests
    c1_cost = [0.0, -1.0, -2.0]
    c2_cost = [0.0, -2.0, -4.0]
    c1 = DiscreteResource([0.2, 0.4, 0.4], c1_cost)
    c2 = DiscreteResource([0.2, 0.4, 0.4], c2_cost)
    problem = DiscreteProblem([c1, c2], 0.5, 10)

    n_instances = 100

    inst = DiscreteInstances(problem, n_instances)

    # correct output formats
    @test length(inst.values) == 2
    @test length(inst.values[1]) == n_instances
    @test length(inst.values[2]) == n_instances

    @test length(inst.costs) == 2
    @test length(inst.costs[1]) == n_instances
    @test length(inst.costs[2]) == n_instances

    # correct cost content
    @test all([
        inst.costs[1][i] == c1_cost[inst.values[1][i]] for i in eachindex(inst.values[1])
    ])
    @test all([
        inst.costs[2][i] == c2_cost[inst.values[2][i]] for i in eachindex(inst.values[2])
    ])
end

@testset "ContinuousResource" begin
    # constructor tests
    not_ok_rv = Normal(5, 5)
    ok_rv = truncated(Normal(5, 5); lower = 0, upper = 5)
    ok_cost(x::Float64)::Float64 = x^2

    @test_throws ArgumentError ContinuousResource(not_ok_rv, ok_cost)
    res = ContinuousResource(ok_rv, ok_cost)

    # rolling values
    values, costs = roll_values(res, 10)
    @test length(values) == 10
    @test length(costs) == 10
    @test all([x >= 0 for x in values])
    @test all([x <= 5 for x in values])
    @test all([ok_cost(values[i]) == costs[i] for i in eachindex(values)])

    values, costs = roll_values(Xoshiro(1), res, 10)
    @test length(values) == 10
    @test length(costs) == 10
    @test all([x >= 0 for x in values])
    @test all([x <= 5 for x in values])
    @test all([ok_cost(values[i]) == costs[i] for i in eachindex(values)])
end

@testset "GaussianCopulaSet" begin
    # constructor
    raw_rvs = [Normal(5, 5), Beta(1.5, 1.5), Weibull(1.5, 1.5)]
    rvs = [truncated(x; lower = 0, upper = 1) for x in raw_rvs]
    resources = [ContinuousResource(x, y -> y^2) for x in rvs]

    cov_mat = [
        1.0 0.5 0.5
        0.5 1.0 0.5
        0.5 0.5 1.0
    ]

    s = GaussianCopulaSet(resources, cov_mat)
    @test true

    # functions
    values, costs = roll_value_set(s, 10)
    @test all([values[i]^2 == costs[i] for i in eachindex(values)])

    values, costs = roll_value_set(Xoshiro(1), s, 10)
    @test all([values[i]^2 == costs[i] for i in eachindex(values)])

end