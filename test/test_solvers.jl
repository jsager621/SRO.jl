#----------------------------
# Continuous Solvers
#----------------------------




#----------------------------
# Discrete Solvers
#----------------------------
function discrete_solver_problems()
    rvs = [[0.1, 0.4, 0.5], [0.1, 0.4, 0.5], [0.5, 0.5]]
    costs = [[0.0, 1.0, 2.0], [0, 1.0, 2.0], [0.0, 5.0]]
    resources = [DiscreteResource(rvs[i], costs[i]) for i in eachindex(rvs)]

    p1 = DiscreteProblem(resources, 0.1, 4)
    p2 = DiscreteProblem(resources, 0.1, 5)
    p3 = DiscreteProblem(resources, 0.1, 6)
    p4 = DiscreteProblem(resources, 1.0, 4)

    return [p1, p2, p3, p4]
end

@testset "discrete_metaheuristics" begin
    problems = discrete_solver_problems()

    # bpso
    args = BPSOArgs(;n_particles = 10, n_cycles = 2)

    sol_1 = bpso(problems[1], args)
    sol_2 = bpso(problems[2], args)
    sol_3 = bpso(problems[3], args)
    sol_4 = bpso(problems[4], args)

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf

    # aco
    args = ACOArgs()

    sol_1 = aco(problems[1], args)
    sol_2 = aco(problems[2], args)
    sol_3 = aco(problems[3], args)
    sol_4 = aco(problems[4], args)

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf

    # one plus one evo
    n_steps = 10

    sol_1 = one_plus_one_evo(problems[1], n_steps)
    sol_2 = one_plus_one_evo(problems[2], n_steps)
    sol_3 = one_plus_one_evo(problems[3], n_steps)
    sol_4 = one_plus_one_evo(problems[4], n_steps)

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf
end

@testset "discrete_optimum" begin
    problems = discrete_solver_problems()

    sol_1 = discrete_optimum(problems[1])
    sol_2 = discrete_optimum(problems[2])
    sol_3 = discrete_optimum(problems[3])
    sol_4 = discrete_optimum(problems[4])

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf

    @test Set(sol_1.resources) == Set(problems[1].resources[1:2])
    @test Set(sol_2.resources) == Set(problems[1].resources)
end

@testset "discrete_simple_heuristics" begin
    problems = discrete_solver_problems()
    sol_1 = take_all(problems[1])
    sol_2 = take_all(problems[2])
    sol_3 = take_all(problems[3])
    sol_4 = take_all(problems[4])

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf
end