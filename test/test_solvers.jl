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
    args = BPSOArgs(; n_particles=10, n_cycles=2)

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

    # n_thread evo
    n_steps = 10

    sol_1 = n_thread_evo(problems[1], n_steps)
    sol_2 = n_thread_evo(problems[2], n_steps)
    sol_3 = n_thread_evo(problems[3], n_steps)
    sol_4 = n_thread_evo(problems[4], n_steps)

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf
end

@testset "discrete_optima" begin
    # regular optimum
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

    # oracle
    inst = DiscreteInstances(Xoshiro(0), problems[3], 10)
    oracle_sol = oracle(inst)
    @test all([x == Inf for x in oracle_sol])

    rv = [0.0, 0.0, 1.0]
    rv_c = [1.0, 1.0, 1.0]
    res = DiscreteResource(rv, rv_c)
    p = DiscreteProblem([res], 0.5, 2)
    cant_fail = DiscreteInstances(p, 10)
    oracle_sol = oracle(cant_fail)
    @test all([x == 1.0 for x in oracle_sol])

end

@testset "discrete_simple_heuristics" begin
    # take_all
    problems = discrete_solver_problems()
    sol_1 = take_all(problems[1])
    sol_2 = take_all(problems[2])
    sol_3 = take_all(problems[3])
    sol_4 = take_all(problems[4])

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf

    # random_feasible
    sol_1 = random_feasible(problems[1])
    sol_2 = random_feasible(problems[2])
    sol_3 = random_feasible(problems[3])
    sol_4 = random_feasible(problems[4])

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf

    # subset_size_sampling
    sol_1 = subset_size_sampling(problems[1], 5)
    sol_2 = subset_size_sampling(problems[2], 5)
    sol_3 = subset_size_sampling(problems[3], 5)
    sol_4 = subset_size_sampling(problems[4], 5)

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf
end

@testset "PropagatingAgent" begin
    problems = discrete_solver_problems()

    host = "127.0.0.1"
    port = 5555

    sol_1 = propagated_agent_solver(problems[1], complete_topology(length(problems[1].resources)), host, port)
    sol_2 = propagated_agent_solver(problems[2], complete_topology(length(problems[2].resources)), host, port)
    sol_3 = propagated_agent_solver(problems[3], complete_topology(length(problems[3].resources)), host, port)
    sol_4 = propagated_agent_solver(problems[4], complete_topology(length(problems[4].resources)), host, port)

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf
end

# function blackboard_agent_solver(
#     problem::DiscreteProblem,
#     n_res_per_agent::Int64,
#     max_cycles::Int64
# )::DiscreteSolution

@testset "BlackBoardSolver" begin
    problems = discrete_solver_problems()

    sol_1 = blackboard_agent_solver(problems[1], 2, 4)
    sol_2 = blackboard_agent_solver(problems[2], 2, 4)
    sol_3 = blackboard_agent_solver(problems[3], 2, 4)
    sol_4 = blackboard_agent_solver(problems[4], 2, 4)

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf
end