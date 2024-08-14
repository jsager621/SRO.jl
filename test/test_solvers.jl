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
    args = BPSOArgs(; n_particles = 10, n_cycles = 2)

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
    @test all([x==Inf for x in oracle_sol])

    rv = [0.0, 0.0, 1.0]
    rv_c = [1.0, 1.0, 1.0]
    res = DiscreteResource(rv, rv_c)
    p = DiscreteProblem([res], 0.5, 2)
    cant_fail = DiscreteInstances(p, 10)
    oracle_sol = oracle(cant_fail)
    @test all([x==1.0 for x in oracle_sol])

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
end

@testset "PropagatingAgent" begin
    problems = discrete_solver_problems()

    HOST = ip"127.0.0.1"
    PORT = 5555
    c = Container()
    c.protocol = TCPProtocol(address = InetAddr(HOST, PORT))


    # agent structure
    agent = propagating_agent_factory(
        c,
        problems[1].resources[1],
        problems[1].p_target,
        problems[1].v_target,
        0.1,
        0.1,
    )

    @test agent.best_cost == Inf
    @test agent.best_agents == [address(agent)]

    # run single agent without neighbors
    run_agent(agent)
    @test agent.best_cost == Inf
    @test agent.best_agents == [address(agent)]

    # run solver logic
    neigh = Bool[
        0 1 1
        1 0 1
        1 1 0
    ]
    addr = InetAddr(HOST, PORT)
    info = 0.5
    term = 0.5

    sol_1 = run_propagated_agent_problem(problems[1], neigh, addr, info, term)
    sol_2 = run_propagated_agent_problem(problems[2], neigh, addr, info, term)
    sol_3 = run_propagated_agent_problem(problems[3], neigh, addr, info, term)
    sol_4 = run_propagated_agent_problem(problems[4], neigh, addr, info, term)

    @test sol_1.cost < Inf
    @test sol_2.cost < Inf
    @test sol_3.cost == Inf
    @test sol_4.cost == Inf
end