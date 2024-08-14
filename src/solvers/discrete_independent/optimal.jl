function discrete_optimum(problem::DiscreteProblem)::DiscreteSolution
    resources = problem.resources
    p_target = problem.p_target
    v_target = problem.v_target

    indices = collect(1:length(resources))
    thread_best_costs = [Inf for _ = 1:Threads.nthreads()]
    thread_best_indices = [Vector{Int64}() for _ = 1:Threads.nthreads()]

    max_values = [r.max_value for r in resources]
    max_cum_sum = cumsum(sort(max_values, rev = true))

    min_resources = 1
    for i in eachindex(max_cum_sum)
        min_resources = i
        if max_cum_sum[i] >= v_target
            break
        end
    end

    Threads.@threads for subset in collect(powerset(indices, min_resources))
        set_cost = sro_target_function(resources[subset], p_target, v_target)
        if set_cost < thread_best_costs[Threads.threadid()]
            thread_best_indices[Threads.threadid()] = subset
            thread_best_costs[Threads.threadid()] = set_cost
        end
    end

    best_thread_index = findmin(thread_best_costs)[2]
    best_cost = minimum(thread_best_costs)
    output_indices = thread_best_indices[best_thread_index]

    return DiscreteSolution(problem, output_indices, resources[output_indices], best_cost)
end

"""
All-knowing oracle solver for discrete SRO instances.
Computes the optimal attainable cost for each instance of the 
problem and returns them as a vector.
"""
function oracle(instances::DiscreteInstances{T})::Vector{T} where {T}
    v_target = instances.problem.v_target
    values = instances.values
    costs = instances.costs

    solutions = zeros(T, length(instances.values))

    for i in eachindex(solutions)
        instance_values = [x[i] for x in values]
        instance_costs = [x[i] for x in costs]

        knapsack_target = sum(instance_values) - v_target
        if knapsack_target < 0
            solutions[i] = Inf
            continue
        end

        not_indices = solve_knapsack_problem(
            profit = instance_costs,
            weight = instance_values,
            capacity = knapsack_target,
        )
        solutions[i] = sum(instance_costs[Not(not_indices)])
    end

    return solutions
end


# knapsack solver taken from:
# https://jump.dev/JuMP.jl/stable/tutorials/linear/knapsack/
function solve_knapsack_problem(;
    profit::Vector{T},
    weight::Vector{Int64},
    capacity::Int64,
)::Vector{Int64} where {T}
    n = length(weight)
    # The profit and weight vectors must be of equal length.
    @assert length(profit) == n
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:n], Bin)
    @objective(model, Max, profit' * x)
    @constraint(model, weight' * x <= capacity)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    chosen_items = [i for i = 1:n if value(x[i]) > 0.5]
    return chosen_items
end
