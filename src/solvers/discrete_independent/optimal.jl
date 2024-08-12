function discrete_optimum(problem::DiscreteProblem)::DiscreteSolution
    resources = problem.resources
    p_target = problem.p_target
    v_target = problem.v_target

    indices = collect(1:length(resources))
    thread_best_costs = [Inf for _ in 1:Threads.nthreads()]
    thread_best_indices = [Vector{Int64}() for _ in 1:Threads.nthreads()]

    max_values = [r.max_value for r in resources]
    max_cum_sum = cumsum(sort(max_values, rev=true))

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

    return DiscreteSolution(
        problem,
        output_indices,
        resources[output_indices],
        best_cost,
    )
end