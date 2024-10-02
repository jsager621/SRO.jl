"""
Given a discrete optimization `problem`, return the full set of resources as a solution.
"""
function take_all(problem::DiscreteProblem{T})::DiscreteSolution{T} where {T}
    return DiscreteSolution(
        problem,
        collect(1:length(problem.resources)),
        problem.resources,
        sro_target_function(problem.resources, problem.p_target, problem.v_target),
    )
end

"""
Given a discrete optimization `problem`, return a random feasible solution.
Works by rolling a random order of resources, determining all feasible solutions in that order and then
uniformly randomly returning one of those feasible solutions.
"""
function random_feasible(problem::DiscreteProblem{T})::DiscreteSolution{T} where {T}
    return random_feasible(Xoshiro(), problem)
end

function random_feasible(
    rng::AbstractRNG,
    problem::DiscreteProblem{T},
)::DiscreteSolution{T} where {T}
    resources = problem.resources
    p_target = problem.p_target
    v_target = problem.v_target

    set_indices =
        sample(rng, collect(1:length(resources)), length(resources), replace = false)

    feasible_indices = Vector{Vector{Int64}}()
    feasible_costs = Vector{T}()

    for i in eachindex(set_indices)
        indices = set_indices[1:i]
        resource_subset = resources[indices]
        cost = sro_target_function(resource_subset, p_target, v_target)

        if cost < Inf
            push!(feasible_costs, cost)
            push!(feasible_indices, indices)
        end
    end

    if isempty(feasible_indices)
        return DiscreteSolution(problem, collect(1:length(resources)), resources, Inf)
    end

    chosen_index = rand(rng, 1:length(feasible_indices))

    return DiscreteSolution(
        problem,
        feasible_indices[chosen_index],
        resources[feasible_indices[chosen_index]],
        feasible_costs[chosen_index]
    )
end

"""
Take `n_samples` random samples of subsets of size 1:length(problem) and 
return the best solution from these subsets 
"""
function subset_size_sampling(problem::DiscreteProblem{T}, n_samples::Int64)::DiscreteSolution{T} where {T}
    return subset_size_sampling(Xoshiro(), problem, n_samples)
end

function subset_size_sampling(rng::AbstractRNG, problem::DiscreteProblem{T}, n_samples::Int64)::DiscreteSolution{T} where {T}
    resources = problem.resources
    p_target = problem.p_target
    v_target = problem.v_target

    full_cost = sro_target_function(resources, p_target, v_target)
    if full_cost == Inf
        return DiscreteSolution(
            problem,
            Int64[],
            DiscreteResource{T}[],
            Inf
        )
    end

    thread_costs = [full_cost for _ in 1:Threads.nthreads()]
    thread_sets = [ones(Bool, length(resources)) for _ in 1:Threads.nthreads()]

    best_cost = full_cost
    select_vector = ones(Bool, length(resources))
    
    # check from size 1 to all but one resource
    for size in 1:length(resources)-1
        base = vcat(ones(Bool, size), zeros(Bool, length(resources)-size))
        Threads.@threads for _ in 1:n_samples
            sample = shuffle(rng, base)
            sample_cost = sro_target_function(resources[sample], p_target, v_target)
            if sample_cost < thread_costs[Threads.threadid()]
                thread_costs[Threads.threadid()] = sample_cost
                thread_sets[Threads.threadid()] = sample
            end
        end
    end

    # sro_target_function(resources[new_select_vector], p_target, v_target)

    m, idx = findmin(thread_costs)
    if m < best_cost
        best_cost = m
        select_vector = thread_sets[idx]
    end
    
    return DiscreteSolution(
        problem,
        collect(1:length(resources))[select_vector],
        resources[select_vector],
        best_cost,
    )
end