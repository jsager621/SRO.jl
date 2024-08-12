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
Works by rollin a random order of resources, determining all feasible solutions in that order and then
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
