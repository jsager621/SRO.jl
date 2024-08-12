"""
Given a discrete optimization `problem`, return the full set of resources as a solution.
"""
function take_all(problem::DiscreteProblem{T})::DiscreteSolution{T} where {T}
    return DiscreteSolution(
        problem,
        collect(1:length(problem.resources)),
        problem.resources,
        sro_target_function(problem.resources, problem.p_target, problem.v_target)
    )
end