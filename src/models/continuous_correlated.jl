using Random
using Copulas
using Distributions

#---------------------------------------
# Resource Definition and Functions
#---------------------------------------
"""
Continuous SRO resource model. 
Values are given by a continuous univariate distribution `v`.
Costs are given by a function `c`: R -> R assigning each value an associated cost.

It is required that `v` has the attributed `lower` and `upper` as in the fields of
the `Truncated` type and that the distribution is actually bounded by these values.

WARNING: This structure does NOT enforce that the lower and upper bounds are actually
held by the distribution. If the distribution can roll values outside of these bounds,
then some of the solvers here can and will break in unforseen ways.
"""
struct ContinuousResource
    v::ContinuousUnivariateDistribution
    c::Function
    lower::Float64
    upper::Float64

    function ContinuousResource(v::ContinuousUnivariateDistribution, c::Function)
        if !hasproperty(v, :lower) || !hasproperty(v, :upper)
            throw(
                ArgumentError(
                    "Resource distribution must have the fields `lower` and `upper`!",
                ),
            )
        end

        if !hasmethod(c, Tuple{AbstractFloat,AbstractFloat})
            throw(ArgumentError("Cost function must have a method from Float to Float!"))
        end

        return new(v, c, v.lower, v.upper)
    end
end

"""
Sample a single `ContinuousResource` `n_values` times.
Computes the cost of each value as defined by the cost function.

# Returns
(`values`, `costs`)
"""
function roll_values(
    r::ContinuousResource,
    n_values::Int64,
)::Tuple{Vector{Float64},Vector{Float64}}
    rng = Xoshiro()
    return roll_values(rng, r, n_values)
end

"""
Sample a single `ContinuousResource` `n_values` times using `rng`.

# Returns
(`values`, `costs`)
"""
function roll_values(
    rng::AbstractRNG,
    r::ContinuousResource,
    n_values::Int64,
)::Tuple{Vector{Float64},Vector{Float64}}
    values = rand(rng, r.v, n_values)
    costs = [r.c(x) for x in r.v]
    return values, costs
end


"""
Universal parent type for sets of correlated continuous resources.
"""
abstract type CorrelatedResources end

"""
Parent type for a set of resources with some specified correlation given by a Copula.
"""
abstract type CopulaResources <: CorrelatedResources end

"""
Set of continuous `resources`` with known correlation given by a gaussian copula.
`distribution` contains the multivariate distribution of the set.
`copula` contains the gaussian copula created by the `cov_mat`.
"""
struct GaussianCopulaSet <: CopulaResources
    resources::Vector{ContinuousResource}
    cov_mat::Matrix{Float64}
    copula::GaussianCopula
    distribution::SklarDist

    function GaussianCopulaSet(
        resources::Vector{ContinuousResource},
        cov_mat::Matrix{Float64},
    )
        marginals = tuple([x.v for x in resources]...)
        copula = GaussianCopula(cov_mat)
        dist = SklarDist(copula, marginals)
        return new(resources, cov_mat, copula, dist)
    end
end

"""
Sample a `GaussianCopulaSet` `n_values` times.

# Returns
(`values`, `costs`) -
As matrices where each row corresponds to one of the resources and each column is one sample of 
the multivariate distribution.
"""
function roll_value_set(
    set::GaussianCopulaSet,
    n_values::Int64,
)::Tuple{Matrix{Float64},Matrix{Float64}}
    rng = Xoshiro()
    return roll_values(rng, set, n_values)
end

"""
Sample a `GaussianCopulaSet` `n_values` times using `rng`.

# Returns
(`values`, `costs`)
"""
function roll_value_set(
    rng::AbstractRNG,
    set::GaussianCopulaSet,
    n_values::Int64,
)::Tuple{Matrix{Float64},Matrix{Float64}}
    values = rand(rng, set.distribution, n_values)

    # compute the cost function of each value in column order
    # start with a copy of the values since we need those values in the calculation anyway
    costs = copy(values)
    for col in eachcol(costs)
        for i in eachindex(col)
            col[i] = set.resources[i].c(col[i])
        end
    end

    return values, costs
end


#---------------------------------------
# Problem Definition
#---------------------------------------
"""
SRO problem consisting of a set of correlated continuous `resources`,
a probability target `p_target` and a value target `v_target`.
"""
struct CorrelatedProblem
    resources::CorrelatedResources
    p_target::Float64
    v_target::Float64
end


#---------------------------------------
# Instances Definition
#---------------------------------------
"""
Set of concrete instances of an SRO `problem`.
The constructor rolls `n_instances` values using the given `rng` by the
distributions of the problem resources and computes their associated costs.
"""
struct CorrelatedInstances
    problem::CorrelatedProblem
    values::Matrix{Float64}
    costs::Matrix{Float64}

    function CorrelatedInstances(
        rng::AbstractRNG,
        problem::CorrelatedProblem,
        n_instances::Int64,
    )
        values, costs = roll_value_set(rng, problem.resources, n_instances)
        return new(problem, values, costs)
    end

    function CorrelatedInstances(problem::CorrelatedProblem, n_instances::Int64)
        rng = Xoshiro()
        return CorrelatedInstances(rng, problem, n_instances)
    end
end


#---------------------------------------
# Solution Definition
#---------------------------------------
"""
Solution structure for a continuous correlated SRO `problem`. Encapsulates different ways to give the solution:
* `resource_indices` - vector of indices of the optimal solution resources as in the `problem.resources` vector
* `resources` - vector of the optimal resources
* `probability` - sum probability of the solution resource set at the target value
* `cost` - expected costs of the solution resource set at the target value
* `is_exact` - flag indicating if the probability and cost values are exact or an estimate - set to true by all heuristic solvers
"""
struct CorrelatedSolution
    problem::CorrelatedProblem
    resource_indices::Vector{Int64}
    resources::Vector{ContinuousResource}
    probability::Float64
    cost::Float64
    is_exact::Bool
end