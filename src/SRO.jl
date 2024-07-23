module SRO

# avoid name collisions with packages used here
using Distributions
import Distributions: cdf, ccdf

# utils
include("utils/utils.jl")

# models
include("models/copula_correlated.jl")
include("models/discrete_independent.jl")
export DiscreteResource, cdf, ccdf, add, DiscreteProblem, DiscreteInstances, DiscreteSolution

# correlated continuous solvers
include("solvers/copula_correlated/fk_fitting.jl")
include("solvers/copula_correlated/metaheuristics.jl")
include("solvers/copula_correlated/oracle.jl")

# independent discrete solvers
include("solvers/discrete_independent/metaheuristics.jl")
include("solvers/discrete_independent/optimal.jl")
include("solvers/discrete_independent/simple_heuristics.jl")
export aco, bpso, discreteOptimum

end