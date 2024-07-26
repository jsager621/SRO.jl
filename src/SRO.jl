module SRO

# avoid name collisions with packages used here
using Distributions
import Distributions: cdf, ccdf

# utils
include("utils/utils.jl")

#---------------------------
# models
#---------------------------
# continuous
include("models/continuous_correlated.jl")
export ContinuousResource, roll_values
export GaussianCopulaResources, roll_value_set
export CorrelatedResources,
    CopulaResources, CorrelatedProblem, CorrelatedInstances, CorrelatedSolution

# discrete
include("models/discrete_independent.jl")
export DiscreteResource, cdf, ccdf, add
export DiscreteProblem, DiscreteInstances, DiscreteSolution

#---------------------------
# Solvers
#---------------------------
# correlated continuous solvers
include("solvers/continuous_correlated/fk_fitting.jl")
include("solvers/continuous_correlated/metaheuristics.jl")
include("solvers/continuous_correlated/oracle.jl")

# independent discrete solvers
include("solvers/discrete_independent/metaheuristics.jl")
include("solvers/discrete_independent/optimal.jl")
include("solvers/discrete_independent/simple_heuristics.jl")
export aco, bpso, discreteOptimum

end