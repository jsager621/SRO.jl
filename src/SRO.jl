module SRO

# avoid name collisions with packages used here
using Random
using Copulas
using Distributions
import Distributions: cdf, ccdf
using StatsBase
using Combinatorics
using ConcurrentCollections
using JuMP
using HiGHS

# utils
include("utils/utils.jl")

#---------------------------
# models
#---------------------------
# continuous
include("models/continuous_correlated.jl")
export ContinuousResource, roll_values
export GaussianCopulaSet, roll_value_set
export ResourceSet, CopulaSet, CorrelatedProblem, CorrelatedInstances, CorrelatedSolution

# discrete
include("models/discrete_independent.jl")
export DiscreteResource, cdf, ccdf, combine, ZERO_RESOURCE
export DiscreteProblem, DiscreteInstances, DiscreteSolution
export sro_target_function

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
include("solvers/discrete_independent/distributed.jl")
export bpso, BPSOArgs
export one_plus_one_evo
export discrete_optimum, oracle, take_all, random_feasible
export AdjacencyMatrix, small_world, fully_connected, ring
export PropagatingAgent, propagating_agent_factory, run_agent, propagated_agent_solver


# benchmarking things, never merged into main!
include("solvers/discrete_independent/mh_multi_no_cache.jl")
export n_thread_evo_no_cache, bpso_no_cache

include("solvers/discrete_independent/mh_multi_simple_cache.jl")
export n_thread_evo_simple_cache, bpso_simple_cache

include("solvers/discrete_independent/mh_multi_thread_cache.jl")
export n_thread_evo_thread_cache, bpso_thread_cache

end