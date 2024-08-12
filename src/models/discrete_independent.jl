using GenericFFT
using InvertedIndices
using Random

#---------------------------------------
# Resource Definition and Functions
#---------------------------------------
"""
Discrete and independent SRO resource model. 
All values are defined from 0 to the maximum value. 
The value vector `v` is mapped to the indices of the probability vector with an offset of -1
so the first probability in `p` is the probability of getting value 0, etc.
The probability vector `p` and cost vector `c` must be of equal length.
"""
struct DiscreteResource{T<:AbstractFloat}
    max_value::Int64
    p::Vector{T}
    c::Vector{T}
    v::Vector{Int64}

    function DiscreteResource(prob::Vector{T}, cost::Vector{T}) where {T<:AbstractFloat}
        if length(prob) != length(cost)
            throw(ArgumentError("Probabililty and cost vector must have equal length!"))
        end

        if !isapprox(sum(prob), 1.0)
            throw(ArgumentError("Probabilities must add up to 1!"))
        end

        max_value = length(prob) - 1
        v = collect(0:max_value)

        return new{T}(max_value, prob, cost, v)
    end
end

const ZERO_RESOURCE = DiscreteResource([1.0], [0.0])

function Base.getindex(r::DiscreteResource, k)
    return (r.p[k], r.c[k], r.v[k])
end

function Base.length(r::DiscreteResource)
    return length(r.p)
end

function cdf(r::DiscreteResource{T})::Vector{T} where {T}
    return cumsum(r.p)
end

function ccdf(r::DiscreteResource{T})::Vector{T} where {T}
    return [1 - n for n in cdf(r)]
end

function convolve(a::Vector{T}, b::Vector{T})::Vector{T} where {T<:AbstractFloat}
    n = length(a)
    m = length(b)

    a_pad = [a; zeros(m - 1)]
    b_pad = [b; zeros(n - 1)]

    raw_convolve = ifft(fft(a_pad) .* fft(b_pad))
    reals = [real(x) for x in raw_convolve]
    filtered = [x < 0 ? T(0.0) : x for x in reals]

    return filtered
end

function convolve(vs::Vector{Vector{T}})::Vector{T} where {T<:AbstractFloat}
    final_length = sum([length(x) for x in vs]) - length(vs) + 1
    paddeds = Vector{Vector{T}}()
    for v in vs
        push!(paddeds, [v; zeros(T, final_length - length(v))])
    end

    ffts = [fft(x) for x in paddeds]
    raw_convolve = ifft(reduce((x, y) -> x .* y, ffts))
    reals = [real(x) for x in raw_convolve]
    filtered = [x < 0 ? T(0.0) : x for x in reals]

    return filtered
end

function add(a::DiscreteResource{T}, b::DiscreteResource{T})::DiscreteResource{T} where {T}
    f3 = convolve(a.p, b.p)

    n = length(a)
    m = length(b)

    f1_pad = [a.p; zeros(m - 1)]
    f2_pad = [b.p; zeros(n - 1)]

    c1_pad = [a.c; zeros(m - 1)]
    c2_pad = [b.c; zeros(n - 1)]

    w1 = f2_pad .* c2_pad
    w2 = f1_pad .* c1_pad

    weighted_cost = ifft((fft(f1_pad) .* fft(w1)) .+ (fft(f2_pad) .* fft(w2)))
    # 
    # ifft(fft(f1 .* (f2 .* c2)) .+ fft(f2 .* (f1 .* c1))) ./ f3
    #

    real_cost = [real(x) for x in weighted_cost]
    c3 = real_cost ./ f3

    return DiscreteResource(f3, c3)
end

function add(resources::Vector{DiscreteResource{T}})::DiscreteResource{T} where {T}
    if length(resources) == 0
        return ZERO_RESOURCE
    end

    if length(resources) == 1
        return resources[1]
    end

    final_length = sum([length(x) for x in resources]) - length(resources) + 1
    f3 = convolve([res.p for res in resources])

    f_paddeds = Vector{Vector{T}}()
    c_paddeds = Vector{Vector{T}}()
    weights = Vector{Vector{T}}()

    for res in resources
        new_f_pad = [res.p; zeros(T, final_length - length(res))]
        new_c_pad = [res.c; zeros(T, final_length - length(res))]
        push!(f_paddeds, new_f_pad)
        push!(c_paddeds, new_c_pad)
        push!(weights, new_f_pad .* new_c_pad)
    end

    f_pads_ffted = [fft(x) for x in f_paddeds]
    ffts = Vector{Vector{ComplexF64}}()
    for i in eachindex(weights)
        # fft of weight * remaining rvs multiplied up
        push!(ffts, fft(weights[i]) .* reduce((x, y) -> x .* y, f_pads_ffted[Not(i)]))
    end

    sum_fft = reduce((x, y) -> x .+ y, ffts)
    weighted_cost = ifft(sum_fft)
    real_cost = [real(x) for x in weighted_cost]
    c3 = real_cost ./ f3

    return DiscreteResource(f3, c3)
end

#---------------------------------------
# Problem Definition
#---------------------------------------
"""
SRO problem consisting of a set of independent discrete `resources`,
a probability target `p_target` and a value target `v_target`.
"""
struct DiscreteProblem{T<:AbstractFloat}
    resources::Vector{DiscreteResource{T}}
    p_target::T
    v_target::Int64
end

#---------------------------------------
# Instances Definition
#---------------------------------------
"""
Set of concrete instances of an SRO `problem`.
The constructor rolls `n_instances` values using the given `rng` by the
distributions of the problem resources.
"""
struct DiscreteInstances{T<:AbstractFloat}
    problem::DiscreteProblem{T}
    values::Vector{Vector{Int64}}
    costs::Vector{Vector{T}}

    function DiscreteInstances(
        rng::AbstractRNG,
        problem::DiscreteProblem{T},
        n_instances::Int64,
    ) where {T<:AbstractFloat}
        values = Vector{Vector{Int64}}()
        costs = Vector{Vector{T}}()

        for res in problem.resources
            rands = rand(rng, n_instances)
            cumu = cdf(res)

            v = [findfirst(n -> n >= x, cumu) for x in rands]
            push!(values, v)

            c = [res.c[x] for x in v]
            push!(costs, c)
        end
        return new{T}(problem, values, costs)
    end

    function DiscreteInstances(problem::DiscreteProblem, n_instances::Int64)
        return DiscreteInstances(Xoshiro(), problem, n_instances)
    end
end

#---------------------------------------
# Solution Definition
#---------------------------------------
"""
Solution structure for a discrete SRO `problem`. Encapsulates different ways to give the solution:
* `resource_indices` - vector of indices of the optimal solution resources as in the `problem.resources` vector
* `resources` - vector of the optimal resources
* `cost` - expected costs of the solution resource set at the target value
"""
struct DiscreteSolution{T<:AbstractFloat}
    problem::DiscreteProblem{T}
    resource_indices::Vector{Int64}
    resources::Vector{DiscreteResource{T}}
    cost::T
end

#---------------------------------------
# Target Function for Optimization
#---------------------------------------
function sro_target_function(
    resources::Vector{DiscreteResource{T}},
    p_target::T,
    v_target::Int64,
)::T where {T}
    if isempty(resources)
        return Inf
    end

    sum_resource = add(resources)
    target_probabilities = [1.0; ccdf(sum_resource)[1:end-1]]
    target_index = v_target + 1

    # set is not feasible because of length
    if length(sum_resource) < target_index
        return Inf
    end

    # set is not feasible because of p_target
    if target_probabilities[target_index] < p_target
        return Inf
    end

    # set is feasible
    return sum_resource.c[target_index]
end
