using OffsetArrays
using GenericFFT

"""
Discrete and independent SRO resource model. 
All values are defined from 0 to the maximum value. 
The value vector __v__ is mapped to the indices of the probability vector.
The probability vector __p__ and cost vector __c__ must be of equal length.
"""
struct DiscreteResource{T<:AbstractFloat}
    max_value::Int64
    p::OffsetVector{T}
    c::OffsetVector{T}
    v::OffsetVector{T}

    function DiscreteResource(prob::Vector{T}, cost::Vector{T})
        max_value = length(prob) - 1
        @assert length(prob) == length(costs) "Probabililty and cost vector must have equal length!"
        @assert isapprox(sum(prob), 1.0) "Probabilities must add up to 1!"

        p = OffsetVector(prob, 0:max_value)
        c = OffsetVector(cost, 0:max_value)
        v = OffsetVector(collect(0:max_value), 0:max_value)

        return new(max_value, p, c, v)
    end
end