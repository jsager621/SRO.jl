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

    function DiscreteResource(prob::Vector{T}, cost::Vector{T}) where {T<:AbstractFloat}
        if length(prob) != length(cost)
            throw(ArgumentError("Probabililty and cost vector must have equal length!"))
        end

        if !isapprox(sum(prob), 1.0)
            throw(ArgumentError("Probabilities must add up to 1!"))
        end

        max_value = length(prob) - 1

        p = OffsetVector(prob, 0:max_value)
        c = OffsetVector(cost, 0:max_value)
        v = OffsetVector(collect(0:max_value), 0:max_value)

        return new{T}(max_value, p, c, v)
    end
end

function Base.getindex(r::DiscreteResource, k)
    return (r.p[k], r.c[k], r.v[k])
end

function Base.length(r::DiscreteResource)
    return length(r.p)
end