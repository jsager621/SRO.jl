using OffsetArrays
import OffsetArrays: no_offset_view
using GenericFFT
using InvertedIndices

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

function cdf(r::DiscreteResource{T})::OffsetVector{T} where T
    return OffsetVector(cumsum(r.p), 0:length(r)-1)
end

function ccdf(r::DiscreteResource{T})::OffsetVector{T} where T
    return [1 - n for n in cdf(r)]
end

function convolve(a::OffsetVector{T, Vector{T}}, b::OffsetVector{T, Vector{T}})::OffsetVector{T, Vector{T}} where {T <: AbstractFloat}
    n = length(a)
    m = length(b)

    a_pad = [no_offset_view(a); zeros(m - 1)]
    b_pad = [no_offset_view(b); zeros(n - 1)]

    raw_convolve = ifft(fft(a_pad) .* fft(b_pad))
    reals = [real(x) for x in raw_convolve]
    filtered = [x < 0 ? T(0.0) : x for x in reals]

    return OffsetVector(filtered, 0:length(filtered)-1)
end

function convolve(vs::Vector{OffsetVector{T, Vector{T}}})::OffsetVector{T, Vector{T}} where {T <: AbstractFloat}
    final_length = sum([length(x) for x in vs]) - length(vs) + 1
    paddeds = Vector{Vector{T}}()
    for v in vs
        push!(paddeds, [no_offset_view(v); zeros(T, final_length - length(v))])
    end

    ffts = [fft(x) for x in paddeds]
    raw_convolve = ifft(reduce((x,y) -> x .* y, ffts))
    reals = [real(x) for x in raw_convolve]
    filtered = [x < 0 ? T(0.0) : x for x in reals]

    return OffsetVector(filtered, 0:length(filtered)-1)
end

function add(a::DiscreteResource{T}, b::DiscreteResource{T})::DiscreteResource{T} where T
    f3 = no_offset_view(convolve(a.p, b.p))
    
    n = length(a)
    m = length(b)

    f1_pad = [no_offset_view(a.p); zeros(m - 1)]
    f2_pad = [no_offset_view(b.p); zeros(n - 1)]

    c1_pad = [no_offset_view(a.c); zeros(m - 1)]
    c2_pad = [no_offset_view(b.c); zeros(n - 1)]

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

function add(resources::Vector{DiscreteResource{T}})::DiscreteResource{T} where T
    final_length = sum([length(x) for x in resources]) - length(resources) + 1
    f3 = no_offset_view(convolve([res.p for res in resources]))

    f_paddeds = Vector{Vector{BigFloat}}()
    c_paddeds = Vector{Vector{BigFloat}}()
    weights = Vector{Vector{T}}()

    for res in resources
        new_f_pad = [no_offset_view(res.p); zeros(T, final_length - length(res))]
        new_c_pad = [no_offset_view(res.c); zeros(T, final_length - length(res))]
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