@unionise begin

"""
    struct Planar{T<:VecOrReal, Tb<:Real, Th} <: Invertible

A Planar flow as described in [1].

[1] - Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows.
"""
struct Planar{T<:VecOrReal, Tb<:VecOrReal, Th, Th′} <: Invertible
    u::T
    w::T
    b::Tb
    h::Th
    h′::Th′
    function Planar(u::T, w::T, b::Tb, h::Th, h′::Th′) where {T, Tb, Th, Th′}
        @assert length(u) == length(w)
        wu = w'u
        û = u + (log1p(exp(wu)) - 1 - wu) * w / sum(abs2, w)
        return new{T, Tb, Th, Th′}(û, w, b, h, h′)
    end
end
dim(p::Planar) = length(p.u)

"""
    apply(f::Planar, z)

Compute the planar transform of `z` specified by `f`.
"""
function apply(f::Planar, z)
    @assert dim(f) == size(z, 1)
    return z .+ f.u .* f.h.(f.w'z .+ f.b)
end

function logdetJ(f::Planar, z)
    @assert dim(f) == size(z, 1)
    return sum(log.(abs.(1 .+ f.h′.(z'f.w .+ f.b) .* f.u'f.w)))
end

end # @unionise
