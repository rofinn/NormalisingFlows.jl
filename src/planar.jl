@unionise begin

"""
    struct Planar{Tu<:AbstractVector{<:Real}, Tw<:AbstractVector{<:Real}, Tb<:Real, Th, Th′} <: Invertible

A Planar flow as described in [1].

[1] - Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows.
"""
struct Planar{Tu<:AbstractVector{<:Real}, Tw<:AbstractVector{<:Real}, Tb<:Real, Th, Th′} <: Invertible
    u::Tu
    w::Tw
    b::Tb
    h::Th
    h′::Th′
    function Planar(u, w::Tw, b::Tb, h::Th, h′::Th′) where {Tw, Tb, Th, Th′}
        @assert length(u) == length(w)
        wu = w'u
        û = sum(abs2, w) != 0 ?
            u + (log1p(exp(wu)) - 1 - wu) * w / sum(abs2, w) :
            zeros(size(u))
        return new{typeof(û), Tw, Tb, Th, Th′}(û, w, b, h, h′)
    end
end
Planar(u::Real, w::Real, b::Real, h, h′) = Planar([u], [w], b, h, h′)
dim(p::Planar) = length(p.u)

"""
    apply(f::Planar, z)

Compute the planar transform of `z` specified by `f`.
"""
function apply(f::Planar, z)
    @assert dim(f) == size(z, 1)
    return z .+ f.u .* f.h.(f.w'z .+ f.b)
end

function logdetJ(f::Planar, z::AbstractVector{<:Real})
    @assert dim(f) == length(z)
    return log(abs(1 + f.h′(z'f.w + f.b) * f.u'f.w))
end

function logdetJ(f::Planar, z)
    @assert dim(f) == size(z, 1)
    return log.(abs.(1 .+ f.h′.(z'f.w .+ f.b) .* f.u'f.w))
end

end # @unionise

const tanh′ = x->1 - tanh(x)^2

"""
    Planar(::typeof(naive_init), rng::AbstractRNG, D::Int)

A Planar transform whose domain is `D`-dimensional. Parameters iid from a standard Normal,
`tanh` nonlinearity used.
"""
Planar(::typeof(naive_init), rng::AbstractRNG, D::Int) =
    Planar(randn(rng, D), randn(rng, D), randn(rng), tanh, tanh′)
function Planar(::typeof(naive_init), rng::AbstractRNG, D::Int, tape::Tape)
    u, w, b = Leaf.(tape, [randn(rng, D), randn(rng, D), randn(rng)])
    return (Planar(u, w, b, tanh, tanh′), [u, w, b])
end

"""
    Planar(::typeof(identity_init), rng::AbstractRNG, D::Int)

A Planar transform whose domain is `D`-dimensional. Parameters chosen to make the function
be the identity.
"""
Planar(::typeof(identity_init), D::Int) = Planar(zeros(D), zeros(D), 2.0, tanh, tanh′)
function Planar(::typeof(identity_init), D::Int, tape::Tape)
    u, w, b = Leaf.(tape, [zeros(D), zeros(D), 2.0])
    return (Planar(u, w, b, tanh, tanh′), [u, w, b])
end
