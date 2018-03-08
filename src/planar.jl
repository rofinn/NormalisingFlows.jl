@unionise begin

const tanh′ = x->1 - tanh(x)^2

"""
    struct Planar{Tu<:AbstractVector{<:Real}, Tw<:AbstractVector{<:Real}, Tb<:Real, Th, Th′} <: Invertible

A Planar flow as described in [1].

[1] - Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows.
"""
struct Planar{
    Tu<:AbstractVector{<:Real},
    Tû<:AbstractVector{<:Real},
    Tw<:AbstractVector{<:Real},
    Tb<:AbstractVector{<:Real},
    Th,
    Th′,
} <: Invertible
    u::Tu
    û::Tû
    w::Tw
    b::Tb
    h::Th
    h′::Th′
    function Planar(u::Tu, w::Tw, b::Tb, h::Th, h′::Th′) where {Tu, Tw, Tb, Th, Th′}
        @assert length(u) == length(w)
        wu = w'u
        û = u + (log1p(exp(wu)) - 1 - wu) * w / (sum(abs2, w) + eps())
        return new{Tu, typeof(û), Tw, Tb, Th, Th′}(u, û, w, b, h, h′)
    end
end
Planar(u::Real, w::Real, b::Real, h, h′) = Planar([u], [w], [b], h, h′)
Planar(u::AbstractVector{<:Real}, w::AbstractVector{<:Real}, b::Real, h, h′) =
    Planar(u, w, [b], h, h′)
Planar(u, w, b) = Planar(u, w, b, tanh, tanh′)
function Planar(θ::Vector, D::Int)
    @assert length(θ) == nparams(Planar, D)
    return Planar(θ[1:D], θ[D+1:2D], θ[2D+1:2D+1])
end
dim(p::Planar) = length(p.u)
params(p::Planar) = [p.u, p.w, p.b]
nparams(::Type{Planar}, D::Int) = 2D + 1

"""
    apply(f::Planar, z)

Compute the planar transform of `z` specified by `f`.
"""
function apply(f::Planar, z)
    @assert dim(f) == size(z, 1)
    return z .+ f.û .* f.h.(f.w'z .+ f.b)
end

function logdetJ(f::Planar, z::AbstractVector{<:Real})
    @assert dim(f) == length(z)
    return log(abs(1 + f.h′(z'f.w + f.b[1]) * f.û'f.w) + eps())
end

function logdetJ(f::Planar, z::AbstractMatrix{<:Real})
    @assert dim(f) == size(z, 1)
    return log.(abs.(1 .+ f.h′.(z'f.w .+ f.b) .* f.û'f.w) .+ eps())
end

end # @unionise

"""
    Planar(::typeof(naive_init), rng::AbstractRNG, D::Int)

A Planar transform whose domain is `D`-dimensional. Parameters iid from a standard Normal,
`tanh` nonlinearity used.
"""
Planar(::typeof(naive_init), rng::AbstractRNG, D::Int) =
    Planar(randn(rng, D), randn(rng, D), randn(rng), tanh, tanh′)
function Planar(::typeof(naive_init), rng::AbstractRNG, D::Int, tape::Tape)
    u, w, b = Leaf.(tape, [randn(rng, D), randn(rng, D), randn(rng, 1)])
    return (Planar(u, w, b, tanh, tanh′), [u, w, b])
end

"""
    Planar(::typeof(identity_init), rng::AbstractRNG, D::Int)

A Planar transform whose domain is `D`-dimensional. Parameters chosen to make the function
be the identity.
"""
Planar(::typeof(identity_init), D::Int) = Planar(zeros(D), zeros(D), [2.0], tanh, tanh′)
function Planar(::typeof(identity_init), D::Int, tape::Tape)
    u, w, b = Leaf.(tape, [zeros(D), zeros(D), [2.0]])
    return (Planar(u, w, b, tanh, tanh′), [u, w, b])
end
