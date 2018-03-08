@unionise begin

"""
    Affine{T<:AbstractVector} <: Invertible

Affine transform with parametes `α` and `log(β)`. Defines function `f(z) = α + diag(β) * z`.
"""
struct Affine{T<:AbstractVector} <: Invertible
    α::T
    logβ::T
    function Affine(α::T, logβ::T) where {T}
        @assert length(α) == length(logβ)
        return new{T}(α, logβ)
    end
end
Affine(α::T, logβ::T) where T<:Real = Affine([α], [logβ])
dim(t::Affine) = length(t.α)

"""
    apply(f::Affine, z)

Compute the affine transform of `z` specified by `f`.
"""
function apply(f::Affine, z::AbstractVecOrMat{<:Real})
    @assert length(f.α) == size(z, 1)
    return f.α .+ exp.(f.logβ) .* z
end

"""
    invert(f::Affine, y)

Apply the inverse transform of `f` to `y` such that `invert(f, apply(f, z)) ≈ z` and
`apply(f, invert(f, z)) ≈ z`.
"""
function invert(f::Affine, y::AbstractVecOrMat{<:Real})
    @assert dim(f) == size(y, 1)
    return (y .- f.α) ./ exp.(f.logβ)
end

function logdetJ(f::Affine, z::AbstractVector{<:Real})
    @assert dim(f) == length(z)
    return sum(f.logβ)
end
function logdetJ(f::Affine, z::AbstractMatrix{<:Real})
    @assert dim(f) == size(z, 1)
    return sum(f.logβ) .* ones(size(z, 2))
end

end # @unionise

"""
    Affine(::typeof(naive_init), rng::AbstractRNG, D::Int, tape::Tape=nothing)

Produce an affine transform whose dimensionality is `D`, in which all parameters are sampled
independently from a standard nomal.
"""
Affine(::typeof(naive_init), rng::AbstractRNG, D::Int) =
    Affine(randn(rng, D), randn(rng, D))
function Affine(::typeof(naive_init), rng::AbstractRNG, D::Int, tape::Tape)
    α, logβ = Leaf.(tape, [randn(rng, D), randn(rng, D)])
    return (Affine(α, logβ), [α, logβ])
end

"""
    Affine(::typeof(identity_init), D::Int)

An Affine transform whose domain is `D`-dimensional. Parameters chosen to make the function
be the identity.
"""
Affine(::typeof(identity_init), D::Int) = Affine(zeros(D), zeros(D))
function Affine(::typeof(identity_init), D::Int, tape::Tape)
    α, logβ = Leaf.(tape, [zeros(D), zeros(D)])
    return (Affine(α, logβ), [α, logβ])
end
