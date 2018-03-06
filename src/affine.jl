@unionise begin

"""
    Affine{Tα, Tlogβ} <: Invertible

Affine transform with parametes `α` and `log(β)`. Defines function `f(z) = α + diag(β) * z`.
"""
struct Affine{T<:VecOrReal} <: Invertible
    α::T
    logβ::T
    function Affine(α::T, logβ::T) where {T}
        @assert length(α) == length(logβ)
        return new{T}(α, logβ)
    end
end
dim(t::Affine) = length(t.α)

"""
    apply(f::Affine, z)

Compute the affine transform of `z` specified by `f`.
"""
function apply(f::Affine, z)
    @assert length(f.α) == size(z, 1)
    return f.α .+ exp.(f.logβ) .* z
end

"""
    invert(f::Affine, y)

Apply the inverse transform of `f` to `y` such that `invert(f, apply(f, z)) ≈ z` and
`apply(f, invert(f, z)) ≈ z`.
"""
function invert(f::Affine, y)
    @assert dim(f) == size(y, 1)
    return (y .- f.α) ./ exp.(f.logβ)
end

# Have to specialise these methods for scalar vs vector due to Nabla bug.
function logdetJ(f::Affine{<:Real}, z)
    @assert dim(f) == size(z, 1)
    return f.logβ * size(z, 2)
end
function logdetJ(f::Affine, z)
    @assert dim(f) == size(z, 1)
    return sum(f.logβ) * size(z, 2)
end

end # @unionise
