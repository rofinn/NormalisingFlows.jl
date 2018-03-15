using Base.LinAlg.BLAS: trmv, trmm, trsv, trsm

@unionise begin

"""
    DiagAffine{T<:AbstractVector} <: Invertible

DiagAffine transform with parametes `α` and `log(β)`. Defines function `f(z) = α + diag(β) * z`.
"""
struct DiagAffine{T<:AbstractVector} <: Invertible
    α::T
    logβ::T
    function DiagAffine(α::T, logβ::T) where {T}
        @assert length(α) == length(logβ)
        return new{T}(α, logβ)
    end
end
DiagAffine(α::T, logβ::T) where T<:Real = DiagAffine([α], [logβ])
function DiagAffine(θ::Vector, D::Int)
    @assert length(θ) == nparams(DiagAffine, D)
    return DiagAffine(θ[1:D], θ[D+1:2D])
end
dim(t::DiagAffine) = length(t.α)

"""
    apply(f::DiagAffine, z)

Compute the affine transform of `z` specified by `f`.
"""
function apply(f::DiagAffine, z::AbstractVecOrMat{<:Real})
    @assert length(f.α) == size(z, 1)
    return f.α .+ exp.(f.logβ) .* z
end

"""
    invert(f::DiagAffine, y)

Apply the inverse transform of `f` to `y` such that `invert(f, apply(f, z)) ≈ z` and
`apply(f, invert(f, z)) ≈ z`.
"""
function invert(f::DiagAffine, y::AbstractVecOrMat{<:Real})
    @assert dim(f) == size(y, 1)
    return (y .- f.α) ./ exp.(f.logβ)
end

function logdetJ(f::DiagAffine, z::AbstractVector{<:Real})
    @assert dim(f) == length(z)
    return sum(f.logβ)
end
function logdetJ(f::DiagAffine, z::AbstractMatrix{<:Real})
    @assert dim(f) == size(z, 1)
    return sum(f.logβ) .* ones(size(z, 2))
end

end # @unionise

params(f::DiagAffine) = [f.α, f.logβ]
nparams(::Type{<:DiagAffine}, D::Int) = 2D

function identity_init!(θ::AbstractVector{<:Real}, ::Type{<:DiagAffine}, ::AbstractRNG, D::Int)
    @assert length(θ) == nparams(DiagAffine, D)
    return fill!(θ, zero(eltype(θ)))
end

@unionise begin

"""
    Affine{TL<:RealMat, TU<:RealMat, Td′<:RealVec, Tb<:RealVec} <: Invertible

A dense affine transform. Parameterised in terms of the LDU decomposition, yielding
O(D^2) application and inversion, and O(D) Jacobian log determinant computation.
"""
struct Affine{TL<:RealMat, TU<:RealMat, Td<:RealVec, Tb<:RealVec} <: Invertible
    L::TL
    U::TU
    logd::Td
    b::Tb
    function Affine(L::TL, U::TU, logd::Td, b::Tb) where {TL, TU, Td, Tb}
        @assert size(L, 1) == size(L, 2)
        @assert size(L) == size(U)
        @assert length(logd) == length(b)
        return new{TL, TU, Td, Tb}(L, U, logd, b)
    end
end

dim(a::Affine) = length(a.b)
nparams(::Type{<:Affine}, D::Int) = 2D * (D + 1)
function Affine(θ::AbstractVector, D::Int)
    @assert length(θ) == nparams(Affine, D)
    L = reshape(θ[1:D^2], D, D)
    U = reshape(θ[D^2+1:2 * D^2], D, D)
    logd = θ[2 * D^2 + 1:2 * D^2 + D]
    b = θ[2 * D^2 + D + 1:nparams(Affine, D)]
    return Affine(L, U, logd, b)
end

function apply(f::Affine, z::AbstractVector{<:Real})
    @assert dim(f) == length(z)
    D = Diagonal(exp.(f.logd))
    return trmv('L', 'N', 'U', f.L, D * trmv('U', 'N', 'U', f.U, z)) .+ f.b
end

function apply(f::Affine, Z::AbstractMatrix{<:Real})
    @assert dim(f) == size(Z, 1)
    D = Diagonal(exp.(f.logd))
    DUδ = D * trmm('L', 'U', 'N', 'U', one(eltype(Z)), f.U, Z)
    return trmm('L', 'L', 'N', 'U', one(eltype(Z)), f.L, DUδ) .+ f.b
end

function invert(f::Affine, z::AbstractVector{<:Real})
    @assert dim(f) == length(z)
    D, δ = Diagonal(exp.(f.logd)), z - f.b
    return trsv('U', 'N', 'U', f.U, D \ trsv('L', 'N', 'U', f.L, δ))
end

function invert(f::Affine, Z::AbstractMatrix{<:Real})
    @assert dim(f) == size(Z, 1)
    D, δ = Diagonal(exp.(f.logd)), Z .- f.b
    tmp = D \ trsm('L', 'L', 'N', 'U', one(eltype(Z)), f.L, δ)
    return trsm('L', 'U', 'N', 'U', one(eltype(Z)), f.U, tmp)
end

function logdetJ(f::Affine, z::AbstractVector{<:Real})
    @assert dim(f) == length(z)
    return sum(f.logd)
end

function logdetJ(f::Affine, Z::AbstractMatrix{<:Real})
    @assert dim(f) == size(Z, 1)
    return sum(f.logd) .* ones(size(Z, 2))
end

end # @unionise

identity_init!(θ::AbstractVector{<:Real}, ::Type{<:Affine}, ::AbstractRNG, D::Int) =
    fill!(θ, zero(eltype(θ)))
