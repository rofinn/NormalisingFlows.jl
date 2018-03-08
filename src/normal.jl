"""
    DiagonalStandardNormal <: AbstactMvNormal

Multivariate Normal distribution with zero mean vector and identity covariance.
"""
struct DiagonalStandardNormal <: AbstractMvNormal
    D::Int
end

const DiagStdNormal = DiagonalStandardNormal

dim(d::DiagonalStandardNormal) = d.D

rand(rng::AbstractRNG, d::DiagonalStandardNormal) = randn(rng, dim(d))
rand(rng::AbstractRNG, d::DiagonalStandardNormal, N::Int) = randn(rng, dim(d), N)

@unionise function logpdf(d::DiagonalStandardNormal, x::AbstractVector{<:Real})
    @assert dim(d) == length(x)
    return -0.5 * (dim(d) * log(2π) + sum(abs2, x))
end
@unionise function logpdf(d::DiagonalStandardNormal, X::AbstractMatrix{<:Real})
    @assert dim(d) == size(X, 1)
    return -0.5 .* ((dim(d) * log(2π)) .+ reshape(mapreducedim(abs2, +, X, 1), size(X, 2)))
end
