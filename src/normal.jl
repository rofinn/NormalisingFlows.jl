import Base: rand, rand!, broadcast

@unionise begin

"""
    Normal{T}

The independent Normal distribution, parameterised in terms of its log standard deviation.
Can be paramterised by either scalars or vectors parameters.
"""
struct Normal{T<:VecOrReal}
    μ::T
    logσ::T
    σ::T
    function Normal(μ::T, logσ::T) where T
        @assert size(μ, 1) == size(logσ, 1)
        return new{T}(μ, logσ, exp.(logσ))
    end
end
dim(d::Normal{<:Real}) = 1
dim(d::Normal{<:VecOrReal}) = length(d.μ)

"""
    rand(rng::AbstractRNG, d::Normal, N::Int=1)

Draw `N` samples from `d` using RNG `rng`. Returned as a `dim(d)` x `N` matrix.
"""
rand(rng::AbstractRNG, d::Normal) = rand!(rng, d, Vector{Float64}(dim(d)))
rand(rng::AbstractRNG, d::Normal, N::Int) = rand!(rng, d, Matrix{Float64}(dim(d), N))

"""
    rand!(rng::AbstractRNG, d::Normal, A::Union{Real, AbstractVecOrMat})

Fill columns of `A` with samples from `d` using RNG `rng`.
"""
function rand!(rng::AbstractRNG, d::Normal, A::Union{Real, AbstractVecOrMat})
    @assert dim(d) == size(A, 1)
    return broadcast!((μ, σ)->μ + σ * randn(rng), A, d.μ, d.σ)
end

"""
    lpdf(d::Normal, x::VecOrReal)

Compute the log probability of `x` under `d`.
"""
lpdf(d::Normal, x::VecOrReal) =
    -0.5 * (dim(d) * log(2π) + 2 * sum(d.logσ) + sum(abs2, (x .- d.μ) ./ d.σ))

"""
    lpdf(d::Normal, x::AbstractMatrix{<:Real})

Compute the log probability of observing each column of `x` jointly under `d`.
"""
function lpdf(d::Normal, X::AbstractMatrix{<:Real})
    @assert dim(d) == size(X, 1)
    D, N = size(X, 1), size(X, 2)
    return -0.5 * N * D * log(2π) - N * sum(d.logσ) - 0.5 * sum(abs2, (X .- d.μ) ./ d.σ)
end

"""
    broadcast(lpdf, d::Normal, X::Union{Real, AbstractVecOrMat})

Compute the probability of each column of `x` under `d`.
"""
function broadcast(lpdf, d::Normal, X::Union{Real, AbstractVecOrMat})
    @assert dim(d) == size(X, 1)
    D, N = size(X, 1), size(X, 2)
    return -0.5 * D * log(2π) - sum(d.logσ) - 0.5 * sum(abs2, (X .- d.μ) ./ d.σ, 1)
end

end # @unionise
