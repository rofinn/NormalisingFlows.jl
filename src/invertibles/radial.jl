@unionise begin

"""
    Radial{Tα<:Real, Tβ<:Real, Tγ<:RealVec} <: Invertible

A Radial Flow as parameterised in [1].

[1] - Trippe, B. L., & Turner, R. E. (2018). Conditional Density Estimation with Bayesian
    Normalising Flows.
"""
struct Radial{Tα<:Real, Tβ<:Real, Tγ<:RealVec} <: Invertible
    α::Tα
    β::Tβ
    γ::Tγ
    Radial(α::Tα, β::Tβ, γ::Tγ) where {Tα, Tβ, Tγ} = 
        new{Tα, Tβ, Tγ}(softplus(α), exp(β) - 1, γ)        
end

dim(r::Radial) = length(r.γ)
params(r::Radial) = [r.α, r.β, r.γ]
nparams(::Type{<:Radial}, D::Int) = D + 2
function Radial(θ::RealVec, D::Int)
    @assert length(θ) == nparams(Radial, D)
    return Radial(θ[1], θ[2], θ[3:D+2])
end


"""
    apply(r::Radial, z)

Compute the Radial transform of `z` specified by `f`.
"""
function apply(f::Radial, z::AbstractVecOrMat{<:Real})
    @assert dim(f) == size(z, 1)
    δ = z .- f.γ
    c = (f.α * f.β) ./ (f.α .+ sqrt.(mapreducedim(abs2, +, δ, 1)))
    return z .+ c .* δ
end

function logdetJ(f::Radial, z::AbstractVector{<:Real})
    @assert dim(f) == length(z)
    return log(abs(1 + f.α^2 * f.β / (f.α + norm(z - f.γ))^2) + eps())
end

function logdetJ(f::Radial, z::AbstractMatrix{<:Real})
    @assert dim(f) == size(z, 1)
    c, δnorms = f.α^2 * f.β, reshape(sqrt.(mapreducedim(abs2, +, z .- f.γ, 1)), size(z, 2))
    return log.(abs.(1 .+ c ./ (f.α .+ δnorms).^2) .+ eps())
end

end # @unionise

function identity_init!(θ::AbstractVector{<:Real}, ::Type{<:Radial}, ::AbstractRNG, D::Int)
    @assert length(θ) == nparams(Radial, D)
    θ[1] = log(expm1(1))
    θ[2] = zero(eltype(θ))
    fill!(view(θ, 3:D+2), zero(eltype(θ)))
    return θ
end
