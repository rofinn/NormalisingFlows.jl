@unionise begin

const tanh′ = x->1 - tanh(x)^2

"""
    BatchPlanar{Tu<:RealMat, Tû<:RealMat, Tw<:RealMat, Tb<:RealMat, Th, Th′} <: Invertible

A batch version of the Planar flow as described in [1].

[1] - Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows.
"""
struct BatchPlanar{TU<:RealMat, TÛ<:RealMat, TW<:RealMat, Tb<:RealMat, Th, Th′} <: Invertible
    U::TU
    Û::TÛ
    W::TW
    b::Tb
    h::Th
    h′::Th′
    function BatchPlanar(u::Tu, w::Tw, b::Tb, h::Th, h′::Th′) where {Tu, Tw, Tb, Th, Th′}
        @assert size(U) == size(W) && size(U, 2) == size(b, 2) && size(b, 1) == 1
        WU = mapreducedim(identity, +, W .* U, 1)
        Û = U .+ (log1p.(exp.(WU)) .- 1 .- WU) ./ (mapreducedim(abs2, +, W, 1) .+ eps())
        return new{TU, typeof(Û), TW, Tb, Th, Th′}(U, Û, W, b, h, h′)
    end
end

dim(p::BatchPlanar) = size(p.U, 1)
params(p::BatchPlanar) = [p.U, p.W, p.b]
nparams(p::BatchPlanar) = 2 * length(p.U) + size(p.U, 2)
nparams(::Type{<:BatchPlanar}, D::Int, P::Int) = (2D + 1) * P

BatchPlanar(U::RealMat, W::RealMat, b::RealMat) = Planar(U, W, b, tanh, tanh′)
function BatchPlanar(θ::Vector, D::Int, P::Int)
    @assert length(θ) == nparams(Planar, D, P)
    DP = D * P
    return Planar(θ[1:DP], θ[DP+1:2DP], θ[2DP + 1:2DP + P])
end

"""
    apply(f::Planar, z)

Compute the planar transform of `z` specified by `f`.
"""
function apply(f::Planar, z::AbstractVecOrMat{<:Real})
    @assert dim(f) == size(z, 1)
    return z .+ mapreducedim(identity, +. f.Û .* f.h.(z'f.W .+ f.b))
end

function logdetJ(f::Planar, z::AbstractVector{<:Real})
    @assert dim(f) == length(z)
    ÛW = mapreducedim(ident, +, f.Û .* f.W, 1)
    return sum(log.(abs.(1 .+ f.h′.(z'f.W .+ f.b) .* ÛW) .+ eps()))
end

function logdetJ(f::Planar, z::AbstractMatrix{<:Real})
    @assert dim(f) == size(z, 1)
    return sum(log.(abs.(1 .+ f.h′.(z'f.w .+ f.b) .* f.û'f.w) .+ eps()), 1)
end

end # @unionise

function identity_init!(θ::AbstractVector{<:Real}, ::Type{<:Planar}, ::AbstractRNG, D::Int)
    @assert length(θ) == nparams(Planar, D)
    fill!(view(θ, 1:nparams(Planar, D)-1), zero(eltype(θ)))
    θ[end] = 2 * one(eltype(θ))
    return θ
end
