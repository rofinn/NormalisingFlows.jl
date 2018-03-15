@unionise begin

const tanh′ = x->1 - tanh(x)^2

"""
    Planar{Tu<:AbstractVector{<:Real}, Tw<:AbstractVector{<:Real}, Tb<:Real, Th, Th′} <: Invertible

A Planar flow as described in [1].

[1] - Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows.
"""
struct Planar{Tu<:RealVec, Tû<:RealVec, Tw<:RealVec, Tb<:Real, Th, Th′} <: Invertible
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
Planar(u::Real, w::Real, b::Real, h, h′) = Planar([u], [w], b, h, h′)
Planar(u, w, b) = Planar(u, w, b, tanh, tanh′)
function Planar(θ::Vector, D::Int)
    @assert length(θ) == nparams(Planar, D)
    return Planar(θ[1:D], θ[D+1:2D], θ[2D+1])
end
dim(p::Planar) = length(p.u)
params(p::Planar) = [p.u, p.w, p.b]
nparams(::Type{<:Planar}, D::Int) = 2D + 1

"""
    apply(f::Planar, z)

Compute the planar transform of `z` specified by `f`.
"""
function apply(f::Planar, z::AbstractVecOrMat{<:Real})
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

function identity_init!(θ::AbstractVector{<:Real}, ::Type{<:Planar}, ::AbstractRNG, D::Int)
    @assert length(θ) == nparams(Planar, D)
    fill!(view(θ, 1:nparams(Planar, D)-1), zero(eltype(θ)))
    θ[end] = 2 * one(eltype(θ))
    return θ
end

# Invert the Planar transform using a root finding method.
function invert(f::Planar, y::AbstractVector{<:Real})
    @assert dim(f) == length(y)
    a, c = f.w'y, f.w'f.û
    try
        α = fzero(α->α + c * tanh(α + f.b) - a, 0.0; order=2)
        return y - f.û .* tanh(α + f.b)
    catch err
        @show err, f.b, a, c
        throw(err)
    end
end

# Batch-inversion method to conform with the defined interface.
function invert(f::Planar, Y::AbstractMatrix{<:Real})
    @assert dim(f) == size(Y, 1)
    Z = similar(Y)
    for n in 1:size(Y, 2)
        Z[:, n] = invert(f, view(Y, :, n))
    end
    return Z
end
