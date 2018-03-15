"""
    abstract type Invertible

Parent type for all invertible transforms.
"""
abstract type Invertible end

"""
    naive_init(T::Type{<:Invertible}, rng::AbstractRNG, D::Int)

Get parameters for some arbitrary valid initialisation for a given `Invertible` type. This
initialisation need not have good performance in any sense, but should not return `Inf`s or
`NaN`s when called with `apply`, `invert`, or `logdetJ`.
"""
naive_init(T::Type{<:Invertible}, rng::AbstractRNG, D::Int) =
    naive_init!(Vector{Float64}(nparams(T, D)), T, rng, D)

"""
    identity_init(T::Type{<:Invertible}, D::Int)

Get parameters that would create an `Invertible` corresponding to the identity function.
"""
identity_init(T::Type{<:Invertible}, rng::AbstractRNG, D::Int) =
    identity_init!(Vector{Float64}(nparams(T, D)), T, rng, D)

# Default naive_init! for all `Invertible`s.
function naive_init!(θ::RealVec, T::Type{<:Invertible}, rng::AbstractRNG, D::Int)
    @assert length(θ) == nparams(T, D)
    return randn!(rng, θ)
end

# Invertible-specific code.
include("invertibles/affine.jl")
include("invertibles/planar.jl")
include("invertibles/radial.jl")

# Constructor methods for each Invertible type with each initialiser type. This list should
# be extended if new Invertibles or initialisers are created.
for TInv in [:DiagAffine, :Affine, :Planar, :Radial]
    for init in [:naive_init, :identity_init]
        @eval $TInv(::typeof($init), rng::AbstractRNG, D::Int) =
            $TInv($init($TInv, rng, D), D)
    end
end

"""
    nparams(f::T) where T<:Invertible

Get the number of parameters of a instantiated `Invertible`.
"""
@inline nparams(f::T) where T<:Invertible = nparams(T, dim(f))
