"""
    abstract type Flow

Supertype for NormalisingFlow and InverseNormalisingFlow.
"""
abstract type Flow end

"""
    naive_init(T::Type{<:Flow}, rng::AbstractRNG, D::Int, invertibles::Vector{<:Invertible})

Get parameters to create a valid `Flow`, doesn't matter how performant it is.
"""
naive_init(T::Type{<:Flow}, rng::AbstractRNG, D::Int, invertibles::AbstractVector) =
    naive_init!(Vector{Float64}(nparams(T, D, invertibles)), T, rng, D, invertibles)

"""
    identity_init(T::Type{<:Flow}, D::Int, invertibles::Vector{<:Invertible})

Get parameters to create a `Flow`, whose transform is the identity function.
"""
identity_init(T::Type{<:Flow}, rng::AbstractRNG, D::Int, invertibles::AbstractVector) =
    identity_init!(Vector{Float64}(nparams(T, D, invertibles)), T, rng, D, invertibles)

# include("flows/normalising_flow.jl")
include("flows/inverse_normalising_flow.jl")

######################## Functions generic to all flows. ########################

dim(d::Flow) = dim(d.p0)
params(d::Flow) = vcat(params.(d.transforms)...)

# Determine number of parameters before instantiating a Flow.
nparams(::Type{<:Flow}, D::Int, ctors::AbstractVector) = sum(nparams.(ctors, D))

# Determine number of parameters using an instantiation of a Flow.
nparams(d::Flow) = sum(nparams.(d.transforms))

for flow in [:NormalisingFlow, :InverseNormalisingFlow]
    @eval @unionise function $flow(p0, θ::Vector{<:Real}, ctors::AbstractVector)
        @assert all(issubtype.(ctors, Invertible))
        @assert length(θ) == sum(nparams.(ctors, dim(p0)))
        pos, transforms = 1, Vector{Invertible}(length(ctors))
        for t in eachindex(ctors)
            Δ = nparams(ctors[t], dim(p0))
            transforms[t] = ctors[t](θ[pos:pos+Δ-1], dim(p0))
            pos += Δ
        end
        return $flow(p0, transforms)
    end
end
