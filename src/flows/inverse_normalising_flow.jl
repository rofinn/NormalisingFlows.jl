"""
    InverseNormalisingFlow{T<:Vector{<:Invertible}}

A normalising flow that operates in reverse. It is thus efficient to compute the log
density of an observation, but not (necessarily) efficent to compute the 
"""
struct InverseNormalisingFlow{T<:Vector{<:Invertible}} <: Flow
    p0::Any
    transforms::T
    function InverseNormalisingFlow(p0, transforms::T) where T<:Vector{<:Invertible}
        @assert all(dim(p0) .== dim.(transforms))
        return new{T}(p0, transforms)
    end
end
const INF = InverseNormalisingFlow

"""
    rand(rng::AbstractRNG, d::INF)

Sample from a INF `d`. Will only work if each transform in `d` has its
`invert` function defined.
"""
function rand(rng::AbstractRNG, d::INF, N::Int=1)
    y = rand(rng, d.p0, N)
    for f in reverse(d.transforms)
        y = invert(f, y)
    end
    return y
end

"""
    logpdf(d::INF, y::RealOrVecOrMat)

Compute the log pdf of an observation `y` of the INF `d`.
"""
function logpdf(d::INF, y::AbstractVecOrMat{<:Real})
    l = zero(eltype(y))
    for k in 1:length(d.transforms)
        l += sum(logdetJ(d.transforms[k], y))
        y = apply(d.transforms[k], y)
    end
    return l + sum(logpdf(d.p0, y))
end

for init in [:naive_init!, :identity_init!]
    @eval begin
    """
        naive_init!(
            θ::AbstractVector{<:Real},
            ::Type{<:INF},
            rng::AbstactRNG,
            D::Int,
            ctors::AbstractVector,
        )

    Fill `θ` with initialisations appropriate for each component of an INF using only
    information available prior to construction.
    """
    function $init(
        θ::AbstractVector{<:Real},
        ::Type{<:INF},
        rng::AbstractRNG,
        D::Int,
        ctors::AbstractVector,
    )
        @assert all(issubtype.(ctors, Invertible))
        @assert length(θ) == nparams(INF, D, ctors)
        pos = 1
        for p in eachindex(ctors)
            Δ = nparams(ctors[p], D)
            $init(view(θ, pos:pos+Δ-1), ctors[p], rng, D)
            pos += Δ
        end
        return θ
    end
    end
end

"""
    INF(::typeof(naive_init), rng::AbstractRNG, D::Int, ctors::AbstractVector)

Initialise an `InverseNormalisingFlow` with the `naive_init`s for each `Invertible`, and
a `DiagonalStandardNormal` as the base distribution.
"""
INF(::typeof(naive_init), rng::AbstractRNG, D::Int, ctors::AbstractVector) =
    INF(DiagStdNormal(D), naive_init(INF, rng, D, ctors), ctors)

"""
    INF(::typeof(identity_init), rng::AbstractRNG, D::Int, ctors::AbstractVector)

Initialise an `InverseNormalisingFlow` with the `identity_init`s for each `Invertible`, and
a `DiagonalStandardNormal` as the base distribution.
"""
INF(::typeof(identity_init), rng::AbstractRNG, D::Int, ctors::AbstractVector) =
    INF(DiagStdNormal(D), identity_init(INF, rng, D, ctors), ctors)
