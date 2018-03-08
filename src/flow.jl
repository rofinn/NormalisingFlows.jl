"""
    InverseNormalisingFlow{T<:Vector{<:Invertible}}

A normalising flow that operates in reverse. It is thus efficient to compute the log
density of an observation, but not (necessarily) efficent to compute the 
"""
struct InverseNormalisingFlow{T<:Vector{<:Invertible}}
    p0::Any
    transforms::T
    function InverseNormalisingFlow(p0, transforms::T) where T<:Vector{<:Invertible}
        @assert all(dim(p0) .== dim.(transforms))
        return new{T}(p0, transforms)
    end
end
dim(d::InverseNormalisingFlow) = dim(d.p0)
params(d::InverseNormalisingFlow) = vcat(params.(d.transforms)...)
nparams(d::InverseNormalisingFlow) = sum(nparams.(d.transforms))
starts(d::InverseNormalisingFlow) = ends(d) .- nparams.(d.transforms) .+ 1
ends(d::InverseNormalisingFlow) = cumsum(nparams.(d.transforms))

"""
    InverseNormalisingFlow(p0, ctors, θ::Vector{<:Real})

Construct an InverseNormalisingFlow from a base distribution `p0`, collection of
transform constructors `ctors`, and a vector of parameters `θ`.
"""
@unionise function InverseNormalisingFlow(p0, ctors, θ::Vector{<:Real})
    D = dim(p0)
    @assert length(θ) == sum(nparams.(ctors, D))
    pos, transforms = 1, Vector{Invertible}(length(ctors))
    for t in eachindex(ctors)
        Δ = nparams(ctors[t], D)
        transforms[t] = ctors[t](θ[pos:pos+Δ-1], D)
        pos += Δ
    end
    return InverseNormalisingFlow(p0, transforms)
end

"""
    rand(rng::AbstractRNG, d::InverseNormalisingFlow)

Sample from a InverseNormalisingFlow `d`. Will only work if each transform in `d` has its
`invert` function defined.
"""
function rand(rng::AbstractRNG, d::InverseNormalisingFlow, N::Int=1)
    y = rand(rng, d.p0, N)
    for f in reverse(d.transforms)
        y = invert(f, y)
    end
    return y
end

"""
    logpdf(d::InverseNormalisingFlow, y::RealOrVecOrMat)

Compute the log pdf of an observation `y` of the InverseNormalisingFlow `d`.
"""
function logpdf(d::InverseNormalisingFlow, y::AbstractVecOrMat{<:Real})
    l = zero(eltype(y))
    for k in 1:length(d.transforms)
        l += sum(logdetJ(d.transforms[k], y))
        y = apply(d.transforms[k], y)
    end
    return l + sum(logpdf(d.p0, y))
end
