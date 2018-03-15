__precompile__(true)

module NormalisingFlows

using Nabla, Distributions, Roots
import Base: rand, broadcast
import Distributions: AbstractMvNormal, logpdf, dim, params

export InverseNormalisingFlow, INF, logpdf, DiagonalStandardNormal, DiagStdNormal, dim,
    invert, DiagAffine, Affine, Planar, Radial, Tanh, params, Invertible, nparams, apply,
    logdetJ

export naive_init, naive_init!, identity_init, identity_init!

const RealVec = AbstractVector{<:Real}
const RealMat = AbstractMatrix{<:Real}

# Some initialisation functions. Invertibles and Flows will define concrete versions.

function naive_init end
function naive_init! end

function identity_init end
function identity_init! end

# Misc. definition.
softplus(x) = log1p(exp(x))

include("normal.jl")
include("invertibles.jl")
include("flows.jl")

end # module
