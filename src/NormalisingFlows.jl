__precompile__(true)

module NormalisingFlows

using Nabla, Distributions
import Base: rand, broadcast
import Distributions: AbstractMvNormal, logpdf, dim, params

export InverseNormalisingFlow, logpdf, DiagonalStandardNormal, DiagStdNormal, dim, invert,
    Affine, Planar, Radial, naive_init, identity_init, params, Invertible, nparams,
    starts, ends

include("normal.jl")
include("transforms.jl")
include("flow.jl")

end # module
