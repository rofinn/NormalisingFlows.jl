__precompile__(true)

module NormalisingFlows

using Nabla, Distributions
import Base: rand, broadcast
import Distributions: AbstractMvNormal, logpdf, dim

export InverseNormalisingFlow, logpdf, DiagonalStandardNormal, dim, invert, Affine, Planar,
    Radial, naive_init, identity_init

include("normal.jl")
include("transforms.jl")
include("flow.jl")

end # module
