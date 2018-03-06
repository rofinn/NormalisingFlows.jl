__precompile__(true)

module NormalisingFlows

using Nabla

export InverseNormalisingFlow, lpdf, Normal, dim, invert, Affine, Planar, Radial

const VecOrReal = Union{AbstractVector{<:Real}, Real}
const RealOrVecOrMat = Union{Real, AbstractVecOrMat{<:Real}}

include("normal.jl")
include("transforms.jl")
include("flow.jl")

end # module
