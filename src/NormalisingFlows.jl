__precompile__(true)

module NormalisingFlows

using Nabla

export InverseNormalisingFlow, Affine, lpdf, Normal, dim, invert

const VecOrReal = Union{AbstractVector{<:Real}, Real}
const RealOrVecOrMat = Union{Real, AbstractVecOrMat{<:Real}}

include("normal.jl")
include("transforms.jl")
include("flow.jl")

end # module
