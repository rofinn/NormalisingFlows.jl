"""
    abstract type Invertible

Parent type for all invertible transforms.
"""
abstract type Invertible end

include("affine.jl")
include("planar.jl")
include("radial.jl")
