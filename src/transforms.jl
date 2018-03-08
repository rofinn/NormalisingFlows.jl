"""
    abstract type Invertible

Parent type for all invertible transforms.
"""
abstract type Invertible end

"""
    naive_init

Argument for constructor of a flow indicating that any old initialisation will do.
"""
function naive_init end

"""
    identity_init

Argument for constructor of a flow indicating that the flow should be the identity function.
"""
function identity_init end

include("affine.jl")
include("planar.jl")
include("radial.jl")
