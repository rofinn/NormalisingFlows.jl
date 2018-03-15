using NormalisingFlows, Base.Test, Optim, Nabla, QuadGK, Distributions

import PDMats
import NormalisingFlows: apply, logdetJ

const __transforms = [DiagAffine, Affine, Planar, Radial]

@testset "NormalisingFlows" begin
    include("normal.jl")
    include("invertibles.jl")
    include("flows.jl")
end
