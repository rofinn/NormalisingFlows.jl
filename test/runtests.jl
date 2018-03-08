using NormalisingFlows, Base.Test, Optim, Nabla, QuadGK, Distributions

import PDMats
import NormalisingFlows: apply, logdetJ

@testset "NormalisingFlows" begin
    include("normal.jl")
    include("transforms.jl")
    include("flow.jl")
end
