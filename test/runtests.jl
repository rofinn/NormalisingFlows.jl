using NormalisingFlows, Base.Test, Optim, Nabla

import NormalisingFlows: apply, logdetJ

@testset "NormalisingFlows" begin
    include("normal.jl")
    include("transforms.jl")
    include("flow.jl")
end
