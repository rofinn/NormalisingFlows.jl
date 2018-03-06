using NormalisingFlows, Base.Test, Optim, Nabla, QuadGK

import NormalisingFlows: apply, logdetJ

@testset "NormalisingFlows" begin

    include("normal.jl")

    @testset "transforms" begin
        include("affine.jl")
        include("planar.jl")
        include("radial.jl")
    end

    include("flow.jl")
end
