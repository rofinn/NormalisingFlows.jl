@testset "planar" begin
    @test params(Planar(1.0, 1.0, 1.0, tanh, tanh)) == [[1.0], [1.0], [1.0]]
end
