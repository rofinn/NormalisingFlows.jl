@testset "planar" begin

    # Test construction and dimensionality.
    let rng = MersenneTwister(123456), h = tanh, h′ = x->1 - tanh(x)^2
        @test_throws MethodError Planar(randn(rng, 5), 5.0, 5.0, h, h′)
        @test_throws AssertionError Planar(randn(rng, 5), randn(rng, 4), 5.0, h, h′)
        @test dim(Planar(randn(rng, 5), randn(rng, 5), 5.0, h, h′)) == 5
        @test dim(Planar(randn(rng), randn(rng), 5.0, h, h′)) == 1
    end

    # Test the application of a planar flow.
    let rng = MersenneTwister(123456), D = 6, N = 10, h = tanh, h′ = x->1 - tanh(x)^2
        flow = Planar(randn(rng, D), randn(rng, D), randn(rng), h, h′)
        @test_throws AssertionError apply(flow, randn(rng, D - 1))
        @test_throws AssertionError apply(flow, randn(rng, D - 1, N))

        X = randn(rng, D, N)
        @test hcat([apply(flow, X[:, n]) for n in 1:N]...) ≈ apply(flow, X) 
    end

    # Test the logdetJ of the planar flow.
    let rng = MersenneTwister(123456), D = 6, N = 10, h = tanh, h′ = x->1 - tanh(x)^2
        flow = Planar(randn(rng, D), randn(rng, D), randn(rng), h, h′)
        @test_throws AssertionError logdetJ(flow, randn(rng, D - 1))
        @test_throws AssertionError logdetJ(flow, randn(rng, D - 1, N))

        X = randn(rng, D, N)
        @test sum([logdetJ(flow, X[:, n]) for n in 1:N]) ≈ logdetJ(flow, X)
    end
end
