@testset "flow" begin

    # Test construction and dims.
    @test_throws AssertionError InverseNormalisingFlow(Normal(0.0, 0.0), [Affine(zeros(5), zeros(5))])
    @test_throws AssertionError InverseNormalisingFlow(Normal(zeros(2), ones(2)), [Affine(0.0, 0.0)])
    @test dim(InverseNormalisingFlow(Normal(0.0, 0.0), [Affine(0.0, 0.0)])) == 1
    @test dim(InverseNormalisingFlow(Normal(0.0, 0.0), [Affine([0.0], [0.0])])) == 1

    # Test basics of log probability computations are consistent with the construction.
    let rng = MersenneTwister(123456)
        flow = InverseNormalisingFlow(Normal(0.0, 0.0), [Affine(0.0, 0.0)])
        @test lpdf(flow, 0.0) == lpdf(Normal(0.0, 0.0), 0.0)
        @test lpdf(flow, 1.0) == lpdf(Normal(0.0, 0.0), 1.0)

        p0, a = Normal(0.0, 0.0), Affine(0.0, 1.0)
        flow, y = InverseNormalisingFlow(p0, [a]), 1.0
        z = apply(a, y)
        @test lpdf(flow, y) == lpdf(p0, z) + logdetJ(a, y)

        p0, a1, a2 = Normal(0.0, 0.0), Affine(0.0, 1.0), Affine(3.0, 2.0)
        flow, y = InverseNormalisingFlow(p0, [a1, a2]), 1.0
        z1, z2 = apply(a1, y), apply(a2, apply(a1, y))
        @test lpdf(flow, y) == logdetJ(a2, z1) + logdetJ(a1, y) + lpdf(p0, z2)
    end

    # Test the mechanics of `rand` for InverseNormalisingFlows.
    let rng = MersenneTwister(123456), N = 10, D = 2
        p0 = Normal(zeros(D), zeros(D))
        a = Affine(ones(D), 1.5 * ones(D))
        flow = InverseNormalisingFlow(p0, [a, a])
        @test size(rand(rng, flow, N)) == (D, N)
    end

    # Test that an affine flow integrates to 1.
    let rng = MersenneTwister(123456)
        p0 = Normal(0.0, 0.0)
        a = Affine(randn(rng), randn(rng))
        flow = InverseNormalisingFlow(p0, [a])
        @test abs(quadgk(y->exp(lpdf(flow, y)), -10, 10)[1] - 1) < 1e-6
    end

    # Test that a Planar flow integrates to 1.
    let rng = MersenneTwister(123456), h = tanh, h′ = x->1 - tanh(x)^2
        p0 = Normal(0.0, 0.0)
        planar = Planar(randn(rng), randn(rng), randn(rng), h, h′)
        flow = InverseNormalisingFlow(p0, [planar])
        @test abs(quadgk(y->exp(lpdf(flow, y)), -20, 20)[1] - 1) < 1e-6
    end

    # Test that we can recover the identity mapping using an Affine transform in spaces of
    # varying dimension and affine transforms.
    let rng = MersenneTwister(123456), N = 10000
        for D = 1:5

            # Construct an INF and sample from it N times.
            p0 = Normal(zeros(D), zeros(D))
            θ = randn(rng, 2D)
            a = Affine(θ[1:D], θ[D+1:end])
            y = rand(rng, InverseNormalisingFlow(p0, [a]), N)

            # Constuct objective function, perform ML inference, and test θ is recovered.
            function obj(θ)
                p0 = Normal(zeros(D), zeros(D))
                transforms = [Affine(θ[1:D], θ[D+1:2D])]
                return -lpdf(InverseNormalisingFlow(p0, transforms), y)
            end
            ∇obj = ∇(obj)
            g!(storage, θ) = (storage .= ∇obj(θ)[1]; storage)
            @test maximum(abs.(optimize(obj, g!, θ).minimizer .- θ)) < 1e-1
        end
    end
end