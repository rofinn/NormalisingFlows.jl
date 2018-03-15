@testset "flow" begin

    # Test construction and dims.
    @test_throws AssertionError InverseNormalisingFlow(DiagonalStandardNormal(1), [DiagAffine(zeros(5), zeros(5))])
    @test_throws AssertionError InverseNormalisingFlow(DiagonalStandardNormal(2), [DiagAffine(0.0, 0.0)])
    @test dim(InverseNormalisingFlow(DiagonalStandardNormal(1), [DiagAffine(0.0, 0.0)])) == 1
    @test dim(InverseNormalisingFlow(DiagonalStandardNormal(1), [DiagAffine([0.0], [0.0])])) == 1

    # Test params.
    let rng = MersenneTwister(123456)
        a = DiagAffine(identity_init, rng, 5)
        p = Planar(identity_init, rng, 5)
        flow = InverseNormalisingFlow(DiagStdNormal(5), [a, p])
        @test length(params(flow)) == 5
        @test params(flow) == vcat(params(a), params(p))
    end

    # Test the mechanics of constructing an INF from an initialiser.
    let rng = MersenneTwister(123456), D = 3, ctors = __transforms
        for (init, inplace_init!) in [(naive_init, naive_init!),
                                     (identity_init, identity_init!)]
            θ = fill!(Vector{Float64}(nparams(INF, D, ctors)), NaN)
            θ = inplace_init!(θ, InverseNormalisingFlow, rng, D, ctors)
            @test all(.!isnan.(θ))
            @test all(.!isinf.(θ))
            @test nparams(INF(DiagStdNormal(D), θ, ctors)) == length(θ)

            @test_throws AssertionError inplace_init!(θ, INF, rng, D + 1, ctors)
            @test_throws TypeError inplace_init!(θ, INF, rng, D, randn(rng, 3))
            @test_throws AssertionError inplace_init!(θ, INF, rng, D, [INF, INF, INF])

            @test length(init(INF, rng, D, ctors)) == length(θ)
            @test nparams(INF(init, rng, D, ctors)) == length(θ)

            @test dim(INF(DiagStdNormal(D), θ, ctors)) == dim(INF(init, rng, D, ctors))
        end
    end

    # Test basics of log probability computations are consistent with the construction.
    let
        flow = InverseNormalisingFlow(DiagonalStandardNormal(1), [DiagAffine(0.0, 0.0)])
        @test logpdf(flow, [0.0]) == logpdf(DiagonalStandardNormal(1), [0.0])
        @test logpdf(flow, [1.0]) == logpdf(DiagonalStandardNormal(1), [1.0])

        p0, a = DiagonalStandardNormal(1), DiagAffine(0.0, 1.0)
        flow, y = InverseNormalisingFlow(p0, [a]), [1.0]
        z = apply(a, y)
        @test logpdf(flow, y) == logpdf(p0, z) + logdetJ(a, y)

        p0, a1, a2 = DiagonalStandardNormal(1), DiagAffine(0.0, 1.0), DiagAffine(3.0, 2.0)
        flow, y = InverseNormalisingFlow(p0, [a1, a2]), [1.0]
        z1, z2 = apply(a1, y), apply(a2, apply(a1, y))
        @test logpdf(flow, y) == logdetJ(a2, z1) + logdetJ(a1, y) + logpdf(p0, z2)
    end

    # Test the mechanics of `rand` for InverseNormalisingFlows.
    let rng = MersenneTwister(123456), N = 10, D = 2
        p0 = DiagonalStandardNormal(D)
        a = DiagAffine(ones(D), 1.5 * ones(D))
        flow = InverseNormalisingFlow(p0, [a, a])
        @test size(rand(rng, flow, N)) == (D, N)
    end

    # Test that each single-layer flow integrates to 1 in 1-dimension.
    for Transform in __transforms
        let rng = MersenneTwister(123456)
            p0 = DiagonalStandardNormal(1)
            flow = InverseNormalisingFlow(p0, [Transform(naive_init, rng, 1)])
            @test abs(quadgk(y->exp(logpdf(flow, [y])), -10, 10)[1] - 1) < 1e-6
        end
    end

    # Test that we approximately recover a random transform given sufficiently many sampled.
    let rng = MersenneTwister(123456), N = 10000
        for D = [1, 5]

            # Construct an INF and sample from it N times.
            p0 = DiagonalStandardNormal(D)
            θ = randn(rng, 2D)
            a = DiagAffine(θ[1:D], θ[D+1:2D])
            y = rand(rng, InverseNormalisingFlow(p0, [a]), N)

            # Constuct objective function, perform ML inference, and test θ is recovered.
            function obj(θ)
                p0 = DiagonalStandardNormal(D)
                transforms = [DiagAffine(θ[1:D], θ[D+1:2D])]
                return -logpdf(InverseNormalisingFlow(p0, transforms), y)
            end
            ∇obj = ∇(obj)
            g!(storage, θ) = (storage .= ∇obj(θ)[1]; storage)
            @test maximum(abs.(optimize(obj, g!, θ).minimizer .- θ)) < 1e-1
        end
    end
end
