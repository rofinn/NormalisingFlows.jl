@testset "normal" begin

    for D in [1, 5]
        let rng = MersenneTwister(123456), N = 10_000_000, D = 5, tol = 1e-2

            # Test dimensionality.
            @test dim(DiagonalStandardNormal(1)) == 1
            @test dim(DiagonalStandardNormal(D)) == D

            # Allocating rng.
            @test abs(mean(rand(rng, DiagonalStandardNormal(1), N))) < tol
            @test abs(mean(rand(rng, DiagonalStandardNormal(D), N))) < tol
            @test abs(std(rand(rng, DiagonalStandardNormal(1), N)) - 1) < tol
            @test abs(std(rand(rng, DiagonalStandardNormal(D), N)) - 1) < tol

            # Test logpdf.
            let x = randn(rng, D)
                d = DiagonalStandardNormal(D)
                d′ = MvNormal(PDMats.ScalMat(D, 1.0))
                @test logpdf(d, x) ≈ logpdf(d′, x)
                for N′ in [1, 7]
                    X = randn(rng, D, N′)
                    @test logpdf(d, X) ≈ logpdf(d′, X)
                end
            end

            # Check that gradients propagate properly through second argument.
            for N′ in [1, 6]
                x = randn(rng, D)
                X = randn(rng, D, N′)
                d = DiagonalStandardNormal(D)
                @test ∇(x->logpdf(d, x))(x)[1] ≈ -x
                @test ∇(X->sum(logpdf(d, X)))(X)[1] ≈ -X
            end
        end
    end
end
