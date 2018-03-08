@testset "affine" begin

    let rng = MersenneTwister(123456)

        # Test that the application and inversion of the transform yields sane results.
        x1, x2 = randn(rng, 2, 5), randn(rng, 1, 5)
        @test apply(Affine(0.0, 0.0), x2) == x2
        @test apply(Affine(1.0, 0.0), x2) == x2 + 1
        @test apply(Affine(ones(2), zeros(2)), x1) == x1 + ones(x1)
        @test apply(Affine(1.0, 0.0), [1.0]) ≈ [2.0]

        # Test that the logdetJ yields some sane results.
        @test logdetJ(Affine(0.0, 0.0), [5.0]) == zero(5.0)
        @test logdetJ(Affine(0.0, 1.0), [5.0]) == one(5.0)
        @test logdetJ(Affine(0.0, 2.0), [5.0]) ≈ 2 * one(5.0)
        @test logdetJ(Affine(0, 0), randn(1, 5)) == zeros(5)
        @test logdetJ(Affine(0, 1), randn(1, 7)) == ones(7)
    end
end
