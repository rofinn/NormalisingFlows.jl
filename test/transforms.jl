@testset "transforms" begin

    let rng = MersenneTwister(123456)

        # Test construction and dimensionality.
        @test_throws MethodError Affine(0.0, randn(5))
        @test_throws TypeError Affine(randn(4, 5), randn(4, 5))
        @test dim(Affine(0.0, 0.0)) == 1
        @test dim(Affine(randn(5), randn(5))) == 5

        # Test that the application and inversion of the transform yields sane results.
        x1, x2 = randn(rng, 2, 5), randn(rng, 1, 5)
        @test_throws AssertionError apply(Affine(0.0, 0.0), x1)
        let a = Affine(0.0, 0.0)
            @test apply(a, x2) == x2
            @test invert(a, apply(a, x2)) ≈ x2
            @test apply(a, invert(a, x2)) ≈ x2
        end
        let a = Affine(1.0, 0.0)
            @test apply(a, x2) == x2 + 1
            @test invert(a, apply(a, x2)) ≈ x2
            @test apply(a, invert(a, x2)) ≈ x2
        end
        let a = Affine(ones(2), zeros(2))
            @test apply(a, x1) == x1 + ones(x1)
            @test invert(a, apply(a, x1)) ≈ x1
            @test apply(a, invert(a, x1)) ≈ x1
        end
        let a = Affine(1.0, 0.0)
            @test apply(a, 1.0) ≈ 2.0
            @test invert(a, apply(a, 1.0)) ≈ 1.0
            @test apply(a, invert(a, 1.0)) ≈ 1.0
        end
        @test apply(Affine(1.0, 0.0), [1])[1] == apply(Affine(1.0, 0.0), 1)


        # Test that the logdetJ yields some sane results.
        @test logdetJ(Affine(0.0, 0.0), 5.0) == 0.0
        @test logdetJ(Affine(0.0, 1.0), 5.0) == 1.0
        @test_throws AssertionError logdetJ(Affine(randn(5), randn(5)), 5.0)
        @test logdetJ(Affine(0.0, 2.0), 5.0) == 2.0
        @test logdetJ(Affine(0, 0), randn(1, 5)) == 0
        @test logdetJ(Affine(0, 1), randn(1, 7)) == 7
    end
end
