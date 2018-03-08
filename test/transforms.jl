@testset "transforms" begin

    # Tests for each flow that can't be expressed in terms of the standard interface.
    include("affine.jl")
    include("planar.jl")
    include("radial.jl")

    # Tests which are generic to all of the flows. Note that these tests essentially define
    # this packages' conventions, and ensure that each flow adhere's to them.
    import NormalisingFlows: Affine, Planar, naive_init, identity_init
    for Transform in [Affine, Planar], D in [1, 5], M in [1, 9]

        let rng = MersenneTwister(123456), transform = Transform(naive_init, rng, D)

            # Test naive construction dimensionality.
            @test dim(transform) == D
            @test dim(Transform(naive_init, rng, D, Tape())[1]) == D

            # Check that AssertionErrors are thrown for incorrect sized args.
            for foo in [apply, invert, logdetJ]
                if method_exists(foo, Tuple{Transform, AbstractVecOrMat{<:Real}})
                    @test_throws AssertionError foo(transform, zeros(D + 1))
                    @test_throws AssertionError foo(transform, zeros(D - 1))
                    @test_throws AssertionError foo(transform, zeros(D + 1, M))
                    @test_throws AssertionError foo(transform, zeros(D - 1, M))
                end
            end

            # Check that size conventions are obeyed.
            for foo in [apply, invert]
                if method_exists(foo, Tuple{Transform, AbstractVecOrMat{<:Real}})
                    @test size(foo(transform, zeros(D))) == (D,)
                    @test size(foo(transform, zeros(D, M))) == (D, M)
                end
            end
            @test size(logdetJ(transform, zeros(D))) == ()
            @test size(logdetJ(transform, zeros(D, M))) == (M,)

            # Check that broadcasting yields the same results as vectorised version.
            for foo in [apply, invert]
                if method_exists(foo, Tuple{Transform, AbstractVecOrMat{<:Real}}) &&
                    method_exists(foo, Tuple{Transform, AbstractMatrix{<:Real}})
                    X = randn(rng, D, M)
                    Y_iterated = hcat([foo(transform, X[:, m]) for m in 1:M]...)
                    @test Y_iterated == foo(transform, X)
                end
            end
            if method_exists(logdetJ, Tuple{Transform, AbstractVector{<:Real}}) &&
                method_exists(logdetJ, Tuple{Transform, AbstractMatrix{<:Real}})
                X = randn(rng, D, M)
                Y_iterated = vcat([logdetJ(transform, X[:, m]) for m in 1:M]...)
                @test Y_iterated ≈ logdetJ(transform, X)
            end

            # Check that `invert` is indeed the inverse of `apply`, if both are defined.
            if method_exists(apply, Tuple{Transform, AbstractVecOrMat{<:Real}}) &&
                method_exists(invert, Tuple{Transform, AbstractVecOrMat{<:Real}})
                let x1 = randn(rng, D), x2 = randn(rng, D, M)
                    @test apply(transform, invert(transform, x1)) ≈ x1
                    @test invert(transform, apply(transform, x1)) ≈ x1
                    @test apply(transform, invert(transform, x2)) ≈ x2
                    @test invert(transform, apply(transform, x2)) ≈ x2
                end
            end

            # Check that identity initialisations indeed yield the identity function.
            if method_exists(Transform, Tuple{typeof(identity_init), Int})
                identity_transform = Transform(identity_init, D)
                let x1 = randn(rng, D), x2 = randn(rng, D, M)
                    @test apply(identity_transform, x1) ≈ x1
                    @test apply(identity_transform, x2) ≈ x2
                    @test logdetJ(identity_transform, x1) ≈ zero(1)
                    @test logdetJ(identity_transform, x2) ≈ zeros(M)
                end
            end
        end
    end
end
