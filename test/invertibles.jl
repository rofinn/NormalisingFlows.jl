@testset "transforms" begin

    # Tests for each flow that can't be expressed in terms of the standard interface.
    include("invertibles/affine.jl")
    include("invertibles/planar.jl")
    include("invertibles/radial.jl")

    # Tests which are generic to all of the flows. Note that these tests essentially define
    # this packages' conventions, and ensure that each flow adhere's to them.
    for Transform in __transforms, D in [1, 5], M in [1, 9]

        # Test construction using *_init!, *_init and Foo(::typeof(*_init), ...).
        # Basic tests to make sure that the functions run.
        let rng = MersenneTwister(123456)
            for (init, inplace_init) in [(naive_init, naive_init!),
                                         (identity_init, identity_init!)]
                θ = fill!(Vector{Float64}(nparams(Transform, D)), NaN)
                @test all(.!isnan.(inplace_init(θ, Transform, rng, D)))
                @test length(θ) == length(init(Transform, rng, D))
                @test nparams(Transform(init, rng, D)) == length(θ)
            end
        end

        let rng = MersenneTwister(123456), transform = Transform(naive_init, rng, D)

            # Test naive construction dimensionality.
            @test dim(transform) == D

            # Check that AssertionErrors are thrown for incorrect sized args, and that
            # outputs are not NaN or Inf.
            for foo in [apply, invert, logdetJ]
                if method_exists(foo, Tuple{Transform, AbstractVector{<:Real}})
                    @test_throws AssertionError foo(transform, zeros(D + 1))
                    @test_throws AssertionError foo(transform, zeros(D - 1))
                    @test all(.!isnan.(foo(transform, randn(rng, D))))
                    @test all(.!isinf.(foo(transform, randn(rng, D))))
                end
                if method_exists(foo, Tuple{Transform, AbstractMatrix{<:Real}})
                    @test_throws AssertionError foo(transform, zeros(D + 1, M))
                    @test_throws AssertionError foo(transform, zeros(D - 1, M))
                    @test all(.!isnan.(foo(transform, randn(rng, D, M))))
                    @test all(.!isinf.(foo(transform, randn(rng, D, M))))
                end
            end

            # Check that size conventions are obeyed.
            for foo in [apply, invert]
                if method_exists(foo, Tuple{Transform, AbstractVector{<:Real}})
                    @test size(foo(transform, zeros(D))) == (D,)
                end
                if method_exists(foo, Tuple{Transform, AbstractMatrix{<:Real}})
                    @test size(foo(transform, zeros(D, M))) == (D, M)
                end
            end
            @test size(logdetJ(transform, zeros(D))) == ()
            @test size(logdetJ(transform, zeros(D, M))) == (M,)

            # Check that broadcasting yields the same results as vectorised version.
            for foo in [apply, invert]
                if method_exists(foo, Tuple{Transform, AbstractVector{<:Real}}) &&
                    method_exists(foo, Tuple{Transform, AbstractMatrix{<:Real}})
                    X = randn(rng, D, M)
                    Y_iterated = hcat([foo(transform, X[:, m]) for m in 1:M]...)
                    @test Y_iterated ≈ foo(transform, X)
                end
            end
            if method_exists(logdetJ, Tuple{Transform, AbstractVector{<:Real}}) &&
                method_exists(logdetJ, Tuple{Transform, AbstractMatrix{<:Real}})
                X = randn(rng, D, M)
                Y_iterated = vcat([logdetJ(transform, X[:, m]) for m in 1:M]...)
                @test Y_iterated ≈ logdetJ(transform, X)
            end

            # Check that `invert` is indeed the inverse of `apply`, if both are defined.
            if method_exists(apply, Tuple{Transform, AbstractVector{<:Real}}) &&
                method_exists(invert, Tuple{Transform, AbstractVector{<:Real}})
                let x1 = randn(rng, D)
                    @test apply(transform, invert(transform, x1)) ≈ x1
                    @test invert(transform, apply(transform, x1)) ≈ x1
                end
            end
            if method_exists(apply, Tuple{Transform, AbstractMatrix{<:Real}}) &&
                method_exists(invert, Tuple{Transform, AbstractMatrix{<:Real}})
                let x2 = randn(rng, D, M)
                    @test apply(transform, invert(transform, x2)) ≈ x2
                    @test invert(transform, apply(transform, x2)) ≈ x2
                end
            end

            # Ensure that the naive_init methods don't provide insane initialisations.
            let transform = Transform(naive_init, rng, D)
                for foo in [apply, invert, logdetJ]
                    if method_exists(foo, Tuple{Transform, AbstractVector{<:Real}})
                        @test all(.!isnan.(foo(transform, randn(rng, D))))
                        @test all(.!isinf.(foo(transform, randn(rng, D))))
                    end
                    if method_exists(foo, Tuple{Transform, AbstractMatrix{<:Real}})
                        @test all(.!isnan.(foo(transform, randn(rng, D, M))))
                        @test all(.!isinf.(foo(transform, randn(rng, D, M))))
                    end
                end
            end

            # Check that identity initialisations indeed yield the identity function.
            if method_exists(Transform, Tuple{typeof(identity_init), AbstractRNG, Int})
                identity_transform = Transform(identity_init, rng, D)
                let x1 = randn(rng, D), x2 = randn(rng, D, M)
                    @test apply(identity_transform, x1) ≈ x1
                    @test apply(identity_transform, x2) ≈ x2
                    @test all(abs.(logdetJ(identity_transform, x1)) .< 1e-12)
                    @test all(abs.(logdetJ(identity_transform, x2)) .< 1e-12)
                end
            end

            @test nparams(transform) == nparams(Transform, D)
        end # let
    end # for Transform...
end
