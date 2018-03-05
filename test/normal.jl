@testset "normal" begin

    import NormalisingFlows: Normal
    let rng = MersenneTwister(123456), N = 10_000_000, D = 5, N′ = 10, tol = 1e-1

        # Test construction and dims.
        @test dim(Normal(0.0, 0.0)) == 1
        @test dim(Normal(zeros(D), zeros(D))) == D
        @test_throws MethodError Normal(0.0, zeros(D))
        @test_throws AssertionError Normal(zeros(1), zeros(2))

        # Allocating rng.
        @test abs(mean(rand(rng, Normal(0.0, 0.0), N))) < tol
        @test abs(mean(rand(rng, Normal(0.5, 0.0), N))) - 0.5 < tol
        @test abs(std(rand(rng, Normal(0.0, 0.0), N)) - 1) < tol
        @test abs(std(rand(rng, Normal(0.0, log(2.0)), N)) - 2) < tol

        # Allocating rng vector.
        @test maximum(abs.(mean(rand(rng, Normal(zeros(D), zeros(D)), N), 2))) < tol
        @test maximum(abs.(mean(rand(rng, Normal(ones(D), zeros(D)), N), 2) - ones(D))) < tol
        @test maximum(abs.(std(rand(rng, Normal(zeros(D), zeros(D)), N), 2) - ones(D))) < tol
        logσ = log.(2 .* ones(D))
        @test maximum(abs.(std(rand(rng, Normal(zeros(D), logσ), N), 2) - exp.(logσ))) < tol

        # In-place rng scalar.
        A = zeros(1, N)
        @test abs(mean(rand!(rng, Normal(0.0, 0.0), A))) < tol
        @test abs(mean(rand!(rng, Normal(0.5, 0.0), A)) - 0.5) < tol
        @test abs(std(rand!(rng, Normal(0.0, 0.0), A)) - 1) < tol
        @test abs(std(rand!(rng, Normal(0.0, log(2.0)), A)) - 2) < tol

        # In-place rng vector.
        A, zs, os = zeros(D, N), zeros(D), ones(D)
        @test_throws AssertionError rand!(rng, Normal(zs, zs), randn(D + 1, N))
        @test maximum(abs.(mean(rand!(rng, Normal(zeros(D), zeros(D)), A), 2))) < tol
        @test maximum(abs.(mean(rand!(rng, Normal(0.5 * os, zs), A), 2) - 0.5 * os)) < tol
        @test maximum(abs.(std(rand!(rng, Normal(zs, zs), A), 2) - os)) < tol
        @test maximum(abs.(std(rand!(rng, Normal(zs, logσ), A), 2) .- exp.(logσ))) < tol

        # Test invariances of lpdf under broadcast vs for loop.
        d, X = Normal(zeros(D), zeros(D)), randn(rng, D, N′)
        @test [lpdf(d, X[:, n]) for n in 1:N′] ≈ vec(lpdf.(d, X))
        @test sum(lpdf.(d, X)) ≈ lpdf(d, X)
    end
end
