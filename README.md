# NormalisingFlows


[![Build Status](https://travis-ci.org/willtebbutt/NormalisingFlows.jl.svg?branch=master)](https://travis-ci.org/willtebbutt/NormalisingFlows.jl) [![Windows Build status](https://ci.appveyor.com/api/projects/status/g0gun5dxbkt631am/branch/master?svg=true)](https://ci.appveyor.com/project/willtebbutt/normalisingflows-jl/branch/master) [![codecov.io](http://codecov.io/github/willtebbutt/NormalisingFlows.jl/coverage.svg?branch=master)](http://codecov.io/github/willtebbutt/NormalisingFlows.jl?branch=master)

This is a WIP. Feel free to play around with it, raise issues / make PRs if you find any bugs or have any suggestions. Particularly useful would be implementations of more invertible transforms. We currently support the Planar and Radial flows, as well as a couple of basic affine flows.

Note that we are currently using slightly different conventions to Distributions.jl; I will make sure to converge on their conventions before any kind of release.

## Example of basic usage

```julia
using NormalisingFlows, Nabla, Optim, PyPlot, Distributions

# Specify toy model and draw some samples from it.
srand(123456);
rng, D, N, d = MersenneTwister(123456), 1, 10000, TDist(3.0);

# Draw some samples. This package assumes all data are in a matrix, with `D` rows and `N` columns.
y = reshape(rand(d, N), D, N);

# Specify objective function which constructs an Inverse Normalising Flow, and some stuff to make it work with Optim.
transforms = [DiagAffine, Planar, Planar, Planar];
obj = θ->-logpdf(INF(DiagStdNormal(D), θ, transforms), y) / N -
                logpdf(DiagStdNormal(length(θ)), θ) / N;
g! = (storage, θ)->(storage .= ∇(obj)(θ)[1]; storage);

# Choose a (potentially crappy) initialisation and optimise.
θ0 = naive_init(INF, rng, D, transforms);
options = Optim.Options(show_every=5, show_trace=true, iterations=100);
θ_opt = optimize(obj, g!, θ0, LBFGS(), options).minimizer;
m = INF(DiagStdNormal(D), θ_opt, transforms);

# Compare logpdf of true model and flow.
println("True logpdf is $(mean(logpdf.(d, y))).")
println("Model logpdf is $(logpdf(m, y) / N).")

# Visualise the logpdf.
ŷ = linspace(-3.0, 3.0, 10000);
plot(ŷ, logpdf.(d, ŷ), "b", label="True");
plot(ŷ, [logpdf(m, [ŷ_]) for ŷ_ in ŷ], "r", label="Flow");
legend();
```
Note the log likelihood seems to have quite a few local minima, so you should expect to have to run this a few times to get a reasonable looking result.
