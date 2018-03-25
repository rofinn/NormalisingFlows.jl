# NormalisingFlows


[![Build Status](https://travis-ci.org/willtebbutt/NormalisingFlows.jl.svg?branch=master)](https://travis-ci.org/willtebbutt/NormalisingFlows.jl) [![Windows Build status](https://ci.appveyor.com/api/projects/status/g0gun5dxbkt631am/branch/master?svg=true)](https://ci.appveyor.com/project/willtebbutt/normalisingflows-jl/branch/master) [![codecov.io](http://codecov.io/github/willtebbutt/NormalisingFlows.jl/coverage.svg?branch=master)](http://codecov.io/github/willtebbutt/NormalisingFlows.jl?branch=master)

This is a WIP. Feel free to play around with it, raise issues / make PRs if you find any bugs
or have any suggestions. Particularly useful would be implementations of more invertible transforms.
We currently support the Planar and Radial flows, as well as a couple of basic affine flows.

Note that we are currently using slightly different conventions to Distributions.jl;
I will make sure to converge on their conventions before any kind of release.

## Example of basic usage

Note: There is currently a bug in Nabla.jl that makes this example break. Please checkout to [this PR](https://github.com/invenia/Nabla.jl/pull/90) to make the example below work for now.

```julia
# Install our packages (e.g., NormalisingFlows, Optim for an optimization function,
# Plots/GR for plotting).
Pkg.clone("https://github.com/willtebbutt/NormalisingFlows.jl")
Pkg.add("Optim")
Pkg.add("Plots")
Pkg.add("GR")

# Load our packages
using NormalisingFlows, Nabla, Distributions	# Installed with NormalisingFlows
using Optim, Plots

# Load the GR backend
gr()

# Specify toy model and draw some samples from it.
srand(123456);
rng, D, N, d = MersenneTwister(123456), 1, 10000, TDist(3.0);

# Draw some samples. This package assumes all data are in a matrix, with `D` rows and `N` columns.
y = reshape(rand(d, N), D, N);

# Specify objective function which constructs an Inverse Normalising Flow, and
# some stuff to make it work with Optim.
transforms = [DiagAffine, Planar, Planar, Planar];

function objective(θ)
    -logpdf(INF(DiagStdNormal(D), θ, transforms), y) / N -
    logpdf(DiagStdNormal(length(θ)), θ) / N
end

update!(storage, θ) = storage .= ∇(objective)(θ)[1]


# Choose a (potentially crappy) initialisation and optimise.
# NOTE: You may want to rerun the lines below to test the model
# on different initializations.
θ0 = naive_init(INF, rng, D, transforms)
options = Optim.Options(show_every=5, show_trace=true, iterations=100)
result = optimize(objective, update!, θ0, LBFGS(), options)
θ_opt = result.minimizer
m = INF(DiagStdNormal(D), θ_opt, transforms);

# Compare logpdf of true model and flow.
println("True logpdf is $(mean(logpdf.(d, y))).")
println("Model logpdf is $(logpdf(m, y) / N).")

# Visualise the logpdf.
ŷ = linspace(-3.0, 3.0, 10000);
p = plot(ŷ, logpdf.(d, ŷ), label="True")
plot!(p, ŷ, [logpdf(m, [ŷ_]) for ŷ_ in ŷ], label="Flow")
```
Note the log likelihood seems to have quite a few local minima, so you should expect to have to run this a few times to get a reasonable looking result.
