#!/usr/bin/env julia
# Standalone MLFriends test — run against the NestedSamplers CLONE project:
#   julia --project=/Users/jayvains/github/personal/NestedSamplers.jl <thisfile>
# (1) unit checks: fit / in / rand / scale!  (2) analytic 2D Gaussian×box logZ.

using NestedSamplers, Distributions, LinearAlgebra, Random, Statistics, Printf
using NestedSamplers: Bounds, Proposals

rng = MersenneTwister(1)

# ---- (1) unit checks --------------------------------------------------------
# curved ridge in the unit cube: a quarter-annulus of points
θ = range(0.1, π/2 - 0.1; length = 300)
ρ = 0.35 .+ 0.03 .* randn(rng, 300)
pts = vcat((0.5 .+ ρ .* cos.(θ))', (0.1 .+ ρ .* sin.(θ))')   # 2 × 300, inside [0,1]²
b = Bounds.fit(Bounds.MLFriends, pts; pointvol = 0)
in_all = all(b.u[:, i] ∈ b for i in 1:size(b.u, 2))
samp = [rand(rng, b) for _ in 1:500]
samp_in = all(s ∈ b for s in samp)
samp_cube = all(all(0 .< s .< 1) for s in samp)
v0 = Bounds.volume(b); Bounds.scale!(b, 2.0); v1 = Bounds.volume(b)
@printf("[unit] maxradiussq=%.4g enlarge=%.4g  | live∈region:%s  rand∈region:%s  rand∈cube:%s  scale×2:%.3f→%.3f(%s)\n",
        b.maxradiussq, b.enlarge, in_all, samp_in, samp_cube, v0, v1, isapprox(v1, 2v0; rtol=1e-6) ? "ok" : "BAD")

# ---- (2) analytic 2D Gaussian × uniform box ---------------------------------
# prior Uniform([-10,10]²); likelihood N(μ,Σ) (normalised) → logZ ≈ -2 log 20.
μ = [3.0, -1.5]; Σ = [1.0 0.5; 0.5 2.0]; mvn = MvNormal(μ, Σ)
logl(x) = logpdf(mvn, x)
prior = [Uniform(-10.0, 10.0), Uniform(-10.0, 10.0)]
model = NestedModel(logl, prior)
logZ_true = -2 * log(20.0)
@printf("\nanalytic logZ = %.4f\n", logZ_true)

for (name, bnd) in (("MLFriends", Bounds.MLFriends), ("MultiEllipsoid", Bounds.MultiEllipsoid))
    sampler = Nested(2, 1000; bounds = bnd, proposal = Proposals.Rejection())
    t0 = time()
    _, st = sample(MersenneTwister(42), model, sampler; dlogz = 0.01,
                   chain_type = Array, progress = false)
    @printf("[%-14s] logZ = %8.4f ± %.4f   err=%.4f   (%.0fs)\n",
            name, st.logz, st.logzerr, abs(st.logz - logZ_true), time() - t0)
end

ok = in_all && samp_in && samp_cube && isapprox(v1, 2v0; rtol = 1e-6)
println(ok ? "\n✅ MLFriends unit checks pass (evidence numbers above)" : "\n❌ unit checks FAIL")
