#!/usr/bin/env julia
# Validate MLFriends gives CORRECT evidence on a CURVED region (the mechanism
# behind the curved-RV bias), tractably in 2D where rejection works.
# Run against the NestedSamplers CLONE project:
#   julia --project=/Users/jayvains/github/personal/NestedSamplers.jl <thisfile>
#
# Target: prior Uniform([-5,5]²); likelihood = exp(-½((‖x‖-r0)/σ)²) — a thin
# ANNULUS at radius r0. A single/multi ellipsoid must cover the whole disk
# (incl. the empty centre): with the rslice proposal the slice scale is set by
# the big ellipsoid → too coarse for the thin ring → mixing/coverage bias.
# MLFriends' balls hug the ring → tight region. Analytic logZ from the radial
# integral is the ground truth.

using NestedSamplers, Distributions, LinearAlgebra, Random, Printf
using NestedSamplers: Bounds, Proposals

const r0 = 2.0; const σ = 0.2; const L = 5.0; const area = (2L)^2
logl(x) = -0.5 * ((norm(x) - r0) / σ)^2
prior = [Uniform(-L, L), Uniform(-L, L)]
model = NestedModel(logl, prior)

# analytic logZ = log( (2π/area) ∫₀^∞ exp(-½((r-r0)/σ)²) r dr )
rs = range(0, L; length = 200_000); dr = step(rs)
I = sum(@. exp(-0.5 * ((rs - r0) / σ)^2) * rs) * dr
logZ_true = log(2π * I / area)
@printf("=== 2D annulus (curved) evidence — analytic logZ = %.4f ===\n", logZ_true)

for (name, bnd, prop) in (
        ("MLFriends+reject",  Bounds.MLFriends,      Proposals.Rejection()),
        ("MultiEll+reject",   Bounds.MultiEllipsoid, Proposals.Rejection()),
        ("MultiEll+rslice",   Bounds.MultiEllipsoid, Proposals.RSlice(slices = 10)))
    sampler = Nested(2, 1000; bounds = bnd, proposal = prop)
    t0 = time()
    _, st = sample(MersenneTwister(42), model, sampler; dlogz = 0.05,
                   chain_type = Array, progress = false)
    @printf("[%-17s] logZ = %8.4f ± %.4f   err=%.4f   (%.0fs)\n",
            name, st.logz, st.logzerr, abs(st.logz - logZ_true), time() - t0)
end
@printf("\nMLFriends should match analytic; if MultiEll+rslice is off, that's the\ncurved-region bias MLFriends removes.\n")
