#!/usr/bin/env julia
# The genuine friends-bound advantage: MULTIMODAL / disconnected posteriors
# where multi-ellipsoid clustering under-covers (cf. the 51-Peg period-comb,
# project_mlfriends_port_assessment). Run against the CLONE project:
#   julia --project=/Users/jayvains/github/personal/NestedSamplers.jl <thisfile>
#
# Target: prior Uniform([-6,6]²); likelihood = Σ of 9 narrow Gaussians on a 3×3
# grid (centres {-3,0,3}², σ=0.3). Analytic logZ = log(9·2πσ² / area).
# MLFriends puts balls on the populated modes (covers all 9); MultiEllipsoid
# must cluster into 9 — if it merges/misses any, logZ is biased.

using NestedSamplers, Distributions, LinearAlgebra, Random, Printf
using NestedSamplers: Bounds, Proposals

const σ = 0.3; const Lb = 6.0; const area = (2Lb)^2
centres = [[x, y] for x in (-3.0, 0.0, 3.0) for y in (-3.0, 0.0, 3.0)]
function logl(p)
    s = -Inf
    for c in centres
        s = log(exp(s) + exp(-0.5 * sum(abs2, p .- c) / σ^2))
    end
    return s
end
prior = [Uniform(-Lb, Lb), Uniform(-Lb, Lb)]
model = NestedModel(logl, prior)
# each Gaussian integrates to 2πσ²; 9 of them; prior density 1/area
logZ_true = log(9 * 2π * σ^2 / area)
@printf("=== 2D 9-mode multimodal evidence — analytic logZ = %.4f ===\n", logZ_true)

for (name, bnd, prop) in (
        ("MLFriends+reject",  Bounds.MLFriends,      Proposals.Rejection()),
        ("MultiEll+reject",   Bounds.MultiEllipsoid, Proposals.Rejection()),
        ("MultiEll+rslice",   Bounds.MultiEllipsoid, Proposals.RSlice(slices = 10)))
    sampler = Nested(2, 1500; bounds = bnd, proposal = prop)
    t0 = time()
    _, st = sample(MersenneTwister(42), model, sampler; dlogz = 0.05,
                   chain_type = Array, progress = false)
    @printf("[%-17s] logZ = %8.4f ± %.4f   err=%.4f   (%.0fs)\n",
            name, st.logz, st.logzerr, abs(st.logz - logZ_true), time() - t0)
end
@printf("\nIf MultiEll under-covers some modes → biased; MLFriends should match analytic.\n")
