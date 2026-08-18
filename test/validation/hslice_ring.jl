#!/usr/bin/env julia
# Validate the HSlice (reflective slice) proposal is CORRECT on a curved problem
# with analytic logZ, before trusting it on the 7D RV. Run against CLONE project:
#   julia --project=/Users/jayvains/github/personal/NestedSamplers.jl <thisfile>
# 2D annulus: prior Uniform([-5,5]²), L=exp(-½((‖x‖-r0)/σ)²). analytic logZ below.

using NestedSamplers, Distributions, LinearAlgebra, Random, Printf
using NestedSamplers: Bounds, Proposals

const r0 = 2.0; const σ = 0.2; const Lb = 5.0; const area = (2Lb)^2
logl(x) = -0.5 * ((norm(x) - r0) / σ)^2
prior = [Uniform(-Lb, Lb), Uniform(-Lb, Lb)]
model = NestedModel(logl, prior)
rs = range(0, Lb; length = 200_000); dr = step(rs)
I = sum(@. exp(-0.5 * ((rs - r0) / σ)^2) * rs) * dr
logZ_true = log(2π * I / area)
@printf("=== HSlice correctness on 2D annulus — analytic logZ = %.4f ===\n", logZ_true)

for (name, prop) in (("rslice", Proposals.RSlice(slices = 10)),
                     ("hslice", Proposals.HSlice(slices = 5, n_steps = 15)))
    sampler = Nested(2, 1000; bounds = Bounds.MultiEllipsoid, proposal = prop)
    t0 = time()
    _, st = sample(MersenneTwister(42), model, sampler; dlogz = 0.05,
                   chain_type = Array, progress = false)
    @printf("[multi + %-7s] logZ = %8.4f ± %.4f   err=%.4f   (%.0fs)\n",
            name, st.logz, st.logzerr, abs(st.logz - logZ_true), time() - t0)
end
@printf("\nhslice err ≲ 0.1 ⇒ the reflective-slice proposal is correct.\n")
