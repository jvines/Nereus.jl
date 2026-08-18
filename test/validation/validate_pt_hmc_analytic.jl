#!/usr/bin/env julia
# Analytic-log-Z gate for the PRODUCTION PT-HMC engine (`_pt_hmc_core`),
# including the adaptive ladder. Unlike test_evidence_gaussian.jl (which feeds a
# self-contained stretch-move ptemcee into the EvidenceAccumulator to validate
# the ESTIMATORS), this drives the actual Hamiltonian-PT code path on a target
# with a closed-form evidence, so it certifies the SAMPLER + LADDER, not just
# the integrator.
#
# Target (everything Gaussian → exact logZ, smooth for HMC, no box walls):
#   prior      = N(x | 0, τ²I)                      (proper, normalised)
#   likelihood = N(x | μ, Σ)                        (normalised)
#   evidence   Z = N(μ | 0, Σ + τ²I)
#   log Z = -½ μᵀ(Σ+τ²I)⁻¹μ - ½ d·log(2π) - ½ log|Σ+τ²I|
# TI identity: prior normalised ⇒ log Z = ∫₀¹ ⟨log L⟩_β dβ exactly.

using Random
using LinearAlgebra
using Printf
using Statistics: mean

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Nereus: _pt_hmc_core, _adapt_ladder

# ----- target -----------------------------------------------------------
const d   = 4
const τ   = 4.0
const μ   = [3.0, -1.5, 2.0, 0.5]
const Σ   = [1.0 0.4 0.0 0.2;
             0.4 2.0 0.3 0.0;
             0.0 0.3 1.5 0.5;
             0.2 0.0 0.5 1.2]
const Σinv     = inv(Σ)
const logdetΣ  = logdet(Σ)
const half_dlog2π = 0.5 * d * log(2π)

logprior(y) = -0.5 * dot(y, y) / τ^2 - 0.5 * d * log(2π * τ^2)
function loglike(y)
    Δ = y .- μ
    return -0.5 * (Δ' * Σinv * Δ) - half_dlog2π - 0.5 * logdetΣ
end
parts(y) = (logprior(y), loglike(y))

# analytic evidence + posterior
const A        = Σ + τ^2 * Matrix(I, d, d)
const logZ_an  = -0.5 * (μ' * (A \ μ)) - half_dlog2π - 0.5 * logdet(A)
const Spost    = inv(inv(Σ) + (1 / τ^2) * Matrix(I, d, d))
const mpost    = Spost * (Σinv * μ)

# ----- run --------------------------------------------------------------
const n_temps  = 14
const W        = 2
rng = MersenneTwister(20260623)

βs0 = Float64[((i - 1) / (n_temps - 1))^2 for i in 1:n_temps]
prior_inits() = [τ .* randn(rng, d) for _ in 1:(n_temps * W)]

@printf("=== PT-HMC analytic-logZ gate (%d-D Gaussian×Gaussian) ===\n", d)
@printf("analytic log Z = %.4f\n", logZ_an)

t0 = time()
# pilot → adapt ladder → final  (mirrors sample_pt_hmc's adaptive path)
pilot = _pt_hmc_core(parts, d, prior_inits(), βs0;
                     n_sweeps = 250, n_warmup = 200, swap_interval = 1,
                     target_accept = 0.8, W = W, progress = false, rng = rng,
                     label = "pilot")
βs = _adapt_ladder(βs0, pilot.varlogL)
@printf("pilot logZ(TI+) = %.4f ; ladder re-gridded\n", pilot.report.ti_plus[1])

res = _pt_hmc_core(parts, d, prior_inits(), βs;
                   n_sweeps = 1500, n_warmup = 400, swap_interval = 1,
                   target_accept = 0.8, W = W, progress = false, rng = rng,
                   label = "final")
@printf("PT-HMC done in %.1fs\n\n", time() - t0)

rep = res.report
# posterior recovery on cold chain
Y = reduce(hcat, res.cold_ys)
m_est = vec(mean(Y, dims = 2))
@printf("posterior mean: est=%s\n              truth=%s\n",
        round.(m_est, digits = 3), round.(mpost, digits = 3))
mean_err = maximum(abs.(m_est .- mpost))

@printf("\nEvidence vs analytic (%.4f):\n", logZ_an)
@printf("  TI  = %9.4f                (err %.4f)\n", rep.ti[1], abs(rep.ti[1] - logZ_an))
@printf("  TI+ = %9.4f ± %.4f      (err %.4f)\n", rep.ti_plus[1], rep.ti_plus[2], abs(rep.ti_plus[1] - logZ_an))
@printf("  SS+ = %9.4f ± %.4f      (err %.4f)\n", rep.ss_plus[1], rep.ss_plus[2], abs(rep.ss_plus[1] - logZ_an))
@printf("  H+  = %9.4f ± %.4f      (err %.4f)\n", rep.hybrid[1], rep.hybrid[2], abs(rep.hybrid[1] - logZ_an))

const TOL_Z = 1.0   # nats — sampler+ladder noise, looser than the estimator-only test
const TOL_M = 0.20  # posterior-mean recovery
errs = abs.([rep.ti[1], rep.ti_plus[1], rep.ss_plus[1], rep.hybrid[1]] .- logZ_an)
spread = maximum([rep.ti[1], rep.ti_plus[1], rep.ss_plus[1], rep.hybrid[1]]) -
         minimum([rep.ti[1], rep.ti_plus[1], rep.ss_plus[1], rep.hybrid[1]])

@printf("\nmax |Z err| = %.4f (tol %.2f) ; estimator spread = %.4f ; mean err = %.4f (tol %.2f)\n",
        maximum(errs), TOL_Z, spread, mean_err, TOL_M)
ok = maximum(errs) < TOL_Z && spread < TOL_Z && mean_err < TOL_M
@printf("\nVerdict: %s\n", ok ? "✅ PT-HMC ANALYTIC GATE PASS" : "❌ FAIL")
exit(ok ? 0 : 1)
