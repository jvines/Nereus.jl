#!/usr/bin/env julia
# Analytic-log-Z validation for the reddemcee TI+/SS+/H+ estimators.
#
# Target:
#   prior     = Uniform([-L, L]^d)              (d=2, L=10)
#   likelihood = N(x | μ, Σ)  (already normalized)
#
# With μ well inside the box and Σ small compared to L, essentially all
# of the Gaussian mass lives inside the prior box. The evidence is then
#
#   Z = ∫ p(x) L(x) dx = (1/(2L)^d) · ∫ N(x|μ,Σ) dx ≈ (2L)^(-d)
#   log Z ≈ -d · log(2L)
#
# For d=2, L=10 → log Z ≈ -2 · log 20 ≈ -5.991.
#
# We run the same self-contained ptemcee from test_ptemcee_gaussian.jl,
# feed every post-burnin (temp × step × walker) log-likelihood into an
# EvidenceAccumulator, then compare TI, TI+, SS+, H+ to the analytic
# value. Tight ladders (n_temps = 12) and long chains give the
# estimators the chance to actually converge — the HD 18599 run that
# motivated this test used only 5 temps × 2000 steps and disagreed by
# ~2000 nats. If THIS test agrees, the HD 18599 disagreement is a
# sampling / convergence problem, not a bug in the estimator code.

using Random
using Statistics: mean, std, cov
using LinearAlgebra: det, inv
using Printf

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Nereus: EvidenceAccumulator, update_evidence!, evidence_report,
               ti_trapezoidal, ti_plus, ss_plus, hybrid_evidence,
               _pchip_integral

# ----- Standalone ptemcee (verbatim copy of test_ptemcee_gaussian) -----
# Same algorithm, but additionally returns per-(temp, step, walker) log L
# so we can feed the evidence accumulator from the same run.
function ptemcee_evidence(log_prior::Function, log_like::Function,
                          init_walkers::Array{Float64,3};
                          betas::Vector{Float64},
                          n_steps::Int, n_burnin::Int,
                          stretch_a::Float64 = 2.0,
                          seed::Int = 1)
    n_temps, n_walkers, n_dim = size(init_walkers)
    @assert n_walkers % 2 == 0 "n_walkers must be even"
    half = n_walkers ÷ 2

    state    = copy(init_walkers)
    logπ_arr = fill(-Inf, n_temps, n_walkers)
    logL_arr = fill(-Inf, n_temps, n_walkers)
    for t in 1:n_temps, w in 1:n_walkers
        x = view(state, t, w, :)
        lp = log_prior(x)
        if isfinite(lp)
            ll = log_like(x)
            if isfinite(ll)
                logπ_arr[t, w] = lp
                logL_arr[t, w] = ll
            end
        end
    end

    rng = MersenneTwister(seed)

    n_keep = n_steps - n_burnin
    samples  = Array{Float64,3}(undef, n_keep, n_walkers, n_dim)
    logL_log = Array{Float64,3}(undef, n_keep, n_temps, n_walkers)

    for step in 1:n_steps
        for (active, inactive) in ((1:half, (half+1):n_walkers),
                                    ((half+1):n_walkers, 1:half))
            for t in 1:n_temps
                β = betas[t]
                for w in active
                    w_partner = rand(rng, inactive)
                    u = rand(rng)
                    z = ((stretch_a - 1) * u + 1)^2 / stretch_a
                    proposal = Vector{Float64}(undef, n_dim)
                    @inbounds for d in 1:n_dim
                        proposal[d] = state[t, w_partner, d] +
                                      z * (state[t, w, d] - state[t, w_partner, d])
                    end
                    lp_prop = log_prior(proposal)
                    if isfinite(lp_prop)
                        ll_prop = log_like(proposal)
                        log_ratio = (n_dim - 1) * log(z) +
                                    β * (ll_prop - logL_arr[t, w]) +
                                    (lp_prop - logπ_arr[t, w])
                        if log(rand(rng)) < log_ratio
                            @inbounds for d in 1:n_dim
                                state[t, w, d] = proposal[d]
                            end
                            logπ_arr[t, w] = lp_prop
                            logL_arr[t, w] = ll_prop
                        end
                    end
                end
            end
        end

        for t in 1:(n_temps - 1)
            Δβ = betas[t] - betas[t + 1]
            for w in 1:n_walkers
                w2 = rand(rng, 1:n_walkers)
                ll1 = logL_arr[t, w]
                ll2 = logL_arr[t + 1, w2]
                log_ratio = Δβ * (ll2 - ll1)
                if log(rand(rng)) < log_ratio
                    for d in 1:n_dim
                        tmp = state[t, w, d]
                        state[t, w, d]      = state[t + 1, w2, d]
                        state[t + 1, w2, d] = tmp
                    end
                    lp_t                  = logπ_arr[t, w]
                    logπ_arr[t, w]        = logπ_arr[t + 1, w2]
                    logπ_arr[t + 1, w2]   = lp_t
                    logL_arr[t, w]        = ll2
                    logL_arr[t + 1, w2]   = ll1
                end
            end
        end

        if step > n_burnin
            keep_step = step - n_burnin
            for w in 1:n_walkers, d in 1:n_dim
                samples[keep_step, w, d] = state[1, w, d]
            end
            for t in 1:n_temps, w in 1:n_walkers
                logL_log[keep_step, t, w] = logL_arr[t, w]
            end
        end
    end

    return (samples = samples, logL_log = logL_log)
end

# ----- 2D Gaussian target ------------------------------------------
const μ_true = [3.0, -1.5]
const Σ_true = [1.0 0.5; 0.5 2.0]
const Σ_inv  = inv(Σ_true)
const log_det_Σ = log(det(Σ_true))
const lo = [-10.0, -10.0]
const hi = [10.0, 10.0]
const d_dim = 2
const log_Z_analytic = -d_dim * log(hi[1] - lo[1])  # = -2 log 20

log_prior_box(x) = all(lo .≤ x .≤ hi) ? 0.0 : -Inf
function log_like_gauss(x)
    Δ = x .- μ_true
    return -0.5 * (Δ' * Σ_inv * Δ) - 0.5 * length(x) * log(2π) - 0.5 * log_det_Σ
end

# ----- Configuration -----------------------------------------------
# A tighter ladder than the HD 18599 run (12 vs 5) to give TI+ enough
# resolution for its curvature estimate. Geometric schedule β_k = r^k.
const n_temps   = 12
const n_walkers = 100
const n_steps   = 3000
const n_burnin  = 500
const r_β       = 0.6   # β ranges from 1 to 0.6^11 ≈ 0.0036

betas_run = Float64[r_β^k for k in 0:(n_temps - 1)]  # decreasing β with index
# EvidenceAccumulator expects β increasing along chain index — flip.
# Internally the sampler uses (cold=β=1, hot=β=0) ordering; the
# accumulator can take either as long as betas is sorted ascending.
# We'll sort ascending when feeding the accumulator.

rng_init = MersenneTwister(42)
init_walkers = Array{Float64,3}(undef, n_temps, n_walkers, d_dim)
for t in 1:n_temps, w in 1:n_walkers
    init_walkers[t, w, 1] = lo[1] + (hi[1] - lo[1]) * rand(rng_init)
    init_walkers[t, w, 2] = lo[2] + (hi[2] - lo[2]) * rand(rng_init)
end

@printf("=== TI+/SS+/H+ analytic-Z validation (2D Gaussian × uniform box) ===\n")
@printf("μ_true = %s\n", μ_true)
@printf("Σ_true = %s\n", Σ_true)
@printf("Prior box: [%s, %s]^%d\n", lo[1], hi[1], d_dim)
@printf("n_temps=%d  n_walkers=%d  n_steps=%d  burnin=%d\n",
        n_temps, n_walkers, n_steps, n_burnin)
@printf("β-ladder: %s\n",
        join(map(b -> @sprintf("%.4f", b), betas_run), ", "))
@printf("Analytic log Z = %.6f\n\n", log_Z_analytic)

t0 = time()
res = ptemcee_evidence(log_prior_box, log_like_gauss, init_walkers;
                       betas    = betas_run,
                       n_steps  = n_steps,
                       n_burnin = n_burnin,
                       seed     = 7)
@printf("ptemcee done in %.1fs\n", time() - t0)

# ----- Posterior recovery sanity -----------------------------------
flat = reshape(res.samples, :, d_dim)
μ_est = vec(mean(flat, dims = 1))
Σ_est = cov(flat)
@printf("\nRecovered μ = [%.3f, %.3f]   (truth: [%.3f, %.3f])\n",
        μ_est[1], μ_est[2], μ_true[1], μ_true[2])
@printf("Recovered Σ:  [%.3f %.3f; %.3f %.3f]   (truth: [%.3f %.3f; %.3f %.3f])\n",
        Σ_est[1,1], Σ_est[1,2], Σ_est[2,1], Σ_est[2,2],
        Σ_true[1,1], Σ_true[1,2], Σ_true[2,1], Σ_true[2,2])

# ----- Evidence accumulator: feed every (step × walker), reordered to
# ascending β (so β[1] = 0.0036 → β[n_temps] = 1.0). Nereus's
# update_evidence! expects ascending β with log_L_per_chain[k] tied to
# betas[k] in that order.
ascending_idx = sortperm(betas_run)
betas_asc = betas_run[ascending_idx]

# Concatenate β-tail (β≈0) by padding accumulator's first slot with the
# prior expectation: at β=0 the tempered density is the prior itself,
# and ⟨log L⟩_prior is a finite number we can compute by Monte Carlo
# from uniform draws. We append β=0 to the ladder so TI integrates over
# the full [0, 1].
const N_PRIOR_DRAW = 50_000
rng_pr = MersenneTwister(99)
prior_logL = Vector{Float64}(undef, N_PRIOR_DRAW)
for i in 1:N_PRIOR_DRAW
    x = (lo[1] + (hi[1] - lo[1]) * rand(rng_pr),
         lo[2] + (hi[2] - lo[2]) * rand(rng_pr))
    prior_logL[i] = log_like_gauss(collect(x))
end

# Build an extended ladder β = [0, betas_asc...]. The accumulator must
# stream log L at β=0; we synthesize those from the prior_logL vector.
betas_ext = vcat(0.0, betas_asc)
n_chains_acc = length(betas_ext)
acc = EvidenceAccumulator(n_chains_acc)

# β=0 first, then β>0 from the ptemcee log
@printf("Feeding accumulator: %d samples per temp\n",
        size(res.logL_log, 1) * n_walkers)

# We'll feed pseudo-synchronous tuples (one log L per temp) per
# (step, walker) — matching how `_sample_pt_transdim` calls
# `update_evidence!`. For β=0 we re-sample a prior log L per call.
n_keep = size(res.logL_log, 1)
for step in 1:n_keep
    for w in 1:n_walkers
        # Build tuple [β=0 logL, β=β_min logL, ..., β=1 logL]
        log_L = Vector{Float64}(undef, n_chains_acc)
        # β = 0: independent prior draw each call
        log_L[1] = prior_logL[mod1((step - 1) * n_walkers + w, N_PRIOR_DRAW)]
        # β > 0: pull from logL_log in ascending β order
        for (j, t_idx) in enumerate(ascending_idx)
            log_L[j + 1] = res.logL_log[step, t_idx, w]
        end
        update_evidence!(acc, log_L, betas_ext)
    end
end

rep = evidence_report(acc, betas_ext)
@printf("\nEvidenceReport (β-ladder length = %d, anchored at β=0):\n", n_chains_acc)
@printf("  TI  (trapezoidal) = %12.6f                       (truth %.6f)\n",
        rep.ti[1], log_Z_analytic)
@printf("  TI+ (PCHIP)       = %12.6f ± %.6f          (truth %.6f)\n",
        rep.ti_plus[1], rep.ti_plus[2], log_Z_analytic)
@printf("  SS+ (geom-bridge) = %12.6f ± %.6f          (truth %.6f)\n",
        rep.ss_plus[1], rep.ss_plus[2], log_Z_analytic)
@printf("  H+  (β* = %.4f)  = %12.6f ± %.6f          (truth %.6f)\n",
        rep.hybrid_beta_star, rep.hybrid[1], rep.hybrid[2], log_Z_analytic)

# ----- Verdict ----------------------------------------------------
const TOL = 0.5  # nats — generous; tight ladders + MC noise allow ~0.1
err_ti   = abs(rep.ti[1]      - log_Z_analytic)
err_tip  = abs(rep.ti_plus[1] - log_Z_analytic)
err_ssp  = abs(rep.ss_plus[1] - log_Z_analytic)
err_hp   = abs(rep.hybrid[1]  - log_Z_analytic)

@printf("\n|errors|: TI=%.4f  TI+=%.4f  SS+=%.4f  H+=%.4f  (tol=%.2f)\n",
        err_ti, err_tip, err_ssp, err_hp, TOL)

verdict_individual = (err_ti < TOL, err_tip < TOL, err_ssp < TOL, err_hp < TOL)
@printf("Per-estimator PASS: TI=%s  TI+=%s  SS+=%s  H+=%s\n",
        verdict_individual...)

# Mutual agreement check: the three estimators should agree with each
# other within their reported errors. This is the actual #11 question.
spread = maximum([rep.ti[1], rep.ti_plus[1], rep.ss_plus[1], rep.hybrid[1]]) -
         minimum([rep.ti[1], rep.ti_plus[1], rep.ss_plus[1], rep.hybrid[1]])
@printf("Spread across estimators: %.4f nats\n", spread)

ok = all(verdict_individual) && spread < 1.0
@printf("\nVerdict: %s\n",
        ok ? "PASS — estimators agree with each other AND with analytic Z" :
             "FAIL — see above")

# ---------------------------------------------------------------------
# REGRESSION (2026-06-13 sign-flip post-mortem): the TI integrators must
# be β-ORDER-AGNOSTIC. Production samplers feed β DESCENDING (1→0,
# cold→hot); pre-fix the integrators walked the array in order →
# ∫₁⁰⟨logL⟩dβ = −logZ, a sign flip that inverted every model-comparison
# Bayes factor. This test was blind to it because it only fed ascending
# β. Deterministic linear-⟨logL⟩ check: ∫₀¹(a+bβ)dβ = a + b/2, exact.
# ---------------------------------------------------------------------
let
    βasc  = collect(0.0:0.1:1.0)
    a0, b0 = -150.0, -8.0                 # ⟨logL⟩(β) = a0 + b0·β (both<0)
    masc  = a0 .+ b0 .* βasc
    exact = a0 + b0 / 2                    # = -154.0
    βdesc = reverse(βasc); mdesc = reverse(masc)
    ti_asc  = ti_trapezoidal(masc,  βasc)
    ti_desc = ti_trapezoidal(mdesc, βdesc)
    pc_asc  = _pchip_integral(βasc,  masc)
    pc_desc = _pchip_integral(βdesc, mdesc)
    @printf("\n=== β-order-agnostic regression (exact ∫ = %.1f) ===\n", exact)
    @printf("  ti_trapezoidal  asc=%.4f  desc=%.4f\n", ti_asc, ti_desc)
    @printf("  _pchip_integral asc=%.4f  desc=%.4f\n", pc_asc, pc_desc)
    order_ok = all(<(1e-6), abs.([ti_asc, ti_desc, pc_asc, pc_desc] .- exact))
    @printf("  order-agnostic + correct sign: %s\n", order_ok)
    global ok = ok && order_ok
end
