# Simulation-Based Calibration (SBC) for Nereus's samplers.
#
# Talts, Betancourt, Simpson, Vehtari & Gelman (2018), "Validating
# Bayesian Inference Algorithms with Simulation-Based Calibration"
# (arXiv:1804.06788).
#
# The third rung of the validation ladder (after bare-algorithm Gaussian
# tests and the production-path recovery matrix in
# `sampler_validation.jl`). Recovery-or-fail-loud only asks "does truth
# land inside the CI"; it has almost no power against a posterior whose
# *width* is wrong. SBC closes that gap.
#
# SBC THEOREM. If θ̃ ~ p(θ), ỹ ~ p(y | θ̃), and {θ_1…θ_L} are L INDEPENDENT
# draws from the posterior p(θ | ỹ) the algorithm produces, then the rank
#       r = #{ l : θ_l < θ̃ }   ∈  {0, …, L}
# is uniform on {0, …, L} — EXACTLY, for a faithful sampler. Deviations:
#   ∪-shaped (var > discrete-uniform) → posterior too NARROW → OVERCONFIDENT
#   ∩-shaped (var < discrete-uniform) → posterior too WIDE   → UNDERCONFIDENT
#   sloped / mean ≠ 0.5               → BIASED.
# So SBC is a goodness-of-fit test of the algorithm's output against the
# intractable posterior the model+prior define — it certifies the machine,
# not the answer, and only under a correct model (white noise here).
#
# ─────────────────────────────────────────────────────────────────────
# This harness was hardened after an adversarial correctness review that
# found (and numerically confirmed) several ways the naive version emits
# WRONG verdicts. The fixes, all implemented below:
#
#  [common-L]  Every kept sim is thinned to the SAME L, and (L+1) % n_bins
#     == 0, so each histogram bin spans an equal integer count of the L+1
#     reachable ranks. Otherwise low/variable L + fixed bins gives a ~29%
#     FALSE-miscalibrated rate for a PERFECT sampler (worse as n_sims ↑).
#     Binning uses ⌊r·n_bins/(L+1)⌋. The old round()-rescale is gone.
#  [ess-honest]  L draws are kept ONLY when the family-aware independent
#     count (min-ESS) ≥ L; a sim with too few independent draws is COUNTED
#     as `insufficient`, never padded with autocorrelated draws (which
#     fabricate ∪/∩) and never silently kept whole on a NaN ESS.
#  [discrete-var]  The ∪/∩ z-score compares the sample var of r/L against
#     the DISCRETE-uniform variance (1/12)(1+2/L), not the continuous 1/12
#     (which biases toward false OVERCONFIDENT at finite L).
#  [survivorship]  Failed sims are counted; if the failure (or
#     insufficient-ESS) fraction is non-negligible the verdict is
#     FAILED-TO-RUN / INSUFFICIENT, NOT a green CALIBRATED — θ-correlated
#     failures would otherwise certify a sampler over only the easy slice.
#  [bonferroni]  The family verdict uses a Šidák/Bonferroni-corrected α
#     over the period-invariant SAMPLED params; the derived eccentricity
#     e_k1 is reported as a diagnostic, NOT triple-counted with its
#     sesinw/secosw coordinates.
#  [no-Tp]  Tp is identifiable only modulo P and its prior spans many
#     periods, so its ABSOLUTE-value rank is non-uniform even for a
#     faithful sampler — Tp is excluded from the calibration verdict (a
#     wrapped-phase rank is the alternative; noted at `SBC_RANK_PARAMS`).
#  [finite-filter]  Non-finite posterior draws are filtered (NaN < v is
#     false in Julia → would deflate ranks) and a NaN-heavy sim is dropped.
#  [coverage]  Every verdict param must be scored on every used sim, else
#     it surfaces as UNTESTED — a renamed/empty slot cannot pass silently.
#  [nested-gated]  The weight-sorted dead-point fallback that nested*
#     return at low n_eff is NOT a posterior draw set; nested-family SBC is
#     gated pending an n_eff/resample flag in chains.info (see run_sbc).
#
# GATE: `sbc_self_test()` now exercises BOTH the analytic i.i.d. kernel
# AND the production glue (the chosen (L,n_bins) gives ~nominal — not 29% —
# false-reject for a perfect sampler; NaN draws are filtered; mis-sized
# draws are flagged ∪/∩). Trust no sampler verdict until it passes.

include(joinpath(@__DIR__, "sampler_validation.jl"))

using Distributions: Chisq, ccdf
using Random, Statistics, Printf

# Period-invariant SAMPLED parameters scored for the calibration verdict.
# Tp is EXCLUDED (modulo-P degeneracy → absolute-rank non-uniform even for
# a faithful sampler; rank mod P if you ever want it). e_k1 is reported
# separately as a derived diagnostic, not counted in the family α (it is a
# deterministic function of sesinw/secosw, already in the family).
const SBC_VERDICT_PARAMS = ["P_k1", "K_k1", "sesinw_k1", "secosw_k1",
                            "gamma_SIM", "sigma_SIM"]
const SBC_DIAG_PARAMS = ["e_k1"]
const SBC_NESTED_FAMILY = Set(["nested", "nested_ins", "nested_dynamic"])

# =====================================================================
# 1. Rank core + uniformity test (kept exact; common-L, discrete-aware)
# =====================================================================

"""
    sbc_rank(v, draws) -> Int

SBC rank of the true value `v` among posterior `draws`: the count of
draws strictly below `v`, an integer in `0:length(draws)`. Assumes
continuous draws (ties measure-zero under the white-noise model); a
parameter with point mass at a bound would need tie randomisation.
"""
sbc_rank(v::Real, draws::AbstractVector{<:Real}) = count(<(v), draws)

"""
    uniformity_test(ranks, L; n_bins) -> NamedTuple

χ² test that SBC `ranks` (ALL on the same `0:L`) are uniform, plus the
discrete-uniform-aware shape moments. Requires `(L+1) % n_bins == 0` so
each bin spans an equal integer number of the `L+1` reachable ranks —
this is what removes the low-L discreteness false-positive. Binning is
`⌊r·n_bins/(L+1)⌋`. `mean`/`var` are of `r/L`; the calibrated reference
is mean `0.5`, var `(1/12)(1+2/L)` (discrete, NOT the continuous `1/12`).
"""
function uniformity_test(ranks::Vector{Int}, L::Int; n_bins::Int = 10)
    N = length(ranks)
    var_ref = (1/12) * (1 + 2/L)
    if N == 0
        return (chi2 = NaN, pval = NaN, counts = zeros(Int, n_bins),
                mean = NaN, var = NaN, var_ref = var_ref, L = L, n = 0)
    end
    (L + 1) % n_bins == 0 ||
        error("uniformity_test: need (L+1) % n_bins == 0 (L=$L, n_bins=$n_bins)")
    counts = zeros(Int, n_bins)
    for r in ranks
        b = clamp(div(r * n_bins, L + 1) + 1, 1, n_bins)
        counts[b] += 1
    end
    E = N / n_bins
    chi2 = sum((c - E)^2 / E for c in counts)
    pval = ccdf(Chisq(n_bins - 1), chi2)
    norm = Float64[r / L for r in ranks]
    return (chi2 = chi2, pval = pval, counts = counts,
            mean = mean(norm), var = var(norm), var_ref = var_ref,
            L = L, n = N)
end

"""
    sbc_shape(t) -> String

Verdict from a `uniformity_test` result, with N-aware z-scores for the
mean (bias) and variance (∪/∩) against the DISCRETE-uniform references.
"""
function sbc_shape(t)
    (isnan(t.pval) || t.n == 0) && return "no-data"
    N = t.n
    z_mean = (t.mean - 0.5) / sqrt(t.var_ref / N)
    # se of the sample variance: use the empirical 4th central moment.
    se_var = sqrt(max(t.var_ref^2 * 2 / (N - 1), 1e-300))   # ~Normal-ish floor
    z_var = (t.var - t.var_ref) / se_var
    flags = String[]
    abs(z_mean) > 3 && push!(flags, @sprintf("BIASED(mean=%.3f, z=%.1f)", t.mean, z_mean))
    z_var >  3 && push!(flags, @sprintf("OVERCONFIDENT ∪ (var=%.4f vs %.4f, z=%.1f)",
                                        t.var, t.var_ref, z_var))
    z_var < -3 && push!(flags, @sprintf("UNDERCONFIDENT ∩ (var=%.4f vs %.4f, z=%.1f)",
                                        t.var, t.var_ref, z_var))
    if isempty(flags)
        return t.pval > 0.01 ? "CALIBRATED" :
               @sprintf("non-uniform (χ²p=%.1e) but no clear shape", t.pval)
    end
    return join(flags, "; ")
end

function sbc_ascii_hist(counts::Vector{Int}; width::Int = 40)
    m = maximum(counts); m == 0 && return "(empty)"
    join([@sprintf("  [%2d] %4d |%s", i, c, "#"^round(Int, width * c / m))
          for (i, c) in enumerate(counts)], "\n")
end

# =====================================================================
# 2. Self-test — analytic kernel control + production-glue controls (GATE)
# =====================================================================

function _sbc_gaussian_ranks(c::Float64, n_sims::Int, L::Int, rng;
                             τ = 3.0, σ = 1.0, n = 10)
    ranks = Vector{Int}(undef, n_sims)
    for k in 1:n_sims
        θ = τ * randn(rng)
        y = θ .+ σ .* randn(rng, n)
        s2 = 1 / (1/τ^2 + n/σ^2); μ = s2 * (sum(y) / σ^2)
        ranks[k] = sbc_rank(θ, μ .+ (c * sqrt(s2)) .* randn(rng, L))
    end
    return ranks
end

# False-reject rate of the χ² test for a PERFECT (exact-uniform) sampler at
# (L, n_bins). With (L+1)%n_bins==0 this must be ~nominal; the unfixed
# version measured ~29% at L=30,n_bins=20.
function _sbc_false_reject_rate(L::Int, n_bins::Int; trials = 400,
                                n_sims = 200, α = 0.01, rng = MersenneTwister(7))
    # Inline binning (same ⌊r·n_bins/(L+1)⌋ rule) so this can evaluate ANY
    # (L, n_bins) — including the unfixed (30,20) the hardened
    # `uniformity_test` now refuses — to demonstrate the discreteness effect.
    rej = 0
    for _ in 1:trials
        ranks = rand(rng, 0:L, n_sims)            # exact discrete uniform
        counts = zeros(Int, n_bins)
        for r in ranks
            counts[clamp(div(r * n_bins, L + 1) + 1, 1, n_bins)] += 1
        end
        E = n_sims / n_bins
        chi2 = sum((c - E)^2 / E for c in counts)
        ccdf(Chisq(n_bins - 1), chi2) < α && (rej += 1)
    end
    return rej / trials
end

"""
    sbc_self_test(; n_sims=2000, L=199, n_bins=10, seed=1) -> Bool

Gate. Validates (a) the analytic i.i.d. kernel — exact posterior reads
CALIBRATED, 0.5×/2× width read ∪/∩ — AND (b) the production glue: the
chosen (L, n_bins) gives ~nominal false-reject for a perfect sampler (the
discreteness fix), and non-finite draws are filtered rather than deflating
ranks. Prints a report; returns `true` iff all controls behave.
"""
function sbc_self_test(; n_sims::Int = 2000, L::Int = 199, n_bins::Int = 10,
                       seed::Int = 1)
    rng = MersenneTwister(seed)
    tp = uniformity_test(_sbc_gaussian_ranks(1.0, n_sims, L, rng), L; n_bins)
    tn = uniformity_test(_sbc_gaussian_ranks(0.5, n_sims, L, rng), L; n_bins)
    tw = uniformity_test(_sbc_gaussian_ranks(2.0, n_sims, L, rng), L; n_bins)

    println("="^72)
    println("SBC SELF-TEST  (analytic kernel + glue;  N=$n_sims, L=$L, n_bins=$n_bins)")
    println("="^72)
    for (lbl, t) in (("POSITIVE (exact i.i.d.)", tp),
                     ("NEGATIVE narrow 0.5×",    tn),
                     ("NEGATIVE wide   2.0×",    tw))
        @printf("%-26s χ²p=%.2e  mean=%.3f  var=%.4f (ref %.4f)  → %s\n",
                lbl, t.pval, t.mean, t.var, t.var_ref, sbc_shape(t))
    end
    ok_pos    = tp.pval > 0.01
    ok_narrow = tn.pval < 1e-3 && tn.var > tn.var_ref
    ok_wide   = tw.pval < 1e-3 && tw.var < tw.var_ref

    # Glue control 1: discreteness false-reject must be ~nominal.
    fr_good = _sbc_false_reject_rate(L, n_bins)
    fr_bad  = _sbc_false_reject_rate(30, 20)       # the unfixed regime
    ok_disc = fr_good < 0.05
    @printf("\nDISCRETENESS  false-reject @ (L=%d,bins=%d)=%.3f  (nominal≈0.01)  |  unfixed (L=30,bins=20)=%.3f\n", L, n_bins, fr_good, fr_bad)

    # Glue control 2: NaN draws must be filtered, not deflate ranks.
    rng2 = MersenneTwister(99); nan_ranks = Int[]
    for _ in 1:n_sims
        θ = randn(rng2); draws = θ .+ randn(rng2, 2L)
        half = vcat(draws[1:L], fill(NaN, L))            # 50% NaN
        clean = filter(isfinite, half)
        push!(nan_ranks, sbc_rank(θ, clean))
    end
    tnan = uniformity_test(nan_ranks, L; n_bins)
    ok_nan = abs(tnan.mean - 0.5) < 0.05
    @printf("NaN-FILTER     mean rank after filtering 50%% NaN = %.3f (want ≈0.5)\n", tnan.mean)

    ok = ok_pos && ok_narrow && ok_wide && ok_disc && ok_nan
    println("-"^72)
    @printf("pos→unif:%s  narrow→∪:%s  wide→∩:%s  discreteness-ok:%s  nan-filter-ok:%s\n",
            ok_pos, ok_narrow, ok_wide, ok_disc, ok_nan)
    println(ok ? "✅ SBC harness VALIDATED — kernel + glue correct" :
                 "❌ SBC harness BROKEN — do NOT trust any sampler verdict")
    println("="^72)
    return ok
end

# =====================================================================
# 3. Nereus SBC — single-planet RV generative model
# =====================================================================
# (Generative cycle below was adversarially reviewed and CONFIRMED correct:
#  prior is data-independent, draw-prior == fit-prior incl. e<1 support,
#  the noise model matches the likelihood, all in bounded/physical space.)

function sbc_overrides()
    return Dict{String, PriorSpec}(
        "P_k1"      => LogUniformPrior(6.0, 24.0),
        "K_k1"      => UniformPrior(0.0, 100.0),
        "gamma_SIM" => UniformPrior(-30.0, 30.0),
        "sigma_SIM" => UniformPrior(0.0, 10.0),
    )
end

function sbc_design(; n::Int = 40, baseline::Float64 = 120.0,
                     rv_err::Float64 = 2.0, seed::Int = 777)
    rng = MersenneTwister(seed)
    return sort(rand(rng, n) .* baseline), fill(rv_err, n)
end

function _sbc_simulate(template_params::Params, template_data::Data,
                       unfrozen_names::Vector{String},
                       theta_b::Vector{Float64}, rng::AbstractRNG)
    th = Theta{Float64}(template_params)
    for (nm, v) in zip(unfrozen_names, theta_b)
        set_param!(th, nm, v)
    end
    preds, _ = rv_predictions(th, template_data)
    jit = theta_b[findfirst(==("sigma_SIM"), unfrozen_names)]
    σ_tot = sqrt.(template_data.rv_err .^ 2 .+ jit^2)
    return preds .+ σ_tot .* randn(rng, length(preds))
end

# Thin a posterior draw vector to EXACTLY `L` ~independent finite draws, or
# return `nothing` when the chain has too few independent draws. `min_ess`
# is the family-aware independent count from `_convergence_diagnostics`.
function _sbc_thin(draws::AbstractVector{<:Real}, min_ess::Real, L::Int)
    finite = filter(isfinite, draws)
    n = length(finite)
    # Need ≥ L genuinely independent draws — no floor that exceeds ESS, and
    # never keep a whole autocorrelated chain on a NaN/zero ESS.
    (isfinite(min_ess) && min_ess >= L && n >= L) || return nothing
    idx = unique(round.(Int, range(1, n; length = L)))
    length(idx) == L || return nothing
    return finite[idx]
end

"""
    run_sbc(sampler; n_sims=128, L=99, n_bins=10, seed, verbose) -> NamedTuple

SBC for one Nereus sampler on the single-planet RV model. Each sim: draw
θ̃ ~ prior, simulate ỹ, fit, thin the posterior to EXACTLY `L` independent
draws (skip the sim — counted `insufficient` — if it cannot reach `L`),
and rank θ̃ in every parameter. Returns ranks, the per-param
`uniformity_test`s, and the run accounting (`n_ok`/`n_fail`/`n_insuff`)
needed for an honest verdict. Requires `(L+1) % n_bins == 0`.
"""
function run_sbc(sampler::String; n_sims::Int = 128, L::Int = 99,
                 n_bins::Int = 10, seed::Int = 20240602, verbose::Bool = true)
    (L + 1) % n_bins == 0 ||
        error("run_sbc: need (L+1) % n_bins == 0 (L=$L, n_bins=$n_bins)")
    if sampler in SBC_NESTED_FAMILY
        error("run_sbc: nested-family SBC is gated — at low n_eff " *
              "sample_$sampler returns weight-SORTED dead points (not posterior " *
              "draws). Surface n_eff/resample_ok in chains.info and skip the " *
              "fallback before SBC-ing $sampler.")
    end

    t, rv_err = sbc_design()
    ov = sbc_overrides()
    tmpl_params, tmpl_data, tmpl_target =
        _build_rv_target(t, zeros(length(t)), rv_err, 1, ov)
    nms = tmpl_params.layout.unfrozen_names
    rank_params = vcat(SBC_VERDICT_PARAMS, SBC_DIAG_PARAMS)
    @assert all(p -> p == "e_k1" || p in nms, rank_params) "rank param not in model"

    ranks = Dict(k => Int[] for k in rank_params)
    n_ok = 0; n_fail = 0; n_insuff = 0

    for s in 1:n_sims
        rng = MersenneTwister(seed + s)
        θb  = Nereus._draw_from_prior(tmpl_target, rng)
        y   = _sbc_simulate(tmpl_params, tmpl_data, nms, θb, rng)
        fit_params, fit_data, fit_target = _build_rv_target(t, y, rv_err, 1, ov)
        r = run_sampler(sampler, fit_params, fit_data, fit_target)
        if !r.ok || r.chains === nothing
            n_fail += 1
            verbose && @printf("  sim %3d  FAIL: %s\n", s, r.ok ? "no chains" : r.err)
            continue
        end
        _, min_ess, _ = _convergence_diagnostics(sampler, r.chains)
        # All params share one stride; decide keep/skip once per sim.
        kept = Dict{String, Vector{Float64}}()
        ok_sim = true
        for k in rank_params
            draws = _derived_samples(r.chains, k)
            thinned = _sbc_thin(draws, min_ess, L)
            if thinned === nothing
                ok_sim = false; break
            end
            kept[k] = thinned
        end
        if !ok_sim
            n_insuff += 1
            verbose && @printf("  sim %3d  INSUFFICIENT (min_ess=%.0f < L=%d)\n",
                               s, min_ess, L)
            continue
        end
        n_ok += 1
        for k in rank_params
            tv = if k == "e_k1"
                se = θb[findfirst(==("sesinw_k1"), nms)]
                sc = θb[findfirst(==("secosw_k1"), nms)]
                sesinw_to_ew(se, sc)[1]
            else
                θb[findfirst(==(k), nms)]
            end
            isfinite(tv) && push!(ranks[k], sbc_rank(tv, kept[k]))
        end
        verbose && s % 16 == 0 &&
            @printf("  sim %3d/%3d  ok=%d fail=%d insuff=%d\n",
                    s, n_sims, n_ok, n_fail, n_insuff)
    end

    # Coverage guard: every verdict param must be scored on every ok sim.
    untested = String[]
    for k in rank_params
        length(ranks[k]) == n_ok || push!(untested, k)
    end

    tests = Dict{String, Any}()
    for k in rank_params
        isempty(ranks[k]) || (tests[k] = uniformity_test(ranks[k], L; n_bins))
    end

    return (sampler = sampler, n_sims = n_sims, L = L, n_bins = n_bins,
            n_ok = n_ok, n_fail = n_fail, n_insuff = n_insuff,
            ranks = ranks, tests = tests, rank_params = rank_params,
            untested = untested)
end

"""
    print_sbc(res; hist=true, α=0.01, fail_thresh=0.05, insuff_thresh=0.20)

Honest verdict: FAILED-TO-RUN if too many sims errored (θ-correlated
failures would otherwise certify only the easy slice); INSUFFICIENT if too
many lacked independent draws; UNTESTED if any verdict param was not
scored on every sim; otherwise a Šidák-corrected CALIBRATED/MISCALIBRATED
over the period-invariant SAMPLED params (e_k1 reported as a diagnostic).
"""
function print_sbc(res; hist::Bool = true, α::Float64 = 0.01,
                   fail_thresh::Float64 = 0.05, insuff_thresh::Float64 = 0.20)
    N = res.n_sims
    println("="^72)
    @printf("SBC: %s   L=%d n_bins=%d   ok=%d fail=%d insuff=%d / %d\n",
            res.sampler, res.L, res.n_bins, res.n_ok, res.n_fail, res.n_insuff, N)
    println("="^72)

    fam = intersect(SBC_VERDICT_PARAMS, collect(keys(res.tests)))
    α_fam = 1 - (1 - α)^(1 / max(length(fam), 1))   # Šidák per-test α
    worst_p = 1.0; worst_k = ""
    for k in vcat(SBC_VERDICT_PARAMS, SBC_DIAG_PARAMS)
        haskey(res.tests, k) || continue
        t = res.tests[k]
        tag = k in SBC_DIAG_PARAMS ? "[diag]" : ""
        if k in fam && t.pval < worst_p
            worst_p = t.pval; worst_k = k
        end
        @printf("%-12s %-6s χ²p=%.2e  mean=%.3f  var=%.4f(ref %.4f)  → %s\n",
                k, tag, t.pval, t.mean, t.var, t.var_ref, sbc_shape(t))
        hist && println(sbc_ascii_hist(t.counts))
    end

    println("-"^72)
    if res.n_fail / N > fail_thresh
        @printf("❌ FAILED-TO-RUN: %d/%d sims errored (> %.0f%%) — verdict withheld; θ-correlated failures would bias SBC to the easy slice.\n", res.n_fail, N, 100fail_thresh)
    elseif res.n_insuff / N > insuff_thresh
        @printf("⚠ INSUFFICIENT: %d/%d sims lacked ≥L independent draws (> %.0f%%) — raise the sampler budget; verdict withheld.\n", res.n_insuff, N, 100insuff_thresh)
    elseif !isempty(res.untested)
        println("❌ UNTESTED params (not scored on every sim): ", res.untested)
    elseif isempty(fam)
        println("❌ no verdict params scored.")
    elseif worst_p < α_fam
        @printf("❌ MISCALIBRATED: %s χ²p=%.2e < Šidák α=%.2e (family of %d) — see flags.\n",
                worst_k, worst_p, α_fam, length(fam))
    else
        @printf("✅ CALIBRATED: all %d verdict params pass (min χ²p=%.2e ≥ Šidák α=%.2e). n_ok=%d.\n", length(fam), worst_p, α_fam, res.n_ok)
    end
    println("="^72)
    return worst_p
end

# =====================================================================
# Entry point: gate, then optional sampler via ARGS.
# =====================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    sbc_self_test() || error("SBC self-test failed — aborting before any sampler run")
    if length(ARGS) >= 1
        sampler = ARGS[1]
        n_sims = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
        println("\nRunning SBC on `$sampler` (n_sims=$n_sims)…\n")
        print_sbc(run_sbc(sampler; n_sims = n_sims))
    end
end
