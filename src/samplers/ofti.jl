# Orbits For The Impatient (OFTI) — Blunt et al. 2017, AJ 153, 229.
#
# Rejection-sampling algorithm for visual-orbit fits with sparse
# observation coverage (orbital arc < ~20%). For each candidate orbit
# drawn from priors, "scale-and-rotate" the orbit so that it exactly
# matches the FIRST relative-astrometry epoch, then accept with
# probability ∝ exp(log-likelihood) for the remaining data.
#
# Why it matters for Nereus: standard MCMC (Pigeons PT, NUTS) gets
# trapped in single modes for sparse-data orbits where (a, e, i, ω, Ω)
# have multiple high-evidence solutions. OFTI samples broadly from the
# prior with the first epoch built in, naturally exploring all modes.
#
# Requirements:
#   - parametrization.mass = :a_driven (a is sampled directly so the
#     scale-and-rotate step can rewrite a)
#   - At least one relAST epoch
#
# References:
#   Blunt et al. 2017, AJ 153, 229 (algorithm)
#   Blunt et al. 2020, AJ 159, 89 (orbitize! v1)

using Random: MersenneTwister, AbstractRNG
import MCMCChains

"""
    ofti_sample(target::NereusTarget;
                n_attempts = 100_000,
                n_calibrate = 0,
                buffer = 0.0,
                planet_idx = 1,
                epoch_idx  = 1,
                seed       = 42,
                show_progress = false) -> MCMCChains.Chains

Run OFTI rejection sampling on a `:a_driven` model with relative-
astrometry data. Returns accepted samples as `MCMCChains.Chains`.

# Algorithm — 2-pass calibrated OFTI (orbitize!-equivalent)

OFTI requires a *fixed* `log_lik_max` reference for the rejection step
to give an unbiased posterior sample. A naive single-pass version that
updates `log_lik_max` on the fly biases early samples (accepted at a
too-permissive threshold) and tightens later, dropping the effective
acceptance rate by ~100×.

The fix is 2-pass:

1. **Pass 1 (calibration).** Run `n_calibrate` candidates: draw from
   priors, scale-and-rotate to anchor epoch, compute the joint
   log-likelihood. Track the maximum but do **not** accept anything.
   This produces a reference `log_lik_max_cal`.

2. **Pass 2 (production).** Run `n_attempts` candidates with the same
   draw + scale-and-rotate procedure, accepting each with probability
   `exp(log_lik − (log_lik_max_cal + buffer))`. The reference is now
   *fixed*, so the kept sample is an unbiased draw from the posterior
   modulo the (a, Ω) Jacobian of the scale-and-rotate transformation.

Per-candidate steps:
1. Draw all unfrozen parameters from their priors (uniform u → bounded
   value via `prior_transform`).
2. Build the model orbit at the anchor relAST epoch and compute the
   predicted (sep_model, PA_model) in mas / radians.
3. Scale `a_kN` by `(sep_obs / sep_model)` and rotate `Omega_kN` by
   `(PA_obs − PA_model)`. After this step the orbit *exactly matches*
   the anchor epoch by construction.
4. If the scaled `a` is outside its prior bounds, reject and continue.
5. Compute the full joint log-likelihood (RV + remaining relAST + HGCA).
6. Accept with probability `exp(log_lik − reference)` (or drop when
   in the calibration pass).

# Arguments

- `n_attempts`: production-pass candidates. Acceptance rate depends on
  data quality and prior breadth — typical few percent, so 100k–1M
  attempts give thousands of accepted samples.
- `n_calibrate`: calibration-pass candidates (default
  `max(n_attempts ÷ 10, 1000)`). Increase if `log_lik_max` is poorly
  resolved (signaled by a large pass-2 overshoot warning).
- `buffer`: added to the calibration `log_lik_max` to guard against
  pass-2 samples landing slightly above the calibration max. Default
  0.0; set to ~log(2)–log(20) if pass 2 reports significant overshoot.
- `planet_idx`: which companion's `a_kN` and `Omega_kN` slots to scale
  and rotate (1-based). Default 1.
- `epoch_idx`: which relAST epoch to anchor (1-based). Default 1.

# Returns

`MCMCChains.Chains` with accepted production-pass samples in
unfrozen-name order. Number of effective independent samples equals
number of accepted candidates (no autocorrelation — rejection
sampling).

If the prior bounds on `a_kN` are too tight, almost all candidates get
rejected at step 4. Widen the bounds (e.g. LogUniform(0.1, 10000)) to
give the scale-and-rotate step room to operate.
"""
function ofti_sample(target::NereusTarget;
                     n_attempts::Integer = 100_000,
                     n_calibrate::Integer = 0,
                     buffer::Real        = 0.0,
                     planet_idx::Integer = 1,
                     epoch_idx::Integer  = 1,
                     seed::Integer       = 42,
                     show_progress::Bool = false)
    rng    = MersenneTwister(seed)
    params = target.params
    layout = params.layout

    params.config.parametrization.mass === :a_driven || throw(ArgumentError(
        "OFTI requires parametrization.mass = :a_driven (need to scale `a` directly)"))

    # Validate data
    relast = target.data.relastrom
    relast === nothing && throw(ArgumentError("OFTI requires relAST data"))
    n_relast(relast) >= 1 || throw(ArgumentError("OFTI requires ≥1 relAST epoch"))
    1 <= epoch_idx <= n_relast(relast) || throw(ArgumentError(
        "epoch_idx out of range"))

    # Anchor observation in mas / radians
    t_anchor   = relast.t[epoch_idx]
    Δra_obs    = relast.ra_off[epoch_idx]
    Δdec_obs   = relast.dec_off[epoch_idx]
    sep_obs    = hypot(Δra_obs, Δdec_obs)
    pa_obs     = atan(Δra_obs, Δdec_obs)   # PA from north (atan(ΔRA, Δdec))

    # Slot lookups
    a_name = "a_k$(planet_idx)"
    Ω_name = "Omega_k$(planet_idx)"
    haskey(layout.name_to_idx, a_name) || throw(ArgumentError(
        "$a_name not in layout — is parametrization.mass = :a_driven?"))
    haskey(layout.name_to_idx, Ω_name) || throw(ArgumentError(
        "$Ω_name not in layout"))
    a_idx_glob = layout.name_to_idx[a_name]
    Ω_idx_glob = layout.name_to_idx[Ω_name]
    a_idx_unf  = findfirst(==(a_idx_glob), layout.unfrozen_idx)
    Ω_idx_unf  = findfirst(==(Ω_idx_glob), layout.unfrozen_idx)
    a_idx_unf === nothing && throw(ArgumentError(
        "$a_name is frozen — OFTI needs to sample / rescale it"))
    Ω_idx_unf === nothing && throw(ArgumentError("$Ω_name is frozen"))
    a_lo = layout.unfrozen_priors[a_idx_unf].lo
    a_hi = layout.unfrozen_priors[a_idx_unf].hi

    n_unf  = length(layout.unfrozen_idx)
    n_cal  = n_calibrate > 0 ? Int(n_calibrate) :
             max(div(Int(n_attempts), 10), 1000)

    # Per-thread buffers — OFTI is embarrassingly parallel (each draw is
    # independent), so we run the production loop with Threads.@threads
    # over the attempt index. Need per-thread `theta`, `x_bounded`, and
    # `rng` to avoid contention. The calibration pass is also threaded
    # but only tracks aggregate stats.
    nthreads = max(1, Threads.nthreads())
    thetas       = [Theta(params)              for _ in 1:nthreads]
    x_bounds     = [zeros(Float64, n_unf)      for _ in 1:nthreads]
    thread_rngs  = [MersenneTwister(seed + 1000*t) for t in 1:nthreads]

    # Per-thread closure. Same logic as before; takes the thread id so
    # it can pick its own buffers, deterministically seeded per thread.
    @inline function _draw_and_evaluate!(tid::Int)
        theta     = thetas[tid]
        x_bounded = x_bounds[tid]
        trng      = thread_rngs[tid]
        for j in 1:n_unf
            ps = layout.unfrozen_priors[j]
            x_bounded[j] = prior_transform(rand(trng), ps)
        end
        set_unfrozen!(theta, x_bounded)

        M_pri = astrom_M_pri(theta)
        plx   = astrom_plx(theta)
        orb, _ = _planet_orbit(theta, planet_idx, M_pri, plx, target.data.t_ref)
        Δra_mod, Δdec_mod = relastrom_offset(orb, t_anchor)
        sep_mod = hypot(Δra_mod, Δdec_mod)
        sep_mod > 0 || return (-Inf, false)
        pa_mod  = atan(Δra_mod, Δdec_mod)

        scale_factor = sep_obs / sep_mod
        a_new = x_bounded[a_idx_unf] * scale_factor
        (a_lo <= a_new <= a_hi) || return (-Inf, false)
        x_bounded[a_idx_unf] = a_new
        x_bounded[Ω_idx_unf] = mod(x_bounded[Ω_idx_unf] + (pa_obs - pa_mod), 2π)
        set_unfrozen!(theta, x_bounded)

        ll = rv_log_likelihood(theta, target.data)
        return (ll, true)
    end

    # ---------- Pass 1: calibration (threaded) ----------------------
    # Each thread tracks its own running max + counts to avoid atomics
    # in the hot loop; we reduce after the parallel block.
    per_t_logL_max  = fill(-Inf, nthreads)
    per_t_finite    = zeros(Int, nthreads)
    per_t_in_bounds = zeros(Int, nthreads)
    pb_cal = ProgressBar("OFTI calibration";
                          total = n_cal, enabled = show_progress)
    # :static — Threads.@threads default :dynamic lets tasks migrate
    # mid-iteration when GC yields fire, which means `tid =
    # Threads.threadid()` becomes stale and per-thread buffers race.
    # See feedback_threading_threadid.md.
    Threads.@threads :static for attempt in 1:n_cal
        tid = Threads.threadid()
        ll, in_bounds = _draw_and_evaluate!(tid)
        if in_bounds
            per_t_in_bounds[tid] += 1
            if isfinite(ll)
                per_t_finite[tid] += 1
                ll > per_t_logL_max[tid] && (per_t_logL_max[tid] = ll)
            end
        end
    end
    log_lik_max_cal = maximum(per_t_logL_max)
    n_finite_cal    = sum(per_t_finite)
    n_in_bounds_cal = sum(per_t_in_bounds)
    update!(pb_cal; n_done = n_cal,
             fields = (:in_bounds => n_in_bounds_cal,
                       :finite => n_finite_cal,
                       :logL_max => log_lik_max_cal))
    finish!(pb_cal)

    n_finite_cal > 0 || throw(ErrorException(
        "OFTI calibration pass: 0 finite log-likelihoods from $n_cal attempts. " *
        "Likely cause: priors don't overlap with the orbit's natural range. " *
        "Widen bounds (especially a_kN) or check the data."))

    log_lik_ref = log_lik_max_cal + Float64(buffer)

    # ---------- Pass 2: production with fixed reference (threaded) ---
    # Accept/reject is per-attempt independent; collect into per-thread
    # accepted vectors, then concatenate. Random acceptance uses the
    # thread's local rng (seeded above), so seeding is reproducible
    # within a (nthreads, seed) configuration.
    per_t_accepted   = [Vector{Vector{Float64}}() for _ in 1:nthreads]
    per_t_loglikes   = [Float64[]                  for _ in 1:nthreads]
    per_t_finite_p   = zeros(Int, nthreads)
    per_t_in_bnds_p  = zeros(Int, nthreads)
    per_t_overshoot  = zeros(Int, nthreads)

    pb_prod = ProgressBar("OFTI production";
                           total = n_attempts, enabled = show_progress)
    Threads.@threads :static for attempt in 1:n_attempts
        tid = Threads.threadid()
        ll, in_bounds = _draw_and_evaluate!(tid)
        if in_bounds
            per_t_in_bnds_p[tid] += 1
            if isfinite(ll)
                per_t_finite_p[tid] += 1
                if ll > log_lik_ref
                    per_t_overshoot[tid] += 1
                    push!(per_t_accepted[tid], copy(x_bounds[tid]))
                    push!(per_t_loglikes[tid], ll)
                elseif rand(thread_rngs[tid]) < exp(ll - log_lik_ref)
                    push!(per_t_accepted[tid], copy(x_bounds[tid]))
                    push!(per_t_loglikes[tid], ll)
                end
            end
        end
    end
    accepted        = reduce(vcat, per_t_accepted)
    log_likes       = reduce(vcat, per_t_loglikes)
    n_finite_prod   = sum(per_t_finite_p)
    n_in_bounds_prod= sum(per_t_in_bnds_p)
    n_overshoot     = sum(per_t_overshoot)
    update!(pb_prod; n_done = n_attempts,
             fields = (:accepted => length(accepted),
                       :overshoot => n_overshoot,
                       :acc_rate => length(accepted) / max(n_attempts, 1)))
    finish!(pb_prod)

    n_acc = length(accepted)
    n_acc > 0 || throw(ErrorException(
        "OFTI: 0 samples accepted from $n_attempts production attempts " *
        "(after $n_cal calibration). Try widening priors, increasing " *
        "n_attempts, or raising buffer (currently $buffer)."))

    if n_finite_prod > 0 && n_overshoot / n_finite_prod > 0.01
        @warn "OFTI: pass-2 log-likelihood exceeded calibration max in $n_overshoot / $n_finite_prod (>1%) — calibration pass undersampled. Increase n_calibrate or set buffer." log_lik_max_cal
    end

    mat = Array{Float64,3}(undef, n_acc, n_unf, 1)
    for i in 1:n_acc
        @inbounds for j in 1:n_unf
            mat[i, j, 1] = accepted[i][j]
        end
    end
    chains = MCMCChains.Chains(mat, layout.unfrozen_names)

    @info "OFTI complete" n_calibrate=n_cal n_attempts n_in_bounds=n_in_bounds_prod n_finite=n_finite_prod n_accepted=n_acc n_overshoot log_lik_max_cal
    return chains
end
