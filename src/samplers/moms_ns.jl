# Trans-dimensional Nested Sampling via Mixtures of Mutually Singular
# distributions — MoMS-NS.
#
# Combines two ingredients:
#   1. The MoMS construction (van den Bergh, Clyde, Raftery, Marsman 2026,
#      arXiv:2604.27791) that maps trans-dim variable selection onto a
#      fixed-dim space (β, γ) with a spike-and-slab prior — already
#      implemented in Nereus as `MoMSBirth` and the matching propose_*
#      methods in transdim/proposals.jl.
#
#   2. Skilling's Nested Sampling (Skilling 2006, AIP 735, 395) on the
#      fixed-dim (β, γ) space. The constrained-likelihood region L > L_*
#      is explored by a small constrained MCMC kernel that reuses MoMS
#      between-model proposals (planet birth/death) plus a single-coordinate
#      Gaussian random walk within the active subset.
#
# The output is a single run that delivers, jointly:
#   * The marginal posterior over the indicator vector γ — i.e. P(N_p = k)
#     and the inclusion probabilities of individual planets.
#   * The posterior over continuous parameters β at every active configuration.
#   * A log-evidence estimate log Z that supports proper Bayes factors
#     between dimensionalities — the headline output the existing trans-dim
#     samplers (RJMCMC / PT / MoMS) do not produce.
#
# This is, to the best of our knowledge, the first trans-dim Nested
# Sampler shipped in pure Julia. The implementation reuses existing
# Nereus infrastructure (TransDimState, MoMSBirth, log_prior,
# _eval_ll, _init_systemics_from_prior!) and adds only the NS skeleton.

using Random
using Statistics: mean, std
using Printf
using MCMCChains

export sample_moms_ns, model_probabilities, bayes_factors

# Inline log-add-exp to avoid pulling in LogExpFunctions
@inline function _logaddexp(a::Real, b::Real)
    a == -Inf && return float(b)
    b == -Inf && return float(a)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

"""
    MoMSNSPoint

Frozen state of a live or dead point in MoMS-NS: parameter values, the
trans-dim indicator vector, and the cached log-prior / log-likelihood.
"""
struct MoMSNSPoint
    values::Vector{Float64}
    planet_active::Vector{Bool}
    noise_active::Vector{Bool}
    log_prior::Float64
    log_L::Float64
end

"""
    sample_moms_ns(target, data; td, kwargs...)
        -> (chains::MCMCChains.Chains, log_evidence, strategy::MoMSBirth)

Trans-dimensional nested sampler operating on the joint (β, γ)
spike-and-slab parameterisation. Returns posterior samples (importance-
resampled to equal weights), the log-evidence estimate, and the
adapted MoMS strategy carrying off-values + scales.

# Required
- `target::NereusTarget` — bounded (`unconstrained=false`) target.
- `data::Data` — observation data.
- `td::TransDimConfig` — trans-dim configuration. Birth strategies in
  `td` are ignored; MoMS-NS always uses MoMS proposals.

# Keywords
- `n_live::Int = 200` — number of live points.
- `dlogz::Float64 = 0.5` — termination criterion on remaining log-Z.
- `n_mcmc::Int = 25` — constrained-MCMC steps when replacing a dead point.
- `init_scale::Float64 = 1.0` — initial MoMS proposal scale (× IQR-stddev).
- `inclusion_prior::Float64 = 0.5` — Bernoulli prior on each γ_k at init,
  also used as the spike-and-slab γ-prior in the corrected log-prior so
  log-Z is properly normalised across N_p configurations.
- `seed::Int = 1`.
- `max_iter::Int = 200_000` — hard upper bound on NS iterations.
- `show_progress::Bool = true` — print log-Z trajectory every `progress_every`
  iterations.
- `progress_every::Int = 200`.
"""
function sample_moms_ns(
    target::NereusTarget,
    data::Data;
    td::TransDimConfig,
    n_live::Int            = 200,
    dlogz::Real            = 0.5,
    n_mcmc::Int            = 25,
    init_scale::Real       = 1.0,
    inclusion_prior::Real  = 0.5,
    seed::Int              = 1,
    max_iter::Int          = 200_000,
    show_progress::Bool    = true,
    progress_every::Int    = 200,
    informed_birth_fraction::Real = 0.0,
    batch_size::Int        = 1,
    warm_start_points::Union{Vector{MoMSNSPoint}, Nothing} = nothing,
)
    # JSON may deliver these floats as Int; normalize.
    dlogz                   = Float64(dlogz)
    init_scale              = Float64(init_scale)
    inclusion_prior         = Float64(inclusion_prior)
    informed_birth_fraction = Float64(informed_birth_fraction)
    0.0 < inclusion_prior < 1.0 || throw(ArgumentError(
        "inclusion_prior must be in (0, 1)"))
    n_live >= 2 || throw(ArgumentError("n_live must be ≥ 2"))
    0.0 <= informed_birth_fraction <= 1.0 || throw(ArgumentError(
        "informed_birth_fraction must be in [0, 1]"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be ≥ 1"))
    batch_size <= n_live ÷ 4 || throw(ArgumentError(
        "batch_size must be ≤ n_live/4 to keep batched-NS bias small"))
    # Noise toggling supported via the existing `propose_noise_birth/death`
    # methods (RJMCMC-style prior-draw proposals). The M-H acceptance in
    # the constrained-MCMC kernel uses prior + proposal ratios with the
    # likelihood replaced by the constraint L > L_thr.

    rng    = MersenneTwister(seed)
    params = target.params

    strategy = MoMSBirth(params; init_scale = init_scale, rng = rng)

    # ---- 1. Initialize n_live points from the spike-and-slab prior --
    # If `warm_start_points` is supplied, use those (typically built from
    # Pathfinder draws via `pathfinder_warmstart_moms_ns`); they bypass
    # the random γ + random params init that traps the chain in low-Np
    # local modes on multi-planet systems.
    live = if warm_start_points !== nothing
        length(warm_start_points) == n_live || throw(ArgumentError(
            "warm_start_points length ($(length(warm_start_points))) " *
            "must equal n_live ($n_live)"))
        copy(warm_start_points)
    else
        out = Vector{MoMSNSPoint}(undef, n_live)
        for i in 1:n_live
            out[i] = _moms_ns_init_point(target, data, strategy, td,
                                          inclusion_prior, rng)
        end
        out
    end

    # ---- 2. NS skeleton ---------------------------------------------
    # Skilling's recursion: at iteration i, the i-th dead point has
    # weight w_i = ΔX_i ≈ exp(-(i-1)/N) - exp(-i/N), and contributes
    # log(L_i) + log(w_i) to log Z (via logaddexp).
    log_Z          = -Inf
    log_X_prev     = 0.0  # log X_0 = 0 (full prior volume)
    inv_N          = 1.0 / n_live

    dead_pts    = MoMSNSPoint[]
    log_w_dead  = Float64[]   # log of width × likelihood (i.e. unnormalised weight)
    log_L_dead  = Float64[]

    iter = 0
    last_report = 0
    pb = ProgressBar("MoMS-NS";
                       total = max_iter, enabled = show_progress)
    while iter < max_iter

        # --- Pick the batch_size worst live points ----------------------
        # batch_size == 1 reduces exactly to Skilling's serial recipe.
        # batch_size > 1 follows dynesty's static-batched approach: kill
        # the b worst points simultaneously, share ΔX_batch evenly across
        # them, and evolve b replacements concurrently against the strict
        # L > max(killed L) constraint.
        if batch_size == 1
            dead_indices = [argmin_log_L(live)]
        else
            sorted_perm = sortperm([p.log_L for p in live])
            dead_indices = sorted_perm[1:batch_size]
        end

        # The most-restrictive L_thr is the max of the killed log_L —
        # replacements must clear this for ALL b parallel evolutions to
        # stay above the truncation boundary.
        L_thr = maximum(live[idx].log_L for idx in dead_indices)

        # ΔX for the batched contraction. log_dX_batch handles both
        # serial (b=1) and batched cases via the same formula.
        log_dX_batch  = log_X_prev + log1p(-exp(-batch_size * inv_N))
        log_w_per     = log_dX_batch - log(batch_size)
        log_X_curr    = log_X_prev - batch_size * inv_N

        # Record each killed point with its own L (for posterior weights)
        # and the shared per-dead weight log_w_per.
        for idx in dead_indices
            push!(dead_pts,    live[idx])
            push!(log_w_dead,  log_w_per)
            push!(log_L_dead,  live[idx].log_L)
            log_Z = _logaddexp(log_Z, log_w_per + live[idx].log_L)
        end

        # Termination check (same as serial NS).
        L_max_live = -Inf
        for p in live
            p.log_L > L_max_live && (L_max_live = p.log_L)
        end
        log_Z_remain = log_X_curr + L_max_live
        Δ = log_Z_remain - log_Z
        if Δ < log(dlogz)
            log_X_prev = log_X_curr
            iter += batch_size
            break
        end

        # --- Replace killed points (parallel via @spawn) -----------------
        # Donors are random survivors (excluding all killed indices).
        survivor_pool = setdiff(1:n_live, dead_indices)
        if batch_size == 1
            # Serial path: avoid @spawn overhead for the common b=1 case.
            donor = survivor_pool[rand(rng, 1:length(survivor_pool))]
            live[dead_indices[1]] = _moms_ns_evolve(live[donor], target,
                                                      data, strategy, td,
                                                      L_thr, n_mcmc,
                                                      inclusion_prior, rng;
                                                      informed_birth_fraction =
                                                          informed_birth_fraction)
        else
            tasks = Vector{Task}(undef, batch_size)
            for j in 1:batch_size
                donor = survivor_pool[rand(rng, 1:length(survivor_pool))]
                # Each task gets its own seeded RNG so the evolutions are
                # reproducible and independent; the master `rng` is only
                # used to draw the seeds + donor indices.
                child_rng = MersenneTwister(rand(rng, UInt64))
                start_pt = live[donor]
                tasks[j] = Threads.@spawn _moms_ns_evolve(start_pt, target,
                                                            data, strategy, td,
                                                            L_thr, n_mcmc,
                                                            inclusion_prior, child_rng;
                                                            informed_birth_fraction =
                                                                informed_birth_fraction)
            end
            for (j, idx) in enumerate(dead_indices)
                live[idx] = fetch(tasks[j])
            end
        end

        log_X_prev = log_X_curr
        iter += batch_size

        # Progress bar — show iter (counting batched as `batch_size`
        # so totals are comparable to serial runs).
        n_active_med = median([count(p.planet_active) for p in live])
        update!(pb; n_done = iter,
                 fields = (:logZ => log_Z,
                           :dlogZ => log_Z_remain - log_Z,
                           :Lthr => L_thr,
                           Symbol("⟨Np⟩") => n_active_med))
    end
    finish!(pb)

    # ---- 3. Add remaining live points (each gets an equal share of
    # the residual prior volume X_iter / n_live).
    log_dX_residual = log_X_prev - log(n_live)
    for p in live
        log_Z = _logaddexp(log_Z, log_dX_residual + p.log_L)
        push!(dead_pts,   p)
        push!(log_w_dead, log_dX_residual)
        push!(log_L_dead, p.log_L)
    end

    if show_progress
        @printf("MoMS-NS finished: %d iterations, %d total points (%d dead + %d live).\n",
                iter, length(dead_pts), iter, n_live)
        @printf("log Z = %+.4f\n", log_Z)
    end

    # ---- 4. Importance-resample to get equally-weighted posterior --
    log_post_w = log_L_dead .+ log_w_dead .- log_Z
    chain = _moms_ns_resample_chain(dead_pts, log_post_w, params, td, rng)

    return chain, log_Z, strategy
end

# =====================================================================
# Live-point initialisation
# =====================================================================

function _moms_ns_init_point(target::NereusTarget, data::Data,
                              strategy::MoMSBirth, td::TransDimConfig,
                              inclusion_prior::Float64,
                              rng::AbstractRNG;
                              max_per_planet_retries::Int = 8)
    params = target.params
    layout = params.layout

    # Mask sized to the FULL noise-model list (consumers index by
    # config.noise_models position; see rjmcmc/transdim_ptemcee fix) with
    # non-toggleables forced active.
    td_state = TransDimState(
        max_planets = td.max_kplanet,
        n_noise = length(params.config.noise_models),
    )
    for (nm_idx, nm) in enumerate(params.config.noise_models)
        if !(td.noise && nm in td.toggleable)
            td_state.noise_active[nm_idx] = true
        end
    end
    theta = Theta{Float64}(params; td = td_state)

    _init_systemics_from_prior!(theta, rng)

    # Override γ values with per-instrument median(rv_i). The default
    # init_systemics draws γ from broad priors — for multi-instrument
    # data those random γ are far from the right zero-points and dominate
    # the residual chi², which forces the constrained log_L threshold
    # to absorb large γ swings before the chain can ever consider planet
    # signals. Seeding γ at the per-instrument median puts each live
    # point on a sane starting calibration; the MCMC still walks γ
    # afterwards if the data prefers something different.
    _seed_gammas_from_data!(theta, data)

    # All planets start inactive (β at off-value). We then attempt to
    # activate each planet independently with probability inclusion_prior.
    # Crucially: if the activation+random-prior-draw of planet k makes
    # log_L non-finite (AMD instability, period collision, NaN-prop on
    # an out-of-support draw, etc.), we DEACTIVATE just that planet and
    # move on. Previously we nuked the entire state to N_p=0 on any
    # failure, which made ~all live points start at N_p=0 for broad-prior
    # multi-planet systems and locked the chain there because the
    # photometry-dominated log-L at N_p=0 is high enough that the
    # constraint L > L_thr never opens up to N_p>0 trajectories.
    for k in 1:td.max_kplanet
        offs   = strategy.off_values[k]
        uf_idx = strategy.slot_indices[k]
        # Always start at off-value for slot k; activation may overwrite.
        for i in eachindex(uf_idx)
            slot = layout.unfrozen_idx[uf_idx[i]]
            theta.values[slot] = offs[i]
        end

        # td.planets=false ⇒ the planet set is FIXED: every slot must be
        # active (the Bernoulli draw previously ran regardless, starting
        # live points planet-less in a fixed-planet noise-only run).
        td.planets && (rand(rng) < inclusion_prior || continue)

        # Try up to `max_per_planet_retries` independent prior draws for
        # this planet's β; accept the first one that gives finite log_L
        # in combination with whatever planets are already active.
        slots_k = planet_slot_indices(layout.planet_blocks[k])
        # Snapshot current values for this planet's slots so we can roll
        # back on failure.
        snap_vals = [theta.values[s] for s in slots_k]
        activate_planet!(theta.td, k)

        success = false
        for attempt in 1:max_per_planet_retries
            for slot in slots_k
                uf_pos = findfirst(==(slot), layout.unfrozen_idx)
                uf_pos === nothing && continue
                prior = layout.unfrozen_priors[uf_pos]
                v = rand(rng, prior.dist)
                lo, hi = bounds(prior)
                theta.values[slot] = clamp(v, lo, hi)
            end
            ll = _eval_ll(theta, data, nothing)
            if isfinite(ll)
                success = true
                break
            end
        end

        if !success
            # Roll back: reset slots to the snapshotted off-values. Only
            # deactivate when the planet set is trans-dim — under
            # td.planets=false the slot must STAY active (fixed planet).
            td.planets && deactivate_planet!(theta.td, k)
            for (i, s) in enumerate(slots_k)
                theta.values[s] = snap_vals[i]
            end
        end
    end

    # Final evaluation. If even N_p=0 gives -Inf (e.g. data is so
    # degenerate the LL breaks at the off-state too), keep the state as
    # an unweighted live point — the NS pruning loop handles -Inf log_L
    # by killing it on the first iteration.
    log_pi = spike_slab_log_prior(theta, strategy, inclusion_prior)
    log_L  = _eval_ll(theta, data, nothing)

    return MoMSNSPoint(
        copy(theta.values),
        copy(theta.td.planet_active),
        copy(theta.td.noise_active),
        Float64(log_pi),
        Float64(log_L),
    )
end

# =====================================================================
# Constrained-MCMC evolution (replace dead point)
# =====================================================================

"""
    _moms_ns_evolve(start, target, data, strategy, td, L_thr, n_mcmc,
                     inclusion_prior, rng)
        -> MoMSNSPoint

Evolve a live point under the constraint log-L > L_thr using a small
MCMC. Each step is either:
  * a between-model MoMS proposal (planet birth/death), with M-H
    acceptance using prior + proposal ratios (no likelihood in the M-H
    ratio — the constraint replaces it), or
  * a within-model single-coordinate Gaussian random walk on the active
    subset, accepted iff prior ratio passes and L > L_thr.
"""
function _moms_ns_evolve(start::MoMSNSPoint, target::NereusTarget,
                          data::Data, strategy::MoMSBirth,
                          td::TransDimConfig, L_thr::Float64,
                          n_mcmc::Int, inclusion_prior::Float64,
                          rng::AbstractRNG;
                          informed_birth_fraction::Float64 = 0.0)
    params = target.params
    layout = params.layout

    # Scratch td sized to the FULL noise-model list — must match the
    # length of the masks carried by MoMSNSPoint (see _moms_ns_init_point).
    td_state = TransDimState(
        max_planets = td.max_kplanet,
        n_noise = length(params.config.noise_models),
    )
    theta = Theta{Float64}(params; td = td_state)
    theta.values .= start.values
    theta.td.planet_active .= start.planet_active
    theta.td.noise_active  .= start.noise_active
    theta.td.n_planets_active = count(start.planet_active)

    log_pi = start.log_prior
    log_L  = start.log_L

    has_toggleable_noise = td.noise && !isempty(td.toggleable)

    for step in 1:n_mcmc
        if rand(rng) < td.transdim_fraction
            # Between-model: with toggleable noise, split between planet
            # flips and noise flips 50/50; otherwise always planet flip.
            do_noise = has_toggleable_noise && (td.planets ?
                                                  rand(rng) < 0.5 : true)
            new_theta, log_q = if do_noise
                # Noise flip: 50/50 birth/death. The existing prior-draw
                # noise proposals' M-H ratio cancels the prior contribution
                # for the new noise params, so it composes correctly with
                # the spike-and-slab planet prior.
                # Within-group swap when the menu declares exclusion groups.
                # NS has no temperature ladder, so like rjmcmc it cannot cross
                # the "none" valley between two members by birth/death; the
                # constrained kernel would have to accept an intermediate state
                # that usually violates L > L_thr. The swap goes straight
                # across. `data` is forwarded so the informed proposals (AD
                # OLS, GP period/amplitude) actually fire.
                if !isempty(td.noise_exclusion_groups) && rand(rng) < 0.5
                    grp = td.noise_exclusion_groups[
                              rand(rng, 1:length(td.noise_exclusion_groups))]
                    propose_noise_swap(theta, rng, grp, td.toggleable; data=data)
                elseif rand(rng) < 0.5
                    propose_noise_birth(theta, rng, td.toggleable; data=data,
                                         exclusion_groups=td.noise_exclusion_groups)
                else
                    propose_noise_death(theta, rng, td.toggleable; data=data)
                end
            else
                # Planet birth/death. With probability
                # `informed_birth_fraction` use a periodogram-driven
                # InformedBirth proposal (so the chain can land planets
                # at real period peaks under the L > L_thr constraint);
                # otherwise the native MoMSBirth Gaussian RW. Same
                # forward+reverse strategy each attempt → detailed
                # balance under the picked strategy's q.
                n_active = theta.td.n_planets_active
                max_k    = td.max_kplanet
                p_birth, _ = _birth_death_probs(n_active, max_k)
                use_informed = informed_birth_fraction > 0 &&
                                rand(rng) < informed_birth_fraction
                # Use JointInformedBirth when photometry is available (it
                # internally falls back to InformedBirth if not). The joint
                # variant adds BLS-derived peaks with active-planet harmonic
                # dedup — necessary for multi-planet trans-dim search on
                # data where transits dominate the likelihood.
                chosen_strategy = use_informed ?
                    (isempty(data.t_phot) ? InformedBirth() : JointInformedBirth()) :
                    strategy
                if rand(rng) < p_birth
                    propose_planet_birth(theta, rng, chosen_strategy; data = data)
                elseif chosen_strategy isa MoMSBirth
                    propose_planet_death(theta, rng, chosen_strategy)
                else
                    # InformedBirth's matching reverse is the strategy-
                    # agnostic uniform-pick death; its Hastings ratio
                    # was computed against this in `propose_planet_birth`.
                    propose_planet_death(theta, rng)
                end
            end
            isfinite(log_q) || continue

            new_log_pi = spike_slab_log_prior(new_theta, strategy, inclusion_prior)
            isfinite(new_log_pi) || continue

            new_log_L = _eval_ll(new_theta, data, nothing)
            (isfinite(new_log_L) && new_log_L > L_thr) || continue

            # NS-constrained M-H: prior + proposal ratios, NO likelihood.
            # The MoMS paper (van den Bergh+ 2026, Eq. 6) proves detailed
            # balance for the Gaussian-RW kernel w.r.t. the POSTERIOR.
            # For Skilling NS we want detailed balance w.r.t. the
            # CONSTRAINED PRIOR π(β,γ)·1[L>L*]. The argument carries
            # over directly: the kernel's reversibility is a property of
            # its (β,γ)-mixed base measure and proposal density, not of
            # the target. Replacing the likelihood with the constraint
            # indicator preserves balance, modulo the hard rejection
            # `new_log_L > L_thr` enforced above.
            log_α = (new_log_pi - log_pi) + log_q
            if log(rand(rng)) < log_α
                theta.values .= new_theta.values
                theta.td.planet_active .= new_theta.td.planet_active
                theta.td.n_planets_active = new_theta.td.n_planets_active
                if has_toggleable_noise
                    theta.td.noise_active .= new_theta.td.noise_active
                end
                log_pi = new_log_pi
                log_L  = new_log_L
            end
        else
            # Within-model: single-coord Gaussian random walk. The within
            # step uses Nereus log_prior internally for its M-H ratio
            # (correct, since the spike-and-slab correction is constant
            # at fixed γ); we re-evaluate via spike_slab_log_prior on
            # accept so log_pi stays consistent for the next γ flip.
            res = _moms_ns_within_step!(theta, data, L_thr, log_pi, log_L,
                                          strategy, inclusion_prior, rng)
            if res !== nothing
                log_pi, log_L = res
            end
        end
    end

    return MoMSNSPoint(
        copy(theta.values),
        copy(theta.td.planet_active),
        copy(theta.td.noise_active),
        Float64(log_pi),
        Float64(log_L),
    )
end

"""Pick a random active unfrozen coordinate, propose a Gaussian step,
accept iff prior allows and L > L_thr. Returns the new (log_pi, log_L)
on accept (with `log_pi` in spike-and-slab units), `nothing` on reject."""
function _moms_ns_within_step!(theta::Theta, data::Data, L_thr::Float64,
                                  log_pi::Float64, log_L::Float64,
                                  strategy::MoMSBirth, inclusion_prior::Float64,
                                  rng::AbstractRNG)
    params = theta.params
    layout = params.layout
    n_unfrozen = length(layout.unfrozen_idx)
    isempty(layout.unfrozen_idx) && return nothing

    # Build active mask: skip unfrozen slots that belong to inactive planets
    skip = falses(n_unfrozen)
    for k in 1:length(theta.td.planet_active)
        theta.td.planet_active[k] && continue
        for slot in planet_slot_indices(layout.planet_blocks[k])
            uf_pos = findfirst(==(slot), layout.unfrozen_idx)
            uf_pos === nothing && continue
            skip[uf_pos] = true
        end
    end
    active_pos = findall(.!skip)
    isempty(active_pos) && return nothing

    j     = active_pos[rand(rng, 1:length(active_pos))]
    slot  = layout.unfrozen_idx[j]
    prior = layout.unfrozen_priors[j]
    lo, hi = bounds(prior)

    # Step scale: ~2% of finite prior width, fallback to current-value-based.
    σ = if isfinite(lo) && isfinite(hi)
        (hi - lo) / 50
    else
        0.05 * abs(theta.values[slot]) + 1e-3
    end

    old_val = theta.values[slot]
    new_val = old_val + σ * randn(rng)
    if (isfinite(lo) && new_val < lo) || (isfinite(hi) && new_val > hi)
        return nothing
    end

    theta.values[slot] = new_val
    new_log_pi = spike_slab_log_prior(theta, strategy, inclusion_prior)
    if !isfinite(new_log_pi)
        theta.values[slot] = old_val
        return nothing
    end
    new_log_L = _eval_ll(theta, data, nothing)
    if !isfinite(new_log_L) || new_log_L <= L_thr
        theta.values[slot] = old_val
        return nothing
    end

    # Symmetric Gaussian proposal cancels — M-H reduces to prior ratio
    log_α = new_log_pi - log_pi
    if log(rand(rng)) < log_α
        return (new_log_pi, new_log_L)
    end
    theta.values[slot] = old_val
    return nothing
end

# =====================================================================
# Posterior reconstruction (importance resampling)
# =====================================================================

"""Importance-resample dead+live points to equal weights and assemble
the same chain layout as `sample_rjmcmc`/`sample_moms` for downstream
post-processing."""
function _moms_ns_resample_chain(points::Vector{MoMSNSPoint},
                                   log_w::Vector{Float64},
                                   params::Params,
                                   td::TransDimConfig,
                                   rng::AbstractRNG)
    layout = params.layout
    n_pts  = length(points)
    n_pts == length(log_w) || error("points and weights length mismatch")

    # Resample size: n_pts (one-to-one). Effective sample size is the
    # standard log-weight variance; if it's tiny the chain will repeat
    # the same handful of points.
    max_lw = maximum(log_w)
    w_norm = exp.(log_w .- max_lw)
    total  = sum(w_norm)
    w_norm ./= total

    n_resample = n_pts
    pick = sample_categorical(w_norm, n_resample, rng)

    # Output columns: unfrozen params + n_planets + noise_active_*
    n_unfrozen   = length(layout.unfrozen_idx)
    noise_names  = Symbol[]
    n_noise_cols = 0
    if td.noise && !isempty(td.toggleable)
        for nm in td.toggleable
            nm_idx = findfirst(==(nm), params.config.noise_models)
            nm_idx === nothing && continue
            push!(noise_names, Symbol("noise_active_$(nm_idx)"))
            n_noise_cols += 1
        end
    end
    n_pl_cols = td.max_kplanet
    planet_names = Symbol[Symbol("planet_active_$(k)") for k in 1:n_pl_cols]
    param_names = vcat(Symbol.(layout.unfrozen_names),
                        [:n_planets], planet_names, noise_names)
    n_cols  = n_unfrozen + 1 + n_pl_cols + n_noise_cols
    samples = Matrix{Float64}(undef, n_resample, n_cols)

    for (out_i, src_i) in enumerate(pick)
        p = points[src_i]
        for (j, uf_idx) in enumerate(layout.unfrozen_idx)
            samples[out_i, j] = p.values[uf_idx]
        end
        col = n_unfrozen + 1
        samples[out_i, col] = Float64(count(p.planet_active))
        for k in 1:n_pl_cols
            samples[out_i, col + k] = Float64(p.planet_active[k])
        end
        if n_noise_cols > 0
            ni = 0
            for nm in td.toggleable
                nm_idx = findfirst(==(nm), params.config.noise_models)
                nm_idx === nothing && continue
                ni += 1
                samples[out_i, col + n_pl_cols + ni] = Float64(p.noise_active[nm_idx])
            end
        end
    end
    return MCMCChains.Chains(samples, param_names)
end

"""Linear-time categorical sampling with replacement. Good enough for
moderate n_pts — the time-critical loops are elsewhere."""
function sample_categorical(probs::AbstractVector{<:Real}, n::Int,
                              rng::AbstractRNG)
    cum = cumsum(probs)
    cum ./= cum[end]
    out = Vector{Int}(undef, n)
    for i in 1:n
        u = rand(rng)
        # binary search
        lo, hi = 1, length(cum)
        while lo < hi
            m = (lo + hi) ÷ 2
            if u <= cum[m]
                hi = m
            else
                lo = m + 1
            end
        end
        out[i] = lo
    end
    return out
end

@inline function argmin_log_L(live::Vector{MoMSNSPoint})
    minL = live[1].log_L
    minI = 1
    @inbounds for i in 2:length(live)
        if live[i].log_L < minL
            minL = live[i].log_L
            minI = i
        end
    end
    return minI
end

# =====================================================================
# Model-probability and Bayes-factor extractors for trans-dim chains
# =====================================================================

"""
    model_probabilities(chains; max_kplanet=nothing) -> Dict{Int, Float64}

Compute the marginal model posterior P(N_p = k | data) from any
trans-dim Nereus chain (`sample_rjmcmc` / `sample_pt` / `sample_moms`
/ `sample_moms_ns`). All four samplers store an `:n_planets` column;
this helper just tabulates it. For MoMS-NS chains the result is
proportional to evidence × prior — i.e. you can read off Bayes factors
directly via `bayes_factors`.

`max_kplanet` defaults to the largest observed N_p in the chain.
"""
function model_probabilities(chains;
                              max_kplanet::Union{Nothing, Int} = nothing)
    np = vec(Array(chains[:n_planets]))
    n  = length(np)
    kmax = max_kplanet === nothing ? Int(maximum(np)) : max_kplanet
    out = Dict{Int, Float64}()
    for k in 0:kmax
        out[k] = count(np .== Float64(k)) / n
    end
    return out
end

"""
    bayes_factors(chains; reference=nothing, model_prior=:uniform)
        -> Dict{Tuple{Int,Int}, Float64}

Bayes factors BF(k₁ vs k₂) = P(N_p=k₁|y)/P(N_p=k₂|y) × p(N_p=k₂)/p(N_p=k₁)
between every observed N_p in the chain. Returns a dict keyed by
`(k₁, k₂)` tuples.

If `reference` is given (e.g. `reference=0`), only return BFs against
that reference value: `Dict(k => BF(k vs reference))`.

`model_prior=:uniform` (default) assumes p(N_p=k) ∝ 1 across observed k.
For other priors, pass `model_prior=k -> p(k)`.
"""
function bayes_factors(chains;
                        reference::Union{Nothing, Int} = nothing,
                        model_prior = :uniform)
    probs = model_probabilities(chains)
    p_model(k) = model_prior === :uniform ? 1.0 : Float64(model_prior(k))
    ks = sort(collect(keys(probs)))
    if reference !== nothing
        reference in ks || throw(ArgumentError(
            "reference k=$reference not present in chain"))
        ref_p = probs[reference]
        ref_p > 0 || throw(ArgumentError(
            "P(N_p = $reference | data) = 0 — cannot use as Bayes-factor reference"))
        out = Dict{Int, Float64}()
        for k in ks
            k == reference && continue
            probs[k] > 0 || (out[k] = 0.0; continue)
            out[k] = (probs[k] / probs[reference]) *
                     (p_model(reference) / p_model(k))
        end
        return out
    end
    out = Dict{Tuple{Int, Int}, Float64}()
    for k1 in ks, k2 in ks
        k1 == k2 && continue
        probs[k2] > 0 || continue
        out[(k1, k2)] = (probs[k1] / probs[k2]) * (p_model(k2) / p_model(k1))
    end
    return out
end

# =====================================================================
# Pathfinder warm-start for MoMS-NS
# =====================================================================

"""
    pathfinder_warmstart_moms_ns(target, data; td, n_live, kwargs...)
        -> Vector{MoMSNSPoint}

Build `n_live` warm-start MoMSNSPoints by running multipath Pathfinder
on the full fixed-dimension posterior (all `td.max_kplanet` slots
forced active), then sampling from the per-run MVN approximations.
Each live point gets a unique parameter draw plus all γ_k = 1.

The aim is to bypass the early-locking trap that hits MoMS-NS on
multi-planet datasets: the chain births one or two planets at random
periods early, the constrained log L threshold rises, and the chain
can never explore enough trans-dim moves to find the true 3- or
4-planet posterior. By starting the live points already inside (or
near) the high-density region of the joint 4-planet posterior, NS
can shrink onto that mode rather than discovering it from scratch.

# Arguments

- `n_live`: number of live points to generate.
- `n_pathfinder_runs`: independent L-BFGS basins (default 16). Each
  draws one MVN approximation; live points are sampled across them.
- `seed`: RNG seed for L-BFGS init points. Distinct from the seed
  passed to `sample_moms_ns`.

# Notes

Pathfinder operates in unconstrained sampler space; we apply
`transform_forward` to map draws back to Nereus parameter values.
The target must be `unconstrained = true` (the default for
`NereusTarget(params, data)`).
"""
function pathfinder_warmstart_moms_ns(
    target::NereusTarget, data::Data;
    td::TransDimConfig,
    n_live::Int,
    n_pathfinder_runs::Int = 16,
    init_scale::Float64 = 1.0,
    inclusion_prior::Float64 = 0.5,
    seed::Int = 42,
    quiet::Bool = false,
)
    target.transform === nothing && throw(ArgumentError(
        "pathfinder_warmstart_moms_ns requires target.transform != nothing " *
        "(build with NereusTarget(...; unconstrained=true))"))

    pf = pathfinder_init(target;
                         n_runs = n_pathfinder_runs,
                         n_draws = max(n_live, n_pathfinder_runs * 4),
                         seed = seed,
                         quiet = quiet)

    params   = target.params
    layout   = params.layout
    pt       = target.transform
    strategy = MoMSBirth(params; init_scale = init_scale,
                          rng = MersenneTwister(seed))

    # Sample n_live draws from Pathfinder's IS-reweighted draws (which
    # already cover the high-ESS portion of the posterior). For live
    # points we want diversity, so randomly pick with replacement when
    # n_live > size(pf.draws, 2).
    n_draws_avail = size(pf.draws, 2)
    rng = MersenneTwister(seed + 1)
    pick = rand(rng, 1:n_draws_avail, n_live)

    points = Vector{MoMSNSPoint}(undef, n_live)
    for i in 1:n_live
        x_unconstrained = pf.draws[:, pick[i]]
        # Map unconstrained ℝⁿ → bounded (prior support) parameter space.
        # Nereus naming is "inverse" for unconstrained→bounded (the
        # inverse of the change-of-variable transform that took bounded
        # priors to ℝⁿ for the sampler).
        y = transform_inverse(x_unconstrained, pt)

        td_state = TransDimState(
            max_planets = td.max_kplanet,
            n_noise = length(params.config.noise_models),
        )
        # All slots active — Pathfinder fit assumes all parameters present
        for k in 1:td.max_kplanet
            activate_planet!(td_state, k)
        end
        # Noise: full-size mask indexed by config.noise_models position
        # (the old 1:length(toggleable) loop used the WRONG index space).
        # Everything starts active, then each exclusion group keeps ONE
        # randomly-chosen member per live point — "all toggleables on"
        # violated user-declared mutual exclusivity (a forbidden config
        # births/deaths can never repair).
        for nm_idx in 1:length(params.config.noise_models)
            activate_noise!(td_state, nm_idx)
        end
        for grp in td.noise_exclusion_groups
            idxs = [findfirst(==(nm), params.config.noise_models)
                    for nm in grp]
            filter!(!isnothing, idxs)
            length(idxs) >= 2 || continue
            keep = idxs[rand(rng, 1:length(idxs))]
            for nm_idx in idxs
                nm_idx == keep && continue
                td_state.noise_active[nm_idx] = false
            end
        end

        theta = Theta{Float64}(params; td = td_state)
        # Place each unfrozen value back into theta.values
        for (j, idx) in enumerate(layout.unfrozen_idx)
            theta.values[idx] = y[j]
        end

        log_pi = spike_slab_log_prior(theta, strategy, inclusion_prior)
        log_L  = _eval_ll(theta, data, nothing)

        points[i] = MoMSNSPoint(
            copy(theta.values),
            copy(theta.td.planet_active),
            copy(theta.td.noise_active),
            Float64(log_pi),
            Float64(log_L),
        )
    end
    return points
end
