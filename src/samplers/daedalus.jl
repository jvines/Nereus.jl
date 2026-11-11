# Trans-dimensional Nested Sampling via Mixtures of Mutually Singular
# distributions — Daedalus.
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
using Statistics: mean, std, cov
using LinearAlgebra: cholesky, Symmetric, LowerTriangular, Diagonal, diag, issuccess
using Printf
using MCMCChains

export sample_daedalus, model_probabilities, bayes_factors

# Inline log-add-exp to avoid pulling in LogExpFunctions
@inline function _logaddexp(a::Real, b::Real)
    a == -Inf && return float(b)
    b == -Inf && return float(a)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

"""
    DaedalusPoint

Frozen state of a live or dead point in Daedalus: parameter values, the
trans-dim indicator vector, and the cached log-prior / log-likelihood.
"""
struct DaedalusPoint
    values::Vector{Float64}
    planet_active::Vector{Bool}
    noise_active::Vector{Bool}
    log_prior::Float64
    log_L::Float64
end

"""
    sample_daedalus(target, data; td, kwargs...)
        -> (chains::MCMCChains.Chains, log_evidence, strategy::MoMSBirth,
            log_evidence_err)

Trans-dimensional nested sampler operating on the joint (β, γ)
spike-and-slab parameterisation. Returns posterior samples (importance-
resampled to equal weights), the log-evidence estimate, the
adapted MoMS strategy carrying off-values + scales, and the log-evidence's
standard error √(H / n_live) (Skilling 2006; H the information).

# Required
- `target::NereusTarget` — bounded (`unconstrained=false`) target.
- `data::Data` — observation data.
- `td::TransDimConfig` — trans-dim configuration. Birth strategies in
  `td` are ignored; Daedalus always uses MoMS proposals.

# Keywords
- `n_live::Int = 200` — number of live points.
- `dlogz::Float64 = 0.5` — termination criterion on remaining log-Z.
- `n_mcmc::Int = 25` — the fewest within-model steps of the constrained walk
  that replaces a dead point (it also continues until one is accepted). Each
  step moves every active coordinate at once by a Gaussian step shaped by the
  live points' covariance, its scale adapting between replacements toward
  half the steps accepted and fixed within each.
- `decorrelation::Real = 16.0` — the walk's length: enough steps, at the
  squared jump per step measured on the walks before it, that its accepted
  steps travel `decorrelation × d` in squared units of the live points'
  covariance (two independent live points are `2d` apart), up to
  `40 × n_mcmc` steps; fixed for each walk. `0` walks `n_mcmc` steps. A walk
  of 25 steps, one coordinate at a time at a fixed 1/50 of its prior width,
  left the replacements close to their donors: on an RV + IAD companion with
  i = 5° log Z spread over 150-169 across 8 seeds, where bridge sampling gives
  165.5 (now 165.6-166.0 over four). With i = 89.5°, 25 steps of the
  covariance-shaped walk put it at 155, against 163.9 at 16 (163.5-164.3 over
  three seeds; 32 and 64 give the same) and bridge sampling's 163.9.
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
function sample_daedalus(
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
    warm_start_points::Union{Vector{DaedalusPoint}, Nothing} = nothing,
    decorrelation::Real    = 16.0,
)
    # JSON may deliver these floats as Int; normalize.
    dlogz                   = Float64(dlogz)
    decorrelation           = Float64(decorrelation)
    init_scale              = Float64(init_scale)
    inclusion_prior         = Float64(inclusion_prior)
    informed_birth_fraction = Float64(informed_birth_fraction)
    0.0 < inclusion_prior < 1.0 || throw(ArgumentError(
        "inclusion_prior must be in (0, 1)"))
    n_live >= 2 || throw(ArgumentError("n_live must be ≥ 2"))
    0.0 <= informed_birth_fraction <= 1.0 || throw(ArgumentError(
        "informed_birth_fraction must be in [0, 1]"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be ≥ 1"))
    batch_size <= n_live ÷ 4 || throw(ArgumentError("batch_size must be ≤ n_live/4"))
    # Noise toggling supported via the existing `propose_noise_birth/death`
    # methods (RJMCMC-style prior-draw proposals). The M-H acceptance in
    # the constrained-MCMC kernel uses prior + proposal ratios with the
    # likelihood replaced by the constraint L > L_thr.

    rng    = MersenneTwister(seed)
    params = target.params

    # Births re-sort planets between same-mode slots, so those slots share one
    # window per circular angle before the off-values are built from them
    # (src/circular.jl). Done once, here, while nothing evaluates the target.
    circ_groups = circular_groups(params; permutable = td.planets)
    unify_circular_groups!(params, circ_groups)
    strategy = MoMSBirth(params; init_scale = init_scale, rng = rng)

    # ---- 1. Initialize n_live points from the spike-and-slab prior --
    # If `warm_start_points` is supplied, use those (typically built from
    # Pathfinder draws via `pathfinder_warmstart_daedalus`); they bypass
    # the random γ + random params init that traps the chain in low-Np
    # local modes on multi-planet systems.
    live = if warm_start_points !== nothing
        length(warm_start_points) == n_live || throw(ArgumentError(
            "warm_start_points length ($(length(warm_start_points))) " *
            "must equal n_live ($n_live)"))
        copy(warm_start_points)
    else
        out = Vector{DaedalusPoint}(undef, n_live)
        for i in 1:n_live
            out[i] = _daedalus_init_point(target, data, strategy, td,
                                          inclusion_prior, rng)
        end
        out
    end

    # Warm-start points come from another run's windows: relabelled into this
    # layout's (by whole periods; their cached densities stay valid).
    _daedalus_relabel!(live, params)

    # The seams of the circular angles, re-cut from the live points every
    # `recut_every` iterations (`_daedalus_recut!`): the constrained-MCMC
    # replacement walks inside each angle's one-period window, walled at its
    # seam, so a mode across a seam would be two pieces that a replacement
    # walk cannot cross between. Exact for nested sampling: a relabel by
    # whole periods leaves the constrained region, the live points' uniform
    # distribution over it and every cached density unchanged.
    recut_every = max(n_live ÷ 2, 25)

    # ---- 2. NS skeleton ---------------------------------------------
    # Skilling's recursion: at iteration i, the i-th dead point has
    # weight w_i = ΔX_i ≈ exp(-(i-1)/N) - exp(-i/N), and contributes
    # log(L_i) + log(w_i) to log Z (via logaddexp).
    log_Z          = -Inf
    log_X_prev     = 0.0  # log X_0 = 0 (full prior volume)

    dead_pts    = DaedalusPoint[]
    log_w_dead  = Float64[]   # log of width × likelihood (i.e. unnormalised weight)
    log_L_dead  = Float64[]

    iter = 0
    last_report = 0
    # The within-model walk's step, in units of the live points' covariance
    # (`_daedalus_proposal`): adapted after every replacement toward half the
    # proposals accepted, as the vendored NestedSamplers RWalk adapts its
    # scale, and frozen within each walk. Started at 2.38/√d.
    n_dim_walk = max(1, length(params.layout.unfrozen_idx))
    walk_scale = Ref(2.38 / sqrt(n_dim_walk))
    # Its length: the squared distance a walk is expected to travel, the
    # squared jumps of its accepted steps summed (`_daedalus_walk_length`),
    # measured on the walks before it and fixed for each.
    walk_jump = Ref(NaN)
    pb = ProgressBar("Daedalus";
                       total = max_iter, enabled = show_progress)
    while iter < max_iter

        # --- Pick the batch_size worst live points ----------------------
        # batch_size == 1 reduces exactly to Skilling's serial recipe.
        # batch_size > 1 kills the b worst points together (lowest first),
        # credits each its own prior volume (`_daedalus_batch_volumes`) and
        # evolves b replacements concurrently against the strict
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

        log_w_batch, log_X_curr = _daedalus_batch_volumes(log_X_prev, n_live,
                                                          batch_size)
        for (j, idx) in enumerate(dead_indices)
            push!(dead_pts,    live[idx])
            push!(log_w_dead,  log_w_batch[j])
            push!(log_L_dead,  live[idx].log_L)
            log_Z = _logaddexp(log_Z, log_w_batch[j] + live[idx].log_L)
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
        # The walks' proposals are built from the live points as they stand
        # (the survivors: the killed ones are below L_thr), and `scale` is
        # frozen for them; it adapts between replacements (below).
        donors_live = live[survivor_pool]
        counts = Ref((0, 0, 0.0))
        nw = _daedalus_walk_length(walk_jump, n_mcmc, n_dim_walk, decorrelation)
        if batch_size == 1
            # Serial path: avoid @spawn overhead for the common b=1 case.
            donor = survivor_pool[rand(rng, 1:length(survivor_pool))]
            live[dead_indices[1]] = _daedalus_evolve(live[donor], target,
                                                      data, strategy, td,
                                                      L_thr, n_mcmc,
                                                      inclusion_prior, rng;
                                                      informed_birth_fraction =
                                                          informed_birth_fraction,
                                                      live = donors_live,
                                                      scale = walk_scale[], n_walk = nw,
                                                      counts)
        else
            tasks = Vector{Task}(undef, batch_size)
            task_counts = [Ref((0, 0, 0.0)) for _ in 1:batch_size]
            for j in 1:batch_size
                donor = survivor_pool[rand(rng, 1:length(survivor_pool))]
                # Each task gets its own seeded RNG so the evolutions are
                # reproducible and independent; the master `rng` is only
                # used to draw the seeds + donor indices.
                child_rng = MersenneTwister(rand(rng, UInt64))
                start_pt = live[donor]
                tc = task_counts[j]
                sc = walk_scale[]
                tasks[j] = Threads.@spawn _daedalus_evolve(start_pt, target,
                                                            data, strategy, td,
                                                            L_thr, n_mcmc,
                                                            inclusion_prior, child_rng;
                                                            informed_birth_fraction =
                                                                informed_birth_fraction,
                                                            live = donors_live,
                                                            scale = sc, n_walk = nw,
                                                            counts = tc)
            end
            for (j, idx) in enumerate(dead_indices)
                live[idx] = fetch(tasks[j])
            end
            counts[] = (sum(c[][1] for c in task_counts), sum(c[][2] for c in task_counts),
                        sum(c[][3] for c in task_counts))
        end
        acc_, rej_, jump_ = counts[]
        _daedalus_update_scale!(walk_scale, acc_, rej_, n_dim_walk)
        # Squared jump per within-model step, smoothed over ~n_live walks.
        if acc_ + rej_ > 0
            j = jump_ / (acc_ + rej_)
            w = 1 / min(n_live, iter)
            walk_jump[] = isfinite(walk_jump[]) ? (1 - w) * walk_jump[] + w * j : j
        end

        log_X_prev = log_X_curr
        iter += batch_size
        iter % recut_every < batch_size &&
            _daedalus_recut!(live, params, circ_groups, strategy)

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

    # Dead points kept the windows of their day; the output is in the last.
    _daedalus_relabel!(dead_pts, params)
    _daedalus_relabel!(live, params)

    # ---- 3. Add remaining live points (each gets an equal share of
    # the residual prior volume X_iter / n_live).
    log_dX_residual = log_X_prev - log(n_live)
    for p in live
        log_Z = _logaddexp(log_Z, log_dX_residual + p.log_L)
        push!(dead_pts,   p)
        push!(log_w_dead, log_dX_residual)
        push!(log_L_dead, p.log_L)
    end

    # Skilling's (2006) error: H the information, nats from prior to
    # posterior, and σ(log Z) = √(H / n_live).
    H = 0.0
    for (lL, lw) in zip(log_L_dead, log_w_dead)
        isfinite(lL) || continue
        H += exp(lL + lw - log_Z) * lL
    end
    log_Z_err = sqrt(max(H - log_Z, 0.0) / n_live)

    if show_progress
        @printf("Daedalus finished: %d iterations, %d total points (%d dead + %d live).\n",
                iter, length(dead_pts), iter, n_live)
        @printf("log Z = %+.4f ± %.4f\n", log_Z, log_Z_err)
    end

    # ---- 4. Importance-resample to get equally-weighted posterior --
    log_post_w = log_L_dead .+ log_w_dead .- log_Z
    chain = _daedalus_resample_chain(dead_pts, log_post_w, params, td, rng)

    return chain, log_Z, strategy, log_Z_err
end

# Within-model steps of the next walk: enough that its accepted steps' squared
# jumps, at `jump` per step (measured on the walks so far), sum to
# `decorrelation × d` in units of the live points' covariance -- where two
# independent live points are 2d apart on average. At least `n_mcmc`, at most
# 40 × n_mcmc; `n_mcmc` until there is a measurement, or with
# `decorrelation = 0`.
function _daedalus_walk_length(jump::Base.RefValue{Float64}, n_mcmc::Int, d::Int,
                               decorrelation::Float64)
    (decorrelation > 0 && isfinite(jump[]) && jump[] > 0) || return n_mcmc
    n = decorrelation * d / jump[]
    return clamp(isfinite(n) ? ceil(Int, n) : 40n_mcmc, n_mcmc, 40 * max(n_mcmc, 1))
end

# The walk's scale after a replacement that accepted `acc` of `acc + rej`
# within-model proposals: the vendored NestedSamplers RWalk's `update_scale!`
# toward 1/2, capped at √d.
function _daedalus_update_scale!(scale::Base.RefValue{Float64}, acc::Int, rej::Int, d::Int)
    acc + rej == 0 && return scale
    ratio = acc / (acc + rej)
    scale[] = min(scale[] * exp((ratio - 0.5) / (0.5 * d)), sqrt(d))
    return scale
end

# The prior volume credited to each of the `b` lowest live points killed
# together, lowest first, and log X after them. The j-th is the lowest of the
# n_live - j + 1 live points left above the j - 1 killed below it, so it
# shrinks log X by 1/(n_live - j + 1), and b = 1 is Skilling's step. The batch
# used to shrink log X by b/n_live and share it evenly, which put log Z about
# (b - 1)·H/(2·n_live) high, as it did NestedSamplers' batch NS
# (vendor/NestedSamplers/src/parallel.jl).
function _daedalus_batch_volumes(log_X::Float64, n_live::Int, b::Int)
    log_w = Vector{Float64}(undef, b)
    for j in 1:b
        dlx = 1.0 / (n_live - j + 1)
        log_w[j] = log_X + log1p(-exp(-dlx))
        log_X -= dlx
    end
    return log_w, log_X
end

# =====================================================================
# Live-point initialisation
# =====================================================================

function _daedalus_init_point(target::NereusTarget, data::Data,
                              strategy::MoMSBirth, td::TransDimConfig,
                              inclusion_prior::Float64,
                              rng::AbstractRNG;
                              max_per_planet_retries::Int = 8)
    params = target.params
    layout = params.layout

    # Mask sized to the FULL noise-model list (consumers index by
    # config.noise_models position; see rjmcmc/transdim_pt_emcee fix) with
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

    # Every coordinate a prior draw, γ included: nested sampling's volumes
    # count from live points drawn from the prior. γ used to be set to each
    # instrument's median RV (`_seed_gammas_from_data!`, as the MCMC samplers
    # start their chains): every live point then had the same γ, the live
    # points' covariance none in γ, and the replacement walks left γ where it
    # was until about a quarter of the run had gone (RV + IAD companion,
    # i = 5°). The evidence counted the prior's γ volume as if the data's γ
    # filled it.
    _init_systemics_from_prior!(theta, rng)

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

    return DaedalusPoint(
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
    _daedalus_evolve(start, target, data, strategy, td, L_thr, n_mcmc,
                     inclusion_prior, rng)
        -> DaedalusPoint

Evolve a live point under the constraint log-L > L_thr using a small
MCMC. Each step is either:
  * a between-model MoMS proposal (planet birth/death), with M-H
    acceptance using prior + proposal ratios (no likelihood in the M-H
    ratio — the constraint replaces it), or
  * a within-model single-coordinate Gaussian random walk on the active
    subset, accepted iff prior ratio passes and L > L_thr.
"""
function _daedalus_evolve(start::DaedalusPoint, target::NereusTarget,
                          data::Data, strategy::MoMSBirth,
                          td::TransDimConfig, L_thr::Float64,
                          n_mcmc::Int, inclusion_prior::Float64,
                          rng::AbstractRNG;
                          informed_birth_fraction::Float64 = 0.0,
                          live::Union{Nothing,Vector{DaedalusPoint}} = nothing,
                          scale::Float64 = 1.0,
                          n_walk::Int = n_mcmc,
                          counts::Union{Nothing,Base.RefValue{Tuple{Int,Int,Float64}}} = nothing)
    params = target.params
    layout = params.layout

    # Scratch td sized to the FULL noise-model list — must match the
    # length of the masks carried by DaedalusPoint (see _daedalus_init_point).
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

    # The within-model proposal of each configuration (`_daedalus_proposal`),
    # built once from `live` and frozen for the walk: a fixed symmetric
    # proposal, so each step leaves the constrained prior invariant.
    props = Dict{Tuple{Vector{Bool},Vector{Bool}},Any}()
    n_within = n_acc = n_rej = steps = 0
    sq_jump = 0.0
    # `n_walk` within-model steps (fixed for the walk; `sample_daedalus` sets
    # it between walks) and at least one accepted: a walk that accepts nothing
    # returns a copy of its donor, two live points where the constrained prior
    # has one (the vendored NestedSamplers RWalk). Capped at 50 × n_walk steps.
    while (n_within < n_walk || n_acc == 0) && steps < 50 * max(n_walk, 1)
        steps += 1
        # td.planets = false: the planet set is fixed, so a between-model
        # move can only toggle noise, and with no toggleable noise there is
        # none to make (as in rjmcmc, moms and the trans-dimensional PT). The
        # planet branch below used to fire regardless: a death switched the
        # fixed planet off whenever the planet-less likelihood cleared L_thr.
        # On an RV + IAD companion at i = 5° (150 live points) one run in five
        # with e fixed, and the run with e free, ended planet-less: every
        # posterior draw on the flat planet-less likelihood, 1,800 nats below
        # the mode (log Z = -1650 against +150 to +169).
        if (td.planets || has_toggleable_noise) && rand(rng) < td.transdim_fraction
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
            # Within-model: a Gaussian step in all active coordinates at once,
            # its covariance the live points' (`_daedalus_proposal`) times
            # `scale`. The M-H ratio is the spike-and-slab prior's, constant in
            # γ here, with the likelihood replaced by the constraint.
            n_within += 1
            key = (copy(theta.td.planet_active), copy(theta.td.noise_active))
            prop = get!(() -> _daedalus_proposal(theta, live), props, key)
            res = _daedalus_within_step!(theta, data, L_thr, log_pi, prop, scale,
                                          strategy, inclusion_prior, rng)
            if res === nothing
                n_rej += 1
            else
                log_pi, log_L, jump = res
                n_acc += 1
                sq_jump += jump
            end
        end
    end
    counts === nothing ||
        (counts[] = (counts[][1] + n_acc, counts[][2] + n_rej, counts[][3] + sq_jump))

    return DaedalusPoint(
        copy(theta.values),
        copy(theta.td.planet_active),
        copy(theta.td.noise_active),
        Float64(log_pi),
        Float64(log_L),
    )
end

# The within-model proposal for the configuration `theta` is in: its active
# unfrozen slots and a lower-triangular factor `L`, a step being `scale · L z`
# with z standard normal. `L` is the Cholesky factor of the covariance of the
# live points in the same configuration -- the shape of the constrained region,
# as dynesty's and the vendored NestedSamplers' bounding ellipsoids are -- when
# there are enough of them (d + 2) and it factorises; otherwise a diagonal of
# each slot's sd over the live points where it is active, or, with fewer than
# two of those (or no `live`), 1/50 of its prior width, the fixed step this
# kernel used to take one coordinate at a time. A slot whose live points share
# one value (to rounding) takes that 1/50 step too: its covariance factorised
# with a pivot of ~1e-16 and froze the slot (γ, when the live points all
# started at the data's).
function _daedalus_proposal(theta::Theta, live::Union{Nothing,Vector{DaedalusPoint}})
    params = theta.params
    layout = params.layout
    act, pos = _active_unfrozen(params, theta.td)
    d = length(act)
    sd = Vector{Float64}(undef, d)
    for (k, p) in enumerate(pos)
        lo, hi = bounds(layout.unfrozen_priors[p])
        sd[k] = isfinite(lo) && isfinite(hi) ? (hi - lo) / 50 :
                0.05 * abs(theta.values[act[k]]) + 1e-3
    end
    live === nothing && return (act = act, pos = pos, L = LowerTriangular(Matrix(Diagonal(sd))))
    same = [q for q in live if q.planet_active == theta.td.planet_active &&
                               q.noise_active == theta.td.noise_active]
    spread(s, m, k) = s > 1e-12 * (abs(m) + sd[k])
    if length(same) >= d + 2
        X = [q.values[a] for q in same, a in act]
        C = cov(X)
        m = vec(mean(X; dims = 1))
        if all(k -> spread(sqrt(max(C[k, k], 0.0)), m[k], k), 1:d)
            F = cholesky(Symmetric(C); check = false)
            issuccess(F) && all(>(0), diag(F.L)) && return (act = act, pos = pos, L = F.L)
        end
    end
    for k in 1:d
        v = Float64[q.values[act[k]] for q in live if _slot_active(q, act[k], params)]
        length(v) >= 2 && spread(std(v), mean(v), k) && (sd[k] = std(v))
    end
    return (act = act, pos = pos, L = LowerTriangular(Matrix(Diagonal(sd))))
end

# Whether theta slot `s` is active at the live point `q` (its planet on, or
# its noise model on).
function _slot_active(q::DaedalusPoint, s::Int, params::Params)
    L = params.layout
    for k in eachindex(q.planet_active)
        s in planet_slot_indices(L.planet_blocks[k]) && return q.planet_active[k]
    end
    for (nm_idx, nm) in enumerate(params.config.noise_models)
        nm_idx <= length(q.noise_active) || continue
        for name in noise_param_names(nm, params.config.instruments)
            get(L.name_to_idx, name, 0) == s && return q.noise_active[nm_idx]
        end
    end
    return true
end

"""Propose `scale · L z` in the active coordinates of `prop`
(`_daedalus_proposal`), accept iff every coordinate stays inside its prior's
bounds, L > L_thr and the prior ratio passes. Returns the new (log_pi, log_L,
squared jump) on accept (with `log_pi` in spike-and-slab units and the jump
`scale² |z|²` in units of the proposal's covariance), `nothing` on reject,
with `theta` restored."""
function _daedalus_within_step!(theta::Theta, data::Data, L_thr::Float64,
                                  log_pi::Float64, prop, scale::Float64,
                                  strategy::MoMSBirth, inclusion_prior::Float64,
                                  rng::AbstractRNG)
    act = prop.act
    isempty(act) && return nothing
    layout = theta.params.layout
    old = theta.values[act]
    z = randn(rng, length(act))
    step = prop.L * z
    for (k, a) in enumerate(act)
        v = old[k] + scale * step[k]
        lo, hi = bounds(layout.unfrozen_priors[prop.pos[k]])
        if (isfinite(lo) && v < lo) || (isfinite(hi) && v > hi)
            theta.values[act] .= old
            return nothing
        end
        theta.values[a] = v
    end
    new_log_pi = spike_slab_log_prior(theta, strategy, inclusion_prior)
    if !isfinite(new_log_pi)
        theta.values[act] .= old
        return nothing
    end
    new_log_L = _eval_ll(theta, data, nothing)
    if !isfinite(new_log_L) || new_log_L <= L_thr
        theta.values[act] .= old
        return nothing
    end
    # Symmetric Gaussian proposal cancels — M-H reduces to prior ratio
    if log(rand(rng)) < new_log_pi - log_pi
        return (new_log_pi, new_log_L, scale^2 * sum(abs2, z))
    end
    theta.values[act] .= old
    return nothing
end

# =====================================================================
# Posterior reconstruction (importance resampling)
# =====================================================================

"""Importance-resample dead+live points to equal weights and assemble
the same chain layout as `sample_rjmcmc`/`sample_moms` for downstream
post-processing."""
function _daedalus_resample_chain(points::Vector{DaedalusPoint},
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

@inline function argmin_log_L(live::Vector{DaedalusPoint})
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
/ `sample_daedalus`). All four samplers store an `:n_planets` column;
this helper just tabulates it. For Daedalus chains the result is
proportional to evidence × prior — i.e. you can read off Bayes factors
directly via `bayes_factors`.

`max_kplanet` defaults to the largest observed N_p in the chain.
"""
function model_probabilities(chains;
                              max_kplanet::Union{Nothing, Int} = nothing)
    np = _draws(chains, :n_planets)
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
# Pathfinder warm-start for Daedalus
# =====================================================================

"""
    pathfinder_warmstart_daedalus(target, data; td, n_live, kwargs...)
        -> Vector{DaedalusPoint}

Build `n_live` warm-start DaedalusPoints by running multipath Pathfinder
on the full fixed-dimension posterior (all `td.max_kplanet` slots
forced active), then sampling from the per-run MVN approximations.
Each live point gets a unique parameter draw plus all γ_k = 1.

The aim is to bypass the early-locking trap that hits Daedalus on
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
  passed to `sample_daedalus`.

# Notes

Pathfinder operates in unconstrained sampler space; we apply
`transform_forward` to map draws back to Nereus parameter values.
The target must be `unconstrained = true` (the default for
`NereusTarget(params, data)`).
"""
function pathfinder_warmstart_daedalus(
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
        "pathfinder_warmstart_daedalus requires target.transform != nothing " *
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

    points = Vector{DaedalusPoint}(undef, n_live)
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

        points[i] = DaedalusPoint(
            copy(theta.values),
            copy(theta.td.planet_active),
            copy(theta.td.noise_active),
            Float64(log_pi),
            Float64(log_L),
        )
    end
    return points
end

# =====================================================================
# Circular seams and polar charts of the live points
# =====================================================================

# Every point's circular angles relabelled into the layout's windows and its
# polar angles folded to [0, π], in place. Exact: the prior and likelihood are
# periodic in the one and symmetric under the other, and the log-prior skips
# inactive slots, so the cached densities stay valid.
function _daedalus_relabel!(pts::AbstractVector{DaedalusPoint}, params::Params)
    L = params.layout
    circ = circular_indices(params)
    for p in pts
        for d in circ
            lo, hi = bounds(L.unfrozen_priors[d])
            s = L.unfrozen_idx[d]
            p.values[s] = circular_relabel(p.values[s], lo, hi)
        end
        _fold_polar_slots!(p.values, params)
    end
    return pts
end

"""
    _daedalus_recut!(live, params, groups, strategy)

Move each circular group's seam to the emptiest arc of the live points' values
(active planets only; [`recut_circular_group!`](@ref)), relabel every live point
into the new windows and fold its polar angles. The MoMS off-values move with
their windows, as at moms' warmup re-cut, so a blind birth keeps its centre
relative to the window rather than landing on the new seam. Births and deaths
read the same off-value, so the replacement kernel stays balanced; the inactive
slots it governs carry no prior or likelihood.
"""
function _daedalus_recut!(live::Vector{DaedalusPoint}, params::Params, groups,
                          strategy::MoMSBirth)
    L = params.layout
    for grp in groups
        draws = Float64[]
        for p in live, (d, k) in grp
            1 <= k <= length(p.planet_active) && !p.planet_active[k] && continue
            push!(draws, p.values[L.unfrozen_idx[d]])
        end
        isempty(draws) && continue
        old = Float64[L.unfrozen_priors[d].lo for (d, _) in grp]
        recut_circular_group!(params, grp, draws) === nothing && continue
        for (j, (d, _)) in enumerate(grp)
            ps = L.unfrozen_priors[d]
            shift = ps.lo - old[j]
            for k in eachindex(strategy.slot_indices)
                q = findfirst(==(d), strategy.slot_indices[k])
                q === nothing ||
                    (strategy.off_values[k][q] = moved_off_value(strategy.off_values[k][q],
                                                                 ps, shift))
            end
        end
    end
    return _daedalus_relabel!(live, params)
end
