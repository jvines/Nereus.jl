# Importance Nested Sampling (Feroz, Hobson, Cameron, Pettitt 2013;
# MultiNest's INS extension; arXiv:1306.2144).
#
# Standard NS computes log Z from the iso-likelihood shell estimator
#   Z ≈ Σ_k L_k · (X_{k-1} − X_k)
# where {L_k} are the accepted dead points and X_k = exp(-k/N) is the
# Skilling prior-mass estimate. Only the *accepted* live-point
# replacements contribute.
#
# Importance NS reuses *every* likelihood evaluation made during the
# run — including the proposals that were rejected for being below the
# current likelihood threshold — by treating the sequence of bound-
# proposals as one importance sampler with target the prior and
# proposal the per-iteration uniform-on-ellipsoid distribution.
# Estimator:
#   Z_INS = (1/N_total_evals) Σ_i L_i · V(E_i) · I[u_i ∈ unit cube]
# where V(E_i) is the volume of the bounding ellipsoid at iteration i.
# Mass that NS throws away as "rejected" is now signal for INS.
#
# Why alongside `sample_nested`: Nereus's `sample_nested` delegates to
# NestedSamplers.jl which hides the rejected-proposal stream. Adding
# INS requires owning the NS loop. This implementation runs a standard
# rejection-sampling NS using NestedSamplers.jl's Bounds machinery
# (Ellipsoid / MultiEllipsoid) for the bound-fitting only.

using NestedSamplers
using Random
using MCMCChains
using LinearAlgebra: det, dot, mul!

export sample_nested_ins, INSResult

"""
    INSResult

Container for a `sample_nested_ins` run.

# Fields
- `chains::MCMCChains.Chains` — posterior samples (importance-resampled).
- `log_z_ns::Float64` — Skilling-style NS log Z (iso-likelihood shells).
- `log_z_ins::Float64` — the sum of `L·V` over every evaluation, divided by
  their number, `V` the volume of the bound it was drawn from. `NaN` unless
  `proposal = :unif`: the walk and slice kernels do not draw from the bound.
  Not an evidence to report even then: each term estimates the evidence
  inside the bound its point was drawn from, which shrinks as the run goes
  on, so it comes out low. The unbiased form (Feroz et al. 2013) weighs each
  point by every bound that contains it. `log_z_ns` is the evidence.
- `n_iters::Int` — outer NS iterations completed.
- `n_evals::Int` — total likelihood evaluations (NS + rejected proposals).
"""
struct INSResult
    chains::MCMCChains.Chains
    log_z_ns::Float64
    log_z_ins::Float64
    n_iters::Int
    n_evals::Int
end

"""
    sample_nested_ins(target, data; kwargs...) -> INSResult

Nested sampling with Nereus's own loop and kernels: the posterior and the
NS log Z (`log_z_ns`), plus, with `proposal = :unif`, a sum over every
likelihood evaluation (`log_z_ins`, see [`INSResult`](@ref): not an evidence
to report).

Operates in **unit-cube space**: `u ∈ [0,1]^n_dim`, prior-transformed
to bounded params via per-coordinate `quantile(prior, u)`. The
bounding ellipsoid is fitted to live points in unit-cube space and
proposals are drawn uniformly from it; both standard NS and INS
machinery use the same draws.

# Keywords
- `n_live::Int=400`
- `bounds::Symbol=:multi` (`:multi` MultiEllipsoid, `:single` Ellipsoid)
- `enlarge::Real=1.25` — ellipsoid enlargement factor (Mukherjee+ 2006).
- `dlogz::Real=0.5` — stopping when log(Z_remaining/Z_done) < dlogz.
- `max_iter::Int=50_000`
- `bound_update_interval::Int=50` — refit ellipsoid every N iterations.
- `n_walks::Union{Int,Nothing}=nothing` — Metropolis steps per replacement
  (`:rwalk`) or slice sweeps (`:slice`). `nothing` → dimension-scaled
  `max(25, 2·n_dim)` (a fixed count under-decorrelates as n_dim grows → logZ
  biased low). Explicit value honored verbatim.
- `walk_scale::Real=1.0` — initial proposal scale: for `:rwalk` the step in
  units of the bounding ellipsoid's axes, adapted between chains toward 50 %
  acceptance and fixed within each; for `:slice` the initial slice width.
- `proposal::Symbol=:rwalk` — within-shell move: `:rwalk` (random walk),
  `:slice` (random-direction slice), or `:unif` (rejection; low-d only).
  These are Nereus's own kernels: every likelihood call they make is
  recorded for `log_z_ins`.
- `min_X_shrinkage::Real=10.0` — min log prior-volume shrinkage before
  the dlogz stop can trigger.
- `seed::Int=1`
- `verbose::Bool=false`
- `show_progress::Bool=true` — a `nested INS` progress line every 10 s, with
  the fields of `sample_nested`'s: iteration, `logz`, `dlogz`, `stage`, `ncall`.
  The stage tracks only the dlogz half of the stopping rule, not
  `min_X_shrinkage`.
"""
function sample_nested_ins(
    target::NereusTarget,
    data::Data;
    n_live::Int = 400,
    bounds::Symbol = :multi,
    enlarge::Real = 1.25,
    dlogz::Real = 0.01,
    max_iter::Int = 200_000,
    bound_update_interval::Int = 50,
    n_walks::Union{Int,Nothing} = nothing,
    walk_scale::Real = 1.0,
    proposal::Symbol = :rwalk,
    min_X_shrinkage::Real = 10.0,
    seed::Int = 1,
    verbose::Bool = false,
    show_progress::Bool = true,
)
    proposal in (:rwalk, :slice, :unif) ||
        throw(ArgumentError("proposal must be :rwalk, :slice, or :unif " *
                            "(got :$proposal)"))
    # JSON config delivers `1` as Int64; the internal hot-loop helpers
    # (e.g. _rwalk_within_shell!) are strictly Float64-typed. Normalize
    # the float-valued kwargs at the boundary so the rest of the
    # function keeps its concrete types.
    enlarge         = Float64(enlarge)
    dlogz           = Float64(dlogz)
    walk_scale      = Float64(walk_scale)
    min_X_shrinkage = Float64(min_X_shrinkage)
    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    # Dimension-scaled within-shell decorrelation: a fixed step count is
    # dimension-blind and under-decorrelates as n_dim grows → logZ biased low
    # (PolyChord `num_repeats ∝ nDims`). `nothing` → max(25, 2·n_dim); an
    # explicit value is honored verbatim. See sample_nested / project_ns_rslice_undermixing.
    n_walks = n_walks === nothing ? max(25, 2 * n_dim) : n_walks
    unfrozen_idx = layout.unfrozen_idx
    unfrozen_priors = layout.unfrozen_priors
    rng = MersenneTwister(seed)
    ns_tr = target.transform === nothing ? () : (target.transform,)

    # Per-slot Theta buffer. NS outer loop is sequential (one replacement
    # per iter, run on slot 1 by default) but the init pass threads over
    # chunks of live points, and RWalk's downstream
    # `transit_log_likelihood` photometry threading both benefit from
    # having a Theta per chunk to avoid contention.
    #
    # Buffers are keyed by CHUNK, never by `Threads.threadid()`: the id
    # spans every threadpool while `Threads.nthreads()` counts only the
    # default one, so `thread_theta[threadid()]` overran this vector on
    # stock Julia >= 1.12 (see src/threading.jl).
    n_slots = _nthread_chunks()
    thread_theta = [Theta{Float64}(params) for _ in 1:n_slots]
    # Per-slot PTWorkspace → ws-aware likelihood (preallocated buffers, no
    # per-call GC; per-planet flux cache + total phot-ll cache). See nested.jl.
    thread_ws = [PTWorkspace(params, params.config.max_kplanet,
                             length(params.config.noise_models);
                             n_obs = length(data.t_rv), n_phot = length(data.t_phot))
                 for _ in 1:n_slots]

    # Likelihood at a unit-cube point u ∈ [0,1]^n_dim
    @inline function eval_loglike(u::AbstractVector{<:Real},
                                    slot::Int = 1)
        theta = thread_theta[slot]
        wb = thread_ws[slot]
        # transform u → bounded x via per-coord prior quantile
        @inbounds for (j, idx) in enumerate(unfrozen_idx)
            ps = unfrozen_priors[j]
            theta.values[idx] = prior_transform(u[j], _ns_prior(ps))
        end
        ll = rv_log_likelihood(theta, data, wb)
        isfinite(ll) || return -Inf
        ll += transit_log_likelihood(theta, data, wb)
        ll += tomogram_log_likelihood(theta, data, wb)  # residual maps
        return ll
    end

    BoundT = bounds === :multi ? NestedSamplers.Bounds.MultiEllipsoid :
             bounds === :single ? NestedSamplers.Bounds.Ellipsoid :
             throw(ArgumentError("bounds must be :multi or :single"))

    # --- Initialize n_live unit-cube live points (threaded) -----------
    # Buffers are keyed by CHUNK (see src/threading.jl), so `slot` is a
    # plain loop counter, in bounds by construction regardless of thread
    # count. `:static` is kept for the 1:1 chunk-to-thread mapping.
    live_u = rand(rng, n_dim, n_live)
    live_logL = Vector{Float64}(undef, n_live)
    init_chunks = _chunk_ranges(n_live, n_slots)
    Threads.@threads :static for slot in 1:length(init_chunks)
        for i in init_chunks[slot]
            live_logL[i] = eval_loglike(@view(live_u[:, i]), slot)
        end
    end
    n_evals = n_live

    # Fit initial bound (with MultiEllipsoid fallback to single Ellipsoid
    # when k-means clustering fails on near-duplicate live points).
    # :unif draws its points from the bound, which must then cover the
    # constrained region, so the bound is bootstrap-expanded for it as in
    # sample_nested (`Bounds.bootstrap_expansion`): fitted to the live points
    # and enlarged by 1.25 in volume, a MultiEllipsoid missed 12-22% of the
    # constrained region of an RM obliquity fit. The walks take only their
    # shape from the bound.
    n_boot = proposal === :unif ? 5 : 0
    pointvol = 1.0 / n_live
    bound = try
        NestedSamplers.Bounds.fit(rng, BoundT, live_u;
                    pointvol = pointvol, enlarge = enlarge, bootstrap = n_boot)
    catch err
        err isa ArgumentError || rethrow(err)
        NestedSamplers.Bounds.fit(rng, NestedSamplers.Bounds.Ellipsoid, live_u;
                    pointvol = pointvol, enlarge = enlarge, bootstrap = n_boot)
    end

    # INS accumulators (one entry per likelihood evaluation made during
    # rejection sampling). Volume is the bound volume at that iter.
    ins_log_L  = Float64[]
    ins_log_V  = Float64[]

    # Standard NS accumulators (dead points)
    dead_logL    = Float64[]
    dead_logwt   = Float64[]    # log of shell weight = log(X_{k-1} - X_k)
    dead_u       = Vector{Vector{Float64}}()
    # Pre-size — bounds the upper end of how many dead points we
    # accumulate before the dlogz check trips. Avoids array-growth
    # copies during the hot loop on long runs.
    sizehint!(dead_logL,  max_iter + n_live)
    sizehint!(dead_logwt, max_iter + n_live)
    sizehint!(dead_u,     max_iter + n_live)
    sizehint!(ins_log_L,  10 * max_iter)
    sizehint!(ins_log_V,  10 * max_iter)

    # Pre-allocated RWalk scratch — re-used every iter. Replaces the
    # `randn(rng, n_dim)` allocation that ran 25-100 × per iter.
    scratch_cur   = Vector{Float64}(undef, n_dim)
    scratch_prop  = Vector{Float64}(undef, n_dim)
    scratch_noise = Vector{Float64}(undef, n_dim)
    worst_save    = Vector{Float64}(undef, n_dim)

    # The random walk's step size, in units of the bound's axes: fixed within
    # each chain, adapted between them (`_rwalk_within_shell!`).
    walk_scale_now = Ref(walk_scale)

    log_z_ns = -Inf
    log_X = 0.0  # log(prior volume contained)
    iter = 0
    pb = ProgressBar("nested INS"; total = 0, print_every = 10.0, enabled = show_progress)
    stage = _NestedStage(dlogz)

    while iter < max_iter
        iter += 1

        # Shell volume (log-stable)
        log_X_new = -iter / n_live
        log_dV = log_X + log1p(-exp(log_X_new - log_X))   # log(X_{k-1} - X_k)

        # Worst live point
        idx_worst = argmin(live_logL)
        logL_worst = live_logL[idx_worst]

        # Shell contribution to NS log Z
        log_wt = log_dV + logL_worst
        log_z_ns = log_z_ns == -Inf ? log_wt :
                   max(log_z_ns, log_wt) + log1p(exp(-abs(log_z_ns - log_wt)))
        push!(dead_logL, logL_worst)
        push!(dead_logwt, log_wt)
        # Save the worst-live coords into a fresh vector (chain holds
        # references; we can't share scratch). Pre-allocated `worst_save`
        # is reused locally then copied. Slight allocation savings vs
        # `copy(live_u[:, idx_worst])` which allocs the column twice.
        @inbounds for d in 1:n_dim
            worst_save[d] = live_u[d, idx_worst]
        end
        push!(dead_u, copy(worst_save))

        log_X = log_X_new

        # Replace worst via the selected within-shell proposal:
        #   :rwalk — Gaussian random walk (robust default, moderate-high d)
        #   :slice — random-direction slice sampling (best in high d /
        #            strongly-correlated posteriors)
        #   :unif  — pure rejection inside the bound (low-d only)
        log_V_cur = log(NestedSamplers.Bounds.volume(bound))
        new_u, new_logL = if proposal === :rwalk
            _rwalk_within_shell!(live_u, live_logL, rng, eval_loglike,
                                  n_dim, n_live, idx_worst, logL_worst,
                                  bound, log_V_cur, n_walks, walk_scale_now,
                                  ins_log_L, ins_log_V,
                                  scratch_cur, scratch_prop, scratch_noise)
        elseif proposal === :slice
            _slice_within_shell!(live_u, live_logL, rng, eval_loglike,
                                  n_dim, n_live, idx_worst, logL_worst,
                                  bound, log_V_cur, n_walks, walk_scale,
                                  ins_log_L, ins_log_V,
                                  scratch_cur, scratch_prop, scratch_noise)
        else  # :unif
            u_r, l_r, bound = _rejection_within_bound!(rng, eval_loglike, n_dim,
                                      logL_worst, bound, log_V_cur,
                                      ins_log_L, ins_log_V;
                                      refit = let pv = exp(log_X) / n_live
                                          () -> _fit_ins_bound(rng, BoundT, live_u, pv,
                                                               enlarge, n_boot)
                                      end,
                                      refit_every = round(Int, 1.5 * n_live))
            (u_r, l_r)
        end
        if new_u === nothing
            # stopping would return a truncated run's evidence
            proposal === :unif && _rejection_failed(iter)
            verbose && @warn "INS: no new candidate accepted at iter $iter; stopping"
            break
        end
        # n_evals = init-pass + all proposals (in-cube + out-of-cube).
        # `ins_log_V` accumulates one entry per proposal, so it's the
        # authoritative count.
        n_evals = n_live + length(ins_log_V)
        @inbounds for d in 1:n_dim
            live_u[d, idx_worst] = new_u[d]
        end
        live_logL[idx_worst] = new_logL

        # Refit bound periodically (after several live points have
        # turned over, the bound needs to track the shrinking support).
        # MultiEllipsoid.fit calls k-means; clustering NaN's on near-
        # duplicate live points (RWalk-in-tight-shell signature). Fall
        # back to a single Ellipsoid bound in that case.
        if iter % bound_update_interval == 0
            # Seams of the circular angles moved off the live points first
            # (`_ns_recut!`): the refit below is in the new chart.
            _ns_recut!(live_u, (dead_u,), params; transforms = ns_tr)
            pv = exp(log_X) / n_live
            try
                bound = NestedSamplers.Bounds.fit(rng, BoundT, live_u;
                    pointvol = pv, enlarge = enlarge, bootstrap = n_boot)
            catch err
                err isa ArgumentError || rethrow(err)
                bound = NestedSamplers.Bounds.fit(rng, NestedSamplers.Bounds.Ellipsoid, live_u;
                    pointvol = pv, enlarge = enlarge, bootstrap = n_boot)
            end
        end

        # Stopping: require BOTH that the remaining-evidence-fraction
        # `dlogz_now` is small AND that the prior volume has actually
        # shrunk by `min_X_shrinkage` log-units (~exp(10) ≈ 22000 ×).
        # The latter prevents premature termination in high-d where
        # `dlogz_now` can look small at startup when log_z_ns is still
        # below log_z_remain by a few hundred nats — that subtraction
        # `log_z_remain - log_z_ns` going negative makes log1p(exp(.))
        # tiny even though we haven't started integrating the peak.
        max_live_logL = maximum(live_logL)
        log_z_remain = log_X + max_live_logL
        dlogz_now = log_z_ns > -Inf ?
                    log1p(exp(log_z_remain - log_z_ns)) :
                    Inf
        done = dlogz_now < dlogz && -log_X > min_X_shrinkage
        # The stopping iteration prints whatever the throttle says, so the
        # run's last line is its final state.
        pb.enabled && update!(pb; n_done = iter, force = done,
                              fields = _nested_fields(stage, log_z_ns, dlogz_now, n_evals))
        done && break

        verbose && iter % 500 == 0 && @info @sprintf(
            "iter %d  log Z_ns=%.3f  Δlog Z=%.4f  log_X=%.2f  n_evals=%d",
            iter, log_z_ns, dlogz_now, log_X, n_evals)
    end
    finish!(pb)

    # Final live-points contribution
    log_X_final = log_X
    for i in 1:n_live
        log_wt = log_X_final - log(n_live) + live_logL[i]
        log_z_ns = log_z_ns == -Inf ? log_wt :
                   max(log_z_ns, log_wt) + log1p(exp(-abs(log_z_ns - log_wt)))
        push!(dead_logL, live_logL[i])
        push!(dead_logwt, log_wt)
        push!(dead_u, copy(live_u[:, i]))
    end

    # --- log(Σ L_i · V_i) − log(N_total_evals), see INSResult -----
    # Defined only for draws from the bound: the walk and slice kernels move
    # within the constraint, and their points have no proposal volume. With
    # :rwalk it came out 1.2-1.7 nats above pt_emcee's evidence on an RM
    # obliquity fit and up to 3.4 above on an eccentric-RV fit, while run_job
    # reported it as the evidence.
    finite_idx = findall(isfinite, ins_log_L)
    if proposal !== :unif
        log_z_ins = NaN
    elseif isempty(finite_idx)
        log_z_ins = -Inf
    else
        terms = ins_log_L[finite_idx] .+ ins_log_V[finite_idx]
        m = maximum(terms)
        log_z_ins = m + log(sum(exp.(terms .- m))) - log(n_evals)
    end

    # --- Posterior chain construction ----------------------------------
    # Systematic importance resampling clones the MAP when the posterior
    # is peaked enough that `n_eff ≪ N`: all N indices land on the same
    # max-weight dead point. To preserve actual posterior diversity we
    # split on `n_eff` (Kish 1965): if it's large enough for unbiased
    # resampling, do the standard systematic draw; otherwise return the
    # top-N unique dead points sorted by weight descending. The latter
    # is biased low (mode-heavy) but it shows the *support* of the
    # posterior, not a degenerate single-point chain.
    n_dead = length(dead_logL)
    log_w = Float64[dead_logwt[i] - log_z_ns for i in 1:n_dead]
    log_w_max = maximum(log_w)
    w = exp.(log_w .- log_w_max)
    w ./= sum(w)
    n_eff = round(Int, 1 / sum(w .^ 2))

    if n_eff >= 200
        # Standard importance resampling
        n_resample = n_eff
        indices = _systematic_resample(rng, w, n_resample)
    else
        # Peaked posterior — take the top dead points by weight, keeping
        # uniqueness. This avoids the resample-collapse failure on real-
        # data 22-d targets.
        perm = sortperm(w; rev = true)
        n_keep = min(max(n_eff, 200), n_dead)
        indices = perm[1:n_keep]
        @warn "INS posterior: n_eff = $(n_eff) (< 200). Returning " *
              "top-$(n_keep) dead points by weight rather than " *
              "systematic resampling. To get a properly resampled " *
              "chain, increase n_live or run longer."
    end
    n_resample = length(indices)

    # Build the bounded-space chain (apply per-coord prior transform)
    samples = Matrix{Float64}(undef, n_resample, n_dim)
    for (i, idx) in enumerate(indices)
        for (j, _) in enumerate(unfrozen_idx)
            samples[i, j] = prior_transform(dead_u[idx][j], _ns_prior(unfrozen_priors[j]))
        end
    end

    param_names = Symbol.(layout.unfrozen_names)
    chains = MCMCChains.Chains(samples, param_names)
    return INSResult(chains, log_z_ns, log_z_ins, iter, n_evals)
end


# =====================================================================
# Within-shell samplers — replace pure rejection sampling for high-d
# =====================================================================

"""
    _rwalk_within_shell!(live_u, live_logL, rng, eval_loglike, n_dim,
                          n_live, idx_worst, logL_worst, bound, log_V_cur,
                          n_walks, scale, ins_log_L, ins_log_V)
    -> (new_u, new_logL) or (nothing, -Inf)

Within-shell random walk: from a live point inside the constraint, chosen at
random among those other than the one being replaced, exactly `n_walks`
Metropolis steps `u + σ·A·z` inside the unit cube and the constraint
`L > logL_worst`, with `z` uniform in the unit ball and `A` the axes of a
bounding ellipsoid around the start (`_walk_ellipsoid`). The step size
`σ = scale[]` is the same for every step of the chain; after it, `scale[]`
moves toward 50 % acceptance, as the vendored `Proposals.RWalk`'s does
(dynesty's rule). A chain with no step accepted returns its start. Every
proposal feeds the INS accumulator (accepted or not, the importance estimator
counts all evaluations the same way).

The walk used to adapt its step inside the chain: growing it when the running
acceptance was low and shrinking it when high, halving it after five
rejections in a row, and counting `n_walks` accepted steps rather than steps.
Where such a chain ends depends on its own acceptances, so the kernel does not
leave the uniform distribution inside the constraint unchanged; on an RM
obliquity fit `sample_nested_ins` and `sample_nested_dynamic` put log Z about
2 nats above pt_emcee's.

Returns `(nothing, -Inf)` only if no live point lies strictly above
`logL_worst`.
"""
function _rwalk_within_shell!(live_u::AbstractMatrix{Float64},
                               live_logL::AbstractVector{Float64},
                               rng::AbstractRNG, eval_loglike::Function,
                               n_dim::Int, n_live::Int, idx_worst::Int,
                               logL_worst::Float64, bound,
                               log_V_cur::Float64, n_walks::Int,
                               scale::Base.RefValue{Float64},
                               ins_log_L::Vector{Float64},
                               ins_log_V::Vector{Float64},
                               # Pre-allocated scratch (caller owns;
                               # avoids per-RWalk-call allocations of
                               # `randn(rng, n_dim)` and the proposal
                               # vector — at ~25 walks × 2M iters this
                               # was a 350 MB-of-GC-pressure cost).
                               scratch_cur::Vector{Float64},
                               scratch_prop::Vector{Float64},
                               scratch_noise::Vector{Float64};
                               slot::Int = 1)
    start_idx = _pick_start(rng, live_logL, idx_worst, logL_worst)
    start_idx == 0 && return (nothing, -Inf)
    @inbounds for d in 1:n_dim
        scratch_cur[d] = live_u[d, start_idx]
    end
    cur_logL = live_logL[start_idx]
    A = _walk_ellipsoid(rng, bound, scratch_cur).axes

    σ = scale[]
    n_accept = 0
    for _ in 1:n_walks
        _rand_ball!(rng, scratch_noise)
        mul!(scratch_prop, A, scratch_noise)
        in_cube = true
        @inbounds for d in 1:n_dim
            x = scratch_cur[d] + σ * scratch_prop[d]
            scratch_prop[d] = x
            (x < 0.0 || x > 1.0) && (in_cube = false)
        end
        if !in_cube
            # out of the cube: a rejected step
            push!(ins_log_L, -Inf)
            push!(ins_log_V, log_V_cur)
            continue
        end
        logL_prop = eval_loglike(scratch_prop, slot)
        push!(ins_log_L, logL_prop)
        push!(ins_log_V, log_V_cur)
        if logL_prop > logL_worst
            @inbounds for d in 1:n_dim
                scratch_cur[d] = scratch_prop[d]
            end
            cur_logL = logL_prop
            n_accept += 1
        end
    end
    # between chains only (vendored `update_scale!`)
    ratio = n_accept / n_walks
    scale[] = min(σ * exp((ratio - 0.5) / (0.5 * n_dim)), sqrt(n_dim))
    return (scratch_cur, cur_logL)
end

# A live point other than `idx_worst` strictly above `logL_worst`, uniformly
# at random; 0 if there is none. (The walks used to take the first such point
# after a random offset, which favoured the point after `idx_worst`.)
function _pick_start(rng::AbstractRNG, live_logL::AbstractVector{Float64},
                     idx_worst::Int, logL_worst::Float64)
    n = 0
    @inbounds for i in eachindex(live_logL)
        (i != idx_worst && live_logL[i] > logL_worst) && (n += 1)
    end
    n == 0 && return 0
    k = rand(rng, 1:n)
    @inbounds for i in eachindex(live_logL)
        if i != idx_worst && live_logL[i] > logL_worst
            k -= 1
            k == 0 && return i
        end
    end
    return 0
end

# The ellipsoid whose axes shape a walk or slice chain starting at `u`: one
# of the bound's ellipsoids holding `u`, at random, as the vendored samplers
# choose (`Bounds.bound_containing`); if none holds it (the bound is refit
# only every `bound_update_interval` iterations), the nearest in its own
# metric. The shape depends on the start only, never on the chain after it.
function _walk_ellipsoid(rng::AbstractRNG, bound, u::AbstractVector{Float64})
    e = NestedSamplers.Bounds.bound_containing(rng, bound, u)
    e === nothing || return e
    ells = bound isa NestedSamplers.Bounds.MultiEllipsoid ? bound.ellipsoids : (bound,)
    best = first(ells)
    dbest = Inf
    for ell in ells
        δ = u .- ell.center
        dd = dot(δ, ell.A * δ)
        dd < dbest && (best = ell; dbest = dd)
    end
    return best
end

# `z` in place: uniform in the unit ball.
function _rand_ball!(rng::AbstractRNG, z::Vector{Float64})
    Random.randn!(rng, z)
    n = length(z)
    r = rand(rng)^(1 / n) / sqrt(sum(abs2, z))
    z .*= r
    return z
end

# Evaluate the constrained log-likelihood at `scratch_cur + t·dir`,
# writing the trial point into `scratch_prop`. Points outside the unit
# cube score -Inf. Every probe feeds the INS accumulator. Shared by the
# slice kernel's step-out and shrink phases.
@inline function _slice_probe!(scratch_prop::Vector{Float64},
                                scratch_cur::Vector{Float64},
                                dir::Vector{Float64}, t::Float64,
                                n_dim::Int, eval_loglike::Function, slot::Int,
                                log_V_cur::Float64,
                                ins_log_L::Vector{Float64},
                                ins_log_V::Vector{Float64})
    in_cube = true
    @inbounds for d in 1:n_dim
        x = scratch_cur[d] + t * dir[d]
        scratch_prop[d] = x
        (x < 0.0 || x > 1.0) && (in_cube = false)
    end
    if !in_cube
        push!(ins_log_L, -Inf)
        push!(ins_log_V, log_V_cur)
        return -Inf
    end
    ll = eval_loglike(scratch_prop, slot)
    push!(ins_log_L, ll)
    push!(ins_log_V, log_V_cur)
    return ll
end

"""
    _slice_within_shell!(live_u, live_logL, rng, eval_loglike, n_dim,
                          n_live, idx_worst, logL_worst, bound, log_V_cur,
                          n_walks, walk_scale, ins_log_L, ins_log_V,
                          scratch_cur, scratch_prop, scratch_noise)
    -> (new_u, new_logL) or (nothing, -Inf)

Within-shell random-direction slice sampling (dynesty `rslice` style).
From a random viable live point, perform `n_walks` slice updates: each
draws an isotropic unit direction, brackets a slice interval by stepping
out while `L > logL_worst`, then shrinks the interval until a point
satisfying the constraint is found. More robust than the random walk in
high dimensions / strongly-correlated posteriors because the slice
interval auto-scales to the local constraint geometry along each
direction. Every likelihood probe feeds the INS accumulator.

`walk_scale` sets the initial slice half-width (`walk_scale / √n_dim`).
"""
function _slice_within_shell!(live_u::AbstractMatrix{Float64},
                               live_logL::AbstractVector{Float64},
                               rng::AbstractRNG, eval_loglike::Function,
                               n_dim::Int, n_live::Int, idx_worst::Int,
                               logL_worst::Float64, bound,
                               log_V_cur::Float64, n_walks::Int,
                               walk_scale::Float64,
                               ins_log_L::Vector{Float64},
                               ins_log_V::Vector{Float64},
                               scratch_cur::Vector{Float64},
                               scratch_prop::Vector{Float64},
                               scratch_noise::Vector{Float64};
                               slot::Int = 1)
    # Start from a random viable live point that isn't the one being
    # replaced (same NS rule as the RWalk kernel).
    rand_offset = rand(rng, 0:(n_live - 1))
    start_idx = 0
    @inbounds for offset in 0:(n_live - 1)
        cand = ((rand_offset + offset) % n_live) + 1
        if cand != idx_worst && live_logL[cand] > logL_worst
            start_idx = cand
            break
        end
    end
    start_idx == 0 && return (nothing, -Inf)
    @inbounds for d in 1:n_dim
        scratch_cur[d] = live_u[d, start_idx]
    end
    cur_logL = live_logL[start_idx]

    w = walk_scale / sqrt(n_dim)   # initial slice half-width
    max_expand = 20
    max_shrink = 50
    n_done = 0

    for _sweep in 1:n_walks
        # Isotropic unit direction in `scratch_noise`.
        Random.randn!(rng, scratch_noise)
        nrm = 0.0
        @inbounds for d in 1:n_dim
            nrm += scratch_noise[d]^2
        end
        nrm = sqrt(nrm)
        nrm == 0.0 && continue
        @inbounds for d in 1:n_dim
            scratch_noise[d] /= nrm
        end

        # Random initial bracket [lo, hi] straddling t = 0.
        r  = rand(rng)
        lo = -w * r
        hi =  w * (1.0 - r)

        # Step out: extend each end while it still satisfies the
        # constraint, capped at `max_expand` doublings of work.
        ex = 0
        while ex < max_expand &&
              _slice_probe!(scratch_prop, scratch_cur, scratch_noise, lo,
                            n_dim, eval_loglike, slot, log_V_cur,
                            ins_log_L, ins_log_V) > logL_worst
            lo -= w
            ex += 1
        end
        ex = 0
        while ex < max_expand &&
              _slice_probe!(scratch_prop, scratch_cur, scratch_noise, hi,
                            n_dim, eval_loglike, slot, log_V_cur,
                            ins_log_L, ins_log_V) > logL_worst
            hi += w
            ex += 1
        end

        # Shrink: sample uniformly in [lo, hi], accept first point inside
        # the constraint, otherwise contract the interval toward 0.
        accepted = false
        for _ in 1:max_shrink
            t  = lo + (hi - lo) * rand(rng)
            ll = _slice_probe!(scratch_prop, scratch_cur, scratch_noise, t,
                               n_dim, eval_loglike, slot, log_V_cur,
                               ins_log_L, ins_log_V)
            if ll > logL_worst
                @inbounds for d in 1:n_dim
                    scratch_cur[d] = scratch_prop[d]
                end
                cur_logL = ll
                accepted = true
                n_done  += 1
                break
            else
                t < 0.0 ? (lo = t) : (hi = t)
            end
        end
        accepted || break   # stuck this sweep — return what we have
    end

    n_done == 0 && return (nothing, -Inf)
    return (scratch_cur, cur_logL)
end

"""Original rejection-sampling within-bound (low-d only). Selected via
`proposal = :unif`; kept for parity / regression testing."""
function _rejection_within_bound!(rng::AbstractRNG,
                                   eval_loglike::Function, n_dim::Int,
                                   logL_worst::Float64, bound,
                                   log_V_cur::Float64,
                                   ins_log_L::Vector{Float64},
                                   ins_log_V::Vector{Float64};
                                   refit = nothing, refit_every::Int = typemax(Int))
    # A draw outside the cube costs no likelihood call, and the bootstrap-
    # expanded bound can lie mostly outside the cube while the constrained
    # region is still large, so those draws have a cap of their own. Each one
    # still counts in the INS normalisation, with L = 0.
    #
    # `refit()` refits the bound every `refit_every` likelihood calls during
    # the search, each refit with a fresh bootstrap expansion, as dynesty and
    # sample_nested do (vendored `draw_uniform`): an expansion that came out
    # far too large is otherwise drawn from until the caps run out. The bound
    # the search ends on is returned with the point.
    ncall = nout = since = 0
    while ncall < 100_000 && nout < 1_000_000
        if refit !== nothing && since ≥ refit_every
            bound = refit()
            log_V_cur = log(NestedSamplers.Bounds.volume(bound))
            since = 0
        end
        u_prop = rand(rng, bound)
        in_cube = all(0 <= u <= 1 for u in u_prop)
        if !in_cube
            push!(ins_log_L, -Inf)
            push!(ins_log_V, log_V_cur)
            nout += 1
            continue
        end
        ncall += 1
        since += 1
        logL_prop = eval_loglike(u_prop)
        push!(ins_log_L, logL_prop)
        push!(ins_log_V, log_V_cur)
        if logL_prop > logL_worst
            return (u_prop, logL_prop, bound)
        end
    end
    return (nothing, -Inf, bound)
end

_rejection_failed(iter) =
    error("no point from the bound reached the likelihood threshold at iteration " *
          "$iter after 100 000 likelihood calls (proposal = :unif): ellipsoids that " *
          "cover this constrained region are too loose for uniform draws, so use " *
          ":rwalk or :slice")

# The bound fitted to the live points; a single Ellipsoid when MultiEllipsoid's
# k-means fails on near-duplicate points.
function _fit_ins_bound(rng, BoundT, live_u, pointvol, enlarge, n_boot)
    try
        NestedSamplers.Bounds.fit(rng, BoundT, live_u;
                    pointvol = pointvol, enlarge = enlarge, bootstrap = n_boot)
    catch err
        err isa ArgumentError || rethrow(err)
        NestedSamplers.Bounds.fit(rng, NestedSamplers.Bounds.Ellipsoid, live_u;
                    pointvol = pointvol, enlarge = enlarge, bootstrap = n_boot)
    end
end


"""Systematic resampling — same low-variance pattern used by sample_pa."""
function _systematic_resample(rng::AbstractRNG, w::Vector{Float64}, n::Int)
    cdf = cumsum(w)
    u0 = rand(rng) / n
    indices = Vector{Int}(undef, n)
    j = 1
    for k in 1:n
        u = u0 + (k - 1) / n
        while j < length(cdf) && cdf[j] < u
            j += 1
        end
        indices[k] = j
    end
    return indices
end
