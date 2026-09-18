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
using LinearAlgebra: det

export sample_nested_ins, INSResult

"""
    INSResult

Container for a `sample_nested_ins` run.

# Fields
- `chains::MCMCChains.Chains` — posterior samples (importance-resampled).
- `log_z_ns::Float64` — Skilling-style NS log Z (iso-likelihood shells).
- `log_z_ins::Float64` — INS log Z (importance over all evaluations).
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

Importance Nested Sampling. Standard NS posterior + log Z + an INS
log Z that uses every likelihood evaluation (incl. rejected proposals).

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
- `walk_scale::Real=1.0` — initial proposal scale.
- `proposal::Symbol=:rwalk` — within-shell move: `:rwalk` (random walk),
  `:slice` (random-direction slice), or `:unif` (rejection; low-d only).
  These are Nereus's own accumulator-aware kernels — NestedSamplers'
  proposal types can't feed the per-evaluation INS estimator, so INS
  cannot reuse them.
- `min_X_shrinkage::Real=10.0` — min log prior-volume shrinkage before
  the dlogz stop can trigger.
- `seed::Int=1`
- `verbose::Bool=false`
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

    # Per-thread Theta buffer. NS outer loop is sequential (one
    # replacement per iter) but the init pass and RWalk's downstream
    # `transit_log_likelihood` photometry threading both benefit from
    # having a Theta per Julia thread to avoid contention.
    n_thr = max(Threads.nthreads(), 1)
    thread_theta = [Theta{Float64}(params) for _ in 1:n_thr]
    # Per-thread PTWorkspace → ws-aware likelihood (preallocated buffers, no
    # per-call GC; per-planet flux cache + total phot-ll cache). See nested.jl.
    thread_ws = [PTWorkspace(params, params.config.max_kplanet,
                             length(params.config.noise_models);
                             n_obs = length(data.t_rv), n_phot = length(data.t_phot))
                 for _ in 1:n_thr]

    # Likelihood at a unit-cube point u ∈ [0,1]^n_dim
    @inline function eval_loglike(u::AbstractVector{<:Real},
                                    tid::Int = 1)
        theta = thread_theta[tid]
        wb = thread_ws[tid]
        # transform u → bounded x via per-coord prior quantile
        @inbounds for (j, idx) in enumerate(unfrozen_idx)
            ps = unfrozen_priors[j]
            theta.values[idx] = prior_transform(u[j], ps)
        end
        ll = rv_log_likelihood(theta, data, wb)
        isfinite(ll) || return -Inf
        ll += transit_log_likelihood(theta, data, wb)
        return ll
    end

    BoundT = bounds === :multi ? NestedSamplers.Bounds.MultiEllipsoid :
             bounds === :single ? NestedSamplers.Bounds.Ellipsoid :
             throw(ArgumentError("bounds must be :multi or :single"))

    # --- Initialize n_live unit-cube live points (threaded) -----------
    live_u = rand(rng, n_dim, n_live)
    live_logL = Vector{Float64}(undef, n_live)
    Threads.@threads :static for i in 1:n_live
        tid = Threads.threadid()
        live_logL[i] = eval_loglike(@view(live_u[:, i]), tid)
    end
    n_evals = n_live

    # Fit initial bound (with MultiEllipsoid fallback to single Ellipsoid
    # when k-means clustering fails on near-duplicate live points).
    pointvol = 1.0 / n_live
    bound = try
        NestedSamplers.Bounds.scale!(
            NestedSamplers.Bounds.fit(BoundT, live_u; pointvol=pointvol),
            enlarge)
    catch err
        err isa ArgumentError || rethrow(err)
        NestedSamplers.Bounds.scale!(
            NestedSamplers.Bounds.fit(
                NestedSamplers.Bounds.Ellipsoid,
                live_u; pointvol = pointvol),
            enlarge)
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

    log_z_ns = -Inf
    log_X = 0.0  # log(prior volume contained)
    iter = 0

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
                                  bound, log_V_cur, n_walks, walk_scale,
                                  ins_log_L, ins_log_V,
                                  scratch_cur, scratch_prop, scratch_noise)
        elseif proposal === :slice
            _slice_within_shell!(live_u, live_logL, rng, eval_loglike,
                                  n_dim, n_live, idx_worst, logL_worst,
                                  bound, log_V_cur, n_walks, walk_scale,
                                  ins_log_L, ins_log_V,
                                  scratch_cur, scratch_prop, scratch_noise)
        else  # :unif
            _rejection_within_bound!(rng, eval_loglike, n_dim,
                                      logL_worst, bound, log_V_cur,
                                      ins_log_L, ins_log_V)
        end
        if new_u === nothing
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
            pv = exp(log_X) / n_live
            try
                bound = NestedSamplers.Bounds.scale!(
                    NestedSamplers.Bounds.fit(BoundT, live_u; pointvol=pv),
                    enlarge)
            catch err
                err isa ArgumentError || rethrow(err)
                bound = NestedSamplers.Bounds.scale!(
                    NestedSamplers.Bounds.fit(
                        NestedSamplers.Bounds.Ellipsoid,
                        live_u; pointvol = pv),
                    enlarge)
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
        if dlogz_now < dlogz && -log_X > min_X_shrinkage
            break
        end

        verbose && iter % 500 == 0 && @info @sprintf(
            "iter %d  log Z_ns=%.3f  Δlog Z=%.4f  log_X=%.2f  n_evals=%d",
            iter, log_z_ns, dlogz_now, log_X, n_evals)
    end

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

    # --- INS log Z = log(Σ L_i · V_i) − log(N_total_evals) -----
    finite_idx = findall(isfinite, ins_log_L)
    if isempty(finite_idx)
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
            samples[i, j] = prior_transform(dead_u[idx][j], unfrozen_priors[j])
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
                          n_walks, walk_scale, ins_log_L, ins_log_V)
    -> (new_u, new_logL) or (nothing, -Inf)

Within-shell RWalk: pick a random live point as the starting state,
take `n_walks` Gaussian random-walk Metropolis steps inside the
[0,1]^n_dim unit cube subject to the constraint `L > logL_worst`.
Step size adapts toward ~50% acceptance over the chain. Each proposal
feeds the INS accumulator (we don't distinguish accepted from rejected
for the importance estimator — all evaluations contribute the same way).

Returns the final accepted state. Falls back to `(nothing, -Inf)` if no
proposal is accepted across all `n_walks` steps (extremely unlikely if
the starting state already satisfies the constraint).
"""
function _rwalk_within_shell!(live_u::AbstractMatrix{Float64},
                               live_logL::AbstractVector{Float64},
                               rng::AbstractRNG, eval_loglike::Function,
                               n_dim::Int, n_live::Int, idx_worst::Int,
                               logL_worst::Float64, bound,
                               log_V_cur::Float64, n_walks::Int,
                               walk_scale::Float64,
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
                               tid::Int = 1)
    # Start state: pick a random viable live point that's NOT the one
    # being replaced. Standard NS rule. Random offset + first-viable
    # scan avoids the `filter` allocation that the old code did per
    # call.
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

    # Initial step scale. Adapted below toward the in-band acceptance.
    σ = walk_scale / sqrt(n_dim)

    max_attempts = max(5 * n_walks, 100)
    n_accept = 0
    attempt = 0
    n_reject_streak = 0
    while n_accept < n_walks && attempt < max_attempts
        attempt += 1
        Random.randn!(rng, scratch_noise)
        in_cube = true
        @inbounds for d in 1:n_dim
            scratch_prop[d] = scratch_cur[d] + σ * scratch_noise[d]
            if scratch_prop[d] < 0.0 || scratch_prop[d] > 1.0
                in_cube = false
            end
        end
        if !in_cube
            push!(ins_log_L, -Inf)
            push!(ins_log_V, log_V_cur)
            n_reject_streak += 1
            if n_reject_streak >= 5
                σ *= 0.5
                n_reject_streak = 0
            end
            continue
        end
        logL_prop = eval_loglike(scratch_prop, tid)
        push!(ins_log_L, logL_prop)
        push!(ins_log_V, log_V_cur)
        if logL_prop > logL_worst
            @inbounds for d in 1:n_dim
                scratch_cur[d] = scratch_prop[d]
            end
            cur_logL = logL_prop
            n_accept += 1
            n_reject_streak = 0
            acc = n_accept / attempt
            # NOTE: this looks inverted against the usual acceptance-targeting
            # law (and against the vendored dynesty-style `update_scale!`), and
            # it does make per-call displacement saturate — an 80-walk chain
            # travels no further than a 10-walk one. But flipping it to the
            # "correct" direction MEASURABLY DEGRADES FITS: on an injected
            # 1-planet RV recovery (P=8 d, 8 seeds, n_live=100, dlogz=0.5) the
            # flipped version recovers 5/8 instead of 8/8, rails three seeds to
            # P ≈ 660-810 d, drops median logZ_ns from -77.4 to -94.0, and
            # widens the NS-vs-INS estimator gap from 2.0 to 8.3 nats.
            # With walk_scale=1.0 the initial σ = 1/√n_dim is a large fraction
            # of the unit cube, and the shrink-on-high-acceptance behaviour is
            # what pulls it down to a workable scale; removing that leaves the
            # walk diffusing into junk long-period modes. Do not "fix" the sign
            # without re-running that benchmark — the real suspects are the
            # bang-bang controller shape and the walk_scale default, not this
            # line alone.
            σ *= acc < 0.2 ? 1.2 : (acc > 0.7 ? 0.8 : 1.0)
        else
            n_reject_streak += 1
            if n_reject_streak >= 5
                σ *= 0.7
                n_reject_streak = 0
            end
        end
    end

    n_accept == 0 && return (nothing, -Inf)
    return (scratch_cur, cur_logL)
end

# Evaluate the constrained log-likelihood at `scratch_cur + t·dir`,
# writing the trial point into `scratch_prop`. Points outside the unit
# cube score -Inf. Every probe feeds the INS accumulator. Shared by the
# slice kernel's step-out and shrink phases.
@inline function _slice_probe!(scratch_prop::Vector{Float64},
                                scratch_cur::Vector{Float64},
                                dir::Vector{Float64}, t::Float64,
                                n_dim::Int, eval_loglike::Function, tid::Int,
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
    ll = eval_loglike(scratch_prop, tid)
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
                               tid::Int = 1)
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
                            n_dim, eval_loglike, tid, log_V_cur,
                            ins_log_L, ins_log_V) > logL_worst
            lo -= w
            ex += 1
        end
        ex = 0
        while ex < max_expand &&
              _slice_probe!(scratch_prop, scratch_cur, scratch_noise, hi,
                            n_dim, eval_loglike, tid, log_V_cur,
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
                               n_dim, eval_loglike, tid, log_V_cur,
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
                                   ins_log_V::Vector{Float64})
    for _ in 1:10_000
        u_prop = rand(rng, bound)
        in_cube = all(0 <= u <= 1 for u in u_prop)
        if !in_cube
            push!(ins_log_L, -Inf)
            push!(ins_log_V, log_V_cur)
            continue
        end
        logL_prop = eval_loglike(u_prop)
        push!(ins_log_L, logL_prop)
        push!(ins_log_V, log_V_cur)
        if logL_prop > logL_worst
            return (u_prop, logL_prop)
        end
    end
    return (nothing, -Inf)
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
