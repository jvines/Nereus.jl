# Dynamic Nested Sampling (Higson, Handley, Hobson, Lasenby 2019;
# Stat. Comput. 29, 891; arXiv:1704.03459).
#
# Standard NS allocates `n_live` live points uniformly across the
# likelihood range, so most of the resolution lands in the bulk where
# the prior has support but the posterior is thin. Dynamic NS targets
# the live-point allocation to where the posterior actually concentrates
# (or to where evidence integration is uncertain), giving the classic
# 10×–100× speedup on posterior estimation at fixed compute.
#
# Algorithm (simplified Higson 2019):
#  1. **Baseline run** — standard NS with `n_live_init` live points for
#     a small number of iterations, producing an initial dead-point
#     sequence (Lᵢ, log Xᵢ).
#  2. **Importance estimation** — for each iteration i compute
#     I_post(L) ∝ exp(logXᵢ + logLᵢ) (posterior-targeted importance)
#     or I_z(L) ∝ exp(logXᵢ + logLᵢ) · (logXᵢ shell width) (evidence-
#     targeted; same shape for our purposes). Smooth and locate the
#     bulk of the importance distribution.
#  3. **Dynamic batches** — pick a target likelihood range
#     [L_lo, L_hi] covering the high-importance region. Initialize
#     `n_live_batch` new live points uniformly in that range (sampled
#     from the dead-point cloud), run NS just within that range until
#     the lowest live point exits at L_hi, accumulating new dead
#     points.
#  4. **Combine** via Higson's iso-likelihood interleaving: merge all
#     dead-point sequences by L, recompute log X for each step as if
#     the union were one effective n_live(L) schedule.
#
# This implementation is a single-loop, single-batch reduction of the
# full Higson algorithm: it runs a baseline pass, identifies where the
# posterior mass concentrates, then runs *one* targeted batch to refine
# that region. Multi-batch is straightforward to add by iterating
# steps 2–3 with a running n_live(L) bookkeeping.

using NestedSamplers
using Random
using MCMCChains
using Statistics: median
using LogExpFunctions: logsumexp, logsubexp

export sample_nested_dynamic, DynamicNSResult

"""
    DynamicNSResult

# Fields
- `chains::MCMCChains.Chains` — resampled posterior, bounded space.
- `log_z::Float64` — combined log Z from baseline + targeted batch.
- `log_z_baseline::Float64`, `log_z_batch::Float64` — per-pass log Z.
- `n_iters_baseline::Int`, `n_iters_batch::Int`
- `n_evals::Int` — total likelihood evaluations.
- `target_L_lo::Float64`, `target_L_hi::Float64` — likelihood range of the batch.
"""
struct DynamicNSResult
    chains::MCMCChains.Chains
    log_z::Float64
    log_z_baseline::Float64
    log_z_batch::Float64
    n_iters_baseline::Int
    n_iters_batch::Int
    n_evals::Int
    target_L_lo::Float64
    target_L_hi::Float64
end

"""
    sample_nested_dynamic(target, data; kwargs...) -> DynamicNSResult

Two-pass dynamic NS. Baseline run with `n_live_init`, then a targeted
batch with `n_live_batch` live points concentrated in the high-
importance likelihood window.

# Keywords
- `n_live_init::Int=200` — live points for the baseline pass.
- `n_live_batch::Int=400` — live points for the targeted batch.
- `bounds::Symbol=:multi`
- `enlarge::Float64=1.25`
- `dlogz_init::Float64=2.0` — looser stopping for baseline (fast initial map).
- `dlogz_batch::Float64=0.5` — tight stopping for the targeted batch.
- `pfrac::Real=0.8` — Higson+ 2019 importance blend (the f-knob).
  `1.0` = pure posterior (parameter estimation), `0.0` = pure evidence,
  `0.8` = dynesty default (posterior-leaning). Use a low value when the
  run is for log Z.
- `maxfrac::Real=0.8` — refine the contiguous likelihood band where the
  blended importance exceeds `maxfrac × max` (dynesty's threshold rule).
- `pad::Int=1` — extra dead points to include each side of that band.
- `max_iter::Int=50_000` per pass.
- `bound_update_interval::Int=50`
- `n_walks::Union{Int,Nothing}=nothing` — Metropolis steps / slice sweeps per
  replacement. `nothing` → dimension-scaled `max(25, 2·n_dim)` (a fixed count
  under-decorrelates as n_dim grows → logZ biased low). Explicit honored.
- `walk_scale::Real=1.0` — initial proposal scale.
- `proposal::Symbol=:rwalk` — within-shell move (`:rwalk`, `:slice`,
  `:unif`); shares Nereus's INS kernels (see `sample_nested_ins`).
- `seed::Int=1`
- `verbose::Bool=false`
"""
function sample_nested_dynamic(
    target::NereusTarget,
    data::Data;
    n_live_init::Int = 200,
    n_live_batch::Int = 400,
    bounds::Symbol = :multi,
    enlarge::Real = 1.25,
    dlogz_init::Real = 2.0,
    dlogz_batch::Real = 0.5,
    pfrac::Real = 0.8,
    maxfrac::Real = 0.8,
    pad::Int = 1,
    max_iter::Int = 50_000,
    bound_update_interval::Int = 50,
    n_walks::Union{Int,Nothing} = nothing,
    walk_scale::Real = 1.0,
    proposal::Symbol = :rwalk,
    seed::Int = 1,
    verbose::Bool = false,
)
    bounds in (:multi, :single) ||
        throw(ArgumentError("bounds must be :multi or :single (got :$bounds)"))
    proposal in (:rwalk, :slice, :unif) ||
        throw(ArgumentError("proposal must be :rwalk, :slice, or :unif " *
                            "(got :$proposal)"))
    0 ≤ pfrac ≤ 1 ||
        throw(ArgumentError("pfrac must be in [0, 1] (got $pfrac)"))
    0 < maxfrac ≤ 1 ||
        throw(ArgumentError("maxfrac must be in (0, 1] (got $maxfrac)"))
    # Normalize float-valued kwargs at the boundary — JSON config sends
    # integers (e.g. `enlarge: 1`) that the strictly-Float64 _ns_inner
    # would otherwise reject.
    enlarge     = Float64(enlarge)
    dlogz_init  = Float64(dlogz_init)
    dlogz_batch = Float64(dlogz_batch)
    walk_scale  = Float64(walk_scale)
    pfrac       = Float64(pfrac)
    maxfrac     = Float64(maxfrac)
    # Dimension-scaled within-shell decorrelation (see sample_nested): a fixed
    # step count under-decorrelates as n_dim grows → logZ biased low. `nothing`
    # → max(25, 2·n_dim) (PolyChord num_repeats ∝ nDims); explicit honored.
    # Flows to both _ns_inner passes below. See project_ns_rslice_undermixing.
    n_walks = n_walks === nothing ?
              max(25, 2 * length(target.params.layout.unfrozen_idx)) : n_walks
    # === Pass 1: baseline NS ===
    verbose && @info "Dynamic NS: baseline pass (n_live=$n_live_init, dlogz=$dlogz_init)"
    base = _ns_inner(target, data;
                     n_live = n_live_init, bounds = bounds, enlarge = enlarge,
                     dlogz = dlogz_init, max_iter = max_iter,
                     bound_update_interval = bound_update_interval,
                     L_min = -Inf, L_max = +Inf,
                     n_walks = n_walks, walk_scale = walk_scale,
                     proposal = proposal,
                     seed = seed, verbose = verbose)

    # Higson+ 2019 importance allocation (dynesty `weight_function`).
    # Two importance functions over the dead-point sequence, blended by
    # `pfrac` (the f-knob):
    #   pweight[i] ∝ exp(logwt_i − log Z)        — normalized posterior
    #               weight; concentrates at the posterior bulk →
    #               minimizes parameter-estimation variance.
    #   zweight[i] ∝ (Z_tot − Z_i) / n_live      — remaining evidence
    #               beyond point i; spreads toward lower L where the
    #               evidence integral's variance lives → minimizes log Z
    #               variance.
    #   weight = pfrac·pweight + (1−pfrac)·zweight
    # `pfrac=1` → pure posterior, `pfrac=0` → pure evidence. The batch
    # then refines the contiguous likelihood band where the blended
    # weight exceeds `maxfrac × max(weight)` (dynesty's threshold rule),
    # padded by `pad` dead points each side.
    n_dead   = length(base.dead_logwt)
    logz_tot = base.log_z                      # = logsumexp(dead_logwt)

    # Posterior weight (normalized).
    log_pw  = base.dead_logwt .- logz_tot
    pweight = exp.(log_pw)
    pweight ./= sum(pweight)

    # Evidence weight: remaining evidence past each dead point. The
    # baseline pass has constant n_live, so the 1/n_live factor is a
    # constant that cancels under normalization and is omitted.
    logz_cum = similar(base.dead_logwt)
    acc = -Inf
    @inbounds for i in 1:n_dead
        acc = logsumexp((acc, base.dead_logwt[i]))
        logz_cum[i] = acc
    end
    zweight = Vector{Float64}(undef, n_dead)
    @inbounds for i in 1:n_dead
        # log(Z_tot − Z_i); the final point has zero remaining → -Inf.
        zweight[i] = logz_cum[i] ≥ logz_tot ? -Inf :
                     logsubexp(logz_tot, logz_cum[i])
    end
    zmax = maximum(zweight)
    zweight = isfinite(zmax) ? exp.(zweight .- zmax) : ones(n_dead)
    zweight ./= sum(zweight)

    # Blend and threshold.
    weight = pfrac .* pweight .+ (1 - pfrac) .* zweight
    thresh = maxfrac * maximum(weight)
    sel    = findall(>(thresh), weight)
    isempty(sel) && (sel = [argmax(weight)])
    idx_lo = max(first(sel) - pad, 1)
    idx_hi = min(last(sel)  + pad, n_dead)
    idx_hi = max(idx_hi, idx_lo + 1)
    L_lo = base.dead_logL[idx_lo]
    L_hi = idx_hi < n_dead ? base.dead_logL[idx_hi] : Inf

    verbose && @info @sprintf(
        "Dynamic NS: target window L ∈ [%.3f, %.3f] (dead indices %d–%d / %d)",
        L_lo, L_hi, idx_lo, idx_hi, n_dead)

    # === Pass 2: targeted batch ===
    # Initialize `n_live_batch` live points by drawing from the baseline
    # dead points with log L ≥ L_lo (uniformly). This seeds the batch
    # in the target region without re-evaluating the likelihood.
    elig = findall(>=(L_lo), base.dead_logL)
    rng = MersenneTwister(seed + 1)
    init_idx = rand(rng, elig, n_live_batch)
    init_u = hcat([copy(base.dead_u[i]) for i in init_idx]...)
    init_logL = base.dead_logL[init_idx]

    batch = _ns_inner(target, data;
                      n_live = n_live_batch, bounds = bounds, enlarge = enlarge,
                      dlogz = dlogz_batch, max_iter = max_iter,
                      bound_update_interval = bound_update_interval,
                      L_min = L_lo, L_max = L_hi,
                      init_u = init_u, init_logL = init_logL,
                      n_walks = n_walks, walk_scale = walk_scale,
                      proposal = proposal,
                      seed = seed + 2, verbose = verbose)

    # === Combine: Higson interleaving (simplified) ===
    # Concatenate dead-point streams and recompute log_X with an
    # effective n_live that is the sum of baseline + batch in the
    # batch's L range. For L outside the batch range, n_live = baseline
    # only.
    all_logL = vcat(base.dead_logL, batch.dead_logL)
    all_u    = vcat(base.dead_u, batch.dead_u)
    ord = sortperm(all_logL)
    all_logL = all_logL[ord]
    all_u    = all_u[ord]

    n_total = length(all_logL)
    log_X = 0.0
    log_z = -Inf
    log_wts = Vector{Float64}(undef, n_total)
    for k in 1:n_total
        in_batch_range = all_logL[k] >= L_lo && all_logL[k] <= L_hi
        n_live_eff = in_batch_range ? (n_live_init + n_live_batch) : n_live_init
        log_X_new = log_X - 1.0 / n_live_eff
        log_dV = log_X + log1p(-exp(log_X_new - log_X))
        log_wt = log_dV + all_logL[k]
        log_wts[k] = log_wt
        log_z = log_z == -Inf ? log_wt :
                max(log_z, log_wt) + log1p(exp(-abs(log_z - log_wt)))
        log_X = log_X_new
    end

    # Posterior chain construction — same n_eff-aware fallback as
    # sample_nested_ins. When n_eff < 200 the systematic resample
    # collapses to the MAP; instead take the top dead points by weight.
    log_w_shift = log_wts .- log_z
    log_w_max = maximum(log_w_shift)
    w = exp.(log_w_shift .- log_w_max)
    w ./= sum(w)
    n_eff = round(Int, 1 / sum(w .^ 2))
    if n_eff >= 200
        n_resample = n_eff
        indices = _systematic_resample(rng, w, n_resample)
    else
        perm = sortperm(w; rev = true)
        n_keep = min(max(n_eff, 200), length(w))
        indices = perm[1:n_keep]
        @warn "Dynamic NS posterior: n_eff = $(n_eff) (< 200). " *
              "Returning top-$(n_keep) dead points by weight rather " *
              "than systematic resampling. Increase n_live_init / " *
              "n_live_batch or run longer."
    end
    n_resample = length(indices)

    layout = target.params.layout
    n_dim = length(layout.unfrozen_idx)
    samples = Matrix{Float64}(undef, n_resample, n_dim)
    for (i, idx) in enumerate(indices)
        for (j, _) in enumerate(layout.unfrozen_idx)
            samples[i, j] = prior_transform(all_u[idx][j], layout.unfrozen_priors[j])
        end
    end

    param_names = Symbol.(layout.unfrozen_names)
    chains = MCMCChains.Chains(samples, param_names)

    return DynamicNSResult(
        chains,
        log_z, base.log_z, batch.log_z,
        base.n_iters, batch.n_iters, base.n_evals + batch.n_evals,
        L_lo, L_hi)
end

# Inner NS loop shared by baseline + batch passes. Accepts L_min /
# L_max so the batch can restrict to a likelihood window.
struct _NSPassResult
    dead_logL::Vector{Float64}
    dead_logwt::Vector{Float64}
    dead_u::Vector{Vector{Float64}}
    log_z::Float64
    n_iters::Int
    n_evals::Int
end

function _ns_inner(target::NereusTarget, data::Data;
                   n_live::Int, bounds::Symbol, enlarge::Float64,
                   dlogz::Float64, max_iter::Int,
                   bound_update_interval::Int,
                   L_min::Float64, L_max::Float64,
                   init_u::Union{Nothing,Matrix{Float64}} = nothing,
                   init_logL::Union{Nothing,Vector{Float64}} = nothing,
                   n_walks::Int = 25,
                   walk_scale::Float64 = 1.0,
                   proposal::Symbol = :rwalk,
                   min_X_shrinkage::Float64 = 10.0,
                   seed::Int, verbose::Bool)
    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    unfrozen_idx = layout.unfrozen_idx
    unfrozen_priors = layout.unfrozen_priors
    rng = MersenneTwister(seed)

    # Per-thread Theta — same rationale as sample_nested_ins (parallel
    # init pass + per-thread isolation for the inner phot threading
    # inside transit_log_likelihood).
    n_thr = max(Threads.nthreads(), 1)
    thread_theta = [Theta{Float64}(params) for _ in 1:n_thr]
    # Per-thread PTWorkspace → ws-aware likelihood (no per-call GC + caches).
    thread_ws = [PTWorkspace(params, params.config.max_kplanet,
                             length(params.config.noise_models);
                             n_obs = length(data.t_rv), n_phot = length(data.t_phot))
                 for _ in 1:n_thr]

    @inline function eval_loglike(u::AbstractVector{<:Real},
                                    tid::Int = 1)
        theta = thread_theta[tid]
        wb = thread_ws[tid]
        @inbounds for (j, idx) in enumerate(unfrozen_idx)
            theta.values[idx] = prior_transform(u[j], unfrozen_priors[j])
        end
        ll = rv_log_likelihood(theta, data, wb)
        isfinite(ll) || return -Inf
        ll += transit_log_likelihood(theta, data, wb)
        return ll
    end

    BoundT = bounds === :multi ? NestedSamplers.Bounds.MultiEllipsoid :
                                  NestedSamplers.Bounds.Ellipsoid

    # Init live points (threaded eval when init_u not provided)
    if init_u !== nothing
        live_u = copy(init_u)
        live_logL = copy(init_logL)
    else
        live_u = rand(rng, n_dim, n_live)
        live_logL = Vector{Float64}(undef, n_live)
        Threads.@threads :static for i in 1:n_live
            tid = Threads.threadid()
            live_logL[i] = eval_loglike(@view(live_u[:, i]), tid)
        end
    end
    n_evals = init_u === nothing ? n_live : 0

    pointvol = 1.0 / n_live
    bound = try
        NestedSamplers.Bounds.scale!(
            NestedSamplers.Bounds.fit(BoundT, live_u; pointvol=pointvol),
            enlarge)
    catch err
        err isa ArgumentError || rethrow(err)
        # Initial live points are too clustered for MultiEllipsoid
        # k-means (e.g. baseline-pass live points already convergence-
        # tight when passed to the batch pass). Fall back to a single
        # Ellipsoid for the starting bound.
        NestedSamplers.Bounds.scale!(
            NestedSamplers.Bounds.fit(
                NestedSamplers.Bounds.Ellipsoid,
                live_u; pointvol = pointvol),
            enlarge)
    end

    dead_logL    = Float64[]
    dead_logwt   = Float64[]
    dead_u       = Vector{Vector{Float64}}()
    sizehint!(dead_logL,  max_iter + n_live)
    sizehint!(dead_logwt, max_iter + n_live)
    sizehint!(dead_u,     max_iter + n_live)

    # Reused RWalk scratch — see sample_nested_ins for rationale.
    scratch_cur   = Vector{Float64}(undef, n_dim)
    scratch_prop  = Vector{Float64}(undef, n_dim)
    scratch_noise = Vector{Float64}(undef, n_dim)
    worst_save    = Vector{Float64}(undef, n_dim)
    # Throw-away INS-style accumulators (dynamic NS doesn't compute
    # INS evidence). Pre-allocate and reuse with empty!() each iter.
    _scratch_L = Float64[]
    _scratch_V = Float64[]
    sizehint!(_scratch_L, 200)
    sizehint!(_scratch_V, 200)

    log_z = -Inf
    log_X = 0.0
    iter = 0

    while iter < max_iter
        iter += 1
        idx_worst = argmin(live_logL)
        logL_worst = live_logL[idx_worst]

        # Batch terminates when worst live exits the target window
        if logL_worst >= L_max
            break
        end

        log_X_new = -iter / n_live
        log_dV = log_X + log1p(-exp(log_X_new - log_X))
        log_wt = log_dV + logL_worst
        log_z = log_z == -Inf ? log_wt :
                max(log_z, log_wt) + log1p(exp(-abs(log_z - log_wt)))
        push!(dead_logL, logL_worst)
        push!(dead_logwt, log_wt)
        @inbounds for d in 1:n_dim
            worst_save[d] = live_u[d, idx_worst]
        end
        push!(dead_u, copy(worst_save))
        log_X = log_X_new

        # Replace worst via the selected within-shell proposal (:rwalk /
        # :slice / :unif). Reuses pre-allocated scratch buffers; the INS
        # accumulators here are discarded (Dynamic NS doesn't compute
        # INS) but are emptied & reused.
        empty!(_scratch_L); empty!(_scratch_V)
        log_V_cur = log(NestedSamplers.Bounds.volume(bound))
        new_u, new_logL = if proposal === :rwalk
            _rwalk_within_shell!(live_u, live_logL, rng, eval_loglike,
                                  n_dim, n_live, idx_worst,
                                  max(logL_worst, L_min), bound,
                                  log_V_cur, n_walks, walk_scale,
                                  _scratch_L, _scratch_V,
                                  scratch_cur, scratch_prop, scratch_noise)
        elseif proposal === :slice
            _slice_within_shell!(live_u, live_logL, rng, eval_loglike,
                                  n_dim, n_live, idx_worst,
                                  max(logL_worst, L_min), bound,
                                  log_V_cur, n_walks, walk_scale,
                                  _scratch_L, _scratch_V,
                                  scratch_cur, scratch_prop, scratch_noise)
        else  # :unif
            _rejection_within_bound!(rng, eval_loglike, n_dim,
                                      max(logL_worst, L_min), bound,
                                      log_V_cur,
                                      _scratch_L, _scratch_V)
        end
        n_evals += length(_scratch_V)
        new_u === nothing && break
        @inbounds for d in 1:n_dim
            live_u[d, idx_worst] = new_u[d]
        end
        live_logL[idx_worst] = new_logL

        if iter % bound_update_interval == 0
            # MultiEllipsoid.fit calls k-means; clustering NaN's when
            # live points become near-duplicate after RWalk shrinks σ
            # in a tight shell. Fall back to single Ellipsoid in that
            # case — slightly less efficient bound geometry but never
            # fails.
            try
                bound = NestedSamplers.Bounds.scale!(
                    NestedSamplers.Bounds.fit(BoundT, live_u;
                                               pointvol = exp(log_X) / n_live),
                    enlarge)
            catch err
                err isa ArgumentError || rethrow(err)
                bound = NestedSamplers.Bounds.scale!(
                    NestedSamplers.Bounds.fit(
                        NestedSamplers.Bounds.Ellipsoid,
                        live_u;
                        pointvol = exp(log_X) / n_live),
                    enlarge)
            end
        end

        # Require both convergence AND minimum prior-volume shrinkage
        # (same fix as sample_nested_ins). Prevents premature stop at
        # startup when log_z is still climbing.
        log_z_remain = log_X + maximum(live_logL)
        dlogz_now = log_z > -Inf ?
                    log1p(exp(log_z_remain - log_z)) : Inf
        if dlogz_now < dlogz && -log_X > min_X_shrinkage
            break
        end
    end

    # Add final live points
    for i in 1:n_live
        log_wt = log_X - log(n_live) + live_logL[i]
        log_z = log_z == -Inf ? log_wt :
                max(log_z, log_wt) + log1p(exp(-abs(log_z - log_wt)))
        push!(dead_logL, live_logL[i])
        push!(dead_logwt, log_wt)
        push!(dead_u, copy(live_u[:, i]))
    end

    return _NSPassResult(dead_logL, dead_logwt, dead_u, log_z, iter, n_evals)
end
