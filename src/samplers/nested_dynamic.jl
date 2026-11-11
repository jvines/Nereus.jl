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
#     [L_lo, L_hi] covering the high-importance region. Draw
#     `n_live_batch` new live points uniformly inside L > L_lo (chains of
#     the proposal from the baseline's points above L_lo), run NS from
#     there until the lowest live point exits at L_hi, accumulating new
#     dead points.
#  4. **Combine** via Higson's iso-likelihood interleaving: merge all
#     dead-point sequences by L, recompute log X for each step with the
#     number of live points of every pass alive at that L.
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
using LogExpFunctions: logsumexp, logsubexp, logaddexp

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
- `walk_scale::Real=1.0` — initial proposal scale (see `sample_nested_ins`).
- `proposal::Symbol=:rwalk` — within-shell move (`:rwalk`, `:slice`,
  `:unif`); shares Nereus's INS kernels (see `sample_nested_ins`).
- `seed::Int=1`
- `verbose::Bool=false`
- `show_progress::Bool=true` — a progress line every 10 s per pass, labelled
  `nested baseline` and `nested batch 1`, with the fields of `sample_nested`'s:
  iteration, `logz`, `dlogz`, `stage` (toward that pass's dlogz threshold; the
  prior-volume shrinkage the stop also requires is not in it) and `ncall`.
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
    show_progress::Bool = true,
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
                     seed = seed, verbose = verbose,
                     label = "nested baseline", show_progress = show_progress)

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
    # `n_live_batch` live points uniform inside L > L_lo, drawn by the
    # proposal from the baseline's points above L_lo (`_ns_inner`, seed_u).
    elig = findall(>(L_lo), base.dead_logL)
    seed_u = reduce(hcat, base.dead_u[elig])
    seed_logL = base.dead_logL[elig]
    log_X_lo = -idx_lo / n_live_init   # the baseline's prior volume above L_lo
    rng = MersenneTwister(seed + 1)

    batch = _ns_inner(target, data;
                      n_live = n_live_batch, bounds = bounds, enlarge = enlarge,
                      dlogz = dlogz_batch, max_iter = max_iter,
                      bound_update_interval = bound_update_interval,
                      L_min = L_lo, L_max = L_hi,
                      seed_u = seed_u, seed_logL = seed_logL, log_X0 = log_X_lo,
                      extra_dead = (base.dead_u,),
                      n_walks = n_walks, walk_scale = walk_scale,
                      proposal = proposal,
                      seed = seed + 2, verbose = verbose,
                      label = "nested batch 1", show_progress = show_progress)

    # === Combine ===
    all_logL, all_u, log_wts, log_z = _merge_passes(base, batch, L_lo)

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
            samples[i, j] = prior_transform(all_u[idx][j], _ns_prior(layout.unfrozen_priors[j]))
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
# `dead_n[i]` is the number of live points the pass had when it removed point
# i: `n_live` for a dead point, n_live, n_live - 1, …, 1 for the live points it
# ends with, which come last in increasing likelihood.
struct _NSPassResult
    dead_logL::Vector{Float64}
    dead_logwt::Vector{Float64}
    dead_u::Vector{Vector{Float64}}
    dead_n::Vector{Int}
    log_z::Float64
    n_iters::Int
    n_evals::Int
end

"""
    _merge_passes(base, batch, L_lo) -> (logL, u, log_wt, log_z)

The baseline pass and a batch started at `L_lo` merged into one run (Higson
et al. 2019; dynesty merges its batches so): every point in increasing
likelihood, its log weight and the log evidence. Each point leaves its pass
with `dead_n` live points in it. In the merged run the live points at a
likelihood L are those of both passes alive there: the baseline's, and the
batch's for L above L_lo, where it starts; each pass's count at L is the one
its first point at or above L was removed with. The prior volume shrinks by
n / (n + 1) at a point removed with n live points.

The merge used to count n_live_init + n_live_batch live points inside the
batch's window [L_lo, L_hi] and n_live_init elsewhere, which miscounted the
batch's live points above L_hi, removed one by one at its end, and the
baseline's own.
"""
function _merge_passes(base::_NSPassResult, batch::_NSPassResult, L_lo::Float64)
    function alive(r, birth, L)
        L > birth || return 0
        i = searchsortedfirst(r.dead_logL, L)
        return i > length(r.dead_logL) ? 0 : r.dead_n[i]
    end
    issorted(base.dead_logL) && issorted(batch.dead_logL) ||
        error("sample_nested_dynamic: a pass removed its points out of likelihood order")
    all_logL = vcat(base.dead_logL, batch.dead_logL)
    all_u    = vcat(base.dead_u, batch.dead_u)
    all_n    = vcat([n + alive(batch, L_lo, L) for (L, n) in zip(base.dead_logL, base.dead_n)],
                    [n + alive(base, -Inf, L) for (L, n) in zip(batch.dead_logL, batch.dead_n)])
    ord = sortperm(all_logL)
    all_logL = all_logL[ord]
    all_u    = all_u[ord]
    all_n    = all_n[ord]

    log_X = 0.0
    log_z = -Inf
    log_wts = Vector{Float64}(undef, length(all_logL))
    for k in eachindex(all_logL)
        log_X_new = log_X - log1p(1 / all_n[k])
        log_wt = log_X + log1p(-exp(log_X_new - log_X)) + all_logL[k]
        log_wts[k] = log_wt
        log_z = logaddexp(log_z, log_wt)
        log_X = log_X_new
    end
    return all_logL, all_u, log_wts, log_z
end

function _ns_inner(target::NereusTarget, data::Data;
                   n_live::Int, bounds::Symbol, enlarge::Float64,
                   dlogz::Float64, max_iter::Int,
                   bound_update_interval::Int,
                   L_min::Float64, L_max::Float64,
                   seed_u::Union{Nothing,Matrix{Float64}} = nothing,
                   seed_logL::Union{Nothing,Vector{Float64}} = nothing,
                   log_X0::Float64 = 0.0,   # log prior volume above L_min
                   extra_dead = (),   # earlier passes' dead points, rotated by a re-cut too
                   n_walks::Int = 25,
                   walk_scale::Float64 = 1.0,
                   proposal::Symbol = :rwalk,
                   min_X_shrinkage::Float64 = 10.0,
                   seed::Int, verbose::Bool,
                   label::String, show_progress::Bool)
    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    unfrozen_idx = layout.unfrozen_idx
    unfrozen_priors = layout.unfrozen_priors
    rng = MersenneTwister(seed)

    # Per-slot Theta — same rationale as sample_nested_ins (parallel init
    # pass + per-slot isolation for the inner phot threading inside
    # transit_log_likelihood).
    #
    # Buffers are keyed by CHUNK, never by `Threads.threadid()`: the id
    # spans every threadpool while `Threads.nthreads()` counts only the
    # default one, so `thread_theta[threadid()]` overran this vector on
    # stock Julia >= 1.12 (see src/threading.jl).
    n_slots = _nthread_chunks()
    thread_theta = [Theta{Float64}(params) for _ in 1:n_slots]
    # Per-slot PTWorkspace → ws-aware likelihood (no per-call GC + caches).
    thread_ws = [PTWorkspace(params, params.config.max_kplanet,
                             length(params.config.noise_models);
                             n_obs = length(data.t_rv), n_phot = length(data.t_phot))
                 for _ in 1:n_slots]

    @inline function eval_loglike(u::AbstractVector{<:Real},
                                    slot::Int = 1)
        theta = thread_theta[slot]
        wb = thread_ws[slot]
        @inbounds for (j, idx) in enumerate(unfrozen_idx)
            theta.values[idx] = prior_transform(u[j], _ns_prior(unfrozen_priors[j]))
        end
        ll = rv_log_likelihood(theta, data, wb)
        isfinite(ll) || return -Inf
        ll += transit_log_likelihood(theta, data, wb)
        ll += tomogram_log_likelihood(theta, data, wb)  # residual maps
        return ll
    end

    BoundT = bounds === :multi ? NestedSamplers.Bounds.MultiEllipsoid :
                                  NestedSamplers.Bounds.Ellipsoid

    # :unif draws its points from the bound, which must then cover the
    # constrained region: bootstrap-expand it, as sample_nested does
    # (`Bounds.bootstrap_expansion`). The walks take only their shape from it.
    n_boot = proposal === :unif ? 5 : 0

    # Reused RWalk scratch — see sample_nested_ins for rationale.
    scratch_cur   = Vector{Float64}(undef, n_dim)
    scratch_prop  = Vector{Float64}(undef, n_dim)
    scratch_noise = Vector{Float64}(undef, n_dim)
    worst_save    = Vector{Float64}(undef, n_dim)
    # Throw-away INS-style accumulators (dynamic NS doesn't compute
    # INS evidence). Pre-allocate and reuse with empty!() each iter;
    # `length(_scratch_V)` after a draw is its likelihood calls.
    _scratch_L = Float64[]
    _scratch_V = Float64[]
    sizehint!(_scratch_L, 200)
    sizehint!(_scratch_V, 200)

    # the random walk's step, fixed within each chain (sample_nested_ins)
    walk_scale_now = Ref(walk_scale)

    # A new point inside L > `Lthr` from the proposal, its chain started at a
    # point of `pool_u` above `Lthr` other than column `skip`. Returns the
    # point (a copy; `nothing` if no point of the pool is above `Lthr`), its
    # log L and the bound, which :unif refits as it searches.
    function draw(pool_u, pool_logL, skip, Lthr, bnd, log_X)
        empty!(_scratch_L); empty!(_scratch_V)
        log_V_cur = log(NestedSamplers.Bounds.volume(bnd))
        n_pool = size(pool_u, 2)
        u, l = if proposal === :rwalk
            _rwalk_within_shell!(pool_u, pool_logL, rng, eval_loglike,
                                  n_dim, n_pool, skip, Lthr, bnd,
                                  log_V_cur, n_walks, walk_scale_now,
                                  _scratch_L, _scratch_V,
                                  scratch_cur, scratch_prop, scratch_noise)
        elseif proposal === :slice
            _slice_within_shell!(pool_u, pool_logL, rng, eval_loglike,
                                  n_dim, n_pool, skip, Lthr, bnd,
                                  log_V_cur, n_walks, walk_scale,
                                  _scratch_L, _scratch_V,
                                  scratch_cur, scratch_prop, scratch_noise)
        else  # :unif
            u_r, l_r, bnd = _rejection_within_bound!(rng, eval_loglike, n_dim,
                                      Lthr, bnd, log_V_cur,
                                      _scratch_L, _scratch_V;
                                      refit = let pv = exp(log_X) / n_pool
                                          () -> _fit_ins_bound(rng, BoundT, pool_u, pv,
                                                               enlarge, n_boot)
                                      end,
                                      refit_every = round(Int, 1.5 * n_pool))
            (u_r, l_r)
        end
        return (u === nothing ? nothing : copy(u)), l, bnd
    end

    # Init live points (threaded eval when there is no seed pool)
    n_evals = 0
    if seed_u !== nothing
        # A batch's live points must be uniform inside L > L_min, as a run's
        # live points are when its threshold reaches L_min. The earlier
        # pass's points above L_min are not: its dead points are spread
        # evenly in log X, not in X, so they crowd toward high L, and drawing
        # the batch's live points from them, with repeats, gave a batch whose
        # lowest live point could already be above L_max. Each live point is
        # drawn as a new point is (dynesty seeds a batch so): a chain of the
        # proposal from one of those points, inside L > L_min.
        pool_bound = _fit_ins_bound(rng, BoundT, seed_u, exp(log_X0) / size(seed_u, 2),
                                    enlarge, n_boot)
        live_u = Matrix{Float64}(undef, n_dim, n_live)
        live_logL = Vector{Float64}(undef, n_live)
        for i in 1:n_live
            u, l, pool_bound = draw(seed_u, seed_logL, 0, L_min, pool_bound, log_X0)
            n_evals += length(_scratch_V)
            u === nothing && proposal === :unif && _rejection_failed(0)
            u === nothing && throw(ArgumentError("no point above L_min to seed the batch from"))
            live_u[:, i] = u
            live_logL[i] = l
        end
    else
        live_u = rand(rng, n_dim, n_live)
        live_logL = Vector{Float64}(undef, n_live)
        # Buffers are keyed by CHUNK (see src/threading.jl), so `slot` is
        # a plain loop counter, in bounds by construction regardless of
        # thread count. `:static` is kept for the 1:1 chunk-to-thread
        # mapping.
        init_chunks = _chunk_ranges(n_live, n_slots)
        Threads.@threads :static for slot in 1:length(init_chunks)
            for i in init_chunks[slot]
                live_logL[i] = eval_loglike(@view(live_u[:, i]), slot)
            end
        end
        n_evals += n_live
    end

    pointvol = exp(log_X0) / n_live
    bound = try
        NestedSamplers.Bounds.fit(rng, BoundT, live_u;
                    pointvol = pointvol, enlarge = enlarge, bootstrap = n_boot)
    catch err
        err isa ArgumentError || rethrow(err)
        # Initial live points are too clustered for MultiEllipsoid
        # k-means (e.g. baseline-pass live points already convergence-
        # tight when passed to the batch pass). Fall back to a single
        # Ellipsoid for the starting bound.
        NestedSamplers.Bounds.fit(rng, NestedSamplers.Bounds.Ellipsoid, live_u;
                    pointvol = pointvol, enlarge = enlarge, bootstrap = n_boot)
    end

    dead_logL    = Float64[]
    dead_logwt   = Float64[]
    dead_u       = Vector{Vector{Float64}}()
    sizehint!(dead_logL,  max_iter + n_live)
    sizehint!(dead_logwt, max_iter + n_live)
    sizehint!(dead_u,     max_iter + n_live)

    log_z = -Inf
    log_X = log_X0
    iter = 0
    pb = ProgressBar(label; total = 0, print_every = 10.0, enabled = show_progress)
    stage = _NestedStage(dlogz)

    while iter < max_iter
        iter += 1
        idx_worst = argmin(live_logL)
        logL_worst = live_logL[idx_worst]

        # Batch terminates when worst live exits the target window
        if logL_worst >= L_max
            break
        end

        log_X_new = log_X0 - iter / n_live
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
        # :slice / :unif).
        new_u, new_logL, bound = draw(live_u, live_logL, idx_worst,
                                      max(logL_worst, L_min), bound, log_X)
        n_evals += length(_scratch_V)
        # stopping would return a truncated run's evidence (sample_nested_ins)
        new_u === nothing && proposal === :unif && _rejection_failed(iter)
        new_u === nothing && break
        @inbounds for d in 1:n_dim
            live_u[d, idx_worst] = new_u[d]
        end
        live_logL[idx_worst] = new_logL

        if iter % bound_update_interval == 0
            # Seams of the circular angles moved off the live points first
            # (`_ns_recut!`, sample_nested_ins): the refit is in the new chart.
            _ns_recut!(live_u, (dead_u, extra_dead...), params;
                       transforms = target.transform === nothing ? () : (target.transform,))
            # MultiEllipsoid.fit calls k-means; clustering NaN's when
            # live points become near-duplicate after RWalk shrinks σ
            # in a tight shell. Fall back to single Ellipsoid in that
            # case — slightly less efficient bound geometry but never
            # fails.
            try
                bound = NestedSamplers.Bounds.fit(rng, BoundT, live_u;
                    pointvol = exp(log_X) / n_live, enlarge = enlarge, bootstrap = n_boot)
            catch err
                err isa ArgumentError || rethrow(err)
                bound = NestedSamplers.Bounds.fit(rng, NestedSamplers.Bounds.Ellipsoid, live_u;
                    pointvol = exp(log_X) / n_live, enlarge = enlarge, bootstrap = n_boot)
            end
        end

        # Require both convergence AND minimum prior-volume shrinkage
        # (same fix as sample_nested_ins). Prevents premature stop at
        # startup when log_z is still climbing.
        log_z_remain = log_X + maximum(live_logL)
        dlogz_now = log_z > -Inf ?
                    log1p(exp(log_z_remain - log_z)) : Inf
        done = dlogz_now < dlogz && -log_X > min_X_shrinkage
        # The stopping iteration prints whatever the throttle says, so the
        # pass's last line is its final state.
        pb.enabled && update!(pb; n_done = iter, force = done,
                              fields = _nested_fields(stage, log_z, dlogz_now, n_evals))
        done && break
    end
    finish!(pb)

    # Add final live points, in increasing likelihood (see `_NSPassResult`)
    dead_n = fill(n_live, length(dead_logL))
    for (j, i) in enumerate(sortperm(live_logL))
        log_wt = log_X - log(n_live) + live_logL[i]
        log_z = log_z == -Inf ? log_wt :
                max(log_z, log_wt) + log1p(exp(-abs(log_z - log_wt)))
        push!(dead_logL, live_logL[i])
        push!(dead_logwt, log_wt)
        push!(dead_u, copy(live_u[:, i]))
        push!(dead_n, n_live - j + 1)
    end

    return _NSPassResult(dead_logL, dead_logwt, dead_u, dead_n, log_z, iter, n_evals)
end
