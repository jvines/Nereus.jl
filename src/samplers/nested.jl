# Nested sampling via NestedSamplers.jl (ellipsoidal decomposition).
#
# Provides proper evidence estimates comparable to dynesty/MultiNest.
# Uses the prior_transform from Nereus's prior system to map
# [0,1]^n → bounded parameter space.

using NestedSamplers
using Random
using MCMCChains
using LogExpFunctions: logaddexp

"""
    sample_nested(target::NereusTarget, data::Data; kwargs...) -> (chains, log_evidence)

Ellipsoidal nested sampling via NestedSamplers.jl. Returns posterior
samples as `MCMCChains.Chains` and a log-evidence estimate.

The target must use `unconstrained=false` (bounded space).

# Keywords
- `n_live::Int=1500` — number of live points. The default is deliberately
  large: a curved (e,ω)/(K,e) RV ridge or a multi-planet decomposition
  needs enough live points for the MultiEllipsoid bound to resolve the
  mode and for the importance weights to carry a healthy effective sample
  size. At `n_live=400` the 2-planet decomposition silently mis-resolved
  (K1/K2 swapped or collapsed) — a confident-wrong posterior. See the
  proposal note below; the two changes are coupled.
- `bounds::Symbol=:multi` — bounding method (`:multi`, `:single`, `:none`)
- `proposal::Symbol=:rslice` — within-bound proposal. One of
  `:rwalk` (random walk), `:rstagger` (staggered random walk; the same walk
  as `:rwalk` now), `:slice` (slices along the bounding ellipsoid's axes),
  `:rslice` (random-direction slice), `:hslice` (Hamiltonian slice) or
  `:unif` (uniform draws from the bound). `:unif` is for low dimension only:
  its bounds are expanded as dynesty expands them (bootstrap) until they
  cover the constrained region, and on a curved posterior in more than a few
  dimensions they get so loose that a new point takes more than 100 000
  likelihood calls, when the run stops with an error. It does on 7- and
  10-parameter RV and RM fits; on a 2-parameter gravity-darkening fit it
  matches pt_emcee. Before the expansion its bounds missed 12–22% of an RM
  fit's constrained region and log Z came out 2 nats high.
  DEFAULT IS `:rslice`, NOT `:rwalk`. RWalk takes isotropic steps scaled
  by a single scalar inside the bounding ellipsoid; on the correlated RV
  ridges it cannot traverse the correlation, its acceptance collapses, the
  scale shrinks toward the "stuck" floor, and the walk localizes — yielding
  TIGHT-BUT-WRONG credible intervals (e.g. e=0.47 reported with a tight CI
  when truth is 0.55; K1/K2 mis-decomposed in the 2-planet case, surviving
  even n_live=3000). `:rslice` slices along random directions through the
  ellipsoid axes, which track the local correlation, so it follows the
  ridge and recovers truth on all three validation generators. `:rwalk`
  remains available for diagnostics but is NOT recommended for RV fits.
  (Those failures were measured while every step of a walk or slice chain
  restarted from the live point it copied, fixed since. With the fix and its
  default chain, `:rwalk`'s posterior and log Z match pt_emcee's on RM
  obliquity and gravity-darkening fits; on an eccentric RV fit its log Z is
  0.25 high, against `:rslice`'s 0.06.)
- `walk_scale::Real=1.0` — initial proposal scale for the walk / slice
  kernels (passed to NestedSamplers as `scale`).
- `n_walks::Union{Int,Nothing}=nothing` — walk steps per replacement for
  `:rwalk` / `:rstagger`. `nothing` → `max(25, 10·n_dim)`.
- `slices::Union{Int,Nothing}=nothing` — slices per replacement. `nothing` →
  `max(5, 5·n_dim)` for `:rslice`, `max(5, 2·n_dim)` for `:hslice` and 5
  for `:slice`, each of whose slices sweeps all n_dim axes of the bound.
  Too short a chain leaves each new point near the live point it started
  from and puts log Z high; how the defaults were chosen is noted where they
  are set. An explicit `n_walks` or `slices` is honored verbatim.
- `enlarge::Real=1.25` — volume factor by which the bounding ellipsoid(s)
  are enlarged after fitting to the live set. For `:unif`, which draws from
  the bound, the factor is the larger of this and the bootstrap expansion.
  The walk and slice proposals take only their directions and scale from
  the bound and move by the likelihood constraint itself, so a bound that
  does not conform to a curved (e.g. eccentric-RV) ridge does not bias
  their log Z: `:rslice` at its default came within 0.06 nats of pt_emcee's
  on an eccentric RV fit. (An earlier note here called that bias
  structural; it was measured while every chain step restarted from the
  copied live point.)
- `dlogz::Real=0.1` — stopping criterion (remaining evidence). Tighter than
  the canonical 0.5 because a loose tolerance let a run terminate while the
  live set was still confined to a wrong sub-mode (premature convergence at
  a much lower logZ). 0.1 forces enough extra iterations for the bound to
  re-expand and find the true mode's bulk.
- `seed::Int=1`
- `show_progress::Bool=true` — a progress line every 10 s: iteration, `logz`,
  the remaining evidence `dlogz` the run stops on, `stage` (how far log dlogz
  has come toward the threshold, in percent) and `ncall`. There is no total, so
  no ETA, and the stage is not a fraction of the run time.
- `seam_pilot::Bool=true` — with full-circle or polar angles in the model,
  first run a small nested pilot (`sample_nested_ins`, `max(50, n_live ÷ 10)`
  live points, `dlogz = 1`) whose live points place each angle's seam in its
  emptiest arc (a polar angle's, of the live points and their mirror images).
  NestedSamplers.jl owns its loop, so the seam cannot move during the run, and
  a mode across it would be two pieces whose shares of the live points drift
  (src/circular.jl).
"""
function sample_nested(
    target::NereusTarget,
    data::Data;
    n_live::Int = 1500,
    bounds::Symbol = :multi,
    proposal::Symbol = :rslice,
    walk_scale::Real = 1.0,
    n_walks::Union{Int,Nothing} = nothing,
    slices::Union{Int,Nothing} = nothing,
    enlarge::Real = 1.25,
    dlogz::Real = 0.1,
    seed::Int = 1,
    parallel::Bool = !_is_under_juliacall(),
    batch_size::Int = 0,   # 0 → auto (nthreads when parallel, else 1)
    show_progress::Bool = true,
    seam_pilot::Bool = true,
)
    bounds in (:multi, :single, :none, :mlfriends) ||
        throw(ArgumentError("bounds must be :multi, :single, :none, or " *
                            ":mlfriends (got :$bounds)"))
    proposal in (:rwalk, :rstagger, :slice, :rslice, :unif, :hslice) ||
        throw(ArgumentError("proposal must be :rwalk, :rstagger, :slice, " *
                            ":rslice, :unif, or :hslice (got :$proposal)"))
    # MLFriends pairs with EITHER the Rejection proposal (:unif — its `rand`
    # draws uniformly from the union-of-balls region; tight but rejection is
    # intractable in high d) OR the slice kernels (:rslice/:slice), which use
    # `axes(MLFriends)` = the FRIENDS-radius metric as the slice scale — much
    # finer than the bounding-ellipsoid extent, the right scale for a thin
    # curved ridge. The random-walk kernels assume ellipsoidal axes differently,
    # so redirect them to :rslice.
    if bounds === :mlfriends && proposal in (:rwalk, :rstagger)
        @info "sample_nested: bounds=:mlfriends does not pair with :$proposal " *
              "(random-walk axes semantics); using :rslice instead"
        proposal = :rslice
    end
    dlogz      = Float64(dlogz)   # JSON config may deliver an Int
    walk_scale = Float64(walk_scale)
    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    rng = MersenneTwister(seed)

    # --- Decorrelation defaults ---------------------------------------------
    # A new live point is the end of a chain of `slices` slices or `n_walks`
    # walk steps from a live point inside the constraint. Too short a chain
    # leaves it near that start: the live points descend from too few
    # parents, each iteration shrinks the prior volume by more than the
    # 1/n_live NS credits, and log Z comes out HIGH. A random-direction slice
    # or a walk step moves the point along about one direction, so the count
    # grows with n_dim (PolyChord's default is 5·nDims repeats, dynesty's
    # rslice 3 + ndim). Each `:slice` slice sweeps all n_dim axes of the
    # bound, so its count is not scaled.
    #
    # Each default is the smallest multiple of n_dim whose log Z, averaged
    # over 4-9 seeds, came within the NS error (0.17-0.22) of pt_emcee's on
    # four fits: an RM obliquity fit at λ = 0° and 180° (10 parameters), an
    # eccentric RV fit (7) and a gravity-darkening fit (2), with 500 live
    # points (300 for the last). Seed-mean log Z minus pt_emcee's:
    #
    #   :rslice 2·n_dim   -0.05  +0.35  +0.07  -0.01   (RM 0°, RM 180°, RV, GD)
    #           3·n_dim   +0.07  +0.23  -0.00  -0.08
    #           5·n_dim   +0.02  +0.13  +0.06  +0.02   ← default
    #   :slice  5         +0.13  +0.09  -0.01  -0.04   ← default
    #   :hslice 2·n_dim   +0.13  +0.06  +0.06  -0.08   ← default
    #   :rwalk  5·n_dim   +0.26  -0.06  +0.26  -0.21
    #           10·n_dim  +0.07  +0.04  +0.18  -0.04   ← default
    #
    # `:rstagger` is the same walk as `:rwalk`. The walk's RV value is the
    # largest of the defaults; with the walks run one at a time
    # (`parallel = false`) it was -0.02 over six seeds, a difference of
    # 1.5 standard errors that is not resolved. The floors keep a
    # low-dimensional fit's chains from getting too short.
    #
    # The count only matters because the steps form one chain, each starting
    # where the last ended (vendor/NestedSamplers/VENDORED.md). The fork used
    # to start every step from the copied live point, so the new point was one
    # step from it whatever the count: the live set descended from ever fewer
    # ancestors and the posterior came out too narrow and off-centre (an RM
    # obliquity fit gave λ sd 1.6–2.1° against pt_emcee's 2.1–2.5°, b's sd at
    # half of pt_emcee's, the jitters up to 1.3 σ high). An earlier note here
    # blamed the ellipsoid bound for an evidence bias on curved RV ridges that
    # more slices did not remove; it was measured before the fix, when more
    # slices could not help, and a slice takes only its direction from the
    # bound.
    slices_resolved = slices !== nothing ? slices :
                      proposal === :slice  ? 5 :
                      proposal === :hslice ? max(5, 2 * n_dim) :
                                             max(5, 5 * n_dim)
    n_walks_resolved = n_walks !== nothing ? n_walks : max(25, 10 * n_dim)
    if slices === nothing || n_walks === nothing
        @info "sample_nested: dimension-scaled decorrelation (n_dim=$n_dim, " *
              "proposal=:$proposal) → slices=$slices_resolved " *
              "n_walks=$n_walks_resolved enlarge=$(Float64(enlarge))"
    end

    # Build prior distributions vector for NestedModel.
    # NestedSamplers accepts a vector of univariate distributions that
    # define the prior transform (unit cube → parameter space).
    #
    # Angles (src/circular.jl): each full-circle or polar angle's seam is
    # placed, before the run, from a small nested pilot whose own loop re-cuts
    # it at every bound refit.
    if seam_pilot && !isempty(seam_indices(params))
        sample_nested_ins(target, data; n_live = max(50, n_live ÷ 10), dlogz = 1.0,
                          seed = seed + 7919, show_progress = false)
    end
    prior_dists = [_ns_prior(layout.unfrozen_priors[j]).dist for j in 1:n_dim]

    # Log-likelihood function: maps bounded parameter vector → log L.
    # NestedSamplers draws from the prior internally via prior_dists,
    # so we must NOT evaluate log_prior here — only the data likelihood.
    #
    # PER-THREAD Theta buffers: batch/parallel NS (`sample_parallel`) calls
    # `loglike` concurrently from K constrained walks under `@threads :static`,
    # so a single shared buffer would race. The scheduling is NestedSamplers',
    # not ours, so these buffers cannot be keyed by a chunk counter the way the
    # loops we own are — they have to stay keyed by `threadid()`. That id spans
    # every threadpool, so the array is sized with `_nthread_slots()`
    # (= maxthreadid()); `Threads.nthreads()` counts the default pool only and
    # is NOT an upper bound on the id (on stock Julia ≥1.12 it is 1 while the
    # work runs on thread 2 — the reported BoundsError). Under `:static` each
    # thread runs one task, so one buffer per thread still removes the race and
    # the per-eval Theta allocation (NS calls loglike ~n_live × n_iter × walks =
    # millions).
    #
    # The slots are filled LAZILY: maxthreadid() counts GC and foreign threads
    # too (32 against nthreads()=16 on `-t auto`), and a PTWorkspace is a large
    # allocation to make eagerly for a thread that will never call the
    # likelihood. Each thread builds its own on first use; the nothing-check is
    # a union split on the hot path, not an allocation.
    n_slots    = _nthread_slots()
    theta_bufs = Vector{Union{Nothing,Theta{Float64}}}(nothing, n_slots)
    # Per-thread PTWorkspace so the ws-aware likelihood path is used: it reuses
    # preallocated planet/cadence buffers (no per-call Vector allocs → no GC
    # thrash over the millions of NS evals) AND the per-planet flux cache + the
    # total phot-ll cache. Same proven path as PT/rjmcmc.
    n_noise_ws = length(params.config.noise_models)
    ws_bufs    = Vector{Union{Nothing,PTWorkspace}}(nothing, n_slots)
    _new_ws()  = PTWorkspace(params, params.config.max_kplanet, n_noise_ws;
                             n_obs = length(data.t_rv), n_phot = length(data.t_phot))
    function loglike(x)
        tid = Threads.threadid()
        # A thread adopted after the buffers were sized — juliacall calling in
        # on a fresh Python thread raises maxthreadid() — carries an id past
        # the end of the arrays. Hand it an uncached buffer instead of throwing.
        if tid <= n_slots
            t  = theta_bufs[tid]
            tb = t === nothing ? (theta_bufs[tid] = Theta{Float64}(params)) : t
            w  = ws_bufs[tid]
            wb = w === nothing ? (ws_bufs[tid] = _new_ws()) : w
        else
            tb = Theta{Float64}(params)
            wb = _new_ws()
        end
        @inbounds for (j, idx) in enumerate(layout.unfrozen_idx)
            tb.values[idx] = x[j]
        end
        ll = rv_log_likelihood(tb, data, wb)
        isfinite(ll) || return -1e300
        ll += transit_log_likelihood(tb, data, wb)
        ll += tomogram_log_likelihood(tb, data, wb) # residual maps
        isfinite(ll) || return -1e300
        return ll
    end

    model = NestedModel(loglike, prior_dists)

    # Select bounding (validated above).
    bounds_type = bounds === :multi     ? NestedSamplers.Bounds.MultiEllipsoid :
                  bounds === :single    ? NestedSamplers.Bounds.Ellipsoid :
                  bounds === :mlfriends ? NestedSamplers.Bounds.MLFriends :
                                          NestedSamplers.Bounds.NoBounds

    # Select proposal (validated above), threading the tuning knobs
    # through to the matching NestedSamplers kernel.
    proposal_type = if proposal === :rwalk
        NestedSamplers.Proposals.RWalk(; ratio=0.5, walks=n_walks_resolved, scale=walk_scale)
    elseif proposal === :rstagger
        NestedSamplers.Proposals.RStagger(; ratio=0.5, walks=n_walks_resolved, scale=walk_scale)
    elseif proposal === :slice
        NestedSamplers.Proposals.Slice(; slices=slices_resolved, scale=walk_scale)
    elseif proposal === :rslice
        NestedSamplers.Proposals.RSlice(; slices=slices_resolved, scale=walk_scale)
    elseif proposal === :hslice
        # Hamiltonian (reflective) slice: `slices` trajectories per new point,
        # each reflecting off the likelihood constraint about the gradient where
        # a step lands outside it (vendor/NestedSamplers Proposals.HSlice).
        NestedSamplers.Proposals.HSlice(; slices=slices_resolved, scale=walk_scale)
    else  # :unif
        NestedSamplers.Proposals.Rejection()
    end

    sampler = Nested(n_dim, n_live;
        bounds = bounds_type,
        proposal = proposal_type,
        enlarge = Float64(enlarge),
    )

    # Run — returns (Chains, state_NamedTuple).
    # `progress=false` is mandatory: NestedSamplers.jl's `nested_isdone`
    # does `if progress`, but recent AbstractMCMC versions pass a
    # `CreateNewProgressBar{String}` sentinel struct (not a Bool) by
    # default, blowing up with `non-boolean used in boolean context`.
    # The fork's own printouts are `\r` lines in another format, so progress
    # comes from Nereus's bar instead, advanced by a callback that both the
    # serial and the batch path call once per iteration.
    # Batch/parallel NS when threading is safe (pure-Julia daemon — NOT
    # juliacall, where Julia threads can deadlock under the Python GIL). K
    # lowest-L points are replaced per iteration via K concurrent constrained
    # walks; correctness verified (matches serial log Z) in the fork. K=1 is
    # the serial sampler. NOTE: if the model likelihood itself threads (the
    # transit likelihood does), outer batch threading would nest — for
    # transit-heavy fits pass `parallel=false`.
    K = batch_size > 0 ? batch_size :
        (parallel ? max(1, Threads.nthreads()) : 1)

    # SLICE-FAMILY PROPOSALS RUN SERIALLY. The batch/parallel path
    # (`sample_parallel`) draws each replacement through `_draw_replacement`,
    # whose Ellipsoid bounds fallback drives Slice/RSlice into a collapsed
    # slice window far more often than the serial single-point loop; the
    # fork then throws "Slice sampling appears to be stuck" from inside a
    # worker task, surfacing as an opaque `TaskFailedException` that aborts
    # the whole fit. The serial single-point path does not hit this (verified
    # crash-free across all validation generators × seeds), so for the
    # slice kernels we force K=1. The batch parallelism is kept for the
    # random-walk kernels, which do not collapse the same way. (Slice
    # proposals are themselves expensive sequential chains, so the lost
    # batch parallelism costs little.)
    slice_family = proposal in (:slice, :rslice, :hslice)
    # MLFriends also runs serially: the parallel `_draw_replacement` refits the
    # bound on every "point-not-in-region" miss, and a freshly-replaced live
    # point is often outside the stale friends region → repeated (even if now
    # cheap) refits. The serial single-point loop triggers far fewer.
    use_parallel = parallel && K > 1 && !slice_family && bounds !== :mlfriends
    # Run serially by re-seeding so the serial path is reproducible whether
    # we reach it directly or via the parallel-failure fallback below.
    run_serial(pb) = NestedSamplers.sample(MersenneTwister(seed), model, sampler;
                                            dlogz=dlogz, progress=false,
                                            callback=_nested_progress(pb, dlogz))
    pb = ProgressBar("nested"; total = 0, print_every = 10.0, enabled = show_progress)
    ns_chains, ns_state = if use_parallel
        # Belt-and-suspenders: even the random-walk batch path can throw from
        # inside a worker task (a collapsed-scale "stuck" walk). A failed
        # PARALLEL run must NOT take the sampler down — catch it, warn LOUD,
        # and re-run SERIALLY (the single-point path is more robust). A real
        # bug would then resurface in the serial run and propagate normally.
        try
            NestedSamplers.sample_parallel(rng, model, sampler;
                                            batch_size=K, parallel=true,
                                            dlogz=dlogz, progress=false,
                                            callback=_nested_progress(pb, dlogz))
        catch err
            finish!(pb)
            @warn "Parallel nested sampling failed (likely a stuck proposal " *
                  "inside a worker task); falling back to the SERIAL sampler. " *
                  "This is slower but more robust." exception = (err, catch_backtrace())
            # The serial run starts again from iteration 1: a bar of its own.
            pb = ProgressBar("nested serial"; total = 0, print_every = 10.0,
                             enabled = show_progress)
            run_serial(pb)
        end
    else
        run_serial(pb)
    end
    finish!(pb)
    log_ev = ns_state.logz

    # NestedSamplers returns weighted samples. Resample to an unweighted
    # posterior with i.i.d. multinomial resampling on the importance weights
    # (see _multinomial_resample below for why NOT systematic resampling).
    ns_params = names(ns_chains, :parameters)
    weights = vec(Array(ns_chains[:weights]))
    n_raw = length(weights)

    # Extract raw parameter matrix
    raw = Matrix{Float64}(undef, n_raw, n_dim)
    for (j, sym) in enumerate(ns_params[1:n_dim])
        raw[:, j] = Array(ns_chains[sym])
    end

    # --- Posterior chain construction (i.i.d. multinomial resampling) ---
    # The Kish effective sample size 1/Σwᵢ² is the HONEST count of
    # independent posterior draws the weighted dead points carry. We must
    # NOT inflate beyond it: returning more rows than n_eff just clones
    # high-weight points and fabricates precision.
    #
    # Resample I.I.D. MULTINOMIAL, not systematic. Systematic (low-variance)
    # resampling walks a single CDF offset and emits a *sorted, correlated*
    # index sequence — the resulting chain looks maximally autocorrelated,
    # so MCMCChains.ess / assess_fit report ESS ~ tens even when n_eff is
    # ~10⁴. That made the convergence guard fire on EVERY nested run
    # (correct ones included), carrying no signal. Independent multinomial
    # draws give an exchangeable chain whose MCMC-ESS ≈ n_eff, restoring
    # ESS as a real discriminator between a healthy fit and a thin /
    # wrong-mode one.
    n_eff = round(Int, 1.0 / sum(weights .^ 2))

    if n_eff >= 200
        n_resample = n_eff
        indices = _multinomial_resample(rng, weights, n_resample)
    else
        # Thin posterior (peaked weights or a collapsed/wrong-mode run):
        # return the top-weight UNIQUE dead points rather than a degenerate
        # clone-the-MAP chain, and WARN LOUD. A low n_eff here is the honest
        # signal that this run must not be trusted as a tight posterior —
        # the same convention as sample_nested_ins / sample_nested_dynamic.
        perm = sortperm(weights; rev = true)
        n_keep = min(max(n_eff, 200), n_raw)
        indices = perm[1:n_keep]
        @warn "Nested posterior: n_eff = $(n_eff) (< 200). Returning " *
              "top-$(n_keep) dead points by weight rather than i.i.d. " *
              "resampling — the posterior is thin (peaked weights or a " *
              "collapsed run). Do NOT trust a tight credible interval from " *
              "this fit; increase n_live, tighten dlogz, or inspect the " *
              "fit-health report. (proposal=$proposal, n_live=$n_live)"
    end
    n_resample = length(indices)

    param_names = Symbol.(layout.unfrozen_names)
    mat = Matrix{Float64}(undef, n_resample, n_dim)
    for (i, idx) in enumerate(indices)
        mat[i, :] = raw[idx, :]
    end

    chains = MCMCChains.Chains(mat, param_names)
    return chains, log_ev
end


# Remaining evidence Δlog Z = log(Z + L_max·X) − log Z, the quantity the fork's
# `nested_isdone` and `sample_parallel` stop on.
_nested_dlogz(state) =
    logaddexp(state.logz, maximum(state.logl) + state.logvol) - state.logz

# A stage measure for the bar, 0 → 1: how far log Δ has come from its first
# usable value toward the stopping threshold `target`. Usable means finite, with
# log Z above the -1e300 a failed likelihood returns: the first dead points are
# often such failures, and Δ ≈ 1e300 there would put the start where any real Δ
# reads as nearly done. It is not the fraction of the run's time (Δ falls fast
# early and slowly late, hence no ETA), and it falls back when the live set
# reaches a higher-likelihood region, since Δ grows again then.
mutable struct _NestedStage
    target::Float64
    log_start::Float64          # NaN until the first usable Δ
end
_NestedStage(target::Real) = _NestedStage(Float64(target), NaN)

function _stage!(s::_NestedStage, logz::Real, dlogz::Real)
    (isfinite(dlogz) && logz > -1e299) || return 0.0
    ld = log(max(dlogz, floatmin(Float64)))
    isnan(s.log_start) && (s.log_start = ld)
    span = s.log_start - log(s.target)
    span > 0 || return 1.0      # it started at or below the threshold
    return clamp((s.log_start - ld) / span, 0.0, 1.0)
end

# The status fields of every nested sampler's bar, in one order. The stage is
# floored, so 100% means the dlogz criterion is met.
_nested_fields(s::_NestedStage, logz, dlogz, ncall) =
    (:logz => logz, :dlogz => dlogz,
     :stage => @sprintf("%d%%", floor(Int, 100 * _stage!(s, logz, dlogz))),
     :ncall => ncall)

# The callback `sample_nested` hands to AbstractMCMC (serial path) and to the
# fork's `sample_parallel` (batch path): one bar update per iteration. `target`
# is the run's dlogz threshold. Both paths stop right after the iteration whose
# Δ ≤ target, so that one prints whatever the throttle says: the last line is
# the run's final state.
function _nested_progress(pb::ProgressBar, target::Real)
    stage = _NestedStage(target)
    return function (rng, model, sampler, sample, state, i; kwargs...)
        pb.enabled || return nothing
        Δ = _nested_dlogz(state)
        update!(pb; n_done = state.it, force = Δ <= target,
                fields = _nested_fields(stage, state.logz, Δ, state.ncall))
        return nothing
    end
end


"""
    _multinomial_resample(rng, weights, n) -> Vector{Int}

I.I.D. multinomial resampling from normalized `weights`: draw `n` indices
independently with probability proportional to weight. Unlike systematic
(low-variance) resampling this does NOT impose an ordering on the output,
so the resampled chain is exchangeable and its autocorrelation-based ESS
reflects the true effective sample size rather than a sorting artifact.
"""
function _multinomial_resample(rng::AbstractRNG, weights::Vector{Float64}, n::Int)
    cdf = cumsum(weights)
    cdf[end] = 1.0                       # guard against fp round-off < 1
    indices = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        indices[i] = searchsortedfirst(cdf, rand(rng))
    end
    return indices
end
