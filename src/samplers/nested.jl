# Nested sampling via NestedSamplers.jl (ellipsoidal decomposition).
#
# Provides proper evidence estimates comparable to dynesty/MultiNest.
# Uses the prior_transform from Nereus's prior system to map
# [0,1]^n → bounded parameter space.

using NestedSamplers
using Random
using MCMCChains

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
  `:rwalk` (random walk), `:rstagger` (staggered random walk),
  `:slice` (axis slice), `:rslice` (random-direction slice), or
  `:unif` (uniform rejection; low-d only).
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
- `walk_scale::Real=1.0` — initial proposal scale for the walk / slice
  kernels (passed to NestedSamplers as `scale`).
- `n_walks::Union{Int,Nothing}=nothing` — Metropolis steps per replacement for
  `:rwalk` / `:rstagger`. `nothing` → dimension-scaled `max(25, 2·n_dim)`
  (a fixed count under-decorrelates as n_dim grows → logZ biased low). An
  explicit value is honored verbatim.
- `slices::Union{Int,Nothing}=nothing` — slices per replacement for
  `:slice` / `:rslice`. `nothing` → `max(5, 2·n_dim)` for `:rslice`
  (PolyChord `num_repeats ∝ nDims`; dynesty `rslice`=`3+ndim`); `:slice`
  already loops all n_dim axes internally so it stays at 5. Explicit honored.
- `enlarge::Real=1.25` — linear volume factor by which the bounding
  ellipsoid(s) are enlarged after fitting to the live set. Raise it if the
  bound is under-covering, BUT note ellipsoidal bounds cannot conform to a
  strongly-curved (e.g. eccentric-RV) likelihood ridge — there the evidence
  bias is structural and `enlarge` does not remove it (use PT/TI evidence).
- `dlogz::Real=0.1` — stopping criterion (remaining evidence). Tighter than
  the canonical 0.5 because a loose tolerance let a run terminate while the
  live set was still confined to a wrong sub-mode (premature convergence at
  a much lower logZ). 0.1 forces enough extra iterations for the bound to
  re-expand and find the true mode's bulk.
- `seed::Int=1`
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

    # --- Dimension-scaled decorrelation defaults --------------------------
    # A single random-direction slice (RSlice) or Metropolis step (RWalk/
    # RStagger) decorrelates the new live point in ~ONE direction, so the
    # number of steps must grow ~linearly with n_dim to produce an
    # independent draw from the constrained prior. A FIXED count (the old
    # slices=5 / n_walks=25) is dimension-blind → under-decorrelates as n_dim
    # grows → logZ biased LOW. This is the PolyChord prescription
    # (`num_repeats ∝ nDims`, default 2·nDims) and dynesty's `rslice`
    # default (`3 + ndim`). We default to PolyChord's 2·nDims, floored at the
    # historical constants so low-d/easy problems never regress, and resolved
    # only when the user left the knob unset (an explicit value is honored
    # verbatim). The non-random `Slice` ALREADY loops over all n_dim axes per
    # iteration (its `slices` is full ndim-sweeps), so it is NOT scaled here —
    # scaling it too would be O(n_dim²).
    #
    # NOTE (scope): dimension scaling fixes the DECORRELATION bias, which
    # dominates on gentle posteriors. On sharply-curved RV ridges (the e–ω–Mo
    # eccentricity banana) the residual evidence bias is the multi-ellipsoid
    # BOUND failing to conform to the ridge — that is NOT removed by more
    # slices or a larger `enlarge`; use PT/TI for evidence there, or the
    # friends-bound (MLFriends) port. See project_ns_rslice_undermixing.
    slices_resolved = slices !== nothing ? slices :
                      (proposal === :slice ? 5 : max(5, 2 * n_dim))
    n_walks_resolved = n_walks !== nothing ? n_walks : max(25, 2 * n_dim)
    if slices === nothing || n_walks === nothing
        @info "sample_nested: dimension-scaled decorrelation (n_dim=$n_dim, " *
              "proposal=:$proposal) → slices=$slices_resolved " *
              "n_walks=$n_walks_resolved enlarge=$(Float64(enlarge))"
    end

    # Build prior distributions vector for NestedModel.
    # NestedSamplers accepts a vector of univariate distributions that
    # define the prior transform (unit cube → parameter space).
    prior_dists = [layout.unfrozen_priors[j].dist for j in 1:n_dim]

    # Log-likelihood function: maps bounded parameter vector → log L.
    # NestedSamplers draws from the prior internally via prior_dists,
    # so we must NOT evaluate log_prior here — only the data likelihood.
    #
    # PER-THREAD Theta buffers: batch/parallel NS (`sample_parallel`) calls
    # `loglike` concurrently from K constrained walks under `@threads :static`,
    # so a single shared buffer would race. One buffer per thread, indexed by
    # `threadid()` (stable under :static), avoids the race AND the per-eval
    # Theta allocation (NS calls loglike ~n_live × n_iter × walks = millions).
    theta_bufs = [Theta{Float64}(params) for _ in 1:max(1, Threads.nthreads())]
    # Per-thread PTWorkspace so the ws-aware likelihood path is used: it reuses
    # preallocated planet/cadence buffers (no per-call Vector allocs → no GC
    # thrash over the millions of NS evals) AND the per-planet flux cache + the
    # total phot-ll cache. Same proven path as PT/rjmcmc; one ws per thread,
    # indexed by threadid() (stable under NS's :static parallel walks).
    n_noise_ws = length(params.config.noise_models)
    ws_bufs = [PTWorkspace(params, params.config.max_kplanet, n_noise_ws;
                           n_obs = length(data.t_rv), n_phot = length(data.t_phot))
               for _ in 1:max(1, Threads.nthreads())]
    function loglike(x)
        tb = theta_bufs[Threads.threadid()]
        wb = ws_bufs[Threads.threadid()]
        @inbounds for (j, idx) in enumerate(layout.unfrozen_idx)
            tb.values[idx] = x[j]
        end
        ll = rv_log_likelihood(tb, data, wb)
        isfinite(ll) || return -1e300
        ll += transit_log_likelihood(tb, data, wb)
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
        # Reflective (Hamiltonian) slice — gradient-aware, for thin curved ridges.
        # `slices` here = number of trajectories; the dimension-scaled default is
        # fine. n_steps fixed at 15.
        NestedSamplers.Proposals.HSlice(; slices=slices_resolved, n_steps=15, scale=walk_scale)
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
    # Disabling the bar sidesteps the API drift entirely — and the
    # progress output is useless under juliacall on the exoautomata
    # workers anyway.
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
    run_serial() = NestedSamplers.sample(MersenneTwister(seed), model, sampler;
                                          dlogz=dlogz, progress=false)
    ns_chains, ns_state = if use_parallel
        # Belt-and-suspenders: even the random-walk batch path can throw from
        # inside a worker task (a collapsed-scale "stuck" walk). A failed
        # PARALLEL run must NOT take the sampler down — catch it, warn LOUD,
        # and re-run SERIALLY (the single-point path is more robust). A real
        # bug would then resurface in the serial run and propagate normally.
        try
            NestedSamplers.sample_parallel(rng, model, sampler;
                                            batch_size=K, parallel=true,
                                            dlogz=dlogz, progress=false)
        catch err
            @warn "Parallel nested sampling failed (likely a stuck proposal " *
                  "inside a worker task); falling back to the SERIAL sampler. " *
                  "This is slower but more robust." exception = (err, catch_backtrace())
            run_serial()
        end
    else
        run_serial()
    end
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
