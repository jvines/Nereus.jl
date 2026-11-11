# NUTS driver via AdvancedHMC.jl with multi-chain parallelism.
#
# NUTS is a LOCAL sampler: it follows the gradient and CANNOT hop
# between period-alias / disjoint modes. Two failure modes had to be
# closed before it could be trusted on the production RV path:
#
#   (1) FROZEN CHAINS. The old metric was a diagonal mass matrix built
#       from `1/g²` at a single PRIOR-DRAW init. At a prior draw the
#       gradients are pathologically large (|g| ~ 5000 on the RV
#       posterior), so inv_mass ~ 1/g² ~ 1e-8 → near-infinite mass →
#       the momentum can't move the position. Combined with a hardcoded
#       ε = 0.1/√dim the low-level loop never built up the windowed
#       samples the StanHMCAdaptor needs, so the chain stayed frozen
#       (ESS ~ 3, R-hat 2-3). FIX: start from a UNIT DiagEuclideanMetric,
#       pick ε with `find_good_stepsize`, and let the StanHMCAdaptor's
#       windowed Welford estimator adapt the diagonal mass matrix from
#       real draws — the canonical AdvancedHMC idiom.
#
#   (2) DISJOINT-MODE MERGE. Independent prior-draw inits drop each
#       chain into a different alias basin; MCMCChains then merges the
#       frozen basins into one CI that can spuriously bracket truth
#       (false recovery). FIX: WARM-START every chain from a short
#       pt_emcee global pre-search so the unimodal/eccentric targets land
#       in the dominant basin and mix (honest R-hat < 1.1). On genuinely
#       multimodal targets warm-start scatters chains across the modes,
#       cross-chain R-hat stays high, and `assess_fit`'s multimodality
#       check fires — NUTS FAILS LOUD instead of silently merging.
#
# OBSERVABILITY: the returned `Chains` carries per-chain NUTS health
# (divergence count, final adapted step size, mean/max tree depth, mean
# acceptance) in `chains.info` so non-convergence is detectable without
# re-deriving it from the trace. See `nuts_diagnostics(chains)`.
#
# CHECKPOINT / RESUME: the sampling loop is `AdvancedHMC.sample`'s, run here
# (`_nuts_loop!`) instead of inside that call, so that what it carries between
# iterations -- the position, the RNG, the metric, the step size and the
# StanHMCAdaptor's windowed state -- can be saved and put back. A resumed run
# is bit-identical to one that ran straight through (src/checkpoint.jl).

using AdvancedHMC
using ForwardDiff
using LogDensityProblemsAD
using MCMCChains
using Random
using Statistics: mean, std

"""
    _draw_from_prior(target, rng; max_attempts=1000) -> Vector{Float64}

Draw a random initial position (in bounded space) that gives a finite
log-posterior. Uses rejection sampling to avoid unphysical regions.
"""
function _draw_from_prior(target::NereusTarget, rng::AbstractRNG;
                           max_attempts::Int=1000, allow_nonfinite::Bool=false)
    params = target.params
    layout = params.layout
    n = length(layout.unfrozen_idx)
    # Use bounded-space target for validation (no transform)
    bounded_target = NereusTarget(params, target.data; unconstrained=false)

    # Unfrozen positions of each planet-mode group's period slots. Random
    # draws must be SORTED within each group before evaluation: the period-
    # ordering hard prior otherwise rejects all but 1/k! of multi-planet
    # draws (k identical slots → e.g. 1/720 for six, so 1000 attempts fail
    # outright on a 6-planet model) and the all-medians fallback is tied →
    # always rejected by the strict inequality.
    P_groups = Vector{Vector{Int}}()
    let modes = params.config.planet_modes, blocks = layout.planet_blocks,
        seen = Dict{Any,Int}()
        for k in eachindex(blocks)
            Ppos = findfirst(==(blocks[k].P), layout.unfrozen_idx)
            Ppos === nothing && continue
            gi = get!(seen, modes[k], length(P_groups) + 1)
            gi > length(P_groups) && push!(P_groups, Int[])
            push!(P_groups[gi], Ppos)
        end
    end

    # A cyclic inclination (src/circular.jl) is drawn on its circle and folded
    # to [0, π]: every walker then starts in the same representative, sin(i)/2
    # distributed, rather than half of them in the mirror one.
    cinc = cyclic_inclination_indices(params)

    last_lp = -Inf
    last_x = Vector{Float64}(undef, n)
    for attempt in 1:max_attempts
        x = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            ps = layout.unfrozen_priors[i]
            u = rand(rng)
            val = quantile(ps, u)
            if !isfinite(val)
                lo, hi = bounds(ps)
                val = if isfinite(lo) && isfinite(hi)
                    (lo + hi) / 2
                elseif isfinite(lo)
                    lo + 1.0
                elseif isfinite(hi)
                    hi - 1.0
                else
                    0.0
                end
            end
            x[i] = val
        end
        fold_cyclic_inclinations!(x, params; idx = cinc)
        # canonical period order within each mode group (draw is in bounded
        # space, so sorting the values at the P positions sorts the periods)
        for grp in P_groups
            length(grp) >= 2 || continue
            vals = sort!([x[p] for p in grp])
            for (j, p) in enumerate(grp)
                x[p] = vals[j]
            end
        end
        lp = LogDensityProblems.logdensity(bounded_target, x)
        last_lp = lp; last_x .= x
        isfinite(lp) && return x
    end

    # Trans-dim callers evaluate the bare target with EVERY noise model active
    # (td === nothing ⇒ all active), which is unconditionally −Inf for a menu
    # holding mutually-eval-incompatible members (StudentT + a GP, ActivityGP +
    # an additive term). Their real finiteness check is per-walker against the
    # actual toggled td_state, so hand them a raw prior draw and let that gate.
    allow_nonfinite && return last_x

    # All `max_attempts` random draws failed. Diagnose the offending
    # log-likelihood component on the last attempt, then fall back to
    # the prior medians (which usually gives a finite log-posterior on
    # a well-posed model). Logs a warning so the user knows the prior-
    # init is not actually random for this walker — but at least the
    # chain can proceed.
    theta_dbg = Theta{Float64}(params)
    set_unfrozen!(theta_dbg, last_x)
    lpr = log_prior(theta_dbg)
    llr = rv_log_likelihood(theta_dbg, target.data)
    llt = isfinite(llr) ? transit_log_likelihood(theta_dbg, target.data) +
                          tomogram_log_likelihood(theta_dbg, target.data) : NaN

    x_med = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ps = layout.unfrozen_priors[i]
        val = prior_centre(ps)
        if !isfinite(val)
            lo, hi = bounds(ps)
            val = isfinite(lo) && isfinite(hi) ? (lo+hi)/2 :
                  isfinite(lo) ? lo+1.0 :
                  isfinite(hi) ? hi-1.0 : 0.0
        end
        x_med[i] = val
    end
    # Grouped period slots get rank-staggered quantiles instead of the
    # shared median — all-equal periods violate the strict ordering prior.
    for grp in P_groups
        m = length(grp)
        m >= 2 || continue
        for (j, p) in enumerate(grp)
            x_med[p] = quantile(layout.unfrozen_priors[p], j / (m + 1))
        end
    end
    lp_med = LogDensityProblems.logdensity(bounded_target, x_med)

    if isfinite(lp_med)
        @warn("_draw_from_prior: $max_attempts random draws all gave " *
              "non-finite log-posterior — falling back to prior medians " *
              "(tid=$(Threads.threadid()), n_dim=$n). " *
              "Last failed-draw breakdown: log_prior=$lpr, rv_ll=$llr, " *
              "transit_ll=$llt. The median fallback works (lp=$lp_med), " *
              "but consider tightening priors or providing `init=...` " *
              "for genuinely random walker init.")
        return x_med
    end

    throw(ErrorException(
        "_draw_from_prior: no finite log-posterior after $max_attempts " *
        "random draws AND the prior medians (lp_med=$lp_med). " *
        "Components on last_x:  log_prior=$lpr  rv_ll=$llr  transit_ll=$llt. " *
        "If one is -Inf or NaN, that's the offending term. " *
        "Common causes: (a) data contains NaN/Inf (check `data.flux` / " *
        "`data.flux_err` / `data.rv` for non-finite values after " *
        "detrending), (b) ephemeris+T0 mismatched putting model transit " *
        "outside the data subset window, (c) per-transit TTV slots not " *
        "covering all observed cycle indices. Fix the model setup or " *
        "provide `init = ...` to bypass prior-init."))
end

"""
    _warmstart_points(target, n_chains, rng; n_temps, n_walkers, n_steps,
                      n_burnin) -> Vector{Vector{Float64}}

Run a short pt_emcee global pre-search and return `n_chains` HIGH-lp
initial positions in BOUNDED space — one per NUTS chain. NUTS is a
local sampler, so a good init is the difference between mixing in the
true basin and freezing in an alias mode.

The points are drawn from the TOP of the cold-chain ensemble: chain 1
gets the single best (max-lp) draw, the rest get the next-best draws.
On a UNIMODAL target all chains land in the same basin and mix (honest
R-hat). On a MULTIMODAL target the top draws come from DIFFERENT modes,
so the chains scatter across them, cross-chain R-hat stays high, and
`assess_fit`'s multimodality check fires — i.e. NUTS fails LOUD rather
than silently merging disjoint basins. We deliberately do NOT collapse
all chains onto the single best draw, which would manufacture a clean
R-hat that hides the multimodality.

Falls back to independent prior draws if the pre-search throws (e.g. a
model shape pt_emcee cannot init) so NUTS still runs — just without the
warm start.
"""
function _warmstart_points(target::NereusTarget, n_chains::Int,
                           rng::AbstractRNG;
                           n_temps::Int = 6, n_walkers::Int = 40,
                           n_steps::Int = 400, n_burnin::Int = 200)
    layout = target.params.layout
    n = length(layout.unfrozen_idx)
    n_walkers_eff = max(n_walkers, 2 * n + 2)
    n_walkers_eff += isodd(n_walkers_eff) ? 1 : 0
    seed = rand(rng, 1:typemax(Int32))
    try
        res = sample_pt_emcee(target, target.data;
                             n_temps = n_temps, n_walkers = n_walkers_eff,
                             n_steps = n_steps, n_burnin = n_burnin,
                             init_strategy = :prior, seed = seed,
                             show_progress = false)
        ch = res.chains
        # Re-chart every full-circle angle from the WHOLE post-burn-in
        # pre-search (src/circular.jl) before NUTS sees the target. NUTS
        # samples in the logit chart, where the 0/2π seam is a wall at y = ±∞
        # it cannot cross. Measured on a posterior centred on Mo = 0: without
        # a moved chart all four chains came back confined to the sliver below
        # 2π, 358° ± 2° against the true 7° ± 5°; with a converged pre-search
        # (warm_steps = 3000) the seam moved off the posterior and NUTS matched
        # pt_emcee. The chart is only as good as the pre-search: the default
        # 400 steps had not located Mo there, and a seam cut from unconverged
        # draws can still land in the posterior -- fit_health's rail check,
        # measured in the window NUTS sampled in, reports that case. Moving
        # the chart before NUTS starts (and before its step size and metric
        # adapt) is exact, and the init points below come from the relabelled
        # draws.
        recenter_circular!(ch, target.params; transforms = (target.transform,))
        pnames = String.(names(ch, :parameters))
        # `Array(ch)` collapses the walker axis into the iteration axis,
        # giving a flat (n_draw × n_param) matrix — exactly the pooled
        # cold-ensemble sample list we want to rank by :lp. Each row is a
        # bounded-space parameter vector; reorder columns to align with
        # layout.unfrozen_names.
        slot_idx = [findfirst(==(nm), pnames) for nm in layout.unfrozen_names]
        any(isnothing, slot_idx) && error(
            "pt_emcee warm-start chains missing a fitted slot")
        lp_col = findfirst(==("lp"), pnames)
        flat = Array(ch)                 # (n_draw, n_param)
        n_draw = size(flat, 1)
        cand   = Vector{Vector{Float64}}()
        cand_lp = Float64[]
        for it in 1:n_draw
            x = Float64[flat[it, j] for j in slot_idx]
            all(isfinite, x) || continue
            lpv = lp_col === nothing ? 0.0 : flat[it, lp_col]
            isfinite(lpv) || continue
            push!(cand, x); push!(cand_lp, lpv)
        end
        isempty(cand) && error("pt_emcee warm-start produced no finite draws")
        order = sortperm(cand_lp; rev = true)
        pts = Vector{Vector{Float64}}(undef, n_chains)
        for c in 1:n_chains
            # chain c takes the c-th best draw (cycling if fewer uniques).
            pts[c] = copy(cand[order[mod1(c, length(order))]])
        end
        return pts
    catch e
        @warn("NUTS warm-start pre-search failed ($(sprint(showerror, e))); " *
              "falling back to independent prior-draw init. NUTS may freeze " *
              "in disjoint modes — check the returned diagnostics / assess_fit.")
        return [_draw_from_prior(target, rng) for _ in 1:n_chains]
    end
end

"""
    nuts_diagnostics(chains) -> NamedTuple | nothing

Per-chain NUTS health attached by [`sample_nuts`](@ref) to
`chains.info`: `n_divergent`, `step_size`, `mean_tree_depth`,
`max_tree_depth`, `mean_accept` (each a vector over chains). Returns
`nothing` if the chains were not produced by `sample_nuts`.
"""
function nuts_diagnostics(chains::MCMCChains.Chains)
    info = chains.info
    haskey(info, :n_divergent) || return nothing
    return (n_divergent     = info.n_divergent,
            step_size        = info.step_size,
            mean_tree_depth  = info.mean_tree_depth,
            max_tree_depth   = info.max_tree_depth,
            mean_accept      = info.mean_accept)
end


"""
    sample_nuts(target::NereusTarget; kwargs...) -> MCMCChains.Chains

Run NUTS on a `NereusTarget`. Returns posterior as `MCMCChains.Chains`.
NUTS is a LOCAL sampler — it cannot jump period-alias / disjoint modes.
Chains are warm-started from a short pt_emcee global pre-search so the
unimodal / eccentric targets land in and mix within the dominant basin;
on genuinely multimodal targets the warm-started chains scatter across
modes, R-hat stays high, and `assess_fit` flags the multimodality (NUTS
fails LOUD, never silently merges). Per-chain divergence count / step
size / tree depth are attached to `chains.info` (see
[`nuts_diagnostics`](@ref)).

# Keywords
- `n_samples::Int=1000`    : post-warmup samples per chain
- `n_warmup::Int=1000`     : warmup (adaptation) steps per chain. Drives
  both Nesterov dual-averaging (step size) and the windowed Welford
  diagonal mass-matrix estimator — too short and the metric never
  adapts, so keep ≥ ~500.
- `n_chains::Int=4`        : number of parallel chains (uses Julia
  threads). ≥2 is required for a meaningful cross-chain R-hat — a
  single chain cannot self-diagnose disjoint-mode freezing.
- `target_accept::Float64=0.8` : target Metropolis acceptance rate
- `ad_backend::Symbol=:ForwardDiff` : autodiff backend. `:ForwardDiff`
  (fastest for ≤15 params), `:Enzyme` (reverse-mode, better for many
  params / GP models), or `:ReverseDiff` (requires `import ReverseDiff`),
  which records its tape afresh at every point.
- `compile_tape::Bool=false` : refused (an `ArgumentError`) with
  `ad_backend = :ReverseDiff`. A compiled tape replays the operations
  recorded at one point, the branches taken there included, at every other
  point, and every Nereus log density branches on the parameter values:
  Kepler's solver stops once it has converged, a point outside the support
  returns -Inf, transit windows and clamps choose what is computed. Its
  values and gradients were wrong away from the point where it was recorded.
  No effect with the other backends. A checkpoint from a run that left it at
  its former default, `true`, is resumed by passing `compile_tape = true`.
- `warm_start::Bool=true`  : warm-start chains from a short pt_emcee
  pre-search. Disable only if you pass `init` or know the target is
  trivially unimodal; with it off, independent prior-draw inits will
  freeze NUTS in disjoint modes on multi-planet RV posteriors.
  Without it, the seams of the full-circle angles are placed by a short
  pathfinder pilot instead (`_seam_pilot!`). Either way the chains re-cut
  them together at the phase boundaries of the adaptation (`_hmc_recut!`), so
  that none is left against a seam the start placed on the posterior.
- `warm_temps`, `warm_walkers`, `warm_steps`, `warm_burnin` : pt_emcee
  pre-search budget (defaults 6 / 40 / 400 / 200).
- `init::Union{Nothing, Vector{Float64}}=nothing` : initial position in
  **bounded** space (overrides `warm_start`). Transformed to
  unconstrained internally if the target uses transforms.
  Each full-circle angle's window is centred on it.
- `seed::Union{Nothing,Integer}=nothing` : run on `MersenneTwister(seed)`.
  `run_job` passes the job's seed here. Give `seed` or `rng`, not both.
- `rng::Union{Nothing,AbstractRNG}=nothing` : the generator to run on;
  `nothing` (with no `seed`) is `Random.default_rng()`.
- `progress::Bool=true`
- `checkpoint=nothing` — path of a state file. The sampler writes its whole
  state there (each chain's position, step size, mass matrix and Stan
  adaptation state, every RNG stream, the circular windows the warm start
  moved, and the kept draws with their NUTS statistics) at the end of the run
  and every `checkpoint_interval` seconds during it, replacing the file
  atomically. `run_job` sets it to `nuts_state.jls` in `output_dir` unless the
  job gives one.
- `checkpoint_interval::Real=900` — seconds between checkpoints during the
  run. Each chain keeps its own clock; the file always holds every chain's
  latest saved state. A chain that has not saved yet, such as one still
  queued behind the others when there are more chains than threads, is held
  there as its start point and RNG stream, so on a resume it starts afresh
  exactly as it would have.
- `resume::Bool=false` — continue from `checkpoint` instead of initialising
  (no warm-start pre-search, no step-size search): a run that was killed, or a
  finished one that needs more draws. `n_samples` is the new TOTAL of kept
  draws per chain, counted from the start of the original run; the draws
  already kept stay, and the new ones are appended. The continuation is
  bit-identical to an uninterrupted run of `n_samples`, whether the original
  stopped inside the warm-up or after it, and `rng` is left as that run leaves
  it. Pass the same `seed`, or an `rng` in the state the original run started
  from (a fresh `MersenneTwister(5)` again, not the one that run advanced).
  Refused, with the differences listed, if the checkpoint came from
  different data, priors, parameters, `n_warmup`, `n_chains`, `target_accept`,
  AD settings, warm-start settings, `init`, `seed`, `rng` or AdvancedHMC
  version. Of the RNG, the fingerprint keeps the starting state of a `seed`ed
  or passed generator; the unseeded default stream is not part of it, so an
  unseeded run can still be continued.
"""
function sample_nuts(
    target::NereusTarget;
    n_samples::Int = 1000,
    n_warmup::Int = 1000,
    n_chains::Int = 4,
    target_accept::Real = 0.8,
    ad_backend::Symbol = :ForwardDiff,
    compile_tape::Bool = false,
    warm_start::Bool = true,
    warm_temps::Int = 6,
    warm_walkers::Int = 40,
    warm_steps::Int = 400,
    warm_burnin::Int = 200,
    init::Union{Nothing, Vector{Float64}} = nothing,
    seed::Union{Nothing,Integer} = nothing,
    rng::Union{Nothing,AbstractRNG} = nothing,
    progress::Bool = true,
    checkpoint::Union{Nothing,AbstractString,Symbol} = nothing,
    checkpoint_interval::Real = 900.0,
    resume::Bool = false,
    kwargs...
)
    seed !== nothing && rng !== nothing && throw(ArgumentError(
        "sample_nuts takes `seed` or `rng`, not both"))
    # Rebound once, here. No closure below captures `rng`, so it is not boxed.
    rng = seed !== nothing ? MersenneTwister(seed) :
          rng === nothing ? Random.default_rng() : rng
    ad_backend === :ReverseDiff && compile_tape && throw(ArgumentError(
        "compile_tape = true is refused: a compiled ReverseDiff tape replays the " *
        "branches the log density took where it was recorded, and Nereus log " *
        "densities branch on the parameter values (Kepler's solver stops once " *
        "converged, a point outside the support returns -Inf, transit windows), " *
        "so its values and gradients are wrong away from that point. Leave it " *
        "false: ReverseDiff then records the tape afresh at every point."))
    target_accept = Float64(target_accept)   # JSON may deliver an Int
    ck_path = checkpoint === nothing ? nothing : String(checkpoint)
    resume && ck_path === nothing && throw(ArgumentError(
        "resume = true needs `checkpoint`, the state file to continue from"))
    params = target.params
    n_total = n_warmup + n_samples           # iterations per chain

    # --- Checkpoint / resume (src/checkpoint.jl) ------------------------
    # Read before the warm-start pre-search: a resumed run skips it, and the
    # step-size search, because what they produced (the moved circular windows,
    # each chain's position, step size and metric) is in the checkpoint. The
    # adaptor is saved as AdvancedHMC's own object, so the AdvancedHMC version
    # is part of the fingerprint. So is the RNG's starting state, read here
    # before anything draws from it: a different seed is a different run.
    # And the space the chains move in: a checkpoint holds unconstrained
    # coordinates, which a bounded target would read as bounded values.
    ck_fp = run_fingerprint(params, target.data; n_warmup, n_chains,
        space = target.transform isa PackedTransforms ? :unconstrained : :bounded,
        target_accept, ad_backend, compile_tape, warm_start, warm_temps,
        warm_walkers, warm_steps, warm_burnin,
        init = init === nothing ? nothing : copy(init),
        seed = seed === nothing ? nothing : Int(seed),
        rng = _rng_fingerprint(rng),
        advancedhmc = string(pkgversion(AdvancedHMC)),
        # The seams re-cut at the adaptation's phase boundaries (`_hmc_recut!`),
        # polar charts too: a checkpoint from before either is another run,
        # and holds no trace or a trace of another layout.
        seam_recut = :adaptation_phases_polar)
    ck = resume ? read_checkpoint(ck_path, "nuts", ck_fp) : nothing
    if ck !== nothing
        for (c, s) in enumerate(ck.chains)
            s.step::Int <= n_total || throw(ArgumentError(
                "chain $c of the checkpoint is at iteration $(s.step); " *
                "n_warmup + n_samples = $n_total would end before it"))
        end
        # Every entry is between the same two re-cuts (`_nuts_recut!` writes them
        # all at once), so the chains resume in one chart.
        allequal(_nuts_started(s) && s.circ !== nothing ? s.circ.recut_at : 0
                 for s in ck.chains) || throw(ArgumentError(
            "the checkpoint's chains stand on both sides of a seam re-cut"))
    end

    if ck === nothing
        # Per-chain bounded-space init points. `init` (if given) pins every
        # chain at the same point; otherwise warm-start from a short pt_emcee
        # pre-search (the disjoint-mode fix), or fall back to independent
        # prior draws when warm_start is off.
        #
        # Without the warm start's re-chart the windows are placed here, before
        # the chains re-cut them during the adaptation (`_hmc_recut!`): centred
        # on a caller's `init`, the one point known to be in the posterior
        # (which also keeps an angle written in the user's window off a moved
        # window's wall, where transform_forward would clamp it), or by a
        # pathfinder pilot before prior draws (`_seam_pilot!`).
        init_points = if init !== nothing
            x0 = _center_windows_on!(params, copy(init);
                                     transforms = (target.transform,))
            [copy(x0) for _ in 1:n_chains]
        elseif warm_start
            _warmstart_points(target, n_chains, rng;
                              n_temps = warm_temps, n_walkers = warm_walkers,
                              n_steps = warm_steps, n_burnin = warm_burnin)
        else
            isempty(seam_indices(params)) ||
                _seam_pilot!(target, rand(rng, 1:typemax(Int32)))
            pts = Vector{Vector{Float64}}(undef, n_chains)
            for c in 1:n_chains
                pts[c] = _draw_from_prior(target, rng)
            end
            pts
        end
        # One RNG stream per chain, seeded from `rng`. A single chain runs on
        # `rng` itself.
        chain_rngs = Vector{AbstractRNG}(undef, n_chains)
        if n_chains == 1
            chain_rngs[1] = rng
        else
            for c in 1:n_chains
                chain_rngs[c] = MersenneTwister(rand(rng, UInt64))
            end
        end
        # Every chain's checkpoint entry exists before any chain runs: its start
        # point and RNG stream, until it saves its own state. A chain queued
        # behind a full thread pool has not started when the running ones save,
        # and the tasks never yield, so it could otherwise have no entry until
        # one of them finished; a checkpoint written in between would lack it.
        # Without a checkpoint the stream is not copied (not every RNG can be).
        snaps = Any[_nuts_unstarted(init_points[c],
                                    ck_path === nothing ? nothing : copy(chain_rngs[c]))
                    for c in 1:n_chains]
    else
        # The windows the saved chains live in (moved by the warm-start
        # re-chart), and `rng` as the uninterrupted run left it once the chains
        # had started.
        for (nm, (lo, _)) in ck.windows
            set_circular_window!(params, findfirst(==(nm), params.layout.unfrozen_names),
                                 lo; transforms = (target.transform,))
        end
        _restore_rng!(rng, ck.rng_master)
        # Each chain's stream as its entry holds it, saved or unstarted. A
        # single chain's stream is `rng` itself.
        chain_rngs = Vector{AbstractRNG}(undef, n_chains)
        for c in 1:n_chains
            chain_rngs[c] = n_chains == 1 ? rng : MersenneTwister(0)
            _restore_rng!(chain_rngs[c], ck.chains[c].rng)
        end
        snaps = collect(Any, ck.chains)
    end

    ckw = ck_path === nothing ? nothing :
          _NutsCheckpoint(ck_path, ck_fp, Float64(checkpoint_interval), snaps,
                          copy(rng), circular_windows(params), ReentrantLock())
    stops = _NUTS_STOP_AFTER[]
    # The seam re-cut during the adaptation, all chains together (`_hmc_recut!`).
    # A chain that ends, stopped or failed, aborts the rendezvous: a re-cut is
    # taken by every chain or none.
    rc = _has_recut_angles(params) ?
         _HMCRecut(params, target.transform, _Rendezvous(n_chains)) : nothing
    chain_kw = (; n_samples, n_warmup, target_accept, ad_backend, rc)
    run_chain(c, rng_c, prog) = try
        _nuts_chain(target, c, snaps[c], ckw, get(stops, c, 0); chain_kw...,
                    rng = rng_c, progress = prog)
    finally
        rc === nothing || _rendezvous_abort!(rc.rv)
    end

    if n_chains == 1
        res = [run_chain(1, chain_rngs[1], progress)]
    else
        # Run chains in parallel. Only show progress on chain 1. Each task
        # owns its own NereusTarget AD wrapper + RNG so there is no shared
        # state; each chain starts from, or resumes, its entry in `snaps`.
        tasks = Vector{Task}(undef, n_chains)
        for c in 1:n_chains
            chain_rng = chain_rngs[c]
            show_progress = progress && (c == 1)
            tasks[c] = Threads.@spawn run_chain(c, chain_rng, show_progress)
        end
        res = [fetch(t) for t in tasks]
    end
    any(isnothing, res) && throw(_NutsStopped())
    n_chains == 1 && return res[1]
    chains_list = res

    # Tag each chain with its chain ID and merge.
    merged = MCMCChains.chainscat(chains_list...)

    # chainscat keeps only the FIRST chain's `.info`; rebuild per-chain
    # diagnostic VECTORS from every chain so the merged object carries
    # full observability.
    function _gather(sym)
        [haskey(c.info, sym) ? getproperty(c.info, sym) : NaN
         for c in chains_list]
    end
    merged = setinfo(merged, (
        n_divergent     = _gather(:n_divergent),
        step_size       = _gather(:step_size),
        mean_tree_depth = _gather(:mean_tree_depth),
        max_tree_depth  = _gather(:max_tree_depth),
        mean_accept     = _gather(:mean_accept),
    ))
    return merged
end

# Test hook, never set by a real run: entry `c` is the iteration at which chain
# `c` stops, checkpointed, as a killed run would (0 or no entry: it does not).
# No `n_samples` can stop a run inside the warm-up, which the resume tests need.
const _NUTS_STOP_AFTER = Ref(Int[])

"""Thrown by `sample_nuts` when the `_NUTS_STOP_AFTER` test hook stopped a chain."""
struct _NutsStopped <: Exception end

# What `sample_nuts` checkpoints into: each chain's latest saved state, plus
# what is fixed once the chains have started (the caller's `rng` and the
# circular windows). Every chain has an entry from the start (`_nuts_unstarted`
# until it saves). Chains save on their own clocks, under `lock`.
struct _NutsCheckpoint
    path::String
    fingerprint::NamedTuple
    interval::Float64
    snaps::Vector{Any}
    rng_master::AbstractRNG
    windows::Dict{String, Tuple{Float64, Float64}}
    lock::ReentrantLock
end

# A chain's entry before it has saved a state of its own: the bounded-space
# point it starts from and a copy of its RNG stream before the start draws from
# it (`nothing` when the run keeps no checkpoint), which is all a fresh start
# needs (`_nuts_chain`).
_nuts_unstarted(init::Vector{Float64}, rng::Union{Nothing,AbstractRNG}) =
    (; step = 0, init = copy(init), rng)
_nuts_started(snap::NamedTuple) = haskey(snap, :z)

"""Put a checkpointed RNG stream back into `rng`, in place."""
function _restore_rng!(rng::AbstractRNG, saved::AbstractRNG)
    applicable(copy!, rng, saved) || throw(ArgumentError(
        "the checkpoint's RNG is a $(typeof(saved)), which cannot be put back " *
        "into the $(typeof(rng)) passed as `rng`"))
    copy!(rng, saved)
    return rng
end

# What the run fingerprint keeps of the RNG a run starts on: its type and a hash
# of its serialized state (every AbstractRNG serializes; not every one defines
# `==`). The task-local default stream is left out, as `nothing`: it is the
# unseeded case, and its state in a new process says nothing about the run.
_rng_fingerprint(::Random.TaskLocalRNG) = nothing
function _rng_fingerprint(rng::AbstractRNG)
    io = IOBuffer()
    serialize(io, rng)
    return (string(typeof(rng)), hash(String(take!(io))))
end

"""
    _nuts_snap(step, z, metric, kernel, adaptor, rng, draws, stats, cs) -> NamedTuple

A chain's checkpoint entry after iteration `step`. Copies everything the chain
goes on mutating. `circ` is the seam re-cut's part (`_HMCRecutChain`'s trace
and the last boundary re-cut), `nothing` for a target without such angles.
"""
function _nuts_snap(step::Int, z, metric, kernel, adaptor, rng::AbstractRNG, draws, stats,
                    cs)
    # One deepcopy for all four: during warm-up the metric's M⁻¹ is the
    # adaptor's variance array itself, and stays one array in the copy.
    live = deepcopy((; z, metric, kernel, adaptor))
    circ = cs === nothing ? nothing : (; trace = deepcopy(cs.trace), cs.recut_at)
    return (; step, live..., rng = copy(rng), draws = copy(draws), stats = copy(stats),
            circ)
end

# Write the checkpoint file from `ckw`'s entries. With `ckw.lock` held.
_nuts_write!(ckw::_NutsCheckpoint) =
    write_checkpoint(ckw.path, "nuts", ckw.fingerprint,
                     (; chains = ckw.snaps, rng_master = ckw.rng_master,
                        windows = ckw.windows))

"""
    _nuts_snapshot!(ckw, c, step, z, metric, kernel, adaptor, rng, draws, stats, cs)

Save chain `c`'s state after iteration `step` as its entry in `ckw`
(`_nuts_snap`) and write the checkpoint file, which holds every chain's latest
entry.
"""
function _nuts_snapshot!(ckw::_NutsCheckpoint, c::Int, step::Int, z, metric, kernel,
                         adaptor, rng::AbstractRNG, draws, stats, cs)
    snap = _nuts_snap(step, z, metric, kernel, adaptor, rng, draws, stats, cs)
    lock(ckw.lock) do
        ckw.snaps[c] = snap
        _nuts_write!(ckw)
    end
    return nothing
end

"""
    _nuts_logdensity_fns(target, ad_backend) -> (ℓ, ∂ℓ)

The log density `ℓ(y)` and its value-and-gradient `∂ℓ(y)` that a NUTS chain
integrates, in the target's own space, by `ad_backend`.

A ReverseDiff tape is never compiled: a compiled tape replays the branches
taken at the point where it was recorded (LogDensityProblemsAD records it at
zeros(dim)) at every other point, and every Nereus log density branches on
the parameter values. On an eccentric orbit whose zeros(dim) is e = 0,
Kepler's solver takes no Newton step there, so the tape took none anywhere;
its log density was off by hundreds of nats and its gradients by as much as
the gradients themselves (test/test_nuts_reversediff.jl).
"""
function _nuts_logdensity_fns(target::NereusTarget, ad_backend::Symbol)
    ℓ_fn, ∂ℓ_fn, _ = _nuts_logdensity_resync(target, ad_backend)
    return ℓ_fn, ∂ℓ_fn
end

# `_nuts_logdensity_fns` and what must run after a circular window moves before
# they are called again: `nothing`, or for the Enzyme path the resync of its
# restore fallback (`_enzyme_resync!`).
function _nuts_logdensity_resync(target::NereusTarget, ad_backend::Symbol)
    if ad_backend === :Enzyme && target.transform isa PackedTransforms
        # Custom Enzyme path with explicit Const arguments.
        enzyme_cfg = EnzymeGradientConfig(target)
        ℓ_fn = y -> enzyme_logdensity_and_gradient!(enzyme_cfg, target, y)[1]
        ∂ℓ_fn = y -> enzyme_logdensity_and_gradient!(enzyme_cfg, target, y)
        return ℓ_fn, ∂ℓ_fn, () -> _enzyme_resync!(enzyme_cfg, target)
    end
    ℓ_ad = LogDensityProblemsAD.ADgradient(ad_backend, target)
    ℓ_fn = y -> LogDensityProblems.logdensity(ℓ_ad, y)
    ∂ℓ_fn = y -> LogDensityProblems.logdensity_and_gradient(ℓ_ad, y)
    return ℓ_fn, ∂ℓ_fn, nothing
end

# =====================================================================
# Seam re-cut during the HMC adaptation (NUTS and pt_hmc)
#
# NUTS and pt_hmc walk in the logit chart, where a circular window's seam is a
# wall at y = ±∞. Without the pt_emcee warm start the windows are placed once,
# by a pathfinder pilot (`_seam_pilot!`), and chains from prior draws then
# found their way to a mode across a seam the pilot had left on the posterior
# (or to a minor mode against one), and stayed there: on an RV fit with Mo and
# ω both near their seams, one of four NUTS chains sat at ω = -2.5°, the wall
# of the pilot's [-362.5°, -2.5°), for its whole run, and the pt_hmc cold chain
# came back in a mode 170° from pt_emcee's.
#
# So the chains re-cut the seams themselves during the adaptation, at its
# phase boundaries: the end of the step-size-only init buffer and every end of
# a metric window (Stan's schedule; the last one opens the term buffer). All
# chains of a run share the windows (they live in the layout), so they meet
# there (`_Rendezvous`): each brings the angles it visited since the previous
# boundary (the second half of the init buffer, whose start is the transient
# from a prior draw), the last to arrive moves each window's seam to the
# emptiest arc of the pooled visits (`recut_circular_param!`, which moves
# nothing unless the seam is crowded), and every chain relabels its position
# into the new windows. A chain held at a wall had piled its visits there, so
# the seam leaves it and it is in the interior of its chart, free to cross to
# the main mode; if it stays in a minor mode it does so inside the window.
# Cyclic inclinations are folded to [0, π] at the same points, as the other
# samplers' re-cuts fold them.
#
# A move re-charts a coordinate, so at a window end, where the adaptor has
# just set the metric from the window's draws in the old chart, those draws
# are re-expressed in the new one and the coordinate's variance re-estimated
# as the Welford estimator would have (`_welford_var`). The step size's dual
# averaging restarts at every window end anyway. Warm-up draws are discarded
# and the adaptation stops before the first kept draw, so the kept chain is
# an ordinary NUTS chain on the final charts.
# =====================================================================

"""
    _HMCRecut(params, transform, rv)

What the chains of one run share for the seam re-cut: the layout (its circular
windows), the target's `PackedTransforms` (`nothing` for a bounded target), the
positions of every angle whose chart moves (`circ`, [`seam_indices`](@ref):
full-circle and polar) and of the polar ones among them (`cinc`, which are also
folded), and the rendezvous of all the chains that adapt together.
"""
struct _HMCRecut
    params::Params
    transform::Union{Nothing,PackedTransforms}
    circ::Vector{Int}
    cinc::Vector{Int}
    rv::_Rendezvous
end
_HMCRecut(params::Params, transform, rv::_Rendezvous) =
    _HMCRecut(params, transform isa PackedTransforms ? transform : nothing,
              seam_indices(params), cyclic_inclination_indices(params), rv)

"""Whether the target has an angle the re-cut handles."""
_has_recut_angles(params::Params) = !isempty(seam_indices(params))

"""
    _hmc_recut_marks(adaptor, n_warmup) -> (marks, ends)

The iterations after whose adaptation step the chains re-cut: the end of the
init buffer and every window end of `adaptor`'s schedule (Stan's), all before
the last warm-up iteration; `ends` are the window ends among them, where the
metric was just set.
"""
function _hmc_recut_marks(adaptor::AdvancedHMC.StanHMCAdaptor, n_warmup::Int)
    st = AdvancedHMC.Adaptation.StanHMCAdaptorState()
    AdvancedHMC.Adaptation.initialize!(st, adaptor.init_buffer, adaptor.term_buffer,
                                       adaptor.window_size, n_warmup)
    ends = filter(b -> 1 <= b < n_warmup, st.window_splits)
    marks = sort!(unique!(filter(b -> 1 <= b < n_warmup, vcat(adaptor.init_buffer, ends))))
    return marks, ends
end

"""
    _HMCRecutChain

One chain's part in the re-cut: the run's `_HMCRecut`, the boundaries, the
bounded values of the re-cut angles it visited since the last one (`trace`,
one vector per angle, full-circle ones first) and the last boundary whose
re-cut it has taken (`recut_at`). `vote`: whether its visits place the seams
(pt_hmc's hot rungs move with the windows but do not choose them). `resync`:
what must run after a window moved, before the chain's log density is called
again (`_nuts_logdensity_resync`), or `nothing`.
"""
mutable struct _HMCRecutChain
    rc::_HMCRecut
    marks::Vector{Int}
    ends::Vector{Int}
    last_mark::Int
    trace::Vector{Vector{Float64}}
    recut_at::Int
    vote::Bool
    resync::Any
end
function _HMCRecutChain(rc::_HMCRecut, adaptor, n_warmup::Int;
                        trace = nothing, recut_at::Int = 0, vote::Bool = true,
                        resync = nothing)
    marks, ends = _hmc_recut_marks(adaptor, n_warmup)
    tr = trace === nothing ? [Float64[] for _ in rc.circ] : trace
    return _HMCRecutChain(rc, marks, ends, isempty(marks) ? 0 : last(marks), tr,
                          recut_at, vote, resync)
end

"""A chain saved at boundary `step` before its re-cut was taken."""
_hmc_recut_pending(cs::_HMCRecutChain, step::Int) = step in cs.marks && cs.recut_at < step

# The angles' coordinates of a target-space point, one at a time.
_hmc_bounded(θ, d::Int, pt) = pt === nothing ? Float64(θ[d]) : _bounded_coord(θ[d], d, pt)
function _hmc_unbounded(x::Float64, d::Int, pt)
    pt === nothing && return x
    tid = pt.type_ids[d]
    tid == TRANSFORM_IDENTITY && return x
    lo = pt.lowers[d]
    tid == TRANSFORM_LOG && return log(max(x - lo, 1e-10))
    t = clamp((x - lo) / (pt.uppers[d] - lo), 1e-10, 1.0 - 1e-10)
    return log(t / (1 - t))
end

"""Record the re-cut angles of target-space position `θ` in the chain's trace."""
function _hmc_trace!(cs::_HMCRecutChain, θ)
    rc = cs.rc
    for (k, d) in enumerate(rc.circ)
        push!(cs.trace[k], _hmc_bounded(θ, d, rc.transform))
    end
    return cs
end

"""The Welford variance estimate AdvancedHMC's `WelfordVar` gives `ys`."""
function _welford_var(ys::AbstractVector{Float64})
    n = length(ys)
    μ = sum(ys) / n
    M = sum(y -> (y - μ)^2, ys)
    return n / ((n + 5) * (n - 1)) * M + 1e-3 * (5 / (n + 5))
end

# The leader's work at a re-cut: every window from the pooled visits of the
# chains that vote (in chain order, so the pool does not depend on who arrived
# last), and each chain's `resync` if one moved, while all of them wait.
# Returns the moved windows as (position, lo, hi).
function _hmc_recut_windows!(rc::_HMCRecut, payloads)
    voters = sort!([p for p in payloads if p.votes !== nothing]; by = p -> p.c)
    moved = Tuple{Int,Float64,Float64}[]
    isempty(voters) && return moved
    tr = rc.transform === nothing ? () : (rc.transform,)
    for (k, d) in enumerate(rc.circ)
        pooled = reduce(vcat, (p.votes[k] for p in voters))
        win = recut_circular_param!(rc.params, d, pooled; transforms = tr)
        win === nothing || push!(moved, (d, win[1], win[2]))
    end
    if !isempty(moved)
        for p in payloads
            p.resync === nothing || p.resync()
        end
    end
    return moved
end

"""
    _hmc_recut!(cs, i, c, z, h, adaptor) -> (moved, z, h) or nothing

Chain `c`'s re-cut after the adaptation step of boundary `i`: meet the other
chains (`nothing` if the rendezvous was aborted, with nothing changed), then
relabel `z`'s position into the moved windows and fold its cyclic
inclinations (into `[0, π]`, which every polar chart holds), re-estimating the
metric of every re-charted coordinate when `i` is a window end. Returns whether anything changed and the chain's new phase
point and Hamiltonian (`h` is rebuilt only if the metric changed, `z` only if
the position or its chart did). Empties the trace.
"""
function _hmc_recut!(cs::_HMCRecutChain, i::Int, c::Int, z, h, adaptor)
    rc = cs.rc
    pt = rc.transform
    nc = length(rc.circ)
    first_phase = i == cs.marks[1]
    votes = cs.vote ?
        [first_phase ? cs.trace[k][(length(cs.trace[k]) ÷ 2 + 1):end] : cs.trace[k]
         for k in 1:nc] : nothing
    # Read before the meeting: the leader moves the windows and the transform.
    x = Float64[_hmc_bounded(z.θ, d, pt) for d in rc.circ]
    moved = _rendezvous!(ps -> _hmc_recut_windows!(rc, ps), rc.rv,
                         (; c, votes, cs.resync))
    moved === _RV_ABORTED && return nothing

    # New bounded values (position and trace) of every coordinate whose value
    # or chart changed.
    win = Dict(d => (lo, hi) for (d, lo, hi) in moved)
    changed = Tuple{Int,Int,Float64,Vector{Float64}}[]   # (k, d, x', trace')
    for (k, d) in enumerate(rc.circ)
        if d in rc.cinc
            # Folded: [0, π] is inside every polar chart, moved or not.
            xf = fold_inclination(x[k])
            tf = Float64[fold_inclination(v) for v in cs.trace[k]]
            (xf == x[k] && tf == cs.trace[k] && !haskey(win, d)) && continue
            push!(changed, (k, d, xf, tf))
        elseif haskey(win, d)
            lo, hi = win[d]
            push!(changed, (k, d, circular_relabel(x[k], lo, hi),
                            Float64[circular_relabel(v, lo, hi) for v in cs.trace[k]]))
        end
    end
    any_change = !isempty(changed)
    if any_change
        if i in cs.ends
            var = adaptor.pc.var
            for (k, d, _, tr) in changed
                length(tr) >= 2 || continue
                var[d] = _welford_var(Float64[_hmc_unbounded(v, d, pt) for v in tr])
            end
            h = AdvancedHMC.update(h, adaptor)
        end
        θ = copy(z.θ)
        for (k, d, xn, _) in changed
            θ[d] = _hmc_unbounded(xn, d, pt)
        end
        z = AdvancedHMC.phasepoint(h, θ, z.r)
    end
    foreach(empty!, cs.trace)
    cs.recut_at = i
    return any_change, z, h
end

"""
    _nuts_recut!(cs, i, c, z, h, κ, adaptor, rng, draws, stats, ckw)
        -> (status, z, h)

A NUTS chain's re-cut at boundary `i` (`_hmc_recut!`), and with a checkpoint a
second meeting at which the last chain writes every chain's state after it,
with the new windows, in one file. Before that write the file holds every
chain before the re-cut and the old windows, after it every chain after the
re-cut and the new ones: never chains in two charts, which one shared target
could not resume. `status`: `:ok`; `:pending` if another chain stopped before
the re-cut, which was not taken (nothing changed); `:lost` if one stopped
between the two meetings (the chain is re-charted, but the file still holds
it before the re-cut).
"""
function _nuts_recut!(cs::_HMCRecutChain, i::Int, c::Int, z, h, κ, adaptor, rng,
                      draws, stats, ckw::Union{Nothing,_NutsCheckpoint})
    r = _hmc_recut!(cs, i, c, z, h, adaptor)
    r === nothing && return (:pending, z, h)
    _, z, h = r
    ckw === nothing && return (:ok, z, h)
    snap = _nuts_snap(i, z, h.metric, κ, adaptor, rng, draws, stats, cs)
    params = cs.rc.params
    w = _rendezvous!(cs.rc.rv, (; c, snap)) do payloads
        lock(ckw.lock) do
            for p in payloads
                ckw.snaps[p.c] = p.snap
            end
            empty!(ckw.windows)
            merge!(ckw.windows, circular_windows(params))
            _nuts_write!(ckw)
        end
        nothing
    end
    return (w === _RV_ABORTED ? :lost : :ok, z, h)
end

"""
    _nuts_chain(target, c, snap, ckw, stop; kwargs...) -> Union{MCMCChains.Chains, Nothing}

Run NUTS chain `c`, on `rng`, from its entry `snap`: from the start point of an
unstarted entry (`_nuts_unstarted`), or from exactly where a saved state of
this chain stopped. Uses the canonical AdvancedHMC idiom: unit diagonal
metric → `find_good_stepsize` → `StanHMCAdaptor` (windowed dual-averaging step
size + Welford diagonal mass-matrix adaptation) → the sampling loop of
`AdvancedHMC.sample(...; drop_warmup = true)`, run here (`_nuts_loop!`) so that
its state can be checkpointed. The per-step `stats` carry divergence flags,
tree depth and adapted step size, summarized into `chains.info`. `ckw` is the
run's checkpoint (`nothing`: none). Returns `nothing` when the test hook
stopped the chain at iteration `stop`.
"""
function _nuts_chain(
    target::NereusTarget, c::Int, snap::NamedTuple, ckw::Union{Nothing,_NutsCheckpoint},
    stop::Int;
    n_samples::Int, n_warmup::Int, target_accept::Float64,
    ad_backend::Symbol,
    rng::AbstractRNG, progress::Bool,
    rc::Union{Nothing,_HMCRecut} = nothing
)
    dim = LogDensityProblems.dimension(target)
    ℓ_fn, ∂ℓ_fn, resync = _nuts_logdensity_resync(target, ad_backend)
    kinetic = AdvancedHMC.GaussianKinetic()

    # `rng` already holds the stream of the chain's entry (`sample_nuts` put
    # it back on a resume).
    if !_nuts_started(snap)
        init_bounded = snap.init::Vector{Float64}
        length(init_bounded) == dim || throw(ArgumentError(
            "init length $(length(init_bounded)) ≠ target dimension $dim"))

        # Transform the bounded-space init to unconstrained space if needed.
        init_y = if target.transform isa PackedTransforms
            transform_forward(init_bounded, target.transform)
        else
            copy(init_bounded)
        end

        # --- Metric: UNIT diagonal mass matrix ------------------------
        # Start from identity and let the StanHMCAdaptor's windowed Welford
        # estimator learn the per-dimension scales from the chain itself.
        # (The old `1/g²`-at-init metric was degenerate: at a prior-draw
        # init the gradients are ~5000, so inv_mass ~ 1e-8 → infinite mass →
        # frozen chain.)
        metric = AdvancedHMC.DiagEuclideanMetric(dim)
        hamiltonian = AdvancedHMC.Hamiltonian(metric, kinetic, ℓ_fn, ∂ℓ_fn)

        # --- Step size: heuristic initial ε, then dual-averaging ------
        initial_ε = AdvancedHMC.find_good_stepsize(rng, hamiltonian, init_y)
        integrator = AdvancedHMC.Leapfrog(initial_ε)
        kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{
            AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))

        # Windowed adaptation: dual-averaging step size + diagonal mass
        # matrix (the canonical Stan warmup schedule).
        adaptor = AdvancedHMC.StanHMCAdaptor(
            AdvancedHMC.MassMatrixAdaptor(metric),
            AdvancedHMC.StepSizeAdaptor(target_accept, integrator))

        # The starting phase point, drawn as `AdvancedHMC.sample` draws it:
        # its momentum is refreshed before use, but the draw advances `rng`.
        hamiltonian, t0 = AdvancedHMC.sample_init(rng, hamiltonian, init_y)
        z = t0.z
        step0 = 0
        draws = Vector{Vector{Float64}}()
        stats = Vector{NamedTuple}()
    else
        # --- Resume: the saved chain, exactly as it stopped ------------
        # Copied, not used as read: `snap` stays the chain's checkpoint entry
        # until its next save, and the loop mutates the adaptor in place.
        live = deepcopy((; snap.z, snap.metric, snap.kernel, snap.adaptor))
        hamiltonian = AdvancedHMC.Hamiltonian(live.metric, kinetic, ℓ_fn, ∂ℓ_fn)
        kernel = live.kernel
        adaptor = live.adaptor
        z = live.z
        step0 = snap.step::Int
        draws = copy(snap.draws::Vector{Vector{Float64}})
        stats = copy(snap.stats::Vector{NamedTuple})
    end
    # The seam re-cut's state: fresh, or the trace and last re-cut saved.
    cs = rc === nothing ? nothing :
         !_nuts_started(snap) ? _HMCRecutChain(rc, adaptor, n_warmup; resync) :
         _HMCRecutChain(rc, adaptor, n_warmup; trace = deepcopy(snap.circ.trace),
                        recut_at = snap.circ.recut_at::Int, resync)

    n_total = n_warmup + n_samples
    sizehint!(draws, n_samples); sizehint!(stats, n_samples)
    pb = ProgressBar("NUTS"; total = n_total, enabled = progress, start = step0)
    done = _nuts_loop!(rng, hamiltonian, kernel, adaptor, z, draws, stats, step0,
                       n_total, n_warmup, c, ckw, stop, pb, cs)
    finish!(pb)
    done || return nothing

    # --- Back-transform to bounded (physical) space -------------------
    post_samples = if target.transform isa PackedTransforms
        [transform_inverse(s, target.transform) for s in draws]
    else
        draws
    end

    param_names = Symbol.(target.params.layout.unfrozen_names)
    n_post = length(post_samples)
    mat = Matrix{Float64}(undef, n_post, dim)
    @inbounds for i in 1:n_post, j in 1:dim
        mat[i, j] = post_samples[i][j]
    end

    # Log-density column (post-warmup) for downstream plotting / the
    # logpost-sanity check in assess_fit.
    if !isempty(stats) && haskey(stats[1], :log_density)
        lp = [Float64(s.log_density) for s in stats]
        mat = hcat(mat, lp)
        push!(param_names, :lp)
    end

    chains = MCMCChains.Chains(mat, param_names)

    # --- Per-chain NUTS health → chains.info (OBSERVABILITY) ----------
    # Divergences (numerical_error), adapted step size, tree depth and
    # acceptance summarized from the post-warmup stats. These make
    # non-convergence detectable without re-deriving it from the trace.
    n_div = count(s -> get(s, :numerical_error, false), stats)
    fin_ε = isempty(stats) ? NaN : Float64(last(stats).step_size)
    tds   = [Float64(get(s, :tree_depth, NaN)) for s in stats]
    accs  = [Float64(get(s, :acceptance_rate, NaN)) for s in stats]
    chains = setinfo(chains, (
        n_divergent     = n_div,
        step_size       = fin_ε,
        mean_tree_depth = isempty(tds) ? NaN : mean(filter(isfinite, tds)),
        max_tree_depth  = isempty(tds) ? NaN :
                          (all(isnan, tds) ? NaN : maximum(filter(isfinite, tds))),
        mean_accept     = isempty(accs) ? NaN : mean(filter(isfinite, accs)),
    ))
    return chains
end

"""
    _nuts_loop!(rng, h, κ, adaptor, z, draws, stats, step0, n_total, n_warmup,
                c, ckw, stop, pb, cs) -> Bool

Iterations `step0 + 1 : n_total` of one NUTS chain: the loop of
`AdvancedHMC.sample(rng, h, κ, θ, n_total, adaptor, n_warmup; drop_warmup =
true)`, step for step and draw for draw, appending the post-warmup positions
and statistics to `draws` and `stats`, plus the seam re-cut at the phase
boundaries of the adaptation (`cs`, a `_HMCRecutChain`, `nothing` without
angles to re-cut; see `_hmc_recut!`). Between iterations the chain is `z`'s
position, `rng`, the metric of `h`, the step size of `κ`, the adaptor's state
and `cs`'s trace, all of which a checkpoint saves. Returns `false` if the test
hook stopped it at `stop`, or another chain stopped before a re-cut that needs
them all.
"""
function _nuts_loop!(rng::AbstractRNG, h, κ, adaptor, z,
                     draws::Vector{Vector{Float64}}, stats::Vector{NamedTuple},
                     step0::Int, n_total::Int, n_warmup::Int, c::Int,
                     ckw::Union{Nothing,_NutsCheckpoint}, stop::Int, pb::ProgressBar,
                     cs::Union{Nothing,_HMCRecutChain})
    last_ck = time()
    # Saved at a boundary whose re-cut the other chains had not reached (the
    # round was aborted, `_nuts_recut!`): the re-cut first, exactly where the
    # uninterrupted run took it, after that iteration's adaptation.
    if cs !== nothing && _hmc_recut_pending(cs, step0)
        st, z, h = _nuts_recut!(cs, step0, c, z, h, κ, adaptor, rng, draws, stats, ckw)
        st === :ok || return false   # its entry is still the one it resumed from
        if step0 == stop
            ckw === nothing ||
                _nuts_snapshot!(ckw, c, step0, z, h.metric, κ, adaptor, rng, draws, stats, cs)
            return false
        end
    end
    for i in (step0 + 1):n_total
        t = AdvancedHMC.transition(rng, h, κ, z)
        tstat = AdvancedHMC.stat(t)
        h, κ, isadapted = AdvancedHMC.Adaptation.adapt!(
            h, κ, adaptor, i, n_warmup, t.z.θ, tstat.acceptance_rate)
        z = t.z
        if i > n_warmup
            push!(draws, z.θ)
            push!(stats, merge(tstat, (is_adapt = isadapted,)))
        end
        update!(pb; n_done = i,
                fields = (:phase => isadapted ? "warmup" : "sampling",
                          :ε => AdvancedHMC.nom_step_size(κ.τ.integrator),
                          :accept => tstat.acceptance_rate))

        # ---- Seam re-cut at the adaptation's phase boundaries -------------
        if cs !== nothing && i <= cs.last_mark
            _hmc_trace!(cs, z.θ)
            if i in cs.marks
                st, z, h = _nuts_recut!(cs, i, c, z, h, κ, adaptor, rng, draws, stats, ckw)
                if st === :pending
                    # Another chain stopped first, so the re-cut cannot be
                    # taken. Saved as it stands, the re-cut pending: the
                    # checkpoint's other entries and windows are of this chart.
                    ckw === nothing ||
                        _nuts_snapshot!(ckw, c, i, z, h.metric, κ, adaptor, rng, draws,
                                        stats, cs)
                    return false
                end
                # :lost -- re-charted, but the others never saved theirs: the
                # checkpoint keeps every chain before this re-cut.
                st === :lost && return false
            end
        end

        # ---- Checkpoint: every `checkpoint_interval` s, on the last
        # iteration, and before the test hook's stop (src/checkpoint.jl) ---
        if ckw !== nothing && (i == n_total || i == stop ||
                               time() - last_ck >= ckw.interval)
            _nuts_snapshot!(ckw, c, i, z, h.metric, κ, adaptor, rng, draws, stats, cs)
            last_ck = time()
        end
        i == stop && return false
    end
    return true
end
