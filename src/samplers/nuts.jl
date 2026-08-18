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
#       ptemcee global pre-search so the unimodal/eccentric targets land
#       in the dominant basin and mix (honest R-hat < 1.1). On genuinely
#       multimodal targets warm-start scatters chains across the modes,
#       cross-chain R-hat stays high, and `assess_fit`'s multimodality
#       check fires — NUTS FAILS LOUD instead of silently merging.
#
# OBSERVABILITY: the returned `Chains` carries per-chain NUTS health
# (divergence count, final adapted step size, mean/max tree depth, mean
# acceptance) in `chains.info` so non-convergence is detectable without
# re-deriving it from the trace. See `nuts_diagnostics(chains)`.

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
    llt = isfinite(llr) ? transit_log_likelihood(theta_dbg, target.data) : NaN

    x_med = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ps = layout.unfrozen_priors[i]
        val = quantile(ps, 0.5)
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

Run a short ptemcee global pre-search and return `n_chains` HIGH-lp
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
model shape ptemcee cannot init) so NUTS still runs — just without the
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
        res = sample_ptemcee(target, target.data;
                             n_temps = n_temps, n_walkers = n_walkers_eff,
                             n_steps = n_steps, n_burnin = n_burnin,
                             init_strategy = :prior, seed = seed,
                             show_progress = false)
        ch = res.chains
        pnames = String.(names(ch, :parameters))
        # `Array(ch)` collapses the walker axis into the iteration axis,
        # giving a flat (n_draw × n_param) matrix — exactly the pooled
        # cold-ensemble sample list we want to rank by :lp. Each row is a
        # bounded-space parameter vector; reorder columns to align with
        # layout.unfrozen_names.
        slot_idx = [findfirst(==(nm), pnames) for nm in layout.unfrozen_names]
        any(isnothing, slot_idx) && error(
            "ptemcee warm-start chains missing a fitted slot")
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
        isempty(cand) && error("ptemcee warm-start produced no finite draws")
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
Chains are warm-started from a short ptemcee global pre-search so the
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
  params / GP models), or `:ReverseDiff` (requires `import ReverseDiff`)
- `compile_tape::Bool=true` : compile ReverseDiff tape for ~2-3x speedup
  (only applies when `ad_backend=:ReverseDiff`)
- `warm_start::Bool=true`  : warm-start chains from a short ptemcee
  pre-search. Disable only if you pass `init` or know the target is
  trivially unimodal; with it off, independent prior-draw inits will
  freeze NUTS in disjoint modes on multi-planet RV posteriors.
- `warm_temps`, `warm_walkers`, `warm_steps`, `warm_burnin` : ptemcee
  pre-search budget (defaults 6 / 40 / 400 / 200).
- `init::Union{Nothing, Vector{Float64}}=nothing` : initial position in
  **bounded** space (overrides `warm_start`). Transformed to
  unconstrained internally if the target uses transforms.
- `rng::AbstractRNG=Random.default_rng()`
- `progress::Bool=true`
"""
function sample_nuts(
    target::NereusTarget;
    n_samples::Int = 1000,
    n_warmup::Int = 1000,
    n_chains::Int = 4,
    target_accept::Real = 0.8,
    ad_backend::Symbol = :ForwardDiff,
    compile_tape::Bool = true,
    warm_start::Bool = true,
    warm_temps::Int = 6,
    warm_walkers::Int = 40,
    warm_steps::Int = 400,
    warm_burnin::Int = 200,
    init::Union{Nothing, Vector{Float64}} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    progress::Bool = true,
    kwargs...
)
    target_accept = Float64(target_accept)   # JSON may deliver an Int

    # Per-chain bounded-space init points. `init` (if given) pins every
    # chain at the same point; otherwise warm-start from a short ptemcee
    # pre-search (the disjoint-mode fix), or fall back to independent
    # prior draws when warm_start is off.
    init_points = if init !== nothing
        [copy(init) for _ in 1:n_chains]
    elseif warm_start
        _warmstart_points(target, n_chains, rng;
                          n_temps = warm_temps, n_walkers = warm_walkers,
                          n_steps = warm_steps, n_burnin = warm_burnin)
    else
        [_draw_from_prior(target, rng) for _ in 1:n_chains]
    end

    if n_chains == 1
        return _run_single_chain(
            target; n_samples, n_warmup, target_accept, ad_backend,
            compile_tape, init_bounded = init_points[1], rng, progress,
            kwargs...)
    else
        return _run_multi_chain(
            target, n_chains; n_samples, n_warmup, target_accept,
            ad_backend, compile_tape, init_points, rng, progress, kwargs...)
    end
end

"""
    _run_multi_chain(target, n_chains; kwargs...) -> MCMCChains.Chains

Run `n_chains` NUTS chains in parallel using Julia threads, then merge
into a single `Chains` object with chain IDs. Each chain starts from its
own warm-start point (`init_points[c]`). Per-chain NUTS diagnostics are
combined into vectors and attached to the merged `chains.info`. Requires
Julia started with multiple threads (`julia -t N`).
"""
function _run_multi_chain(
    target::NereusTarget, n_chains::Int;
    n_samples, n_warmup, target_accept, ad_backend, compile_tape,
    init_points, rng, progress, kwargs...
)
    # Generate per-chain RNG seeds from the base RNG.
    seeds = [rand(rng, UInt64) for _ in 1:n_chains]

    # Run chains in parallel. Only show progress on chain 1. `:static`
    # so the tasks do not migrate threads mid-run (per the threadid
    # buffer-race lesson elsewhere in this codebase); each task owns its
    # own NereusTarget AD wrapper + RNG so there is no shared state.
    tasks = Vector{Task}(undef, n_chains)
    for c in 1:n_chains
        chain_rng = MersenneTwister(seeds[c])
        show_progress = progress && (c == 1)
        ip = init_points[c]
        tasks[c] = Threads.@spawn _run_single_chain(
            target; n_samples, n_warmup, target_accept, ad_backend,
            compile_tape, init_bounded = ip, rng = chain_rng,
            progress = show_progress, kwargs...)
    end

    chains_list = [fetch(t) for t in tasks]

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

"""
    _run_single_chain(target; kwargs...) -> MCMCChains.Chains

Run a single NUTS chain. Internal workhorse called by `sample_nuts`.
Uses the canonical AdvancedHMC idiom: unit diagonal metric →
`find_good_stepsize` → `StanHMCAdaptor` (windowed dual-averaging step
size + Welford diagonal mass-matrix adaptation) → high-level `sample`
with `drop_warmup=true`. The per-step `stats` carry divergence flags,
tree depth and adapted step size, summarized into `chains.info`.
"""
function _run_single_chain(
    target::NereusTarget;
    n_samples::Int, n_warmup::Int, target_accept::Float64,
    ad_backend::Symbol, compile_tape::Bool,
    init_bounded::Vector{Float64},
    rng::AbstractRNG, progress::Bool, kwargs...
)
    dim = LogDensityProblems.dimension(target)
    length(init_bounded) == dim || throw(ArgumentError(
        "init length $(length(init_bounded)) ≠ target dimension $dim"))

    # Transform the bounded-space init to unconstrained space if needed.
    init_y = if target.transform isa PackedTransforms
        transform_forward(init_bounded, target.transform)
    else
        copy(init_bounded)
    end

    # --- AD gradient wrapper ------------------------------------------
    if ad_backend === :Enzyme && target.transform isa PackedTransforms
        # Custom Enzyme path with explicit Const arguments.
        enzyme_cfg = EnzymeGradientConfig(target)
        ℓ_fn = y -> enzyme_logdensity_and_gradient!(enzyme_cfg, target, y)[1]
        ∂ℓ_fn = y -> enzyme_logdensity_and_gradient!(enzyme_cfg, target, y)
    else
        ad_kwargs = Dict{Symbol,Any}()
        if ad_backend === :ReverseDiff && compile_tape
            ad_kwargs[:compile] = Val(true)
        end
        ℓ_ad = LogDensityProblemsAD.ADgradient(ad_backend, target; ad_kwargs...)
        ℓ_fn = y -> LogDensityProblems.logdensity(ℓ_ad, y)
        ∂ℓ_fn = y -> LogDensityProblems.logdensity_and_gradient(ℓ_ad, y)
    end

    # --- Metric: UNIT diagonal mass matrix ----------------------------
    # Start from identity and let the StanHMCAdaptor's windowed Welford
    # estimator learn the per-dimension scales from the chain itself.
    # (The old `1/g²`-at-init metric was degenerate: at a prior-draw
    # init the gradients are ~5000, so inv_mass ~ 1e-8 → infinite mass →
    # frozen chain.)
    metric = AdvancedHMC.DiagEuclideanMetric(dim)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, AdvancedHMC.GaussianKinetic(),
                                          ℓ_fn, ∂ℓ_fn)

    # --- Step size: heuristic initial ε, then dual-averaging ----------
    initial_ε = AdvancedHMC.find_good_stepsize(rng, hamiltonian, init_y)
    integrator = AdvancedHMC.Leapfrog(initial_ε)
    kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{
        AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))

    # Windowed adaptation: dual-averaging step size + diagonal mass
    # matrix (the canonical Stan warmup schedule).
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric),
        AdvancedHMC.StepSizeAdaptor(target_accept, integrator))

    # --- Sample (high-level API drives adaptation correctly) ----------
    samples_y, stats = AdvancedHMC.sample(
        rng, hamiltonian, kernel, init_y, n_warmup + n_samples,
        adaptor, n_warmup; drop_warmup = true, progress = progress,
        verbose = false)

    # --- Back-transform to bounded (physical) space -------------------
    post_samples = if target.transform isa PackedTransforms
        [transform_inverse(s, target.transform) for s in samples_y]
    else
        samples_y
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
