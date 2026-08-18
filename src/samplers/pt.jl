# Parallel tempering driver via Pigeons.jl with evidence estimation.

using Pigeons
using MCMCChains
using Random

# --- Pigeons interface for NereusTarget ----------------------------

# Make NereusTarget callable (Pigeons log_potential interface).
(target::NereusTarget)(x) = LogDensityProblems.logdensity(target, x)

# Initial state: draw from prior, transform to unconstrained space.
function Pigeons.initialization(target::NereusTarget, rng::AbstractRNG, ::Int)
    x_bounded = _draw_from_prior(target, rng)
    if target.transform isa PackedTransforms
        return transform_forward(x_bounded, target.transform)
    else
        return x_bounded
    end
end

# Reference: auto-tuned Gaussian. Pre-seed with zeros/ones so the
# variational reference is valid before the first tuning round.
function Pigeons.default_reference(target::NereusTarget)
    dim = LogDensityProblems.dimension(target)
    return GaussianReference(
        mean = Dict{Symbol,Any}(:singleton_variable => zeros(dim)),
        standard_deviation = Dict{Symbol,Any}(:singleton_variable => ones(dim)),
        first_tuning_round = 3,
    )
end

# Use SliceSampler (gradient-free, robust) as default explorer.
Pigeons.default_explorer(::NereusTarget) = SliceSampler()

# --- Multi-init wrapper --------------------------------------------
#
# Wraps a NereusTarget with a per-replica initialization matrix
# (n_dim × n_chains, in unconstrained sampler space). When passed to
# `pigeons(...)`, replica `c` is initialized at column `c` of the matrix
# rather than from a prior draw. Used to seed Pigeons walkers with
# Pathfinder draws (or any external warmup), which is the fix for the
# multimodal-posterior pathology where prior-init PT gets stuck in the
# nearest basin.

"""
    InitializedTarget{T<:NereusTarget}

Wrapper that overrides `Pigeons.initialization` to return columns of a
pre-computed `inits` matrix (n_dim × n_chains) instead of prior draws.
All other LogDensityProblems / Pigeons interface calls delegate to the
inner `NereusTarget`.
"""
struct InitializedTarget{T<:NereusTarget}
    inner::T
    inits::Matrix{Float64}     # n_dim × n_chains, unconstrained space
end

(t::InitializedTarget)(x) = LogDensityProblems.logdensity(t.inner, x)
LogDensityProblems.logdensity(t::InitializedTarget, x) =
    LogDensityProblems.logdensity(t.inner, x)
LogDensityProblems.dimension(t::InitializedTarget) =
    LogDensityProblems.dimension(t.inner)
LogDensityProblems.capabilities(::Type{<:InitializedTarget}) =
    LogDensityProblems.LogDensityOrder{0}()

function Pigeons.initialization(t::InitializedTarget, rng::AbstractRNG,
                                replica_idx::Int)
    n_chains = size(t.inits, 2)
    col = mod1(replica_idx, n_chains)
    return Vector{Float64}(t.inits[:, col])
end

Pigeons.default_reference(t::InitializedTarget) =
    Pigeons.default_reference(t.inner)
Pigeons.default_explorer(::InitializedTarget) = SliceSampler()

"""
    sample_pt(target::NereusTarget; kwargs...) -> (chains, log_evidence)

Run parallel tempering on a `NereusTarget` using Pigeons.jl. Returns
posterior as `MCMCChains.Chains` and the stepping-stone log-evidence
estimate.

# Keywords
- `n_rounds::Int=15`     : PT adaptation rounds (samples double each round)
- `n_chains::Int=10`     : number of tempering chains
- `seed::Int=1`          : random seed
- `show_report::Bool=true`
- `td::Union{TransDimConfig, Nothing}=nothing` : trans-dim config (opt-in)
- `init::Union{Nothing, AbstractMatrix{<:Real}, NamedTuple}=nothing` :
  per-replica initial points in unconstrained sampler space. Pass an
  `n_dim × n_chains` matrix, or the NamedTuple returned by
  [`pathfinder_init`](@ref) (its `.draws` field is used; columns are
  cycled if fewer than `n_chains`). When `nothing` (default), Pigeons
  initializes each replica with a fresh prior draw — fine for unimodal
  posteriors, but gets stuck in the nearest basin for multimodal
  problems (HD 159062-class). Multi-init via Pathfinder fixes this.
"""
function sample_pt(
    target::NereusTarget;
    n_rounds::Int = 15,
    n_chains::Int = 10,
    seed::Int = 1,
    show_report::Bool = true,
    td::Union{TransDimConfig, Nothing} = nothing,
    init::Union{Nothing, AbstractMatrix{<:Real}, NamedTuple} = nothing,
    early_stop_thresh::Real = 0.0,
    early_stop_min_rounds::Int = 8,
    within_model::Symbol = :slice,
    backend::Symbol = :nereus,
    explorer::Symbol = :slice,
)
    early_stop_thresh = Float64(early_stop_thresh)   # JSON may deliver an Int
    # PT parallelises chain updates via Threads.@threads. With nthreads()
    # == 1 the @threads loop runs serially, so n_chains chains take
    # n_chains× longer than they should — easily a >10× hit on real-data
    # runs. Warn loudly so users restart with `julia -t auto` instead of
    # discovering this 6 hours into a multi-day run.
    if Threads.nthreads() == 1 && n_chains > 1
        @warn "sample_pt: Julia is running single-threaded but you are using $(n_chains) chains. " *
              "PT chains parallelise via Threads.@threads — with 1 thread they will run serially " *
              "(>10× slower). Restart Julia with `julia -t auto` (or `JULIA_NUM_THREADS=auto`) to fix."
    end
    backend in (:nereus, :pigeons) || throw(ArgumentError(
        "backend must be :nereus (in-house, default) or :pigeons; got :$backend"))

    init_mat_for_transdim() = if init === nothing
        nothing
    elseif init isa NamedTuple
        haskey(init, :draws) || throw(ArgumentError(
            "init NamedTuple must have a :draws field (got keys $(keys(init)))"))
        Matrix{Float64}(init.draws)
    else
        Matrix{Float64}(init)
    end

    # In-house PT (default). Two cases:
    #   - User supplied a TransDimConfig → trans-dim run, as before.
    #   - User left td === nothing → fixed-dim run; build a "null"
    #     config so the optimised zero-alloc threaded RWM hot path
    #     runs as a plain fixed-dim PT (no birth/death moves).
    # The in-house path is ~10× faster than Pigeons on Nereus targets
    # because Pigeons' slice-sampling explore!() loop is not
    # parallelised by default. Backend :pigeons is retained for
    # cross-checks but never the default.
    if backend === :nereus
        td_user_supplied = td !== nothing
        td_eff = td_user_supplied ? td :
                  TransDimConfig(;
                       max_kplanet       = target.params.config.max_kplanet,
                       planets           = false,
                       noise             = false,
                       transdim_fraction = 0.0,
                  )
        chains, log_ev, n_evals = _sample_pt_transdim(target, td_eff;
            n_rounds=n_rounds, n_chains=n_chains, seed=seed,
            show_report=show_report,
            early_stop_thresh=early_stop_thresh,
            early_stop_min_rounds=early_stop_min_rounds,
            init=init_mat_for_transdim(),
            within_model=within_model)
        # Preserve historical return shapes: 2-tuple for fixed-dim
        # (matches the old Pigeons-backed path); 3-tuple for trans-dim
        # (matches the old in-house path). Test scripts depend on this.
        return td_user_supplied ? (chains, log_ev, n_evals) : (chains, log_ev)
    end

    # ---- backend = :pigeons ----
    # Retained for cross-checks only; no trans-dim support here.
    td === nothing || throw(ArgumentError(
        "backend=:pigeons does not support td; use backend=:nereus (default) for trans-dim."))

    pigeons_target = if init === nothing
        target
    else
        inits_mat = if init isa NamedTuple
            haskey(init, :draws) || throw(ArgumentError(
                "init NamedTuple must have a :draws field (got keys $(keys(init)))"))
            Matrix{Float64}(init.draws)
        else
            Matrix{Float64}(init)
        end
        n_dim = LogDensityProblems.dimension(target)
        size(inits_mat, 1) == n_dim || throw(ArgumentError(
            "init matrix has $(size(inits_mat,1)) rows, expected $n_dim (n_dim of target)"))
        size(inits_mat, 2) >= 1 || throw(ArgumentError(
            "init matrix must have at least one column"))
        InitializedTarget(target, inits_mat)
    end

    # Pigeons per-temperature explorer. `:slice` (default) is gradient-free,
    # robust to hard prior walls, and matches Nereus's traditional choice.
    # `:automala` uses AutoMALA — an adaptive MALA/HMC-flavored explorer —
    # which leverages the ForwardDiff gradients Nereus targets already
    # expose. Good for smooth high-d posteriors; loses on multimodal /
    # sharp-ridge targets where slice mixes better.
    pigeons_explorer = if explorer === :slice
        Pigeons.SliceSampler()
    elseif explorer === :automala
        Pigeons.AutoMALA()
    else
        throw(ArgumentError("explorer must be :slice or :automala; got :$explorer"))
    end

    pt = pigeons(;
        target = pigeons_target,
        n_rounds = n_rounds,
        n_chains = n_chains,
        seed = seed,
        explorer = pigeons_explorer,
        record = [Pigeons.traces; record_default(); round_trip],
        show_report = show_report,
    )

    log_ev = stepping_stone(pt)

    # Extract posterior samples from the target chain and
    # back-transform to bounded (physical) space.
    raw_samples = sample_array(pt)  # (n_samples, dim+1, n_chains) — last col is log_density
    n_post = size(raw_samples, 1)
    dim = LogDensityProblems.dimension(target)
    param_names = Symbol.(target.params.layout.unfrozen_names)

    mat = Matrix{Float64}(undef, n_post, dim)
    for i in 1:n_post
        y = Vector{Float64}(raw_samples[i, 1:dim, 1])
        if target.transform isa PackedTransforms
            mat[i, :] = transform_inverse(y, target.transform)
        else
            mat[i, :] = y
        end
    end

    # Append log_density column
    lp = Float64.(raw_samples[:, end, 1])
    mat = hcat(mat, lp)
    push!(param_names, :lp)

    chains = MCMCChains.Chains(mat, param_names)
    return chains, log_ev
end

"""
    sample_pt_warm(target::NereusTarget; pathfinder_kwargs..., pt_kwargs...)
        -> (chains, log_evidence, pathfinder_result)

Convenience wrapper: runs [`pathfinder_init`](@ref) first to produce
multi-basin initial points, then `sample_pt` with `init=` set to the
Pathfinder draws. This is the recommended path for multimodal
posteriors (long-period astrometric binaries, under-constrained
visual orbits, etc.) where prior-init PT gets stuck in one basin.

# Keywords
- `n_pathfinder_runs::Int=16` — number of independent L-BFGS Pathfinder
  runs. Higher → better mode coverage. 16 is a good default for
  HD 159062-class problems; bump to 32+ for very rough landscapes.
- `n_pathfinder_draws::Int=max(2*n_chains, 200)` — total draws after
  Pathfinder ESS reweighting; must be ≥ `n_chains`.
- `backend::Symbol=:auto` — which PT engine to run.
  * `:auto` (default) → `:ptemcee` for fixed-dim runs (`td === nothing`),
    `:nereus` for trans-dim runs (`td !== nothing`).
  * `:ptemcee` → prior-init parallel-tempered ensemble (`sample_ptemcee`),
    with NO Pathfinder warm-start. The prior-dispersed ensemble + stretch
    moves navigate sharp, curved astrometric ridges and recover the orbit
    at threaded speed (HD 159062: a≈57, R̂<1.05, ESS≫100, ~0.4 min).
    Pathfinder is deliberately dropped here: its MVN approximation is poor
    on such ridges (Pareto k ≫ 0.5) and seeds every walker into one
    spurious basin, *false-converging* the ensemble to a wrong orbit with
    pristine-looking diagnostics (HD 159062 Pathfinder-seeded: a=34.5 vs
    true ≈58). Fixed-dim only. This is the recommended default.
  * `:pigeons` → seed Pigeons' adaptively-tempered PT with the Pathfinder
    draws (unconstrained space). It mixes, but is single-threaded for
    `NereusTarget` (~8× slower) and, on HD 159062, non-reproducible across
    seeds (runs landed at a = 115 / 41 / 78 / 64). Kept as an explicit
    opt-in; fixed-dim only.
  * `:nereus` → seed the in-house RWM/slice PT with the draws transformed
    to bounded space. Required for trans-dim (`td !== nothing`); its fixed
    quadratic β-ladder and single-coordinate explorer do NOT navigate
    ultra-sharp curved ridges (cold chain decouples across the β=0.79→1.0
    gap and freezes at the seed → HD 159062 a ≈ 115 vs true ≈ 61). Use for
    trans-dim.
- ptemcee-path keywords (used when `backend` resolves to `:ptemcee`):
  `n_temps=16`, `n_walkers=44`, `n_steps=8000`, `n_burnin=4000`,
  `adapt_ladder=true`; `init_strategy=:prior` is fixed. `sample_ptemcee`
  auto-raises `n_walkers` to `2·n_dim+2` — scale it with dimension for
  high-d fixed-dim fits. Returns `(chains, log_evidence, nothing)`.
- `:pigeons`/`:nereus` keywords pass through to `sample_pt`
  (`n_pathfinder_runs`, `n_rounds`, `n_chains`, `seed`, `show_report`).
"""
function sample_pt_warm(
    target::NereusTarget;
    n_pathfinder_runs::Int = 16,
    n_pathfinder_draws::Int = 0,
    n_rounds::Int = 15,
    n_chains::Int = 10,
    seed::Int = 1,
    show_report::Bool = true,
    td::Union{TransDimConfig, Nothing} = nothing,
    early_stop_thresh::Real = 0.0,
    early_stop_min_rounds::Int = 8,
    within_model::Symbol = :slice,
    backend::Symbol = :auto,
    init_strategy::Symbol = :pathfinder,
    n_temps::Int = 16,
    n_walkers::Int = 44,
    n_steps::Int = 8000,
    n_burnin::Int = 4000,
    adapt_ladder::Bool = true,
)
    early_stop_thresh = Float64(early_stop_thresh)   # JSON may deliver an Int
    n_draws = n_pathfinder_draws == 0 ? max(2 * n_chains, 200) : n_pathfinder_draws
    n_draws >= n_chains || throw(ArgumentError(
        "n_pathfinder_draws ($n_draws) must be ≥ n_chains ($n_chains)"))

    # Resolve the PT engine. Default (:auto): prior-init parallel-tempered
    # ensemble (ptemcee) for fixed-dim, in-house RJMCMC-PT for trans-dim.
    # Neither :ptemcee nor :pigeons can do trans-dim, so a trans-dim request
    # must use :nereus.
    backend in (:auto, :ptemcee, :pigeons, :nereus) || throw(ArgumentError(
        "backend must be :auto, :ptemcee, :pigeons, or :nereus; got :$backend"))
    init_strategy in (:pathfinder, :prior) || throw(ArgumentError(
        "init_strategy must be :pathfinder or :prior; got :$init_strategy"))
    backend_eff = backend === :auto ?
                    (td === nothing ? :ptemcee : :nereus) : backend
    if backend_eff in (:ptemcee, :pigeons) && td !== nothing
        throw(ArgumentError(
            "sample_pt_warm: backend=:$backend_eff does not support trans-dim (td); " *
            "use backend=:nereus (or :auto) for trans-dim runs."))
    end

    # --- Fixed-dim default: prior-init parallel-tempered ensemble --------
    # Deliberately NO Pathfinder here. On sharp, curved astrometric
    # posteriors the Pathfinder MVN warm-start *false-converges* the
    # ensemble — its approximation is poor (Pareto k ≫ 0.5) and seeds every
    # walker into one spurious basin, producing pristine-looking R̂/ESS at a
    # wrong orbit (HD 159062 Pathfinder-seeded: a=34.5 vs true ≈58). The
    # prior-dispersed ensemble + stretch moves instead navigate the ridge
    # and recover robustly at threaded speed (HD 159062: a≈57, R̂<1.05,
    # ~0.4 min). `sample_ptemcee` auto-raises n_walkers to ≥ 2·n_dim+2.
    # Returns the historical fixed-dim 3-tuple (chains, log_evidence, pf)
    # with pf = nothing (no Pathfinder run on this path).
    if backend_eff === :ptemcee
        res = sample_ptemcee(target, target.data;
                             n_temps = n_temps, n_walkers = n_walkers,
                             n_steps = n_steps, n_burnin = n_burnin,
                             adapt_ladder = adapt_ladder,
                             init_strategy = :prior, seed = seed,
                             show_progress = show_report)
        return res.chains, res.log_evidence, nothing
    end

    # --- init_strategy=:prior — skip Pathfinder, seed chains from the prior --
    # Pathfinder is the dominant cost on data-rich trans-dim targets: n_runs
    # L-BFGS basins, each driving ForwardDiff gradients through the SERIAL,
    # un-cached Dual photometry path (~155k points × every L-BFGS iter). It
    # also collapses chains toward one MVN basin (Pareto k > 1) and seeds all
    # planets active (suppressing the trans-dim birth/death that discovers weak
    # signals). Prior-dispersed init is faster AND finds weak planets — the
    # same call the fixed-dim :ptemcee path already makes. init=nothing makes
    # _sample_pt_transdim cold-start from the prior (all_active=false).
    if init_strategy === :prior || n_pathfinder_runs <= 0
        pt_target = if backend_eff === :pigeons
            target.transform === nothing ?
                NereusTarget(target.params, target.data; unconstrained = true) : target
        else
            target
        end
        pt_result = sample_pt(pt_target;
                               td = td, n_rounds = n_rounds, n_chains = n_chains,
                               seed = seed, show_report = show_report, init = nothing,
                               backend = backend_eff,
                               early_stop_thresh = early_stop_thresh,
                               early_stop_min_rounds = early_stop_min_rounds,
                               within_model = within_model)
        if td === nothing
            chains, log_ev = pt_result
            return chains, log_ev, nothing
        else
            chains, log_ev, n_evals = pt_result
            return chains, log_ev, n_evals, nothing
        end
    end

    # Pathfinder *must* run in unconstrained space. Its LBFGS-fitted
    # MVN approximations don't respect [lo, hi] bounds — when fed a
    # bounded-space target, the MVN draws routinely land outside
    # support, return -Inf log-density, and the PSIS reweighting step
    # crashes with `sum(weights) == NaN` once enough draws are -Inf
    # (the normalisation does `-Inf − -Inf = NaN`). NOI-106823 on
    # `pt_warm` hit this consistently with a bounded target from
    # `runner.jl`. The fix is to construct an unconstrained copy of
    # the target for the warm-start, then transform draws back to
    # bounded space before seeding PT.
    pf_target = target.transform === nothing ?
                  NereusTarget(target.params, target.data;
                                unconstrained = true) :
                  target
    pf = pathfinder_init(pf_target;
                         n_runs = n_pathfinder_runs,
                         n_draws = n_draws,
                         seed = seed)

    # Seed PT chains from the per-run Pathfinder draws (one draw per
    # L-BFGS basin) rather than from the IS-reweighted pool. With
    # high-dim trans-dim targets the PSIS reweighting routinely
    # produces Pareto k > 1, which collapses `pf.draws` toward
    # whichever basin Pathfinder happened to favour. `per_run_draws`
    # preserves one starting point per L-BFGS basin, which is what we
    # actually want for multi-chain PT initialisation. When
    # `n_chains > n_runs` we wrap around; when `n_chains < n_runs` we
    # take the first `n_chains` runs.
    n_avail = size(pf.per_run_draws, 2)
    cols = [((i - 1) % n_avail) + 1 for i in 1:n_chains]
    inits = pf.per_run_draws[:, cols]

    # Map the Pathfinder seeds into the space the chosen backend expects:
    #   - :pigeons operates in *unconstrained* space (it wraps the
    #     unconstrained `target` in an `InitializedTarget`), so the
    #     per-run draws are passed through as-is.
    #   - :nereus (in-house trans-dim path) operates in *bounded* space
    #     (it writes columns straight into `theta.values`), so the draws
    #     must be inverse-transformed first. `pf_target` always has a
    #     transform (ensured above).
    inits_to_pass = if backend_eff === :pigeons
        inits
    else
        bounded = Matrix{Float64}(undef, size(inits)...)
        for c in 1:size(inits, 2)
            @views bounded[:, c] = transform_inverse(inits[:, c], pf_target.transform)
        end
        bounded
    end

    # Pigeons must run on an unconstrained target (good geometry, and the
    # `inits` are in `pf_target`'s unconstrained space). When the caller's
    # `target` is bounded, `pf_target` is the unconstrained copy we built
    # above; in the common case (`target` already unconstrained) the two
    # are identical. The in-house path operates in bounded space and uses
    # the original `target`.
    pt_target = backend_eff === :pigeons ? pf_target : target

    pt_result = sample_pt(pt_target;
                           td = td,
                           n_rounds = n_rounds,
                           n_chains = n_chains,
                           seed = seed,
                           show_report = show_report,
                           init = inits_to_pass,
                           backend = backend_eff,
                           early_stop_thresh = early_stop_thresh,
                           early_stop_min_rounds = early_stop_min_rounds,
                           within_model = within_model)
    # Preserve historical return signature: 3-tuple for fixed-dim
    # (chains, log_ev, pf), 4-tuple for trans-dim (..., n_evals, pf).
    if td === nothing
        chains, log_ev = pt_result
        return chains, log_ev, pf
    else
        chains, log_ev, n_evals = pt_result
        return chains, log_ev, n_evals, pf
    end
end

# =====================================================================
# Trans-dimensional PT
# =====================================================================
#
# Pigeons' state management assumes fixed-dim vectors for its
# built-in tempering and reference distribution. For trans-dim, we
# run PT "manually" using the RJMCMC infrastructure with temperature
# ladders, since Pigeons' internal tempering multiplies the
# log-likelihood by β which is exactly what we need for birth/death.
#
# Architecture: multiple RJMCMC chains at different temperatures,
# with periodic replica swaps. Evidence via stepping-stone-like
# estimate from the temperature ladder.

"""
    TransDimPTState

State of one replica in the trans-dim PT sampler.
"""
mutable struct TransDimPTState
    theta::Theta{Float64}
    log_pi::Float64
    log_L::Float64
end

"""
    _sample_pt_transdim(target, td; kwargs...) -> (chains, log_evidence)

Trans-dimensional parallel tempering. Runs multiple RJMCMC chains
at different temperatures with replica swaps.
"""
function _sample_pt_transdim(
    target::NereusTarget,
    td::TransDimConfig;
    n_rounds::Int = 15,
    n_chains::Int = 10,
    seed::Int = 1,
    show_report::Bool = true,
    early_stop_thresh::Float64 = 0.0,
    early_stop_min_rounds::Int = 8,
    init::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    within_model::Symbol = :slice,
)
    within_model in (:slice, :rwm) || throw(ArgumentError(
        "within_model must be :slice or :rwm; got :$within_model"))
    rng = MersenneTwister(seed)
    params = target.params
    data = target.data
    layout = params.layout
    n_unfrozen = length(layout.unfrozen_idx)

    # Temperature ladder (geometric spacing from 0 to 1)
    betas = [i == 1 ? 0.0 : ((i - 1) / (n_chains - 1))^2 for i in 1:n_chains]
    betas[end] = 1.0  # cold chain

    # Validate init if provided. Trans-dim chains operate in bounded
    # space (`unconstrained=false`), so init values are taken to be in
    # bounded space directly — Pathfinder draws need to be transformed
    # to bounded space before being passed through `sample_pt_warm`'s
    # init bridge (handled there).
    if init !== nothing
        size(init, 1) == n_unfrozen || throw(ArgumentError(
            "init has $(size(init, 1)) rows, expected $n_unfrozen unfrozen params"))
        size(init, 2) >= n_chains || throw(ArgumentError(
            "init has $(size(init, 2)) columns, need ≥ n_chains = $n_chains"))
    end

    # Initialize replicas with per-chain RNGs for thread safety
    replicas = Vector{TransDimPTState}(undef, n_chains)
    chain_rngs = [MersenneTwister(seed + c) for c in 1:n_chains]
    for c in 1:n_chains
        # Activate all planet slots up to max_kplanet when either:
        #  (a) trans-dim is OFF for planets (`td.planets == false`) —
        #      this is the fixed-dim case routed through here; planets
        #      defined in params.config must all stay on, otherwise
        #      n_planets in the chain reports 0 and the plotters skip
        #      phase-folds, and the likelihood ignores planet params.
        #  (b) trans-dim is ON but Pathfinder init was supplied —
        #      we trust the warmup placement and let death proposals
        #      prune from there instead of paying the cold-discovery cost.
        all_active = !td.planets || (init !== nothing && td.planets)
        # Mask sized to the FULL noise-model list — all consumers index by
        # config.noise_models position; a length(toggleable) mask left
        # always-on models at low config indices silently inactive and made
        # births of high-index toggleables a BoundsError (see rjmcmc fix).
        td_state = TransDimState(
            max_planets = td.max_kplanet,
            n_noise = length(params.config.noise_models),
        )
        for (nm_idx, nm) in enumerate(params.config.noise_models)
            if !(td.noise && nm in td.toggleable)
                td_state.noise_active[nm_idx] = true
            end
        end
        if all_active
            for k in 1:td.max_kplanet
                activate_planet!(td_state, k)
            end
        end
        theta = Theta{Float64}(params; td=td_state)
        if init !== nothing
            # Take column c of init (in bounded space) into theta.values
            for (j, uf_idx) in enumerate(layout.unfrozen_idx)
                theta.values[uf_idx] = Float64(init[j, c])
            end
        else
            _init_systemics_from_prior!(theta, chain_rngs[c])
        end
        log_pi = log_prior(theta)
        log_L = _eval_ll(theta, data)
        replicas[c] = TransDimPTState(theta, log_pi, log_L)
    end

    # Per-chain likelihood counters and workspaces (avoid contention + allocations)
    chain_ctrs = [LikelihoodCounter() for _ in 1:n_chains]
    n_noise = length(params.config.noise_models)
    n_obs_rv = length(data.t_rv)
    n_phot   = length(data.t_phot)
    # Donor buffer must hold every OTHER replica (n_chains − 1); anything
    # smaller truncates the donor pool the birth proposals draw from.
    chain_ws = [PTWorkspace(params, td.max_kplanet, n_noise;
                              n_obs=n_obs_rv, n_phot=n_phot,
                              max_donors=max(1, n_chains - 1)) for _ in 1:n_chains]
    widths = _prior_widths(layout)

    # Total iterations: rounds double (like Pigeons)
    n_samples_per_round = 2
    total_samples = n_samples_per_round * (2^n_rounds - 1)

    # Noise column metadata
    noise_nm_indices = Int[]  # indices into params.config.noise_models
    if td.noise && !isempty(td.toggleable)
        for nm in td.toggleable
            nm_idx = findfirst(==(nm), params.config.noise_models)
            nm_idx !== nothing && push!(noise_nm_indices, nm_idx)
        end
    end
    n_noise_cols = length(noise_nm_indices)

    # Storage for cold chain samples
    cold_samples = Vector{Vector{Float64}}()
    cold_n_planets = Vector{Float64}()
    cold_noise = Vector{Vector{Float64}}()
    # Per-slot activity, serialized as planet_active_<k> columns. Without
    # them every consumer (science_tables, posterior_plots, label_switching)
    # falls back to "slot k active iff k ≤ n_planets" — an ORDER assumption
    # that is only as good as the packing of active slots, and wrong for any
    # draw taken between a mid-slot death and the next birth.
    n_pl_cols = td.max_kplanet
    cold_planet_active = Vector{Vector{Float64}}()

    iter = 0
    steps_between_swaps = 1
    t_round_start = time()

    # Early-termination state — track N_p posterior stability across rounds
    # to bail out of the doubling-iter rounds once the trans-dim coordinate
    # has converged. Off by default (early_stop_thresh = 0).
    prev_np_post = Float64[]
    rounds_actually_run = n_rounds

    # Single-line dynesty-style progress bar across all rounds. Total =
    # 2 + 4 + … + 2^n_rounds = 2(2^n_rounds − 1).
    pb = ProgressBar("PT trans-dim";
                       total = total_samples,
                       enabled = show_report)

    # Reddemcee-style evidence accumulator (TI+/SS+/H+). Accumulation
    # starts after warmup_rounds — Pigeons-style, drop the early
    # doubling rounds where ⟨log L⟩_β is dominated by transient.
    evidence_acc = EvidenceAccumulator(n_chains)
    evidence_warmup_rounds = max(1, n_rounds ÷ 2)
    logL_buf = Vector{Float64}(undef, n_chains)

    for round in 1:n_rounds
        n_iter_this_round = n_samples_per_round * 2^(round - 1)
        round_start = time()

        for _ in 1:n_iter_this_round
            iter += 1

            # --- Multiple MCMC steps per chain (parallel) ----------------
            Threads.@threads for c in 1:n_chains
                rep = replicas[c]
                beta = betas[c]
                crng = chain_rngs[c]
                cctr = chain_ctrs[c]
                ws = chain_ws[c]

                for _ in 1:steps_between_swaps
                    if rand(crng) < td.transdim_fraction
                        # Build population in pre-allocated buffer (no alloc)
                        ws.n_pop = 0
                        if td.planets
                            # The counter must never outrun the buffer: the
                            # view below is built from it. Sized at
                            # n_chains − 1 above, so the cap is a safety net
                            # that does not fire on this path.
                            cap = length(ws.population)
                            for (j, r) in enumerate(replicas)
                                j == c && continue
                                r.theta.td === nothing && continue
                                r.theta.td.n_planets_active > 0 || continue
                                ws.n_pop ≥ cap && break
                                ws.n_pop += 1
                                ws.population[ws.n_pop] = r.theta
                            end
                        end
                        pop_view = ws.n_pop > 0 ? view(ws.population, 1:ws.n_pop) : nothing

                        # Split trans-dim moves: planet vs noise
                        if td.planets && td.noise && !isempty(td.toggleable)
                            if rand(crng) < 0.5
                                _pt_transdim_move!(rep, data, td, beta, crng;
                                                   population=pop_view, ctr=cctr, ws=ws)
                            else
                                _pt_noise_move!(rep, data, td, beta, crng; ctr=cctr, ws=ws)
                            end
                        elseif td.planets
                            _pt_transdim_move!(rep, data, td, beta, crng;
                                               population=pop_view, ctr=cctr, ws=ws)
                        elseif td.noise && !isempty(td.toggleable)
                            _pt_noise_move!(rep, data, td, beta, crng; ctr=cctr, ws=ws)
                        end
                    else
                        if within_model === :rwm
                            # Adapt only during the first half of warmup;
                            # diminishing-adaptation theory requires the
                            # adaptation to vanish in the limit. Round
                            # midpoint is a reasonable cut.
                            adapt = round <= max(1, n_rounds ÷ 2)
                            _within_model_move_rwm!(rep.theta, data, crng,
                                                      ws, rep.log_pi,
                                                      rep.log_L;
                                                      beta=beta, adapt=adapt,
                                                      ctr=cctr) do new_lpi, new_lL
                                rep.log_pi = new_lpi
                                rep.log_L  = new_lL
                            end
                        else
                            _pt_within_move!(rep, data, beta, 1.0, widths, crng, ws; ctr=cctr)
                        end
                    end
                end
            end

            # --- Replica swaps (serial) ----------------------------------
            for c in 1:(n_chains - 1)
                _try_swap!(replicas, betas, c, c + 1, rng)
            end

            # Reddemcee evidence accumulator update (post-warmup only).
            # Replicas hold the current state at each beta; their log_L
            # is the chain-local log-likelihood at that temperature.
            if round > evidence_warmup_rounds
                @inbounds for k in 1:n_chains
                    logL_buf[k] = replicas[k].log_L
                end
                update_evidence!(evidence_acc, logL_buf, betas)
            end

            # Store cold chain sample
            rep_cold = replicas[end]
            sample = Vector{Float64}(undef, n_unfrozen)
            @inbounds for (j, uf_idx) in enumerate(layout.unfrozen_idx)
                sample[j] = rep_cold.theta.values[uf_idx]
            end
            push!(cold_samples, sample)
            push!(cold_n_planets, Float64(n_p(rep_cold.theta)))
            if n_pl_cols > 0
                ctd = rep_cold.theta.td
                push!(cold_planet_active,
                      Float64[ctd !== nothing && ctd.planet_active[k] ? 1.0 : 0.0
                              for k in 1:n_pl_cols])
            end
            # Noise state
            if n_noise_cols > 0
                ns_state = Float64[is_noise_active(rep_cold.theta.td, ni) ? 1.0 : 0.0
                                   for ni in noise_nm_indices]
                push!(cold_noise, ns_state)
            end

            # Progress bar update — running N_p posterior + cold log-L.
            np_running = vec(cold_n_planets)
            np_post = join([@sprintf("%2.0f", 100 * count(==(Float64(k)), np_running) / length(np_running))
                             for k in 0:td.max_kplanet], "/")
            update!(pb;
                     n_done = length(cold_samples),
                     fields = (:round => @sprintf("%d/%d", round, n_rounds),
                               :Np => np_post * "%",
                               :logL => rep_cold.log_L))
        end

        # Early-stop: terminate when the cold-chain N_p posterior has
        # stabilised. Cumulative cost of rounds k+1..n_rounds is dominated
        # by the doubling tail, so terminating at round 11 instead of 15
        # saves ~half the wall time on a converged problem.
        #
        # Only meaningful when td.planets == true (trans-dim over planet
        # count). For fixed-dim runs the N_p posterior is a delta function
        # at max_kplanet and the Δmax metric is trivially zero from the
        # second qualifying round onward, causing a premature early-stop
        # before any rounds with meaningful effective-sample-size have
        # run. Skip the check entirely in the fixed-dim case — fall
        # through to the normal n_rounds budget.
        if early_stop_thresh > 0 && round >= early_stop_min_rounds && td.planets
            curr_np_post = Float64[
                count(==(Float64(k)), cold_n_planets) / max(1, length(cold_n_planets))
                for k in 0:td.max_kplanet
            ]
            if !isempty(prev_np_post)
                Δmax = maximum(abs.(curr_np_post .- prev_np_post))
                if Δmax < early_stop_thresh
                    finish!(pb;
                             final = @sprintf("PT early-stop at round %d/%d: N_p posterior Δmax = %.4f < %.4f",
                                              round, n_rounds, Δmax, early_stop_thresh))
                    rounds_actually_run = round
                    break
                end
            end
            prev_np_post = curr_np_post
        end
    end
    finish!(pb)

    # Build Chains
    n_post = length(cold_samples)
    planet_col_names = Symbol[Symbol("planet_active_$(k)") for k in 1:n_pl_cols]
    noise_col_names = Symbol[Symbol("noise_active_$(ni)") for ni in noise_nm_indices]
    param_names_out = vcat(Symbol.(layout.unfrozen_names), [:n_planets],
                           planet_col_names, noise_col_names)
    n_out_cols = n_unfrozen + 1 + n_pl_cols + n_noise_cols
    mat = Matrix{Float64}(undef, n_post, n_out_cols)
    for i in 1:n_post
        mat[i, 1:n_unfrozen] = cold_samples[i]
        mat[i, n_unfrozen + 1] = cold_n_planets[i]
        for k in 1:n_pl_cols
            mat[i, n_unfrozen + 1 + k] = cold_planet_active[i][k]
        end
        for (j, _) in enumerate(noise_nm_indices)
            mat[i, n_unfrozen + 1 + n_pl_cols + j] =
                n_noise_cols > 0 ? cold_noise[i][j] : 0.0
        end
    end

    chains = MCMCChains.Chains(mat, param_names_out)

    # Evidence — reddemcee-style TI+/SS+/H+ from the accumulator
    # (Peña & Jenkins 2026, arXiv:2509.24870). TI+ is the primary
    # estimator; SS+ and H+ are reported as cross-checks. If no
    # post-warmup iterations ran (n_rounds ≤ warmup), fall back to the
    # final-replica trapezoidal TI to keep something on the wire.
    report = evidence_acc.started ?
              evidence_report(evidence_acc, betas) :
              EvidenceReport(
                  (_thermodynamic_integration(replicas, betas), 0.0),
                  (_thermodynamic_integration(replicas, betas), Inf),
                  (NaN, NaN), (NaN, NaN), NaN,
              )
    log_ev = report.ti_plus[1]

    if show_report
        total_evals = sum(c.count for c in chain_ctrs)
        @info "Trans-dim PT: $(n_post) samples, $(n_chains) chains, $(total_evals) likelihood evals"
        @info "log Z estimators (post-warmup samples only):"
        @info "  TI  (trapezoidal) = $(round(report.ti[1], digits=2))"
        @info "  TI+ (PCHIP)       = $(round(report.ti_plus[1], digits=2)) ± $(round(report.ti_plus[2], digits=3))"
        @info "  SS+ (geom-bridge) = $(round(report.ss_plus[1], digits=2)) ± $(round(report.ss_plus[2], digits=3))"
        @info "  H+  (β*=$(round(report.hybrid_beta_star, digits=3))) = $(round(report.hybrid[1], digits=2)) ± $(round(report.hybrid[2], digits=3))"
    end

    total_evals = sum(c.count for c in chain_ctrs)
    return chains, log_ev, total_evals
end

# --- PT internal helpers -----------------------------------------------

function _pt_transdim_move!(rep::TransDimPTState, data::Data,
                             td::TransDimConfig, beta::Float64,
                             rng::AbstractRNG;
                             population::Union{AbstractVector{Theta{Float64}}, Nothing}=nothing,
                             ctr::Union{LikelihoodCounter, Nothing}=nothing,
                             ws::Union{PTWorkspace, Nothing}=nothing)
    theta = rep.theta
    n_active = theta.td.n_planets_active
    max_k = td.max_kplanet
    scratch = ws !== nothing ? ws.scratch_theta : nothing

    p_birth, _ = _birth_death_probs(n_active, max_k)
    do_birth = rand(rng) < p_birth

    # Strategy selected for BOTH directions so the birth/death pair is
    # reversible within the selected strategy (see the note in
    # `_planet_move!` in rjmcmc.jl and test_birth_death_reversibility.jl).
    strategy = _select_strategy(td, rng)
    new_theta, log_q = do_birth ?
        propose_planet_birth(theta, rng, strategy;
                              data=data, population=population,
                              scratch=scratch) :
        propose_planet_death(theta, rng, strategy; scratch=scratch)

    !isfinite(log_q) && return

    new_log_pi = log_prior(new_theta)
    !isfinite(new_log_pi) && return

    new_log_L = ws !== nothing ? _eval_ll(new_theta, data, ctr, ws) :
                                 _eval_ll(new_theta, data, ctr)
    !isfinite(new_log_L) && return

    # Tempered acceptance: β multiplies likelihood ratio
    log_α = beta * (new_log_L - rep.log_L) + (new_log_pi - rep.log_pi) + log_q

    if log(rand(rng)) < log_α
        rep.theta.values .= new_theta.values
        rep.theta.td.n_planets_active = new_theta.td.n_planets_active
        rep.theta.td.planet_active .= new_theta.td.planet_active
        if !isempty(rep.theta.td.noise_active)
            rep.theta.td.noise_active .= new_theta.td.noise_active
        end
        rep.log_pi = new_log_pi
        rep.log_L = new_log_L
    end
end

function _pt_noise_move!(rep::TransDimPTState, data::Data,
                          td::TransDimConfig, beta::Float64,
                          rng::AbstractRNG;
                          ctr::Union{LikelihoodCounter, Nothing}=nothing,
                          ws::Union{PTWorkspace, Nothing}=nothing)
    theta = rep.theta
    scratch = ws !== nothing ? ws.scratch_theta : nothing

    # Within-group swap on the COLD side only (beta > 0.3): the hot chains
    # already cross the "none" valley between two exclusion-group members by
    # ordinary birth/death, because tempering flattens the intermediate state.
    # At beta ~ 1 that valley is deep and birth/death cannot cross it, so an
    # entrenched replica never leaves and the occupancy stops being P(M|D).
    # Same rationale and same gate as transdim_ptemcee.
    if beta > 0.3 && !isempty(td.noise_exclusion_groups) && rand(rng) < 0.5 &&
       _any_group_member_active(theta, td.noise_exclusion_groups)
        # Needs an active member to swap FROM; otherwise fall through to
        # birth/death rather than burning the move (see rjmcmc for the full
        # note — a high swap rate with no fallthrough freezes the chain).
        grp = _pick_active_group(theta, td.noise_exclusion_groups, rng)
        cand, lq = propose_noise_swap(theta, rng, grp, td.toggleable; data=data)
        isfinite(lq) || return
        cand_lp = log_prior(cand)
        isfinite(cand_lp) || return
        cand_ll = ws !== nothing ? _eval_ll(cand, data, ctr, ws) :
                                    _eval_ll(cand, data, ctr)
        isfinite(cand_ll) || return
        if log(rand(rng)) < beta*(cand_ll - rep.log_L) + (cand_lp - rep.log_pi) + lq
            rep.theta.values .= cand.values
            rep.theta.td.noise_active .= cand.td.noise_active
            rep.log_pi = cand_lp; rep.log_L = cand_ll
        end
        return
    end

    # 50/50 birth/death. `data` MUST be forwarded or every informed proposal
    # (AD's OLS coefficients, the GP period/amplitude hints) silently degrades
    # to a blind prior draw.
    if rand(rng) < 0.5
        new_theta, log_q = propose_noise_birth(theta, rng, td.toggleable;
                                                scratch=scratch, data=data,
                                                exclusion_groups=td.noise_exclusion_groups)
    else
        new_theta, log_q = propose_noise_death(theta, rng, td.toggleable;
                                                scratch=scratch, data=data)
    end

    !isfinite(log_q) && return

    new_log_pi = log_prior(new_theta)
    !isfinite(new_log_pi) && return

    new_log_L = ws !== nothing ? _eval_ll(new_theta, data, ctr, ws) :
                                 _eval_ll(new_theta, data, ctr)
    !isfinite(new_log_L) && return

    # Tempered acceptance
    log_α = beta * (new_log_L - rep.log_L) + (new_log_pi - rep.log_pi) + log_q

    if log(rand(rng)) < log_α
        rep.theta.values .= new_theta.values
        rep.theta.td.noise_active .= new_theta.td.noise_active
        rep.log_pi = new_log_pi
        rep.log_L = new_log_L
    end
end

function _pt_within_move!(rep::TransDimPTState, data::Data,
                           beta::Float64, scale::Float64,
                           widths::Vector{Float64},
                           rng::AbstractRNG;
                           ctr::Union{LikelihoodCounter, Nothing}=nothing)
    accepted = _within_model_move!(rep.theta, data, rng, scale, widths,
                                    rep.log_pi, rep.log_L;
                                    beta=beta, ctr=ctr) do new_lpi, new_lL
        rep.log_pi = new_lpi
        rep.log_L = new_lL
    end
    return accepted
end

function _pt_within_move!(rep::TransDimPTState, data::Data,
                           beta::Float64, scale::Float64,
                           widths::Vector{Float64},
                           rng::AbstractRNG,
                           ws::PTWorkspace;
                           ctr::Union{LikelihoodCounter, Nothing}=nothing)
    accepted = _within_model_move!(rep.theta, data, rng, scale, widths,
                                    rep.log_pi, rep.log_L, ws;
                                    beta=beta, ctr=ctr) do new_lpi, new_lL
        rep.log_pi = new_lpi
        rep.log_L = new_lL
    end
    return accepted
end

function _try_swap!(replicas::Vector{TransDimPTState},
                     betas::Vector{Float64}, i::Int, j::Int,
                     rng::AbstractRNG)
    # Metropolis swap criterion
    log_α = (betas[j] - betas[i]) * (replicas[i].log_L - replicas[j].log_L)
    if log(rand(rng)) < log_α
        replicas[i], replicas[j] = replicas[j], replicas[i]
    end
end

function _thermodynamic_integration(replicas::Vector{TransDimPTState},
                                     betas::Vector{Float64})
    # Simple estimate: use current log_L from each replica
    # (in production, we'd accumulate mean <log L>_β over iterations)
    n = length(betas)
    n < 2 && return 0.0
    log_z = 0.0
    for i in 1:(n - 1)
        db = betas[i + 1] - betas[i]
        log_z += db * (replicas[i].log_L + replicas[i + 1].log_L) / 2
    end
    return log_z
end
