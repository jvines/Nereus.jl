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
- `n_warmup_rounds::Union{Nothing,Int}=nothing` : rounds of warmup. The RWM
  step sizes adapt over them, the circular seams are re-cut at their end, and
  the evidence accumulates only after them. `nothing` is `max(1, n_rounds ÷ 2)`,
  or, with `resume = true`, the warmup of the run being continued, so extending
  a run does not move its warmup.
- `checkpoint=nothing` : path of a state file. The sampler writes its whole
  state there (every replica's position, model state and cached log-densities,
  every RNG stream, the RWM step sizes and their counters, the likelihood
  counters, the cold-chain draws, the evidence accumulators, the circular
  windows and warmup trace, and the early-stop history) at the end of the run,
  before an early stop, and every `checkpoint_interval` seconds during it,
  replacing the file atomically. `run_job` sets it to `pt_state.jls` in
  `output_dir` unless the job gives one.
- `checkpoint_interval::Real=900` : seconds between checkpoints during the run.
- `resume::Bool=false` : continue from `checkpoint` instead of initialising (no
  prior draws, no Pathfinder): a run that was killed, or a finished one that
  needs more rounds. `n_rounds` is the new TOTAL, counted from the start of the
  original run; the draws already kept stay, and the new ones are appended. The
  continuation is bit-identical to an uninterrupted run of `n_rounds` with the
  same warmup. A run that stopped early carries on, with
  the early-stop check run again from the next round (`early_stop_thresh = 0`
  turns it off). Refused, with the differences listed, if the checkpoint came
  from different data, priors, parameters, chains, seed, warmup, within-model
  kernel, trans-dim settings or initialisation, or if `n_rounds` would end
  before it. Not available for a trans-dim run whose planet births use
  `InformedBirth` or `JointInformedBirth` (the default `td` does): their
  periodogram peaks live in process-global caches keyed on `objectid(rng)`,
  which a checkpoint cannot hold, so such a run is not checkpointed (a warning
  says so) and `resume` is refused.

Circular angles (`src/circular.jl`): at the end of the warmup rounds each
full-circle seam is moved to the emptiest arc of the cold replica's recent
history and every replica is relabelled, so the returned draws are in one
chart. With planet births on (`td.planets`), same-mode planet slots share one
window per angle, set before the first state is drawn and re-cut together. An
`init` is relabelled into the current windows (a copy; the caller's matrix is
not touched). The layout (and `target.transform`) keep the moved window.
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
    init_strategy::Symbol = :prior,
    n_pathfinder_runs::Int = 16,
    n_pathfinder_draws::Int = 0,
    n_warmup_rounds::Union{Nothing,Int} = nothing,
    checkpoint::Union{Nothing,AbstractString,Symbol} = nothing,
    checkpoint_interval::Real = 900.0,
    resume::Bool = false,
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
    init_strategy in (:prior, :pathfinder) || throw(ArgumentError(
        "init_strategy must be :prior (default) or :pathfinder; got :$init_strategy"))

    # Pathfinder warm start. Was the separate `pt_warm` engine, which was not
    # a sampler: it computed Pathfinder draws and delegated straight back here.
    # An initialisation strategy is not an algorithm, so it is a keyword.
    #
    # NOT offered on `pt_emcee`, deliberately -- see the note there. Pathfinder's
    # MVN approximation is poor on sharp curved ridges (Pareto k >> 0.5) and
    # seeds every walker into one spurious basin, producing pristine R-hat/ESS
    # at a wrong orbit (HD 159062: a=34.5 vs true ~58).
    #
    # `init_key` is what the run started from, for the checkpoint fingerprint:
    # the caller's matrix, the Pathfinder settings (its draws follow from them
    # and `seed`), or prior draws. Taken before Pathfinder runs, which a resume
    # skips: the saved replicas replace any starting point.
    init_mat = _pt_init_matrix(init)
    init_key = init_mat !== nothing ? hash(init_mat) :
               init_strategy === :pathfinder && n_pathfinder_runs > 0 ?
                   (:pathfinder, n_pathfinder_runs, n_pathfinder_draws) : :prior
    if !resume && init_strategy === :pathfinder && init_mat === nothing &&
       n_pathfinder_runs > 0
        init_mat = _pathfinder_inits(target, n_chains;
                                     n_runs  = n_pathfinder_runs,
                                     n_draws = n_pathfinder_draws,
                                     seed    = seed)
    end

    # In-house PT (default). Two cases:
    #   - User supplied a TransDimConfig → trans-dim run, as before.
    #   - User left td === nothing → fixed-dim run; build a "null"
    #     config so the optimised zero-alloc threaded RWM hot path
    #     runs as a plain fixed-dim PT (no birth/death moves).
    # This is ~10x faster than Pigeons on Nereus targets, because Pigeons'
    # slice-sampling explore!() loop is not parallelised by default. The
    # Pigeons path survives as `_sample_pt_pigeons` for cross-checks; it is
    # deliberately not an engine.
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
        init=init_mat,
        within_model=within_model,
        n_warmup_rounds, checkpoint, checkpoint_interval, resume, init_key)
    # Preserve historical return shapes: 2-tuple for fixed-dim
    # (matches the old Pigeons-backed path); 3-tuple for trans-dim
    # (matches the old in-house path). Test scripts depend on this.
    return td_user_supplied ? (chains, log_ev, n_evals) : (chains, log_ev)

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

`n_warmup_rounds`, `checkpoint`, `checkpoint_interval` and `resume` are
[`sample_pt`](@ref)'s. `init_key` stands for the starting point in the
checkpoint fingerprint (`sample_pt` passes it; `nothing` derives it from
`init`). `halt_after` is a test hook: with a `checkpoint`, the run writes it
after iteration `halt_after` and throws `InterruptException`, as a killed run
would.
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
    n_warmup_rounds::Union{Nothing,Int} = nothing,
    checkpoint::Union{Nothing,AbstractString,Symbol} = nothing,
    checkpoint_interval::Real = 900.0,
    resume::Bool = false,
    init_key = nothing,
    halt_after::Int = 0,
)
    within_model in (:slice, :rwm) || throw(ArgumentError(
        "within_model must be :slice or :rwm; got :$within_model"))
    n_warmup_rounds === nothing || n_warmup_rounds >= 1 || throw(ArgumentError(
        "n_warmup_rounds must be ≥ 1; got $n_warmup_rounds"))
    resume && checkpoint === nothing && throw(ArgumentError(
        "resume = true needs `checkpoint`, the state file to continue from"))
    # Planet births from the informed proposals (InformedBirth,
    # JointInformedBirth) read periodogram / BLS peaks from process-global
    # caches (src/transdim/birth_strategies.jl: _PEAK_CACHES, _BLS_CACHES),
    # keyed on objectid(rng) and the active set and refreshed every
    # INFORMED_CACHE_INTERVAL calls. Those peaks and counters are chain state,
    # and the keys are address-derived hashes, so they cannot be saved and put
    # back: such a run cannot be continued bit-identically, and is neither
    # checkpointed nor resumed rather than continued approximately.
    informed_births = td.planets && td.transdim_fraction > 0 &&
        any(s -> s isa Union{InformedBirth, JointInformedBirth}, td.birth_strategies)
    resume && informed_births && throw(ArgumentError(
        "resume = true: this run proposes planet births with InformedBirth / " *
        "JointInformedBirth, whose periodogram caches (process-global, keyed on " *
        "objectid(rng)) cannot be saved, so it cannot be continued bit-identically"))
    if informed_births && checkpoint !== nothing
        @warn "sample_pt: not checkpointing $checkpoint: informed planet births " *
              "(InformedBirth / JointInformedBirth) keep process-global periodogram " *
              "caches that a checkpoint cannot hold, so this run cannot be resumed."
    end
    ck_path = checkpoint === nothing || informed_births ? nothing : String(checkpoint)
    ck_every = Float64(checkpoint_interval)
    rng = MersenneTwister(seed)
    params = target.params
    data = target.data
    layout = params.layout
    n_unfrozen = length(layout.unfrozen_idx)

    # Temperature ladder (geometric spacing from 0 to 1)
    betas = [i == 1 ? 0.0 : ((i - 1) / (n_chains - 1))^2 for i in 1:n_chains]
    betas[end] = 1.0  # cold chain

    # Total iterations: rounds double (like Pigeons). Round r ends at iteration
    # n_samples_per_round * (2^r - 1).
    n_samples_per_round = 2
    total_samples = n_samples_per_round * (2^n_rounds - 1)

    # Warmup rounds: RWM adaptation, the circular re-cut at their end, and the
    # evidence only after them. A resume that does not name them keeps the
    # saved run's (`max(1, n_rounds ÷ 2)` of its own n_rounds, typically), so
    # extending a run does not move its warmup.
    evidence_warmup_rounds = something(n_warmup_rounds,
                                       resume ? _pt_saved_warmup(ck_path) : nothing,
                                       max(1, n_rounds ÷ 2))

    # --- Checkpoint / resume (src/checkpoint.jl) ------------------------
    # The fingerprint is the target plus every setting the chain depends on;
    # n_rounds and the early-stop settings only decide how long it runs. On
    # resume the replicas are not drawn: the saved state is put back just
    # before the main loop, which carries on from iteration `iter0 + 1`.
    ck_fp = run_fingerprint(params, data; n_chains, seed,
        warmup_rounds = evidence_warmup_rounds, within_model,
        td = _pt_td_key(td),
        init = init_key !== nothing ? init_key :
               init === nothing ? :prior : hash(Matrix{Float64}(init)))
    ck = resume ? read_checkpoint(ck_path, "pt", ck_fp) : nothing
    iter0 = ck === nothing ? 0 : ck.iter::Int
    iter0 <= total_samples || throw(ArgumentError(
        "the checkpoint is at iteration $iter0 (round $(_pt_round_of(iter0))); " *
        "n_rounds = $n_rounds would end before it"))

    # Validate init if provided. Trans-dim chains operate in bounded
    # space (`unconstrained=false`), so init values are taken to be in
    # bounded space directly — Pathfinder draws need to be transformed
    # to bounded space before being passed to sample_pt's
    # init bridge (handled there).
    if init !== nothing
        size(init, 1) == n_unfrozen || throw(ArgumentError(
            "init has $(size(init, 1)) rows, expected $n_unfrozen unfrozen params"))
        size(init, 2) >= n_chains || throw(ArgumentError(
            "init has $(size(init, 2)) columns, need ≥ n_chains = $n_chains"))
    end

    # Circular angles (src/circular.jl). Every birth ends in
    # `_sort_group_periods!`, which copies raw values between same-mode slots,
    # so with `td.planets` on those slots share one window per angle -- set
    # here, before any replica is drawn. Nothing else moves planet blocks: with
    # births off every angle keeps a window of its own.
    circ_groups = circular_groups(params; permutable = td.planets)
    unify_circular_groups!(params, circ_groups; transforms = (target.transform,))

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
        if ck !== nothing
            # Resume: nothing is drawn or evaluated; the saved replica is put
            # back below, before the main loop.
            replicas[c] = TransDimPTState(theta, NaN, NaN)
            continue
        elseif init !== nothing
            # Take column c of init (in bounded space) into theta.values,
            # relabelled into the current circular windows: the init is written
            # in the user's windows, or Pathfinder's, which the layout need not
            # share any more. A copy, so the caller's matrix is left alone.
            x0 = circular_relabel_point!(Float64[init[j, c] for j in 1:n_unfrozen],
                                         params)
            for (j, uf_idx) in enumerate(layout.unfrozen_idx)
                theta.values[uf_idx] = x0[j]
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
                       enabled = show_report,
                       start = iter0)

    # Reddemcee-style evidence accumulator (TI+/SS+/H+). Accumulation
    # starts after warmup_rounds — Pigeons-style, drop the early
    # doubling rounds where ⟨log L⟩_β is dominated by transient.
    # `evidence_warmup_rounds` is set above, with the fingerprint.
    evidence_acc = EvidenceAccumulator(n_chains)
    logL_buf = Vector{Float64}(undef, n_chains)

    # Circular angles (src/circular.jl): the cold replica's ACTIVE values over
    # the second half of the warmup iterations, pooled per window group,
    # decide where each seam goes, and every replica is relabelled at the end
    # of the last warmup round, so the rounds that feed the evidence run in one
    # chart. Safe here, unlike multi-chain rjmcmc / moms: all replicas live in
    # this one loop and the re-cut runs between rounds, outside the threaded
    # sweep. The trace is kept even when this run ends with its warmup (no
    # re-cut then: nothing follows), so a resume that extends it re-cuts
    # exactly where an uninterrupted run would have.
    warmup_iters = n_samples_per_round * (2^evidence_warmup_rounds - 1)
    circ_from = warmup_iters - warmup_iters ÷ 2
    circ = isempty(circ_groups) ? nothing :
           _CircularWarmupTrace(params, circ_groups, warmup_iters ÷ 2)

    # --- Resume: put the saved run back exactly as it stopped ----------
    # In place: the threaded sweep's closure captures replicas, chain_rngs,
    # chain_ctrs and chain_ws, and rebinding a captured variable would box it.
    # Replicas get their saved positions, model states and cached densities
    # (the Theta objects stay; swaps only ever permute them). The workspace
    # likelihood caches are not saved: they are keyed on hashes of what they
    # were computed from, so a cold cache recomputes the same values.
    if ck !== nothing
        for c in 1:n_chains
            rep = replicas[c]
            rep.theta.values .= ck.values[c]
            _pt_copy_td_state!(rep.theta.td, ck.td_states[c])
            rep.log_pi = ck.log_pi[c]::Float64
            rep.log_L  = ck.log_L[c]::Float64
            copy!(chain_rngs[c], ck.chain_rngs[c])
            chain_ctrs[c].count = ck.n_evals[c]::Int
            ws = chain_ws[c]
            ws.rwm_sigmas .= ck.rwm_sigmas[c]
            ws.rwm_attempts .= ck.rwm_attempts[c]
            ws.rwm_accepts .= ck.rwm_accepts[c]
        end
        copy!(rng, ck.rng)
        append!(cold_samples, ck.cold_samples)
        append!(cold_n_planets, ck.cold_n_planets)
        append!(cold_noise, ck.cold_noise)
        append!(cold_planet_active, ck.cold_planet_active)
        for f in fieldnames(EvidenceAccumulator)
            setfield!(evidence_acc, f, getfield(ck.evidence, f))
        end
        prev_np_post = ck.prev_np_post::Vector{Float64}
        # Windows the saved replicas live in (moved by the warmup re-cut).
        for (nm, (lo, _)) in ck.windows
            set_circular_window!(params, findfirst(==(nm), layout.unfrozen_names),
                                 lo; transforms = (target.transform,))
        end
        if circ !== nothing
            if ck.circ_draws === nothing          # already re-cut
                circ = nothing
            else
                foreach(append!, circ.draws, ck.circ_draws)
                circ.tick = ck.circ_tick::Int
            end
        end
        iter = iter0
        # A run that ended with its warmup skipped the re-cut (nothing followed
        # it). This one goes on, so it happens now, where it would have.
        if circ !== nothing && iter0 == warmup_iters && evidence_warmup_rounds < n_rounds
            _pt_recut_circular!(circ, params, replicas, cold_samples, data,
                                chain_ctrs, chain_ws, target)
            circ = nothing
        end
    end

    # Where the main loop picks up: `r_done` whole rounds are behind, and `k0`
    # iterations of the next one. A checkpoint on a round boundary is always
    # written after that round's closing steps (early-stop check, re-cut).
    r_done = 0
    while n_samples_per_round * (2^(r_done + 1) - 1) <= iter0
        r_done += 1
    end
    k0 = iter0 - n_samples_per_round * (2^r_done - 1)
    last_ck = time()

    for round in (r_done + 1):n_rounds
        n_iter_this_round = n_samples_per_round * 2^(round - 1)
        round_start = time()

        for it in (round == r_done + 1 ? k0 + 1 : 1):n_iter_this_round
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
                            # Adapt only during warmup (by default the first
                            # half of the rounds); diminishing-adaptation
                            # theory requires the adaptation to vanish in the
                            # limit. Round midpoint is a reasonable cut.
                            adapt = round <= evidence_warmup_rounds
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
            if circ !== nothing && circ_from < iter <= warmup_iters
                _record_circular_warmup!(circ, rep_cold.theta)
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

            # Checkpoint every `checkpoint_interval` s (src/checkpoint.jl). Not
            # on a round's last iteration: that one is written after the
            # round's closing steps below, so a boundary checkpoint never
            # leaves them half done.
            if ck_path !== nothing && it < n_iter_this_round &&
               (time() - last_ck >= ck_every || iter == halt_after)
                _pt_write_checkpoint(ck_path, ck_fp, iter, replicas, rng,
                    chain_rngs, chain_ctrs, chain_ws, cold_samples,
                    cold_n_planets, cold_noise, cold_planet_active,
                    evidence_acc, prev_np_post, circ, params)
                last_ck = time()
                iter == halt_after && throw(InterruptException())
            end
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
        #
        # On a stop the checkpoint is written first, with this round's N_p
        # posterior as the one to compare against: a resume carries on and
        # checks again from the next round.
        stop_early = false
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
                    stop_early = true
                end
            end
            prev_np_post = curr_np_post
        end

        # The warmup re-cut, when rounds follow it in this run. A run that
        # stops here early, or ends with its warmup, leaves it to a resume.
        if !stop_early && circ !== nothing && round == evidence_warmup_rounds &&
           evidence_warmup_rounds < n_rounds
            _pt_recut_circular!(circ, params, replicas, cold_samples, data,
                                chain_ctrs, chain_ws, target)
            circ = nothing
        end

        # Checkpoint: every `checkpoint_interval` s, on the last round, and
        # before an early stop (src/checkpoint.jl).
        if ck_path !== nothing && (stop_early || round == n_rounds ||
                                   time() - last_ck >= ck_every || iter == halt_after)
            _pt_write_checkpoint(ck_path, ck_fp, iter, replicas, rng,
                chain_rngs, chain_ctrs, chain_ws, cold_samples,
                cold_n_planets, cold_noise, cold_planet_active,
                evidence_acc, prev_np_post, circ, params)
            last_ck = time()
            iter == halt_after && throw(InterruptException())
        end
        stop_early && break
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

# `sample_pt`'s `init` as an n_dim × n_chains matrix, or `nothing`.
_pt_init_matrix(::Nothing) = nothing
_pt_init_matrix(init::AbstractMatrix{<:Real}) = Matrix{Float64}(init)
function _pt_init_matrix(init::NamedTuple)
    haskey(init, :draws) || throw(ArgumentError(
        "init NamedTuple must have a :draws field (got keys $(keys(init)))"))
    return Matrix{Float64}(init.draws)
end

# The round iteration `iter` falls in (rounds end at 2(2^r - 1)).
function _pt_round_of(iter::Int)
    r = 1
    while 2 * (2^r - 1) < iter
        r += 1
    end
    return r
end

# The trans-dim settings as plain values for the checkpoint fingerprint, which
# compares with `isequal`: the config holds vectors of structs, which compare by
# identity, so the noise models and birth strategies go in as their `repr`. With
# noise births on, the module switches for their informed proposals go in too.
_pt_td_key(td::TransDimConfig) = (
    planets = td.planets, max_kplanet = td.max_kplanet, noise = td.noise,
    toggleable = repr.(td.toggleable),
    birth_strategies = repr.(td.birth_strategies),
    birth_weights = td.birth_weights, transdim_fraction = td.transdim_fraction,
    noise_exclusion_groups = [repr.(g) for g in td.noise_exclusion_groups],
    alias_jump_fraction = td.alias_jump_fraction,
    informed_noise = td.noise ? (GP_INFORMED_BIRTH[], AD_INFORMED_BIRTH[]) : nothing)

# The warmup of the run the checkpoint at `path` holds, or `nothing` when there
# is no pt checkpoint there (`read_checkpoint` then says what is wrong). Read
# ahead of the fingerprint, so a resume that does not name `n_warmup_rounds`
# keeps the saved run's.
function _pt_saved_warmup(path::AbstractString)
    isfile(path) || return nothing
    ck = try
        open(deserialize, path)
    catch
        return nothing
    end
    (ck isa NamedTuple && get(ck, :sampler, nothing) == "pt" &&
     ck.fingerprint isa NamedTuple) || return nothing
    w = get(ck.fingerprint, :warmup_rounds, nothing)
    return w isa Int ? w : nothing
end

# Copy a replica's saved model state into its live one, in place (every field,
# so a field added to TransDimState is not silently left behind).
function _pt_copy_td_state!(dst::TransDimState, src::TransDimState)
    for f in fieldnames(TransDimState)
        v = getfield(src, f)
        v isa AbstractArray ? copyto!(getfield(dst, f), v) : setfield!(dst, f, v)
    end
    return dst
end

# Everything `_sample_pt_transdim`'s main loop carries from one iteration to the
# next, written atomically to `path` (src/checkpoint.jl). `circ_draws = nothing`
# marks the warmup re-cut as done (or never needed).
function _pt_write_checkpoint(path, fp, iter, replicas, rng, chain_rngs,
                              chain_ctrs, chain_ws, cold_samples, cold_n_planets,
                              cold_noise, cold_planet_active, evidence_acc,
                              prev_np_post, circ, params)
    write_checkpoint(path, "pt", fp, (; iter,
        values = [r.theta.values for r in replicas],
        td_states = [r.theta.td for r in replicas],
        log_pi = Float64[r.log_pi for r in replicas],
        log_L = Float64[r.log_L for r in replicas],
        rng, chain_rngs, n_evals = Int[c.count for c in chain_ctrs],
        rwm_sigmas = [ws.rwm_sigmas for ws in chain_ws],
        rwm_attempts = [ws.rwm_attempts for ws in chain_ws],
        rwm_accepts = [ws.rwm_accepts for ws in chain_ws],
        cold_samples, cold_n_planets, cold_noise, cold_planet_active,
        evidence = evidence_acc, prev_np_post,
        circ_draws = circ === nothing ? nothing : circ.draws,
        circ_tick = circ === nothing ? 0 : circ.tick,
        windows = circular_windows(params)))
end

# The end-of-warmup circular re-cut: move the seams, relabel every replica and
# every stored draw into the new windows, and refresh the replicas' cached
# densities.
function _pt_recut_circular!(circ, params, replicas, cold_samples, data,
                             chain_ctrs, chain_ws, target)
    moved = _recut_circular_warmup!(circ, params,
                                    (r.theta for r in replicas);
                                    transforms = (target.transform,))
    # The warmup draws already stored go into the new chart too, so
    # the returned chain is in one chart from its first row.
    for m in moved, s in cold_samples
        s[m.pos] = circular_relabel(s[m.pos], m.lo, m.hi)
    end
    # Same density in either chart; re-evaluated only so the cached
    # values are those of the stored point, bit for bit. Slot c's
    # workspace, as the sweep pairs them.
    if !isempty(moved)
        for c in eachindex(replicas)
            rep = replicas[c]
            rep.log_pi = log_prior(rep.theta)
            rep.log_L = _eval_ll(rep.theta, data, chain_ctrs[c], chain_ws[c])
        end
    end
    return nothing
end

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
    # Same rationale and same gate as transdim_pt_emcee.
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

# =====================================================================
# Pigeons cross-check path
# =====================================================================
#
# NOT an engine and not in ENGINES. Pigeons runs the same algorithm as
# sample_pt, single-threaded for NereusTarget (~8x slower), cannot do
# trans-dim, and on HD 159062 was non-reproducible across seeds
# (a = 115 / 41 / 78 / 64). It exists to cross-check the in-house PT,
# which is a testing concern, not a user-facing choice. Exposing it as
# `backend` put an implementation swap in the same namespace as the
# algorithm selector; that is what this refactor removed.
#
# No warmup re-cut of circular angles: Pigeons owns the replicas, and in its
# logit space the seam is at y = ±∞. The output-side re-cut is all it gets.
function _sample_pt_pigeons(
    target::NereusTarget;
    n_rounds::Int = 15,
    n_chains::Int = 10,
    seed::Int = 1,
    show_report::Bool = true,
    td::Union{TransDimConfig, Nothing} = nothing,
    init::Union{Nothing, AbstractMatrix{<:Real}, NamedTuple} = nothing,
    explorer::Symbol = :slice,
)
    td === nothing || throw(ArgumentError(
        "_sample_pt_pigeons does not support td; use sample_pt for trans-dim."))

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
    _pathfinder_inits(target, n_chains; n_runs, n_draws, seed) -> Matrix{Float64}

`n_chains` PT starting points, one per independent Pathfinder L-BFGS basin,
returned in BOUNDED space ready for `sample_pt`'s `init`.

Two details that are load-bearing, both learned the hard way:

1. **Pathfinder must run in unconstrained space.** Its L-BFGS-fitted MVN
   approximations do not respect `[lo, hi]` bounds. Fed a bounded target, the
   MVN draws land outside support, return `-Inf` log-density, and the PSIS
   reweighting divides `-Inf - -Inf` into `NaN` once enough draws are `-Inf`.
   NOI-106823 hit this consistently from `runner.jl`. So an unconstrained copy
   of the target is built when needed, and the draws are transformed back.

2. **Seed from `per_run_draws`, not the reweighted pool.** On high-dimensional
   trans-dim targets PSIS routinely gives Pareto k > 1, which collapses
   `pf.draws` toward whichever basin Pathfinder happened to favour. One draw
   per L-BFGS basin is what multi-chain PT actually wants. `n_chains > n_runs`
   wraps around; `n_chains < n_runs` takes the first `n_chains`.
"""
function _pathfinder_inits(target::NereusTarget, n_chains::Int;
                           n_runs::Int = 16, n_draws::Int = 0, seed::Int = 1)
    nd = n_draws == 0 ? max(2 * n_chains, 200) : n_draws
    nd >= n_chains || throw(ArgumentError(
        "n_pathfinder_draws ($nd) must be >= n_chains ($n_chains)"))

    pf_target = target.transform === nothing ?
                  NereusTarget(target.params, target.data; unconstrained = true) :
                  target
    pf = pathfinder_init(pf_target; n_runs = n_runs, n_draws = nd, seed = seed)

    n_avail = size(pf.per_run_draws, 2)
    cols  = [((i - 1) % n_avail) + 1 for i in 1:n_chains]
    inits = pf.per_run_draws[:, cols]

    bounded = Matrix{Float64}(undef, size(inits)...)
    for c in 1:size(inits, 2)
        @views bounded[:, c] = transform_inverse(inits[:, c], pf_target.transform)
    end
    return bounded
end
