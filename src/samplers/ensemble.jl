# Affine-invariant ensemble MCMC (emcee-style): the Goodman & Weare (2010)
# stretch move.
#
# Gradient-free, embarrassingly parallel, good for multi-modal posteriors.
# The standard sampler used in most RV exoplanet papers.
#
# Walker initialization follows astroEMPEROR: each walker is drawn
# independently from the prior, invalid draws (logp or logL = -Inf)
# are rejected and redrawn until all walkers are valid.
#
# The step loop is AffineInvariantMCMC.sample's, written out here so the run can
# checkpoint and resume (src/checkpoint.jl). The package runs the whole chain in
# one call: it cannot be stopped part-way, and it recomputes the walkers'
# log-densities on entry instead of taking them. `_stretch_step!` draws the same
# random numbers in the same order and does the same arithmetic, so a seeded run
# at `de_move = 0` is bit-identical to what the package returned;
# test/test_ensemble_resume.jl holds it to that. By default a share `de_move` of
# the walker updates are differential-evolution moves instead
# (src/samplers/de_move.jl, `_ensemble_step!`).

using AffineInvariantMCMC
using Random
using MCMCChains

"""
    sample_ensemble(target::NereusTarget; kwargs...) -> chains

Affine-invariant ensemble MCMC (Goodman & Weare 2010): the stretch move (scale
a = 2) mixed with a differential-evolution move (`de_move`). At `de_move = 0` it
is, step for step, the algorithm of AffineInvariantMCMC.jl. Returns posterior as
`MCMCChains.Chains`, with the collective-drift check of each chain's walkers
([`ensemble_drift`](@ref)) in `chains.info.drift`.

Works in unconstrained space (with transforms) by default.

# Keywords
- `n_walkers::Int=50` — number of walkers (must be ≥ 2 × n_dim)
- `n_steps::Int=5000` — steps per walker
- `n_burnin::Int=1000` — burn-in steps to discard
- `thinning::Int=1` — thinning factor
- `init::Union{Nothing, Vector{Float64}}=nothing` — if provided, walkers scatter around this point; otherwise drawn from prior (EMPEROR-style)
- `seed::Int=1`
- `de_move::Real=DE_MOVE_DEFAULT` — as in `sample_pt_emcee`: per walker and
  step, the probability of a differential-evolution move, x' = x + γ(x_j − x_k)
  + ε with j ≠ k from the other half of the walkers, in place of the stretch
  move (src/samplers/de_move.jl). On the sampling coordinates, like the
  stretch. `0` is AffineInvariantMCMC's stretch-only kernel, draw for draw.
- `n_chains::Int=1` — independent ensembles run on Julia threads, chain `c`
  seeded with `seed + c - 1`. They share the circular windows: the burn-in
  re-cut is taken from the walkers of all of them, at a point where each has
  stopped at the same step.
- `show_progress::Bool=true` — progress bar (the first chain's, when
  `n_chains > 1`).
- `prune_stranded::Bool=true` — every 50 steps through the first three quarters
  of burn-in, move walkers stranded where the posterior cannot carry mass onto
  the ones that found it, as `sample_pt_emcee` does for each rung (see
  `_pt_prune_stranded!`, which also gives the mass bound that makes it safe).
  A stretch move cannot bring a walker into a narrow mode from far away, and
  without this one stranded walker in 32 recorded 3 % of the draws 90° or more
  from an RM λ constrained to ±2°, or at the seam of its window, where the
  logit's plateau holds it for good. Burn-in only: every recorded step is
  ordinary MCMC.
- `n_scouts::Int=16` — without `init`, the walkers drawn from the prior are
  joined by up to `n_walkers ÷ 2` scouts in place of as many of them: prior
  draws moved by slice updates until they stop climbing, less those the bound
  of the stranded-walker pruning shows to be where the posterior cannot carry
  mass (src/samplers/scouts.jl). The pruning only moves walkers onto ones that
  found the mode; when none of a chain's did, it had nothing to move them to,
  and one chain in four of an RV + IAD companion at i = 5° sat 1,500 nats
  below the mode, at i = 161°, for the whole run. `0` draws every walker from
  the prior.
- `checkpoint=nothing` — path of a state file. The sampler writes its whole
  state there (the walkers, their log-densities, the RNG, the step reached and
  the draws kept so far) at the end of the run and every `checkpoint_interval`
  seconds during it, replacing the file atomically. With `n_chains > 1` each
  chain has its own file, `_chain<c>` added before the extension. `run_job`
  sets it to `ensemble_state.jls` in `output_dir` unless the job gives one.
- `checkpoint_interval::Real=900` — seconds between checkpoints during the run.
- `resume::Bool=false` — continue from `checkpoint` instead of initialising:
  a run that was killed, or a finished one that needs more steps. `n_steps` is
  the new TOTAL, counted from the start of the original run; the draws already
  kept stay, and the new ones are appended. The continuation is bit-identical to
  an uninterrupted run of `n_steps`. Refused, with the differences listed, if
  the checkpoint came from different data, priors, parameters, circular windows,
  walkers, burn-in, thinning, seed, `init`, `n_chains`, `de_move` or
  parameterisation.
"""
function sample_ensemble(
    target::NereusTarget;
    n_walkers::Int = 50,
    n_steps::Int = 5000,
    n_burnin::Int = 1000,
    thinning::Int = 1,
    init::Union{Nothing, Vector{Float64}} = nothing,
    seed::Int = 1,
    n_chains::Int = 1,
    show_progress::Bool = true,
    checkpoint::Union{Nothing,AbstractString,Symbol} = nothing,
    checkpoint_interval::Real = 900.0,
    resume::Bool = false,
    prune_stranded::Bool = true,
    n_scouts::Int = 16,
    de_move::Real = DE_MOVE_DEFAULT,
)
    n_scouts >= 0 || throw(ArgumentError("n_scouts must be ≥ 0"))
    p_de = _de_check(de_move)
    ck_path = checkpoint === nothing ? nothing : String(checkpoint)
    resume && ck_path === nothing && throw(ArgumentError(
        "resume = true needs `checkpoint`, the state file to continue from"))
    ck_every = Float64(checkpoint_interval)

    # Multi-chain: fan out `n_chains` independent ensemble runs on
    # Julia threads. Each chain re-draws its walker init and runs
    # the stretch move on its own state. Useful both for wall-clock speed
    # (Goodman-Weare within-chain walker stepping is single-threaded here)
    # and for between-chain R-hat convergence diagnostics.
    #
    # The chains share ONE Params, so the burn-in re-cut of the circular
    # seams (which moves the layout's windows) is taken at a rendezvous
    # (src/circular.jl): on each due step every chain stops, the last to arrive
    # re-cuts from all their walkers and relabels every chain's, and all carry
    # on. Their checkpoints are read here, before any starts, so a resume puts
    # the shared windows back once (`_ensemble_multichain_setup!`).
    if n_chains > 1
        n_dim = length(target.params.layout.unfrozen_idx)
        nw = max(n_walkers, 2 * n_dim + 2)
        fps = [_ensemble_fingerprint(target, nw, n_burnin, thinning, seed + c - 1,
                                     n_chains, init, prune_stranded, n_scouts, p_de)
               for c in 1:n_chains]
        cks = Any[resume ? read_checkpoint(_chain_checkpoint_path(ck_path, c),
                                           "ensemble", fps[c]) : nothing
                  for c in 1:n_chains]
        rv = _ensemble_multichain_setup!(target, cks, max(n_steps, 2),
                                         n_burnin * thinning)
        tasks = Vector{Task}(undef, n_chains)
        for c in 1:n_chains
            ck_c = ck_path === nothing ? nothing : _chain_checkpoint_path(ck_path, c)
            tasks[c] = Threads.@spawn try
                _ensemble_chain(target, n_walkers, n_steps,
                    n_burnin, thinning, init, seed + c - 1, n_chains,
                    show_progress && c == 1, ck_c, ck_every, resume;
                    pre = (fp = fps[c], ck = cks[c]), rendezvous = rv, prune_stranded,
                    n_scouts, de_move = p_de, check_drift = false)
            finally
                rv === nothing || _rendezvous_leave!(rv)
            end
        end
        chs = [fetch(t) for t in tasks]
        # The drift check runs on each chain's walkers (one ensemble each) and
        # reports them together, once.
        drift = _ensemble_drift_chains([c.info.walker_cube for c in chs],
                                       target.params; sampler = "ensemble",
                                       de_move = p_de)
        out = MCMCChains.chainscat([_strip_walker_cube(c) for c in chs]...)
        return MCMCChains.setinfo(out, (; drift))
    end
    return _ensemble_chain(target, n_walkers, n_steps, n_burnin, thinning, init,
                           seed, n_chains, show_progress, ck_path, ck_every, resume;
                           prune_stranded, n_scouts, de_move = p_de)
end

# Chain `c`'s state file in a multi-chain run: "dir/ensemble_state.jls" ->
# "dir/ensemble_state_chain2.jls".
function _chain_checkpoint_path(path::AbstractString, c::Int)
    base, ext = splitext(path)
    return string(base, "_chain", c, ext)
end

# What an ensemble checkpoint must match (src/checkpoint.jl). The walkers are
# stored in the sampling coordinates, so the circular windows and whether the
# target is unconstrained are part of it. So is turning the stranded-walker
# pruning off, or scouting with other than 16; the defaults leave the
# fingerprint as it was before either existed. `de_move` enters when it is not
# 0, the stretch-only kernel every checkpoint before the move ran.
_ensemble_fingerprint(target::NereusTarget, n_walkers, n_burnin, thinning, seed,
                      n_chains, init, prune_stranded::Bool = true,
                      n_scouts::Int = 16, de_move::Float64 = DE_MOVE_DEFAULT) =
    run_fingerprint(target.params, target.data; n_walkers, n_burnin, thinning,
        seed, n_chains, init = init === nothing ? nothing : copy(init),
        unconstrained = target.transform !== nothing,
        windows = circular_windows(target.params),
        (prune_stranded ? (;) : (; prune_stranded = false))...,
        (n_scouts == 16 ? (;) : (; n_scouts))..., _de_fp(de_move)...)

"""
    _ensemble_recut!(x, params, transform, rv) -> Bool

The burn-in re-cut of one chain's walkers `x` ([`recut_walkers!`](@ref)). With
a rendezvous `rv` (several chains over one `Params`) the chains re-cut
together: the last to arrive pools every chain's walkers, re-cuts from all of
them and writes each chain's back. Returns whether any walker moved (then the
caller redoes its log-densities).
"""
function _ensemble_recut!(x::Matrix{Float64}, params::Params, transform,
                          rv::Union{Nothing,_Rendezvous})
    rv === nothing && return recut_walkers!(x, params, transform)
    return _rendezvous!(rv, x) do xs
        X = reduce(hcat, xs)
        recut_walkers!(X, params, transform) || return false
        o = 0
        for xc in xs
            xc .= view(X, :, (o + 1):(o + size(xc, 2)))
            o += size(xc, 2)
        end
        return true
    end
end

"""
    _ensemble_multichain_setup!(target, cks, n_steps, n_burn_steps)
        -> Union{Nothing,_Rendezvous}

Before the chains of a multi-chain ensemble start: put back the shared windows
of a resumed run (those of the furthest chain; any chain saved in older charts
is relabelled into them, `_ensemble_rechart_checkpoint!`), and return the
rendezvous for the burn-in re-cuts still to come -- `nothing` when there are
none, or when the resumed chains are out of step (different numbers of re-cuts
left), in which case the windows stay as they are: a chain still re-cutting
would move them under one that is not.
"""
function _ensemble_multichain_setup!(target::NereusTarget, cks, n_steps::Int,
                                     n_burn_steps::Int)
    params = target.params
    isempty(seam_indices(params)) && return nothing
    trs = target.transform === nothing ? () : (target.transform,)
    steps = Int[ck === nothing ? 0 : ck.step::Int for ck in cks]
    have = Bool[ck !== nothing && hasproperty(ck, :windows) for ck in cks]
    if any(have)
        ref = argmax([have[c] ? steps[c] : -1 for c in eachindex(cks)])
        wref = cks[ref].windows
        for c in eachindex(cks)
            have[c] && cks[c].windows != wref || continue
            _ensemble_rechart_checkpoint!(target, cks[c], wref)
        end
        _restore_shared_windows!(params, wref, trs)
        params.layout.spin_axis .= cks[ref].spin_axis
    end
    B = n_burn_steps
    left = [count(s -> _pt_recut_due(s, B), (st + 1):min(B, n_steps)) for st in steps]
    return allequal(left) && first(left) > 0 ? _Rendezvous(length(cks)) : nothing
end

# A chain's checkpointed walkers and kept draws, saved in the charts
# `ck.windows`, moved into the charts `wref`: through the bounded coordinates,
# relabelled by whole periods. Through a logit that changes the walkers'
# coordinates and Jacobians, so their log-densities are redone. Serial: called
# before any chain runs.
function _ensemble_rechart_checkpoint!(target::NereusTarget, ck, wref)
    params, tr = target.params, target.transform
    trs = tr === nothing ? () : (tr,)
    L = params.layout
    circ = seam_indices(params)
    kept = ck.kept::Array{Float64, 3}
    cols = [view(ck.x, :, j) for j in axes(ck.x, 2)]
    append!(cols, view(kept, :, j, s) for j in axes(kept, 2), s in axes(kept, 3))
    _restore_shared_windows!(params, ck.windows, trs)
    W = [tr === nothing ? Vector{Float64}(y) : transform_inverse(Vector{Float64}(y), tr)
         for y in cols]
    _restore_shared_windows!(params, wref, trs)
    for (y, w) in zip(cols, W)
        for i in circ
            lo, hi = bounds(L.unfrozen_priors[i])
            w[i] = circular_relabel(w[i], lo, hi)
        end
        y .= tr === nothing ? w : transform_forward(w, tr)
    end
    for j in axes(ck.x, 2)
        lp = LogDensityProblems.logdensity(target, ck.x[:, j])
        ck.lastll[j] = isfinite(lp) ? lp : -1e300
    end
    return ck
end

# One stretch-move step of the whole ensemble, in place: each half of the
# walkers moved in turn against the other. AffineInvariantMCMC.sample's inner
# loop, written the same way so that it draws the same numbers in the same order
# (`rand(rng, n)` fills a vector differently from n scalar draws) and rounds the
# same way. `lastll` holds each walker's log-density and is updated with it.
function _stretch_step!(logpost, x::Matrix{Float64}, lastll::Vector{Float64},
                        rng::AbstractRNG, divisions, a::Float64 = 2.0)
    for (active, inactive) in divisions
        zs = map(u -> ((a - 1) * u + 1)^2 / a, rand(rng, length(active)))
        proposals = map(i -> zs[i] * x[:, active[i]] +
                             (1 - zs[i]) * x[:, rand(rng, inactive)], 1:length(active))
        newll = map(logpost, proposals)
        for (j, w) in enumerate(active)
            z = zs[j]
            logratio = (size(x, 1) - 1) * log(z) + newll[j] - lastll[w]
            if log(rand(rng)) < logratio
                lastll[w] = newll[j]
                x[:, w] = proposals[j]
            end
        end
    end
    return x
end

# One step of the whole ensemble with the DE move mixed in (`de_move` > 0): each
# half moved in turn against the other, each walker by a DE move with
# probability `p_de` and by the stretch move otherwise, all its proposals
# evaluated, then accepted in walker order, as the stretch-only step does. A DE
# proposal is symmetric given the other half (src/samplers/de_move.jl), so its
# ratio is the posterior ratio alone; the stretch keeps its z^(d-1). ε's scale
# is the other half's spread, γ's base 2.38/√(2d). Proposals are built in `ws`,
# allocated once per chain.
struct _EnsembleDEWork
    P::Matrix{Float64}
    lq::Vector{Float64}
    newll::Vector{Float64}
    scale::Vector{Float64}
end
_EnsembleDEWork(nd::Int, n_walkers::Int) =
    _EnsembleDEWork(Matrix{Float64}(undef, nd, cld(n_walkers, 2)),
                    zeros(cld(n_walkers, 2)), zeros(cld(n_walkers, 2)), zeros(nd))

function _ensemble_step!(ws::_EnsembleDEWork, logpost, x::Matrix{Float64},
                         lastll::Vector{Float64}, rng::AbstractRNG, divisions,
                         p_de::Float64, a::Float64 = 2.0)
    nd = size(x, 1)
    base = _de_gamma0(nd)
    for (active, inactive) in divisions
        lo, hi = first(inactive), last(inactive)
        @inbounds for d in 1:nd
            ws.scale[d] = _de_spread(view(x, d, inactive))
        end
        for (i, w) in enumerate(active)
            col = view(ws.P, :, i)
            if _de_chosen(rng, p_de)
                j, k = _de_pair(rng, lo, hi)
                _de_fill!(col, x, w, j, k, _de_gamma(rng, base), ws.scale, rng)
                ws.lq[i] = 0.0
            else
                z = ((a - 1) * rand(rng) + 1)^2 / a
                p = rand(rng, inactive)
                @inbounds for d in 1:nd
                    col[d] = z * x[d, w] + (1 - z) * x[d, p]
                end
                ws.lq[i] = (nd - 1) * log(z)
            end
        end
        for i in eachindex(active)
            ws.newll[i] = logpost(view(ws.P, :, i))
        end
        for (i, w) in enumerate(active)
            if log(rand(rng)) < ws.lq[i] + ws.newll[i] - lastll[w]
                lastll[w] = ws.newll[i]
                @inbounds for d in 1:nd
                    x[d, w] = ws.P[d, i]
                end
            end
        end
    end
    return x
end

# The two halves the stretch move alternates between, as AffineInvariantMCMC
# splits them.
function _stretch_divisions(n_walkers::Int)
    batch1 = 1:div(n_walkers, 2)
    batch2 = (div(n_walkers, 2) + 1):n_walkers
    return [(batch1, batch2), (batch2, batch1)]
end

# One ensemble run: initialise (or restore from the checkpoint), step, and
# return the kept draws as `MCMCChains.Chains`.
function _ensemble_chain(target::NereusTarget, n_walkers::Int, n_steps::Int,
                         n_burnin::Int, thinning::Int,
                         init::Union{Nothing, Vector{Float64}}, seed::Int,
                         n_chains::Int, show_progress::Bool,
                         ck_path::Union{Nothing, String}, ck_every::Float64,
                         resume::Bool; pre = nothing,
                         rendezvous::Union{Nothing,_Rendezvous} = nothing,
                         prune_stranded::Bool = true, n_scouts::Int = 16,
                         de_move::Float64 = DE_MOVE_DEFAULT, check_drift::Bool = true)
    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    rng = MersenneTwister(seed)

    n_walkers = max(n_walkers, 2 * n_dim + 2)  # emcee requirement
    n_steps = max(n_steps, 2)                  # as AffineInvariantMCMC.sample

    # Log-posterior function
    has_transform = target.transform !== nothing
    function logpost(x::AbstractVector{Float64})
        lp = LogDensityProblems.logdensity(target, x)
        return isfinite(lp) ? lp : -1e300
    end

    # --- Checkpoint / resume (src/checkpoint.jl) ------------------------
    # The walkers are stored in the sampling coordinates, so the windows of the
    # circular parameters and whether the target is unconstrained are part of
    # what the run depends on. On resume nothing is drawn: the saved state
    # replaces initialisation and the loop carries on from step `step0 + 1`.
    # With several chains `sample_ensemble` has read the checkpoint (`pre`).
    ck_fp = pre === nothing ?
        _ensemble_fingerprint(target, n_walkers, n_burnin, thinning, seed,
                              n_chains, init, prune_stranded, n_scouts, de_move) : pre.fp
    ck = pre !== nothing ? pre.ck :
         resume ? read_checkpoint(ck_path, "ensemble", ck_fp) : nothing
    step0 = ck === nothing ? 0 : ck.step::Int
    step0 <= n_steps || throw(ArgumentError(
        "the checkpoint is at step $step0; n_steps = $n_steps would end before it"))

    # Walker positions (n_dim × n_walkers), their log-densities, and the thinned
    # post-burn-in states kept for the output.
    x = Matrix{Float64}(undef, n_dim, n_walkers)
    lastll = Vector{Float64}(undef, n_walkers)
    n_saved = div(n_steps, thinning)
    kept = Array{Float64}(undef, n_dim, n_walkers, max(n_saved - n_burnin, 0))
    # Median log L at the last stranded-walker prune (`_pt_prune_stranded!`).
    prune_med = [NaN]

    if ck === nothing
        # Initialize walkers
        if init !== nothing
            # Scatter around provided init point (init is in bounded space)
            length(init) == n_dim || throw(ArgumentError(
                "init length ($(length(init))) must match n_dim ($n_dim)"))
            # Relabelled into the layout's circular windows first (src/circular.jl):
            # a reused target carries moved windows, and an angle outside its own
            # is out of support, or clamped onto the wall by transform_forward.
            x_init = circular_relabel_point!(copy(init), params)
            center = has_transform ? transform_forward(x_init, target.transform) : x_init
            for j in 1:n_walkers
                x[:, j] = center .+ 1e-4 .* randn(rng, n_dim)
            end
        else
            # EMPEROR-style: draw each walker from prior, reject invalid
            # ones (logp or logL = -Inf), redraw until all valid.
            valid = falses(n_walkers)
            max_attempts = 1000
            for attempt in 1:max_attempts
                all(valid) && break
                for j in 1:n_walkers
                    valid[j] && continue
                    x_bounded = _draw_from_prior(target, rng)
                    if has_transform
                        x[:, j] = transform_forward(x_bounded, target.transform)
                    else
                        x[:, j] = x_bounded
                    end
                    lp = logpost(x[:, j])
                    valid[j] = lp > -1e200
                end
            end
            n_valid = count(valid)
            n_valid == n_walkers || @warn "Only $n_valid / $n_walkers walkers initialized with finite posterior after $max_attempts attempts"
            # Scouts (src/samplers/scouts.jl) in place of up to half the
            # walkers, at random: the pruning can then move the rest onto a
            # mode even when no prior-drawn walker of this chain finds it.
            if n_scouts > 0
                uf = layout.unfrozen_idx
                draw!(th, r) = (th.values[uf] .= _draw_from_prior(target, r); th)
                surv = _scout_survivors(Theta{Float64}(params), target.data, rng,
                                        _prior_widths(layout); draw!, n_scouts)
                k = min(length(surv), n_walkers ÷ 2)
                for (j, v) in zip(randperm(rng, n_walkers)[1:k], surv[1:k])
                    x[:, j] = has_transform ? transform_forward(v[uf], target.transform) :
                                              v[uf]
                end
            end
        end
        for j in 1:n_walkers
            lastll[j] = logpost(x[:, j])
        end
    else
        # --- Resume: put the saved run back exactly as it stopped -------
        x .= ck.x::Matrix{Float64}
        lastll .= ck.lastll::Vector{Float64}
        copy!(rng, ck.rng::MersenneTwister)
        ck_kept = ck.kept::Array{Float64, 3}
        kept[:, :, 1:size(ck_kept, 3)] .= ck_kept
        hasproperty(ck, :prune_med) && (prune_med .= ck.prune_med)
        # Windows (and spin axis) the saved walkers live in, moved by the re-cut;
        # with several chains put back, once for all, by `sample_ensemble`.
        if pre === nothing && hasproperty(ck, :windows)
            for (nm, (lo, _)) in ck.windows
                set_circular_window!(params, findfirst(==(nm), layout.unfrozen_names),
                                     lo; transforms = has_transform ? (target.transform,) : ())
            end
            params.layout.spin_axis .= ck.spin_axis
        end
    end

    # --- Main loop ----------------------------------------------------
    divisions = _stretch_divisions(n_walkers)
    de_ws = _EnsembleDEWork(n_dim, n_walkers)
    pb = ProgressBar("ensemble"; total = n_steps, enabled = show_progress,
                     start = step0)
    last_ck = time()
    for step in (step0 + 1):n_steps
        if de_move > 0
            _ensemble_step!(de_ws, logpost, x, lastll, rng, divisions, de_move)
        else
            _stretch_step!(logpost, x, lastll, rng, divisions)
        end
        # ---- Stranded walkers (burn-in only): sample_pt_emcee's pruning, one
        # rung at β = 1. Before the re-cut, so the seam is cut from the cleaned
        # walkers. On the unconstrained target the walkers' coordinates are the
        # logit's, a line for every parameter: no angle is offset on a circle.
        prune_stranded && step % 50 == 0 && step <= (3 * n_burnin * thinning) ÷ 4 &&
            _ensemble_prune!(x, lastll, target, logpost, rng, prune_med)
        # ---- Burn-in re-cut (src/circular.jl): on pt_emcee's schedule, the
        # walkers folded, spin-flipped and every full-circle seam moved to the
        # emptiest arc, so a mode across a seam is not split between two walls
        # that stretch moves cannot carry walkers between. A move through the
        # logit changes the walkers' Jacobians: their log-densities are redone.
        # Concurrent chains share `params`, whose windows a re-cut moves: they
        # re-cut together, at a rendezvous (`_ensemble_recut!`), or -- after a
        # resume that left them out of step -- not at all.
        if step <= n_burnin * thinning && _pt_recut_due(step, n_burnin * thinning) &&
           (n_chains == 1 || rendezvous !== nothing) &&
           _ensemble_recut!(x, params, target.transform, rendezvous)
            for j in 1:n_walkers
                lastll[j] = logpost(x[:, j])
            end
        end
        if step % thinning == 0
            k = div(step, thinning) - n_burnin
            k > 0 && (kept[:, :, k] .= x)
        end
        update!(pb; n_done = step)

        # ---- Checkpoint: every `checkpoint_interval` s and on the last step
        if ck_path !== nothing && (step == n_steps || time() - last_ck >= ck_every)
            n_kept = max(div(step, thinning) - n_burnin, 0)
            write_checkpoint(ck_path, "ensemble", ck_fp, (; step, x, lastll, rng,
                kept = kept[:, :, 1:n_kept], windows = circular_windows(params),
                spin_axis = copy(params.layout.spin_axis), prune_med = copy(prune_med)))
            last_ck = time()
        end
    end
    show_progress && finish!(pb)

    # Discard burn-in and flatten across walkers
    n_keep = n_saved - n_burnin
    n_keep > 0 || throw(ArgumentError(
        "n_burnin ($n_burnin) ≥ total steps ($n_saved)"))

    # Back-transform to bounded space and flatten
    n_total = n_keep * n_walkers
    param_names = Symbol.(layout.unfrozen_names)
    mat = Matrix{Float64}(undef, n_total, n_dim)

    idx = 0
    for s in 1:n_keep
        for w in 1:n_walkers
            idx += 1
            y = kept[:, w, s]
            if has_transform
                mat[idx, :] = transform_inverse(y, target.transform)
            else
                mat[idx, :] = y
            end
        end
    end

    # The drift check reads each walker's own sequence: (step, param, walker).
    cube = permutedims(reshape(mat, n_walkers, n_keep, n_dim), (2, 3, 1))
    if check_drift
        drift = _ensemble_drift_chains([cube], params; sampler = "ensemble", de_move)
        return MCMCChains.Chains(mat, param_names; info = (; drift))
    end
    # Several chains: `sample_ensemble` checks them together, and drops this.
    return MCMCChains.Chains(mat, param_names; info = (; walker_cube = cube))
end

# A chain without the walker cube `_ensemble_chain` hands `sample_ensemble`.
_strip_walker_cube(c::MCMCChains.Chains) = MCMCChains.setinfo(c, NamedTuple())

# The ensemble's stranded walkers moved onto the ones that found the mode:
# `_pt_prune_stranded!` on a one-rung ladder at β = 1, in the sampling
# coordinates. Its mass bound needs the prior normalised there, which the
# prior times the transform's Jacobian is. The split into (log prior + log
# Jacobian, log L) is evaluated once per walker per call (`_logdensity_parts`);
# a moved walker's log-density is then redone through `logpost`, so the walkers
# left alone keep theirs bit for bit, and a call that moves none draws nothing
# from `rng`. `med` is the record the pruning keeps between calls.
function _ensemble_prune!(x::Matrix{Float64}, lastll::Vector{Float64},
                          target::NereusTarget, logpost, rng, med::Vector{Float64})
    nd, nw = size(x)
    state = Array{Float64,3}(undef, 1, nw, nd)
    lπ = Matrix{Float64}(undef, 1, nw)
    lL = Matrix{Float64}(undef, 1, nw)
    for w in 1:nw
        state[1, w, :] .= view(x, :, w)
        lπ[1, w], lL[1, w] = _logdensity_parts(target, view(x, :, w))
    end
    circ = target.transform === nothing ? circular_indices(target.params) : Int[]
    moved = _pt_prune_stranded!(state, lπ, lL, [1.0], rng;
                                evalf = y -> _logdensity_parts(target, y),
                                circ, med_logL = med)
    moved[1] == 0 && return 0
    for w in 1:nw
        any(j -> state[1, w, j] != x[j, w], 1:nd) || continue
        x[:, w] .= view(state, 1, w, :)
        lastll[w] = logpost(view(x, :, w))
    end
    return moved[1]
end
