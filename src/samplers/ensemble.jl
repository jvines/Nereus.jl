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
# is bit-identical to what the package returned; test/test_ensemble_resume.jl
# holds it to that.

using AffineInvariantMCMC
using Random
using MCMCChains

"""
    sample_ensemble(target::NereusTarget; kwargs...) -> chains

Affine-invariant ensemble MCMC (Goodman & Weare 2010), step for step the
algorithm of AffineInvariantMCMC.jl (stretch scale a = 2). Returns posterior as
`MCMCChains.Chains`.

Works in unconstrained space (with transforms) by default.

# Keywords
- `n_walkers::Int=50` — number of walkers (must be ≥ 2 × n_dim)
- `n_steps::Int=5000` — steps per walker
- `n_burnin::Int=1000` — burn-in steps to discard
- `thinning::Int=1` — thinning factor
- `init::Union{Nothing, Vector{Float64}}=nothing` — if provided, walkers scatter around this point; otherwise drawn from prior (EMPEROR-style)
- `seed::Int=1`
- `n_chains::Int=1` — independent ensembles run on Julia threads, chain `c`
  seeded with `seed + c - 1`.
- `show_progress::Bool=true` — progress bar (the first chain's, when
  `n_chains > 1`).
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
  walkers, burn-in, thinning, seed, `init`, `n_chains` or parameterisation.
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
)
    ck_path = checkpoint === nothing ? nothing : String(checkpoint)
    resume && ck_path === nothing && throw(ArgumentError(
        "resume = true needs `checkpoint`, the state file to continue from"))
    ck_every = Float64(checkpoint_interval)

    # Multi-chain: fan out `n_chains` independent ensemble runs on
    # Julia threads. Each chain re-draws its walker init and runs
    # the stretch move on its own state, so concurrent execution
    # is race-free. Useful both for wall-clock speed (Goodman-Weare
    # within-chain walker stepping is single-threaded here)
    # and for between-chain R-hat convergence diagnostics.
    if n_chains > 1
        tasks = Vector{Task}(undef, n_chains)
        for c in 1:n_chains
            ck_c = ck_path === nothing ? nothing : _chain_checkpoint_path(ck_path, c)
            tasks[c] = Threads.@spawn _ensemble_chain(target, n_walkers, n_steps,
                n_burnin, thinning, init, seed + c - 1, n_chains,
                show_progress && c == 1, ck_c, ck_every, resume)
        end
        return MCMCChains.chainscat([fetch(t) for t in tasks]...)
    end
    return _ensemble_chain(target, n_walkers, n_steps, n_burnin, thinning, init,
                           seed, n_chains, show_progress, ck_path, ck_every, resume)
end

# Chain `c`'s state file in a multi-chain run: "dir/ensemble_state.jls" ->
# "dir/ensemble_state_chain2.jls".
function _chain_checkpoint_path(path::AbstractString, c::Int)
    base, ext = splitext(path)
    return string(base, "_chain", c, ext)
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
                         resume::Bool)
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
    ck_fp = run_fingerprint(params, target.data; n_walkers, n_burnin, thinning,
        seed, n_chains, init = init === nothing ? nothing : copy(init),
        unconstrained = has_transform, windows = circular_windows(params))
    ck = resume ? read_checkpoint(ck_path, "ensemble", ck_fp) : nothing
    step0 = ck === nothing ? 0 : ck.step::Int
    step0 <= n_steps || throw(ArgumentError(
        "the checkpoint is at step $step0; n_steps = $n_steps would end before it"))

    # Walker positions (n_dim × n_walkers), their log-densities, and the thinned
    # post-burn-in states kept for the output.
    x = Matrix{Float64}(undef, n_dim, n_walkers)
    lastll = Vector{Float64}(undef, n_walkers)
    n_saved = div(n_steps, thinning)
    kept = Array{Float64}(undef, n_dim, n_walkers, max(n_saved - n_burnin, 0))

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
    end

    # --- Main loop ----------------------------------------------------
    divisions = _stretch_divisions(n_walkers)
    pb = ProgressBar("ensemble"; total = n_steps, enabled = show_progress,
                     start = step0)
    last_ck = time()
    for step in (step0 + 1):n_steps
        _stretch_step!(logpost, x, lastll, rng, divisions)
        if step % thinning == 0
            k = div(step, thinning) - n_burnin
            k > 0 && (kept[:, :, k] .= x)
        end
        update!(pb; n_done = step)

        # ---- Checkpoint: every `checkpoint_interval` s and on the last step
        if ck_path !== nothing && (step == n_steps || time() - last_ck >= ck_every)
            n_kept = max(div(step, thinning) - n_burnin, 0)
            write_checkpoint(ck_path, "ensemble", ck_fp, (; step, x, lastll, rng,
                kept = kept[:, :, 1:n_kept]))
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

    chains = MCMCChains.Chains(mat, param_names)
    return chains
end
