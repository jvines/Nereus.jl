# Elliptical slice sampling via EllipticalSliceSampling.jl.
#
# Designed for models with Gaussian process priors. ESS exploits the
# Gaussian prior structure for efficient gradient-free sampling.
#
# Two modes:
#   1. Standalone: approximate all priors as Gaussian (useful for
#      characterization near a known mode).
#   2. GP-only: sample GP hyperparameters with ESS while holding other
#      parameters fixed (for use in Gibbs-like schemes).

using EllipticalSliceSampling
using Distributions: MvNormal
using Random
using MCMCChains

"""
    sample_ess(target::NereusTarget, data::Data; kwargs...) -> chains

Elliptical slice sampling (Murray, Adams & MacKay 2010) via
EllipticalSliceSampling.jl. Returns posterior as `MCMCChains.Chains`.

Uses a Gaussian approximation to the prior centered on `init` (or
the prior midpoint) with diagonal covariance from the prior widths.
Best for characterization near a known mode, especially with GP
noise models.

Works in bounded space (`unconstrained=false`).

# Keywords
- `n_samples::Int=5000` — number of samples
- `n_burnin::Int=1000` — burn-in to discard
- `init::Union{Nothing, Vector{Float64}}=nothing` — center of Gaussian prior approximation
- `seed::Int=1`
- `checkpoint=nothing` — path of a state file. The sampler writes its whole
  state there (the step count, the RNG, the current draw with its
  log-likelihood, and the draws kept so far) at the end of the run and every
  `checkpoint_interval` seconds during it, replacing the file atomically. With
  `n_chains > 1` each chain has its own file, `<stem>.chain<c><ext>`. `run_job`
  sets it to `ess_state.jls` in `output_dir` unless the job gives one.
- `checkpoint_interval::Real=900` — seconds between checkpoints during the run.
- `resume::Bool=false` — continue from `checkpoint` instead of initialising:
  a run that was killed, or a finished one that needs more samples.
  `n_samples` is the new TOTAL after burn-in, counted from the start of the
  original run; the draws already kept stay, and the new ones are appended.
  The continuation is bit-identical to an uninterrupted run of `n_samples`.
  Refused, with the differences listed, if the checkpoint came from different
  data, priors, parameters, burn-in, seed, chain count, `init`, or a different
  Gaussian or bounds (circular windows moved since).
"""
function sample_ess(
    target::NereusTarget,
    data::Data;
    n_samples::Int = 5000,
    n_burnin::Int = 1000,
    init::Union{Nothing, Vector{Float64}} = nothing,
    seed::Int = 1,
    n_chains::Int = 1,
    checkpoint::Union{Nothing,AbstractString,Symbol} = nothing,
    checkpoint_interval::Real = 900.0,
    resume::Bool = false,
)
    runs = _sample_ess(target, data; n_steps = n_burnin + n_samples, n_burnin,
                       init, seed, n_chains, checkpoint, checkpoint_interval, resume)

    param_names = Symbol.(target.params.layout.unfrozen_names)
    chains = map(runs) do r
        r.n_oob > 0 && @warn "ESS: discarded $(r.n_oob) / $(r.n_post) out-of-bounds samples"
        size(r.draws, 1) > 0 ||
            error("All ESS samples were out of bounds — Gaussian prior is too wide")
        MCMCChains.Chains(r.draws, param_names)
    end
    # Multi-chain: the chains concatenate into one Chains object with chain
    # IDs preserved.
    return n_chains == 1 ? only(chains) : MCMCChains.chainscat(chains...)
end

# The sampler proper: `n_steps` ESS transitions per chain, burn-in included,
# so a test can stop a run inside burn-in, where a killed run would have left
# its last checkpoint. Returns, per chain, the kept draws (draw × parameter),
# the number of post-burn-in steps taken and how many of them fell out of
# bounds.
function _sample_ess(target::NereusTarget, data::Data; n_steps::Int, n_burnin::Int,
                     init::Union{Nothing, Vector{Float64}}, seed::Int, n_chains::Int,
                     checkpoint::Union{Nothing,AbstractString,Symbol},
                     checkpoint_interval::Real, resume::Bool)
    ck_path = checkpoint === nothing ? nothing : String(checkpoint)
    resume && ck_path === nothing && throw(ArgumentError(
        "resume = true needs `checkpoint`, the state file to continue from"))

    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)

    # Build Gaussian prior approximation from the actual prior bounds.
    # Center on init (or prior midpoint), width from prior support.
    #
    # The centre is `init` relabelled into the layout's circular windows
    # (src/circular.jl): a target reused after a fit carries moved windows,
    # and an angle centred outside its own would have most of its samples
    # discarded as out of bounds below. The widths still read `init` as the
    # caller wrote it, so they do not change with the chart.
    x0 = init === nothing ? nothing : circular_relabel_point!(copy(init), params)
    centers = Vector{Float64}(undef, n_dim)
    widths = Vector{Float64}(undef, n_dim)
    for (j, ps) in enumerate(layout.unfrozen_priors)
        lo, hi = ps.lo, ps.hi
        if init !== nothing
            centers[j] = x0[j]
            # Width: small fraction of the init value or prior range.
            # ESS is for characterization — widths should be posterior-scale,
            # not prior-scale.
            if isinf(lo) || isinf(hi)
                widths[j] = max(abs(init[j]) * 0.1, 1.0)
            else
                widths[j] = max(min((hi - lo) / 20, abs(init[j]) * 0.1), 1e-10)
            end
        else
            if isinf(lo) || isinf(hi)
                centers[j] = 0.0
                widths[j] = 100.0
            else
                centers[j] = (lo + hi) / 2
                widths[j] = max((hi - lo) / 20, 1e-10)
            end
        end
    end

    prior = MvNormal(centers, widths .^ 2)
    # The box the draws are kept in: the prior bounds, i.e. the circular
    # windows as they are now.
    lo = Float64[ps.lo for ps in layout.unfrozen_priors]
    hi = Float64[ps.hi for ps in layout.unfrozen_priors]

    # --- Checkpoint / resume (src/checkpoint.jl) ------------------------
    # One state file per chain. Chain c is ESS seeded with `seed + c - 1`;
    # its draws depend on the Gaussian above, the bounds they are filtered
    # against and the burn-in, so those are fingerprinted. Every checkpoint
    # is read here, before any chain starts, so a refusal reaches the caller
    # as an ArgumentError rather than a failed task.
    paths = Union{Nothing,String}[
        ck_path === nothing ? nothing :
        n_chains == 1 ? ck_path :
        (s = splitext(ck_path); string(s[1], ".chain", c, s[2])) for c in 1:n_chains]
    fps = [run_fingerprint(params, data; n_burnin, seed, n_chains, chain = c,
                           init = init === nothing ? nothing : copy(init),
                           ess_mean = copy(centers), ess_sd = copy(widths),
                           bounds = [lo hi])
           for c in 1:n_chains]
    cks = Any[resume ? read_checkpoint(paths[c], "ess", fps[c]) : nothing
              for c in 1:n_chains]
    for ck in cks
        ck === nothing && continue
        step0 = ck.step::Int
        step0 <= n_steps || throw(ArgumentError(
            "the checkpoint is at step $step0 (burn-in included); n_burnin + " *
            "n_samples = $n_steps would end before it"))
    end

    # Multi-chain: `n_chains` independent ESS chains in parallel via Julia
    # threads. Each chain gets its own seed, theta buffer and workspace
    # (captured per-call by the loglike closure), so concurrent execution is
    # race-free; the Gaussian is immutable and only read.
    if n_chains == 1
        return [_ess_chain(target, data, prior, lo, hi, n_steps, n_burnin, seed,
                           paths[1], fps[1], cks[1], checkpoint_interval)]
    end
    tasks = Vector{Task}(undef, n_chains)
    for c in 1:n_chains
        tasks[c] = Threads.@spawn _ess_chain(target, data, prior, lo, hi, n_steps,
                                             n_burnin, seed + c - 1, paths[c],
                                             fps[c], cks[c], checkpoint_interval)
    end
    return [fetch(t) for t in tasks]
end

# One chain: builds its likelihood, puts a checkpoint's state back, runs.
function _ess_chain(target::NereusTarget, data::Data, prior, lo::Vector{Float64},
                    hi::Vector{Float64}, n_steps::Int, n_burnin::Int, seed::Int,
                    ck_path::Union{Nothing,String}, ck_fp::NamedTuple, ck,
                    checkpoint_interval::Real)
    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    rng = MersenneTwister(seed)

    # Log-likelihood only — ESS handles the Gaussian prior internally.
    # No hard bounds here: hard walls break ESS's slice correctness.
    # Out-of-bounds samples are discarded after sampling.
    #
    # Pre-allocate the Theta buffer once and reuse across calls.
    # ESS sampling is single-threaded within a chain, so the captured buffer
    # is race-free.
    theta_buf = Theta{Float64}(params)
    # ws-aware likelihood (no per-call GC + caches). ESS is single-threaded so
    # one shared workspace is race-free.
    ws_buf = PTWorkspace(params, params.config.max_kplanet,
                         length(params.config.noise_models);
                         n_obs = length(data.t_rv), n_phot = length(data.t_phot))
    function loglike(x)
        @inbounds for (j, idx) in enumerate(layout.unfrozen_idx)
            theta_buf.values[idx] = x[j]
        end
        ll = rv_log_likelihood(theta_buf, data, ws_buf)
        isfinite(ll) || return -1e300
        ll += transit_log_likelihood(theta_buf, data, ws_buf)
        # The residual maps are data too. Without this term ESS sampled a
        # Doppler-tomography target as if it had no maps at all.
        ll += tomogram_log_likelihood(theta_buf, data)
        isfinite(ll) || return -1e300
        return ll
    end
    model = ESSModel(prior, loglike)

    # Kept draws, in bounds and after burn-in, `n_dim` values each.
    kept = Float64[]

    # --- Resume: put the saved chain back exactly as it stopped ----------
    # The ESS state is the current draw, its log-likelihood and the buffer
    # the next prior draw is written into; with the RNG it is everything the
    # next step reads. Nothing here is captured by `loglike`.
    step0 = 0
    state = nothing
    if ck !== nothing
        step0 = ck.step::Int
        copy!(rng, ck.rng::MersenneTwister)
        state = ck.ess_state::EllipticalSliceSampling.ESSState
        append!(kept, ck.kept::Vector{Float64})
    end

    _ess_steps!(kept, rng, model, state, step0, n_steps, n_burnin, lo, hi,
                ck_path, ck_fp, checkpoint_interval)

    n_keep = length(kept) ÷ n_dim
    n_post = max(n_steps - n_burnin, 0)
    return (draws = permutedims(reshape(kept, n_dim, n_keep)), n_post,
            n_oob = n_post - n_keep)
end

# The step loop, behind a function barrier so `state` is concretely typed
# whether it came from a checkpoint or from the first step. Step 1 draws the
# start from the Gaussian, as `AbstractMCMC.sample` does; steps after
# `n_burnin` are kept when they lie inside [lo, hi].
function _ess_steps!(kept::Vector{Float64}, rng::MersenneTwister, model, state,
                     step0::Int, n_steps::Int, n_burnin::Int, lo::Vector{Float64},
                     hi::Vector{Float64}, ck_path::Union{Nothing,String},
                     ck_fp::NamedTuple, checkpoint_interval::Real)
    mcmc_step = EllipticalSliceSampling.AbstractMCMC.step
    sampler = ESS()
    last_ck = time()
    for step in (step0 + 1):n_steps
        draw, state = step == 1 ? mcmc_step(rng, model, sampler) :
                                  mcmc_step(rng, model, sampler, state)
        if step > n_burnin
            in_bounds = true
            @inbounds for j in eachindex(lo)
                if draw[j] < lo[j] || draw[j] > hi[j]
                    in_bounds = false
                    break
                end
            end
            in_bounds && append!(kept, draw)
        end

        # ---- Checkpoint: every `checkpoint_interval` s and on the last step
        # (src/checkpoint.jl); ESS has no early stop. -----------------------
        if ck_path !== nothing && (step == n_steps ||
                                   time() - last_ck >= checkpoint_interval)
            write_checkpoint(ck_path, "ess", ck_fp,
                             (; step, rng, ess_state = state, kept))
            last_ck = time()
        end
    end
    return kept
end
