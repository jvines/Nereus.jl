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
"""
function sample_ess(
    target::NereusTarget,
    data::Data;
    n_samples::Int = 5000,
    n_burnin::Int = 1000,
    init::Union{Nothing, Vector{Float64}} = nothing,
    seed::Int = 1,
    n_chains::Int = 1,
)
    # Multi-chain: spawn `n_chains` independent ESS chains in parallel
    # via Julia threads. Each chain gets its own seed and theta buffer
    # (captured per-call by the loglike closure), so concurrent execution
    # is race-free. Chains concatenate into a single Chains object with
    # chain IDs preserved.
    if n_chains > 1
        tasks = Vector{Task}(undef, n_chains)
        for c in 1:n_chains
            tasks[c] = Threads.@spawn sample_ess(
                target, data;
                n_samples = n_samples, n_burnin = n_burnin,
                init = init, seed = seed + c - 1, n_chains = 1)
        end
        return MCMCChains.chainscat([fetch(t) for t in tasks]...)
    end

    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    rng = MersenneTwister(seed)

    # Build Gaussian prior approximation from the actual prior bounds.
    # Center on init (or prior midpoint), width from prior support.
    centers = Vector{Float64}(undef, n_dim)
    widths = Vector{Float64}(undef, n_dim)
    for (j, ps) in enumerate(layout.unfrozen_priors)
        lo, hi = ps.lo, ps.hi
        if init !== nothing
            centers[j] = init[j]
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

    # Log-likelihood only — ESS handles the Gaussian prior internally.
    # No hard bounds here: hard walls break ESS's slice correctness.
    # Out-of-bounds samples are discarded after sampling.
    #
    # Pre-allocate the Theta buffer once and reuse across calls.
    # ESS sampling is single-threaded in EllipticalSliceSampling.jl, so
    # the captured buffer is race-free.
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
        return ll
    end

    model = ESSModel(prior, loglike)
    raw = EllipticalSliceSampling.sample(rng, model, ESS(), n_samples + n_burnin;
                                          progress=false)

    # Discard burn-in, then filter out-of-bounds samples
    samples = raw[(n_burnin + 1):end]

    # Keep only samples within prior bounds
    valid = Vector{Vector{Float64}}()
    for s in samples
        in_bounds = true
        for (j, ps) in enumerate(layout.unfrozen_priors)
            if s[j] < ps.lo || s[j] > ps.hi
                in_bounds = false
                break
            end
        end
        in_bounds && push!(valid, s)
    end

    n_oob = length(samples) - length(valid)
    n_oob > 0 && @warn "ESS: discarded $n_oob / $(length(samples)) out-of-bounds samples"

    n_keep = length(valid)
    n_keep > 0 || error("All ESS samples were out of bounds — Gaussian prior is too wide")

    param_names = Symbol.(layout.unfrozen_names)
    mat = Matrix{Float64}(undef, n_keep, n_dim)
    for (i, s) in enumerate(valid)
        mat[i, :] = s
    end

    chains = MCMCChains.Chains(mat, param_names)
    return chains
end
