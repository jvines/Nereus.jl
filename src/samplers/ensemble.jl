# Affine-invariant ensemble MCMC (emcee-style) via AffineInvariantMCMC.jl.
#
# Gradient-free, embarrassingly parallel, good for multi-modal posteriors.
# The standard sampler used in most RV exoplanet papers.
#
# Walker initialization follows astroEMPEROR: each walker is drawn
# independently from the prior, invalid draws (logp or logL = -Inf)
# are rejected and redrawn until all walkers are valid.

using AffineInvariantMCMC
using Random
using MCMCChains

"""
    sample_ensemble(target::NereusTarget; kwargs...) -> chains

Affine-invariant ensemble MCMC (Goodman & Weare 2010) via
AffineInvariantMCMC.jl. Returns posterior as `MCMCChains.Chains`.

Works in unconstrained space (with transforms) by default.

# Keywords
- `n_walkers::Int=50` — number of walkers (must be ≥ 2 × n_dim)
- `n_steps::Int=5000` — steps per walker
- `n_burnin::Int=1000` — burn-in steps to discard
- `thinning::Int=1` — thinning factor
- `init::Union{Nothing, Vector{Float64}}=nothing` — if provided, walkers scatter around this point; otherwise drawn from prior (EMPEROR-style)
- `seed::Int=1`
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
)
    # Multi-chain: fan out `n_chains` independent ensemble runs on
    # Julia threads. Each chain re-draws its walker init and runs
    # AffineInvariantMCMC on its own state, so concurrent execution
    # is race-free. Useful both for wall-clock speed (Goodman-Weare
    # within-chain walker stepping is single-threaded in the package)
    # and for between-chain R-hat convergence diagnostics.
    if n_chains > 1
        tasks = Vector{Task}(undef, n_chains)
        for c in 1:n_chains
            tasks[c] = Threads.@spawn sample_ensemble(
                target;
                n_walkers = n_walkers, n_steps = n_steps,
                n_burnin = n_burnin, thinning = thinning,
                init = init, seed = seed + c - 1, n_chains = 1)
        end
        return MCMCChains.chainscat([fetch(t) for t in tasks]...)
    end

    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    rng = MersenneTwister(seed)

    n_walkers = max(n_walkers, 2 * n_dim + 2)  # emcee requirement

    # Log-posterior function
    has_transform = target.transform !== nothing
    function logpost(x::AbstractVector{Float64})
        lp = LogDensityProblems.logdensity(target, x)
        return isfinite(lp) ? lp : -1e300
    end

    # Initialize walkers
    x0 = Matrix{Float64}(undef, n_dim, n_walkers)
    if init !== nothing
        # Scatter around provided init point (init is in bounded space)
        length(init) == n_dim || throw(ArgumentError(
            "init length ($(length(init))) must match n_dim ($n_dim)"))
        center = has_transform ? transform_forward(init, target.transform) : copy(init)
        for j in 1:n_walkers
            x0[:, j] = center .+ 1e-4 .* randn(rng, n_dim)
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
                    x0[:, j] = transform_forward(x_bounded, target.transform)
                else
                    x0[:, j] = x_bounded
                end
                lp = logpost(x0[:, j])
                valid[j] = lp > -1e200
            end
        end
        n_valid = count(valid)
        n_valid == n_walkers || @warn "Only $n_valid / $n_walkers walkers initialized with finite posterior after $max_attempts attempts"
    end

    # Run
    chain, llhoodvals = AffineInvariantMCMC.sample(
        logpost, n_walkers, x0, n_steps, thinning;
        rng = rng,
    )
    # chain is (n_dim, n_walkers, n_steps)
    n_steps_out = size(chain, 3)

    # Discard burn-in and flatten across walkers
    n_keep = n_steps_out - n_burnin
    n_keep > 0 || throw(ArgumentError(
        "n_burnin ($n_burnin) ≥ total steps ($n_steps_out)"))

    # Back-transform to bounded space and flatten
    n_total = n_keep * n_walkers
    param_names = Symbol.(layout.unfrozen_names)
    mat = Matrix{Float64}(undef, n_total, n_dim)

    idx = 0
    for s in (n_burnin + 1):n_steps_out
        for w in 1:n_walkers
            idx += 1
            y = chain[:, w, s]
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
