# Pathfinder warmup wrapper.
#
# Pathfinder (Zhang, Carpenter, Gelman, Vehtari 2022, JMLR 23, 306;
# arXiv:2108.03782) finds good NUTS / Pigeons initial points by
# running L-BFGS to the MAP, then constructing multivariate-normal
# approximations to the posterior at each L-BFGS iterate, and picking
# the one with maximum effective sample size (ESS) under importance
# sampling.
#
# Wraps `Pathfinder.jl` (Seth Axen, MIT). The wrapper exposes a clean
# Nereus-native API: pass a NereusTarget, get back initial unfrozen
# vectors (in unconstrained space) suitable for seeding NUTS or
# Pigeons walkers, plus the chosen Gaussian approximation as a
# Distributions.MvNormal.

using Pathfinder: pathfinder, multipathfinder
using Random: MersenneTwister, AbstractRNG
using Optim: LBFGS
using LineSearches: BackTracking
using Logging: Logging
using Printf
import LogDensityProblems
import Transducers: ThreadedEx, SequentialEx

"""
    pathfinder_init(target::NereusTarget;
                    n_runs::Int = 8,
                    n_draws::Int = 100,
                    seed::Int = 42,
                    quiet::Bool = true)
    -> NamedTuple{(:draws, :per_run_draws, :fit_distribution,
                    :psis_result, :pathfinder_result)}

Run multipath Pathfinder on `target`. Returns:

- `draws::Matrix{Float64}` — `n_unfrozen × n_draws` array of points in
  the unconstrained sampler space, after PSIS importance reweighting
  against the union of L-BFGS runs.
- `per_run_draws::Matrix{Float64}` — `n_unfrozen × n_runs` matrix with
  one draw from each independent L-BFGS run's MVN approximation. This
  preserves basin diversity even when `psis_result.pareto_shape > 0.7`
  collapses the IS-reweighted `draws` onto a single basin. Prefer
  this for seeding multi-chain samplers when basin coverage matters.
- `fit_distribution::Distributions.MixtureModel` — uniformly-weighted
  mixture over each run's chosen MVN approximation.
- `psis_result` — Pareto-smoothed-importance-sampling diagnostics.
  `psis_result.pareto_shape > 0.7` indicates the mixture-of-Gaussians
  approximation is a poor fit; values close to 0 indicate it works well.
- `pathfinder_result` — the raw Pathfinder.jl object.

# Arguments

- `n_runs`: number of independent L-BFGS runs from random initial
  points. Each run produces its own Gaussian approximation; the best
  is selected. Default 8 — good coverage of multimodal posteriors
  without excessive cost.
- `n_draws`: total samples to return after ESS reweighting.
- `seed`: RNG seed for the L-BFGS initial points.
- `quiet`: when `true` (default), suppresses the chatty Pathfinder /
  PSIS `@warn` output for high Pareto-k or rejected L-BFGS updates,
  and replaces it with a single one-line Nereus summary printed at
  return. The diagnostics are still available on the returned
  NamedTuple.

# Use cases

1. **NUTS warmup**: pass a column of `draws` as the `init` kwarg to
   `sample_nuts`, and use `cov(fit_distribution)` as the initial mass
   matrix.
2. **Pigeons / PT seeding**: pass `n_chains` columns of
   `per_run_draws` as the initial walkers (one per chain). When
   `n_chains == n_runs` this guarantees one chain per L-BFGS basin.
3. **OFTI initialization**: pass Pathfinder draws into the OFTI
   prior-draw step rather than uniform priors — much higher
   acceptance rate.

For multimodal posteriors (HD 159062-class), use `n_runs=16` or more
so each L-BFGS run lands in a different basin.
"""
function pathfinder_init(target::NereusTarget;
                         n_runs::Int = 8,
                         n_draws::Int = 100,
                         seed::Int = 42,
                         quiet::Bool = true)
    n_dim = LogDensityProblems.dimension(target)

    # multipathfinder expects (logp, ndim, ndraws) and we pass our target
    # directly because Pathfinder's logdensity_and_gradient interface
    # is satisfied by NereusTarget via the LogDensityProblems API.
    #
    # L-BFGS with BackTracking line search rather than the default
    # HagerZhang. NereusTarget returns `-Inf` at hard boundaries
    # (e ≥ 1 in the (sesinw, secosw) parametrization, etc.) and
    # ForwardDiff propagates `NaN` gradients through those points.
    # HagerZhang asserts `isfinite(phi_c) && isfinite(dphi_c)` and
    # crashes; BackTracking just rejects the step and shrinks.
    optimizer = LBFGS(linesearch = BackTracking())
    rng = MersenneTwister(seed)

    # Suppress Pathfinder/PSIS @warn flood — Pareto k > 1 is the rule
    # for high-dimensional trans-dim posteriors and the warning fires
    # ~once per L-BFGS run plus once for the final PSIS step. We emit
    # one Nereus-controlled summary instead.
    #
    # `executor=SequentialEx()` is required, NOT a performance choice.
    # Pathfinder defaults to `ThreadedEx()`, which dispatches each
    # L-BFGS basin to a different Julia worker thread. Under juliacall
    # (every production exoautomata worker), concurrent Julia code is
    # explicitly unsupported: PythonCall's GIL handshake and Julia's
    # thread pool deadlock — all 17 Julia threads park on futex while
    # the main Python thread blocks inside `juliacall.Main.__call__`.
    # See juliacall's own FAQ:
    #   https://juliapy.github.io/PythonCall.jl/stable/faq/#Is-PythonCall/JuliaCall-thread-safe?
    # We previously masked this with `n_threads == 1 ? Seq : Threaded`,
    # which kept the deadlock latent for any worker that ran with
    # JULIA_NUM_THREADS > 1 (the default for the exoautomata pool).
    # Sequential it is — the n_runs L-BFGS basins are quick enough
    # that the wall-time hit is acceptable, and the trans-dim PT
    # sampler downstream still benefits from `JULIA_NUM_THREADS` for
    # its per-temperature ensemble.
    exec = SequentialEx()

    result = if quiet
        Logging.with_logger(Logging.NullLogger()) do
            multipathfinder(
                target, n_draws;
                nruns = n_runs,
                rng = rng,
                optimizer = optimizer,
                executor = exec,
            )
        end
    else
        multipathfinder(
            target, n_draws;
            nruns = n_runs,
            rng = rng,
            optimizer = optimizer,
            executor = exec,
        )
    end

    # Per-L-BFGS-run draws: one column per independent run, sampled
    # from that run's MVN approximation. Preserves basin diversity
    # even when the IS-reweighted `result.draws` collapses to one mode.
    per_run_draws = Matrix{Float64}(undef, n_dim, n_runs)
    for (i, pr) in enumerate(result.pathfinder_results)
        if pr === nothing
            # Failed L-BFGS run — fall back to mean of the global mixture
            # (all-runs average) so the chain still starts in a reasonable
            # place rather than at a frozen-zero vector.
            per_run_draws[:, i] = mean(result.fit_distribution)
            continue
        end
        # Each run's `draws` is `(n_dim, n_per_run_draws)`. Take the
        # first column — it's a sample from that run's MVN.
        d = pr.draws
        per_run_draws[:, i] = size(d, 2) > 0 ? d[:, 1] : mean(pr.fit_distribution)
    end

    if quiet
        # One concise summary line. ESS / Pareto-k come from the PSIS
        # step on the mixture-of-Gaussians; these are the most useful
        # numbers for the user.
        psis = result.psis_result
        if psis !== nothing
            k = psis.pareto_shape
            assessment = k > 1     ? "very bad — basin diversity unreliable" :
                          k > 0.7   ? "bad — using per-run draws for diversity" :
                          k > 0.5   ? "ok"                                    :
                                       "good"
            @info @sprintf("Pathfinder: %d L-BFGS runs, %d draws, Pareto k = %.2f (%s)",
                            n_runs, n_draws, k, assessment)
        else
            @info @sprintf("Pathfinder: %d L-BFGS runs, %d draws (no PSIS)",
                            n_runs, n_draws)
        end
    end

    return (
        draws            = Matrix(result.draws),
        per_run_draws    = per_run_draws,
        fit_distribution = result.fit_distribution,
        psis_result      = result.psis_result,
        pathfinder_result = result,
    )
end
