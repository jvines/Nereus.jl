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
                    quiet::Bool = true,
                    n_scouts::Int = 16)
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
- `n_scouts`: the L-BFGS runs start from scouted prior draws, each run from
  one of those the stranded-walker bound keeps, picked at random
  (src/samplers/scouts.jl), rather than from Pathfinder's own uniform draws
  on (-2, 2) in the unconstrained coordinates. From those, all 8 runs on an
  RV + IAD companion with free e stopped on a plateau at e = 0.885, 34,000
  nats below the mode, and the draws were all there. `0` keeps Pathfinder's
  starts.
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
                         quiet::Bool = true,
                         n_scouts::Int = 16)
    n_scouts >= 0 || throw(ArgumentError("n_scouts must be ≥ 0"))
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

    run(init) = if quiet
        Logging.with_logger(Logging.NullLogger()) do
            multipathfinder(
                target, n_draws;
                nruns = n_runs,
                init = init,
                rng = rng,
                optimizer = optimizer,
                executor = exec,
            )
        end
    else
        multipathfinder(
            target, n_draws;
            nruns = n_runs,
            init = init,
            rng = rng,
            optimizer = optimizer,
            executor = exec,
        )
    end
    # The starts: scouted prior draws (src/samplers/scouts.jl) in the
    # coordinates Pathfinder runs in, or its own with `n_scouts = 0`.
    init1 = n_scouts == 0 ? nothing :
        [target.transform === nothing ? x : transform_forward(x, target.transform)
         for x in _cold_starts(target, n_runs, rng, n_scouts)]
    result = run(init1)

    # Angles. A full-circle angle is charted on one walled period, a polar
    # angle on one too, first [-π/2, 3π/2], whose seam is the far image of the
    # equator. L-BFGS started on one side of a mode across a seam climbs into
    # the wall (its logit coordinate runs off to ±∞) and fits half a mode. So
    # the first pass is a pilot: its draws place each seam in their emptiest
    # arc (a polar one's, of the draws and their mirror images) and fold the
    # polar angles (`recut_walkers!`, which moves the target's windows), and if
    # a seam moved or a polar angle railed, Pathfinder runs again in the new
    # charts, each run started from the pilot run's draw.
    init2 = _pathfinder_recut!(result, target, n_runs)
    init2 === nothing || (result = run(init2))

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

"""
    _pathfinder_recut!(result, target, n_runs) -> Union{Nothing,Vector{Vector{Float64}}}

The angle check after a Pathfinder pilot `result`. Pools the draws (the
importance-resampled ones and every run's own), folds the polar angles and
moves each full-circle or polar seam to the emptiest arc of the pool (a polar
one's of the pool and its mirror image; [`recut_walkers!`](@ref), which moves
`target`'s windows and transform). Returns `nothing` when nothing
needs a second pass, else one start per run for it: each run's first draw (the
resampled draws stand in for a failed run), in the new charts. A second pass is
needed when a seam moved, or when more than 10 % of the draws of a polar angle
sit within 0.05 rad of its chart's walls (railed; about 1.6 % would for draws
spread over the whole circle).
"""
function _pathfinder_recut!(result, target::NereusTarget, n_runs::Int)
    params, tr = target.params, target.transform
    cinc = cyclic_inclination_indices(params)
    isempty(cinc) && isempty(circular_indices(params)) && return nothing
    blocks = Matrix{Float64}[Matrix{Float64}(result.draws)]
    first_col = Int[]
    o = size(blocks[1], 2)
    for pr in result.pathfinder_results
        if pr === nothing || size(pr.draws, 2) == 0
            push!(first_col, 0)
            continue
        end
        push!(blocks, Matrix{Float64}(pr.draws))
        push!(first_col, o + 1)
        o += size(pr.draws, 2)
    end
    D = reduce(hcat, blocks)
    size(D, 2) == 0 && return nothing
    L = params.layout
    railed = false
    if !isempty(cinc)
        near = zeros(Int, length(cinc))
        for j in axes(D, 2)
            x = tr === nothing ? D[:, j] : transform_inverse(D[:, j], tr)
            for (m, d) in enumerate(cinc)
                lo, hi = bounds(L.unfrozen_priors[d])
                (x[d] - lo < 0.05 || hi - x[d] < 0.05) && (near[m] += 1)
            end
        end
        railed = any(>(0.1 * size(D, 2)), near)
    end
    w0 = circular_windows(params)
    recut_walkers!(D, params, tr)
    (railed || circular_windows(params) != w0) || return nothing
    n_res = size(blocks[1], 2)
    return [Vector{Float64}(D[:, first_col[i] > 0 ? first_col[i] :
                                 (n_res > 0 ? mod1(i, n_res) : 1)])
            for i in 1:min(n_runs, length(first_col))]
end

"""
    _seam_pilot!(target, seed) -> Bool

The circular windows placed by a pathfinder pilot ([`pathfinder_init`](@ref),
which re-cuts the seams from its own draws), for the samplers that would
otherwise start in the layout's charts: NUTS and pt_hmc without their pt_emcee
warm start, whose chains re-cut the seams themselves only once their adaptation
has run to a phase boundary (`_hmc_recut!`). They walk in the logit chart,
where a seam is a wall at ±∞, and a mode across it is two slivers, one per
wall, that no chain crosses
(NUTS on Mo ≈ 0 without a moved chart: 358° ± 2° against the true 7° ± 5°).
A polar angle's chart is placed the same way: its first chart's seam is the
mirror image of i = 90°, which an edge-on posterior's mirror copy straddles.
Returns whether a window moved. A pilot that fails leaves the charts as they
were, with a warning.
"""
function _seam_pilot!(target::NereusTarget, seed::Integer)
    params = target.params
    (target.transform === nothing || isempty(seam_indices(params))) && return false
    w0 = circular_windows(params)
    try
        pathfinder_init(target; n_runs = 4, n_draws = 200, seed = Int(seed))
    catch err
        err isa InterruptException && rethrow()
        @warn "the seam pilot (pathfinder) failed: the circular windows stay " *
              "where they were, and a posterior across a seam will be cut by it" exception = err
        return false
    end
    return circular_windows(params) != w0
end

"""
    _center_windows_on!(params, x; transforms = ()) -> x

Every full-circle angle's window moved to be centred on `x`'s value (bounded
coordinates), the seam opposite it, and `x` relabelled into the new windows. A
polar angle's seam goes to the pole farther from `x` and its mirror image
([`polar_centre_lo`](@ref)): opposite `x` would be the mirror of an edge-on
value. For a sampler started from a caller's `init`: the start is the one
point known to be in the posterior.
"""
function _center_windows_on!(params::Params, x::AbstractVector; transforms = ())
    for d in seam_indices(params)
        ps = params.layout.unfrozen_priors[d]
        lo = is_cyclic_inclination(ps) ? polar_centre_lo(x[d]) : x[d] - (ps.hi - ps.lo) / 2
        set_circular_window!(params, d, lo; transforms)
    end
    return circular_relabel_point!(x, params)
end
