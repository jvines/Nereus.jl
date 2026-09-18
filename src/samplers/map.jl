# MAP optimization + Laplace evidence approximation via Optim.jl.
#
# Fast point estimates for quick-look analysis before running full MCMC.
# Laplace approximation gives a rough log-evidence from the Hessian at
# the MAP — useful for model comparison when NS is too expensive.

using Optim
using ADTypes: AutoForwardDiff
using ForwardDiff
using LinearAlgebra
using MCMCChains

"""
    MAPResult

Result of MAP optimization.

`sample_map` is a **point estimate**: it reports a single optimum, never a
credible interval. The `converged`/`railed` flags exist so a caller can tell
whether that point is trustworthy.

# Fields
- `x_map::Vector{Float64}` — MAP estimate (bounded/physical space)
- `log_posterior::Float64` — log-posterior at MAP
- `log_evidence_laplace::Float64` — Laplace approximation to log Z
- `hessian::Matrix{Float64}` — Hessian of -log posterior at MAP (information matrix)
- `param_names::Vector{String}` — parameter names
- `converged::Bool` — `true` only when the optimizer converged **and** no
  parameter is railed against a prior bound. A railed point is never reported
  as converged (see `railed`).
- `railed::Bool` — `true` when at least one sampled parameter sits at a prior
  bound with the objective still pushing outward (the posterior wants to leave
  the box). A railed optimum is an artefact of the prior support, not a real
  mode, so it must never be presented as a converged MAP.
- `railed_params::Vector{String}` — names of the parameters that railed
  (empty when `railed == false`). Useful for telling the user *which* prior to
  widen.
- `n_basins::Int` — number of *distinct* local optima the multistart landed in
  (clustered in normalized parameter space). `> 1` means the posterior is
  multimodal as seen by the optimizer.
- `dominance::Float64` — log-posterior gap between the reported optimum and the
  best *distinct* runner-up basin (`+Inf` when only one basin was found). A
  point estimate can only be trusted as "the" mode when it clearly dominates
  the competing basins; a small gap means the multistart found a thicket of
  comparable local modes and MAP cannot certify which (if any) is the global
  one — so `converged` is forced `false` in that case.
"""
struct MAPResult
    x_map::Vector{Float64}
    log_posterior::Float64
    log_evidence_laplace::Float64
    hessian::Matrix{Float64}
    param_names::Vector{String}
    converged::Bool
    railed::Bool
    railed_params::Vector{String}
    n_basins::Int
    dominance::Float64
end

"""
    _detect_railed(x_bounded, target, bound_rtol) -> Vector{Int}

Return the indices of unfrozen parameters that are *railed* against a prior
bound at the candidate point `x_bounded` (bounded/physical space).

A parameter is flagged only when BOTH hold:

1. it lies within `bound_rtol · (hi - lo)` of a finite bound, AND
2. the physical objective (−log-posterior, no transform Jacobian) keeps
   *improving* as the parameter moves toward that bound — i.e. the posterior
   wants to escape the box. This is checked by a one-sided finite-difference
   probe just inside the bound.

The directional test is what separates an *artefact* (LBFGS pinned in a
saturated logit corner because the true mode lies outside the prior) from a
*legitimate* optimum that simply rests on a physical floor (jitter ≈ 0,
K ≈ 0 for a non-detection) where the objective is genuinely minimised at the
boundary and the gradient points inward. The former is untrustworthy; the
latter is a real MAP and must not be flagged.

Bounds drawn from a half-open / unbounded prior (an endpoint `build_transform`
encoded as a non-finite, mapped to 0.0) are ignored: we read the true bound
from the prior, not the transform's clamp.
"""
function _detect_railed(x_bounded::AbstractVector{<:Real},
                         target::NereusTarget, bound_rtol::Float64)
    layout = target.params.layout
    priors = layout.unfrozen_priors
    n = length(x_bounded)
    railed = Int[]

    # Physical (Jacobian-free) negative log-posterior in bounded space.
    bounded_target = target.transform === nothing ? target :
        NereusTarget(target.params, target.data; unconstrained = false)
    neg_phys(x) = begin
        lp = LogDensityProblems.logdensity(bounded_target, x)
        return isfinite(lp) ? -lp : Inf
    end

    f0 = neg_phys(collect(float.(x_bounded)))
    isfinite(f0) || return railed   # can't probe a non-finite optimum

    @inbounds for i in 1:n
        ps = priors[i]
        is_fixed(ps) && continue
        lo, hi = bounds(ps)
        (isfinite(lo) && isfinite(hi)) || continue   # only fully-bounded params
        span = hi - lo
        span > 0 || continue
        tol = bound_rtol * span
        xi = float(x_bounded[i])
        # Probe step: a small fraction of the span, pointing INTO the box.
        h = max(10 * tol, 1e-6 * span)

        at_lo = (xi - lo) <= tol
        at_hi = (hi - xi) <= tol
        (at_lo || at_hi) || continue

        # Physical-floor exemption. A parameter resting on a hard
        # non-negativity floor (lower bound ≈ 0 — jitter/σ ≥ 0, K ≥ 0,
        # amplitude ≥ 0) is a LEGITIMATE optimum, not a prior-escape
        # artefact: the posterior cannot "want to leave" past zero (there is
        # no physical region below it). The inward-probe test below ALWAYS
        # fires on such a floor — inflating a variance/amplitude away from
        # zero strictly worsens an already-adequate fit, so f_in > f0
        # unconditionally — which would falsely flag the correct MAP
        # (a non-detection K≈0, or a noise model with no excess jitter) as
        # railed. So skip it. This is what the docstring promises ("a
        # parameter legitimately resting on a physical floor … is not
        # flagged"); the one-sided inward difference alone cannot deliver it
        # because a floor and a true escape look identical to it (both have
        # an inward-increasing objective). A nonzero lower bound (e.g. a
        # period bracket P ∈ [6,24], lo = 6) is a mere prior choice, NOT a
        # physical floor, so it stays subject to the directional test.
        if at_lo && abs(lo) <= 1e-6 * span
            continue
        end

        # Move inward by h and see whether the objective got worse. If
        # stepping into the box raises −log-post (f_in > f0), the optimum
        # genuinely sits at the bound with an OUTWARD gradient ⇒ railed.
        # If it drops (f_in < f0), the interior is better and this is just
        # a near-bound interior optimum (not railed).
        xprobe = collect(float.(x_bounded))
        if at_lo
            xprobe[i] = min(xi + h, hi)
        else
            xprobe[i] = max(xi - h, lo)
        end
        f_in = neg_phys(xprobe)
        # Strictly worse inside (with a tiny margin to ignore FP noise on a
        # truly flat floor) ⇒ the box is binding ⇒ railed.
        if isfinite(f_in) && f_in > f0 + 1e-6 * (1 + abs(f0))
            push!(railed, i)
        end
    end
    return railed
end

"""
    _same_basin(xa, xb, target, basin_rtol) -> Bool

Whether two bounded-space optima sit in the same local basin, judged by the
maximum per-parameter distance normalized by each prior's span. Unbounded /
half-bounded params are normalized by their finite span when available and
skipped otherwise (the bounded orbital params P/K/e dominate the basin identity
anyway).
"""
function _same_basin(xa::AbstractVector, xb::AbstractVector,
                     target::NereusTarget, basin_rtol::Float64)
    priors = target.params.layout.unfrozen_priors
    @inbounds for i in eachindex(xa)
        lo, hi = bounds(priors[i])
        (isfinite(lo) && isfinite(hi) && hi > lo) || continue
        if abs(xa[i] - xb[i]) / (hi - lo) > basin_rtol
            return false
        end
    end
    return true
end

"""
    _basin_summary(xs, fvals, keep, best, target; basin_rtol, dom_margin)
        -> (n_basins, dominance, dominant)

Cluster the kept multistart optima `xs` (bounded space, lower `fvals` = better)
into distinct basins and report:

- `n_basins`  : number of distinct basins among the kept starts,
- `dominance` : log-posterior gap between the `best` start and the best
  optimum in a *different* basin (`+Inf` if there is none),
- `dominant`  : whether that gap clears `dom_margin`.

A point-estimate MAP can only be trusted as *the* mode when it dominates the
competing basins; otherwise the optimizer found a thicket of comparable local
modes (the classic RV period/eccentricity degeneracy) and cannot certify the
global one. The caller forces `converged=false` when `!dominant`.
"""
function _basin_summary(xs::Vector{Vector{Float64}}, fvals::Vector{Float64},
                        keep::Vector{Int}, best::Int, target::NereusTarget;
                        basin_rtol::Float64, dom_margin::Float64)
    # Count distinct basins among the kept starts (greedy clustering).
    reps = Int[]
    for s in keep
        if !any(r -> _same_basin(xs[s], xs[r], target, basin_rtol), reps)
            push!(reps, s)
        end
    end
    n_basins = length(reps)

    # Best objective among kept starts that are NOT in the winner's basin.
    best_other = Inf
    for s in keep
        s == best && continue
        if !_same_basin(xs[s], xs[best], target, basin_rtol)
            best_other = min(best_other, fvals[s])
        end
    end
    # dominance = (−log-post gap) = f_other − f_best  (≥ 0 since best is min).
    dominance = isfinite(best_other) ? (best_other - fvals[best]) : Inf
    dominant = dominance >= dom_margin
    return n_basins, dominance, dominant
end

"""
    sample_map(target::NereusTarget; kwargs...) -> MAPResult

Find the MAP (Maximum A Posteriori) estimate via gradient-based
optimization and compute the Laplace approximation to the evidence.

Works in unconstrained space, back-transforms the result.

This is a **point estimate** and never returns a credible interval. The
returned `MAPResult` carries `converged` and `railed` flags so the caller can
tell whether the optimum is trustworthy: a point pinned against a prior bound
(the posterior wanting to escape the box) is reported with `railed=true` and
`converged=false`, never as a clean MAP.

# Keywords
- `init::Union{Nothing, Vector{Float64}}=nothing` — initial guess (unconstrained)
- `method::Symbol=:LBFGS` — `:LBFGS`, `:BFGS`, or `:NelderMead`
- `maxiter::Int=10000`
- `g_tol::Float64=1e-8`
- `n_starts::Int=32` — number of multistarts (fanned out across threads). A
  single start lands in a poor local mode or a bound-pinned corner; even 8 is
  too sparse to cover the RV period axis. 32 grid+eccentricity-seeded starts
  (see below) reliably visit the global Keplerian basin on the validation
  generators while staying far cheaper than a full NS / PT run. Pass
  `init=...` to seed start 1 explicitly.
- `seed::Int=42` — base RNG seed for the prior-draw multistarts.
- `bound_rtol::Float64=1e-3` — a parameter is *at* a bound when it lies within
  `bound_rtol · (hi - lo)` of either endpoint. Railing is then confirmed only
  if the objective keeps improving toward that bound (outward gradient), so a
  parameter legitimately resting on a physical floor (e.g. jitter ≈ 0 with the
  objective minimised there) is *not* flagged.
- `basin_rtol::Float64=0.02` — two optima are the *same* local basin when every
  bounded parameter agrees to within `basin_rtol · (hi - lo)`. Used to count
  distinct basins and find the runner-up for the dominance check.
- `dom_margin::Float64=50.0` — the reported optimum must beat the best
  *distinct* runner-up basin by at least `dom_margin` nats of log-posterior to
  be reported as `converged`. On a multimodal RV posterior the multistart
  often lands in a thicket of comparable local modes; a point estimate cannot
  certify which is global, so without a dominant winner `converged` is forced
  `false` (honest "not-trustworthy" rather than a silent wrong mode). The
  margin is deliberately large: a genuine high-S/N planetary signal beats its
  period/eccentricity aliases by many tens of nats (Δχ²/2 in the dozens), so a
  true global recovery clears 50 nats easily, while a merely-deep alias that
  dominates a missed-global by only ~10-40 nats does NOT — closing the residual
  "dominates the found basins but the true global was never visited" hole. The
  cost is that MAP will not certify weak or poorly-covered signals as
  converged; that is the correct, conservative behaviour for a point estimate
  (use `sample_pt` / `sample_nested` for those).
"""
function sample_map(
    target::NereusTarget;
    init::Union{Nothing, Vector{Float64}} = nothing,
    method::Symbol = :LBFGS,
    maxiter::Int = 10000,
    g_tol::Float64 = 1e-8,
    n_starts::Int = 32,
    seed::Int = 42,
    bound_rtol::Float64 = 1e-3,
    basin_rtol::Float64 = 0.02,
    dom_margin::Float64 = 50.0,
)
    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    has_transform = target.transform !== nothing

    # Objective: minimize the negative PHYSICAL-space log-posterior.
    #
    # LBFGS still works in unconstrained `y`-space (so the bounds are
    # handled by the transform and there are no boundaries to hit), but
    # the quantity we maximise is the bounded-space posterior
    #   log p(θ|D) = log_prior(θ) + log L(θ)
    # WITHOUT the inverse-transform log-Jacobian.
    #
    # `LogDensityProblems.logdensity(target, y)` on a transformed target
    # returns `lp + ll + lt + lj`, i.e. the unconstrained-space density.
    # That density's mode is NOT the physical posterior mode: the
    # bounded→unconstrained Jacobian for a logit transform,
    # `log(hi-lo) + log(s) + log(1-s)`, peaks at `s = 0.5` (the bound
    # midpoint). For `sesinw`/`secosw` that midpoint is `e ≈ 0`, so a
    # naïve MAP on the unconstrained density is pulled to low-eccentricity
    # / boundary corners and rails away from the true orbit (the HD 17289
    # failure: a literature orbit at e≈0.515 / K≈1414 m/s collapses to
    # e≈0.125 / K & P at their prior bounds with a 3553 m/s residual).
    #
    # We therefore subtract the log-Jacobian back out to recover the
    # physical log-posterior. We reuse the guarded `logdensity` (which
    # already handles all the NaN/Inf edge cases) and remove only `lj`.
    function neg_logpost(x)
        if has_transform
            lp = LogDensityProblems.logdensity(target, x)
            isfinite(lp) || return 1e300
            lj = transform_logabsdetjac_inv(x, target.transform)
            isfinite(lj) || return 1e300
            return -(lp - lj)
        else
            lp = LogDensityProblems.logdensity(target, x)
            return isfinite(lp) ? -lp : 1e300
        end
    end

    # Build the n_starts initial points up-front so we can fan out
    # in parallel. Start 1 honors the user's `init` (or prior draw with
    # the given seed); starts 2..n_starts draw from the prior with
    # derived seeds, then have their orbital periods spread on a log-grid
    # and their eccentricities pushed off the e≈0 origin. Random prior
    # draws alone almost never land a start in the true period basin (the
    # period axis is log-vast and LBFGS cannot cross period barriers) and
    # are biased toward e≈0 (uniform sesinw/secosw ⇒ p(e)∝e but the e=0
    # corner is a strong local attractor). The grid + eccentricity spread
    # systematically cover those axes so the global Keplerian basin is
    # actually visited — without the cost of a full periodogram or NS run.
    #
    # Index the period (LogUniform) and e-component (sesinw/secosw,
    # esinw/ecosw) parameters once, by name.
    nmv = layout.unfrozen_names
    period_idx = Int[i for i in 1:n_dim if startswith(nmv[i], "P_k")]
    e_idx = Int[i for i in 1:n_dim
                if startswith(nmv[i], "sesinw_k") ||
                   startswith(nmv[i], "secosw_k") ||
                   startswith(nmv[i], "esinw_k")  ||
                   startswith(nmv[i], "ecosw_k")]

    function _seed_start(xb::Vector{Float64}, s::Int, rng::AbstractRNG)
        # Spread each period across a log-grid; stagger per planet so a
        # multi-planet fit doesn't put every planet on the same period.
        for (pj, i) in enumerate(period_idx)
            lo, hi = bounds(layout.unfrozen_priors[i])
            (isfinite(lo) && isfinite(hi) && lo > 0) || continue
            denom = max(n_starts - 1, 1)
            frac = mod((s - 2) + 0.5 * (pj - 1), denom) / denom
            xb[i] = exp(log(lo) + frac * (log(hi) - log(lo)))
        end
        # Push eccentricity components off the origin on a fraction of the
        # starts so the e=0 attractor doesn't capture every restart. Keep
        # within the unit disk (e ≤ ~0.81 ⇒ |component| ≤ 0.9 each).
        if !isempty(e_idx) && iseven(s)
            for i in e_idx
                lo, hi = bounds(layout.unfrozen_priors[i])
                r = 0.9 * (2 * rand(rng) - 1)
                xb[i] = clamp(r, lo, hi)
            end
        end
        return xb
    end

    inits = Vector{Vector{Float64}}(undef, n_starts)
    if init === nothing
        base_rng = MersenneTwister(seed)
        xb = _draw_from_prior(target, base_rng)
        inits[1] = has_transform ? transform_forward(xb, target.transform) : xb
    else
        inits[1] = copy(init)
    end
    for s in 2:n_starts
        rng_s = MersenneTwister(seed + s)
        xb = _draw_from_prior(target, rng_s)
        xb = _seed_start(xb, s, rng_s)
        inits[s] = has_transform ? transform_forward(xb, target.transform) : xb
    end

    # Select optimizer
    optimizer = if method === :LBFGS
        Optim.LBFGS()
    elseif method === :BFGS
        Optim.BFGS()
    else
        Optim.NelderMead()
    end

    opts = Optim.Options(;
        iterations = maxiter,
        g_tol = g_tol,
        show_trace = false,
    )

    function _run_one_start(x0::Vector{Float64})
        if method === :NelderMead
            res = Optim.optimize(neg_logpost, x0, optimizer, opts)
        else
            F0 = neg_logpost(x0)
            od = Optim.OnceDifferentiable(neg_logpost, x0, F0,
                                           zeros(n_dim), AutoForwardDiff())
            res = Optim.optimize(od, x0, optimizer, opts)
        end
        return res
    end

    # Fan out across threads when n_starts > 1 and we have threads.
    # Keep the convergence + log-post info from each start so we can
    # pick the global best.
    results = Vector{Any}(undef, n_starts)
    if n_starts > 1 && Threads.nthreads() > 1
        tasks = [Threads.@spawn _run_one_start(inits[s]) for s in 1:n_starts]
        for s in 1:n_starts
            results[s] = fetch(tasks[s])
        end
    else
        for s in 1:n_starts
            results[s] = _run_one_start(inits[s])
        end
    end

    # Candidate selection.
    #
    # Report the global-minimum-objective finite start — that is the best
    # posterior value the multistart found, i.e. the MAP estimate. We do
    # NOT silently swap to a worse non-railed point: if the best point is
    # railed or non-dominant we still report it, but flag it (railed /
    # converged=false) so the caller knows it isn't trustworthy. This is
    # more honest than handing back a different, worse point that merely
    # looks clean.
    fvals = Float64[Optim.minimum(r) for r in results]
    xs_bounded = Vector{Vector{Float64}}(undef, n_starts)
    for s in 1:n_starts
        xo = Optim.minimizer(results[s])
        xs_bounded[s] = has_transform ? transform_inverse(xo, target.transform) : xo
    end
    finite_ok = [isfinite(fvals[s]) for s in 1:n_starts]
    keep = findall(finite_ok)

    best = if !isempty(keep)
        keep[argmin(fvals[keep])]
    else
        argmin(fvals)   # all non-finite; report the least-bad
    end
    result = results[best]

    x_opt = Optim.minimizer(result)
    # Reported log-posterior is the PHYSICAL-space posterior at the MAP
    # (the optimizer objective), i.e. log_prior + log_likelihood with no
    # transform Jacobian.
    log_post = -Optim.minimum(result)

    # Railing test on the reported point: a bound-pinned corner (e.g. K
    # railed to 0 with jitter soaking up the signal) has a vanishing
    # saturated-logit y-gradient, so `Optim.converged` reports `true` even
    # though the box is binding. The directional bound test catches it.
    railed_idx = _detect_railed(xs_bounded[best], target, bound_rtol)
    railed = !isempty(railed_idx)

    # Multimodality / dominance diagnostic over the finite starts. A MAP
    # point estimate is only trustworthy as "the" mode when it clearly
    # dominates the competing basins; otherwise the multistart found a
    # thicket of comparable local modes (RV period/eccentricity degeneracy)
    # and we cannot certify the global one — flag it not-converged rather
    # than presenting a silent wrong mode. (This cannot detect a global mode
    # that NO start visited; that residual risk is why MAP is a quick-look
    # estimator, not a posterior — see the docstring.)
    n_basins, dominance, dominant = if isempty(keep)
        (0, 0.0, false)
    else
        _basin_summary(xs_bounded, fvals, keep, best, target;
                       basin_rtol = basin_rtol, dom_margin = dom_margin)
    end

    # A railed optimum is never "converged" (the box is binding, the point is
    # a prior-support artefact); a non-dominant optimum is never "converged"
    # (the mode is ambiguous). Either way `converged=false` is the honest call.
    converged = Optim.converged(result) && !railed && dominant

    # Hessian at MAP for Laplace approximation.
    #
    # The optimizer maximises the physical posterior (no Jacobian), but
    # the Laplace evidence is the Gaussian-integral approximation of
    #   Z = ∫ p(θ) L(θ) dθ.
    # When a transform is present that integral is evaluated in
    # unconstrained `y`-space, where the integrand is the FULL
    # unconstrained density `f_unc(y) = log_prior + log_lik + log_J`
    # (the Jacobian is exactly the change-of-variables factor that keeps
    # the integral equal to the physical evidence). So the Laplace term
    # must use the unconstrained density's value and curvature at the
    # MAP point — NOT the physical posterior used for the search.
    #
    # The Hessian and the subsequent eigvals can fail (NaN/Inf entries,
    # LAPACK error) when the likelihood landscape is degenerate at the
    # optimum — e.g. flat ridges or a kernel parameter at its bound. In
    # those cases we return NaN for the Laplace evidence and a zero
    # Hessian, but still report the MAP point itself, since the
    # optimizer succeeded.
    neg_logdensity_unc(x) = begin
        lp = LogDensityProblems.logdensity(target, x)
        return isfinite(lp) ? -lp : 1e300
    end
    H, log_ev_laplace = try
        Hloc = ForwardDiff.hessian(neg_logdensity_unc, x_opt)
        if any(!isfinite, Hloc)
            (zeros(n_dim, n_dim), NaN)
        else
            eigs = eigvals(Hloc)
            log_dens_unc = -neg_logdensity_unc(x_opt)
            ev = if all(isfinite, eigs) && all(eigs .> 0) && isfinite(log_dens_unc)
                log_dens_unc + n_dim / 2 * log(2π) - sum(log.(eigs)) / 2
            else
                NaN
            end
            (Hloc, ev)
        end
    catch
        (zeros(n_dim, n_dim), NaN)
    end

    # MAP in bounded space — already computed for the railing test.
    x_bounded = xs_bounded[best]
    railed_names = String[layout.unfrozen_names[i] for i in railed_idx]

    if railed
        @warn("sample_map: MAP optimum is railed against a prior bound " *
              "for $(railed_names) — the posterior wants to leave the box. " *
              "Reporting converged=false / railed=true. This point is a " *
              "prior-support artefact, not a mode; widen the offending prior " *
              "(or seed `init`) before trusting it.")
    elseif !dominant && !isempty(keep)
        # Informational (not a fault): MAP is a point estimate and simply
        # can't certify a mode on a multimodal posterior. Use @info so the
        # internal callers that only want `x_map` (PT `:map_scatter` init,
        # detrend noise-only fits) don't spam @warn on every call, while the
        # not-trustworthy verdict still surfaces in the returned flags + log.
        @info("sample_map: the MAP optimum does not dominate the competing " *
              "local basins (found $n_basins distinct basins; best beats the " *
              "runner-up by only $(round(dominance, digits=2)) < " *
              "$(round(dom_margin, digits=2)) nats). The posterior is " *
              "multimodal and this point estimate cannot certify the global " *
              "mode — reporting converged=false. Use a global sampler " *
              "(sample_pt / sample_nested) or seed `init` near the expected " *
              "orbit.")
    end

    return MAPResult(
        x_bounded,
        log_post,
        isfinite(log_ev_laplace) ? log_ev_laplace : NaN,
        H,
        layout.unfrozen_names,
        converged,
        railed,
        railed_names,
        n_basins,
        dominance,
    )
end

"""Print MAP result summary."""
function Base.show(io::IO, r::MAPResult)
    flags = String[]
    r.railed && push!(flags, "RAILED at $(r.railed_params)")
    r.n_basins > 1 && push!(flags,
        "$(r.n_basins) basins, dominance=$(round(r.dominance, digits=2)) nats")
    status = isempty(flags) ? "converged=$(r.converged)" :
             "converged=$(r.converged); " * join(flags, "; ")
    println(io, "MAP Result ($status)")
    println(io, "  log p(θ|D) = $(round(r.log_posterior, digits=2))")
    println(io, "  log Z (Laplace) ≈ $(round(r.log_evidence_laplace, digits=2))")
    for (i, name) in enumerate(r.param_names)
        @printf(io, "  %-20s  %.6f\n", name, r.x_map[i])
    end
end
