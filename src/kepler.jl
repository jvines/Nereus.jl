# Kepler's equation solver.
#
# Solves M = E - e sin(E) for the eccentric anomaly E given mean anomaly M
# and eccentricity e. Uses the Danby-style higher-order Newton iteration
# from Murray & Dermott "Solar System Dynamics" §2.4 — 4th-order convergent,
# so 3-5 iterations suffice for e < 0.99 to machine precision.
#
# For tight-loop performance inside MCMC, use the in-place vector form
# `kepler_solve!(E, M, e)`.
#
# ANALYTIC GRADIENT: Instead of letting AD trace through the Newton
# iteration (which creates ~10 tape nodes per observation), we provide
# a ChainRules `rrule` that uses the implicit function theorem:
#
#   F(E, M, e) = E - e sin(E) - M = 0
#   ∂E/∂M = 1 / (1 - e cos E)
#   ∂E/∂e = sin(E) / (1 - e cos E)
#
# This replaces ~60% of the AD tape with two trig evaluations per
# observation. The forward pass still uses the Newton solver at full
# Float64 speed; only the backward pass uses the analytic derivatives.
# ForwardDiff gets the same derivatives from a `Dual` method (below).
# Backends that trace the iteration itself (ReverseDiff, Enzyme) get them
# from the Newton step taken on the iteration that converges.

using ChainRulesCore
import ForwardDiff

"""
    kepler_solve(M, e; tol=1e-10, max_iter=30, strict=false) -> E

Solve Kepler's equation `M = E - e sin(E)` for `E`, the eccentric anomaly.

Uses a 4th-order Danby iteration (Murray & Dermott 1999, §2.4). Converges
to `|M - E + e sin(E)| < tol` in a handful of iterations for bound orbits
(`0 ≤ e < 1`). For parabolic / hyperbolic orbits the caller's problem —
this routine targets the `e < 1` regime.

Generic over `M, e` element type so the routine works with `Float64`,
`Float32`, `ForwardDiff.Dual`, etc.

# Arguments
- `M`: Mean anomaly (radians). Any real value; internally normalised.
- `e`: Eccentricity. Expected in `[0, 1)`.

# Keywords
- `tol`: Residual tolerance on `|M - E + e sin(E)|`. Default `1e-10`.
- `max_iter`: Safety cap on Newton iterations. Default `30`.
- `strict`: If `true`, throw an `ErrorException` on non-convergence
  instead of warning. Useful for debugging and validation runs.
  Default `false`.

# Returns
- `E`: Eccentric anomaly (radians). Not wrapped to any canonical range —
  subsequent `sin(E)`, `cos(E)` handle periodicity naturally.

# Derivatives
`∂E/∂M = 1 / (1 - e cos E)` and `∂E/∂e = sin E / (1 - e cos E)`, the
derivatives of the solution (implicit function theorem), for every
backend: ForwardDiff through a `Dual` method that solves on the values,
ChainRules through an `rrule`, and a backend that traces the iteration
through the Newton step taken on the iteration that meets `tol`. Without
that step a traced derivative was the derivative of the last iterate, not
of the solution: at `e = 0` the initial guess `M + 0.85 sign(sin M) e`
already meets `tol`, and `∂E/∂e` came back `0.85 sign(sin M)` instead of
`sin M`.

# Convergence handling
If the iteration exhausts `max_iter` without reaching `tol`, emits a
`Base.@warn` (throttled to the first 5 occurrences per session via
`maxlog=5`) and returns the current `E`. High-eccentricity orbits
(`e > 0.95`) at certain mean anomalies can need more than the default
30 iterations; in that case either increase `max_iter` or set
`strict=true` and handle the error. Silent non-convergence was a
deliberate design change — the old behaviour returned an unconverged
`E` with no diagnostic, which caused subtle posterior bias in high-e
fits.
"""
@inline function kepler_solve(M::Real, e::Real;
                       tol::Real=1e-10, max_iter::Int=30, strict::Bool=false)
    # Promote M to a concrete float type — Julia's Irrational{:π} etc.
    # can't be used in oftype(M, 0.85) or as constructor targets.
    M = float(M)
    e = float(e)

    # Numerical-stability fold: NUTS warm-up can propose extreme Tp
    # values that produce huge |M| (= 2π·(t-Tp)/P), at which the
    # Murray-Dermott initial guess and the Newton convergence both
    # degrade. Fold M into [-π, π) for the iteration; the answer
    # E satisfies `E - e·sin E = M` because shifting M by 2πk shifts
    # E by 2πk too (sin/cos are 2π-periodic). We add the offset back
    # at the end so callers that check the round-trip identity still
    # see consistent E for any input M.
    two_π = oftype(M, 2π)
    π_t   = oftype(M, π)
    M_folded = mod(M + π_t, two_π) - π_t
    M_offset = M - M_folded
    M = M_folded

    # Initial guess (Murray & Dermott eq. 2.64 variant).
    # sign(sin(M)) picks the right hemisphere; 0.85e is a well-known
    # empirical choice that balances robustness and quick convergence.
    #
    # Note: Markley 1995 cubic-polynomial guess was tried here; it
    # converges in fewer iterations but costs ~50 ns of extra setup
    # math (pow(2/3) and sqrt) per call, which negates the iteration
    # savings for low-e orbits (e < 0.5, the common case in exoplanet
    # RV+transit fits). The Murray-Dermott guess is faster end-to-end
    # for our workload despite needing 2-3 Danby iterations on average.
    E = M + sign(sin(M)) * oftype(M, 0.85) * e
    f_last = E  # will be overwritten inside the loop; kept for the warn

    converged = false
    @inbounds for _ in 1:max_iter
        sinE, cosE = sincos(E)
        f   = E - e * sinE - M
        f_last = f
        if abs(f) < tol
            # One Newton step more, which moves E by less than tol / f'(E).
            # It is what makes an AD backend that traces this loop carry
            # the derivatives of the solution: with f(E) ≈ 0 the step's
            # derivative is ∂E = (∂M + sin E ∂e) / (1 - e cos E), whatever
            # the iterate's own was. Taken also when f is exactly 0, as at
            # e = 0, where the iterate is the initial guess and its ∂E/∂e is
            # 0.85 sign(sin M). Skipped only where f'(E) = 0 (e = 1, E = 0).
            fp = 1 - e * cosE
            iszero(fp) || (E -= f / fp)
            converged = true
            break
        end

        # Derivatives for Danby 4th-order step.
        fp   = 1 - e * cosE          # f'(E)
        fpp  = e * sinE              # f''(E)
        fppp = e * cosE              # f'''(E)

        # Three-stage higher-order Newton.
        dE1 = -f / fp
        dE2 = -f / (fp + oftype(f, 0.5) * fpp * dE1)
        dE3 = -f / (fp + oftype(f, 0.5) * fpp * dE2 + fppp * dE2 * dE2 / 6)

        E += dE3
    end

    if !converged
        if strict
            error("kepler_solve did not converge: M=$M, e=$e, " *
                  "residual=$f_last after $max_iter iterations")
        else
            @warn("kepler_solve did not converge",
                  M=M, e=e, residual=f_last, max_iter=max_iter, maxlog=5)
        end
    end
    # Add the 2πk offset back so the round-trip `E - e·sin E ≈ M_input`
    # holds for callers that check it. Doesn't affect sin(E)/cos(E).
    return E + M_offset
end

# ForwardDiff: E from the solver on the values, its partials from the implicit
# function theorem. Differentiating the iteration instead gave the derivative
# of whichever iterate met the tolerance (see the docstring), and costs a Dual
# operation per step. The value is the one the Float64 method returns, to the
# bit. Nested Duals (Hessians) recurse through `ForwardDiff.value`.
@inline function kepler_solve(M::ForwardDiff.Dual{Tg}, e::ForwardDiff.Dual{Tg};
                              kwargs...) where {Tg}
    ev = ForwardDiff.value(e)
    E = kepler_solve(ForwardDiff.value(M), ev; kwargs...)
    sinE, cosE = sincos(E)
    ∂E = (ForwardDiff.partials(M) + sinE * ForwardDiff.partials(e)) / (1 - ev * cosE)
    return ForwardDiff.Dual{Tg}(E, ∂E)
end
# A Dual with a plain number, or Duals of two tags (nested differentiation):
# promoted to one Dual type, then the method above.
@inline kepler_solve(M::ForwardDiff.Dual, e::ForwardDiff.Dual; kwargs...) =
    kepler_solve(promote(M, e)...; kwargs...)
@inline kepler_solve(M::ForwardDiff.Dual, e::Real; kwargs...) =
    kepler_solve(promote(M, e)...; kwargs...)
@inline kepler_solve(M::Real, e::ForwardDiff.Dual; kwargs...) =
    kepler_solve(promote(M, e)...; kwargs...)

"""
    kepler_solve!(E, M, e; kwargs...)

In-place vector form. Populates `E[i]` with the solution for `M[i]` and a
single scalar eccentricity `e`. Lengths of `E` and `M` must match.

# Arguments
- `E`: Output array, same length as `M`.
- `M`: Input mean anomalies.
- `e`: Scalar eccentricity.
"""
function kepler_solve!(E::AbstractVector{<:Real}, M::AbstractVector{<:Real}, e::Real;
                        kwargs...)
    @assert length(E) == length(M) "E and M must have the same length"
    @inbounds for i in eachindex(M, E)
        E[i] = kepler_solve(M[i], e; kwargs...)
    end
    return E
end

"""
    kepler_solve(M::AbstractVector, e; kwargs...) -> E

Allocating vector form. Returns a new array with element type inferred
from `M` and `e` (so autodiff `Dual` types are preserved).
"""
function kepler_solve(M::AbstractVector{<:Real}, e::Real; kwargs...)
    return kepler_solve.(M, e; kwargs...)
end

# =====================================================================
# ChainRules rrule — analytic gradient via implicit function theorem
# =====================================================================
#
# F(E, M, e) = E - e sin(E) - M = 0
# By the implicit function theorem:
#   ∂E/∂M = -∂F/∂M / (∂F/∂E) = 1 / (1 - e cos E)
#   ∂E/∂e = -∂F/∂e / (∂F/∂E) = sin(E) / (1 - e cos E)
#
# This is exact (no approximation) and replaces AD tracing through
# the entire Newton iteration. The forward pass runs the solver at
# full Float64 speed; only the pullback uses the analytic derivatives.

function ChainRulesCore.rrule(::typeof(kepler_solve), M::Real, e::Real; kwargs...)
    # Forward pass: solve at native speed (Float64, no AD overhead).
    E = kepler_solve(M, e; kwargs...)

    function kepler_solve_pullback(ΔE)
        sinE, cosE = sincos(E)
        denom = 1 - e * cosE
        # ∂E/∂M = 1 / denom
        ΔM = ΔE / denom
        # ∂E/∂e = sinE / denom
        Δe = ΔE * sinE / denom
        return ChainRulesCore.NoTangent(), ΔM, Δe
    end

    return E, kepler_solve_pullback
end
