# Sky-plane projection of a Keplerian orbit, via PlanetOrbits.jl.
#
# Nereus does not re-implement Markley's Kepler solver, the Thiele-Innes
# transform, or the visual-orbit machinery. PlanetOrbits.jl (sefffal,
# MIT) is the canonical Julia primitive for these and is fully autodiff-
# clean (ForwardDiff, Enzyme, ChainRules extensions). We depend on it.
#
# Convention boundary:
#   - Nereus samples (P, K, e, ω, M0, i, Ω) per planet plus a system-
#     level M_pri and plx (parallax).
#   - PlanetOrbits.orbit(...) wants (M, a, e, i, ω, Ω, tp, plx) where
#     M is the *total* mass (M_pri + M_sec) and `a` is the *relative*
#     semi-major axis between the two bodies.
#
# So the projection step does:
#   1. Derive M_sec from RV semi-amplitude K via the standard mass
#      function: K = (2πG/P)^(1/3) · M_sec sin i / (M_pri+M_sec)^(2/3) /
#      sqrt(1 - e²).
#   2. Compute a from Kepler's third law a^3 = (M_pri+M_sec)·P^2 (in
#      M_sun / AU / yr units).
#   3. Convert M0 → tp via Nereus's existing time-anchor helpers.
#   4. Build a `Visual{KepOrbit}` and call `orbitsolve(orbit, t)`.
#
# Two outputs are needed downstream:
#   * Companion sky position relative to the primary (for `RelAstromData`):
#     `raoff(sol)`, `decoff(sol)`.
#   * Stellar reflex motion (for HGCA / IAD absolute astrometry):
#     `raoff(sol, M_sec)`, `decoff(sol, M_sec)`. Sign is opposite of the
#     companion's offset, scaled by M_sec / (M_pri + M_sec).
#
# All quantities are generic over element type — passing a ForwardDiff
# `Dual` for any sampled parameter propagates gradients through to the
# sky-plane offset.

using PlanetOrbits
import LinearAlgebra

# ---------------------------------------------------------------------
# Mass function: derive M_sec from K, P, e, sin i, M_pri
# ---------------------------------------------------------------------
#
#   f(M) = M_sec^3 sin^3 i / (M_pri + M_sec)^2
#        = K^3 · P · (1 - e²)^(3/2) / (2π G)
#
# In SI, K [m/s], P [s], masses in kg → f(M) [kg]. We want Nereus-
# native units: K [m/s], P [days], masses [M_sun] → f(M) [M_sun].

const _SEC_PER_DAY = 86400.0

# f_M [M_sun] = K[m/s]^3 · P[days] · (1-e²)^(3/2) · _F_M_FACTOR
# GM_sun from src/constants.jl, not a local G × a local solar mass: this file
# carried its own 1.989e30 kg, so the astrometric mass function ran on a
# different GM than the RV one in src/derived.jl.
const _F_M_FACTOR = _SEC_PER_DAY / (2π * GM_SUN_SI)
# Numeric value ≈ 1.0364e-16. Sanity: K=12.7 m/s, P=4332 d, e=0.05 (Jupiter)
# → f_M ≈ 9.18e-10 M_sun → M_sec sin i ≈ 9.72e-4 M_sun ≈ 1.02 M_J. ✓

"""
    mass_function(K, P, e) -> f_M

RV mass function f(M) = M_sec³ sin³ i / (M_pri + M_sec)² in solar masses.

`K` in m/s, `P` in days, `e` dimensionless. The mass function is the
RHS-only quantity, derived directly from observable RV properties.
"""
function mass_function(K::Real, P::Real, e::Real)
    one_minus_e2 = max(1 - e * e, zero(e))
    return _F_M_FACTOR * K^3 * P * one_minus_e2^(3/2)
end

"""
    mass_ratio_from_F(F) -> x

Positive root of `x³ − F x² − 2F x − F = 0`, the mass function in the
variables `x = M_sec / M_pri` and `F = f / (M_pri sin³ i)`
(`x³ = F (1 + x)²`), by Cardano's method.

The cubic has exactly one real root for every `F > 0`: its discriminant is
`−F² (4F + 27) < 0`. Descartes allows two further negative roots, but they are
complex, so the trigonometric branch is never needed. With `x = y + F/3` the
depressed cubic is `y³ + p y + q = 0`,

    p = −2F − F²/3,   q = −F − 2F²/3 − 2F³/27,

and `(q/2)² + (p/3)³ = F² (4F + 27) / 108` exactly, so the square root needs no
subtraction. `y = u + v` with `u = ∛(−q/2 + √D)`; the other cube root
`−q/2 − √D` cancels catastrophically at small `F`, so `v = −p/(3u)` instead.
Every term is positive: no cancellation anywhere on `F ∈ (0, ∞)`.

Branches only on the primal value, so it is type-generic and ForwardDiff-safe.
`F ≤ 0` returns 0.
"""
@inline function mass_ratio_from_F(F::T) where {T<:Real}
    F > 0 || return zero(T)
    three = T(3)
    sD = F * sqrt((4F + 27) / 108)
    mq2 = F / 2 + F^2 / three + F^3 / 27     # −q/2 > 0
    mp3 = (2F + F^2 / three) / three         # −p/3 > 0
    u = cbrt(mq2 + sD)
    return u + mp3 / u + F / three
end

"""
    msec_from_K(K, P, e, sin_i, M_pri) -> M_sec

Companion mass from the exact RV mass function
`M_sec³ sin³ i / (M_pri + M_sec)² = f(K, P, e)`, no `M_sec << M_pri`
approximation. The cubic in `x = M_sec / M_pri` is solved analytically
([`mass_ratio_from_F`](@ref), Cardano), with `F = f / (M_pri sin³ i)`.

`K` in m/s, `P` in days, `M_pri` in M_sun. Returns M_sec in M_sun.
"""
function msec_from_K(K::Real, P::Real, e::Real, sin_i::Real,
                     M_pri::Real)
    sin_i_safe = max(abs(sin_i), oftype(sin_i, 1e-12))
    K_safe = max(K, zero(K))
    f_M = mass_function(K_safe, P, e)
    F = f_M / (sin_i_safe^3 * M_pri)
    return mass_ratio_from_F(F) * M_pri
end

# ---------------------------------------------------------------------
# Kepler's third law: a from P, M_total
# ---------------------------------------------------------------------

"""
    a_from_P(P_days, M_total) -> a_AU

Semi-major axis of the relative orbit in AU, given orbital period in
days and total mass in solar masses (Kepler's third law).

  a³ [AU³] = M_total [M_sun] · (P [yr])²

The year is `KEPLER_YEAR_DAYS`, the one that makes that exact — and the one
PlanetOrbits uses when it turns `(a, M)` back into a period, so the orbit
`build_orbit` hands it has the period it was given. With the Julian year it
came out 1.89e-5 long (see the constant).
"""
function a_from_P(P_days::Real, M_total::Real)
    P_yr = P_days / KEPLER_YEAR_DAYS
    return cbrt(M_total * P_yr * P_yr)
end

# ---------------------------------------------------------------------
# Build a PlanetOrbits orbit from Nereus-native parameters
# ---------------------------------------------------------------------

"""
    build_orbit(P, e, ω, Ω, i, M_pri, M_sec, tp, plx) -> Visual{KepOrbit}

Construct a `Visual{KepOrbit}` (PlanetOrbits.jl) from Nereus-native
parameters.

# Arguments
- `P::Real`     : orbital period (days)
- `e::Real`     : eccentricity
- `ω::Real`     : argument of periastron (rad)
- `Ω::Real`     : longitude of ascending node (rad)
- `i::Real`     : inclination (rad)
- `M_pri::Real` : primary (host) mass (M_sun)
- `M_sec::Real` : companion mass (M_sun)
- `tp::Real`    : time of periastron passage (MJD)
- `plx::Real`   : parallax (mas)

Returns an autodiff-clean `Visual{KepOrbit{T}, T}` where `T` is the
common element type of the inputs.

Internally: `M_total = M_pri + M_sec`, `a = a_from_P(P, M_total)`. The
PlanetOrbits convention is M = total mass and a = relative separation.
"""
function build_orbit(P::Real, e::Real, ω::Real, Ω::Real, i::Real,
                     M_pri::Real, M_sec::Real, tp::Real, plx::Real)
    M_total = M_pri + M_sec
    a = a_from_P(P, M_total)
    return orbit(M = M_total, a = a, e = e, i = i, ω = ω, Ω = Ω,
                 tp = tp, plx = plx)
end

# ---------------------------------------------------------------------
# Sky-plane projection at a single time
# ---------------------------------------------------------------------

"""
    relastrom_offset(orbit, t) -> (Δra, Δdec)

Companion's sky-plane offset relative to the host star at time `t`
(MJD), in mas. Use this for `RelAstromData` likelihood evaluation.

Wraps `PlanetOrbits.orbitsolve(orbit, t)` followed by `raoff` / `decoff`.
"""
@inline function relastrom_offset(orb, t::Real)
    sol = orbitsolve(orb, t)
    return raoff(sol), decoff(sol)
end

"""
    star_reflex_offset(orbit, t, M_sec) -> (Δra*, Δdec)

Stellar reflex sky-plane offset at time `t` (MJD), in mas. This is the
*star's* displacement relative to the system barycentre, scaled by
`M_sec / (M_pri + M_sec)` and sign-flipped relative to the companion.

Use this for absolute astrometry likelihoods (HGCA, Hipparcos IAD,
Gaia epoch astrometry).

`M_sec` in M_sun.
"""
@inline function star_reflex_offset(orb, t::Real, M_sec::Real)
    sol = orbitsolve(orb, t)
    return raoff(sol, M_sec), decoff(sol, M_sec)
end

# ---------------------------------------------------------------------
# Hoisted reflex kernel — the same offset, with the orbit-constant part
# computed once instead of once per epoch
# ---------------------------------------------------------------------
#
# `star_reflex_offset` above is the REFERENCE and stays that way. It is also
# the wrong shape for an IAD likelihood: every one of a source's abscissae
# (824 for a Gaia DR4 source, 611 for merged Hipparcos + DR4) shares ONE orbit,
# yet `orbitsolve` re-derives that orbit's geometry at every single epoch --
# the true anomaly from the eccentric anomaly (`atan` + a second `sincos`),
# `r = p/(1 + e cos ν)`, two `OrbitSolution` structs, `totalmass`, and
# `rad2as*1e3/dist` twice. Measured over 824 abscissae that is 92 us, of which
# ~39 us is recomputing quantities that did not change.
#
# Thiele-Innes collapses all of it into four constants. With
#
#     X = cos E - e,    Y = sqrt(1 - e^2) sin E
#
# the in-plane position is r cos ν = a X and r sin ν = a Y, so substituting
# r cos(ν+ω) = a(X cos ω - Y sin ω) and r sin(ν+ω) = a(X sin ω + Y cos ω) into
# PlanetOrbits' `posx`/`posy` (orbit-keplerian.jl:798-813)
#
#     posx = r cos(ν+ω) sinΩ + r sin(ν+ω) cosi cosΩ
#     posy = r cos(ν+ω) cosΩ - r sin(ν+ω) cosi sinΩ
#
# gives posx = X·B + Y·G and posy = X·A + Y·F with the classical elements
#
#     A =  a(cos ω cosΩ - sin ω cosi sinΩ)     B = a(cos ω sinΩ + sin ω cosi cosΩ)
#     F = -a(sin ω cosΩ + cos ω cosi sinΩ)     G = a(cos ω cosi cosΩ - sin ω sinΩ)
#
# Pre-multiplying those by the reflex-and-units factor -M_sec/M_tot ·
# rad2as·1e3/dist leaves the per-epoch work at: one Kepler solve, one `sincos`,
# and four multiply-adds.
#
# This is an ALGEBRAIC identity, not an approximation, and the Kepler solve is
# still PlanetOrbits' own (`kepler_solver`, Markley for e < 1 -- the same call
# `orbitsolve` makes), so E is bit-identical. What differs is the order of the
# floating-point operations after it, so agreement with `star_reflex_offset` is
# to rounding (~1e-14 relative), not bit-for-bit. `test_reflex_kernel.jl` pins
# it to `star_reflex_offset` over the parameter ranges these fits actually
# reach, and that test is what must catch it if PlanetOrbits ever changes the
# internals this reads.

"""
    ReflexKernel

Orbit-constant part of a stellar reflex offset, in Thiele-Innes form, with the
reflex mass ratio and the AU→mas conversion already folded in. Build one per
companion per likelihood call with [`_reflex_kernel`](@ref), then evaluate it at
each epoch with [`_reflex_offset`](@ref).
"""
struct ReflexKernel{T}
    n_per_day::T    # mean motion [rad/day]
    tp::T           # epoch of periastron passage [MJD]
    e::T
    sqrt1me2::T     # sqrt(1 - e^2)
    sA::T; sB::T; sF::T; sG::T   # Thiele-Innes elements, pre-scaled [mas]
end

"""
    ReflexFallback

Carries an orbit this file has no closed form for, so it is evaluated by
`star_reflex_offset` exactly as before. Reached only if `_planet_orbit` ever
produces something other than a `Visual{KepOrbit}`, or a hyperbolic orbit.
"""
struct ReflexFallback{O,T}
    orb::O
    M_sec::T
end

"""
    _reflex_fast_applicable(orb) -> Bool

Whether `_reflex_kernel`'s closed form covers `orb`. Callers test this over the
WHOLE set of orbits before building any kernels, so that the vector they build
has one concrete element type — see `_iad_residuals!`.

False for any orbit that is not a `Visual{KepOrbit}`, and for a hyperbolic one:
`sqrt(1 - e^2)` does not exist there and PlanetOrbits switches to a different
anomaly relation. `_planet_orbit` clamps e below 1 (`_anomaly_e`) before building
the orbit, so the hyperbolic case is unreachable from the likelihood; this keeps a caller
that does not clamp on the reference path instead of handing it a NaN.
"""
_reflex_fast_applicable(orb::PlanetOrbits.VisualOrbit{<:Any,<:PlanetOrbits.KepOrbit}) =
    orb.parent.e < 1
_reflex_fast_applicable(orb) = false

"""
    _reflex_kernel(orb, M_sec) -> ReflexKernel

Hoist everything about `(orb, M_sec)` that does not depend on epoch. Defined
only where [`_reflex_fast_applicable`](@ref) holds, and returns exactly one
concrete type so a vector of these stays type-stable.
"""
function _reflex_kernel(orb::PlanetOrbits.VisualOrbit{<:Any,<:PlanetOrbits.KepOrbit},
                        M_sec::Real)
    p = orb.parent
    # -M_sec/M_tot puts the STAR opposite the companion about the barycentre
    # (PlanetOrbits scales raoff/decoff by exactly this, PlanetOrbits.jl:958-963);
    # rad2as*1e3/dist is its AU->mas, as in `raoff(::OrbitSolutionVisual)`.
    scale = -M_sec / p.M * (PlanetOrbits.rad2as * oftype(orb.dist, 1e3) / orb.dist)
    sinω, cosω = sincos(p.ω)
    a = p.a
    A =  a * (cosω * p.cosΩ - sinω * p.cosi * p.sinΩ)
    B =  a * (cosω * p.sinΩ + sinω * p.cosi * p.cosΩ)
    F = -a * (sinω * p.cosΩ + cosω * p.cosi * p.sinΩ)
    G =  a * (cosω * p.cosi * p.cosΩ - sinω * p.sinΩ)
    return ReflexKernel(promote(p.n / PlanetOrbits.year2day_julian, p.tp, p.e,
                                sqrt(1 - p.e^2),
                                scale * A, scale * B, scale * F, scale * G)...)
end

"""
    _reflex_offset(kernel, t) -> (Δra*, Δdec)

Stellar reflex offset [mas] at epoch `t` [MJD]. Equal to
`star_reflex_offset(orb, t, M_sec)` to floating-point rounding.
"""
@inline function _reflex_offset(k::ReflexKernel, t::Real)
    MA = k.n_per_day * (t - k.tp)
    _, sE, cE = _markley_sc(MA, k.e)
    X = cE - k.e
    Y = k.sqrt1me2 * sE
    return (k.sB * X + k.sG * Y, k.sA * X + k.sF * Y)
end

"""
    _markley_sc(M, e) -> (E, sin E, cos E)

Markley (1995) Kepler solver that also RETURNS the sine and cosine it already
computed, instead of making the caller take a second `sincos`.

This is PlanetOrbits' `kepler_solver(M, e, Markley())` (kepsolve-markley.jl:31)
line for line, with one change: upstream writes `f2, f3 = e .* sincos(E1)` for
its own corrector and then discards `sincos(E1)`, so `_reflex_offset` used to
call `sincos(E)` all over again — measured at 18.5% of a whole likelihood
evaluation, because it is a second libm trig call at every one of a source's 824
abscissae, 3.3 million times a fit.

`E = E1 + δ5` and `|δ5| < 1e-10` over the range these fits reach, so sin and cos
at `E` come from the angle-sum identity with a two-term series for the tiny
rotation — a truncation of order `δ5³/6 ~ 1e-31`, far below Float64 resolution.
Measured against `sincos(kepler_solver(...))`: max |Δ reflex offset| = 4.9e-17
mas, against a Gaia along-scan σ of 0.049 mas.

`E` itself is returned BIT-IDENTICAL to upstream — every operation feeding it is
the same in the same order — which is what `test_reflex_kernel.jl` pins, so an
upstream change to the corrector fails loudly here rather than drifting.
"""
@inline function _markley_sc(_M::Real, e::Real)
    M = rem2pi(_M, RoundNearest)
    T = float(promote_type(typeof(M), typeof(e)))
    if iszero(M) || iszero(e)
        s, c = sincos(T(M))
        return (T(M), s, c)
    end
    pi2 = abs2(T(pi))
    α  = (3 * pi2 + 8 * (pi2 - pi * abs(M)) / (5 * (1 + e))) / (pi2 - 6)
    d  = 3 * (1 - e) + α * e
    q  = 2 * α * d * (1 - e) - M * M
    r  = 3 * α * d * (d - 1 + e) * M + M * M * M
    w  = cbrt(abs2(abs(r) + sqrt(q * q * q + r * r)))
    E1 = (2 * r * w / @evalpoly(w, q * q, q, 1) + M) / d
    # Upstream's `e .* sincos(E1)`, keeping the sin/cos rather than dropping them.
    sE1, cE1 = sincos(E1)
    f2 = e * sE1
    f3 = e * cE1
    f0 = E1 - f2 - M
    f1 = 1 - f3
    δ3 = -f0 / (f1 - f0 * f2 / (2 * f1))
    δ4 = -f0 / @evalpoly(δ3, f1, f2 / 2, f3 / 6)
    δ5 = -f0 / @evalpoly(δ4, f1, f2 / 2, f3 / 6, -f2 / 24)
    d2 = δ5 * δ5
    sd = δ5 * (1 - d2 / 6)
    cd = 1 - d2 / 2
    return (E1 + δ5, sE1 * cd + cE1 * sd, cE1 * cd - sE1 * sd)
end

@inline _reflex_offset(k::ReflexFallback, t::Real) =
    star_reflex_offset(k.orb, t, k.M_sec)

"""
    star_reflex_pm(orbit, t, M_sec) -> (μ_ra*, μ_dec)

Stellar reflex proper motion at time `t` (MJD), in mas/yr. This is the
instantaneous reflex PM induced by the companion at this epoch.

Used by HGCA Mode A (compare instantaneous PM at three reference
epochs to catalog values). Mode B (epoch-window absorption) is Phase 2
and uses `star_reflex_offset` integrated over scan times instead.
"""
@inline function star_reflex_pm(orb, t::Real, M_sec::Real)
    sol = orbitsolve(orb, t)
    return pmra(sol, M_sec), pmdec(sol, M_sec)
end

# ---------------------------------------------------------------------
# GOST window-averaged stellar reflex PM (HGCA Mode B)
# ---------------------------------------------------------------------

"""
    gost_window_avg_pm(orb, gost, M_sec; window = nothing) -> (μra_avg, μdec_avg)

Compute the orbit's contribution to the catalog-reported (pmra, pmdec)
when fitting a five-parameter solution to the Gaia mission's actual
along-scan transits — i.e. the Gaia-window-averaged stellar reflex PM
at the GOST mean epoch. This is HGCA Mode B (Brandt 2018 ApJS 239 §4),
the upgrade over Mode A's instantaneous-PM-at-Gaia-reference-epoch
treatment that matters for orbits with `P` comparable to or shorter
than Gaia's ~3-yr mission window.

# Algorithm

For each GOST transit `i = 1..N`:
- Compute the orbit-induced stellar reflex along-scan offset
  `Δη_orbit_i = Δra·sin ψ_i + Δdec·cos ψ_i` from
  `star_reflex_offset(orb, gost.t[i], M_sec)`.
- Build the design vector
  `x_i = (sin ψ_i, cos ψ_i, plx_factor_i, sin ψ_i · dt_i, cos ψ_i · dt_i)`
  where `dt_i = (gost.t[i] − t_ref) / 365.25` in Julian years.

Then solve the 5-param linear least squares
  `(X^T X) q_orbit = X^T Δη_orbit`
for `q_orbit = (Δra0, Δdec0, Δϖ, Δμα, Δμδ)`. The `(Δμα, Δμδ)`
components are the orbit-induced *catalog-PM shift* — they are what
Gaia's 5-param fit would report as proper motion if the only signal
were the orbital reflex.

`window = (mjd_start, mjd_end)` restricts the fit to the transits inside
it. A catalogue proper motion comes from ONE data release, so the model of
it must be fitted through that release's transits only: pass
`GAIA_DR3_WINDOW` (or `GAIA_DR2_WINDOW`) — a GOST plan is usually fetched
over the whole mission, and fitting through all of it averages the reflex
over years the catalogue never saw. `nothing` uses every transit.

The reference epoch `t_ref` is set to the mean of the transits used, which
minimizes the covariance between (ra0, μα) and (dec0, μδ) and is the
empirical "best" reference for these fits (Brandt 2018 §3, htof's
`AstrometricFitter.find_optimal_central_epoch`).

Uniform per-transit weight is used. A future refinement could weight
by Gaia's per-transit AL precision (varies with G-mag) but for
characterization-quality orbits the uniform-weight approximation is
small relative to the catalog covariance.

Returns 0-magnitude PMs if the GOST data is degenerate (5×5 design
matrix not positive-definite) — extremely rare for ≥ 5 transits with
diverse scan angles, but defensive.
"""
function gost_window_avg_pm(orb, gost, M_sec::Real;
                            window::Union{Nothing, Tuple{Real, Real}} = nothing)
    Δq = gost_5param_fit(orb, gost, M_sec; window = window)
    return (Δq[4], Δq[5])
end

"""
    gost_5param_fit(orb, gost, M_sec; t_ref=mean(gost.t), window=nothing) -> Δq::Vector{T}

Forecast the orbit-induced **5-parameter astrometric shift** that Gaia's
catalog pipeline would absorb when fitting through the orbital reflex.
Returns `Δq = (Δα₀, Δδ₀, Δϖ, Δμα*, Δμδ)` in `(mas, mas, mas, mas/yr,
mas/yr)`, evaluated at reference epoch `t_ref` (default: mean of GOST
transit times — minimizes (position, PM) covariance per Brandt 2018 §3
and htof's `find_optimal_central_epoch`).

Algorithm: per-transit design vector
`x_i = (sin ψ_i, cos ψ_i, plx_factor_i, sin ψ_i · dt_i, cos ψ_i · dt_i)`
with `dt_i = (gost.t[i] − t_ref) / 365.25`. The orbit-induced along-scan
reflex `Δη_orbit_i = Δra · sin ψ + Δdec · cos ψ` is the "data" the
5-param solution would absorb. Solve `(XᵀX) Δq = Xᵀ Δη` in closed form
via Cholesky; degenerate design returns zeros.

`window = (mjd_start, mjd_end)` fits only the transits inside it (both ends
inclusive), and the default `t_ref` becomes their mean. Use the data
release's window (`GAIA_DR3_WINDOW`, `GAIA_DR2_WINDOW`) whenever the result
models a published catalogue quantity. Fewer than 5 transits in the window
returns zeros, as for a short plan; `Data` refuses such a plan up front
when a Gaia-epoch product depends on it.

`t_ref` may also be a pair `(t_α, t_δ)`: the position in each coordinate at
its own epoch, as the catalogues tabulate them (see [`scan_5param_fit`](@ref),
which this is the GOST case of).

This is the Gaia-side htof primitive. `gost_window_avg_pm` is the
backward-compat 2-tuple wrapper returning only `(Δμα*, Δμδ)`.

Uniform per-transit weight; per-transit AL precision is a future
refinement (G-mag-dependent — small relative to the published 5×5
catalog covariance for characterization-quality orbits).
"""
gost_5param_fit(orb, gost, M_sec::Real;
                t_ref::Union{Nothing, Real, Tuple{Real, Real}} = nothing,
                window::Union{Nothing, Tuple{Real, Real}} = nothing) =
    scan_5param_fit(orb, gost, M_sec; t_ref = t_ref, window = window)

# ---------------------------------------------------------------------
# Five-parameter fit of the reflex through a mission's scans
# ---------------------------------------------------------------------
#
# A catalogue proper motion (Hipparcos, Gaia DR2, Gaia DR3) is not the
# instantaneous reflex at some epoch: it is the proper motion of a linear
# five-parameter solution fitted to that mission's along-scan abscissae. So the
# model of it is that same fit, run on the reflex. One solver serves both
# missions; only where the scan geometry comes from differs:
#
#   GOST plan (Gaia)          t, ψ (sincos taken here), parallax factor; every
#                             transit weighs the same.
#   IADData (Hipparcos IAD)   t, sin ψ / cos ψ (cached at construction),
#                             parallax factor, and 1/σ² from the abscissa
#                             errors — the weights the reduction used. The
#                             abscissae themselves are never read.
#
# Both use the along-scan projection Δη = Δα*·sin ψ + Δδ·cos ψ that
# `_iad_residuals!` uses (see `along_scan_projection`), and the design columns
# are built from the same sin ψ, cos ψ, so a pure linear motion comes back
# exactly in either case — what `test_hip_epoch_models.jl` checks.

_scan_n(s::IADData) = n_iad(s)
@inline _scan_t(s::IADData, j::Int) = s.t[j]
@inline _scan_sc(s::IADData, j::Int) = (s.sinpsi[j], s.cospsi[j])
@inline _scan_plxf(s::IADData, j::Int) = s.parallax_factor[j]
@inline _scan_w(s::IADData, j::Int) = s.weight[j]
_scan_tsum(s::IADData) = sum(s.t)

# A GOST plan, or anything carrying its three columns.
_scan_n(s) = length(s.t)
@inline _scan_t(s, j::Int) = s.t[j]
@inline _scan_sc(s, j::Int) = sincos(s.psi[j])
@inline _scan_plxf(s, j::Int) = s.parallax_factor[j]
@inline _scan_w(s, j::Int) = 1.0
_scan_tsum(s) = sum(s.t)

_ref_epochs(::Nothing, t_mean::Float64) = (t_mean, t_mean)
_ref_epochs(t::Real, ::Float64) = (Float64(t), Float64(t))
_ref_epochs(t::Tuple{Real, Real}, ::Float64) = (Float64(t[1]), Float64(t[2]))

_kernel_eltype(::ReflexKernel{T}) where {T} = T
_kernel_eltype(k::ReflexFallback) = typeof(k.M_sec)

"""
    scan_5param_fit(orb, scans, M_sec; t_ref = nothing, window = nothing) -> Δq::Vector{T}

The five-parameter solution `Δq = (Δα*, Δδ, Δϖ, Δμα*, Δμδ)` in
`(mas, mas, mas, mas/yr, mas/yr)` that a catalogue's linear astrometric fit
absorbs from the stellar reflex of `(orb, M_sec)`, given the scans that fit
was made to: the model of a catalogue position and proper motion.

`scans` is a `GOSTData` scan plan (Gaia: uniform weights) or an `IADData`
(Hipparcos intermediate data: each abscissa weighted by `1/abscissa_err²`, as
in the reduction; only the scan geometry is read). Per scan `j` the fitted
model is

    η_j = Δα* sin ψ_j + Δδ cos ψ_j + Δϖ f_j
          + Δμα* sin ψ_j (t_j − t_α)/yr + Δμδ cos ψ_j (t_j − t_δ)/yr

with `f_j` the along-scan parallax factor and `yr = 365.25 d`, solved by
weighted least squares against the reflex's along-scan projection.

- `t_ref`: the epoch(s) the fitted position refers to. `nothing` is the mean
  epoch of the scans used; a number sets both coordinates; a pair `(t_α, t_δ)`
  sets each on its own, which is how the HGCA and G23H tabulate their central
  epochs (`epoch_ra_hip`, `epoch_dec_hip`, …). The proper motion does not
  depend on it — moving a reference epoch only re-mixes the position column —
  the position does.
- `window = (mjd_start, mjd_end)`: only the scans inside (both ends
  inclusive), e.g. `HIPPARCOS_WINDOW`, `GAIA_DR3_WINDOW`.

Fewer than 5 scans in the window, or a degenerate design, returns zeros.

Generic over the element type, so ForwardDiff Duals pass through: the normal
matrix is data-only and stays `Float64`, the orbit enters only through the
right-hand side. Abscissae that share an epoch exactly (a Hipparcos
great-circle's several, typically ~3 per epoch) share one Kepler solve.
"""
function scan_5param_fit(orb, scans, M_sec::Real;
                         t_ref::Union{Nothing, Real, Tuple{Real, Real}} = nothing,
                         window::Union{Nothing, Tuple{Real, Real}} = nothing)
    T = typeof(M_sec)
    if window === nothing
        lo, hi = -Inf, Inf
        n = _scan_n(scans)
        n >= 5 || return zeros(T, 5)
        t_mean = _scan_tsum(scans) / n
    else
        lo, hi = Float64(window[1]), Float64(window[2])
        n = 0; t_sum = 0.0
        @inbounds for j in 1:_scan_n(scans)
            t = _scan_t(scans, j)
            lo <= t <= hi || continue
            n += 1; t_sum += t
        end
        n >= 5 || return zeros(T, 5)
        t_mean = t_sum / n
    end
    t_ra, t_dec = _ref_epochs(t_ref, t_mean)

    # Same hoist as `_iad_residuals!`: ONE orbit, `n` scans, so reduce the
    # orbit to its epoch-independent constants once instead of re-deriving its
    # geometry inside `orbitsolve` at every scan. The fast/fallback choice is
    # made HERE, for the whole scan, and a CONCRETE kernel goes to the barrier
    # below — deciding per scan would make the kernel a `Union` and put a
    # dynamic dispatch back on the inner loop.
    return _reflex_fast_applicable(orb) ?
        _scan_5param_accumulate(_reflex_kernel(orb, M_sec), scans, M_sec, t_ra, t_dec, lo, hi) :
        _scan_5param_accumulate(ReflexFallback(orb, M_sec), scans, M_sec, t_ra, t_dec, lo, hi)
end

"""
    _scan_5param_accumulate(rk, scans, M_sec, t_ra, t_dec, lo, hi) -> Δq (5-vector)

The normal-equation accumulation and solve behind [`scan_5param_fit`](@ref).
Separate so it is a function barrier: it specialises on the concrete types of
`rk` and `scans`, which is what keeps `_reflex_offset` and the scan accessors
static calls in the loop. Scans outside `[lo, hi]` are skipped in place rather
than copied out, so a windowed fit allocates nothing more than an unwindowed
one.

For a GOST plan with one reference epoch this is the old GOST-only
accumulation operation for operation (its weight is exactly 1.0, so `w·x` is
`x` bit for bit), and its results are unchanged to the last bit.
"""
function _scan_5param_accumulate(rk, scans, M_sec::Real, t_ra::Float64, t_dec::Float64,
                                 lo::Float64, hi::Float64)
    T = promote_type(typeof(M_sec), _kernel_eltype(rk))

    # XᵀWX is data-only, so it is accumulated (and factorised) in Float64
    # whatever `T` is; only XᵀWΔη carries the orbit.
    A11 = 0.0; A12 = 0.0; A13 = 0.0; A14 = 0.0; A15 = 0.0
    A22 = 0.0; A23 = 0.0; A24 = 0.0; A25 = 0.0
    A33 = 0.0; A34 = 0.0; A35 = 0.0
    A44 = 0.0; A45 = 0.0
    A55 = 0.0
    v1 = zero(T); v2 = zero(T); v3 = zero(T); v4 = zero(T); v5 = zero(T)

    t_prev = NaN
    Δra = zero(T); Δdec = zero(T)
    @inbounds for j in 1:_scan_n(scans)
        t = _scan_t(scans, j)
        lo <= t <= hi || continue
        if t != t_prev
            o = _reflex_offset(rk, t)
            Δra = T(o[1]); Δdec = T(o[2])
            t_prev = t
        end
        s, c = _scan_sc(scans, j)
        w    = _scan_w(scans, j)
        x1 = s
        x2 = c
        x3 = _scan_plxf(scans, j)
        x4 = s * ((t - t_ra) / 365.25)
        x5 = c * ((t - t_dec) / 365.25)
        d_eta = Δra * s + Δdec * c
        wx1 = w * x1; wx2 = w * x2; wx3 = w * x3; wx4 = w * x4; wx5 = w * x5

        v1 += wx1 * d_eta
        v2 += wx2 * d_eta
        v3 += wx3 * d_eta
        v4 += wx4 * d_eta
        v5 += wx5 * d_eta

        A11 += wx1 * x1
        A12 += wx1 * x2
        A13 += wx1 * x3
        A14 += wx1 * x4
        A15 += wx1 * x5
        A22 += wx2 * x2
        A23 += wx2 * x3
        A24 += wx2 * x4
        A25 += wx2 * x5
        A33 += wx3 * x3
        A34 += wx3 * x4
        A35 += wx3 * x5
        A44 += wx4 * x4
        A45 += wx4 * x5
        A55 += wx5 * x5
    end

    A = Matrix{Float64}(undef, 5, 5)
    A[1,1]=A11; A[1,2]=A12; A[1,3]=A13; A[1,4]=A14; A[1,5]=A15
    A[2,1]=A12; A[2,2]=A22; A[2,3]=A23; A[2,4]=A24; A[2,5]=A25
    A[3,1]=A13; A[3,2]=A23; A[3,3]=A33; A[3,4]=A34; A[3,5]=A35
    A[4,1]=A14; A[4,2]=A24; A[4,3]=A34; A[4,4]=A44; A[4,5]=A45
    A[5,1]=A15; A[5,2]=A25; A[5,3]=A35; A[5,4]=A45; A[5,5]=A55
    v_vec = T[v1, v2, v3, v4, v5]

    chol = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A); check = false)
    LinearAlgebra.issuccess(chol) || return zeros(T, 5)
    return chol \ v_vec
end

# ---------------------------------------------------------------------
# Nereus-side convenience: build orbit + M_sec from a Theta
# ---------------------------------------------------------------------

"""
    _planet_orbit(theta, k, M_pri, plx) -> (orbit, M_sec)

Internal helper. Decode planet `k`'s orbital state from the current
`Theta`, derive the companion mass via the RV mass function (the
default `:K_driven` mass parametrization), and build a
`Visual{KepOrbit}` ready for `orbitsolve`.

Returns the orbit and the derived companion mass `M_sec` (M_sun) so
the caller can pass `M_sec` into `raoff(sol, M_sec)` /
`pmra(sol, M_sec)` for absolute-astrometry queries.

Inclination is read via `planet_inc(theta, k)` (radians) and Ω via
`planet_Omega(theta, k)`. The time anchor is decoded by `anchor_to_tp`
with the already-decoded `(e, ω)`.
"""
function _planet_orbit(theta::Theta, k::Int, M_pri::Real, plx::Real)
    # 4-arg form is a thin wrapper over the 5-arg form with t_ref=0.
    # Wrong under parametrization.time === :Mo, the one anchor measured from
    # t_ref; Tp, Tc and phic do not read it.
    return _planet_orbit(theta, k, M_pri, plx, 0.0)
end

"""
    _planet_orbit(theta, k, M_pri, plx, t_ref) -> (orbit, M_sec)

Form that takes the data reference epoch explicitly. Preferred over
the no-`t_ref` form, which assumes `t_ref = 0` and so is wrong under
`parametrization.time = :Mo`.
"""
function _planet_orbit(theta::Theta, k::Int, M_pri::Real, plx::Real,
                       t_ref::Real)
    P     = planet_P(theta, k)
    e, ω  = planet_e_w(theta, k)
    inc   = planet_inc(theta, k)
    Ω     = planet_Omega(theta, k)
    t_anc = planet_time_anchor(theta, k)
    # Clamp e to physical range. The (sesinw, secosw) parametrization
    # has unconstrained Cartesian priors that can produce e > 1; the RV
    # path absorbs this in `true_anomaly` via the same clamp (`_anomaly_e`).
    # PlanetOrbits' KepOrbit needs e ∈ [0, 1) so we apply the clamp here
    # too. No-op for every e in [0, 1), so the astrometric orbit is the RV's
    # (the clamp was to 0.9999, and above it the two were different orbits);
    # protects autodiff during line search.
    # `clamp` keeps e itself, partials included, at e = 0, where
    # `max(e, zero(e))` returned the constant and lost ∂/∂e.
    e_safe = _anomaly_e(e)
    pc = theta.params.config.parametrization
    Tp = anchor_to_tp(pc.time, t_anc, P, e_safe, ω, t_ref, pc.t_ref_c)
    # Get M_sec via the parametrization-aware accessor: in :K_driven
    # this internally inverts the mass function from the sampled K;
    # in :M_sec_driven it reads the directly-sampled value.
    block = theta.params.layout.planet_blocks[k]
    if block isa SB2Block
        # SB2: BOTH masses derived from (K_A, K_B, inc) — the passed M_pri is
        # ignored (M_A is not an independent input for a double-lined binary).
        # Build the orbit with (M_A, M_B) so M_total and a/R are correct, and
        # the reflex M_B/M_total = K_A/(K_A+K_B) reproduces the RV amplitudes.
        M_A_sb2, M_B_sb2 = _sb2_masses(theta, k)
        orb = build_orbit(P, e_safe, ω, Ω, inc, M_A_sb2, M_B_sb2, Tp, plx)
        M_sec = M_B_sb2
        # Luminous-companion photocenter: the observed photocenter reflex is
        # the star-A barycentric reflex scaled by (1 − f_light/f_mass),
        # f_mass = M_B/M_total = K_A/(K_A+K_B). star_reflex_* / GOST scale the
        # reflex LINEARLY in the mass argument (orbit fixed at true masses), so
        # returning an EFFECTIVE reflex mass applies the correction at this one
        # chokepoint for IAD/GOST/HGCA/G23H. relAST uses relastrom_offset and
        # is excluded (unresolved SB2). f_light=0 (dark) ⇒ factor 1.
        f_light = system_f_light(theta)
        if f_light != 0
            K_A = theta.values[block.KA]; K_B = theta.values[block.KB]
            f_mass = K_A / (K_A + K_B)
            M_sec = M_sec * (one(M_sec) - f_light / f_mass)
        end
        return orb, M_sec
    end

    M_sec = planet_M_sec(theta, k)
    orb = build_orbit(P, e_safe, ω, Ω, inc, M_pri, M_sec, Tp, plx)
    return orb, M_sec
end
