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

const _G_SI       = 6.6743e-11      # m³/(kg·s²)
const _M_SUN_KG   = 1.989e30        # kg
const _SEC_PER_DAY = 86400.0

# f_M [M_sun] = K[m/s]^3 · P[days] · (1-e²)^(3/2) · _F_M_FACTOR
const _F_M_FACTOR = _SEC_PER_DAY / (2π * _G_SI * _M_SUN_KG)
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
    msec_from_K(K, P, e, sin_i, M_pri) -> M_sec

Solve f(M) = M_sec³ sin³ i / (M_pri + M_sec)² for M_sec given the
observed RV semi-amplitude `K`, period `P`, eccentricity `e`, sine-
inclination `sin_i`, and primary mass `M_pri` (M_sun).

Closed-form not available in general — the equation is cubic in M_sec.
Solve via Newton iteration starting from the M_sec ≪ M_pri approximation
M_sec ≈ (f_M · M_pri²)^(1/3) / sin i. Typically converges in 3–5
iterations.

`K` in m/s, `P` in days, `M_pri` in M_sun. Returns M_sec in M_sun.

For low-mass companions (planets) the starter is already an excellent
approximation and one Newton step is enough.
"""
function msec_from_K(K::Real, P::Real, e::Real, sin_i::Real,
                     M_pri::Real)
    sin_i_safe = max(abs(sin_i), oftype(sin_i, 1e-12))
    K_safe = max(K, zero(K))
    f_M = mass_function(K_safe, P, e)
    # Reparametrize via q = M_sec / M_total ∈ [0, 1):
    #   f_M = M_sec³ sin³i / M_total²
    #   M_sec = q M_pri / (1 - q)
    #   ⇒ q³ × M_pri × sin³i / (1-q) = f_M
    #   ⇒ q³ + r q − r = 0,  with r = f_M / (M_pri sin³i) × (1-q) absorbed
    # Solve the cubic q³ × sin³i × M_pri − f_M (1-q) = 0 by damped
    # Newton starting at the small-q approximation, with backtracking
    # to keep q ∈ (0, 1).
    sin3 = sin_i_safe^3
    # Small-q starter: q ≈ (f_M / (sin³i × M_pri))^(1/3)
    r_const = f_M / (sin3 * M_pri)
    q_start = cbrt(r_const)
    q = min(max(q_start, oftype(q_start, 1e-12)), oftype(q_start, 0.999))

    # 30 Newton iterations: F(q) = q³ × sin³i × M_pri − f_M (1-q)
    #   F'(q) = 3 q² × sin³i × M_pri + f_M
    @inbounds for _ in 1:30
        F  = q^3 * sin3 * M_pri - f_M * (1 - q)
        Fp = 3 * q^2 * sin3 * M_pri + f_M
        Δ  = F / Fp
        # Backtrack: keep q ∈ (1e-12, 0.999)
        step = Δ
        for _bt in 1:20
            q_new = q - step
            if q_new > oftype(q, 1e-12) && q_new < oftype(q, 0.999)
                q = q_new
                break
            end
            step /= 2
        end
    end
    # Convert back to M_sec
    return q * M_pri / (1 - q)
end

# ---------------------------------------------------------------------
# Kepler's third law: a from P, M_total
# ---------------------------------------------------------------------

"""
    a_from_P(P_days, M_total) -> a_AU

Semi-major axis of the relative orbit in AU, given orbital period in
days and total mass in solar masses (Kepler's third law).

  a³ [AU³] = M_total [M_sun] · (P [yr])²
"""
function a_from_P(P_days::Real, M_total::Real)
    P_yr = P_days / 365.25
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
    gost_window_avg_pm(orb, gost, M_sec) -> (μra_avg, μdec_avg)

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

The reference epoch `t_ref` is set to the mean of `gost.t`, which
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
function gost_window_avg_pm(orb, gost, M_sec::Real)
    Δq = gost_5param_fit(orb, gost, M_sec)
    return (Δq[4], Δq[5])
end

"""
    gost_5param_fit(orb, gost, M_sec; t_ref=mean(gost.t)) -> Δq::Vector{T}

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

This is the Gaia-side htof primitive. `gost_window_avg_pm` is the
backward-compat 2-tuple wrapper returning only `(Δμα*, Δμδ)`.

Uniform per-transit weight; per-transit AL precision is a future
refinement (G-mag-dependent — small relative to the published 5×5
catalog covariance for characterization-quality orbits).
"""
function gost_5param_fit(orb, gost, M_sec::Real;
                          t_ref::Union{Nothing, Real} = nothing)
    n = n_gost(gost)
    T = typeof(M_sec)
    n >= 5 || return zeros(T, 5)

    t_ref_gost = t_ref === nothing ? sum(gost.t) / n : Float64(t_ref)

    A11 = zero(T); A12 = zero(T); A13 = zero(T); A14 = zero(T); A15 = zero(T)
    A22 = zero(T); A23 = zero(T); A24 = zero(T); A25 = zero(T)
    A33 = zero(T); A34 = zero(T); A35 = zero(T)
    A44 = zero(T); A45 = zero(T)
    A55 = zero(T)
    v1 = zero(T); v2 = zero(T); v3 = zero(T); v4 = zero(T); v5 = zero(T)

    @inbounds for j in 1:n
        Δra, Δdec = star_reflex_offset(orb, gost.t[j], M_sec)
        s, c = sincos(gost.psi[j])
        plxf = gost.parallax_factor[j]
        dt   = (gost.t[j] - t_ref_gost) / 365.25
        x1 = s
        x2 = c
        x3 = plxf
        x4 = s * dt
        x5 = c * dt
        d_eta = Δra * s + Δdec * c

        v1 += x1 * d_eta
        v2 += x2 * d_eta
        v3 += x3 * d_eta
        v4 += x4 * d_eta
        v5 += x5 * d_eta

        A11 += x1 * x1
        A12 += x1 * x2
        A13 += x1 * x3
        A14 += x1 * x4
        A15 += x1 * x5
        A22 += x2 * x2
        A23 += x2 * x3
        A24 += x2 * x4
        A25 += x2 * x5
        A33 += x3 * x3
        A34 += x3 * x4
        A35 += x3 * x5
        A44 += x4 * x4
        A45 += x4 * x5
        A55 += x5 * x5
    end

    A = Matrix{T}(undef, 5, 5)
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
`planet_Omega(theta, k)`. Time-anchor decoding follows the
parametrization: `:Mo` and `:Tp` map straight through to a Tp value,
`:Tc` is converted via the existing `tc_to_tp` helper using the
already-decoded `(e, ω)`.
"""
function _planet_orbit(theta::Theta, k::Int, M_pri::Real, plx::Real)
    # 4-arg form is a thin wrapper over the 5-arg form with t_ref=0.
    # Only correct when parametrization.time === :Tp (so the time anchor
    # is already a Tp); other anchors (Mo, Tc) need t_ref to convert.
    return _planet_orbit(theta, k, M_pri, plx, 0.0)
end

"""
    _planet_orbit(theta, k, M_pri, plx, t_ref) -> (orbit, M_sec)

Form that takes the data reference epoch explicitly. Preferred over
the no-`t_ref` form — the no-`t_ref` form assumes `t_ref = 0` and is
only correct when `parametrization.time = :Tp` (so the time anchor is
already a Tp).
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
    # path absorbs this in `true_anomaly` via the same min/max clamp.
    # PlanetOrbits' KepOrbit needs e ∈ [0, 1) so we apply the clamp here
    # too. No-op for valid samples; protects autodiff during line search.
    e_safe = min(max(e, zero(e)), oftype(e, 0.9999))
    time_kind = theta.params.config.parametrization.time
    Tp = _t_anchor_to_tp(time_kind, t_anc, P, e_safe, ω, t_ref)
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

@inline function _t_anchor_to_tp(time_kind::Symbol, t_anc::Real, P::Real,
                                  e::Real, ω::Real, t_ref::Real)
    if time_kind === :Mo
        return mo_to_tp(t_anc, P, t_ref)
    elseif time_kind === :Tp
        return t_anc
    else  # :Tc
        return tc_to_tp(t_anc, P, e, ω)
    end
end
