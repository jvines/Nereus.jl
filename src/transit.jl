using Transits: QuadLimbDark, compute
import Transits
import ForwardDiff

# Transit light-curve model.
#
# This file provides:
#   - transit_flux_uniform(z, p)    — uniform-disk transit (exact, no LD).
#   - kipping_q_to_u(q1, q2)        — Kipping 2013 reparametrisation of
#                                     quadratic LD into [0, 1]^2.
#   - transit_flux(z, p, u1, u2)    — top-level entry, currently dispatches
#                                     to uniform + small-planet correction
#                                     and raises on the full LD case.
#
# Scope note: the full Mandel & Agol 2002 / Agol 2020 quadratic LD
# treatment is intentionally deferred to a later pass, where we'll drop
# in `Limbdark.jl` (Luger group, autodiff-aware) once its API is
# verified. For now, `transit_flux_uniform` gives us correct geometry
# with zero external dependencies, sufficient to exercise the sampling
# machinery end-to-end.
#
# Coordinate conventions:
#   z : sky-projected centre-to-centre distance in units of R_star
#   p : planet radius in units of R_star (p > 0)
#   Flux is normalised so out-of-transit = 1 and the obscured fraction
#   is subtracted from 1.

# ---------------------------------------------------------------------
# Uniform-disk transit flux
# ---------------------------------------------------------------------

"""
    transit_flux_uniform(z, p) -> F

Normalised flux during transit of an opaque circular planet of radius
`p` (in stellar radii) in front of a uniform-brightness stellar disk,
for sky-projected separation `z` (in stellar radii).

Analytical formula:

    F(z, p) = 1 - A(z, p) / π

where `A(z, p)` is the lens area (overlap of two disks with radii 1
and `p` at centre separation `z`). Three regimes:

  - `z ≥ 1 + p`       : no overlap, `F = 1`.
  - `z ≤ 1 - p`       : planet inside star, `F = 1 - p²`.
  - `z ≤ p - 1`       : star inside planet (only if `p > 1`), `F = 0`.
  - otherwise         : partial overlap, computed from the standard
                        circle-circle lens formula.

Generic over element type so autodiff traces through.
"""
function transit_flux_uniform(z::Real, p::Real)
    # No overlap.
    if z >= 1 + p
        return one(z * p)
    end
    # Star entirely behind planet (p > 1 only).
    if p > 1 && z <= p - 1
        return zero(z * p)
    end
    # Planet entirely inside stellar disk.
    if z <= 1 - p
        return 1 - p * p
    end
    # Partial overlap — circle-circle lens area.
    #
    # A = cos⁻¹((z² + 1 - p²)/(2z)) +
    #     p² cos⁻¹((z² + p² - 1)/(2 z p)) -
    #     0.5 √((1+z+p)(1+z-p)(z+p-1)(1-z+p))
    #
    # (Mandel & Agol 2002, §3, eqs. 1 for the uniform case.)
    #
    # Both acos arguments are ±1 at the contacts and round past them within an
    # ulp or two of z = 1 - p, 1 + p and p - 1, where acos threw a DomainError
    # (about one input in three at z = 1 - p + 1 ulp). Clamped: an argument in
    # [-1, 1] is untouched, and a clamped one is the contact's own value, so the
    # flux is continuous through it.
    z2 = z * z
    p2 = p * p
    k1 = acos(clamp((z2 + 1 - p2) / (2 * z), -1, 1))
    k2 = acos(clamp((z2 + p2 - 1) / (2 * z * p), -1, 1))
    k3 = sqrt(max(zero(z), (1 + z + p) * (1 + z - p) * (z + p - 1) * (1 - z + p)))
    A = k1 + p2 * k2 - k3 / 2
    return 1 - A / π
end

# ---------------------------------------------------------------------
# Kipping 2013 quadratic LD reparametrisation
# ---------------------------------------------------------------------

"""
    kipping_q_to_u(q1, q2) -> (u1, u2)

Kipping 2013 transform from `(q1, q2) ∈ [0, 1]²` to the physical
quadratic limb-darkening coefficients `(u1, u2)`:

    u1 = 2 √q1 · q2
    u2 = √q1 · (1 - 2 q2)

The `(q1, q2)` space is the natural box to place a uniform prior on
because it maps exactly to the physically allowed quadratic-LD
triangle in `(u1, u2)` space — no rejected draws, no reparametrisation
Jacobian beyond a constant.
"""
function kipping_q_to_u(q1::Real, q2::Real)
    sqrt_q1 = sqrt(q1)
    u1 = 2 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1 - 2 * q2)
    return u1, u2
end

"""
    kipping_u_to_q(u1, u2) -> (q1, q2)

Inverse of `kipping_q_to_u`. Defined for physically allowed `(u1, u2)`
pairs; returns `(q1, q2)` that may lie outside `[0, 1]²` if the input
is unphysical.
"""
function kipping_u_to_q(u1::Real, u2::Real)
    s = u1 + u2
    q1 = s * s
    q2 = u1 / (2 * s)
    return q1, q2
end

"""
    transit_flux(z, p, u1, u2) -> F

Quadratic limb-darkened transit flux: Transits.jl's `compute` (Agol 2020
formulation) as restructured in `_quad_flux`, which does not throw at the
contacts. For `u1 = u2 = 0` falls back to the exact uniform-disk formula.
"""
function transit_flux(z::Real, p::Real, u1::Real, u2::Real)
    if iszero(u1) && iszero(u2)
        return transit_flux_uniform(z, p)
    end
    ld = QuadLimbDark([u1, u2])
    return _quad_flux(ld, z, p)
end

"""
    transit_flux(ld::QuadLimbDark, z, p) -> F

Pre-built-LD overload. Use this when you have a single (u1, u2) pair
that's reused across many `(z, p)` queries — typical for the per-
instrument inner loop of the photometry likelihood. Avoids the
QuadLimbDark constructor (which calls compute_gn and a normalization
divide) on every photometric point.

The QuadLimbDark struct stores `u_n[1] = -1`, `u_n[2] = u1`, `u_n[3] = u2`,
so checking the latter two against zero recovers the uniform-disk
fast-path that `transit_flux(z, p, u1, u2)` does.
"""
function transit_flux(ld::QuadLimbDark, z::Real, p::Real)
    if iszero(ld.u_n[2]) && iszero(ld.u_n[3])
        return transit_flux_uniform(z, p)
    end
    return _quad_flux(ld, z, p)
end

# `Transits.compute(::QuadLimbDark, b, r)` (Transits 0.4.1, src/polynomial/quad.jl
# and poly.jl) restructured: the same formulas, and the same operations in the
# same order for every value that reaches the result, so the flux -- and, for
# ForwardDiff duals, its partials -- are the same to the bit (test_transit_flux.jl
# checks both). What changes: `compute_uniform`, `compute_linear` and
# `compute_quadratic` are inlined with no keyword plumbing; the triangle area
# (`sqarea_triangle`) is computed only on the partial-overlap branch, the one
# that uses it; atan(kite_area2, r² − 1 − b²) is computed once, where
# `compute_uniform` and `compute_quadratic` each computed it; and the values
# nothing uses (k, sqbrinv, sqonembmr2, onemr2mb2, onemr2pb2, kck) are gone.
# Bulirsch's `cel` is called as before. 12-15 per cent less time per in-contact
# evaluation (32-34 instead of 37-39 ns across an NGTS-33 b chord). The types
# follow `compute`: T is the float type of `b`, which is what its `one(T)`,
# `zero(T)` and `convert(T, π)` are.
#
# Two departures, both one ulp inside the second and third contacts, b = 1 − r;
# every other input is bit-identical. (1) There onembpr2/(4br) is below eps/2,
# k² = 1 + onembpr2/(4br) rounds to exactly 1, and the k² ≤ 1 branch takes kc²
# from (r − 1 + b), which is −1e-16: `compute` dies in sqrt with a DomainError
# (134 of ~378 000 random inputs), and a sampler that met that geometry aborted.
# kc² is clamped at 0, which touches no kc² ≥ 0; and k² == 1 makes `cel` replace
# kc by eps anyway, so the flux is the tangency's, continuous with its
# neighbours. (2) At r = 0.5, b = prevfloat(0.5), `compute` returns a flux off
# by 2π/3 · g₁ · norm (see the b + r == 1 branch).
function _quad_flux(ld::QuadLimbDark, b::Real, r::Real)
    T = float(typeof(b))
    if b ≥ 1 + r || iszero(r)
        return one(T)                           # unobscured
    elseif r ≥ 1 + b
        return zero(T)                          # completely obscured
    end
    g_n = ld.g_n
    r2 = r^2
    b2 = b^2

    if iszero(b)                                # annular ellipse
        onemr2 = 1 - r2
        sqrt1mr2 = sqrt(onemr2)
        flux = g_n[1] * onemr2 + 2 / 3 * g_n[2] * sqrt1mr2^3
        if ld.n_max > 1
            flux -= g_n[3] * 2 * r2 * onemr2
        end
        return flux * π * ld.norm
    end

    onembmr2 = (r + 1 - b) * (1 - r + b)
    onembmr2inv = inv(onembmr2)
    br = b * r
    fourbr = 4 * br
    fourbrinv = inv(fourbr)
    sqbr = sqrt(br)
    onembpr2 = (1 - r - b) * (1 + b + r)
    k2 = max(0, onembpr2 * fourbrinv + 1)
    if k2 > 1
        kc2 = k2 > 2 ? 1 - inv(k2) : onembpr2 * onembmr2inv
    else
        kc2 = k2 > 0.5 ? (r - 1 + b) * (b + r + 1) * fourbrinv : 1 - k2
    end
    kc2 < 0 && (kc2 = zero(kc2))                # the tangency (see above)
    kc = sqrt(kc2)

    # uniform term (compute_uniform)
    if b ≤ 1 - r
        s0 = π * (1 - r2)
        kap0 = convert(T, π)
        kite_area2 = zero(T)
        Πmkap1 = kap0                           # not read on this branch
    else
        kite_area2 = sqrt(Transits.sqarea_triangle(one(T), r, b))
        r2m1 = (r - 1) * (r + 1)
        kap0 = atan(kite_area2, r2m1 + b2)
        Πmkap1 = atan(kite_area2, r2m1 - b2)
        s0 = Πmkap1 - r2 * kap0 + 0.5 * kite_area2
    end
    flux = g_n[1] * s0
    ld.n_max == 0 && return flux * ld.norm

    # linear term (compute_linear; b ≠ 0 here)
    if b == r
        if r == 0.5                                             # case 6
            Λ1 = π - 4 / 3
        elseif r < 0.5                                          # case 5
            m = 4 * r2
            Eofk = Transits.cel(m, one(T), one(T), 1 - m)
            Em1mKdm = Transits.cel(m, one(T), one(T), zero(T))
            Λ1 = π + 2 / 3 * ((2 * m - 3) * Eofk - m * Em1mKdm) +
                 (b - r) * 4 * r * (Eofk - 2 * Em1mKdm)
        else                                                    # case 7
            m = 4 * r2
            minv = inv(m)
            Eofk = Transits.cel(minv, one(T), one(T), 1 - minv)
            Em1mKdm = Transits.cel(minv, one(T), one(T), zero(T))
            Λ1 = π + 1 / 3 * ((2 * m - 3) * Em1mKdm - m * Eofk) / r -
                 (b - r) * 2 * (2 * Eofk - Em1mKdm)
        end
    elseif b + r > 1                                            # cases 2, 8
        Πofk, Eofk, Em1mKdm = Transits.cel(k2, kc, (b - r)^2 * kc2, zero(T), one(T),
                                           one(T), 3 * kc2 * (b - r) * (b + r), kc2,
                                           zero(T))
        Λ1 = onembmr2 * (Πofk + (-3 + 6 * r2 + 2 * br) * Em1mKdm - fourbr * Eofk) /
             (3 * sqbr)
    elseif b + r < 1                                            # cases 3, 9
        bmrdbpr = (b - r) / (b + r)
        μ = 3 * bmrdbpr * onembmr2inv
        p = bmrdbpr^2 * onembpr2 * onembmr2inv
        Πofk, Eofk, Em1mKdm = Transits.cel(inv(k2), kc, p, 1 + μ, one(T), one(T),
                                           p + μ, kc2, zero(T))
        Λ1 = 2 * sqrt(onembmr2) * (onembpr2 * Πofk - (4 - 7 * r2 - b2) * Eofk) / 3
    else                                                        # b + r == 1
        # `compute` writes 2π·(r > 0.5) here, which is 2π·(r > b) when b = 1 − r,
        # and s1 below subtracts 2π·(r > b): the two cancel. At r = 0.5,
        # b = prevfloat(0.5), the third contact rounded onto this branch, they
        # disagree, and s1 was off by 2π/3 -- a flux of -0.013 where its
        # neighbours read 0.732. (r > b) is the same Bool at every other input.
        Λ1 = 2 * acos(1 - 2 * r) - 2 * π * (r > b) -
             (4 / 3 * (3 + 2 * r - 8 * r2) + 8 * (r + b - 1) * r) * sqrt(r * (1 - r))
    end
    s1 = ((1 - T(r > b)) * 2π - Λ1) / 3
    flux += g_n[2] * s1
    ld.n_max == 1 && return flux * ld.norm

    # quadratic term (compute_quadratic)
    η2 = r2 * ((r2 + b2) + b2)
    if k2 > 1
        four_pi_eta = 2 * π * (η2 - 1)
    else
        # compute_uniform's Πmkap1 where it was computed (b > 1 - r)
        Πmk = b ≤ 1 - r ? atan(kite_area2, (r - 1) * (r + 1) - b2) : Πmkap1
        four_pi_eta = 2 * (-Πmk + η2 * kap0 - 0.25 * kite_area2 * (1 + 5 * r2 + b2))
    end
    flux += g_n[3] * (2 * s0 + four_pi_eta)
    return flux * ld.norm
end

# =====================================================================
# Sky-projected separation z(t)
# =====================================================================

"""
    sky_separation(t, P, e, ω, Tp, a_Rs) -> z

Sky-projected planet-star centre distance in units of R_star at time t.
Uses the full eccentric orbit:
  r(f) = a(1-e²)/(1+e cos(f))
  z = (a/R*) * sqrt(1 - sin²(i) sin²(ω+f)) * r/(a)
    = r/R* * sqrt(1 - sin²(i) sin²(ω+f))

For the simplified case where the transit is short compared to the
orbital period, z ≈ a/R* * sqrt(sin²(ω+f)*cos²(i) + cos²(ω+f)).

We compute the exact formula using the full orbital position.
"""
function sky_separation(t::Real, P::Real, e::Real, ω::Real,
                         Tp::Real, b::Real, a_Rs::Real)
    return _sky_separation_signed(t, P, e, ω, Tp, b, a_Rs)[1]
end

# `sky_separation`'s z together with sin(ω+f), which is > 0 when the planet is
# in front of the star (transit side) and <= 0 behind it (occultation side). z
# alone cannot tell the two apart: it depends on sin²(ω+f) only.
@inline function _sky_separation_signed(t::Real, P::Real, e::Real, ω::Real,
                                        Tp::Real, b::Real, a_Rs::Real)
    two_pi = oftype(t, 2π)
    M = two_pi * (t - Tp) / P
    E = kepler_solve(M, e)
    f = true_anomaly(E, e)

    # Orbital radius in units of semi-major axis
    r_over_a = (1 - e^2) / (1 + e * cos(f))

    # Inclination from impact parameter: b = a/R* cos(i) (1-e²)/(1+e sin(ω))
    # => cos(i) = b * (1 + e*sin(ω)) / (a/R* * (1-e²))
    e_factor = (1 + e * sin(ω)) / (1 - e^2)
    cos_i = b * e_factor / a_Rs
    sin_i_sq = 1 - cos_i^2

    # Sky-projected separation
    sin_wf, cos_wf = sincos(ω + f)
    z = a_Rs * r_over_a * sqrt(max(1 - sin_i_sq * sin_wf^2, zero(t)))
    return z, sin_wf
end

# Sky positions for the photometric likelihood, which evaluates one orbit at
# every cadence and sub-sample of a call (~10⁴). `planet_sky_position` and
# `sky_separation` rebuild sin ω, cos ω, √(1−e²), cos i and sin i at each time;
# `_sky_orbit` builds them once per call. The phase then comes from one
# sincos(E), by the half-angle identity
#
#     cos f = (cos E − e) / (1 − e cos E),   sin f = √(1−e²) sin E / (1 − e cos E),
#
# instead of sincos(E/2), an atan, and cos f and sincos(f + ω) after it. When e is
# exactly zero -- a Float64 0, or a Dual whose value and partials are all zero,
# i.e. a fixed e -- E = M and Kepler's equation is not solved; a free e that
# happens to be zero keeps the solver, so ∂/∂e still propagates.
#
# `ef` is the e of the true anomaly: `true_anomaly`'s clamp (`_anomaly_e`) by
# default, as `planet_sky_position` and `sky_separation` have it, or e itself for
# the workspace refresh of an ordinary planet, which never clamped. The two agree
# for every e in [0, 1), every e a likelihood evaluates an orbit at. (The clamp
# was to [0, 0.9999], and above 0.9999 the two were different models.)
struct _SkyOrbit{T}
    P::T
    Tp::T
    e::T            # Kepler's equation and the orbital radius
    ef::T           # the true anomaly's e
    sq::T           # √(1 − ef²)
    sω::T
    cω::T
    ome2::T         # 1 − e²
    aR::T
    cosi::T         # `planet_sky_position`'s inclination (clamped to [−1, 1])
    sini::T
    sin_i_sq::T     # `sky_separation`'s 1 − cos² i (cos i not clamped)
    circ::Bool      # e exactly zero and not a free parameter: E = M
end

@inline _is_const_zero(x::AbstractFloat) = iszero(x)
@inline _is_const_zero(x::ForwardDiff.Dual) =
    _is_const_zero(ForwardDiff.value(x)) && iszero(ForwardDiff.partials(x))
@inline _is_const_zero(x::Real) = false

@inline function _sky_orbit(P::Real, e::Real, ω::Real, Tp::Real, b::Real, a_Rs::Real,
                            ef::Real = _anomaly_e(e))
    T = promote_type(typeof(P), typeof(e), typeof(ω), typeof(Tp), typeof(b),
                     typeof(a_Rs), typeof(ef))
    sω, cω = sincos(ω)
    ome2 = 1 - e * e
    # as `_sky_from_phase` (rm.jl) builds it
    cosi = clamp(b * (1 + e * sω) / max(a_Rs * ome2, eps()), -1.0, 1.0)
    sini = sqrt(max(1 - cosi * cosi, zero(cosi)))
    # as `_sky_separation_signed` builds it
    cos_i = b * ((1 + e * sω) / ome2) / a_Rs
    return _SkyOrbit{T}(P, Tp, e, ef, sqrt(max(1 - ef * ef, zero(ef))), sω, cω,
                        ome2, a_Rs, cosi, sini, 1 - cos_i * cos_i, _is_const_zero(e))
end

# cos f, sin f, r/a and sin(ω + f) at time t.
@inline function _orbit_phase(o::_SkyOrbit, t::Real)
    M = 2π * (t - o.Tp) / o.P
    E = o.circ ? M : kepler_solve(M, o.e)
    sinE, cosE = sincos(E)
    denom = 1 - o.ef * cosE
    cosf = (cosE - o.ef) / denom
    sinf = o.sq * sinE / denom
    r_over_a = o.ome2 / (1 + o.e * cosf)
    return cosf, sinf, r_over_a, o.sω * cosf + o.cω * sinf
end

# `planet_sky_position(t, ...)`: (x_sky, y_sky, z_los) in stellar radii.
@inline function _sky_position(o::_SkyOrbit, t::Real)
    cosf, sinf, r_over_a, sin_wf = _orbit_phase(o, t)
    cos_wf = o.cω * cosf - o.sω * sinf
    r = o.aR * r_over_a
    return (r * (-cos_wf), r * (sin_wf * o.cosi), r * (sin_wf * o.sini))
end

# `_sky_separation_signed(t, ...)`: z and sin(ω + f), > 0 with the planet in front.
@inline function _sky_separation_signed(o::_SkyOrbit, t::Real)
    _, _, r_over_a, sin_wf = _orbit_phase(o, t)
    z = o.aR * r_over_a * sqrt(max(1 - o.sin_i_sq * sin_wf * sin_wf, zero(sin_wf)))
    return z, sin_wf
end

"""
    rho_s_to_a_Rs(rho_s, P) -> a/R*

Kepler's third law: a/R* = (G ρ_s P² / (3π))^(1/3).
ρ_s in solar units (ρ_sun), P in days.
"""
function rho_s_to_a_Rs(rho_s::Real, P::Real)
    # G * ρ_sun * day² / (3π) in CGS, then take cube root
    # Using: G = 6.674e-8 cm³/(g s²), ρ_sun = 1.411 g/cm³, day = 86400 s
    # G * ρ_sun = 9.413e-8 /s², * day² = 9.413e-8 * 86400² = 702.6
    # Factor: (G ρ_sun day² / (3π))^(1/3) = (702.6 / 3π)^(1/3) * (ρ/ρ_sun * P²)^(1/3)
    # = 4.209 * (ρ/ρ_sun * P²)^(1/3)
    # Actually, let me just use the exact constant.
    # a/R* = ( (G / (3π)) * ρ_s * P² )^(1/3)
    # with ρ_s in g/cm³ and P in seconds:
    # G/(3π) = 6.674e-8 / (3π) = 7.077e-9 cm³/(g s²)
    # P_sec = P * 86400
    # a/R* = (7.077e-9 * ρ_s * (P*86400)²)^(1/3)
    #
    # For ρ_s in solar units (ρ_sun = 1.411 g/cm³):
    # a/R* = (7.077e-9 * 1.411 * ρ_rel * 86400² * P²)^(1/3)
    #       = (7.077e-9 * 1.411 * 7.4649e9 * ρ_rel * P²)^(1/3)
    #       = (74.48 * ρ_rel * P²)^(1/3)
    rho_cgs = rho_s * 1.411  # convert solar units to g/cm³
    G_cgs = 6.674e-8         # cm³ g⁻¹ s⁻²
    P_sec = P * 86400.0
    return cbrt(G_cgs * rho_cgs * P_sec^2 / (3π))
end
