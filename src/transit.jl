using Transits: QuadLimbDark, compute

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
    z2 = z * z
    p2 = p * p
    k1 = acos((z2 + 1 - p2) / (2 * z))
    k2 = acos((z2 + p2 - 1) / (2 * z * p))
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

Quadratic limb-darkened transit flux via Transits.jl (Agol 2020
formulation). For `u1 = u2 = 0` falls back to the exact uniform-disk
formula.
"""
function transit_flux(z::Real, p::Real, u1::Real, u2::Real)
    if iszero(u1) && iszero(u2)
        return transit_flux_uniform(z, p)
    end
    ld = QuadLimbDark([u1, u2])
    return compute(ld, z, p)
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
    return compute(ld, z, p)
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
    return z
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
