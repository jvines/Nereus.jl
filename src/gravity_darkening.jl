# Gravity-darkened transits (von Zeipel 1924; Barnes 2009).
#
# A rotating star is oblate, and its local effective gravity — hence its local
# effective temperature — falls from pole to equator. A transiting planet crosses
# regions of different surface brightness, so the light curve becomes ASYMMETRIC
# in a way that depends on the sky-projected obliquity λ AND on the stellar
# inclination i★ SEPARATELY. That separation is the point: an RM or Doppler
# tomogram gives only λ, so ψ (the true 3-D obliquity) is otherwise unreachable
# without a rotation period. Gravity darkening supplies i★ directly.
#
# The effect scales as (v_eq / v_crit)^2, so it is self-testing: a detection means
# the star is a fast rotator seen at low inclination; a non-detection puts a lower
# bound on i★. Both outcomes constrain ψ.
#
# COST. A full 2-D integration of the occulted flux per cadence is ~1e7 operations
# per likelihood call and is not affordable inside a sampler. For a small planet
# the effect factorises: Mandel–Agol already treats the limb-darkening integral
# exactly, and gravity darkening modulates the AMPLITUDE of the blocked flux by
# the local surface brightness relative to the disc average. So
#
#     F_gd(t) = 1 - [1 - F_MA(z, rp, u1, u2)] * I(x, y) / <I>
#
# which costs one extra brightness evaluation per cadence. The error is O(rp^2)
# in the brightness gradient across the planet — for rp = 0.116 that is well under
# the photometric precision.
#
# CONVENTIONS (shared with rm.jl so λ means the same thing in both):
#   x_sky  along the projected orbital motion, in stellar radii
#   y_sky  along the projected impact-parameter axis: +b at mid-transit, exactly
#          as `planet_sky_position` returns it
#   λ      sky-projected angle between the stellar spin axis and the orbit normal.
#          The projected spin axis is the line through (sin λ, cos λ) -- where
#          the RM kernels' sub-planet velocity x cos λ − y sin λ vanishes.
#   i★     angle between the spin vector and the line of sight, on (0°, 180°),
#          measured the way the orbital inclination is, so that
#              cos ψ = cos i★ cos i + sin i★ sin i cos λ.
#          90° = equator-on. Below 90° the visible pole sits at
#          −sin i★ (sin λ, cos λ) on the sky: for an aligned orbit that is the
#          far side of disc centre from the transit chord, as it must be when
#          the orbit lies in the equatorial plane. Above 90° it is the mirror
#          point. Darkening cannot tell a spin vector from its reverse, so
#          (i★, λ) and (180° − i★, λ + 180°) are the same star here; it takes a
#          measurement of the SENSE of rotation (RM, tomography) to split them.
#   β      gravity-darkening exponent; 0.25 for a radiative envelope (von Zeipel),
#          ~0.08 for convective (Lucy 1967). A9V here ⇒ radiative.

const _OMEGA_CRIT_INTERNAL = sqrt(8 / 27)   # break-up in the potential below
"""
    roche_flattening(ω_frac) -> f

Oblateness f = 1 − R_pole/R_eq for a Roche-model star rotating at `ω_frac` of its
critical (break-up) angular velocity. Exact for the Roche potential:
R_eq/R_pole = (3/ω_frac) cos[(π + acos(ω_frac))/3] for ω_frac ∈ (0,1].
"""
function roche_flattening(ω_frac::Real)
    ω = clamp(float(ω_frac), 0.0, 1.0)
    ω < 1e-6 && return 0.0
    ω >= 1.0 && return 1.0 - 2.0 / 3.0                # exact: R_eq/R_pol = 3/2
    ratio = (3.0 / ω) * cos((π + acos(ω)) / 3.0)      # R_eq / R_pole
    return 1.0 - 1.0 / ratio
end

"""
    roche_radius(θ, ω_frac) -> r

Stellar radius at colatitude `θ` (0 = pole) in units of the POLAR radius, for a
Roche model at `ω_frac` = Ω/Ω_crit. Closed form — the Roche surface
`1/r + (ω²/2) r² sin²θ = 1` is a depressed cubic with the standard solution

    r(θ) = (3 / (ω_frac sinθ)) cos[(π + acos(ω_frac sinθ)) / 3]

which is exact at break-up (r = 3/2 at the equator, for every star) and reduces
to 1 as ω_frac → 0. The previous damped fixed-point iteration converged to the
WRONG root approaching break-up — the surface equation has two roots there and
iteration walks to the larger, unphysical one — giving r(π/2) = 1.466 instead of
the exact 1.5.
"""
function roche_radius(θ::Real, ω_frac::Real)
    ω = clamp(float(ω_frac), 0.0, 1.0)
    w = ω * abs(sin(θ))
    w < 1e-9 && return one(w)                        # pole, or no rotation
    return (3 / w) * cos((π + acos(min(w, 1.0))) / 3)
end

"""
    local_gravity(θ, ω_frac) -> g

Effective gravity (centrifugal included) at colatitude `θ`, normalised to the
POLAR value. Used by von Zeipel to set the local temperature.
"""
function local_gravity(θ::Real, ω_frac::Real)
    ω = clamp(float(ω_frac), 0.0, 0.9999) * _OMEGA_CRIT_INTERNAL
    r = roche_radius(θ, ω_frac)          # takes the FRACTION, converts internally
    s, c = sin(θ), cos(θ)
    g_r = -1.0 / r^2 + ω^2 * r * s^2         # radial component
    g_θ = ω^2 * r * s * c                    # latitudinal component
    return sqrt(g_r^2 + g_θ^2)
end

# `local_gravity(θ, ω_frac)^2` from c2 = cos²θ and s2 = sin²θ, for the per-cadence
# brightness and the disc-average nodes, which know cos θ and not θ: no acos, and
# no sin and cos after it.
#
# The Roche radius solves 1/r + k r² = 1 with k = (Ω²/2) sin²θ. While
# w = ω_frac sinθ <= 0.7 it is taken by four Newton steps from r = 1 (the left
# side is convex, so they climb monotonically to the root): at round-off there
# (5e-16 at w = 0.7), and more accurate than `roche_radius`'s closed form at
# small w, where (3/w) cos[(π + acos w)/3] cancels (3e-10 relative at
# w = 1e-6). Above 0.7 Newton slows as the two roots of the cubic merge at
# break-up (an error of 1e-2 at w = 0.999), so the closed form is used.
@inline function _gd_gravity2(c2, s2, ω_frac)
    ωf = clamp(float(ω_frac), 0.0, 1.0)
    Ω2 = (clamp(ωf, 0.0, 0.9999) * _OMEGA_CRIT_INTERNAL)^2   # Ω², as in local_gravity
    w2 = ωf^2 * s2
    if w2 <= 0.49
        k = Ω2 * s2 / 2
        r = one(k)
        for _ in 1:4                     # r -= f/f', f = 1/r + k r² - 1, times r²
            r2 = r * r
            r -= (r + k * r2 * r2 - r2) / (2k * r2 * r - 1)
        end
    else
        w = sqrt(w2)
        r = (3 / w) * cos((π + acos(min(w, one(w)))) / 3)
    end
    g_r = Ω2 * r * s2 - 1 / r^2
    return g_r^2 + Ω2^2 * r^2 * s2 * c2
end

"""
    gd_brightness(x_sky, y_sky, i_star, λ, ω_frac, β) -> I

Surface brightness at sky position `(x_sky, y_sky)` (stellar radii, origin at disc
centre) for a gravity-darkened star, normalised so the DISC-AVERAGED brightness is
1. Returns 1.0 for a point off the disc, and 1.0 exactly when ω_frac = 0.

The sky position is rotated into the stellar frame by λ, then the colatitude
follows from i★, and von Zeipel gives T ∝ g^β with the bolometric brightness
∝ T^4 ∝ g^{4β}.
"""
function gd_brightness(x_sky::Real, y_sky::Real, i_star::Real, λ::Real,
                       ω_frac::Real, β::Real = 0.25)
    return gd_brightness(gd_context(i_star, λ, ω_frac, β), x_sky, y_sky)
end

"""
    gd_context(i_star, λ, ω_frac, β) -> ctx

Everything in the brightness map that does NOT depend on where the planet is:
the two rotation matrices, and the single scale factor that carries both the
polar-gravity normalisation and the disc average.

This exists for speed. The photometric likelihood evaluates the brightness once
per cadence -- 61 000 times per call for a representative A-star transit -- and
the disc average is an integral over the disc, so it belongs here, once per
call, and never in the per-cadence path. It is not cached across calls (see
`_gd_disc_mean`): a shared cache needs a lock, and parallel walkers then queue on
it for values none of them reuse.
"""
@inline function gd_context(i_star::Real, λ::Real, ω_frac::Real, β::Real = 0.25)
    ω = _gd_omega(ω_frac)
    ω < 1e-6 && return _gd_context(i_star, λ, ω, β, nothing)
    return _gd_context(i_star, λ, ω, β, _gd_disc_mean(i_star, ω, β))
end

# The rotation rate the brightness map uses: Ω/Ω_crit, kept below break-up.
@inline _gd_omega(ω_frac) = clamp(float(ω_frac), 0.0, 0.999)

# `gd_context` with the disc average `mean` already computed, for ω from
# `_gd_omega`. The likelihood builds one context per (planet, band) and the
# average depends on neither λ nor the planet, so `_gd_contexts` computes it once
# per distinct β and passes it here.
@inline function _gd_context(i_star::Real, λ::Real, ω::Real, β::Real, mean)
    sλ, cλ = sincos(λ)
    si, ci = sincos(i_star)
    if ω < 1e-6
        # No rotation: the map is uniform. `scale = 0` flags the flat case so the
        # per-cadence path can return 1.0 without touching the Roche solver.
        return (sλ = sλ, cλ = cλ, si = si, ci = ci, ω = 0.0,
                p = 4 * float(β), scale = 0.0)
    end
    # The gravity is normalised to the pole (g = 1 there exactly), so the scale is
    # the inverse disc average alone.
    return (sλ = sλ, cλ = cλ, si = si, ci = ci, ω = ω, p = 4 * float(β), scale = 1 / mean)
end

"""
    gd_brightness(ctx, x_sky, y_sky) -> I

Surface brightness at sky position `(x_sky, y_sky)` in stellar radii, origin at
disc centre, normalised so the disc average is 1. Returns 1.0 off the disc, and
1.0 exactly when the star is not rotating.
"""
@inline function gd_brightness(ctx, x_sky::Real, y_sky::Real)
    ctx.scale == 0 && return one(promote_type(typeof(x_sky), typeof(ctx.scale)))
    ρ2 = x_sky^2 + y_sky^2
    ρ2 >= 1 && return one(promote_type(typeof(x_sky), typeof(ctx.scale)))

    # Component of the sky position along the projected spin axis. The axis is
    # the line through (sin λ, cos λ) in `planet_sky_position`'s frame, and the
    # minus sign is which end of it leans toward the observer (see CONVENTIONS).
    # This read `-x sλ + y cλ`, correct for y_sky = −b at mid-transit; when
    # `planet_sky_position` moved to +b the map was left behind, mirrored in y
    # against the chord it is evaluated on -- λ entered as 180° − λ.
    yr = -(x_sky * ctx.sλ + y_sky * ctx.cλ)
    zr = sqrt(max(1 - ρ2, zero(ρ2)))              # toward the observer

    # spin axis = (0, sin i★, cos i★): +y' is its sky projection, +z is toward us.
    # i★ = 0 (pole-on) must put the DISC CENTRE at the pole (θ = 0).
    z_star = yr * ctx.si + zr * ctx.ci
    # cos²θ and sin²θ of the colatitude θ: the map depends on nothing else, and
    # is N/S symmetric. g^p = exp((p/2) log g²).
    c2 = min(z_star^2, one(z_star))
    g2 = _gd_gravity2(c2, 1 - c2, ctx.ω)
    return exp(ctx.p / 2 * log(g2)) * ctx.scale
end

# Disc-averaged brightness: the normalisation depends only on (i★, ω, β), not on
# where the planet is, so it is computed once per likelihood call, per band (see
# `_gd_contexts`), never per cadence.
#
# It is NOT cached across calls. A global cache needed a lock, and with the sampler
# evaluating many walkers in parallel every walker queued on that lock to compute a
# value no other walker would reuse: the key was (i★, ω, β) rounded to 1e-5, every
# proposal moves i★ and ω, so the cache never hit, grew without bound, and on a
# near-miss returned the value computed at whichever nearby point got there first,
# which made the likelihood depend on thread scheduling. Computing it directly
# also keeps d<I>/di★ in any AD gradient.
#
# The average is the projected-area-weighted mean over the visible disc,
#
#     <I> = ∫ f μ dx dy / ∫ μ dx dy,   f = (g(θ)/g_pole)^{4β},  μ = √(1 − x² − y²),
#
# with cos θ = |n·s| for the surface normal n and the spin axis s. It reduces to
# 1-D integrals in colatitude, exactly:
#
#   * on the sphere dx dy = μ dΩ, so the numerator is ∫_visible f μ² dΩ and the
#     denominator is 2π/3;
#   * f(n) = f(−n) (the map is N/S symmetric) and μ² is even, so the integral
#     over the visible hemisphere is half the one over the whole sphere: the
#     terminator drops out;
#   * over the whole sphere, with u = cos θ and φ the azimuth about s,
#     μ = u cos i★ + √(1−u²) sin i★ cos φ and ∫₀^{2π} μ² dφ = 2π [u² cos² i★ +
#     (1−u²) sin² i★ / 2].
#
# Hence
#
#     <I> = 3 [cos² i★ M₂ + sin² i★ (M₀ − M₂)/2],     M_k = ∫₀¹ f(u) u^k du,
#
# and i★ enters in closed form. f is analytic in u on [0, 1] (r and g depend on
# sin² θ = 1 − u²), so Gauss–Legendre converges geometrically: 32 nodes are at
# round-off for ω ≤ 0.99 and within 5e-12 at the 0.999 clamp. The 96 x 96
# midpoint grid this replaced was off by 1e-7 at ω = 0.1 and 8e-5 near
# break-up, and cost 7 200 `local_gravity` calls per band instead of 32.
#
# f depends on β only through the power, so log g at the nodes is computed once
# per ω (`_gd_node_logs`) and each band's average from it (`_gd_disc_mean_nodes`).
# The nodes are tuples of Float64 constants: AD sees only the arithmetic in
# (i★, ω, β).

# Gauss–Legendre nodes and weights on [0, 1] (Newton on the three-term recurrence).
function _gauss_legendre_01(n::Int)
    u = zeros(n); w = zeros(n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))
        for _ in 1:100
            p0, p1 = 1.0, z
            for k in 2:n
                p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
            end
            dz = p1 / (n * (z * p1 - p0) / (z^2 - 1))
            z -= dz
            abs(dz) <= 4eps() && break
        end
        p0, p1 = 1.0, z
        for k in 2:n
            p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
        end
        dp = n * (z * p1 - p0) / (z^2 - 1)
        u[i] = (1 + z) / 2                       # node on [0, 1]
        w[i] = 1 / ((1 - z^2) * dp^2)            # weight on [0, 1]
    end
    return u, w
end

const _GD_NQ = 32
# cos²θ and sin²θ at the nodes, and the weights of M₀ and M₂
const _GD_C2, _GD_S2, _GD_W0, _GD_W2 = let (u, w) = _gauss_legendre_01(_GD_NQ)
    (Tuple(u .^ 2), Tuple(1 .- u .^ 2), Tuple(w), Tuple(w .* u .^ 2))
end

# log g (pole-normalised) at the quadrature nodes: the β-independent part of <I>.
@inline _gd_node_logs(ω_frac) =
    ntuple(k -> log(_gd_gravity2(_GD_C2[k], _GD_S2[k], ω_frac)) / 2, Val(_GD_NQ))

# <I> for one β from the node logs.
@inline function _gd_disc_mean_nodes(lg::NTuple{N}, i_star, β) where {N}
    si, ci = sincos(i_star)
    p = 4β
    m0 = m2 = zero(promote_type(eltype(lg), typeof(p)))
    @inbounds for k in 1:N
        f = exp(p * lg[k])
        m0 += _GD_W0[k] * f
        m2 += _GD_W2[k] * f
    end
    return 3 * (ci^2 * m2 + si^2 * (m0 - m2) / 2)
end

_gd_disc_mean_raw(i_star, ω_frac, β) = _gd_disc_mean_nodes(_gd_node_logs(ω_frac), i_star, β)

_gd_disc_mean(i_star, ω_frac, β) = _gd_disc_mean_raw(i_star, ω_frac, β)

"""
    transit_flux_gd(x_sky, y_sky, z_los, rp, u1, u2, i_star, λ, ω_frac, β) -> F

Relative flux during a gravity-darkened transit. Uses Mandel–Agol for the exact
limb-darkened shape and scales the blocked flux by the local surface brightness:

    F = 1 - [1 - F_MA(z, rp, u1, u2)] * I(x, y)

with `I` normalised to unit disc average, so ω_frac = 0 reduces EXACTLY to the
standard model. `z_los > 0` selects inferior conjunction (planet in front); the
occultation is returned as no event.
"""
function transit_flux_gd(x_sky::Real, y_sky::Real, z_los::Real, rp::Real,
                         u1::Real, u2::Real, i_star::Real, λ::Real,
                         ω_frac::Real, β::Real = 0.25)
    z_los > 0 || return 1.0                        # secondary eclipse: no transit
    z = sqrt(x_sky^2 + y_sky^2)
    f_ma = transit_flux(z, rp, u1, u2)
    f_ma >= 1.0 && return 1.0
    I = gd_brightness(x_sky, y_sky, i_star, λ, ω_frac, β)
    return 1.0 - (1.0 - f_ma) * I
end

"""
    omega_frac_from_veq(v_eq, M_s, R_s) -> ω_frac = Ω/Ω_crit

Convert an equatorial velocity [km/s] to a fraction of break-up ANGULAR velocity,
for stellar mass [M_sun] and POLAR radius [R_sun].

The distinction matters. `roche_radius`, `roche_flattening` and `local_gravity`
are all parametrised by Ω/Ω_crit, but v_eq/v_crit is a DIFFERENT ratio, because
the star's equatorial radius is not the break-up equatorial radius:

    Ω/Ω_crit = (v_eq/v_crit) · (3/2) / (R_eq/R_pol)

and R_eq/R_pol is itself a function of Ω/Ω_crit, so this is solved by iteration
(it converges in a few steps; the map is a contraction away from break-up).
Returning v_eq/v_crit directly — as this did — understated the oblateness by up
to a factor 3/2, which is a factor ~2 in the light-curve asymmetry.

Both limits are exact: at break-up v_eq = v_crit and R_eq/R_pol = 3/2, so the
answer is 1; for slow rotation R_eq → R_pol and Ω/Ω_crit → (3/2)(v_eq/v_crit).
"""
function omega_frac_from_veq(v_eq::Real, M_s::Real, R_s::Real)
    # Roche critical rotation: Ω_crit² = 8GM/(27 R_pol³), and at break-up the
    # equatorial radius is 3/2 R_pol, so
    #     v_crit = Ω_crit · (3/2) R_pol = sqrt(2GM / (3 R_pol)).
    # Using sqrt(GM/R_eq) instead (the Keplerian orbital speed at the equator)
    # understates ω by ~1.8x.
    Rpol = R_s * R_SUN_M
    v_crit = sqrt(2 * GM_SUN_SI * M_s / (3 * Rpol)) / 1e3     # km/s
    vr = clamp(v_eq / v_crit, 0.0, 1.0)
    vr <= 0 && return zero(vr)
    vr >= 1 && return one(vr)

    # v_eq/v_crit = ω · R_eq(ω) / (3/2 R_pol) is strictly increasing on (0,1),
    # so invert it by bisection. A fixed-point iteration on the same relation
    # is NOT safe: the map is decreasing in ω and its slope exceeds 1 near
    # break-up, where it converged to 0.917 for a true 0.99.
    h(w) = w * roche_radius(π / 2, w) / 1.5 - vr
    lo, hi = 0.0, 1.0
    for _ in 1:64
        mid = 0.5 * (lo + hi)
        h(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    ω = 0.5 * (lo + hi)

    # Bisection compares only the value part, so under ForwardDiff the result
    # carries no derivative. Two Newton steps in the caller's number type fix
    # that: at a Newton fixed point h(ω) = 0 holds in the Dual algebra, so
    # dω/dv_eq comes out right even though h′ here is a plain finite difference.
    if !(vr isa AbstractFloat)
        δ = 1e-6
        dh = (h(min(ω + δ, 1.0)) - h(max(ω - δ, 0.0))) / (min(ω + δ, 1.0) - max(ω - δ, 0.0))
        for _ in 1:2
            ω = ω - h(ω) / dh
        end
    end
    return ω
end

"""
    transit_flux_gd(ld::QuadLimbDark, x_sky, y_sky, z_los, rp, i_star, λ, ω_frac, β) -> F

`QuadLimbDark` overload, for the photometric likelihood: it already caches a
per-instrument `ld` object, so re-deriving u1/u2 per cadence would be wasted work.
Identical physics to the (u1, u2) method.
"""
function transit_flux_gd(ld::QuadLimbDark, x_sky::Real, y_sky::Real, z_los::Real,
                         rp::Real, i_star::Real, λ::Real, ω_frac::Real,
                         β::Real = 0.25)
    z_los > 0 || return one(promote_type(typeof(x_sky), typeof(rp)))
    z = sqrt(x_sky^2 + y_sky^2)
    f_ma = transit_flux(ld, z, rp)
    f_ma >= 1 && return f_ma
    I = gd_brightness(x_sky, y_sky, i_star, λ, ω_frac, β)
    return 1 - (1 - f_ma) * I
end

"""
    gd_beta_band(λ_eff_nm, T_eff; β = 0.25) -> β_eff

Effective gravity-darkening exponent for a filter of effective wavelength
`λ_eff_nm` [nm] observing a star of effective temperature `T_eff` [K].

Gravity darkening is CHROMATIC and the difference is not small. von Zeipel gives
the local temperature, T ∝ g^β, and the BOLOMETRIC flux then goes as T^4 — but a
filter measures the Planck function over its own passband, so the brightness
responds as T^n with

    n = dlnB_λ/dlnT = x / (1 - e^{-x}),    x = hc / (λ k T)

`n = 4` is the BOLOMETRIC (Stefan–Boltzmann) value and holds monochromatically
only where x ≈ 3.92, i.e. λ T ≈ 3.67e-3 m·K. For a 7437 K star that is 493 nm:
bluer than this and the band is MORE sensitive than bolometric, redder and it is
less. n ≈ 4.14 in g′ (475 nm), ≈ 2.75 in i′ (763 nm), ≈ 2.69 in the TESS band
(786 nm). Since the model raises `g` to the power `4β_eff`, matching the physics
means

    β_eff = β · n / 4

Using the bolometric β for the TESS band overstates the light-curve asymmetry by
~50% (0.25 vs 0.168), which for a NON-detection turns into a stronger claimed
lower limit on i_star than the data support.

Feed the result to the `gd_beta_<INST>` prior:

    priors["gd_beta_TESS"] = FixedPrior(gd_beta_band(786.0, 7437.0))
"""
function gd_beta_band(λ_eff_nm::Real, T_eff::Real; β::Real = 0.25)
    h = 6.62607015e-34; c = 2.99792458e8; kB = 1.380649e-23
    x = h * c / (λ_eff_nm * 1e-9 * kB * T_eff)
    n = x / (1 - exp(-x))                       # dlnB_λ / dlnT
    return β * n / 4
end

"Effective wavelengths [nm] of the filters Nereus fits most often."
const GD_LAMBDA_EFF = Dict(
    "TESS" => 786.5, "Kepler" => 639.0, "CHEOPS" => 643.0,
    "NGTS" => 697.0, "g" => 475.0, "r" => 622.0, "i" => 763.0, "z" => 905.0,
    "B" => 445.0, "V" => 551.0, "R" => 658.0, "I" => 806.0,
)

"""
    transit_flux_gd(ld::QuadLimbDark, ctx, x_sky, y_sky, z_los, rp) -> F

Flux with a precomputed brightness context (`gd_context`). This is the form the
photometric likelihood uses; the argument-list forms above build a context per
call and are for tests and one-off evaluation.
"""
@inline function transit_flux_gd(ld::QuadLimbDark, ctx, x_sky::Real, y_sky::Real,
                                 z_los::Real, rp::Real)
    z_los > 0 || return one(promote_type(typeof(x_sky), typeof(rp)))
    f_ma = transit_flux(ld, sqrt(x_sky^2 + y_sky^2), rp)
    f_ma >= 1 && return f_ma
    return 1 - (1 - f_ma) * gd_brightness(ctx, x_sky, y_sky)
end
