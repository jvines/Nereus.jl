# Keplerian orbit primitives: true anomaly, RV model, parametrization
# conversions. All functions are generic over element type so autodiff
# frameworks can trace through them.
#
# Conventions follow Murray & Dermott (1999) and Perryman "Exoplanet
# Handbook" (2018):
# - ω (omega): argument of periastron, radians. Measured from ascending
#   node to periastron in the direction of motion.
# - f: true anomaly, radians. Measured from periastron.
# - M: mean anomaly at time t, `M(t) = M0 + 2π (t - t_ref) / P` where
#   `M0` is the mean anomaly at reference epoch `t_ref`.
# - K: RV semi-amplitude, m/s.
# - P: orbital period, days (or whatever unit t is in — must match).

# ---------------------------------------------------------------------
# True anomaly from eccentric anomaly
# ---------------------------------------------------------------------

"""
    true_anomaly(E, e) -> f

Compute the true anomaly `f` from eccentric anomaly `E` and eccentricity
`e`. Uses the atan2-form to stay well-conditioned and single-valued over
`E ∈ [-π, π]`.

    f = 2 atan2( √(1+e) sin(E/2),  √(1-e) cos(E/2) )

For `e = 0`, `f = E` (modulo branch). For `e → 1`, behaves correctly as
long as `E ≠ π`.
"""
@inline function true_anomaly(E::Real, e::Real)
    # Guard against unphysical e (sampler can transiently propose e > 1
    # or e < 0 via sesinw/secosw, even in unconstrained space where
    # the bijector maps each component independently but doesn't enforce
    # the joint constraint s²+c² < 1).
    # Use max/min instead of clamp for better autodiff trace (no branch).
    e_safe = min(max(e, zero(e)), oftype(e, 0.9999))
    arg_plus  = 1 + e_safe
    arg_minus = max(1 - e_safe, oftype(e, 1e-10))  # prevent sqrt(negative)
    sqrt_one_plus  = sqrt(arg_plus)
    sqrt_one_minus = sqrt(arg_minus)
    sinE2, cosE2 = sincos(E / 2)
    return 2 * atan(sqrt_one_plus * sinE2, sqrt_one_minus * cosE2)
end

# ---------------------------------------------------------------------
# RV Keplerian model (Mo parametrization)
# ---------------------------------------------------------------------

"""
    rv_keplerian(t, P, K, e, ω, M0, t_ref) -> rv

Single-planet radial velocity at time `t`:

    rv(t) = K · [ cos(f + ω) + e cos(ω) ]

where the true anomaly `f` is computed from the mean anomaly

    M(t) = M0 + 2π (t - t_ref) / P

via Kepler's equation. The reference epoch `t_ref` sets the zero of
`M0`; conventionally this is `median(t_rv)` for numerical conditioning.

For `e = 0` reduces to `K cos(M + ω)`.

Broadcast-friendly: pass a vector for `t` to get a vector of rv values
(`rv_keplerian.(t, P, K, e, ω, M0, t_ref)`).
"""
function rv_keplerian(t::Real, P::Real, K::Real, e::Real, ω::Real,
                       M0::Real, t_ref::Real)
    M = M0 + TWO_PI * (t - t_ref) / P
    E = kepler_solve(M, e)
    f = true_anomaly(E, e)
    return K * (cos(f + ω) + e * cos(ω))
end

"""
    rv_keplerian_sum(t, params, t_ref) -> rv

Sum of RV contributions from multiple Keplerian orbits. `params` is an
iterable of `(P, K, e, ω, M0)` tuples (or any collection supporting
unpacking of 5 elements).

Returns the total radial velocity at time `t`.
"""
function rv_keplerian_sum(t::Real, params, t_ref::Real)
    rv = zero(t)
    for (P, K, e, ω, M0) in params
        rv += rv_keplerian(t, P, K, e, ω, M0, t_ref)
    end
    return rv
end

# ---------------------------------------------------------------------
# Parametrization conversions: e, ω ↔ (sesinw, secosw) ↔ (esinw, ecosw)
# ---------------------------------------------------------------------
#
# Why bother? Sampling directly in (e, ω) is problematic because at low
# e, ω is unconstrained. sesinw/secosw (Eastman+ 2013) and esinw/ecosw
# parameterizations flatten the geometry and handle this cleanly.

"""
    sesinw_to_ew(sesinw, secosw) -> (e, ω)

Convert `(√e sin ω, √e cos ω)` → `(e, ω)`. The inverse map is
`ew_to_sesinw`. `ω` is returned in `(-π, π]`.
"""
function sesinw_to_ew(sesinw::Real, secosw::Real)
    e = sesinw^2 + secosw^2
    ω = atan(sesinw, secosw)
    return e, ω
end

"""
    esinw_to_ew(esinw, ecosw) -> (e, ω)

Convert `(e sin ω, e cos ω)` → `(e, ω)`. The inverse map is
`ew_to_esinw`. `ω` is returned in `(-π, π]`.
"""
function esinw_to_ew(esinw::Real, ecosw::Real)
    e = sqrt(esinw^2 + ecosw^2)
    ω = atan(esinw, ecosw)
    return e, ω
end

"""
    ew_to_sesinw(e, ω) -> (sesinw, secosw)

Inverse of `sesinw_to_ew`.
"""
function ew_to_sesinw(e::Real, ω::Real)
    sqrt_e = sqrt(e)
    sinω, cosω = sincos(ω)
    return sqrt_e * sinω, sqrt_e * cosω
end

"""
    ew_to_esinw(e, ω) -> (esinw, ecosw)

Inverse of `esinw_to_ew`.
"""
function ew_to_esinw(e::Real, ω::Real)
    sinω, cosω = sincos(ω)
    return e * sinω, e * cosω
end

# ---------------------------------------------------------------------
# Time anchor conversions: Mo, Tp, Tc
# ---------------------------------------------------------------------
#
# Mo = mean anomaly at reference epoch t_ref
# Tp = time of periastron passage
# Tc = time of inferior conjunction (transit center)
#
# Conventions follow EMPEROR (Hou et al. 2012):
#   M(t) = 2π (t - Tp) / P
#   At t = t_ref: Mo = 2π (t_ref - Tp) / P
#   At transit:   f = π/2 - ω

"""
    mo_to_tp(Mo, P, t_ref) -> Tp

Convert mean anomaly at epoch to time of periastron passage.
"""
function mo_to_tp(Mo::Real, P::Real, t_ref::Real)
    return t_ref - Mo * P / oftype(Mo, 2π)
end

"""
    tp_to_mo(Tp, P, t_ref) -> Mo

Convert time of periastron passage to mean anomaly at epoch.
"""
function tp_to_mo(Tp::Real, P::Real, t_ref::Real)
    return oftype(Tp, 2π) * (t_ref - Tp) / P
end

"""
    tc_to_tp(Tc, P, e, ω) -> Tp

Convert time of transit center to time of periastron passage.
Transit occurs at true anomaly `f = π/2 - ω` (inferior conjunction).
"""
function tc_to_tp(Tc::Real, P::Real, e::Real, ω::Real)
    f_tr = oftype(ω, π) / 2 - ω
    # Guard e ≥ 1: active planets always have e < 1 (the likelihood rejects
    # e ≥ 1), but INACTIVE trans-dim slots carry stale unbounded junk (e > 1),
    # and post-fit consumers (PPC, predictions) iterate all slots. The result
    # for such a slot is unused (its K is masked out), but sqrt((1-e)/(1+e))
    # must not crash. No-op for the physical e < 1 range.
    ec = clamp(e, zero(e), oftype(e, 1) - eps(oftype(e, 1)))
    E_tr = 2 * atan(tan(f_tr / 2) * sqrt((1 - ec) / (1 + ec)))
    return Tc - P / oftype(Tc, 2π) * (E_tr - ec * sin(E_tr))
end

"""
    tp_to_tc(Tp, P, e, ω) -> Tc

Convert time of periastron passage to time of transit center.
"""
function tp_to_tc(Tp::Real, P::Real, e::Real, ω::Real)
    f_tr = oftype(ω, π) / 2 - ω
    # Guard e ≥ 1 (see tc_to_tp): inactive trans-dim slots carry stale junk
    # (e > 1) that post-fit consumers iterate; sqrt((1-e)/(1+e)) must not crash.
    ec = clamp(e, zero(e), oftype(e, 1) - eps(oftype(e, 1)))
    E_tr = 2 * atan(tan(f_tr / 2) * sqrt((1 - ec) / (1 + ec)))
    return Tp + P / oftype(Tp, 2π) * (E_tr - ec * sin(E_tr))
end
