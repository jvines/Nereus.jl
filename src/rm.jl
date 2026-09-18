# Rossiter-McLaughlin effect — flux-weighted (subtracted-light) model.
#
# ATTRIBUTION: this is Ohta, Taruya & Suto (2005), ApJ 622 1118, in its
# leading-order form. It was previously labelled "Hirano+ 2011" throughout this
# file, which is WRONG and cost real time on a real analysis: Hirano+ 2011
# (ApJ 742 69) is a DIFFERENT model that accounts for how a CCF responds to
# line-profile distortion, and is not implemented here. See rm_signal_arome for
# the CCF-matched alternative (Boue+ 2013).
#
# Computes the RV anomaly induced by a transiting planet occulting parts
# of the rotating stellar disk. The subtracted-light treatment of
# Ohta+ 2005 gives, to leading order,
#
#   ΔRV_RM(t) = -Δflux(t) · v_p(t)
#
# captures the essential physics, where:
#   - Δflux(t) is the transit flux deficit (≥ 0 in transit, 0 outside)
#   - v_p(t) is the line-of-sight stellar velocity at the planet's
#     sky-plane position, in units consistent with Δflux being a
#     fraction (so Δflux ≤ 1).
#
# The line-of-sight velocity at the occulted region depends on the
# stellar projected rotation `V·sin(i_*)` and the sky-projected
# stellar obliquity `λ`:
#
#   v_p(t) = V·sin(i_*) · x_p(t)
#
# where x_p is the planet's position perpendicular to the projected
# stellar spin axis, in units of stellar radius:
#
#   x_p(t) =  x_sky(t) · cos(λ) − y_sky(t) · sin(λ)
#
# (x_sky, y_sky) are the planet's sky-plane offsets from the stellar
# centre, with x_sky along the orbital motion direction and y_sky the
# impact-parameter axis. For near-circular orbits during transit:
#
#   x_sky(t) ≈ (a/R_*) · (2π/P) · (t − T_c)
#   y_sky(t) ≈ b                                  [impact parameter]
#
# The full Ohta+ 2005 correction including limb darkening multiplies
# v_p by a factor f(b, p, u₁, u₂) close to unity for typical
# transiting systems; we omit it in the leading-order form because
# the Δflux already encodes the LD profile via the underlying transit
# model (Transits.jl QuadLimbDark).
#
# Sign convention: when the planet occults the blue-shifted limb
# (approaching half of the rotating disk), ΔRV_RM is *positive* —
# the unobstructed flux-weighted mean shifts redward. With our
# (x_sky cos λ − y_sky sin λ) ordering and λ measured from the orbit
# normal to the stellar spin axis, the formula
# below produces the correct sign.

using LinearAlgebra: norm

"""
    rm_signal(x_sky, y_sky, Δflux, v_sini, λ) -> ΔRV_RM [m/s]

Leading-order flux-weighted (subtracted-light) RM anomaly, Ohta+ 2005.

# Arguments
- `x_sky`, `y_sky` — planet sky-plane offsets from the stellar centre,
  in units of stellar radii. `x_sky` along the orbital motion, `y_sky`
  the impact-parameter direction.
- `Δflux` — fractional flux deficit at this time (`0 ≤ Δflux ≤ 1`).
  Zero outside transit; nonzero only when the planet occults the disk.
- `v_sini` — stellar projected rotation `V·sin(i_*)`, m/s.
- `λ` — sky-projected obliquity, radians.

Returns ΔRV in m/s, to be added to the orbit-only RV prediction.
"""
@inline function rm_signal(x_sky::Real, y_sky::Real, Δflux::Real,
                            v_sini::Real, λ::Real)
    Δflux > 0 || return zero(promote_type(typeof(x_sky), typeof(v_sini)))
    s, c = sincos(λ)
    x_p = x_sky * c - y_sky * s
    return -Δflux * v_sini * x_p
end

"""
    rm_signal_arome(x_sky, y_sky, Δflux, v_sini, λ, σ0, βp) -> ΔRV_RM [m/s]

Rossiter-McLaughlin anomaly as measured by **fitting a Gaussian to a CCF** —
Boué et al. 2013 (A&A 550, A53), Eq. 15 — the formulation implemented by ARoME:

    v_CCF = −(2σ₀²/(σ₀²+β_p²))^{3/2} · f · v_p · exp(−v_p²/[2(σ₀²+β_p²)])

with `f = Δflux` the occulted flux fraction and `v_p = v_sini · x_*` the mean
line-of-sight velocity of the occulted region.

This matters because a Gaussian fit to a CCF does **not** return the
flux-weighted mean subtracted velocity that [`rm_signal`](@ref) computes. When
the planet occults a patch far out in the line wings (|v_p| ≫ σ₀) the notch
barely moves the fitted centroid, and the leading-order form over-predicts;
near line centre it under-predicts. `rm_signal` is the σ₀ ≫ v_p limit of this.

- `σ0` — dispersion of the Gaussian fitted to the OUT-OF-TRANSIT CCF [m/s].
  Measure it from the data; do NOT derive it from `v_sini`. Boué+2013 quote a
  validity limit of v·sin i ≲ 20 km/s, but that limit comes from *approximating*
  σ₀, not from Eq. 15: validated against direct CCF simulation, feeding a
  measured σ₀ holds the expression to ~10% at v·sin i = 40 km/s, versus a factor
  0.41–1.70 shape error for the flux-weighted form at the same rotation.
- `βp` — dispersion of the sub-planet (local) line profile [m/s]: the intrinsic
  stellar line width convolved with the instrumental profile.

Use [`rm_signal`](@ref) only for slow rotators (v·sin i ≲ β_p), where the three
formulations agree to ~10%.
"""
@inline function rm_signal_arome(x_sky::Real, y_sky::Real, Δflux::Real,
                                   v_sini::Real, λ::Real, σ0::Real, βp::Real)
    TT = promote_type(typeof(x_sky), typeof(v_sini), typeof(σ0))
    Δflux > 0 || return zero(TT)
    s, c = sincos(λ)
    v_p = v_sini * (x_sky * c - y_sky * s)
    w2 = σ0 * σ0 + βp * βp
    w2 > 0 || return -TT(Δflux) * TT(v_p)      # degenerate: fall back to Ohta/Winn
    pref = (2 * σ0 * σ0 / w2)^TT(1.5)
    return -TT(Δflux) * TT(v_p) * pref * exp(-v_p * v_p / (2 * w2))
end

"""
    rm_signal_hirano(x_sky, y_sky, Δflux, v_sini, λ, βs, βp) -> ΔRV_RM [m/s]

Rossiter-McLaughlin anomaly as measured by **cross-correlation / forward-model
(iodine-technique) RV pipelines** — Hirano et al. 2011 (ApJ 742, 69), Eq. 27,
the Gaussian analytic form (derived in Hirano et al. 2010, ApJ 709, 458, §2.3):

    Δv = −f · v_p · {2β_*²/(β_*²+β_p²)}^{3/2} · [1 − κ + κ²/2],
    κ  = v_p²/(β_*²+β_p²)

with `f = Δflux` the occulted flux fraction and `v_p = v_sini · x_*` the mean
line-of-sight velocity of the occulted region. Derived by maximizing the
cross-correlation of the in-transit spectrum against the out-of-transit
template — the estimator used by iodine-cell reductions (HIRES, HDS, PFS) —
with every broadening kernel approximated as a Gaussian, to first order in `f`.
The bracket is exp(−κ) truncated at O(κ²): Hirano+2010 expand in κ and keep
through the quintic velocity term, which is the form Eq. 27 prints and the one
the community fits as "the Hirano model".

- `βs` — Gaussian width of the OUT-OF-TRANSIT stellar template line [m/s],
  INCLUDING rotation, macroturbulence and the instrumental profile. Measure it
  from the template; lacking that, Hirano+2010 (appendix) give
  `βs² ≈ βp² + (v·sin i / 1.31)²` (Gaussian LSQ fit to the rotation kernel,
  quadratic limb darkening).
- `βp` — Gaussian width of the sub-planet (local) line profile [m/s]: thermal +
  microturbulent + instrumental; Hirano+2011 (§4) fold macroturbulence in via
  `βp = sqrt(β² + ζ²)`.

WIDTH CONVENTION — NOT the same as [`rm_signal_arome`](@ref): Hirano's
Gaussians are `exp(−v²/β²)`, so `βs`, `βp` here are 1/e HALF-WIDTHS, i.e.
`√2 ×` the dispersions `σ0`, `βp` that `rm_signal_arome` takes
(`σ0 = βs/√2`). Feeding dispersions to this kernel under-widths the line and
inflates the wing signal.

Regimes: preferred over [`rm_signal`](@ref) for iodine-technique RVs of
moderate rotators, where the CCF-shape response matters and the flux-weighted
form mis-scales the amplitude (the v_p³ term here is what the Winn+2005
empirical calibrations were fitting). It reduces to `rm_signal` (−f·v_p) when
`v_sini ≪ βp` (βs → βp, prefactor → 1, κ → 0) — there all three kernels agree
to ~10%. Because the bracket truncates exp(−κ), keep κ ≲ 0.6 (few-% truncation
error): beyond that the bracket turns back up (min 0.5 at κ = 1) instead of
decaying, over-predicting the signal when a rapid rotator's limb is occulted.
In that regime use `rm_signal_arome` — same first-order-in-f Gaussian theory
with the exponential kept resummed (identical modulo the √2 convention) — or
Hirano+2011's full Fourier formula (their Eq. 16, not implemented here).
"""
@inline function rm_signal_hirano(x_sky::Real, y_sky::Real, Δflux::Real,
                                    v_sini::Real, λ::Real, βs::Real, βp::Real)
    TT = promote_type(typeof(x_sky), typeof(v_sini), typeof(βs))
    Δflux > 0 || return zero(TT)
    s, c = sincos(λ)
    v_p = v_sini * (x_sky * c - y_sky * s)
    w2 = βs * βs + βp * βp
    w2 > 0 || return -TT(Δflux) * TT(v_p)      # degenerate: fall back to Ohta/Winn
    pref = (2 * βs * βs / w2)^TT(1.5)
    κ = v_p * v_p / w2
    return -TT(Δflux) * TT(v_p) * pref * (1 - κ + κ * κ / 2)
end

"""
    planet_sky_position(t, P, e, ω, Tp, b, a_Rs) -> (x_sky, y_sky, z_los)

Planet's sky-plane offset from the stellar centre at time `t`, in
units of stellar radius. Returns `(x_sky, y_sky, z_los)` where `x_sky` is
along the orbital motion direction, `y_sky` is the impact-parameter
axis (constant at `b` for circular orbits at mid-transit), and `z_los`
is the line-of-sight coordinate — positive when the planet is in FRONT
of the star (transit), negative when behind it (occultation).

`z_los` is required to distinguish the two conjunctions: `x_sky` and
`y_sky` alone are identical at inferior and superior conjunction, so a
sky-separation test cannot tell a transit from an occultation.

For circular orbits this reduces to
    `x_sky(t) = (a/R_*) sin(M), y_sky(t) = b · cos(M)` where
M is the mean anomaly relative to mid-transit. For eccentric orbits
we use the true anomaly and the standard sky-projection formulae
(Winn 2010 Eq. 53–55).
"""
function planet_sky_position(t::Real, P::Real, e::Real, ω::Real,
                              Tp::Real, b::Real, a_Rs::Real)
    M = 2π * (t - Tp) / P
    E = kepler_solve(M, e)
    f = true_anomaly(E, e)
    # Distance from star (in stellar radii), accounting for eccentricity.
    r = a_Rs * (1 - e * e) / (1 + e * cos(f))
    # Sky-plane components (Winn 2010 Eq. 53–55).
    # Use the impact parameter to recover sin(i): b = (a_Rs)(1-e²)/(1+e sin ω) · cos(i)
    # We approximate cos(i) ≈ b / a_Rs · (1+e sin ω)/(1-e²) ; in practice we just
    # decompose along the projected orbit axis with the same b-driven scaling.
    one_minus_e2 = 1 - e * e
    cosi = b * (1 + e * sin(ω)) / max(a_Rs * one_minus_e2, eps())
    cosi = clamp(cosi, -1.0, 1.0)
    sini = sqrt(max(1 - cosi * cosi, zero(cosi)))
    fw = f + ω
    x_sky = r * (-cos(fw))           # along orbital motion
    # SIGN: +b at mid-transit, matching this function's own docstring and
    # `shadow_track` in tomography.jl. It returned -b until 2026-08-10, which
    # silently mirrored the lambda axis of every RV-RM fit relative to every
    # tomographic one: v_p^RM(lambda) == v_p^tomo(-lambda). Nothing pinned the
    # sign, because the only test on it took abs(). For the RM kernels a flip is
    # a pure relabelling lambda -> -lambda, but `gd_brightness` is NOT symmetric
    # under (y, lambda) -> (-y, -lambda), so gravity-darkened fits were affected
    # in substance: ~55 ppm peak at a hot-Jupiter-on-A-star geometry, growing
    # with rotation.
    y_sky = r * (sin(fw) * cosi)     # impact-parameter axis
    z_los = r * (sin(fw) * sini)     # line of sight: > 0 ⇒ planet in front
    return (x_sky, y_sky, z_los)
end

"""
    rm_signal_at_time(t, P, e, ω, Tp, b, a_Rs, Δflux, v_sini, λ) -> ΔRV [m/s]

Convenience wrapper: compute the RM contribution at time `t` given the
full orbit + transit-window flux deficit. Returns 0 when `Δflux ≤ 0`.
"""
@inline function rm_signal_at_time(t::Real, P::Real, e::Real, ω::Real,
                                     Tp::Real, b::Real, a_Rs::Real,
                                     Δflux::Real, v_sini::Real, λ::Real)
    Δflux > 0 || return zero(promote_type(typeof(t), typeof(v_sini)))
    x_sky, y_sky, z_los = planet_sky_position(t, P, e, ω, Tp, b, a_Rs)
    # Only a transit produces an RM anomaly; behind the star there is none.
    z_los > 0 || return zero(promote_type(typeof(t), typeof(v_sini)))
    return rm_signal(x_sky, y_sky, Δflux, v_sini, λ)
end

"""
    _decode_rm_state(theta, p_idx, Ps) -> (n_rm, rm_state) | (0, nothing)

Internal helper for the RV likelihoods. Pre-decodes everything needed
to evaluate the Rossiter-McLaughlin contribution at each RV epoch.

Returns `(0, nothing)` when no active planet has `:RM` in its
`PlanetDataSources`. Otherwise returns `(n_rm, state)` where `state`
is a named tuple of pre-computed per-RM-planet caches:

- `j_active::Vector{Int}` — index `j ∈ 1..n_rv_planets` of the RM
  planet inside the existing `Ps/es/ws/Tps` arrays
- `λ::Vector{T}` — sky-projected obliquity
- `b::Vector{T}`, `rr::Vector{T}` — transit geometry
- `a_Rs::Vector{T}` — semi-major axis in stellar radii
- `v_sini::T` — system-level V·sin(i_*)
- `u1::T`, `u2::T` — limb-darkening coefficients (Kipping-decoded
  from first PM instrument; q1=q2=0 → uniform-disk fast path)

Returns `n_rm = -1` (sentinel) if a required quantity is missing
(e.g. M_s and rho_s both unset). The caller treats that as a fit
failure and returns -Inf log-likelihood.
"""
function _decode_rm_state(theta::Theta{T}, p_idx,
                           Ps::AbstractVector{T}) where {T}
    config  = theta.params.config
    layout  = theta.params.layout
    modes   = config.planet_modes

    # Count RM-enabled active planets (flux-weighted OR Reloaded OR ARoME)
    n_rm = 0
    j = 0
    for k in p_idx
        block = layout.planet_blocks[k]
        has_K(block) || continue
        j += 1
        has_any_rm(modes[k]) && (n_rm += 1)
    end
    n_rm == 0 && return (0, nothing)

    j_active    = Vector{Int}(undef, n_rm)
    λs          = Vector{T}(undef, n_rm)
    bs          = Vector{T}(undef, n_rm)
    rrs         = Vector{T}(undef, n_rm)
    a_Rs        = Vector{T}(undef, n_rm)
    is_reloaded = Vector{Bool}(undef, n_rm)
    is_arome    = Vector{Bool}(undef, n_rm)

    use_rho = config.parametrization.use_rho_s
    rho_val = use_rho ? theta.values[layout.systemic.rho_s] : zero(T)
    M_s = config.M_s
    R_s = config.R_s
    have_ms_rs = !isnan(M_s) && !isnan(R_s) && R_s > 0

    r = 0
    j = 0
    for k in p_idx
        block = layout.planet_blocks[k]
        has_K(block) || continue
        j += 1
        has_any_rm(modes[k]) || continue
        r += 1
        j_active[r]    = j
        is_reloaded[r] = has_rm_r(modes[k])
        is_arome[r]    = has_rm_a(modes[k])

        # λ from explicit layout slot
        λs[r] = planet_lambda(theta, k)

        # b, rr from transit geometry (planet must have PM mode — RM
        # requires PM by construction; we keep the gate cheap and just
        # decode b_rr).
        b_raw, rr_raw = planet_b_rr(theta, k)
        if config.parametrization.geom === :r1r2
            # b_raw, rr_raw came back as r1, r2 — convert to (b, rr).
            r1, r2 = b_raw, rr_raw
            rr_for_rm = r1 * r2
            b_for_rm  = r1 * (1 - rr_for_rm) / 2
            bs[r], rrs[r] = b_for_rm, rr_for_rm
        else
            bs[r], rrs[r] = b_raw, rr_raw
        end

        # a_Rs: rho_s path (preferred) or M_s+R_s fallback
        if use_rho
            a_Rs[r] = rho_s_to_a_Rs(rho_val, Ps[j])
        elseif have_ms_rs
            P_s = Ps[j] * T(86400.0)
            GM = T(1.3271244e26) * M_s
            a_cm = cbrt(GM * P_s^2 / (4 * T(π)^2))
            a_Rs[r] = a_cm / (R_s * T(6.9570e10))
        else
            return (-1, nothing)
        end
    end

    v_sini = system_vsini(theta)

    # LD from first PM instrument's (q1, q2). Defaults to uniform disk
    # when no PM instrument is configured.
    u1, u2 = zero(T), zero(T)
    n_pm_inst = length(layout.systemic.ld_q1)
    if n_pm_inst > 0 && layout.systemic.ld_q1[1] > 0
        q1 = theta.values[layout.systemic.ld_q1[1]]
        q2 = theta.values[layout.systemic.ld_q2[1]]
        u1, u2 = kipping_q_to_u(q1, q2)
    end

    # ARoME line widths (zero unless some planet uses :RM_A).
    σ0 = zero(T); βp = zero(T)
    if layout.systemic.rm_sigma_ccf > 0
        σ0 = theta.values[layout.systemic.rm_sigma_ccf]
        βp = theta.values[layout.systemic.rm_beta_p]
    end

    return (n_rm, (; j_active, λs, bs, rrs, a_Rs, v_sini, u1, u2,
                     is_reloaded, is_arome, σ0, βp))
end

"""
    rm_contribution(t, n_rm, state, Ps, es, ws, Tps) -> ΔRV [m/s]

Per-observation RM contribution for the RV likelihood inner loop.
Sums the in-transit RM anomalies from all RM-enabled planets at time
`t`. Returns 0 when no planet is currently in transit.
"""
@inline function rm_contribution(t::Real, n_rm::Int, state,
                                   Ps::AbstractVector{T},
                                   es::AbstractVector{T},
                                   ws::AbstractVector{T},
                                   Tps::AbstractVector{T}) where {T}
    n_rm == 0 && return zero(T)
    Δ = zero(T)
    @inbounds for r in 1:n_rm
        j = state.j_active[r]
        x_sky, y_sky, z_los = planet_sky_position(t, Ps[j], es[j], ws[j], Tps[j],
                                                   state.bs[r], state.a_Rs[r])
        # Reject superior conjunction: the sky-separation test below is satisfied
        # at BOTH conjunctions, so without this the occultation would produce a
        # mirrored RM anomaly of equal magnitude.
        z_los > 0 || continue
        z2 = x_sky * x_sky + y_sky * y_sky
        limit = 1 + state.rrs[r]
        z2 > limit * limit && continue
        z = sqrt(z2)
        Δflux = 1 - transit_flux(z, state.rrs[r], state.u1, state.u2)
        Δflux > 0 || continue
        Δ += if state.is_arome[r]
            rm_signal_arome(x_sky, y_sky, Δflux, state.v_sini, state.λs[r],
                             state.σ0, state.βp)
        elseif state.is_reloaded[r]
            rm_reloaded_signal(x_sky, y_sky, state.rrs[r],
                                state.u1, state.u2,
                                state.v_sini, state.λs[r], Δflux)
        else
            rm_signal(x_sky, y_sky, Δflux, state.v_sini, state.λs[r])
        end
    end
    return Δ
end

"""
    rm_reloaded_signal(x_p, y_p, rr, u1, u2, v_sini, λ, Δflux;
                        N_grid = 21) -> ΔRV [m/s]

Reloaded RM contribution (Cegla+ 2016). Integrates the intensity-
weighted line-of-sight velocity over the planet's sky-projected disk,
then multiplies by `−Δflux`:

    ΔRV = −Δflux · ⟨v_los⟩_planet-disk

`⟨v_los⟩` uses the same quadratic-LD intensity `I(μ) = 1 − u₁(1−μ) −
u₂(1−μ)²` and solid-body rotation `v_los = V·sin(i_*)·(x·cos λ −
y·sin λ)` as the flux-weighted form. In the small-rr limit it reduces to it
formula (the disk-mean velocity → the velocity at the planet centre).

`N_grid` is the per-axis Cartesian sub-cell count over the bounding
box `[-rr, +rr]²`. Cegla+ 2016 used 51×51; 21×21 (≈260 cells) is
sub-percent for rr ≲ 0.1.

Limitations of this v1: solid-body rotation only (no differential
rotation), no convective blueshift, no granulation noise. Add those as
optional terms in a follow-up.
"""
@inline function rm_reloaded_signal(x_p::Real, y_p::Real, rr::Real,
                                      u1::Real, u2::Real,
                                      v_sini::Real, λ::Real,
                                      Δflux::Real;
                                      N_grid::Int = 21)
    TT = promote_type(typeof(x_p), typeof(v_sini), typeof(Δflux))
    Δflux > 0 || return zero(TT)
    sin_λ, cos_λ = sincos(λ)
    step  = (2 * TT(rr)) / N_grid
    rr2   = TT(rr) * TT(rr)
    sum_F  = zero(TT)
    sum_FV = zero(TT)
    @inbounds for ix in 1:N_grid
        dx = -TT(rr) + (ix - TT(0.5)) * step
        for iy in 1:N_grid
            dy = -TT(rr) + (iy - TT(0.5)) * step
            (dx * dx + dy * dy) > rr2 && continue
            x = TT(x_p) + dx
            y = TT(y_p) + dy
            r2 = x * x + y * y
            r2 >= one(TT) && continue   # outside stellar disk
            μ = sqrt(one(TT) - r2)
            one_minus_μ = one(TT) - μ
            I = one(TT) - u1 * one_minus_μ - u2 * one_minus_μ * one_minus_μ
            I > 0 || continue
            v_los = TT(v_sini) * (x * cos_λ - y * sin_λ)
            sum_F  += I
            sum_FV += I * v_los
        end
    end
    sum_F > 0 || return zero(TT)
    v_mean = sum_FV / sum_F
    return -TT(Δflux) * v_mean
end
