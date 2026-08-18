# Astrometry log-likelihoods.
#
# Two terms in Phase 1:
#
#  * `relastrom_log_likelihood`  — Gaussian in tangent-plane (ΔRA, Δdec)
#                                  for each `RelAstromData` epoch, with
#                                  optional RA-Dec correlation per epoch.
#                                  Multi-companion: each row's
#                                  `planet_idx` selects which orbit to
#                                  evaluate.
#
#  * `hgca_log_likelihood`       — quadratic form in 3-PM residuals
#                                  (Hipparcos, long-baseline, Gaia) per
#                                  axis, with the per-axis 3×3 covariance.
#                                  HGCA Mode A: instantaneous PM at the
#                                  three reference epochs, no scan-window
#                                  absorption (Mode B / IAD+GOST is
#                                  Phase 2).
#
# Analytic marginalization of nuisance linear parameters (parallax,
# barycenter PM components) lives in `marginalization.jl` and is called
# from `hgca_log_likelihood` when enabled.
#
# Element-type generic: passes ForwardDiff Duals through. The
# PlanetOrbits-derived sky-plane evaluation in `projection.jl` is the
# only cost driver per epoch.

using LinearAlgebra: cholesky, Symmetric, issuccess

# ---------------------------------------------------------------------
# Relative astrometry
# ---------------------------------------------------------------------

"""
    relastrom_chi2_one(Δra_obs, Δdec_obs, σra, σdec, corr,
                      Δra_mod, Δdec_mod) -> χ²

χ² for one (RA, Dec) measurement with covariance

    Σ = [σra²        ρ σra σdec;
         ρ σra σdec  σdec²      ]

(ρ = `corr`). Inverts in closed form to avoid building 2×2 matrices in
the inner loop. Returns the log-determinant + Mahalanobis form
together so the full Gaussian log-likelihood is just `-0.5 χ²`.

Returns a tuple `(χ²_quadratic, log_det_Σ)`.
"""
@inline function relastrom_chi2_one(σra::Real, σdec::Real, corr::Real,
                                    rra::Real, rdec::Real)
    one_minus_ρ² = max(1 - corr * corr, oftype(corr, 1e-12))
    inv_factor = inv(σra * σdec * one_minus_ρ²)
    quad = (rra * rra / σra^2 + rdec * rdec / σdec^2 -
            2 * corr * rra * rdec / (σra * σdec)) / one_minus_ρ²
    log_det = 2 * log(σra * σdec) + log(one_minus_ρ²)
    return quad, log_det
end

"""
    relastrom_log_likelihood(theta, data) -> ll

Gaussian log-likelihood of all `RelAstromData` epochs. Loops over
each epoch, decodes the orbit for the binding planet `k`, projects to
sky-plane offset via `relastrom_offset`, and accumulates `-0.5 χ²` plus
a `-0.5 log det Σ - log(2π)` term per measurement.

Returns 0 if `data.relastrom` is empty.
"""
function relastrom_log_likelihood(theta::Theta{T}, data) where {T<:Real}
    relast = data.relastrom
    relast === nothing && return zero(T)
    n = n_relast(relast)
    n == 0 && return zero(T)

    M_pri = astrom_M_pri(theta)
    plx   = astrom_plx(theta)
    t_ref = data.t_ref
    log_2π = oftype(plx, log(2π))

    # Memoize the per-planet orbit construction across epochs. Previously
    # `_planet_orbit` was called once per epoch even though it returns
    # the same orbit when the planet index is unchanged — for visual
    # binaries (single planet, 30-50 epochs) this rebuilt the orbit
    # 30-50× per likelihood call. Tracks `prev_k` to skip rebuilds when
    # consecutive epochs target the same planet.
    ll = zero(T)
    prev_k = -1
    orb = nothing
    @inbounds for j in 1:n
        k    = relast.planet_idx[j]
        # relAST is a resolved-companion measurement — skip SB2 binary blocks
        # (unresolved; their reflex is absolute-astrometry-only). Invalid config,
        # but guard so it can't silently fit a companion position for an SB2.
        is_sb2(theta.params.layout.planet_blocks[k]) && continue
        t_j  = relast.t[j]
        if k != prev_k
            orb, _ = _planet_orbit(theta, k, M_pri, plx, t_ref)
            prev_k = k
        end
        Δra_mod, Δdec_mod = relastrom_offset(orb, t_j)

        rra  = relast.ra_off[j]  - Δra_mod
        rdec = relast.dec_off[j] - Δdec_mod
        σra  = relast.ra_err[j]
        σdec = relast.dec_err[j]
        ρ    = relast.corr[j]
        quad, logdet = relastrom_chi2_one(σra, σdec, ρ, rra, rdec)
        ll += -0.5 * (quad + logdet) - log_2π
    end
    return ll
end

# ---------------------------------------------------------------------
# HGCA absolute astrometry (Mode A: instantaneous PM at 3 epochs)
# ---------------------------------------------------------------------

"""
    hgca_log_likelihood(theta, data) -> ll

Gaussian log-likelihood of HGCA, treating the three epochs as
independent in time and using the **2×2 within-epoch RA-Dec
covariance** for each (Brandt 2021 Appendix Eq. 1).

Per-epoch term: `(r_k − μ_b)ᵀ C_k⁻¹ (r_k − μ_b)` where
`r_k = (pmra_obs[k] − reflex_RA[k], pmdec_obs[k] − reflex_Dec[k])` and
`μ_b = (μ_α*,bary, μ_δ,bary)` is the system barycentric PM (a 2-vector,
shared across the three epochs).

The barycentric PM is **analytically marginalized jointly in (RA, Dec)**:
optimal `μ_b = A⁻¹ v` where `A = Σ_k C_k⁻¹` (2×2) and
`v = Σ_k C_k⁻¹ r_k` (2-vector). Marginalized χ² is then
`Σ_k r_kᵀ C_k⁻¹ r_k − vᵀ A⁻¹ v` plus the Gaussian normalization.

Returns 0 if `data.hgca` is `nothing`.
"""
function hgca_log_likelihood(theta::Theta{T}, data) where {T<:Real}
    hgca = data.hgca
    hgca === nothing && return zero(T)

    M_pri  = astrom_M_pri(theta)
    plx    = astrom_plx(theta)
    t_ref  = data.t_ref
    log_2π = oftype(plx, log(2π))

    # Sum of stellar reflex PM at the three epochs from all astrometry-
    # bearing companions.
    #
    # Hip epoch (ie=1) and Gaia epoch (ie=3) are catalog INSTANTANEOUS PMs
    # at their reference epochs. The Hipparcos–Gaia epoch (ie=2) is NOT
    # instantaneous: it is the MEAN reflex velocity over the ~25 yr baseline
    # = (Gaia pos − Hip pos)/baseline (Brandt 2021 Eq. 1), handled below.
    #
    # Gaia epoch: if GOST scan-plan data are supplied, use the orbit-induced
    # catalog-PM shift from a 5-param fit through the actual GOST transits
    # (HGCA Mode B; Brandt 2018 §4); otherwise the instantaneous reflex PM at
    # the Gaia reference epoch. Mode B matters for `P` ≲ Gaia mission window.
    pmra_mod  = zeros(T, 3)
    pmdec_mod = zeros(T, 3)
    use_gost_mode_b = data.gost !== nothing
    for k in planet_indices(theta)
        block = theta.params.layout.planet_blocks[k]
        has_AS(block) || continue
        # Per-planet astrometric coupling. A companion the RV has established
        # can still be astrometrically undetected; when the mask says so it
        # contributes NO reflex here, and its inc/Omega carry their priors.
        # Defaults true, so fixed-dim behaviour is unchanged.
        is_planet_as_active(theta, k) || continue
        orb, M_sec = _planet_orbit(theta, k, M_pri, plx, t_ref)
        for ie in 1:3
            t_e = hgca.epochs[ie]
            if ie == 2
                # HG epoch: the catalog "Hipparcos–Gaia" PM is the SCALED
                # POSITIONAL DIFFERENCE (Gaia pos − Hip pos)/baseline — i.e. the
                # MEAN reflex velocity over the ~25 yr baseline, not the
                # instantaneous PM at the midpoint (Brandt 2021 Eq. 1). For
                # P ≫ baseline (e.g. HD 159062, P=354 yr) the two agree to
                # <0.1 mas/yr, but for P comparable to the baseline (e.g.
                # HD 4747, P≈33 yr → ~75% of an orbit over the baseline) the
                # reflex is highly non-linear and the instantaneous-PM
                # approximation badly biases the inferred mass low.
                oH = star_reflex_offset(orb, hgca.epochs[1], M_sec)   # Hip
                oG = star_reflex_offset(orb, hgca.epochs[3], M_sec)   # Gaia
                dt_yr = (hgca.epochs[3] - hgca.epochs[1]) / oftype(t_e, 365.25)
                μra, μdec = (oG[1] - oH[1]) / dt_yr, (oG[2] - oH[2]) / dt_yr
            elseif ie == 3 && use_gost_mode_b
                μra, μdec = gost_window_avg_pm(orb, data.gost, M_sec)
            else
                μra, μdec = star_reflex_pm(orb, t_e, M_sec)
            end
            pmra_mod[ie]  += μra
            pmdec_mod[ie] += μdec
        end
    end

    # Build the 2-vector residuals and accumulate (A, v, Σ_k r'C⁻¹r)
    # via per-epoch 2×2 inversion.
    A11 = zero(T); A12 = zero(T); A22 = zero(T)        # Σ_k C_k⁻¹
    v1  = zero(T); v2  = zero(T)                        # Σ_k C_k⁻¹ r_k
    rCr = zero(T)                                       # Σ_k r_kᵀ C_k⁻¹ r_k
    log_det_C_sum = zero(T)
    @inbounds for k in 1:3
        # 2×2 covariance for epoch k
        a = hgca.cov_ep[k][1, 1]   # σ_RA²
        b = hgca.cov_ep[k][1, 2]   # σ_RA σ_Dec ρ
        c = hgca.cov_ep[k][2, 2]   # σ_Dec²
        det_C = a * c - b * b
        # 2×2 inverse: (1/det) [c -b; -b a]
        invC11 =  c / det_C
        invC12 = -b / det_C
        invC22 =  a / det_C
        # Residuals
        rra_k  = hgca.pmra[k]  - pmra_mod[k]
        rdec_k = hgca.pmdec[k] - pmdec_mod[k]
        # rᵀ C⁻¹ r
        rCr += invC11 * rra_k * rra_k + 2 * invC12 * rra_k * rdec_k +
               invC22 * rdec_k * rdec_k
        # C⁻¹ r → contributes to v
        v1 += invC11 * rra_k + invC12 * rdec_k
        v2 += invC12 * rra_k + invC22 * rdec_k
        # Accumulate A = Σ C⁻¹
        A11 += invC11
        A12 += invC12
        A22 += invC22
        # Log-det
        log_det_C_sum += log(det_C)
    end

    # Marginalized barycentric μ_b = A⁻¹ v
    det_A = A11 * A22 - A12 * A12
    invA11 =  A22 / det_A
    invA12 = -A12 / det_A
    invA22 =  A11 / det_A
    # vᵀ A⁻¹ v
    vAv = invA11 * v1 * v1 + 2 * invA12 * v1 * v2 + invA22 * v2 * v2
    χ2_min = rCr - vAv
    # Log-norm of marginalization: log det(A) is the prior-independent
    # multiplicative factor in the Gaussian integration over μ_b.
    log_norm = log_det_C_sum + log(det_A)

    # Total HGCA log-likelihood: −0.5 × (χ²_min + log_norm) − 3·log(2π)
    # (3 epochs × 2 DOF = 6, but 2 DOF are absorbed in the marginalization,
    # leaving 4 effective DOF; the −2·log(2π) accounts for the joint μ_b
    # marginalization Jacobian).
    return -0.5 * (χ2_min + log_norm) - 2 * log_2π
end

# ---------------------------------------------------------------------
# Hipparcos IAD (along-scan abscissa) — scaffolding
# ---------------------------------------------------------------------

"""
    along_scan_projection(Δra, Δdec, ψ) -> Δη

Project a tangent-plane (ΔRA·cos δ, Δδ) offset onto the along-scan
direction at angle ψ (rad) measured from local RA:

    Δη = Δra · sin ψ + Δdec · cos ψ

This is the standard Hipparcos / Gaia along-scan decomposition (see
e.g. Brandt 2018 Eq. 3, htof's `AlongScanMotion` class).

Sign convention: ψ = 0 points along +Dec, ψ = π/2 points along +RA,
so the along-scan unit vector in (RA*, Dec) is (sin ψ, cos ψ). This is
PINNED to htof: `_parse_van_leeuwen_iad` sets ψ = atan2(CPSI, SPSI),
which equals htof's `scan_angle` θ (htof: file CPSI = sin θ, SPSI =
cos θ; `special_parse.to_along_scan_basis(ra,dec,θ) = dec·cos θ +
ra·sin θ`). Verified against htof's parser to machine precision on
HIP 24205 (residuals & parallax factors identical). NB: an earlier
ψ = atan2(SPSI, CPSI) silently swapped RA↔Dec in the reflex
projection — harmless to the 5-param offset/PM marginalisation (same
2-D span) but it scrambled the orbit reflex against the data.
"""
@inline function along_scan_projection(Δra::Real, Δdec::Real, ψ::Real)
    s, c = sincos(ψ)
    return Δra * s + Δdec * c
end

"""
    iad_log_likelihood(theta, data) -> ll

Marginalized log-likelihood of Hipparcos IAD: per-transit Gaussian
on the along-scan abscissa residual after subtracting the orbit's
along-scan reflex AND analytically marginalizing over a five-parameter
correction `(Δα0, Δδ0, Δϖ, Δμα*, Δμδ)` to the catalog astrometric
solution. This is the algorithm of htof (Brandt 2018, ApJS 239 31)
and van Leeuwen 2007 §17.3 — re-fit the 5-parameter solution at every
likelihood call given the current orbit, then evaluate the residuals.

# Algorithm

For each transit `i = 1..n`:

  predicted_along_scan_i = Δη_orbit_i +
                           Δα0 · sin ψ_i + Δδ0 · cos ψ_i +
                           Δϖ  · plx_factor_i +
                           Δμα* · sin ψ_i · pm_factor_i +
                           Δμδ  · cos ψ_i · pm_factor_i

where `Δη_orbit_i = Σ_k along_scan_projection(reflex_k(t_i), ψ_i)`
sums the stellar reflex from every astrometry-bearing companion.

Define the design vector `xᵢ = (sin ψ_i, cos ψ_i, plx_factor_i,
sin ψ_i · pm_factor_i, cos ψ_i · pm_factor_i)` and the orbit-corrected
data `r_i = abscissa_i − Δη_orbit_i`. Under flat priors on the 5-param
catalog correction `q`, the marginalized χ² is

    χ²_min(orbit) = rᵀ W r − vᵀ A⁻¹ v

with `W = diag(1/σ_i²)`, `A = Xᵀ W X` (5×5), `v = Xᵀ W r` (5-vec).
The Gaussian normalization picks up `−½ log det A` from the
marginalization Jacobian and `−Σ log σ_i` from the per-transit
Gaussian.

Note that `A` and `Σ log σ_i` depend only on the IAD design (ψ, σ,
plx_factor, pm_factor), not on the orbit. They contribute additive
constants to the log-likelihood — fine for orbit-fitting and for log-Z
estimates, modulo a transit-independent offset shared with any other
model.

Requires `n ≥ 5` transits (otherwise the 5-param correction is
underdetermined and the likelihood is undefined). Returns 0 if
`data.iad` is `nothing` or empty.
"""
function iad_log_likelihood(theta::Theta{T}, data) where {T<:Real}
    iad = data.iad
    iad === nothing && return zero(T)
    n = n_iad(iad)
    n == 0 && return zero(T)
    # Need ≥ 5 transits for the marginalization to be well-posed.
    # Real Hipparcos sources have 30–150; this guard catches synthetic
    # under-determined cases cleanly.
    n >= 5 || return zero(T)

    # If a published Gaia DR3 5-param solution is supplied alongside IAD
    # AND we have GOST scan plans for the orbit's Gaia-epoch shift,
    # dispatch to the joint htof-style likelihood. This is the htof
    # primary use case: marginalize over the catalog-truth 5-vector
    # using BOTH Hipparcos abscissae and the Gaia 5-param solution as
    # constraints on a single shared `q = (α₀, δ₀, ϖ, μα*, μδ)`.
    if data.gaia_dr3 !== nothing && data.gost !== nothing
        return _iad_gaia_joint_log_likelihood(theta, data, iad)
    end

    M_pri  = astrom_M_pri(theta)
    plx    = astrom_plx(theta)
    t_ref  = data.t_ref
    log_2π = oftype(plx, log(2π))

    # Pre-build per-planet (orb, M_sec) ONCE per likelihood call. The
    # original `for j in epochs / for k in planets / build orb` pattern
    # rebuilt the same orbit n_iad × n_planets times per call (Hipparcos
    # IAD: 30-150 transits × few planets = hundreds of needless orbit
    # constructions). Now built once per planet per call.
    active_ks = Int[]
    orbs      = Any[]
    M_secs    = T[]
    for k in planet_indices(theta)
        block = theta.params.layout.planet_blocks[k]
        has_AS(block) || continue
        # Per-planet astrometric coupling. A companion the RV has established
        # can still be astrometrically undetected; when the mask says so it
        # contributes NO reflex here, and its inc/Omega carry their priors.
        # Defaults true, so fixed-dim behaviour is unchanged.
        is_planet_as_active(theta, k) || continue
        orb_k, M_sec_k = _planet_orbit(theta, k, M_pri, plx, t_ref)
        push!(active_ks, k)
        push!(orbs, orb_k)
        push!(M_secs, M_sec_k)
    end

    # Stage 1 — build orbit-subtracted residuals r_i = abscissa_i − Δη_orbit_i.
    r = Vector{T}(undef, n)
    @inbounds for j in 1:n
        t_j = iad.t[j]
        ψ_j = iad.psi[j]
        Δη_mod = zero(T)
        for ki in eachindex(active_ks)
            Δra, Δdec = star_reflex_offset(orbs[ki], t_j, M_secs[ki])
            Δη_mod += along_scan_projection(Δra, Δdec, ψ_j)
        end
        r[j] = iad.abscissa[j] - Δη_mod
    end

    # Stage 2 — accumulate (rᵀ W r), (Xᵀ W r), (Xᵀ W X). The design vector
    # x_j = (sin ψ, cos ψ, plx_factor, sin ψ · pm_factor, cos ψ · pm_factor).
    A11 = zero(T); A12 = zero(T); A13 = zero(T); A14 = zero(T); A15 = zero(T)
    A22 = zero(T); A23 = zero(T); A24 = zero(T); A25 = zero(T)
    A33 = zero(T); A34 = zero(T); A35 = zero(T)
    A44 = zero(T); A45 = zero(T)
    A55 = zero(T)
    v1 = zero(T); v2 = zero(T); v3 = zero(T); v4 = zero(T); v5 = zero(T)
    rWr = zero(T)
    log_sigma_sum = zero(T)

    @inbounds for j in 1:n
        s, c = sincos(iad.psi[j])
        plxf = iad.parallax_factor[j]
        pmf  = iad.pm_factor[j]
        x1 = s
        x2 = c
        x3 = plxf
        x4 = s * pmf
        x5 = c * pmf
        σ  = iad.abscissa_err[j]
        w  = 1 / (σ * σ)
        rj = r[j]

        rWr += w * rj * rj
        log_sigma_sum += log(σ)

        v1 += w * x1 * rj
        v2 += w * x2 * rj
        v3 += w * x3 * rj
        v4 += w * x4 * rj
        v5 += w * x5 * rj

        A11 += w * x1 * x1
        A12 += w * x1 * x2
        A13 += w * x1 * x3
        A14 += w * x1 * x4
        A15 += w * x1 * x5
        A22 += w * x2 * x2
        A23 += w * x2 * x3
        A24 += w * x2 * x4
        A25 += w * x2 * x5
        A33 += w * x3 * x3
        A34 += w * x3 * x4
        A35 += w * x3 * x5
        A44 += w * x4 * x4
        A45 += w * x4 * x5
        A55 += w * x5 * x5
    end

    # Build symmetric A as a small 5×5 matrix and Cholesky-factor it.
    # A is data-only (no orbit dependence) so its element type is real;
    # but to keep dispatch generic we promote to T.
    A = Matrix{T}(undef, 5, 5)
    A[1,1]=A11; A[1,2]=A12; A[1,3]=A13; A[1,4]=A14; A[1,5]=A15
    A[2,1]=A12; A[2,2]=A22; A[2,3]=A23; A[2,4]=A24; A[2,5]=A25
    A[3,1]=A13; A[3,2]=A23; A[3,3]=A33; A[3,4]=A34; A[3,5]=A35
    A[4,1]=A14; A[4,2]=A24; A[4,3]=A34; A[4,4]=A44; A[4,5]=A45
    A[5,1]=A15; A[5,2]=A25; A[5,3]=A35; A[5,4]=A45; A[5,5]=A55

    v_vec = T[v1, v2, v3, v4, v5]

    # Solve A · q_opt = v. Use Cholesky on A (positive-definite by
    # construction when the design matrix has rank 5).
    chol = cholesky(Symmetric(A); check = false)
    if !issuccess(chol)
        # Design rank-deficient — typically because `parallax_factor`
        # and/or `pm_factor` were not supplied (zero-vector defaults).
        # Fall back to a non-marginalized Gaussian: just penalize the
        # orbit-residual `r` against the per-transit error. This loses
        # the catalog-correction degree-of-freedom absorption but keeps
        # the likelihood meaningful for forward-modeling-only test
        # inputs.
        return -0.5 * rWr - log_sigma_sum - 0.5 * n * log_2π
    end
    q_opt = chol \ v_vec

    χ²_min = rWr - (v1*q_opt[1] + v2*q_opt[2] + v3*q_opt[3] +
                    v4*q_opt[4] + v5*q_opt[5])
    log_det_A = 2 * (log(chol.L[1,1]) + log(chol.L[2,2]) + log(chol.L[3,3]) +
                     log(chol.L[4,4]) + log(chol.L[5,5]))

    # Marginalized log-likelihood:
    #   −½ χ²_min − Σ log σ_i − ½ log det A − (n − 5)/2 · log(2π)
    # The (n−5) factor reflects the 5 absorbed catalog DOF; A's log-det
    # is the marginalization Jacobian (analogous to the HGCA term).
    return -0.5 * χ²_min - log_sigma_sum - 0.5 * log_det_A -
           0.5 * (n - 5) * log_2π
end

# ---------------------------------------------------------------------
# Joint Hipparcos IAD + Gaia DR3 5-param (htof port)
# ---------------------------------------------------------------------
#
# When BOTH `data.iad` and `data.gaia_dr3` (+ `data.gost` scan plan)
# are present, we compute the htof-equivalent joint likelihood:
# marginalize over a SHARED catalog 5-vector q = (α₀, δ₀, ϖ, μα*, μδ)
# constrained simultaneously by the Hipparcos along-scan abscissae
# AND the published Gaia DR3 5-param solution.
#
# This recovers more information than either piece alone:
#   * IAD alone (`iad_log_likelihood`) marginalizes a Hipparcos-epoch
#     5-vector. Sees only Hip-era astrometry.
#   * Gaia DR3 alone is degenerate without an anchor.
#   * Joint: the SAME q is propagated linearly to both Hip and Gaia
#     reference epochs via the PM transport, breaking the degeneracy
#     and yielding the long-baseline acceleration constraint htof was
#     built to compute.
#
# Mathematical setup
# ------------------
# Let `q` (5-vector) be the catalog 5-param solution at the *Hipparcos*
# reference epoch (we choose this as the canonical reference; Gaia's
# 5-vector at its own reference epoch is then `P(Δt) q` with `P(Δt)`
# the linear PM-transport matrix).
#
#   P(Δt) =  [ 1  0  0  Δt 0 ;
#              0  1  0  0  Δt;
#              0  0  1  0  0 ;
#              0  0  0  1  0 ;
#              0  0  0  0  1 ]
#
# Hipparcos transit `i` measures along-scan abscissa
#   a_h_i = X_h_i · q + Δη_orb_h_i + ε_i,    ε_i ~ N(0, σ_h_i²)
# where X_h_i = (sin ψ, cos ψ, plx_factor, sin ψ · Δt_i, cos ψ · Δt_i)
# is the IAD design row at the Hip reference epoch.
#
# Gaia publishes `y_g` (5-vector) with covariance Σ_g, related to q by
#   y_g = P(Δt_g) q + Δq_orb + η,    η ~ N(0, Σ_g)
# where Δt_g = (gaia_dr3.t_ref − iad.t_ref) / yr, and Δq_orb is the
# orbit-induced 5-param shift forecast by `gost_5param_fit` (computed
# at the Gaia reference epoch via GOST scan plans).
#
# Marginalize q (flat prior):
#   A = Σ_i (1/σ²) X_h_i X_h_iᵀ + P(Δt)ᵀ Σ_g⁻¹ P(Δt)         (5×5)
#   v = Σ_i (1/σ²) X_h_i (a_h_i − Δη_orb_h_i)
#       + P(Δt)ᵀ Σ_g⁻¹ (y_g − Δq_orb)                          (5-vec)
#   χ²_min = Σ_i (1/σ²)(a_h_i − Δη_orb_h_i)²
#          + (y_g − Δq_orb)ᵀ Σ_g⁻¹ (y_g − Δq_orb) − vᵀA⁻¹v
#
# Log-likelihood:
#   ln L = −½ χ²_min − Σ log σ_h_i − ½ log det Σ_g − ½ log det A
#          − ½ (n_h + 5 − 5) log(2π)
#        = −½ χ²_min − Σ log σ_h_i − ½ log det Σ_g − ½ log det A
#          − ½ n_h log(2π)
#
# The (n_h + 5 − 5) DOF reflects: n_h IAD + 5 Gaia − 5 marginalized.
#
# Numerics
# --------
# Σ_g is data-only — Cholesky once per likelihood call. A is built per
# call (orbit-dependent through Δq_orb on the RHS, but only through v;
# A's *structure* is data-only, so log_det_A is constant over orbit
# samples — kept here for log-Z compatibility). The Hipparcos block
# of A and `Σ log σ` are likewise constant.
#
# The orbit's contribution comes through:
#   * Δη_orb_h_i  (per-IAD-transit reflex projection)
#   * Δq_orb     (Gaia 5-param shift forecast from GOST)

"""
    _iad_gaia_joint_log_likelihood(theta, data, iad) -> ll

Joint Hipparcos IAD + Gaia DR3 5-param marginalized log-likelihood.
Internal helper called by `iad_log_likelihood` when both data sources
are present. See block comment above for the math.

Element-type generic: `T = element type of theta` floats through
orbit-evaluated quantities (Δη and Δq_orb).
"""
function _iad_gaia_joint_log_likelihood(theta::Theta{T}, data, iad) where {T<:Real}
    gaia    = data.gaia_dr3
    gost    = data.gost
    n_h     = n_iad(iad)
    M_pri   = astrom_M_pri(theta)
    plx     = astrom_plx(theta)
    t_ref   = data.t_ref
    log_2π  = oftype(plx, log(2π))

    # PM transport: Hipparcos reference (use iad.t mean as Hip ref) →
    # Gaia reference (gaia.t_ref). Δt in Julian years.
    t_ref_hip = sum(iad.t) / n_h
    Δt_yr     = (gaia.t_ref - t_ref_hip) / 365.25

    # Per-planet orbit pre-build (same pattern as iad_log_likelihood).
    active_ks = Int[]
    orbs      = Any[]
    M_secs    = T[]
    for k in planet_indices(theta)
        block = theta.params.layout.planet_blocks[k]
        has_AS(block) || continue
        # Per-planet astrometric coupling. A companion the RV has established
        # can still be astrometrically undetected; when the mask says so it
        # contributes NO reflex here, and its inc/Omega carry their priors.
        # Defaults true, so fixed-dim behaviour is unchanged.
        is_planet_as_active(theta, k) || continue
        orb_k, M_sec_k = _planet_orbit(theta, k, M_pri, plx, t_ref)
        push!(active_ks, k)
        push!(orbs, orb_k)
        push!(M_secs, M_sec_k)
    end

    # Stage 1: per-IAD-transit reflex residual r_h_i = a_h_i − Δη_orb_h_i.
    # Stage 2: orbit-induced Gaia 5-param shift Δq_orb (sum across planets).
    # Stage 3: assemble A = A_h + P(Δt)ᵀ Σ_g⁻¹ P(Δt), v = v_h + P(Δt)ᵀ Σ_g⁻¹ (y_g − Δq_orb).
    # Stage 4: χ²_min, log dets, return.

    # --- Stage 1: Hipparcos residuals + design accumulation
    A11 = zero(T); A12 = zero(T); A13 = zero(T); A14 = zero(T); A15 = zero(T)
    A22 = zero(T); A23 = zero(T); A24 = zero(T); A25 = zero(T)
    A33 = zero(T); A34 = zero(T); A35 = zero(T)
    A44 = zero(T); A45 = zero(T)
    A55 = zero(T)
    v1 = zero(T); v2 = zero(T); v3 = zero(T); v4 = zero(T); v5 = zero(T)
    rWr = zero(T)
    log_sigma_sum = zero(T)

    @inbounds for j in 1:n_h
        t_j = iad.t[j]
        ψ_j = iad.psi[j]
        # Orbit reflex projection
        Δη_mod = zero(T)
        for ki in eachindex(active_ks)
            Δra, Δdec = star_reflex_offset(orbs[ki], t_j, M_secs[ki])
            Δη_mod += along_scan_projection(Δra, Δdec, ψ_j)
        end
        r_j = iad.abscissa[j] - Δη_mod
        # Design at the Hipparcos reference epoch: pm_factor counts
        # years from t_ref_hip; iad's stored pm_factor is years from
        # the *catalog* reference, which differs by a constant shift
        # (absorbed by α₀ and δ₀ — keeps the PM column orthogonal to
        # position columns through the central-epoch trick).
        s, c = sincos(ψ_j)
        plxf = iad.parallax_factor[j]
        pmf  = (t_j - t_ref_hip) / 365.25
        x1 = s
        x2 = c
        x3 = plxf
        x4 = s * pmf
        x5 = c * pmf
        σ  = iad.abscissa_err[j]
        w  = 1 / (σ * σ)

        rWr += w * r_j * r_j
        log_sigma_sum += log(σ)

        v1 += w * x1 * r_j
        v2 += w * x2 * r_j
        v3 += w * x3 * r_j
        v4 += w * x4 * r_j
        v5 += w * x5 * r_j

        A11 += w * x1 * x1; A12 += w * x1 * x2; A13 += w * x1 * x3; A14 += w * x1 * x4; A15 += w * x1 * x5
        A22 += w * x2 * x2; A23 += w * x2 * x3; A24 += w * x2 * x4; A25 += w * x2 * x5
        A33 += w * x3 * x3; A34 += w * x3 * x4; A35 += w * x3 * x5
        A44 += w * x4 * x4; A45 += w * x4 * x5
        A55 += w * x5 * x5
    end

    # --- Stage 2: orbit-induced Gaia 5-param shift, summed across planets.
    # gost_5param_fit returns Δq at GOST mean epoch; we re-evaluate it
    # at the GAIA published reference epoch by passing gaia.t_ref.
    Δq_orb = zeros(T, 5)
    for ki in eachindex(active_ks)
        Δq_k = gost_5param_fit(orbs[ki], gost, M_secs[ki]; t_ref = gaia.t_ref)
        @. Δq_orb += Δq_k
    end

    # --- Stage 3: Gaia constraint contribution
    # Residual y_g − Δq_orb (5-vector)
    y_minus_Δq = T[gaia.params[1] - Δq_orb[1],
                   gaia.params[2] - Δq_orb[2],
                   gaia.params[3] - Δq_orb[3],
                   gaia.params[4] - Δq_orb[4],
                   gaia.params[5] - Δq_orb[5]]

    # Cholesky-factor Σ_g (data-only); use it to compute Σ_g⁻¹ y' and
    # P(Δt)ᵀ Σ_g⁻¹ P(Δt) for the joint A.
    chol_Σg = cholesky(Symmetric(gaia.cov); check = false)
    if !issuccess(chol_Σg)
        # Σ_g not positive-definite — should be caught at construction
        # time, but guard defensively.
        return convert(T, -Inf)
    end
    Σg_inv_y = chol_Σg \ y_minus_Δq         # 5-vector
    log_det_Σg = 2 * (log(chol_Σg.L[1,1]) + log(chol_Σg.L[2,2]) +
                       log(chol_Σg.L[3,3]) + log(chol_Σg.L[4,4]) +
                       log(chol_Σg.L[5,5]))

    # P(Δt) is a 5×5 sparse matrix — only off-diagonal entries are
    # P[1,4] = Δt and P[2,5] = Δt. We build it explicitly to keep the
    # code legible; performance impact is negligible (5×5 once per call).
    P = Matrix{T}(undef, 5, 5)
    fill!(P, zero(T))
    @inbounds for k in 1:5
        P[k, k] = one(T)
    end
    P[1, 4] = T(Δt_yr)
    P[2, 5] = T(Δt_yr)

    # Pᵀ Σ_g⁻¹ y' (5-vec)
    PᵀΣinv_y = P' * Σg_inv_y
    @inbounds begin
        v1 += PᵀΣinv_y[1]; v2 += PᵀΣinv_y[2]; v3 += PᵀΣinv_y[3]
        v4 += PᵀΣinv_y[4]; v5 += PᵀΣinv_y[5]
    end

    # Pᵀ Σ_g⁻¹ P (5×5) → solve Σ_g · Y = P for Y (5×5), then Pᵀ · Y.
    Y_g = chol_Σg \ P
    PᵀΣinvP = P' * Y_g
    @inbounds begin
        A11 += PᵀΣinvP[1,1]; A12 += PᵀΣinvP[1,2]; A13 += PᵀΣinvP[1,3]; A14 += PᵀΣinvP[1,4]; A15 += PᵀΣinvP[1,5]
        A22 += PᵀΣinvP[2,2]; A23 += PᵀΣinvP[2,3]; A24 += PᵀΣinvP[2,4]; A25 += PᵀΣinvP[2,5]
        A33 += PᵀΣinvP[3,3]; A34 += PᵀΣinvP[3,4]; A35 += PᵀΣinvP[3,5]
        A44 += PᵀΣinvP[4,4]; A45 += PᵀΣinvP[4,5]
        A55 += PᵀΣinvP[5,5]
    end

    # Gaia residual quadratic: (y_g − Δq_orb)ᵀ Σ_g⁻¹ (y_g − Δq_orb)
    rGaia = zero(T)
    @inbounds for k in 1:5
        rGaia += y_minus_Δq[k] * Σg_inv_y[k]
    end

    # --- Stage 4: solve and assemble
    A = Matrix{T}(undef, 5, 5)
    A[1,1]=A11; A[1,2]=A12; A[1,3]=A13; A[1,4]=A14; A[1,5]=A15
    A[2,1]=A12; A[2,2]=A22; A[2,3]=A23; A[2,4]=A24; A[2,5]=A25
    A[3,1]=A13; A[3,2]=A23; A[3,3]=A33; A[3,4]=A34; A[3,5]=A35
    A[4,1]=A14; A[4,2]=A24; A[4,3]=A34; A[4,4]=A44; A[4,5]=A45
    A[5,1]=A15; A[5,2]=A25; A[5,3]=A35; A[5,4]=A45; A[5,5]=A55
    v_vec = T[v1, v2, v3, v4, v5]

    chol_A = cholesky(Symmetric(A); check = false)
    if !issuccess(chol_A)
        # Joint design rank-deficient — extremely rare with both Hip and
        # Gaia constraints (each pins 5 DOF). Fall back to non-marginalized.
        return -0.5 * (rWr + rGaia) - log_sigma_sum -
               0.5 * log_det_Σg - 0.5 * (n_h + 5) * log_2π
    end
    q_opt = chol_A \ v_vec
    χ²_min = rWr + rGaia - (v1*q_opt[1] + v2*q_opt[2] + v3*q_opt[3] +
                             v4*q_opt[4] + v5*q_opt[5])
    log_det_A = 2 * (log(chol_A.L[1,1]) + log(chol_A.L[2,2]) +
                      log(chol_A.L[3,3]) + log(chol_A.L[4,4]) +
                      log(chol_A.L[5,5]))

    # ln L = −½ χ²_min − Σ log σ_h − ½ log det Σ_g − ½ log det A − ½ n_h log(2π)
    return -0.5 * χ²_min - log_sigma_sum - 0.5 * log_det_Σg -
           0.5 * log_det_A - 0.5 * n_h * log_2π
end

# ---------------------------------------------------------------------
# Gaia GOST (along-scan transit forward model) — scaffolding
# ---------------------------------------------------------------------

"""
    gost_log_likelihood(theta, data) -> ll

Returns 0 by design — GOST's likelihood contribution is **folded into
`hgca_log_likelihood`** as the HGCA Mode-B upgrade (Brandt 2018 §4).

When `data.gost` and `data.hgca` are both present, the Gaia-epoch
reflex term in the HGCA likelihood is replaced from the instantaneous
`star_reflex_pm(orb, hgca.epochs[3], M_sec)` value by the
GOST-window-averaged value from `gost_window_avg_pm(orb, data.gost,
M_sec)`. The latter does a 5-parameter fit to the orbit-induced
along-scan signal across the actual Gaia mission transits — i.e.
exactly what Gaia's catalog pipeline does, modulo the per-transit
weighting.

Mode B differs from Mode A only when the orbit's period is comparable
to or shorter than Gaia's ~3-yr mission window. For Phase 1 targets
(P ≳ 30 yr), the two agree to well below the HGCA error budget; the
Mode B path keeps the formulation honest for short-period regimes.

Without HGCA, GOST alone is uninformative (no paired multi-epoch
reference) and returns 0.

Future Gaia DR4 epoch astrometry would warrant a real per-transit
Gaussian against `data.gost`; that needs a `GaiaEpochData` type that
isn't in Nereus yet.
"""
function gost_log_likelihood(theta::Theta{T}, data) where {T<:Real}
    gost = data.gost
    gost === nothing && return zero(T)
    n = n_gost(gost)
    n == 0 && return zero(T)

    # Forward-model the projection so the call path is exercised, then
    # discard the result (stub). This keeps the function autodiff-clean
    # even before a real likelihood lands — a user adding `data.gost`
    # cannot accidentally NaN the log-density.
    M_pri = astrom_M_pri(theta)
    plx   = astrom_plx(theta)
    t_ref = data.t_ref

    # Pre-build per-planet orbits once (see iad_log_likelihood comment)
    active_ks = Int[]
    orbs      = Any[]
    M_secs    = T[]
    for k in planet_indices(theta)
        block = theta.params.layout.planet_blocks[k]
        has_AS(block) || continue
        # Per-planet astrometric coupling. A companion the RV has established
        # can still be astrometrically undetected; when the mask says so it
        # contributes NO reflex here, and its inc/Omega carry their priors.
        # Defaults true, so fixed-dim behaviour is unchanged.
        is_planet_as_active(theta, k) || continue
        orb_k, M_sec_k = _planet_orbit(theta, k, M_pri, plx, t_ref)
        push!(active_ks, k)
        push!(orbs, orb_k)
        push!(M_secs, M_sec_k)
    end

    Δη_sink = zero(T)
    @inbounds for j in 1:n
        t_j = gost.t[j]
        ψ_j = gost.psi[j]
        for ki in eachindex(active_ks)
            Δra, Δdec = star_reflex_offset(orbs[ki], t_j, M_secs[ki])
            Δη_sink += along_scan_projection(Δra, Δdec, ψ_j)
        end
    end
    # TODO: replace with htof port — for now return 0 regardless of
    # the forward model. Multiplying by zero keeps autodiff happy and
    # documents the intent ("evaluated, then discarded").
    return zero(T) * Δη_sink
end

# ---------------------------------------------------------------------
# G23H (Thompson+ 2026) — 5-PM joint absolute astrometry
# ---------------------------------------------------------------------

"""
    g23h_log_likelihood(theta, data) -> ll

Joint log-likelihood for the Thompson+ 2026 G23H catalog
(arXiv:2602.00235): five proper-motion measurements per source —
Hipparcos, Hipparcos→Gaia long-baseline, Gaia DR2, Gaia DR3−DR2
derived, Gaia DR3 — with a full 10×10 covariance that captures the
DR2↔DR3 cross-correlation arising from shared transits.

# Algorithm

For each of the five reference epochs `t_e` we forward-model the
stellar reflex PM `(μ_α*_e, μ_δ_e) = Σ_k star_reflex_pm(orb_k, t_e,
M_sec_k)` over astrometry-bearing companions. Stack the five 2-vector
residuals into a 10-vector `r ∈ ℝ^10` ordered
`(pmra_1, pmdec_1, ..., pmra_5, pmdec_5)`.

The system barycentric PM `μ_b ∈ ℝ²` is shared across all five
epochs and analytically marginalized: with design matrix `X ∈ ℝ^{10×2}`
whose row `2k−1` is `(1, 0)` and row `2k` is `(0, 1)`, the optimal
`μ_b = (XᵀΣ⁻¹X)⁻¹ XᵀΣ⁻¹ r` and the marginalized chi-squared is

    χ²_min = rᵀ Σ⁻¹ r − vᵀ A⁻¹ v

with `A = XᵀΣ⁻¹X` (2×2) and `v = XᵀΣ⁻¹r` (2-vector). Returns

    ln ℒ = −½ (χ²_min + log det Σ + log det A) − 4·log(2π)

(10 obs − 2 marginalized = 8 effective DOF → −8/2 ·log(2π).)

If `data.gost` is also present, the Gaia DR3 epoch (k=5) reflex is
upgraded to the GOST-window-averaged value via `gost_window_avg_pm`,
matching the HGCA Mode B treatment in `hgca_log_likelihood`.

Returns 0 if `data.g23h` is `nothing`.

# Notes
- G23H is a strict superset of HGCA (Hip + HG + DR3 are 3 of its 5
  PMs). Providing both `data.hgca` and `data.g23h` will double-count
  three measurements; users should pick one. Nereus does not throw
  on this — diagnose via posterior-predictive checks.
- UEVA, IAD jitter, and Gaia RV-variability extensions of Thompson+
  2026 are NOT included here. They require separate Nereus data
  types (`GaiaUEVAData`, `GaiaRVVarData`) — future work.
"""
function g23h_log_likelihood(theta::Theta{T}, data) where {T<:Real}
    g23h = data.g23h
    g23h === nothing && return zero(T)

    M_pri  = astrom_M_pri(theta)
    plx    = astrom_plx(theta)
    t_ref  = data.t_ref
    log_2π = oftype(plx, log(2π))

    # Forward-model stellar reflex PM at each of 5 G23H epochs.
    pmra_mod  = zeros(T, 5)
    pmdec_mod = zeros(T, 5)
    use_gost_mode_b = data.gost !== nothing
    for k in planet_indices(theta)
        block = theta.params.layout.planet_blocks[k]
        has_AS(block) || continue
        # Per-planet astrometric coupling. A companion the RV has established
        # can still be astrometrically undetected; when the mask says so it
        # contributes NO reflex here, and its inc/Omega carry their priors.
        # Defaults true, so fixed-dim behaviour is unchanged.
        is_planet_as_active(theta, k) || continue
        orb, M_sec = _planet_orbit(theta, k, M_pri, plx, t_ref)
        for ie in 1:5
            t_e = g23h.epochs[ie]
            if ie == 5 && use_gost_mode_b
                μra, μdec = gost_window_avg_pm(orb, data.gost, M_sec)
            else
                μra, μdec = star_reflex_pm(orb, t_e, M_sec)
            end
            pmra_mod[ie]  += μra
            pmdec_mod[ie] += μdec
        end
    end

    # Build 10-vector of residuals.
    r = Vector{T}(undef, 10)
    @inbounds for ie in 1:5
        r[2*ie - 1] = g23h.pmra[ie]  - pmra_mod[ie]
        r[2*ie    ] = g23h.pmdec[ie] - pmdec_mod[ie]
    end

    # Cholesky-factor the (data-only) 10×10 covariance once. Σ has no
    # orbit dependence so its log-det is constant; we still add it for
    # log-Z compatibility with the rest of the joint likelihood.
    chol_Σ = cholesky(Symmetric(g23h.cov); check = false)
    issuccess(chol_Σ) ||
        throw(ErrorException("G23H covariance not positive-definite"))

    # Σ⁻¹ r (10-vec). The Cholesky factorization solves this in
    # one back-substitution.
    Σinv_r = chol_Σ \ r

    # X is the 10×2 selector for the system barycentric PM (μ_α*, μ_δ).
    # Each pair of rows in X is the 2×2 identity, so:
    #   v = Xᵀ Σinv_r    (2-vec)  = (Σ over odd indices, Σ over even)
    #   A = Xᵀ Σ⁻¹ X     (2×2)    requires the full Σ⁻¹.
    v1 = zero(T); v2 = zero(T)
    @inbounds for ie in 1:5
        v1 += Σinv_r[2*ie - 1]
        v2 += Σinv_r[2*ie    ]
    end

    # Build A = Xᵀ Σ⁻¹ X by summing the appropriate Σ⁻¹ blocks.
    # Σ⁻¹ = chol_Σ \ I, but we only need its 2×2 reduction. Solve
    # the 10×2 system Σ Y = X for Y, then A = Xᵀ Y.
    X_mat = zeros(T, 10, 2)
    @inbounds for ie in 1:5
        X_mat[2*ie - 1, 1] = one(T)
        X_mat[2*ie    , 2] = one(T)
    end
    Y = chol_Σ \ X_mat   # 10×2

    A11 = zero(T); A12 = zero(T); A22 = zero(T)
    @inbounds for ie in 1:5
        A11 += Y[2*ie - 1, 1]
        A12 += Y[2*ie - 1, 2]
        A22 += Y[2*ie    , 2]
    end
    # Note: A12 == Σ_{odd} Y[odd, 2] == Σ_{even} Y[even, 1] by
    # symmetry of (X^T Σ^-1 X). We use the odd-row form.

    rΣr = zero(T)
    @inbounds for j in 1:10
        rΣr += r[j] * Σinv_r[j]
    end

    # Solve 2×2 A·μ_b = v in closed form.
    det_A = A11 * A22 - A12 * A12
    if !(det_A > 0)
        return convert(T, -Inf)
    end
    μb1 = ( A22 * v1 - A12 * v2) / det_A
    μb2 = (-A12 * v1 + A11 * v2) / det_A
    χ²_min = rΣr - (v1 * μb1 + v2 * μb2)

    log_det_Σ = 2 * (log(chol_Σ.L[1,1])  + log(chol_Σ.L[2,2])  +
                     log(chol_Σ.L[3,3])  + log(chol_Σ.L[4,4])  +
                     log(chol_Σ.L[5,5])  + log(chol_Σ.L[6,6])  +
                     log(chol_Σ.L[7,7])  + log(chol_Σ.L[8,8])  +
                     log(chol_Σ.L[9,9])  + log(chol_Σ.L[10,10]))
    log_det_A = log(det_A)

    # 10 observations, 2 marginalized → 8 effective DOF.
    return -0.5 * (χ²_min + log_det_Σ + log_det_A) - 4 * log_2π
end

# ---------------------------------------------------------------------
# Orchestration: combined astrometry log-likelihood
# ---------------------------------------------------------------------

"""
    astrom_log_likelihood(theta, data) -> ll

Sum of relative-astrometry, HGCA, IAD, and GOST log-likelihoods, plus
the optional O'Neil 2019 observation-based prior contribution if
enabled in the parametrization config. Returns 0 if no astrometry
data is present.
"""
function astrom_log_likelihood(theta::Theta{T}, data) where {T<:Real}
    ll = relastrom_log_likelihood(theta, data) +
         hgca_log_likelihood(theta, data) +
         iad_log_likelihood(theta, data) +
         gost_log_likelihood(theta, data) +
         g23h_log_likelihood(theta, data)
    if theta.params.config.parametrization.obs_prior &&
       data.relastrom !== nothing
        ll += obs_based_log_prior(theta, data)
    end
    return ll
end

"""
    obs_based_log_prior(theta, data) -> Δlogp

O'Neil 2019 (AJ 158, 4) observation-based prior for relative astrometry.
Counters the bias of standard log-flat priors on (a, e) when the
observation arc covers a small fraction of the orbit — the standard
prior over-weights configurations where the companion is "currently"
at large projected separation versus orbits that spend most of their
time at large separation (Kepler's 2nd law).

For each relAST epoch the contribution is

    (1 / (σ_RA · σ_Dec)) · |2(e²-2)sin(E) + e(3M + sin(2E)) + 3M cos(E)| /
        (6 √(1-e²))

summed over epochs, multiplied by `((G·M_tot·P) / (2π⁴))^(1/3)` and
returned as `−2 log(jacobian)`. (orbitize!'s implementation, ported.)

The prior is per-companion; for multi-companion systems the term sums
contributions from each AS-bearing planet's relAST epochs (binding via
`planet_idx`).
"""
function obs_based_log_prior(theta::Theta{T}, data) where {T<:Real}
    relast = data.relastrom
    relast === nothing && return zero(T)
    n = n_relast(relast)
    n == 0 && return zero(T)

    # G in M_sun · AU³ · yr⁻²: the constant from Kepler's 3rd law
    # P[yr]² = a[AU]³ / M_total[M_sun] is exact when 4π² ≡ G in these
    # units. We therefore use 4π² for G·M_tot in solar/AU/yr.
    fourpi2 = oftype(zero(T), 4 * π^2)

    M_pri = astrom_M_pri(theta)
    t_ref = data.t_ref

    jac_total = zero(T)
    for k in planet_indices(theta)
        block = theta.params.layout.planet_blocks[k]
        has_AS(block) || continue
        # Per-planet astrometric coupling. A companion the RV has established
        # can still be astrometrically undetected; when the mask says so it
        # contributes NO reflex here, and its inc/Omega carry their priors.
        # Defaults true, so fixed-dim behaviour is unchanged.
        is_planet_as_active(theta, k) || continue
        # relAST is a RESOLVED-companion measurement — N/A for an unresolved
        # SB2 binary (its absolute-astrometry reflex goes through the
        # star_reflex_* paths instead). Exclude it defensively.
        is_sb2(block) && continue
        P_d   = planet_P(theta, k)
        e, ω  = planet_e_w(theta, k)
        M_sec = planet_M_sec(theta, k)
        M_tot = M_pri + M_sec
        t_anc = planet_time_anchor(theta, k)
        time_kind = theta.params.config.parametrization.time
        Tp = _t_anchor_to_tp(time_kind, t_anc, P_d, e, ω, t_ref)

        e_safe = min(max(e, zero(e)), oftype(e, 0.9999))
        sqrt_1me2 = sqrt(1 - e_safe * e_safe)
        # Prefactor: ((G M_tot P) / (2π^4))^(1/3) in AU·yr⁻¹·M_sun
        # → simplifies in solar/AU/yr units to (M_tot P_yr / (2π²))^(1/3)
        P_yr = P_d / oftype(P_d, 365.25)
        prefactor = cbrt(M_tot * P_yr / (fourpi2 / 2))

        contrib = zero(T)
        for j in 1:n
            relast.planet_idx[j] == k || continue
            t_obs = relast.t[j]
            σra   = relast.ra_err[j]
            σdec  = relast.dec_err[j]
            # Mean anomaly at t_obs (radians)
            M_obs = oftype(P_d, 2π) * (t_obs - Tp) / P_d
            E_obs = kepler_solve(M_obs, e_safe)
            # O'Neil 2019 Eq. 33 numerator
            num = abs(
                2 * (e_safe^2 - 2) * sin(E_obs) +
                e_safe * (3 * M_obs + sin(2 * E_obs)) +
                3 * M_obs * cos(E_obs)
            )
            contrib += (1 / (σra * σdec)) * num / (6 * sqrt_1me2)
        end
        jac_total += contrib * prefactor
    end

    # Guard against zero / degenerate jacobian
    j_safe = max(jac_total, oftype(jac_total, 1e-300))
    return -2 * log(j_safe)
end

# ---------------------------------------------------------------------
# Internal helpers — declared here because they require both projection
# and the planet-block layout. The block-side accessors `astrom_plx`,
# `active_n_p`, `has_AS`, `_planet_orbit` are defined where the planet
# block subtypes live (parameters.jl extension).
# ---------------------------------------------------------------------

# Forward declarations — implementations in parameters.jl + model.jl
# (after the astrometry block subtypes are defined). Kept here for
# grep-ability.
# function astrom_plx(theta)::T               # parallax in mas
# function has_AS(block)::Bool                # data-source membership
# function _planet_orbit(theta, k, M_pri, plx) -> (orbit, M_sec)
