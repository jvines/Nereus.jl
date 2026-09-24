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

# Per-imager jitter

`RelAstromData.inst` indexes `InstrumentConfig.as_names`, and each imager
contributes an additive jitter `s_m` (mas) through
`σ² → σ² + s_m²` on both axes. Two imagers of the same companion — GPI
and SPHERE, say — therefore carry separate error budgets, which is the
point: published relative-astrometry uncertainties are routinely
optimistic in instrument-specific ways, and one shared jitter lets the
better-calibrated instrument absorb the other's systematics.

Name no astrometric instruments (the default) and no jitter slot exists:
`as_jitter` returns exactly zero, the reported σ are used verbatim, and
the arithmetic is untouched — no `sqrt(σ² + 0²)` round-trip.

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
    # Per-imager jitter, hoisted out of the epoch loop. Empty vector when no
    # astrometric instruments are configured, which is the flag the loop
    # below tests — a zero-length vector means "reported errors, verbatim".
    n_as_slots = length(theta.params.layout.systemic.as_jitter)
    jit2 = if n_as_slots == 0
        T[]
    else
        T[(x -> x * x)(as_jitter(theta, m)) for m in 1:n_as_slots]
    end

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
        m    = relast.inst[j]
        if !isempty(jit2) && m <= length(jit2) && jit2[m] != 0
            # Jitter is an INDEPENDENT error added to each axis, i.e.
            # Σ → Σ_reported + s²·I. That leaves the off-diagonal alone, so
            # inflating the marginals while keeping ρ would quietly inflate
            # the covariance too — a correlated error the instrument never
            # reported. Rescale ρ to hold σ_ra·σ_dec·ρ fixed.
            σra0, σdec0 = relast.ra_err[j], relast.dec_err[j]
            σra  = sqrt(σra0^2  + jit2[m])
            σdec = sqrt(σdec0^2 + jit2[m])
            ρ    = relast.corr[j] * (σra0 * σdec0) / (σra * σdec)
        else
            σra  = relast.ra_err[j]
            σdec = relast.dec_err[j]
            ρ    = relast.corr[j]
        end
        quad, logdet = relastrom_chi2_one(σra, σdec, ρ, rra, rdec)
        ll += -0.5 * (quad + logdet) - log_2π
    end
    return ll
end

# ---------------------------------------------------------------------
# HGCA absolute astrometry (Mode A: instantaneous PM at 3 epochs)
# ---------------------------------------------------------------------

"""
    _hgca_model_pm(theta, hgca, data, M_pri, plx, t_ref) -> (pmra_mod, pmdec_mod)

The model proper motion at the three HGCA epochs: the summed stellar reflex
of every astrometrically active companion, with the Hipparcos–Gaia epoch as
the mean reflex velocity over the baseline and the Gaia epoch through GOST
(Mode B) when scan plans are supplied. The part of `hgca_log_likelihood`
that `plot_pm_residuals` must reproduce exactly, so it lives here once.
"""
function _hgca_model_pm(theta::Theta{T}, hgca, data, M_pri, plx, t_ref) where {T<:Real}
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
    return pmra_mod, pmdec_mod
end

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

    pmra_mod, pmdec_mod = _hgca_model_pm(theta, hgca, data, M_pri, plx, t_ref)

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


# ---------------------------------------------------------------------
# Multi-instrument along-scan design
# ---------------------------------------------------------------------
#
# One `IADData` may carry transits from several intermediate-astrometry
# missions (Hipparcos IAD and Gaia DR4 epoch astrometry both live in this
# container). The marginalised nuisance vector is then
#
#     q = (Δα₀¹, Δδ₀¹, μα*, μδ, Δα₀², Δδ₀², …)         n_q = 2 + 2·n_inst
#
# — parallax and BOTH proper-motion components SHARED across instruments,
# only the along-scan zero point split per instrument.
#
# That split is the whole design, and it is not the obvious one. Giving
# each mission its own free five-vector destroys the science: the
# Hipparcos↔Gaia signal IS the proper-motion difference across the ~25 yr
# gap (this is what HGCA exploits), and marginalising a free μ per mission
# under a flat prior integrates that difference away. You would be left
# with two short-baseline fits and no long lever.
#
# Sharing μ is safe even though the two missions measure `pm_factor` from
# different epochs (Hipparcos: years from 1991.25; Gaia DR4: from J2017.5).
# Shifting an instrument's time origin by δ changes its model by
# μα*·sinψ·δ + μδ·cosψ·δ, which lies exactly in that instrument's own
# (sinψ, cosψ) zero-point columns — so the per-instrument zero point
# absorbs the origin difference, and the frame offset between the two
# reference positions, in one go.
#
# Column ORDER is deliberate: instrument 1's zero point occupies columns
# 1-2 and the shared block columns 3-5, so a single-instrument container
# reproduces the previous hand-unrolled 5×5 layout
# (sinψ, cosψ, plx_factor, sinψ·pm_factor, cosψ·pm_factor) column for
# column — and therefore bit for bit. χ²_min and log det A are invariant
# under column permutation in exact arithmetic, but the Cholesky is not
# invariant in the last bits, and this package ships tracked reproduction
# artifacts computed with the single-mission path.

"""
    _iad_n_q(n_inst) -> Int

Size of the marginalised nuisance vector: two shared parameters
(μα*, μδ) plus an along-scan zero point (Δα₀, Δδ₀) per instrument.

The parallax is NOT in here. It used to be — a free ϖ marginalised under a
flat prior — while `astrom_plx(theta)` supplied a SEPARATE sampled parallax
that scaled the orbit. Two parameters for one physical quantity, fully
decoupled: measured on HD 114762 (558 DR4 transits), the conditional ϖ̂ from
this solve sat at 25.72-25.77 mas (Gaia DR3: 25.36 ± 0.30) and moved by
0.05 mas while the sampled plx was swept 22 → 36 mas. The abscissae pin the
parallax, Nereus computed that on every likelihood call, discarded it, and
left the parallax that sets the mass free to be dragged along the
a0 ∝ M_sec·ϖ degeneracy — which is why the published run lands 25σ off an
informative Normal(25.36, 0.30) prior. See `_iad_residuals!`, where the
sampled parallax now enters the model deterministically instead.
"""
@inline _iad_n_q(n_inst::Int) = 2 + 2 * n_inst

"""
    _iad_pos_cols(n_inst) -> Vector{Int}

First zero-point column of each instrument. Instrument 1 sits at columns
1-2, ahead of the shared proper-motion block at 3-4; every later
instrument follows the shared block.

Was `2m + 2` when the shared block was three wide (ϖ, μα*, μδ). The
parallax column is gone — see `_iad_n_q` — so the shared block is two wide
and later instruments start one column earlier.
"""
_iad_pos_cols(n_inst::Int) = Int[m == 1 ? 1 : 2m + 1 for m in 1:n_inst]

# Accumulate into the upper triangle only — `Symmetric(A)` reads that half.
@inline function _iad_acc!(A::AbstractMatrix, i::Int, j::Int, val)
    @inbounds i <= j ? (A[i, j] += val) : (A[j, i] += val)
    return nothing
end

"""
    _iad_ref_offset(q, s, c, plxf, pmf) -> Real

The catalogue solution `q = (Δα₀, Δδ₀, ϖ, μα*, μδ)` projected onto this
transit's scan direction — what must be ADDED BACK to an O−C residual to
recover the full along-scan abscissa.

Hipparcos `RES` is a residual against the catalogue solution on header
line 11 of the van Leeuwen record; Gaia's `centroid_pos_al` is the full
abscissa. Sharing ϖ and μ between them is only meaningful once both mean
the same thing: unreconstructed, Hipparcos' ϖ column means "correction to
the catalogue parallax" and Gaia's means "the parallax".

With ONE instrument this addition lands entirely inside the column span of
the design matrix, which the flat-prior marginalisation projects out — so
it changes nothing, which is why it was never needed before.
"""
@inline function _iad_ref_offset(q::NTuple{5, Float64}, s, c, plxf, pmf)
    return q[1] * s + q[2] * c + q[3] * plxf + q[4] * (s * pmf) + q[5] * (c * pmf)
end

"""
    _iad_ref_common(iad) -> NTuple{5, Float64}

The catalogue five-vector every instrument's abscissae are expressed
RELATIVE TO once they have been put on a common footing.

Instruments disagree about what their abscissae mean: Hipparcos stores
O−C residuals against its catalogue solution, Gaia stores full abscissae.
Putting them together means picking one origin and shifting everything to
it. The naive choice — "full abscissae for everyone" — is algebraically
fine and numerically bad: a real Hipparcos row is ~2 mas of residual on
top of ~200 mas of catalogue sky path (μ ≈ 180 mas/yr over a 1.2 yr
baseline), so rWr inflates by ~10⁴ while χ²_min stays put, and χ²_min is
computed as `rWr − vᵀA⁻¹v`, a difference of two nearly equal numbers.
For a high-proper-motion star (Barnard's, 10.4 arcsec/yr) that cancels
most of the available precision and puts a staircase in the AD gradient.

So the common origin is a CATALOGUE solution, not zero: the first
instrument that stores residuals donates its own. Every residual
instrument then keeps its small numbers, and each absolute instrument has
that solution subtracted — which is just as small, because the two
catalogues are measuring the same star and agree to order a mas. The
marginalised shared block then means "correction to this catalogue
solution" rather than "the parallax", which is exactly what the
single-mission code has always meant.
"""
function _iad_ref_common(iad)
    @inbounds for m in eachindex(iad.ref_params)
        iad.abscissa_kind[m] === :residual && any(!=(0.0), iad.ref_params[m]) &&
            return iad.ref_params[m]
    end
    return ntuple(_ -> 0.0, 5)
end

"""
    _iad_ref_offsets(iad) -> (offsets, needs)

Per-instrument five-vector to ADD to that instrument's stored abscissae to
express them against the common origin, and a flag for whether it is
non-zero at all.

For instrument `m` the stored value is `a = w − X·ref_m`, and we want
`a' = w − X·ref_common`, so `offsets[m] = ref_m − ref_common`.

With ONE instrument this is identically zero, so nothing is added and the
single-mission arithmetic is untouched down to the sign of zero. That is
not luck: one instrument is always its own common origin.
"""
function _iad_ref_offsets(iad)
    ref_c = _iad_ref_common(iad)
    offs = [ntuple(k -> iad.ref_params[m][k] - ref_c[k], 5)
            for m in eachindex(iad.ref_params)]
    return offs, Bool[any(!=(0.0), o) for o in offs]
end

# The no-astrometric-planet case, hoisted to a const so that path neither
# allocates nor hands `_iad_residuals!` an abstract element type.
const _NO_ORBITS = PlanetOrbits.AbstractOrbit[]

"""
    _iad_active_orbits(theta, M_pri, plx, t_ref) -> (active_ks, orbs, M_secs)

Build `(orbit, M_sec)` once per astrometrically-active companion.

Without this the `for transit / for planet / build orbit` pattern rebuilds
the same orbit `n_transits × n_planets` times per likelihood call — for
Hipparcos that is 30-150 transits times a few planets, all identical.
"""
function _iad_active_orbits(theta::Theta{T}, M_pri, plx, t_ref) where {T<:Real}
    active_ks = Int[]
    M_secs    = T[]
    # NOT `Any[]`. `_iad_residuals!` reads `orbs[ki]` once per abscissa, so an
    # abstract element type puts a dynamic dispatch and a boxed return on the
    # innermost loop of the whole fit -- 824 of them per likelihood call on a
    # Gaia DR4 source, measured at 151 ns and 214 B each: 216 us and 172 KiB per
    # call, against 92 us and 0.2 KiB with a concrete eltype. That 182 KiB/eval
    # also put 24% of a fit's wall time in GC. `_planet_orbit` is not inferrable
    # (the planet-block container is abstractly typed), so take the element type
    # from the first orbit at run time; `_iad_residuals!` and `gost_5param_fit`
    # are function barriers and specialise on whatever concrete vector they get.
    orbs      = nothing
    for k in planet_indices(theta)
        block = theta.params.layout.planet_blocks[k]
        has_AS(block) || continue
        # Per-planet astrometric coupling. A companion the RV has established
        # can still be astrometrically undetected; when the mask says so it
        # contributes NO reflex here, and its inc/Omega carry their priors.
        # Defaults true, so fixed-dim behaviour is unchanged.
        is_planet_as_active(theta, k) || continue
        orb_k, M_sec_k = _planet_orbit(theta, k, M_pri, plx, t_ref)
        orbs === nothing && (orbs = Vector{typeof(orb_k)}())
        push!(active_ks, k)
        push!(orbs, orb_k)
        push!(M_secs, M_sec_k)
    end
    # Every branch of `_planet_orbit` (including the SB2 one) ends in the same
    # `build_orbit` call, so one theta never mixes orbit types and the typed
    # vector never has to widen.
    return active_ks, (orbs === nothing ? _NO_ORBITS : orbs), M_secs
end


"""
    _iad_residuals!(r, iad, orbs, M_secs, plx) -> r

`r_i = w_i − Δη_orbit_i`, where `w_i` is the FULL along-scan abscissa
(the stored value plus its instrument's catalogue solution, when the
stored value is an O−C residual) and `Δη_orbit_i` is the summed stellar
reflex of every astrometry-bearing companion projected onto scan `i`.

The reconstruction always uses the instrument's own stored `pm_factor`,
because that is the time origin the catalogue proper motion refers to —
independently of which time origin the design columns use. The two differ
by a constant in the instrument's zero-point span, which the
marginalisation absorbs.
"""
function _iad_residuals!(r::AbstractVector{T}, iad, orbs, M_secs, plx) where {T<:Real}
    # Every abscissa shares these orbits, so reduce each one to its
    # epoch-independent Thiele-Innes constants ONCE rather than re-deriving the
    # geometry inside `orbitsolve` at all 824 of them
    # (`_reflex_kernel`, src/astrometry/projection.jl).
    #
    # The fast/fallback choice is made HERE, for the whole vector, and never
    # per element. Returning `ReflexKernel` for one orbit and `ReflexFallback`
    # for another would make this a `Vector{Union{...}}`, and `_reflex_offset`
    # would go back to being a dynamic dispatch on the innermost loop --
    # measured, that costs more than the hoist saves (236 KiB/eval against
    # 12.5). Both branches below build a CONCRETE vector and hand it to the
    # same function barrier, which specialises on it.
    if all(_reflex_fast_applicable, orbs)
        ks = [_reflex_kernel(orbs[ki], M_secs[ki]) for ki in eachindex(orbs)]
        return _iad_residuals_kernels!(r, iad, ks, plx)
    else
        fb = [ReflexFallback(orbs[ki], M_secs[ki]) for ki in eachindex(orbs)]
        return _iad_residuals_kernels!(r, iad, fb, plx)
    end
end

"""
    _kepler_from_neighbour(e, M, E0, s0, c0) -> (E, sin E, cos E, ok)

Kepler solution at mean anomaly `M`, refined from a neighbouring epoch's already
solved `(E0, sin E0, cos E0)` instead of from scratch.

The abscissae of one field-of-view transit sit within 40 s of each other
(`grp_head`, src/astrometry/data.jl), across which the mean anomaly moves ~2e-6
rad for a 1183 d orbit — so Newton from the group's head converges in two steps,
and the sine and cosine come from angle-sum identities over those two tiny
corrections rather than two more libm calls.

`ok` is the point: the last thing computed is the actual Kepler residual
`E - e sin E - M`, so the caller finds out whether the refinement really landed
rather than assuming it. It does not for a short-period, high-eccentricity orbit
whose group spans an appreciable fraction of a radian near periastron, and the
caller then pays for a full solve. That makes this an optimisation with no
accuracy regime to reason about: it is either as good as Markley or it is not
used.
"""
@inline function _kepler_from_neighbour(e, δ, E0, s0, c0)
    # Everything is expressed as an INCREMENT on the head's solution. `δ` is the
    # mean-anomaly step across the group, formed from the time difference, and
    # the target anomaly is never materialised. That matters: `_markley_sc`
    # reduces its argument with `rem2pi`, so `E0` solves the head's REDUCED
    # anomaly, and differencing a raw sibling anomaly against it would carry
    # every 2π wrap into δ. On a prior draw with |M| ~ 1e6 that produced a
    # mirror-asymmetry of 2e13 in the node-flip test — caught there, not here.
    #
    # Since the head satisfies E0 - e sin E0 = M_head exactly, Kepler's residual
    # at the head for the sibling's anomaly is just -δ.
    d1 = δ / (1 - e * c0)
    s1, c1 = _rotate_by(s0, c0, d1)
    E1 = E0 + d1
    # f(E1) for the sibling = (E1 - e sin E1) - (E0 - e sin E0) - δ, a difference
    # of O(1) quantities with no large anomaly anywhere in it.
    f1 = (E1 - e * s1) - (E0 - e * s0) - δ
    d2 = -f1 / (1 - e * c1)
    s2, c2 = _rotate_by(s1, c1, d2)
    E2 = E1 + d2
    f2 = (E2 - e * s2) - (E0 - e * s0) - δ
    return (E2, s2, c2, abs(f2) <= 1e-12)
end

# (sin(x+d), cos(x+d)) from (sin x, cos x) for a SMALL d, by angle-sum with a
# three-term series. Error ~ d^6/720: exact to Float64 for the |d| < 1e-2 this
# is used at, and when d is larger the caller's residual check rejects the
# result anyway.
@inline function _rotate_by(s, c, d)
    d2 = d * d
    sd = d * (1 - d2 / 6 * (1 - d2 / 20))
    cd = 1 - d2 / 2 * (1 - d2 / 12)
    return (s * cd + c * sd, c * cd - s * sd)
end

# Grouped path: one Markley solve per epoch group instead of one per abscissa.
# Only for a SINGLE companion — the overwhelmingly common astrometric fit —
# because carrying a per-kernel head state for several would need a buffer, and
# the multi-companion case falls through to the generic loop below.
function _iad_residuals_kernels!(r::AbstractVector{T}, iad,
                                 kernels::AbstractVector{<:ReflexKernel},
                                 plx) where {T<:Real}
    length(kernels) == 1 || return _iad_residuals_generic!(r, iad, kernels, plx)
    k = kernels[1]
    ref_off, needs_ref = _iad_ref_offsets(iad)
    Δplx = plx - _iad_ref_common(iad)[3]
    head = 0
    t_head = zero(eltype(iad.t))
    E0 = s0 = c0 = zero(typeof(k.e))
    @inbounds for j in eachindex(r)
        h = iad.grp_head[j]
        local sE, cE
        if h != head
            head = h
            t_head = iad.t[j]
            E0, s0, c0 = _markley_sc(k.n_per_day * (t_head - k.tp), k.e)
            sE, cE = s0, c0
        else
            # The step across the group, straight from the time difference, so
            # it stays ~1e-6 rad instead of inheriting the anomaly's magnitude.
            δ = k.n_per_day * (iad.t[j] - t_head)
            _, sE, cE, ok = _kepler_from_neighbour(k.e, δ, E0, s0, c0)
            if !ok
                _, sE, cE = _markley_sc(k.n_per_day * (iad.t[j] - k.tp), k.e)
            end
        end
        X = cE - k.e
        Y = k.sqrt1me2 * sE
        Δη_mod = (k.sB * X + k.sG * Y) * iad.sinpsi[j] +
                 (k.sA * X + k.sF * Y) * iad.cospsi[j] +
                 Δplx * iad.parallax_factor[j]
        m = iad.inst[j]
        if needs_ref[m]
            r[j] = iad.abscissa[j] +
                   _iad_ref_offset(ref_off[m], iad.sinpsi[j], iad.cospsi[j],
                                   iad.parallax_factor[j], iad.pm_factor[j]) - Δη_mod
        else
            r[j] = iad.abscissa[j] - Δη_mod
        end
    end
    return r
end

_iad_residuals_kernels!(r::AbstractVector, iad, kernels, plx) =
    _iad_residuals_generic!(r, iad, kernels, plx)

"""
    _iad_residuals_generic!(r, iad, kernels, plx) -> r

The residual loop proper, one full Kepler solve per abscissa per companion.
Separate from `_iad_residuals!` so it is a function barrier: it specialises on
the concrete element type of `kernels`, which is what keeps `_reflex_offset` a
static call inside the innermost loop. Used for several companions, and for any
orbit the closed-form kernel declines (`ReflexFallback`).
"""
function _iad_residuals_generic!(r::AbstractVector{T}, iad, kernels, plx) where {T<:Real}
    ref_off, needs_ref = _iad_ref_offsets(iad)
    # The shared block means "correction to `ref_c`" (see `_iad_ref_common`),
    # so the parallax the model has to supply is the correction implied by the
    # SAMPLED parallax, not its absolute value. For a single absolute
    # instrument (Gaia epoch data) `ref_c` is all zeros and this is just
    # `plx · plx_factor`.
    Δplx = plx - _iad_ref_common(iad)[3]
    @inbounds for j in eachindex(r)
        t_j = iad.t[j]
        # `iad.sinpsi`/`iad.cospsi` instead of `along_scan_projection(.., ψ_j)`
        # and a second `sincos` on the residual branch below: ψ is data, and
        # this loop runs 824 times per evaluation and 3.3M evaluations per fit.
        # Same values, so the residuals are bit-identical (max|Δ| = 0 measured).
        s = iad.sinpsi[j]
        c = iad.cospsi[j]
        Δη_mod = zero(T)
        for ki in eachindex(kernels)
            Δra, Δdec = _reflex_offset(kernels[ki], t_j)
            Δη_mod += Δra * s + Δdec * c
        end
        # Parallax is a sampled parameter, not a marginalised nuisance, so its
        # along-scan term belongs in the model alongside the orbit reflex.
        Δη_mod += Δplx * iad.parallax_factor[j]
        m = iad.inst[j]
        if needs_ref[m]
            w_j = iad.abscissa[j] + _iad_ref_offset(ref_off[m], s, c,
                                                    iad.parallax_factor[j],
                                                    iad.pm_factor[j])
            r[j] = w_j - Δη_mod
        else
            r[j] = iad.abscissa[j] - Δη_mod
        end
    end
    return r
end

"""
    _iad_normal_equations!(A, v, iad, r, pm_fac, pos_col) -> (rWr, Σlogσ)

Accumulate the weighted normal equations `A = XᵀWX`, `v = XᵀWr` for the
multi-instrument design, plus `rᵀWr` and `Σ log σ`.

Each transit contributes exactly four non-zero design entries: its own
instrument's two zero-point columns `(sinψ, cosψ)` and the two shared
columns `(sinψ·pm_factor, cosψ·pm_factor)`. Every other instrument's
columns are zero for this row and are simply not touched.

There is no parallax column: the sampled parallax is a known quantity and
its term `plx · plx_factor` is subtracted in `_iad_residuals!` instead of
being marginalised away here. See `_iad_n_q`.

`pm_fac` is passed in rather than read off `iad` because the joint
Hipparcos+Gaia path re-centres it on each instrument's own mean epoch.
"""
function _iad_normal_equations!(A::AbstractMatrix{T}, v::AbstractVector{T},
                                iad, r::AbstractVector{T},
                                pm_fac::AbstractVector{<:Real},
                                pos_col::Vector{Int}) where {T<:Real}
    rWr = zero(T)
    # ψ, σ and therefore Σ log σ are DATA: cached on the IADData at construction
    # rather than rebuilt here on every one of a fit's millions of evaluations
    # (measured 11.8 us -> 5.1 us on 824 abscissae). The cached sum accumulates
    # in the same order this loop used, so it is the identical Float64.
    log_sigma_sum = convert(T, iad.log_sigma_sum)
    @inbounds for j in eachindex(r)
        s    = iad.sinpsi[j]
        c    = iad.cospsi[j]
        pmf  = pm_fac[j]
        p    = pos_col[iad.inst[j]]
        cols = (p, p + 1, 3, 4)
        xs   = (s, c, s * pmf, c * pmf)
        w  = iad.weight[j]
        rj = r[j]

        rWr += w * rj * rj

        for a in 1:4
            wa = w * xs[a]
            v[cols[a]] += wa * rj
            for b in a:4
                _iad_acc!(A, cols[a], cols[b], wa * xs[b])
            end
        end
    end
    return rWr, log_sigma_sum
end

"""
    _iad_marginalised_residuals!(out, iad, r, q_opt, pm_fac, pos_col) -> out

`r` minus the fitted catalogue solution, transit by transit -- what the
marginalisation actually leaves behind, for diagnostics that need the
residuals themselves rather than the χ².

Lives HERE, beside `_iad_normal_equations!`, because it must use that
function's design row and nothing else: `cols = (p, p+1, 3, 4)` with
`xs = (s, c, s·pmf, c·pmf)`. A second hand-written copy in the plotting
code is exactly what went wrong -- it still carried the pre-removal 5-wide
catalogue layout `(Δα₀, Δδ₀, ϖ, μα*, μδ)`, so it multiplied μα* by the
parallax factor, shifted both proper-motion terms one column, and read
`q_opt[5]`, which does not exist for a single instrument (`_iad_n_q(1)` is
4). Under `@inbounds` that read past the end of the vector instead of
throwing, and `iad_residuals.png` came out empty with χ²/N = NaN.
"""
function _iad_marginalised_residuals!(out::AbstractVector, iad, r::AbstractVector,
                                      q_opt::AbstractVector, pm_fac, pos_col)
    @inbounds for j in eachindex(r)
        s    = iad.sinpsi[j]     # cached; see `_iad_normal_equations!`
        c    = iad.cospsi[j]
        pmf  = pm_fac[j]
        p    = pos_col[iad.inst[j]]
        out[j] = r[j] - (q_opt[p] * s + q_opt[p + 1] * c +
                         q_opt[3] * s * pmf + q_opt[4] * c * pmf)
    end
    return out
end

"""
    _iad_solve(A, v, rWr, n_q) -> (χ²_min, log_det_A) or nothing

Cholesky-solve the marginalisation and return `χ²_min = rᵀWr − vᵀA⁻¹v`
together with `log det A`. Returns `nothing` when `A` is not
positive-definite — a rank-deficient design, which the callers handle by
falling back to an unmarginalised Gaussian.

The accumulation order is written out rather than delegated to `dot` /
`sum`: both may reassociate, and the single-instrument path has to land on
the same bits as the hand-unrolled code it replaces.
"""
function _iad_solve(A::AbstractMatrix{T}, v::AbstractVector{T}, rWr::T, n_q::Int) where {T<:Real}
    chol = cholesky(Symmetric(A); check = false)
    issuccess(chol) || return nothing
    q_opt = chol \ v
    vq = v[1] * q_opt[1]
    @inbounds for i in 2:n_q
        vq += v[i] * q_opt[i]
    end
    ld = log(chol.L[1, 1])
    @inbounds for i in 2:n_q
        ld += log(chol.L[i, i])
    end
    return rWr - vq, 2 * ld
end

"""
    iad_log_likelihood(theta, data) -> ll

Marginalized log-likelihood of intermediate astrometry: per-transit
Gaussian on the along-scan abscissa after subtracting the orbit's
along-scan reflex AND analytically marginalizing over the catalog
astrometric solution under flat priors. This is the algorithm of htof
(Brandt 2018, ApJS 239 31) and van Leeuwen 2007 §17.3 — re-fit the
astrometric solution at every likelihood call given the current orbit,
then evaluate the residuals.

Handles one or several instruments. `IADData` carries a per-transit
`inst` index (Hipparcos IAD and Gaia DR4 epoch astrometry share this
container and this likelihood), and the nuisance vector is

    q = (Δα₀¹, Δδ₀¹, μα*, μδ, Δα₀², Δδ₀², …)         n_q = 2 + 2·n_inst

— the parallax and BOTH proper-motion components shared, only the
along-scan zero point per instrument. Sharing μ is the entire reason to
combine two missions: the Hipparcos↔Gaia signal is the proper-motion
difference across the ~25 yr gap. See the block comment above
`_iad_n_q` for why a free five-vector per mission would throw that away.

# Algorithm

For each transit `i = 1..n` belonging to instrument `m`:

  predicted_along_scan_i = Δη_orbit_i +
                           Δα0ᵐ · sin ψ_i + Δδ0ᵐ · cos ψ_i +
                           ϖ_sampled · plx_factor_i +
                           μα*  · sin ψ_i · pm_factor_i +
                           μδ   · cos ψ_i · pm_factor_i

where `Δη_orbit_i = Σ_k along_scan_projection(reflex_k(t_i), ψ_i)`
sums the stellar reflex from every astrometry-bearing companion.

The design row has four non-zero entries — instrument `m`'s two
zero-point columns `(sin ψ_i, cos ψ_i)` and the two shared columns
`(sin ψ_i · pm_factor_i, cos ψ_i · pm_factor_i)` — and zeros in every
other instrument's columns. The parallax is NOT marginalised: `ϖ` above
is the sampled `astrom_plx(theta)`, subtracted into the model. The orbit-corrected data is
`r_i = w_i − Δη_orbit_i`, with `w_i` the FULL abscissa: for an instrument
whose abscissae are O−C residuals (Hipparcos `RES`) the catalogue
solution in `IADData.ref_params` is added back first, so that a shared
ϖ and μ mean the same thing for every instrument. Under flat priors on
`q` the marginalized χ² is

    χ²_min(orbit) = rᵀ W r − vᵀ A⁻¹ v

with `W = diag(1/σ_i²)`, `A = Xᵀ W X` (n_q × n_q), `v = Xᵀ W r`.
The Gaussian normalization picks up `−½ log det A` from the
marginalization Jacobian and `−Σ log σ_i` from the per-transit
Gaussian.

Note that `A` and `Σ log σ_i` depend only on the IAD design (ψ, σ,
plx_factor, pm_factor), not on the orbit. They contribute additive
constants to the log-likelihood — fine for orbit-fitting and for log-Z
estimates, modulo a transit-independent offset shared with any other
model.

Requires `n ≥ n_q` transits (otherwise the correction is
underdetermined and the likelihood is undefined) — 4 for one
instrument, 6 for two. Returns 0 if `data.iad` is `nothing` or empty.

With one instrument this reduces to the previous hand-unrolled 5×5
marginalization bit for bit: same column order, same accumulation order,
same Cholesky. `test/astrometry/test_iad_multi_instrument.jl` pins that
with `===` on the returned `Float64`.
"""
function iad_log_likelihood(theta::Theta{T}, data) where {T<:Real}
    iad = data.iad
    iad === nothing && return zero(T)
    n = n_iad(iad)
    n == 0 && return zero(T)
    n_inst = n_iad_inst(iad)
    n_q    = _iad_n_q(n_inst)
    # Need MORE than n_q transits: at n == n_q the design is exactly
    # determined, so χ²_min ≡ 0 and every orbit fits perfectly — a
    # silent-wrong result, not a likelihood. Real Hipparcos sources have
    # 30–150 and Gaia epoch data hundreds; this guard catches synthetic
    # under-determined cases cleanly.
    n > n_q || return zero(T)

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

    active_ks, orbs, M_secs = _iad_active_orbits(theta, M_pri, plx, t_ref)

    # Stage 1 — full abscissa minus orbit reflex.
    r = Vector{T}(undef, n)
    _iad_residuals!(r, iad, orbs, M_secs, plx)

    # Stage 2 — accumulate (rᵀ W r), (Xᵀ W r), (Xᵀ W X) over the
    # multi-instrument design.
    A = zeros(T, n_q, n_q)
    v = zeros(T, n_q)
    rWr, log_sigma_sum =
        _iad_normal_equations!(A, v, iad, r, iad.pm_factor, _iad_pos_cols(n_inst))

    solved = _iad_solve(A, v, rWr, n_q)
    if solved === nothing
        # Design rank-deficient — typically because `parallax_factor`
        # and/or `pm_factor` were not supplied (zero-vector defaults).
        # Fall back to a non-marginalized Gaussian: just penalize the
        # orbit-residual `r` against the per-transit error. This loses
        # the catalog-correction degree-of-freedom absorption but keeps
        # the likelihood meaningful for forward-modeling-only test
        # inputs. Nothing here depends on n_q: no parameter was absorbed.
        #
        # This branch is a DIFFERENT model (q ≡ 0), not a degraded version of
        # the same one, and it is discontinuous with the marginalised branch.
        # With one instrument that is a tolerable answer for a forward-model
        # test input. With several it usually means one instrument has no
        # usable scan geometry and is quietly dragging the whole joint fit
        # into a model nobody asked for, so say so.
        n_inst > 1 && @warn("iad_log_likelihood: the $(n_q)-parameter design " *
            "over $(n_inst) instruments is rank-deficient, so the catalogue " *
            "solution is NOT being marginalised — the likelihood has fallen " *
            "back to holding it at zero. Usually one instrument has no " *
            "parallax_factor/pm_factor, or its scan angles are degenerate. " *
            "The returned value is not comparable with a marginalised one.",
            maxlog = 1)
        return -0.5 * rWr - log_sigma_sum - 0.5 * n * log_2π
    end
    χ²_min, log_det_A = solved

    # Marginalized log-likelihood:
    #   −½ χ²_min − Σ log σ_i − ½ log det A − (n − n_q)/2 · log(2π)
    # The (n − n_q) factor reflects the absorbed catalog DOF — 4 for one
    # instrument, 2 + 2·n_inst in general; A's log-det is the
    # marginalization Jacobian (analogous to the HGCA term).
    return -0.5 * χ²_min - log_sigma_sum - 0.5 * log_det_A -
           0.5 * (n - n_q) * log_2π
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
    n_inst  = n_iad_inst(iad)
    n_q     = _iad_n_q(n_inst)
    pos_col = _iad_pos_cols(n_inst)
    M_pri   = astrom_M_pri(theta)
    plx     = astrom_plx(theta)
    t_ref   = data.t_ref
    log_2π  = oftype(plx, log(2π))

    # Each instrument's design is re-centred on its OWN mean epoch, which
    # keeps its proper-motion columns orthogonal to its position columns.
    # Re-centring shifts the model by a constant times (sinψ, cosψ), i.e.
    # by something inside that instrument's own zero-point span, so it
    # cannot move the marginalised answer — it only conditions the solve.
    #
    # For a single instrument this is `sum(iad.t)/n_h`, spelled exactly as
    # the pre-generalisation code spelled it: `sum` may reassociate under
    # `@simd`, so a hand-rolled loop is not guaranteed to land on the same
    # last bit, and this path must be unchanged in output.
    t_ref_inst = Vector{Float64}(undef, n_inst)
    if n_inst == 1
        t_ref_inst[1] = sum(iad.t) / n_h
    else
        cnt = zeros(Int, n_inst)
        acc = zeros(Float64, n_inst)
        @inbounds for j in 1:n_h
            m = iad.inst[j]
            acc[m] += iad.t[j]
            cnt[m] += 1
        end
        @inbounds for m in 1:n_inst
            t_ref_inst[m] = acc[m] / cnt[m]
        end
    end
    pm_fac = Vector{Float64}(undef, n_h)
    @inbounds for j in 1:n_h
        pm_fac[j] = (iad.t[j] - t_ref_inst[iad.inst[j]]) / 365.25
    end

    # PM transport: instrument 1's reference epoch → Gaia's (gaia.t_ref).
    # Instrument 1 is the frame the published Gaia five-parameter solution
    # is tied to here; with two intermediate-astrometry instruments the
    # published solution also duplicates the second one's data, which
    # `Data` warns about at construction.
    Δt_yr = (gaia.t_ref - t_ref_inst[1]) / 365.25

    active_ks, orbs, M_secs = _iad_active_orbits(theta, M_pri, plx, t_ref)

    # --- Stage 1: per-transit residuals + design accumulation
    r = Vector{T}(undef, n_h)
    _iad_residuals!(r, iad, orbs, M_secs, plx)
    A = zeros(T, n_q, n_q)
    v = zeros(T, n_q)
    rWr, log_sigma_sum = _iad_normal_equations!(A, v, iad, r, pm_fac, pos_col)

    # --- Stage 2: orbit-induced Gaia 5-param shift, summed across planets.
    # gost_5param_fit returns Δq at GOST mean epoch; we re-evaluate it
    # at the GAIA published reference epoch by passing gaia.t_ref.
    Δq_orb = zeros(T, 5)
    for ki in eachindex(active_ks)
        Δq_k = gost_5param_fit(orbs[ki], gost, M_secs[ki]; t_ref = gaia.t_ref)
        @. Δq_orb += Δq_k
    end

    # --- Stage 3: Gaia constraint contribution
    #
    # Which rows of the published solution are usable depends on whether the
    # caller supplied a position.
    #
    # `GaiaDR3Data` documents α₀ = δ₀ = 0 as "no published position-shift
    # constraint, only PM + parallax", and `load_gaia_dr3` produces exactly
    # that by default (its fiducial IS the Gaia position, so the offsets come
    # out identically zero). Feeding those two zeros through as MEASUREMENTS
    # — values of zero carrying the published sub-mas σ — is not a null
    # constraint, it is an extremely tight one: it asserts the star sits at
    # the Hipparcos frame origin at the Gaia epoch. By then the star has
    # moved μ·Δt; for a 180 mas/yr star over 25 yr that is ~4500 mas against
    # a ~0.3 mas error, and the fit either contorts the orbit to cover it or
    # returns a χ² of order 10⁷. The two catalogues do not share a
    # tangent-plane origin and nothing in the data says how far apart they
    # are.
    #
    # So when the positions are zero, use the (ϖ, μα*, μδ) MARGINAL — the
    # 3×3 sub-block of the COVARIANCE, which marginalises over the position
    # rather than conditioning on it. Those three are epoch-independent in a
    # linear sky model, so no transport is needed either. A caller who has
    # genuinely put both catalogues in one frame passes non-zero offsets and
    # gets the full 5×5 constraint, unchanged.
    use_pos = !(gaia.params[1] == 0 && gaia.params[2] == 0)
    m_g     = use_pos ? 5 : 3
    rows_g  = use_pos ? (1:5) : (3:5)

    # The IAD rows are expressed against `ref_c` (see `_iad_ref_common`), so
    # the shared block means "correction to that catalogue solution". The
    # published Gaia vector is absolute, so it has to be shifted onto the
    # same origin or the two blocks would be constraining different
    # quantities — a Hipparcos block pulling ϖ toward 0 (its residuals say
    # the catalogue was right) while Gaia pulls it toward 30 mas. That
    # mismatch is what made this path unusable with real data.
    ref_c = _iad_ref_common(iad)
    # The parallax and proper-motion rows shift by ref_c directly. The POSITION
    # rows do not: q's position is where the star is NOW relative to the
    # reference solution, so what must come off the measurement is the sky path
    # ref_c itself predicts at the Gaia epoch — its position PLUS its proper
    # motion times the full baseline from the catalogue epoch. Subtracting the
    # untransported ref_c leaves μ_cat·Δt of error, ~4500 mas for a 180 mas/yr
    # star against a ~0.3 mas position error.
    #
    # The baseline runs from where the instrument's stored pm_factor is zero
    # (the catalogue epoch), which is δ₁ years before the re-centring epoch
    # Δt_yr is measured from. δ₁ is the mean stored pm_factor of instrument 1.
    δ₁ = let acc = 0.0, cnt = 0
        @inbounds for j in 1:n_h
            iad.inst[j] == 1 || continue
            acc += iad.pm_factor[j]; cnt += 1
        end
        cnt == 0 ? 0.0 : acc / cnt
    end
    Δt_cat = Δt_yr + δ₁
    ref_shift = (ref_c[1] + ref_c[4] * Δt_cat, ref_c[2] + ref_c[5] * Δt_cat,
                 ref_c[3], ref_c[4], ref_c[5])
    # The parallax is a SAMPLED parameter, not a marginalised one, so the
    # published Gaia parallax constrains it directly: its model value moves to
    # the residual side instead of getting a column in `P`. `ref_shift[3]`
    # is `ref_c[3]`, so row 3 reduces to `ϖ_gaia − Δq_orb[3] − plx`. Keeping
    # the full `Σ_g` means the ϖ-μ cross-covariance is still honoured.
    Δplx_model = plx - ref_c[3]
    y_minus_Δq = T[gaia.params[k] - ref_shift[k] - Δq_orb[k] -
                   (k == 3 ? Δplx_model : zero(T)) for k in rows_g]
    Σ_g = use_pos ? gaia.cov : gaia.cov[3:5, 3:5]

    # Cholesky-factor Σ_g (data-only); use it to compute Σ_g⁻¹ y' and
    # Pᵀ Σ_g⁻¹ P for the joint A.
    chol_Σg = cholesky(Symmetric(Σ_g); check = false)
    if !issuccess(chol_Σg)
        # Σ_g not positive-definite — should be caught at construction
        # time, but guard defensively.
        return convert(T, -Inf)
    end
    Σg_inv_y = chol_Σg \ y_minus_Δq
    log_det_Σg = let ld = log(chol_Σg.L[1,1])
        @inbounds for i in 2:m_g
            ld += log(chol_Σg.L[i, i])
        end
        2 * ld
    end

    # P maps the marginalised nuisance vector onto the published Gaia
    # vector: the parallax and proper-motion rows read the shared block, and
    # — when positions are in play — the position rows read instrument 1's
    # zero point plus the Δt transport across the epoch gap. It is m_g×n_q;
    # with one instrument and a supplied position that is exactly the 5×5
    # transport matrix this path used before (identity plus P[1,4] =
    # P[2,5] = Δt).
    # The parallax row of the published vector reads NO column — the sampled
    # parallax already came off in `y_minus_Δq`. The shared proper-motion
    # block sits at columns 3-4 now that ϖ is gone (was 4-5).
    P = zeros(T, m_g, n_q)
    @inbounds if use_pos
        P[1, pos_col[1]]     = one(T)
        P[2, pos_col[1] + 1] = one(T)
        # row 3 = parallax: no column.
        P[4, 3] = one(T)
        P[5, 4] = one(T)
        P[1, 3] = T(Δt_yr)
        P[2, 4] = T(Δt_yr)
    else
        # rows_g = 3:5 → (parallax, μα*, μδ); the parallax row stays empty.
        P[2, 3] = one(T)
        P[3, 4] = one(T)
    end

    # Pᵀ Σ_g⁻¹ y' (n_q-vec)
    PᵀΣinv_y = P' * Σg_inv_y
    @inbounds for i in 1:n_q
        v[i] += PᵀΣinv_y[i]
    end

    # Pᵀ Σ_g⁻¹ P (n_q×n_q) → solve Σ_g · Y = P for Y (m_g×n_q), then Pᵀ · Y.
    Y_g = chol_Σg \ P
    PᵀΣinvP = P' * Y_g
    @inbounds for i in 1:n_q, j in i:n_q
        A[i, j] += PᵀΣinvP[i, j]
    end

    # Gaia residual quadratic: (y_g − Δq_orb)ᵀ Σ_g⁻¹ (y_g − Δq_orb)
    rGaia = zero(T)
    @inbounds for k in 1:m_g
        rGaia += y_minus_Δq[k] * Σg_inv_y[k]
    end

    # --- Stage 4: solve and assemble
    solved = _iad_solve(A, v, rWr + rGaia, n_q)
    if solved === nothing
        # Joint design rank-deficient — extremely rare with both Hip and
        # Gaia constraints (each pins 5 DOF). Fall back to non-marginalized.
        return -0.5 * (rWr + rGaia) - log_sigma_sum -
               0.5 * log_det_Σg - 0.5 * (n_h + m_g) * log_2π
    end
    χ²_min, log_det_A = solved

    # ln L = −½ χ²_min − Σ log σ_h − ½ log det Σ_g − ½ log det A
    #        − ½ (n_h + m_g − n_q) log(2π)
    # n_h IAD rows + m_g Gaia rows − n_q marginalized. With one instrument
    # and a supplied position, n_q = m_g = 5 and the Gaia block cancels
    # exactly, leaving −½ n_h log(2π) as before.
    return -0.5 * χ²_min - log_sigma_sum - 0.5 * log_det_Σg -
           0.5 * log_det_A - 0.5 * (n_h + m_g - n_q) * log_2π
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

If you have real Gaia DR4 *epoch* astrometry, do not route it here:
GOST is only a predicted scan plan (when Gaia looked, at what angle),
with no measured abscissae to fit. Load the published per-CCD
along-scan measurements with [`read_gaia_epoch_votable`](@ref) and fit them
through [`iad_log_likelihood`](@ref), which takes DR4 epoch data
directly — the DR4 abscissa model is the Hipparcos-IAD model.
"""
function gost_log_likelihood(theta::Theta{T}, data) where {T<:Real}
    # Unconditionally zero: GOST carries no measurement to fit (the struct is
    # t, psi, parallax_factor, along_scan_pos -- a scan plan, not abscissae).
    # Its information enters through gost_window_avg_pm / gost_5param_fit on
    # the HGCA, DR3 and G23H paths instead. `Data(...)` warns when GOST is
    # supplied without one of those, so an inert GOST is never silent.
    #
    # This used to forward-model every scan epoch and multiply the result by
    # zero, "to keep the call path exercised". That cost one Kepler solve per
    # epoch per active planet on EVERY astrometric likelihood evaluation --
    # astrom_log_likelihood sums this term unconditionally -- and it was not
    # even safe: `zero(T) * Δη_sink` is NaN when the forward model returns NaN
    # or Inf, so the loop introduced exactly the failure it claimed to prevent.
    return zero(T)
end

# ---------------------------------------------------------------------
# G23H (Thompson+ 2026) — 5-PM joint absolute astrometry
# ---------------------------------------------------------------------

"""
    _g23h_model_pm(theta, g23h, data, M_pri, plx, t_ref) -> (pmra_mod, pmdec_mod)

The model proper motion at the five G23H epochs: the summed stellar reflex of
every astrometrically active companion, the DR3 epoch through GOST (Mode B)
when scan plans are supplied. The part of `g23h_log_likelihood` that
`plot_g23h_residuals` must reproduce exactly, so it lives here once.
"""
function _g23h_model_pm(theta::Theta{T}, g23h, data, M_pri, plx, t_ref) where {T<:Real}
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
    return pmra_mod, pmdec_mod
end

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

    pmra_mod, pmdec_mod = _g23h_model_pm(theta, g23h, data, M_pri, plx, t_ref)

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
        P_yr = P_d / oftype(P_d, KEPLER_YEAR_DAYS)   # G ≡ 4π² year, as above
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
