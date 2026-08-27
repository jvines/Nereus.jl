# Astrometry data containers.
#
# Mirrors the per-source-type pattern of `data.jl`: thin structs holding
# pre-staged numeric arrays + validation. No loaders, no catalog scrapers,
# no FITS readers (those live in a Python sidecar that hands Nereus plain
# numbers, per spec decision D5).
#
# Two data types are supported in Phase 1:
#
#  * `RelAstromData`  — relative astrometry of a resolved companion w.r.t.
#                       the host star. Tangent-plane offsets in mas with
#                       optional RA/Dec correlation. Per-companion: each
#                       entry binds to a planet index `k`.
#
#  * `HGCAData`       — Hipparcos-Gaia Catalog of Accelerations row.
#                       Three calibrated proper-motion vectors (Hip, HG
#                       long-baseline, Gaia) with per-axis 3×3 covariance.
#                       Single instance per system.
#
# Conventions:
#   - All angles in radians, separations in mas, RVs in m/s, masses in
#     solar masses. Epochs in MJD (consistent with the rest of Nereus).
#   - HGCA epochs are stored in MJD too, converted from the conventional
#     decimal-year tabulation at load time.
#   - `corr` (RA–Dec correlation) is a single number per epoch, in
#     [-1, 1]; 0 means uncorrelated (the common case).
#
# Phase 1 scope: HGCA EDR3 with per-epoch 2×2 RA/Dec covariance (the
# within-epoch RA-Dec correlation is tabulated as `pmra_pmdec_*` and
# can be substantial — HIP 3850 has ρ_Gaia = 0.66). The three HGCA
# epochs are treated as independent in time, matching orvara's
# likelihood (Brandt 2021, Appendix Eq. 1). Cross-epoch correlations
# along a single axis exist physically (HG_PM = (Gaia_pos − Hip_pos)
# / 25 yr shares position uncertainties with Hip and Gaia PMs) but are
# not in the HGCA tabulation; orvara ignores them too. Hipparcos IAD
# and Gaia DR3 epoch astrometry are Phase 2.

using LinearAlgebra: cholesky, Symmetric, isposdef

"""
    RelAstromData

Relative astrometry of a resolved companion: tangent-plane offsets
(ΔRA·cos δ, Δδ) of the companion relative to the host star, in mas.

Each row binds to a single planet `k` via `planet_idx`. A multi-companion
system contributes one or more rows per companion; the planet-idx field
disambiguates them in the likelihood.

# Fields
- `t::Vector{Float64}`           : observation times (MJD)
- `ra_off::Vector{Float64}`      : ΔRA × cos δ (mas)
- `dec_off::Vector{Float64}`     : Δδ (mas)
- `ra_err::Vector{Float64}`      : 1σ on ra_off (mas)
- `dec_err::Vector{Float64}`     : 1σ on dec_off (mas)
- `corr::Vector{Float64}`        : RA-Dec correlation per epoch ∈ [-1, 1]
- `planet_idx::Vector{Int}`      : planet index per epoch (1-based)

# Invariants
- All vectors have matching length.
- `ra_err`, `dec_err` strictly positive.
- `|corr| ≤ 1`.
- `planet_idx` strictly positive.
"""
struct RelAstromData
    t::Vector{Float64}
    ra_off::Vector{Float64}
    dec_off::Vector{Float64}
    ra_err::Vector{Float64}
    dec_err::Vector{Float64}
    corr::Vector{Float64}
    planet_idx::Vector{Int}
end

"""
    RelAstromData(; t, ra_off, dec_off, ra_err, dec_err, corr=nothing, planet_idx=nothing)

Keyword constructor. Defaults `corr` to all zeros and `planet_idx` to all
ones (single-companion case).
"""
function RelAstromData(;
    t::AbstractVector{<:Real},
    ra_off::AbstractVector{<:Real},
    dec_off::AbstractVector{<:Real},
    ra_err::AbstractVector{<:Real},
    dec_err::AbstractVector{<:Real},
    corr::Union{Nothing, AbstractVector{<:Real}} = nothing,
    planet_idx::Union{Nothing, AbstractVector{<:Integer}} = nothing,
)
    n = length(t)
    n > 0 || throw(ArgumentError("RelAstromData requires at least one epoch"))
    length(ra_off)  == n || throw(ArgumentError("ra_off length mismatch"))
    length(dec_off) == n || throw(ArgumentError("dec_off length mismatch"))
    length(ra_err)  == n || throw(ArgumentError("ra_err length mismatch"))
    length(dec_err) == n || throw(ArgumentError("dec_err length mismatch"))
    all(>(0), ra_err)  || throw(ArgumentError("ra_err entries must be > 0"))
    all(>(0), dec_err) || throw(ArgumentError("dec_err entries must be > 0"))

    corr_vec = corr === nothing ? zeros(Float64, n) : Vector{Float64}(corr)
    length(corr_vec) == n || throw(ArgumentError("corr length mismatch"))
    all(x -> -1 <= x <= 1, corr_vec) ||
        throw(ArgumentError("corr entries must be in [-1, 1]"))

    pidx_vec = planet_idx === nothing ? ones(Int, n) : Vector{Int}(planet_idx)
    length(pidx_vec) == n || throw(ArgumentError("planet_idx length mismatch"))
    all(>(0), pidx_vec) || throw(ArgumentError("planet_idx entries must be 1-based positive integers"))

    return RelAstromData(
        Vector{Float64}(t),
        Vector{Float64}(ra_off),
        Vector{Float64}(dec_off),
        Vector{Float64}(ra_err),
        Vector{Float64}(dec_err),
        corr_vec,
        pidx_vec,
    )
end

"""
    n_relast(d::RelAstromData) -> Int

Number of relative-astrometry epochs (separation/position-angle or
(dRA, dDec) measurements of a resolved companion).
"""
n_relast(d::RelAstromData) = length(d.t)

"""
    HGCAData

Hipparcos-Gaia Catalog of Accelerations row (Brandt 2021, EDR3 edition).
Three calibrated proper-motion vectors at three epochs:

  1. Hipparcos epoch (~1991.25)
  2. Long-baseline Hipparcos→Gaia (~2004.6 for EDR3)
  3. Gaia EDR3 epoch (~2016.0)

Each epoch carries a 2×2 within-epoch RA-Dec covariance matrix
(σ²_RA, σ²_Dec, σ_RA·σ_Dec·ρ_RA-Dec). The within-epoch correlations
ρ are tabulated in HGCA as `pmra_pmdec_hip`/`pmra_pmdec_hg`/
`pmra_pmdec_gaia` and can be substantial (HIP 3850 has ρ_Gaia = 0.66).
The three epochs are treated as independent in time — orvara's
likelihood does the same (Brandt 2021, Appendix Eq. 1).

# Fields
- `epochs::NTuple{3, Float64}`     : (Hip, HG, Gaia) MJD
- `pmra::NTuple{3, Float64}`       : (Hip, HG, Gaia) μ_α* (mas/yr)
- `pmdec::NTuple{3, Float64}`      : (Hip, HG, Gaia) μ_δ (mas/yr)
- `cov_ep::NTuple{3, Matrix{Float64}}` : 3 × (2×2) within-epoch covariance.
                                          Layout per matrix: [σ²_RA   σ_RA·σ_Dec·ρ;
                                                              σ_RA·σ_Dec·ρ   σ²_Dec]
- `plx::Float64`                   : Gaia parallax (mas)
- `plx_err::Float64`               : 1σ parallax uncertainty (mas)
- `hip_id::Int`                    : Hipparcos catalog ID

# Invariants
- Each 2×2 covariance matrix is symmetric positive-definite.
- `plx_err > 0`, `plx > 0`.
"""
struct HGCAData
    epochs::NTuple{3, Float64}
    pmra::NTuple{3, Float64}
    pmdec::NTuple{3, Float64}
    cov_ep::NTuple{3, Matrix{Float64}}
    plx::Float64
    plx_err::Float64
    hip_id::Int
end

"""
    HGCAData(; epochs, pmra, pmdec, sigma_pmra, sigma_pmdec, corr_pmra_pmdec,
              plx, plx_err, hip_id)

Keyword constructor. Pass per-epoch 1σ errors and within-epoch
RA-Dec correlations; the constructor builds the three 2×2 matrices.

# Arguments
- `sigma_pmra::NTuple{3, Real}`     : (Hip, HG, Gaia) σ on pmra (mas/yr)
- `sigma_pmdec::NTuple{3, Real}`    : (Hip, HG, Gaia) σ on pmdec (mas/yr)
- `corr_pmra_pmdec::NTuple{3, Real}`: (Hip, HG, Gaia) ρ_RA-Dec ∈ [−1, 1]

Use `mjd_epochs(epochs_jyear)` to convert decimal-year epoch tuples to MJD.
"""
function HGCAData(;
    epochs::NTuple{3, <:Real},
    pmra::NTuple{3, <:Real},
    pmdec::NTuple{3, <:Real},
    sigma_pmra::NTuple{3, <:Real},
    sigma_pmdec::NTuple{3, <:Real},
    corr_pmra_pmdec::NTuple{3, <:Real} = (0.0, 0.0, 0.0),
    plx::Real,
    plx_err::Real,
    hip_id::Integer,
)
    plx > 0     || throw(ArgumentError("plx must be > 0"))
    plx_err > 0 || throw(ArgumentError("plx_err must be > 0"))
    cov_eps = ntuple(k -> begin
        σra  = Float64(sigma_pmra[k])
        σdec = Float64(sigma_pmdec[k])
        ρ    = Float64(corr_pmra_pmdec[k])
        σra > 0    || throw(ArgumentError("sigma_pmra[$k] must be > 0"))
        σdec > 0   || throw(ArgumentError("sigma_pmdec[$k] must be > 0"))
        -1 < ρ < 1 || throw(ArgumentError("corr_pmra_pmdec[$k] must be in (-1, 1)"))
        Matrix{Float64}([σra^2          ρ*σra*σdec;
                         ρ*σra*σdec     σdec^2     ])
    end, 3)
    return HGCAData(
        NTuple{3, Float64}(Float64.(epochs)),
        NTuple{3, Float64}(Float64.(pmra)),
        NTuple{3, Float64}(Float64.(pmdec)),
        cov_eps,
        Float64(plx),
        Float64(plx_err),
        Int(hip_id),
    )
end

"""
    IADData

Hipparcos Intermediate Astrometric Data — per-epoch along-scan
abscissa residuals for a single source, as redistributed by van
Leeuwen 2007 ("Hipparcos: The New Reduction") and re-tabulated by
Brandt 2018 / htof.

Each Hipparcos transit is a 1-D along-scan abscissa (the projection
of the source position onto the great-circle scan direction at that
epoch). The scan-direction angle ψ — the angle of the scan track
relative to local RA — varies from transit to transit because
Hipparcos's scanning law rotates the satellite continuously. **Per-
epoch ψ is therefore essential**: each measurement constrains only
the orbit's projection onto its own scan direction, and information
about the orthogonal direction comes only from the variety of ψ
sampled across the mission. A single transit cannot pin down a 2-D
position; the ensemble does.

For a stellar reflex (Δα*, Δδ) at epoch t_i, the modeled along-scan
displacement is

    Δη_model(t_i) = Δα*(t_i) sin ψ_i + Δδ(t_i) cos ψ_i

(Brandt 2018 Eq. 3, and equivalently htof's `parallactic_motion` and
`AlongScanMotion` decomposition).

# Fields
- `t::Vector{Float64}`         : transit epochs (MJD)
- `abscissa::Vector{Float64}`  : along-scan residual (mas)
- `abscissa_err::Vector{Float64}` : 1σ on abscissa (mas)
- `psi::Vector{Float64}`       : along-scan direction angle (rad);
                                  the angle of the great-circle scan
                                  direction relative to local RA at
                                  this transit
- `parallax_factor::Vector{Float64}` : Hipparcos-tabulated parallax
                                  factor at this transit (used to
                                  multiply Δϖ when fitting parallax;
                                  preserved for downstream consumers)
- `pm_factor::Vector{Float64}` : along-scan time baseline relative
                                  to the catalog reference epoch in
                                  Julian years (used to multiply Δμ
                                  when re-fitting proper motion in a
                                  full IAD likelihood)

# Invariants
- All vectors have matching length.
- `abscissa_err` strictly positive.
- `psi` is not range-checked (it's in the Hipparcos sign convention,
  which can wrap).
"""
struct IADData
    t::Vector{Float64}
    abscissa::Vector{Float64}
    abscissa_err::Vector{Float64}
    psi::Vector{Float64}
    parallax_factor::Vector{Float64}
    pm_factor::Vector{Float64}
end

"""
    IADData(; t, abscissa, abscissa_err, psi, parallax_factor=nothing, pm_factor=nothing)

Keyword constructor. `parallax_factor` and `pm_factor` default to
zeros when not supplied (forward-modeling-only use case).
"""
function IADData(;
    t::AbstractVector{<:Real},
    abscissa::AbstractVector{<:Real},
    abscissa_err::AbstractVector{<:Real},
    psi::AbstractVector{<:Real},
    parallax_factor::Union{Nothing, AbstractVector{<:Real}} = nothing,
    pm_factor::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n = length(t)
    n > 0 || throw(ArgumentError("IADData requires at least one transit"))
    length(abscissa)     == n || throw(ArgumentError("abscissa length mismatch"))
    length(abscissa_err) == n || throw(ArgumentError("abscissa_err length mismatch"))
    length(psi)          == n || throw(ArgumentError("psi length mismatch"))
    all(>(0), abscissa_err) ||
        throw(ArgumentError("abscissa_err entries must be > 0"))
    # Fail loudly on non-finite input. A NaN epoch (e.g. a NULL Gaia
    # `obs_time_bary_corr`) otherwise propagates silently through pm_factor
    # into a non-finite log-likelihood, which reads as a bad model rather
    # than as bad data.
    for (nm, v) in (("t", t), ("abscissa", abscissa), ("abscissa_err", abscissa_err),
                    ("psi", psi))
        all(isfinite, v) ||
            throw(ArgumentError("IADData: $nm contains non-finite entries " *
                                "($(count(!isfinite, v)) of $n)"))
    end

    plx_fac_vec = parallax_factor === nothing ? zeros(Float64, n) :
                  Vector{Float64}(parallax_factor)
    length(plx_fac_vec) == n ||
        throw(ArgumentError("parallax_factor length mismatch"))
    all(isfinite, plx_fac_vec) ||
        throw(ArgumentError("IADData: parallax_factor contains non-finite entries " *
                            "($(count(!isfinite, plx_fac_vec)) of $n)"))

    pm_fac_vec = pm_factor === nothing ? zeros(Float64, n) :
                 Vector{Float64}(pm_factor)
    length(pm_fac_vec) == n ||
        throw(ArgumentError("pm_factor length mismatch"))
    all(isfinite, pm_fac_vec) ||
        throw(ArgumentError("IADData: pm_factor contains non-finite entries " *
                            "($(count(!isfinite, pm_fac_vec)) of $n)"))

    return IADData(
        Vector{Float64}(t),
        Vector{Float64}(abscissa),
        Vector{Float64}(abscissa_err),
        Vector{Float64}(psi),
        plx_fac_vec,
        pm_fac_vec,
    )
end

"""
    n_iad(d::IADData) -> Int

Number of along-scan abscissa measurements in an intermediate
astrometric data set.

Covers Hipparcos IAD and Gaia DR4 epoch astrometry alike -- the DR4
per-CCD abscissa model is the Hipparcos-IAD model, so both load into
`IADData` and are counted here.
"""
n_iad(d::IADData) = length(d.t)

"""
    GOSTData

Gaia Generic Object Slope-angle Tracker (GOST) — the Gaia mission
scan plan for a chosen sky position. Each row predicts a Gaia
along-scan transit: when the source crosses the focal plane and at
what scan angle. Used to forward-model the orbit's projection onto
each transit's scan direction.

GOST does not contain a measurement; it is a deterministic prediction
of the spacecraft attitude. To form a likelihood you compare the
forward-modeled along-scan residual to the per-transit Gaia DR3 epoch
astrometry (where available) or — more commonly — use GOST to compute
the Hipparcos-Gaia long-baseline proper motion in the same way htof
does for HGCA Mode B.

# Fields
- `t::Vector{Float64}`          : transit epochs (MJD)
- `psi::Vector{Float64}`        : along-scan angle (rad)
- `parallax_factor::Vector{Float64}` : along-scan parallax factor
                                  at this transit (Gaia-convention,
                                  signed)
- `along_scan_pos::Vector{Float64}` : predicted geometric along-scan
                                  position contribution at this
                                  transit (mas), from the catalog
                                  five-parameter solution sans the
                                  reflex (i.e. the path the source
                                  would follow if it were single)

# Invariants
- All vectors have matching length.

The precise GOST CSV column names depend on which query interface
was used (the public ESAC GOST tool tabulates several variants); the
fields above are the minimal set needed to evaluate
`gost_log_likelihood`.
"""
struct GOSTData
    t::Vector{Float64}
    psi::Vector{Float64}
    parallax_factor::Vector{Float64}
    along_scan_pos::Vector{Float64}
end

"""
    GOSTData(; t, psi, parallax_factor=nothing, along_scan_pos=nothing)

Keyword constructor. `parallax_factor` and `along_scan_pos` default
to zeros if not supplied.
"""
function GOSTData(;
    t::AbstractVector{<:Real},
    psi::AbstractVector{<:Real},
    parallax_factor::Union{Nothing, AbstractVector{<:Real}} = nothing,
    along_scan_pos::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n = length(t)
    n > 0 || throw(ArgumentError("GOSTData requires at least one transit"))
    length(psi) == n || throw(ArgumentError("psi length mismatch"))

    plx_fac_vec = parallax_factor === nothing ? zeros(Float64, n) :
                  Vector{Float64}(parallax_factor)
    length(plx_fac_vec) == n ||
        throw(ArgumentError("parallax_factor length mismatch"))

    asp_vec = along_scan_pos === nothing ? zeros(Float64, n) :
              Vector{Float64}(along_scan_pos)
    length(asp_vec) == n ||
        throw(ArgumentError("along_scan_pos length mismatch"))

    return GOSTData(
        Vector{Float64}(t),
        Vector{Float64}(psi),
        plx_fac_vec,
        asp_vec,
    )
end

"""
    n_gost(d::GOSTData) -> Int

Number of predicted Gaia scan-plan transits.

GOST is scan geometry, not measurement: these rows say when Gaia looked
and at what angle. See [`gost_log_likelihood`](@ref) for how they enter
the fit.
"""
n_gost(d::GOSTData) = length(d.t)

"""
    G23HData

Thompson+ 2026 G23H ("Gaia DR2/DR3 + Hipparcos") joint catalog row,
the higher-information successor to HGCA EDR3 for astrometric
acceleration analysis (arXiv:2602.00235).

G23H tabulates **five proper-motion measurements** per source instead
of HGCA's three:

  1. Hipparcos epoch (~1991.25)
  2. Hipparcos→Gaia long-baseline (~2004.6)
  3. Gaia DR2 (~2015.0)
  4. Gaia DR3−DR2 derived (mean ~2015.5)
  5. Gaia DR3 (~2016.0)

Crucially, DR2 and DR3 share transit data — their PMs are correlated
(estimated `ρ_{DR2,DR3}` from the ratio of matched transits). The
catalog therefore stores a **full 10×10 covariance matrix** for the
ten PM scalars `(pmra, pmdec)_k for k=1..5`, with the DR2↔DR3 4×4
cross-block populated and other off-diagonals zero (or near-zero) by
construction. Nereus treats the 10×10 covariance opaquely — the
caller is responsible for assembling it from the published catalog
values, typically via the Python sidecar that reads the Arrow/feather
catalog file.

Components NOT included in this data type (Phase 1 scope; future work
to match the full Thompson+ 2026 likelihood):
  - **UEVA** (astrometric excess noise constraint with cube-root
    transformation) — requires a separate `GaiaUEVAData` type.
  - **Hipparcos IAD jitter** term — already exposed via `IADData`.
  - **Gaia RV variability** non-central chi-squared term — requires a
    separate `GaiaRVVarData` type.

# Fields
- `epochs::NTuple{5, Float64}` — (Hip, HG, DR2, DR3−DR2, DR3) MJD
- `pmra::NTuple{5, Float64}` — five PMs in RA (mas/yr)
- `pmdec::NTuple{5, Float64}` — five PMs in Dec (mas/yr)
- `cov::Matrix{Float64}` — 10×10 covariance for
  `[pmra_1, pmdec_1, ..., pmra_5, pmdec_5]` (mas²/yr²)
- `plx::Float64` — parallax (mas)
- `plx_err::Float64` — 1σ on parallax (mas)
- `hip_id::Int` — Hipparcos catalog ID

# Invariants
- `cov` is 10×10, symmetric, positive-definite.
- `plx > 0`, `plx_err > 0`.
"""
struct G23HData
    epochs::NTuple{5, Float64}
    pmra::NTuple{5, Float64}
    pmdec::NTuple{5, Float64}
    cov::Matrix{Float64}
    plx::Float64
    plx_err::Float64
    hip_id::Int
end

"""
    G23HData(; epochs, pmra, pmdec, cov, plx, plx_err, hip_id)

Keyword constructor. The caller passes a 10×10 covariance matrix
already populated with within-epoch 2×2 blocks AND the DR2↔DR3 4×4
cross-block (off-diagonal index pairs `(5:6, 9:10)` and `(9:10, 5:6)`
under the `[pmra_1, pmdec_1, ..., pmra_5, pmdec_5]` ordering).
"""
function G23HData(;
    epochs::NTuple{5, <:Real},
    pmra::NTuple{5, <:Real},
    pmdec::NTuple{5, <:Real},
    cov::AbstractMatrix{<:Real},
    plx::Real,
    plx_err::Real,
    hip_id::Integer,
)
    plx > 0     || throw(ArgumentError("plx must be > 0"))
    plx_err > 0 || throw(ArgumentError("plx_err must be > 0"))
    size(cov) == (10, 10) ||
        throw(ArgumentError("cov must be 10×10 (got $(size(cov)))"))
    cov_sym = Symmetric(Matrix{Float64}(cov))
    isposdef(cov_sym) ||
        throw(ArgumentError("cov must be symmetric positive-definite"))
    return G23HData(
        NTuple{5, Float64}(Float64.(epochs)),
        NTuple{5, Float64}(Float64.(pmra)),
        NTuple{5, Float64}(Float64.(pmdec)),
        Matrix{Float64}(cov_sym),
        Float64(plx),
        Float64(plx_err),
        Int(hip_id),
    )
end

"""
    n_g23h(::G23HData) -> Int

Number of G23H constraint rows, which is always 5 -- the two proper-motion
components at each of two epochs plus their scaled positional difference.
"""
n_g23h(::G23HData) = 5

"""
    GaiaDR3Data

Published Gaia DR3 five-parameter astrometric solution + 5×5 covariance
for a single source. Pairs with `GOSTData` (the predicted scan plan)
to form the htof-equivalent joint Hipparcos+Gaia likelihood when also
combined with `IADData`.

# Fields
- `params::NTuple{5, Float64}` — published 5-vector at `t_ref` MJD,
  ordered `(α₀ [mas], δ₀ [mas], ϖ [mas], μα* [mas/yr], μδ [mas/yr])`.
  `α₀` and `δ₀` are stored as small *offsets* from a fiducial
  reference position (typically the Gaia DR3 published `(α, δ)`
  at `t_ref`), not absolute angles — keeps numerics well-conditioned
  in the joint refit. Setting both to 0 means "no published
  position-shift constraint, only PM + parallax."
- `cov::Matrix{Float64}` — 5×5 published covariance, ordered to
  match `params`.
- `t_ref::Float64` — Gaia DR3 reference epoch (MJD). DR3 uses
  J2016.0 = MJD 57388.0.

# Invariants
- `cov` is symmetric positive-definite.
- `t_ref` is finite.

# Why offsets, not absolute α/δ
The Gaia 5-param fit is locally linear in `q = (α₀, δ₀, ϖ, μα*, μδ)`
about the published values. Treating `params` as offsets keeps the
joint linear system numerically well-conditioned (the published
catalog covariance is mas-scale, while absolute α is ~10⁹ mas).
The orbit-induced shifts `Δq` are also offsets by construction.

# htof correspondence
htof exposes this as `GaiaeData.parse_gaia_table(...).fit_5param()`.
Nereus keeps the data and the fit logic separated: `GaiaDR3Data`
holds the published values; `gost_5param_fit` (in projection.jl)
forecasts the orbit's contribution; `iad_log_likelihood` does the
marginalized joint comparison when `IADData` is present.
"""
struct GaiaDR3Data
    params::NTuple{5, Float64}
    cov::Matrix{Float64}
    t_ref::Float64
end

"""
    GaiaDR3Data(; params, cov, t_ref)

Keyword constructor. `params` order: `(α₀, δ₀, ϖ, μα*, μδ)` in
`(mas, mas, mas, mas/yr, mas/yr)`. `α₀` and `δ₀` are offsets from
the catalog reference position (set to 0 if the position isn't
informative for your fit). `t_ref` in MJD; Gaia DR3 uses J2016.0
= MJD 57388.0 (use `jyear_to_mjd(2016.0)`).
"""
function GaiaDR3Data(;
    params::NTuple{5, <:Real},
    cov::AbstractMatrix{<:Real},
    t_ref::Real,
)
    size(cov) == (5, 5) ||
        throw(ArgumentError("cov must be 5×5 (got $(size(cov)))"))
    isfinite(t_ref) ||
        throw(ArgumentError("t_ref must be finite"))
    cov_sym = Symmetric(Matrix{Float64}(cov))
    isposdef(cov_sym) ||
        throw(ArgumentError("cov must be symmetric positive-definite"))
    return GaiaDR3Data(
        NTuple{5, Float64}(Float64.(params)),
        Matrix{Float64}(cov_sym),
        Float64(t_ref),
    )
end

"""
    jyear_to_mjd(y) -> mjd

Convert a decimal Julian year (e.g. 1991.25) to MJD. JYear 2000.0 is
defined to be JD 2451545.0 (J2000.0) which is MJD 51544.5.
"""
jyear_to_mjd(y::Real) = (y - 2000.0) * 365.25 + 51544.5

"""
    mjd_epochs(epochs_jyear::NTuple{N, <:Real}) -> NTuple{N, Float64}

Vectorized form of `jyear_to_mjd` for any-length epoch tuple. Used
by HGCA (3 epochs) and G23H (5 epochs) constructors.
"""
mjd_epochs(epochs_jyear::NTuple{N, <:Real}) where {N} =
    NTuple{N, Float64}(jyear_to_mjd.(epochs_jyear))
