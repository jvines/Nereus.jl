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
- `inst::Vector{Int}`            : imager index per epoch (1-based), the
                                   astrometric twin of `rv_inst`. ORTHOGONAL
                                   to `planet_idx`: that says WHICH COMPANION
                                   an epoch measures, this says WHICH
                                   INSTRUMENT measured it. GPI and SPHERE
                                   observing the same companion are two
                                   instruments and one planet index; they get
                                   separate `sigma_as_<name>` jitters (see
                                   [`relastrom_log_likelihood`](@ref)), which
                                   is the point — their astrometric error
                                   budgets are not the same and a shared one
                                   lets the better instrument absorb the
                                   other's systematics.

# Invariants
- All vectors have matching length.
- `ra_err`, `dec_err` strictly positive.
- `|corr| ≤ 1`.
- `planet_idx` strictly positive.
- `inst` strictly positive; entries index `InstrumentConfig.as_names`.
"""
struct RelAstromData
    t::Vector{Float64}
    ra_off::Vector{Float64}
    dec_off::Vector{Float64}
    ra_err::Vector{Float64}
    dec_err::Vector{Float64}
    corr::Vector{Float64}
    planet_idx::Vector{Int}
    inst::Vector{Int}
end

"""
    RelAstromData(; t, ra_off, dec_off, ra_err, dec_err, corr=nothing,
                    planet_idx=nothing, inst=nothing)

Keyword constructor. Defaults `corr` to all zeros, and `planet_idx` and
`inst` to all ones (the single-companion, single-imager case).
"""
function RelAstromData(;
    t::AbstractVector{<:Real},
    ra_off::AbstractVector{<:Real},
    dec_off::AbstractVector{<:Real},
    ra_err::AbstractVector{<:Real},
    dec_err::AbstractVector{<:Real},
    corr::Union{Nothing, AbstractVector{<:Real}} = nothing,
    planet_idx::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    inst::Union{Nothing, AbstractVector{<:Integer}} = nothing,
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

    inst_vec = inst === nothing ? ones(Int, n) : Vector{Int}(inst)
    length(inst_vec) == n || throw(ArgumentError("inst length mismatch"))
    all(>(0), inst_vec) || throw(ArgumentError("inst entries must be 1-based positive integers"))
    # Gap-free, as for `IADData`: `inst` indexes `InstrumentConfig.as_names`
    # positionally, so a gap silently shifts every later imager onto the wrong
    # jitter parameter.
    let n_i = maximum(inst_vec)
        sort!(unique(inst_vec)) == collect(1:n_i) || throw(ArgumentError(
            "RelAstromData: inst must be gap-free 1-based (got " *
            "$(sort(unique(inst_vec))), expected 1:$n_i) — it indexes " *
            "InstrumentConfig.as_names by position"))
    end

    return RelAstromData(
        Vector{Float64}(t),
        Vector{Float64}(ra_off),
        Vector{Float64}(dec_off),
        Vector{Float64}(ra_err),
        Vector{Float64}(dec_err),
        corr_vec,
        pidx_vec,
        inst_vec,
    )
end

"""
    n_relast(d::RelAstromData) -> Int

Number of relative-astrometry epochs (separation/position-angle or
(dRA, dDec) measurements of a resolved companion).
"""
n_relast(d::RelAstromData) = length(d.t)

"""
    n_relast_inst(d::RelAstromData) -> Int

Number of distinct imagers contributing relative astrometry.
"""
n_relast_inst(d::RelAstromData) = isempty(d.inst) ? 0 : maximum(d.inst)

"""
    merge_relast(sources::RelAstromData...) -> RelAstromData
    merge_relast(sources::AbstractVector{RelAstromData}) -> RelAstromData

Concatenate relative-astrometry sources into one container, numbering the
imagers in the order given. The relative-astrometry twin of
[`merge_iad`](@ref):

```julia
relast = merge_relast(gpi_epochs, sphere_epochs)   # imagers 1 and 2
```

`planet_idx` is carried through untouched — it says which COMPANION each
epoch measures and is orthogonal to which instrument measured it. Two
imagers tracking one companion give two instruments and one planet index.
"""
function merge_relast(sources::AbstractVector{RelAstromData})
    isempty(sources) && throw(ArgumentError("merge_relast needs at least one source"))
    length(sources) == 1 && return only(sources)
    t = Float64[]; ra = Float64[]; dec = Float64[]
    ra_e = Float64[]; dec_e = Float64[]; corr = Float64[]
    pidx = Int[]; inst = Int[]
    offset = 0
    for src in sources
        append!(t, src.t); append!(ra, src.ra_off); append!(dec, src.dec_off)
        append!(ra_e, src.ra_err); append!(dec_e, src.dec_err)
        append!(corr, src.corr); append!(pidx, src.planet_idx)
        append!(inst, src.inst .+ offset)
        offset += n_relast_inst(src)
    end
    return RelAstromData(; t = t, ra_off = ra, dec_off = dec, ra_err = ra_e,
                         dec_err = dec_e, corr = corr, planet_idx = pidx,
                         inst = inst)
end

merge_relast(sources::RelAstromData...) = merge_relast(collect(RelAstromData, sources))

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

# Per-coordinate epochs

The HGCA tabulates a central epoch per mission AND per coordinate
(`epoch_ra_hip`, `epoch_dec_hip`, `epoch_ra_gaia`, `epoch_dec_gaia`): the
epoch at which that coordinate's position is uncorrelated with its proper
motion. They are not one epoch: across the catalogue the Hipparcos RA and Dec
epochs differ by up to 1.2 yr, the Gaia ones by up to 1.3 yr (HIP 103836:
1991.20 / 1991.39 and 2015.96 / 2016.11). The Hipparcos–Gaia proper motion is
the positional difference between those epochs, coordinate by coordinate
(Brandt 2018 §3; orvara and Octofitter model it the same way):

    μα*,HG = (α*_Gaia(t_α,Gaia) − α*_Hip(t_α,Hip)) / (t_α,Gaia − t_α,Hip)

and likewise in Dec with the Dec epochs. `epochs_ra` and `epochs_dec` carry
them, as `(Hip, HG, Gaia)` in MJD with the HG entry the midpoint of the other
two; [`hgca_log_likelihood`](@ref) fits the reflex to those epochs. `epochs`
stays the one-per-quantity tuple used for plotting and `t_ref`.

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
- `epochs_ra::NTuple{3, Float64}`  : (Hip, HG, Gaia) central epochs of RA (MJD)
- `epochs_dec::NTuple{3, Float64}` : (Hip, HG, Gaia) central epochs of Dec (MJD)

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
    epochs_ra::NTuple{3, Float64}
    epochs_dec::NTuple{3, Float64}
end

# The pre-per-coordinate positional form: both coordinates at `epochs`.
HGCAData(epochs, pmra, pmdec, cov_ep, plx, plx_err, hip_id) =
    HGCAData(epochs, pmra, pmdec, cov_ep, plx, plx_err, hip_id,
             NTuple{3, Float64}(Float64.(epochs)), NTuple{3, Float64}(Float64.(epochs)))

"""
    HGCAData(; epochs, pmra, pmdec, sigma_pmra, sigma_pmdec, corr_pmra_pmdec,
              plx, plx_err, hip_id, epochs_ra = epochs, epochs_dec = epochs)

Keyword constructor. Pass per-epoch 1σ errors and within-epoch
RA-Dec correlations; the constructor builds the three 2×2 matrices.

# Arguments
- `sigma_pmra::NTuple{3, Real}`     : (Hip, HG, Gaia) σ on pmra (mas/yr)
- `sigma_pmdec::NTuple{3, Real}`    : (Hip, HG, Gaia) σ on pmdec (mas/yr)
- `corr_pmra_pmdec::NTuple{3, Real}`: (Hip, HG, Gaia) ρ_RA-Dec ∈ [−1, 1]
- `epochs_ra`, `epochs_dec`         : the catalogue's per-coordinate central
                                      epochs (MJD), see the type's docstring.
                                      Default to `epochs` for both
                                      coordinates; [`load_hgca_row`](@ref)
                                      fills them from the catalogue.

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
    epochs_ra::Union{Nothing, NTuple{3, <:Real}} = nothing,
    epochs_dec::Union{Nothing, NTuple{3, <:Real}} = nothing,
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
    ep  = NTuple{3, Float64}(Float64.(epochs))
    era = epochs_ra  === nothing ? ep : NTuple{3, Float64}(Float64.(epochs_ra))
    ede = epochs_dec === nothing ? ep : NTuple{3, Float64}(Float64.(epochs_dec))
    _check_catalogue_epochs(era, ede, 1, 3, "HGCAData")
    return HGCAData(
        ep,
        NTuple{3, Float64}(Float64.(pmra)),
        NTuple{3, Float64}(Float64.(pmdec)),
        cov_eps,
        Float64(plx),
        Float64(plx_err),
        Int(hip_id),
        era, ede,
    )
end

# The per-coordinate epochs set the divisor of a positional-difference proper
# motion, so they must be finite and the two ends of that difference distinct.
function _check_catalogue_epochs(era, ede, i0::Int, i1::Int, who::String)
    (all(isfinite, era) && all(isfinite, ede)) ||
        throw(ArgumentError("$who: epochs_ra / epochs_dec must be finite"))
    (era[i1] != era[i0] && ede[i1] != ede[i0]) || throw(ArgumentError(
        "$who: the epochs at the two ends of a positional difference coincide " *
        "(epochs_ra[$i0] = $(era[i0]), epochs_ra[$i1] = $(era[i1]), " *
        "epochs_dec[$i0] = $(ede[i0]), epochs_dec[$i1] = $(ede[i1]))"))
    return nothing
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
- `inst::Vector{Int}`          : 1-based instrument index per transit,
                                  the astrometric twin of `rv_inst` /
                                  `phot_inst`. All-ones for a single
                                  mission. Two missions in one container
                                  (Hipparcos IAD = 1, Gaia DR4 epoch = 2)
                                  share the parallax and both proper-motion
                                  components but get their own along-scan
                                  zero point — see [`iad_log_likelihood`](@ref).
                                  Build it with [`merge_iad`](@ref) rather
                                  than by hand.
- `ref_params::Vector{NTuple{5, Float64}}` : ONE entry per instrument —
                                  the catalogue five-parameter solution
                                  `(Δα₀, Δδ₀, ϖ, μα*, μδ)` in
                                  `(mas, mas, mas, mas/yr, mas/yr)` that
                                  was SUBTRACTED to form `abscissa`, in the
                                  same order as the design columns. Added
                                  back by the likelihood to recover a full
                                  abscissa. All-zeros when nothing was
                                  subtracted.
- `abscissa_kind::Vector{Symbol}` : ONE entry per instrument, `:absolute`
                                  or `:residual`. `:absolute` means
                                  `abscissa` is already a full along-scan
                                  coordinate (Gaia `centroid_pos_al`);
                                  `:residual` means it is O−C against
                                  `ref_params` (Hipparcos `RES`). The
                                  distinction cannot be inferred from
                                  `ref_params` — a Hipparcos file read
                                  without its header has zeros there and
                                  is still a residual.

# Invariants
- All per-transit vectors have matching length.
- `abscissa_err` strictly positive.
- `psi` is not range-checked (it's in the Hipparcos sign convention,
  which can wrap).
- `inst` is 1-based and gap-free: `sort(unique(inst)) == 1:n_inst`. A gap
  would mean an instrument with no transits and two unconstrained
  zero-point columns, i.e. a rank-deficient design at every likelihood
  call.
- `ref_params` and `abscissa_kind` have one entry per instrument.
- With more than one instrument, a `:residual` instrument must carry a
  non-zero `ref_params`: without it its abscissae cannot be put on the
  same footing as an `:absolute` instrument's, and sharing ϖ and μ across
  the two would be comparing a correction to a full quantity.
"""
struct IADData
    t::Vector{Float64}
    abscissa::Vector{Float64}
    abscissa_err::Vector{Float64}
    psi::Vector{Float64}
    parallax_factor::Vector{Float64}
    pm_factor::Vector{Float64}
    inst::Vector{Int}
    ref_params::Vector{NTuple{5, Float64}}
    abscissa_kind::Vector{Symbol}
    # --- derived from the columns above, computed once at construction -------
    # These are DATA, not model: they do not depend on a single sampled
    # parameter, yet the likelihood used to rebuild all of them on every
    # evaluation -- `sincos(psi)` twice per abscissa (once inside
    # `along_scan_projection` in the residual loop, again in
    # `_iad_normal_equations!`, and a third time on the residual-abscissa
    # branch), `1/sigma^2` per abscissa, and the constant `sum(log sigma)` from
    # scratch over all 824 of them. Measured on a Gaia DR4 source that was
    # 15.3 us of a 68.5 us evaluation, i.e. 22%, repeated 3.3 million times a
    # fit. Caching them is bit-identical, not merely close: the same `sincos`
    # values, and `log_sigma_sum` accumulated in the same left-to-right order
    # the loop used (verified max|delta| = 0 on the residuals, rWr, A and v).
    sinpsi::Vector{Float64}
    cospsi::Vector{Float64}
    weight::Vector{Float64}          # 1 / abscissa_err^2
    log_sigma_sum::Float64           # sum(log(abscissa_err)), left to right
    # Index of the first abscissa of `j`'s epoch group (== j when j leads one).
    #
    # Intermediate astrometry is not one measurement per epoch. Gaia epoch data
    # is per-CCD: a field-of-view transit produces 8-9 abscissae ~5 s apart
    # (measured on two real DR4 sources: 824 -> 93 groups of exactly 9 spanning
    # 39.7 s, and 558 -> 63 of 9 spanning 38.9 s). Hipparcos is stronger still --
    # its grouped abscissae share an epoch EXACTLY (max span 0.0 s on HIP 64426),
    # differing only in scan angle. Across 40 s the mean anomaly of any orbit
    # these fits reach barely moves, so one Kepler solve serves the whole group
    # (see `_iad_residuals_kernels!`), instead of nine identical ones.
    #
    # A group is a CONTIGUOUS run of one instrument inside `_IAD_GROUP_TOL` days
    # of its head, so nothing is assumed about global ordering: unsorted or
    # unclustered input simply yields singleton groups and the old behaviour.
    grp_head::Vector{Int}
end

"""
Epoch-group window, in days. 0.01 d = 14.4 min, comfortably wider than a Gaia
field-of-view transit (~40 s) and far narrower than the gap between transits.
"""
const _IAD_GROUP_TOL = 0.01

"""
    IADData(; t, abscissa, abscissa_err, psi, parallax_factor=nothing,
              pm_factor=nothing, inst=nothing, ref_params=nothing,
              abscissa_kind=nothing)

Keyword constructor. `parallax_factor` and `pm_factor` default to
zeros when not supplied (forward-modeling-only use case).

`inst` defaults to all-ones — one instrument, which is what every
single-mission data set is. `ref_params` and `abscissa_kind` are
per-INSTRUMENT (not per-transit) and default to all-zero five-vectors
and `:absolute` respectively, i.e. "the abscissae are already full
along-scan coordinates and nothing needs adding back".

To combine two missions, load each one separately and join them with
[`merge_iad`](@ref); it renumbers the instruments for you and carries
each source's reference solution across.
"""
function IADData(;
    t::AbstractVector{<:Real},
    abscissa::AbstractVector{<:Real},
    abscissa_err::AbstractVector{<:Real},
    psi::AbstractVector{<:Real},
    parallax_factor::Union{Nothing, AbstractVector{<:Real}} = nothing,
    pm_factor::Union{Nothing, AbstractVector{<:Real}} = nothing,
    inst::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    ref_params::Union{Nothing, AbstractVector} = nothing,
    abscissa_kind::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing,
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

    # --- instrument index -------------------------------------------
    # Gap-free 1-based numbering. A gap means an instrument with no
    # transits, hence two zero-point columns nothing constrains — the
    # design matrix would be rank-deficient at every likelihood call and
    # the marginalisation would silently fall back to its unmarginalised
    # branch. Catch it here instead.
    inst_vec = inst === nothing ? ones(Int, n) : Vector{Int}(inst)
    length(inst_vec) == n ||
        throw(ArgumentError("inst length ($(length(inst_vec))) must match t length ($n)"))
    all(>(0), inst_vec) ||
        throw(ArgumentError("IADData: inst entries must be 1-based positive integers"))
    n_inst = maximum(inst_vec)
    sort!(unique(inst_vec)) == collect(1:n_inst) || throw(ArgumentError(
        "IADData: inst must be gap-free 1-based (got instruments " *
        "$(sort(unique(inst_vec))), expected 1:$n_inst) — an instrument with " *
        "no transits leaves its two along-scan zero-point columns unconstrained"))

    # --- per-instrument catalogue reference solution -----------------
    if ref_params === nothing
        ref_vec = [ntuple(_ -> 0.0, 5) for _ in 1:n_inst]
    else
        length(ref_params) == n_inst || throw(ArgumentError(
            "ref_params has $(length(ref_params)) entries but there are " *
            "$n_inst instrument(s) — it is per instrument, not per transit"))
        ref_vec = Vector{NTuple{5, Float64}}(undef, n_inst)
        for (m, p) in enumerate(ref_params)
            length(p) == 5 || throw(ArgumentError(
                "ref_params[$m] must have 5 entries (Δα₀, Δδ₀, ϖ, μα*, μδ)"))
            all(isfinite, p) || throw(ArgumentError(
                "ref_params[$m] contains non-finite entries"))
            ref_vec[m] = NTuple{5, Float64}(Float64.(Tuple(p)))
        end
    end

    if abscissa_kind === nothing
        kind_vec = fill(:absolute, n_inst)
    elseif abscissa_kind isa Symbol
        kind_vec = fill(abscissa_kind, n_inst)
    else
        kind_vec = Vector{Symbol}(abscissa_kind)
    end
    length(kind_vec) == n_inst || throw(ArgumentError(
        "abscissa_kind has $(length(kind_vec)) entries but there are " *
        "$n_inst instrument(s) — it is per instrument, not per transit"))
    for (m, k) in enumerate(kind_vec)
        k === :absolute || k === :residual || throw(ArgumentError(
            "abscissa_kind[$m] = :$k — must be :absolute or :residual"))
    end

    # Each instrument owns two zero-point columns, so with fewer than two
    # transits it cannot even nominally determine them — it contributes zero
    # residual degrees of freedom and drags the joint design toward
    # rank-deficiency. Only checked when there is more than one instrument,
    # so a one-row single-mission container stays constructible.
    if n_inst > 1
        for m in 1:n_inst
            cnt = count(==(m), inst_vec)
            cnt >= 2 || throw(ArgumentError(
                "IADData: instrument $m has $cnt transit(s). Each instrument " *
                "carries its own two along-scan zero-point columns and needs " *
                "at least 2 transits to constrain them."))
        end
    end

    # A non-zero reference on an :absolute instrument is a contradiction:
    # :absolute says nothing was subtracted to form these abscissae, and
    # ref_params says what was. Left unchecked it silently shifts that
    # instrument's abscissae by a catalogue sky path.
    for m in 1:n_inst
        (kind_vec[m] === :absolute && any(!=(0.0), ref_vec[m])) && throw(ArgumentError(
            "IADData: instrument $m is :absolute but carries a non-zero " *
            "ref_params $(ref_vec[m]). :absolute means the abscissae are " *
            "already full along-scan coordinates and nothing was subtracted. " *
            "If they are O−C residuals against that solution, pass " *
            "abscissa_kind = :residual."))
    end

    # A residual instrument with no reference solution cannot be put on
    # the same footing as an absolute one: its ϖ column means "correction
    # to the catalogue parallax" while the other's means "the parallax".
    # Sharing ϖ and μ across the two would then be comparing different
    # quantities — the exact error this container exists to prevent. Alone
    # it is harmless (the marginalisation projects the whole catalogue
    # solution out), so only reject it when there is something to combine
    # it WITH.
    if n_inst > 1
        for m in 1:n_inst
            (kind_vec[m] === :residual && all(==(0.0), ref_vec[m])) && throw(ArgumentError(
                "IADData: instrument $m carries :residual abscissae but an " *
                "all-zero ref_params, so its full abscissae cannot be " *
                "reconstructed. Combining it with another instrument would " *
                "share ϖ/μ between a catalogue CORRECTION and a full " *
                "quantity. Supply the catalogue five-parameter solution it " *
                "was differenced against (Hipparcos: header line 11 of the " *
                "van Leeuwen residual record)."))
        end
    end

    # Data-only quantities the likelihood would otherwise rebuild on every
    # evaluation (see the struct's field comments). `log_sigma_sum` accumulates
    # left to right, exactly as `_iad_normal_equations!` used to, so the cached
    # value is bit-identical to the one the loop produced.
    psi_vec = Vector{Float64}(psi)
    err_vec = Vector{Float64}(abscissa_err)
    t_vec   = Vector{Float64}(t)
    sin_vec = Vector{Float64}(undef, n)
    cos_vec = Vector{Float64}(undef, n)
    w_vec   = Vector{Float64}(undef, n)
    log_sigma_sum = 0.0
    @inbounds for j in 1:n
        sin_vec[j], cos_vec[j] = sincos(psi_vec[j])
        σ = err_vec[j]
        w_vec[j] = 1 / (σ * σ)
        log_sigma_sum += log(σ)
    end

    # Epoch groups (see the `grp_head` field comment).
    grp_vec = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        if j == 1 || inst_vec[j] != inst_vec[j - 1] ||
           !(0 <= t_vec[j] - t_vec[grp_vec[j - 1]] <= _IAD_GROUP_TOL)
            grp_vec[j] = j
        else
            grp_vec[j] = grp_vec[j - 1]
        end
    end

    return IADData(
        t_vec,
        Vector{Float64}(abscissa),
        err_vec,
        psi_vec,
        plx_fac_vec,
        pm_fac_vec,
        inst_vec,
        ref_vec,
        kind_vec,
        sin_vec,
        cos_vec,
        w_vec,
        log_sigma_sum,
        grp_vec,
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
    n_iad_inst(d::IADData) -> Int

Number of intermediate-astrometry INSTRUMENTS in `d` -- 1 for a single
mission, 2 for a joint Hipparcos IAD + Gaia DR4 epoch container, and so on.

This is the `n_inst` that sizes the marginalisation in
[`iad_log_likelihood`](@ref): it solves for `2 + 2*n_inst` linear nuisance
parameters, a shared `(ϖ, μα*, μδ)` plus one along-scan zero point per
instrument.
"""
n_iad_inst(d::IADData) = length(d.ref_params)

"""
    merge_iad(sources::IADData...) -> IADData
    merge_iad(sources::AbstractVector{IADData}) -> IADData

Concatenate intermediate-astrometry sources into one container, numbering
the instruments in the order given. This is how a joint Hipparcos + Gaia
fit is assembled:

```julia
hip  = fetch_hip_iad(24205)                       # instrument 1
gaia = read_gaia_epoch_votable(xml, source_id).iad  # instrument 2
data = Data(; iad = merge_iad(hip, gaia), t_rv = ..., rv = ..., rv_err = ...)
```

Instrument order is the argument order, NOT sorted — the instruments have
different reference epochs and different abscissa conventions, so an order
nobody chose is a debugging hazard when reading the fitted per-instrument
zero points. A source that is itself multi-instrument keeps its internal
ordering and is appended after the ones before it.

Merging is where a residual-vs-absolute mismatch becomes fatal, so the
`IADData` constructor's check fires here: a `:residual` source with no
catalogue reference solution cannot be combined with anything.
"""
function merge_iad(sources::AbstractVector{IADData})
    isempty(sources) && throw(ArgumentError("merge_iad needs at least one source"))
    length(sources) == 1 && return only(sources)

    t        = Float64[]
    abscissa = Float64[]
    abs_err  = Float64[]
    psi      = Float64[]
    plx_fac  = Float64[]
    pm_fac   = Float64[]
    inst     = Int[]
    refs     = NTuple{5, Float64}[]
    kinds    = Symbol[]

    offset = 0
    for src in sources
        append!(t,        src.t)
        append!(abscissa, src.abscissa)
        append!(abs_err,  src.abscissa_err)
        append!(psi,      src.psi)
        append!(plx_fac,  src.parallax_factor)
        append!(pm_fac,   src.pm_factor)
        append!(inst,     src.inst .+ offset)
        append!(refs,     src.ref_params)
        append!(kinds,    src.abscissa_kind)
        offset += n_iad_inst(src)
    end

    return IADData(; t = t, abscissa = abscissa, abscissa_err = abs_err,
                   psi = psi, parallax_factor = plx_fac, pm_factor = pm_fac,
                   inst = inst, ref_params = refs, abscissa_kind = kinds)
end

merge_iad(sources::IADData...) = merge_iad(collect(IADData, sources))

"""
    iad_for_inst(d::IADData, m::Integer) -> IADData

The transits of instrument `m` alone, as a fresh single-instrument
`IADData` carrying that instrument's reference solution.

Use it for anything that is a per-MISSION question rather than a joint
one — the 5/7/9-parameter solution ladder
([`astrom_logZ`](@ref)) is the main example: "is this source a 5p or a 9p
astrometric solution?" has one answer per catalogue, not one across a
25-year gap.
"""
function iad_for_inst(d::IADData, m::Integer)
    1 <= m <= n_iad_inst(d) || throw(ArgumentError(
        "instrument $m out of range (this IADData has $(n_iad_inst(d)))"))
    keep = findall(==(Int(m)), d.inst)
    return IADData(; t = d.t[keep], abscissa = d.abscissa[keep],
                   abscissa_err = d.abscissa_err[keep], psi = d.psi[keep],
                   parallax_factor = d.parallax_factor[keep],
                   pm_factor = d.pm_factor[keep],
                   ref_params = [d.ref_params[m]],
                   abscissa_kind = [d.abscissa_kind[m]])
end

"""
    clip_transit_outliers(iad; nsigma = 5.0, psi_tol = 1e-4, min_ccds = 3,
                          label = "", verbose = true) -> (iad_clean, report)

Reject along-scan abscissae that disagree with the other CCDs of their own
field-of-view transit. Opt-in; `iad` itself is never modified.

# Why within a transit, and only there

Gaia epoch astrometry is per CCD: one field-of-view transit is 8-9 abscissae
read ~5 s apart (`grp_head`), at one scan angle to 3e-5 rad and one parallax
factor. Across those 40 s nothing the model contains can move -- a 2 mas
reflex on an 84 d orbit shifts by 1e-4 mas, a 600 mas/yr proper motion by
1e-3 mas -- so the CCDs of one transit are repeated measurements of a single
along-scan coordinate, and an outlier among them is identifiable WITHOUT any
orbit or astrometric model. That makes this safe for exactly the case a model
clip is not: a few-mas wobble. A clip against a no-orbit five-parameter
solution flags the wobble itself; this cannot see it.

Measured on the DR4 pre-release: HD 114762's 63 transits have a within-
transit χ²/dof of 9.8, the same excess its orbit fit shows (χ²/N ≈ 9.6) --
the excess is CCDs disagreeing with their own transit, not the orbit. One
transit scatters by ±5 mas across its CCDs and five more have a single CCD
0.6-5.6 mas off. Gaia-4: 1.38 within transits, one CCD 1.0 mas off.

# The rule

For each transit of at least `min_ccds` abscissae whose scan angles agree to
`psi_tol` rad (Hipparcos IAD records sharing an epoch are different scans:
on HIP 64426 their scan angles differ by 1e-4 to 1e-2 rad, so none of its
records is tested):

1. Take the transit median `c` of the abscissae.
2. Flag CCD `j` if `|a_j − c| > nsigma · σ_j`, `σ_j` its reported error.
3. If the flagged CCDs are not a strict minority, there is no consensus on
   where the star was and the WHOLE transit is rejected.

Before step 1 each abscissa is moved to its transit head's scan angle,
`a_j − η⊥_j · Δψ_j`, where `η⊥` is the across-scan position from a no-orbit
five-parameter fit of the instrument. That term is ≤ 0.017 mas on HD 114762
(582 mas/yr) but reaches several tenths of a mas for a Barnard's-star proper
motion, which would otherwise read as an outlier. The fit only supplies that
geometric correction: an orbit it misses by a few mas changes it by 1e-4 mas.

`nsigma = 5`: with ~10³ abscissae per source, Gaussian errors give ~6e-4
false rejections per source; with the reported errors ~15% low (both DR4
sources above, outliers removed) it is still 4.3 true σ, ~0.015 per source.

# Report

`report` is a NamedTuple: `n_in`, `n_rejected`, `nsigma`; per rejected
abscissa (indices into the INPUT `iad`) `index`, `t`, `inst`, `transit` (the
transit's index, in order, within its instrument), `ccd` (position within the
transit), `deviation` (mas, from the transit median), `z` (`deviation/σ`) and
`whole_transit`; and `n_transits_tested`, `n_transits_rejected`. With
`verbose` the same goes to the log, one line per rejected abscissa.

# Cost of the alternatives

A heavy-tailed likelihood (Student-t) or a good/bad mixture per abscissa would
not throw anything away, but either breaks the analytic marginalisation of the
catalogue solution: the integral over `q` is no longer Gaussian, so `q` (4 to
6 parameters, strongly correlated with the orbit) would have to be sampled or
Laplace-approximated on every call, plus an outlier fraction and scale per
instrument. Both also judge outliers against the orbit model rather than
against the transit's own CCDs.
"""
function clip_transit_outliers(iad::IADData; nsigma::Real = 5.0, psi_tol::Real = 1e-4,
                               min_ccds::Int = 3, label::AbstractString = "",
                               verbose::Bool = true)
    nsigma > 0 || throw(ArgumentError("nsigma must be positive, got $nsigma"))
    min_ccds >= 3 || throw(ArgumentError(
        "min_ccds must be at least 3: with two CCDs that disagree there is no " *
        "telling which one is wrong"))
    n = n_iad(iad)

    # Across-scan position per abscissa, from a no-orbit five-parameter fit of
    # each instrument. Only ever multiplied by the 3e-5 rad scan-angle drift
    # inside a transit, so its own error does not matter (see docstring).
    across = zeros(n)
    for m in 1:n_iad_inst(iad)
        idx = findall(==(m), iad.inst)
        sw  = 1 ./ iad.abscissa_err[idx]
        pmf = iad.pm_factor[idx]
        X = hcat(iad.sinpsi[idx], iad.cospsi[idx], iad.parallax_factor[idx],
                 iad.sinpsi[idx] .* pmf, iad.cospsi[idx] .* pmf)
        q = (sw .* X) \ (sw .* iad.abscissa[idx])
        all(isfinite, q) || continue
        @inbounds for (k, j) in enumerate(idx)
            Δα = q[1] + q[4] * pmf[k]
            Δδ = q[2] + q[5] * pmf[k]
            across[j] = Δα * iad.cospsi[j] - Δδ * iad.sinpsi[j]
        end
    end

    rej = Int[]; ccd = Int[]; trn = Int[]; dev = Float64[]; zs = Float64[]
    whole = Bool[]
    n_tested = 0; n_whole = 0
    transit_no = zeros(Int, n_iad_inst(iad))
    j = 1
    while j <= n
        h = iad.grp_head[j]
        k = j
        while k < n && iad.grp_head[k + 1] == h
            k += 1
        end
        g = j:k
        m = iad.inst[j]
        transit_no[m] += 1
        j = k + 1
        length(g) >= min_ccds || continue
        dψ = [rem2pi(iad.psi[i] - iad.psi[h], RoundNearest) for i in g]
        maximum(abs, dψ) <= psi_tol || continue
        n_tested += 1
        a = [iad.abscissa[i] - across[i] * dψ[ii] for (ii, i) in enumerate(g)]
        c = median(a)
        d = a .- c
        z = d ./ iad.abscissa_err[g]
        bad = abs.(z) .> nsigma
        any(bad) || continue
        all_out = 2 * count(!, bad) <= length(g)       # no strict majority left
        all_out && (n_whole += 1)
        for (ii, i) in enumerate(g)
            (bad[ii] || all_out) || continue
            push!(rej, i); push!(ccd, ii); push!(trn, transit_no[m])
            push!(dev, d[ii]); push!(zs, z[ii]); push!(whole, all_out)
        end
    end

    report = (n_in = n, n_rejected = length(rej), nsigma = Float64(nsigma),
              index = rej, t = iad.t[rej], inst = iad.inst[rej], transit = trn,
              ccd = ccd, deviation = dev, z = zs, whole_transit = whole,
              n_transits_tested = n_tested, n_transits_rejected = n_whole)

    if verbose
        tag = isempty(label) ? "" : " ($label)"
        lines = ["MJD $(round(report.t[r]; digits = 5))  inst $(report.inst[r])  " *
                 "transit $(report.transit[r]) CCD $(report.ccd[r])  " *
                 "Δ = $(round(report.deviation[r]; digits = 3)) mas " *
                 "($(round(report.z[r]; digits = 1))σ)" *
                 (report.whole_transit[r] ? "  [whole transit]" : "")
                 for r in eachindex(rej)]
        @info "clip_transit_outliers$tag: rejected $(length(rej)) of $n abscissae " *
              "at $(nsigma)σ from their transit median ($n_tested transits tested, " *
              "$n_whole rejected whole)" * (isempty(lines) ? "" : "\n  " * join(lines, "\n  "))
    end

    isempty(rej) && return iad, report
    keep = setdiff(1:n, rej)
    for m in 1:n_iad_inst(iad)
        any(i -> iad.inst[i] == m, keep) || throw(ArgumentError(
            "clip_transit_outliers rejected every abscissa of instrument $m"))
    end
    clean = IADData(; t = iad.t[keep], abscissa = iad.abscissa[keep],
                    abscissa_err = iad.abscissa_err[keep], psi = iad.psi[keep],
                    parallax_factor = iad.parallax_factor[keep],
                    pm_factor = iad.pm_factor[keep], inst = iad.inst[keep],
                    ref_params = iad.ref_params, abscissa_kind = iad.abscissa_kind)
    return clean, report
end

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
    GAIA_DR2_WINDOW, GAIA_DR3_WINDOW

The spans of Gaia observations each release's astrometry was fitted to, as
`(start, end)` in MJD: DR2 2014-07-25 10:30 to 2016-05-23 11:35 UTC
(Lindegren et al. 2018), EDR3/DR3 2014-07-25 10:30 to 2017-05-28 08:44 UTC
(Lindegren et al. 2021). A GOST scan plan covers the whole mission, so a
release's five-parameter solution is reproduced from the transits inside its
window only: the `window` keyword of `gost_5param_fit` in the
likelihoods (HGCA, G23H, Gaia DR3 + IAD), `_gost_window` in the plots.
"""
const GAIA_DR2_WINDOW = (56863.4375, 57531.4826)
const GAIA_DR3_WINDOW = (56863.4375, 57901.3639)

"""
    HIPPARCOS_WINDOW

The span of the Hipparcos mission's observations, `(start, end)` in MJD: the
Julian years 1989.84 to 1993.20. Every unrejected record of the 2007
reduction's intermediate data (the `ResRec_JavaTool_2014` bundle
[`fetch_hip_iad`](@ref) reads, all sources) lies between EPOCH −1.4049 and
+1.9458 yr from J1991.25, i.e. 1989.8451 to 1993.1958; the window is that
range rounded outwards. The Hipparcos epoch of the HGCA and G23H models is a
five-parameter fit through the scans inside it — which also keeps the Gaia
transits of a merged Hipparcos + Gaia `IADData` out of that fit.
"""
const HIPPARCOS_WINDOW = ((1989.84 - 2000.0) * 365.25 + 51544.5,
                          (1993.20 - 2000.0) * 365.25 + 51544.5)

# Scans of `t` inside `win`, both ends inclusive.
_n_in_window(t::AbstractVector{<:Real}, win::Tuple{Real, Real}) =
    count(x -> win[1] <= x <= win[2], t)

"""
    _gost_window(gost, (t0, t1)) -> GOSTData or nothing

The transits of `gost` with `t0 ≤ t ≤ t1`, or `nothing` when fewer than five
fall inside: a five-parameter fit needs five, and `gost_5param_fit` returns
zeros below that, which would read as a real projection.
"""
function _gost_window(gost::GOSTData, win::Tuple{Real, Real})
    keep = (gost.t .>= win[1]) .& (gost.t .<= win[2])
    count(keep) >= 5 || return nothing
    return GOSTData(gost.t[keep], gost.psi[keep], gost.parallax_factor[keep],
                    gost.along_scan_pos[keep])
end

"""
    G23HData

Thompson+ 2026 G23H ("Gaia DR2/DR3 + Hipparcos") joint catalog row,
the higher-information successor to HGCA EDR3 for astrometric
acceleration analysis (arXiv:2602.00235).

G23H tabulates **five proper-motion measurements** per source instead
of HGCA's three:

  1. Hipparcos epoch (~1991.25) — adopted from the HGCA
  2. Hipparcos→Gaia long-baseline (~2004) — adopted from the HGCA
  3. Gaia DR2 (~2015.5)
  4. Gaia DR3−DR2 scaled position difference (~2015.75)
  5. Gaia DR3 (~2016.0)

# What each one is, and at which epochs

Thompson+ 2026 §II–III and Table 1, and Octofitter's `G23HObs`, which is the
catalogue's reference forward model:

  - 1, 3, 5 are the proper motions of five-parameter fits over the
    Hipparcos, DR2 and DR3 observations.
  - 2 is `(α*_DR3(t_DR3) − α*_Hip(t_Hip)) / (t_DR3 − t_Hip)`, per coordinate.
  - 4 is `(α*_DR3(t_DR3) − α*_DR2(t_DR2)) / (t_DR3 − t_DR2)`, per coordinate.

Every `t` there is the catalogue's CENTRAL epoch of that release and
coordinate (`epoch_ra_hip`, `epoch_dec_hip`, `epoch_ra_dr2`, …, the epochs
the paper propagates both catalogues to before differencing), not the DR2
and DR3 reference epochs J2015.5 and J2016.0. Pass them as `epochs_ra` and
`epochs_dec`, ordered like `epochs`; the HG and DR3−DR2 entries of those are
not used by the model (their divisors come from the end-point epochs).

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
- `epochs_ra::NTuple{5, Float64}` — central epochs of RA, ordered like
  `epochs` (MJD): the catalogue's `epoch_ra_hip`, `epoch_ra_hg`,
  `epoch_ra_dr2`, `epoch_ra_dr32`, `epoch_ra_dr3`
- `epochs_dec::NTuple{5, Float64}` — the same for Dec

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
    epochs_ra::NTuple{5, Float64}
    epochs_dec::NTuple{5, Float64}
end

# The pre-per-coordinate positional form: both coordinates at `epochs`.
G23HData(epochs, pmra, pmdec, cov, plx, plx_err, hip_id) =
    G23HData(epochs, pmra, pmdec, cov, plx, plx_err, hip_id,
             NTuple{5, Float64}(Float64.(epochs)), NTuple{5, Float64}(Float64.(epochs)))

"""
    G23HData(; epochs, pmra, pmdec, cov, plx, plx_err, hip_id,
               epochs_ra = epochs, epochs_dec = epochs)

Keyword constructor. The caller passes a 10×10 covariance matrix
already populated with within-epoch 2×2 blocks AND the DR2↔DR3 4×4
cross-block (off-diagonal index pairs `(5:6, 9:10)` and `(9:10, 5:6)`
under the `[pmra_1, pmdec_1, ..., pmra_5, pmdec_5]` ordering).

`epochs_ra` / `epochs_dec` are the catalogue's per-coordinate central epochs
(MJD; the catalogue tabulates Julian years, so `mjd_epochs` them). They set the
reference epochs of the Hipparcos, DR2 and DR3 fits and the divisors of the
two positional differences, so pass them whenever you have the catalogue row;
both default to `epochs`.
"""
function G23HData(;
    epochs::NTuple{5, <:Real},
    pmra::NTuple{5, <:Real},
    pmdec::NTuple{5, <:Real},
    cov::AbstractMatrix{<:Real},
    plx::Real,
    plx_err::Real,
    hip_id::Integer,
    epochs_ra::Union{Nothing, NTuple{5, <:Real}} = nothing,
    epochs_dec::Union{Nothing, NTuple{5, <:Real}} = nothing,
)
    plx > 0     || throw(ArgumentError("plx must be > 0"))
    plx_err > 0 || throw(ArgumentError("plx_err must be > 0"))
    size(cov) == (10, 10) ||
        throw(ArgumentError("cov must be 10×10 (got $(size(cov)))"))
    cov_sym = Symmetric(Matrix{Float64}(cov))
    isposdef(cov_sym) ||
        throw(ArgumentError("cov must be symmetric positive-definite"))
    ep  = NTuple{5, Float64}(Float64.(epochs))
    era = epochs_ra  === nothing ? ep : NTuple{5, Float64}(Float64.(epochs_ra))
    ede = epochs_dec === nothing ? ep : NTuple{5, Float64}(Float64.(epochs_dec))
    _check_catalogue_epochs(era, ede, 1, 5, "G23HData")   # Hip → DR3
    _check_catalogue_epochs(era, ede, 3, 5, "G23HData")   # DR2 → DR3
    return G23HData(
        ep,
        NTuple{5, Float64}(Float64.(pmra)),
        NTuple{5, Float64}(Float64.(pmdec)),
        Matrix{Float64}(cov_sym),
        Float64(plx),
        Float64(plx_err),
        Int(hip_id),
        era, ede,
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
