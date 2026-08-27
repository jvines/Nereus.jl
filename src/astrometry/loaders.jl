# Loaders for public astrometry catalog formats.
#
# Phase 1 supports the HGCA EDR3 FITS file (Brandt 2021), the de facto
# public catalog of Hipparcos-Gaia 3-PM vectors. The file is hosted at
# physics.ucsb.edu/~tbrandt/HGCA_vEDR3.fits (~20 MB) and ships with
# orvara, orbitize!, and Octofitter as the canonical input.
#
# The loader is *thin*: pull a single row by Hipparcos ID, repackage as
# `HGCAData`. Nereus itself never traverses the full table during a
# fit — pre-staging is on the user.
#
# Covariance structure:
#   HGCA tabulates per-epoch 1σ errors AND per-epoch RA-Dec correlations
#   (`pmra_pmdec_hip`/`_hg`/`_gaia`). It does NOT tabulate cross-epoch
#   correlations along a single axis. orvara's likelihood treats the
#   three epochs as independent in time (Brandt 2021, Appendix Eq. 1)
#   but uses the 2×2 within-epoch covariance fully.
#
#   Nereus follows the same factorization. The within-epoch
#   correlations can be substantial — HIP 3850 has ρ_Gaia = 0.66 —
#   so they cannot be ignored.

using FITSIO

"""
    load_hgca_row(fits_path::AbstractString, hip_id::Integer) -> HGCAData

Read a single row of the HGCA EDR3 catalog (`HGCA_vEDR3.fits` from
Brandt 2021) and return an `HGCAData` with full per-epoch 2×2 RA-Dec
covariance.

Epochs are tabulated in HGCA per-axis (`epoch_ra_*`, `epoch_dec_*`); we
take the RA epochs as the canonical times since the RA and Dec epochs
agree to within rounding for all sources.
"""
function load_hgca_row(fits_path::AbstractString, hip_id::Integer)
    isfile(fits_path) || throw(ArgumentError("HGCA file not found: $fits_path"))
    f = FITS(fits_path)
    try
        hdu = f[2]
        hip_col = read(hdu, "hip_id")
        idx = findfirst(==(hip_id), hip_col)
        idx === nothing && throw(ArgumentError(
            "Hipparcos ID $hip_id not found in HGCA EDR3 catalog"))

        # Epochs (Julian year → MJD)
        epoch_hip_jy  = read(hdu, "epoch_ra_hip")[idx]
        epoch_gaia_jy = read(hdu, "epoch_ra_gaia")[idx]
        # HG epoch is the midpoint of the two missions
        epoch_hg_jy   = (epoch_hip_jy + epoch_gaia_jy) / 2

        # Proper motions (mas/yr)
        pmra_hip   = read(hdu, "pmra_hip")[idx]
        pmdec_hip  = read(hdu, "pmdec_hip")[idx]
        pmra_hg    = read(hdu, "pmra_hg")[idx]
        pmdec_hg   = read(hdu, "pmdec_hg")[idx]
        pmra_gaia  = read(hdu, "pmra_gaia")[idx]
        pmdec_gaia = read(hdu, "pmdec_gaia")[idx]

        # Per-epoch 1σ errors
        σ_pmra_hip   = read(hdu, "pmra_hip_error")[idx]
        σ_pmdec_hip  = read(hdu, "pmdec_hip_error")[idx]
        σ_pmra_hg    = read(hdu, "pmra_hg_error")[idx]
        σ_pmdec_hg   = read(hdu, "pmdec_hg_error")[idx]
        σ_pmra_gaia  = read(hdu, "pmra_gaia_error")[idx]
        σ_pmdec_gaia = read(hdu, "pmdec_gaia_error")[idx]

        # Within-epoch RA-Dec correlations
        ρ_hip  = read(hdu, "pmra_pmdec_hip")[idx]
        ρ_hg   = read(hdu, "pmra_pmdec_hg")[idx]
        ρ_gaia = read(hdu, "pmra_pmdec_gaia")[idx]

        # Parallax
        plx     = read(hdu, "parallax_gaia")[idx]
        plx_err = read(hdu, "parallax_gaia_error")[idx]

        return HGCAData(
            epochs = mjd_epochs((epoch_hip_jy, epoch_hg_jy, epoch_gaia_jy)),
            pmra   = (Float64(pmra_hip),  Float64(pmra_hg),  Float64(pmra_gaia)),
            pmdec  = (Float64(pmdec_hip), Float64(pmdec_hg), Float64(pmdec_gaia)),
            sigma_pmra        = (Float64(σ_pmra_hip),  Float64(σ_pmra_hg),  Float64(σ_pmra_gaia)),
            sigma_pmdec       = (Float64(σ_pmdec_hip), Float64(σ_pmdec_hg), Float64(σ_pmdec_gaia)),
            corr_pmra_pmdec   = (Float64(ρ_hip),       Float64(ρ_hg),       Float64(ρ_gaia)),
            plx     = Float64(plx),
            plx_err = Float64(plx_err),
            hip_id  = Int(hip_id),
        )
    finally
        close(f)
    end
end

"""
    load_orvara_rv(rv_path::AbstractString;
                    instrument_label::AbstractString="HIRES",
                    jd_to_mjd::Bool=true) -> NamedTuple

Load orvara-format RV file (single instrument, JD epochs, m/s).

orvara format:
    # comments...
    Epoch[JD]   RV[m/s] RV_err[m/s] InstrumentID
    2452832.91668  89.22  1.62  0
    ...

Returns `(t, rv, rv_err, inst)` ready for `Data(...)`. By default
converts JD → MJD (Nereus / PlanetOrbits convention).
"""
function load_orvara_rv(rv_path::AbstractString;
                        jd_to_mjd::Bool=true)
    rows = Tuple{Float64,Float64,Float64,Int}[]
    for line in eachline(rv_path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        toks = split(s)
        length(toks) >= 4 || continue
        t   = parse(Float64, toks[1])
        rv  = parse(Float64, toks[2])
        err = parse(Float64, toks[3])
        ins = parse(Int,     toks[4])
        push!(rows, (t, rv, err, ins))
    end
    isempty(rows) && throw(ArgumentError("RV file empty: $rv_path"))
    t   = [r[1] - (jd_to_mjd ? 2_400_000.5 : 0.0) for r in rows]
    rv  = [r[2] for r in rows]
    err = [r[3] for r in rows]
    # 1-based instrument indices (orvara is 0-based)
    ins = [r[4] + 1 for r in rows]
    return (t=t, rv=rv, rv_err=err, rv_inst=ins)
end

"""
    load_orvara_relast(path::AbstractString) -> RelAstromData

Load orvara-format relative astrometry file (decimal year, separation
in arcsec, PA in deg, optional ρ-PA correlation, planet ID).

orvara format:
    # comments...
    # Date    Sep   Err_sep   PA   Err_PA   Corr_Sep_PA   PlanetID
     2012.48  2.594 0.014    298.6  0.1      0            0
     ...

Converts to Nereus conventions:
  - `t` in MJD (decimal year → MJD)
  - `ra_off`, `dec_off` in **mas** (separation [arcsec] × 1000, projected
    via PA [degrees])
  - `corr` is the (Δra, Δdec) correlation; orvara's column is
    `Corr_Sep_PA` which we propagate after the (sep, PA) → (ΔRA, Δdec)
    transform.
  - `planet_idx` is 1-based (orvara is 0-based)

The (sep, PA) → (ΔRA, Δdec) transform with full error propagation:
    ΔRA  =  sep · sin(PA)
    Δdec =  sep · cos(PA)
    σ²_ΔRA  = σ²_sep · sin²(PA) + (sep·σ_PA)² · cos²(PA)  + 2 ρ_sP · sep · σ_sep · σ_PA · sin(PA) cos(PA)
    σ²_Δdec = σ²_sep · cos²(PA) + (sep·σ_PA)² · sin²(PA)  − 2 ρ_sP · sep · σ_sep · σ_PA · sin(PA) cos(PA)
    cov(ΔRA, Δdec) = σ²_sep · sin(PA) cos(PA) − (sep·σ_PA)² · sin(PA) cos(PA) + ρ_sP · sep · σ_sep · σ_PA · (cos² − sin²)
"""
function load_orvara_relast(path::AbstractString)
    rows = Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Int}[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        toks = split(s)
        length(toks) >= 7 || continue
        date  = parse(Float64, toks[1])
        sep   = parse(Float64, toks[2])
        ε_sep = parse(Float64, toks[3])
        pa    = parse(Float64, toks[4])
        ε_pa  = parse(Float64, toks[5])
        ρ_sP  = parse(Float64, toks[6])
        pid   = parse(Int,     toks[7])
        push!(rows, (date, sep, ε_sep, pa, ε_pa, ρ_sP, pid))
    end
    isempty(rows) && throw(ArgumentError("relAST file empty: $path"))

    n = length(rows)
    t = Vector{Float64}(undef, n)
    ra_off = Vector{Float64}(undef, n)
    dec_off = Vector{Float64}(undef, n)
    ra_err = Vector{Float64}(undef, n)
    dec_err = Vector{Float64}(undef, n)
    corr = Vector{Float64}(undef, n)
    pidx = Vector{Int}(undef, n)

    for (i, r) in enumerate(rows)
        date, sep, ε_sep, pa_deg, ε_pa_deg, ρ_sP, pid = r
        t[i] = jyear_to_mjd(date)
        # Convert sep from arcsec to mas, PA from deg to rad
        sep_mas = sep * 1000.0
        ε_sep_mas = ε_sep * 1000.0
        pa = deg2rad(pa_deg)
        ε_pa = deg2rad(ε_pa_deg)

        sin_pa, cos_pa = sincos(pa)
        ra_off[i]  = sep_mas * sin_pa
        dec_off[i] = sep_mas * cos_pa

        # Error propagation: var(Δra), var(Δdec), cov(Δra, Δdec)
        v_sep = ε_sep_mas^2
        v_pa  = (sep_mas * ε_pa)^2
        c_sP  = ρ_sP * ε_sep_mas * (sep_mas * ε_pa)
        var_ra  = v_sep * sin_pa^2 + v_pa * cos_pa^2 + 2 * c_sP * sin_pa * cos_pa
        var_dec = v_sep * cos_pa^2 + v_pa * sin_pa^2 - 2 * c_sP * sin_pa * cos_pa
        cov_rd  = (v_sep - v_pa) * sin_pa * cos_pa + c_sP * (cos_pa^2 - sin_pa^2)
        ra_err[i]  = sqrt(max(var_ra,  0.0))
        dec_err[i] = sqrt(max(var_dec, 0.0))
        # Pearson correlation of (Δra, Δdec)
        denom = ra_err[i] * dec_err[i]
        corr[i] = denom > 0 ? clamp(cov_rd / denom, -1.0, 1.0) : 0.0

        pidx[i] = pid + 1   # 0-based → 1-based
    end

    return RelAstromData(
        t = t, ra_off = ra_off, dec_off = dec_off,
        ra_err = ra_err, dec_err = dec_err,
        corr = corr, planet_idx = pidx,
    )
end

# ---------------------------------------------------------------------
# Hipparcos IAD (van Leeuwen 2007 — "Hipparcos: The New Reduction")
# ---------------------------------------------------------------------

"""
    load_hip_iad(path::AbstractString;
                 epoch_col::Int = 1,
                 parallax_factor_col::Int = 2,
                 pm_factor_col::Int = 3,
                 abscissa_col::Int = 4,
                 abscissa_err_col::Int = 5,
                 psi_col::Int = 6,
                 epoch_kind::Symbol = :jyear,
                 epoch_offset::Float64 = 0.0) -> IADData

Read a Hipparcos Intermediate Astrometric Data file in van Leeuwen
2007 redistribution format and return an `IADData`.

The IAD format is documented in Volume 3 of "Hipparcos: The New
Reduction of the Raw Data" (van Leeuwen 2007). The redistribution
DVD ships a per-source ASCII table where each row is a single
transit. Column meanings (defaults below match the redistribution
order Brandt 2018 / htof use):

  1. Epoch (decimal year, with `epoch_offset` typically 1991.25)
  2. Parallax factor (signed, dimensionless)
  3. Proper-motion factor (Δyear from catalog reference epoch,
     typically 1991.25)
  4. Along-scan abscissa residual (mas)
  5. 1σ on abscissa residual (mas)
  6. Scan-direction angle ψ (rad)

Comment lines (`#`) are skipped. Whitespace-delimited, one transit
per line. Override the `*_col` keywords if your file format puts
columns elsewhere.

`epoch_kind` controls how column 1 is interpreted:
  - `:jyear` (default): decimal Julian year, converted via
    `jyear_to_mjd`. Add `epoch_offset` first if your column stores
    `Δyear` from a reference epoch (e.g. some redistributions store
    `epoch − 1991.25`; pass `epoch_offset = 1991.25` then).
  - `:mjd`: already in MJD, used as-is.

This is a deliberately minimal parser — for production use, prefer
htof's parser via the planned EMPEROR II Python sidecar that hands
Nereus pre-staged numbers (D5).
"""
function load_hip_iad(path::AbstractString;
                      epoch_col::Int = 1,
                      parallax_factor_col::Int = 2,
                      pm_factor_col::Int = 3,
                      abscissa_col::Int = 4,
                      abscissa_err_col::Int = 5,
                      psi_col::Int = 6,
                      epoch_kind::Symbol = :jyear,
                      epoch_offset::Float64 = 0.0)
    isfile(path) || throw(ArgumentError("IAD file not found: $path"))
    rows = Tuple{Float64,Float64,Float64,Float64,Float64,Float64}[]
    ncols_needed = max(epoch_col, parallax_factor_col, pm_factor_col,
                       abscissa_col, abscissa_err_col, psi_col)
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        toks = split(s)
        length(toks) >= ncols_needed || continue
        ep   = parse(Float64, toks[epoch_col])
        pxf  = parse(Float64, toks[parallax_factor_col])
        pmf  = parse(Float64, toks[pm_factor_col])
        absc = parse(Float64, toks[abscissa_col])
        aerr = parse(Float64, toks[abscissa_err_col])
        ψ    = parse(Float64, toks[psi_col])
        push!(rows, (ep, pxf, pmf, absc, aerr, ψ))
    end
    isempty(rows) && throw(ArgumentError("IAD file empty: $path"))

    n = length(rows)
    t   = Vector{Float64}(undef, n)
    pxf = Vector{Float64}(undef, n)
    pmf = Vector{Float64}(undef, n)
    absc = Vector{Float64}(undef, n)
    aerr = Vector{Float64}(undef, n)
    ψ    = Vector{Float64}(undef, n)
    for (i, r) in enumerate(rows)
        ep_raw, pxf_i, pmf_i, absc_i, aerr_i, ψ_i = r
        ep_eff = ep_raw + epoch_offset
        if epoch_kind === :jyear
            t[i] = jyear_to_mjd(ep_eff)
        elseif epoch_kind === :mjd
            t[i] = ep_eff
        else
            throw(ArgumentError("epoch_kind must be :jyear or :mjd"))
        end
        pxf[i]  = pxf_i
        pmf[i]  = pmf_i
        absc[i] = absc_i
        aerr[i] = aerr_i
        ψ[i]    = ψ_i
    end

    return IADData(
        t = t,
        abscissa = absc,
        abscissa_err = aerr,
        psi = ψ,
        parallax_factor = pxf,
        pm_factor = pmf,
    )
end

# ---------------------------------------------------------------------
# Gaia GOST (Generic Object Slope-angle Tracker)
# ---------------------------------------------------------------------

"""
    load_gost(path::AbstractString;
              t_col::Union{Symbol,AbstractString} = "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]",
              psi_col::Union{Symbol,AbstractString} = "scanAngle[rad]",
              parallax_factor_col::Union{Symbol,AbstractString} = "parallaxFactorAlongScan",
              along_scan_pos_col::Union{Symbol,AbstractString} = "Position[Heliocentric_Astrometric_Position]_AlongScan",
              t_kind::Symbol = :bjd_tcb,
              delim::Char = ',') -> GOSTData

Read a Gaia GOST CSV export and return a `GOSTData`.

The Gaia GOST tool (https://gaia.esac.esa.int/gost/) returns CSV
files with a header row. **The exact column names depend on which
GOST query was used** — the four columns above are the canonical
ones for the public "single source forecast" mode. If your CSV uses
different names, override the `*_col` keywords.

`t_kind` controls how the time column is interpreted:
  - `:bjd_tcb` (default): Barycentric Julian Date (TCB), converted
    to MJD via `t_mjd = t_bjd − 2_400_000.5`. (TCB and TDB differ by
    < 0.1 s after J2000 — for orbit-fitting at the mas level this is
    immaterial, but a future port should resolve the time scale
    properly per htof.)
  - `:mjd`: already in MJD, used as-is.

The `along_scan_pos` column is optional: if missing, GOSTData is
constructed with along-scan position = 0. Nothing reads it as an
observed value: GOST is a scan-geometry auxiliary and contributes
through the HGCA/DR3 window-average, not through a likelihood of its
own (see [`gost_log_likelihood`](@ref)).

Whitespace tolerance: tokens are stripped and double-quotes removed.
"""
function load_gost(path::AbstractString;
                   t_col::Union{Symbol,AbstractString} =
                       "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]",
                   psi_col::Union{Symbol,AbstractString} = "scanAngle[rad]",
                   parallax_factor_col::Union{Symbol,AbstractString} =
                       "parallaxFactorAlongScan",
                   along_scan_pos_col::Union{Symbol,AbstractString} =
                       "Position[Heliocentric_Astrometric_Position]_AlongScan",
                   t_kind::Symbol = :bjd_tcb,
                   delim::Char = ',')
    isfile(path) || throw(ArgumentError("GOST file not found: $path"))

    lines = String[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        push!(lines, line)
    end
    isempty(lines) && throw(ArgumentError("GOST file empty: $path"))

    header = [strip(replace(t, '"' => "")) for t in split(lines[1], delim)]
    function _idx(want)
        w = String(want)
        i = findfirst(==(w), header)
        i === nothing && throw(ArgumentError(
            "GOST column \"$w\" not found in header: $header"))
        return i
    end
    i_t   = _idx(t_col)
    i_psi = _idx(psi_col)
    i_pxf = _idx(parallax_factor_col)
    i_asp = nothing
    try
        i_asp = _idx(along_scan_pos_col)
    catch _err
        i_asp = nothing  # along_scan_pos is optional
    end

    n = length(lines) - 1
    t   = Vector{Float64}(undef, n)
    psi = Vector{Float64}(undef, n)
    pxf = Vector{Float64}(undef, n)
    asp = i_asp === nothing ? zeros(Float64, n) : Vector{Float64}(undef, n)

    for (i, raw) in enumerate(lines[2:end])
        toks = [strip(replace(t, '"' => "")) for t in split(raw, delim)]
        t_raw = parse(Float64, toks[i_t])
        if t_kind === :bjd_tcb
            t[i] = t_raw - 2_400_000.5
        elseif t_kind === :mjd
            t[i] = t_raw
        else
            throw(ArgumentError("t_kind must be :bjd_tcb or :mjd"))
        end
        psi[i] = parse(Float64, toks[i_psi])
        pxf[i] = parse(Float64, toks[i_pxf])
        if i_asp !== nothing
            asp[i] = parse(Float64, toks[i_asp])
        end
    end

    return GOSTData(
        t = t,
        psi = psi,
        parallax_factor = pxf,
        along_scan_pos = asp,
    )
end

# ---------------------------------------------------------------------
# Gaia DR3 5-param catalog row → GaiaDR3Data
# ---------------------------------------------------------------------
#
# Gaia DR3 publishes its 5-parameter astrometric solution per source via
# the `gaiadr3.gaia_source` ADQL table. Each source carries:
#
#   * 5 parameters: ra (deg), dec (deg), parallax (mas), pmra (mas/yr),
#     pmdec (mas/yr)
#   * 5 1σ uncertainties: ra_error (mas), dec_error (mas),
#     parallax_error (mas), pmra_error (mas/yr), pmdec_error (mas/yr)
#   * 10 cross-correlation coefficients (5×4/2 off-diagonal):
#     `ra_dec_corr`, `ra_parallax_corr`, `ra_pmra_corr`, `ra_pmdec_corr`,
#     `dec_parallax_corr`, `dec_pmra_corr`, `dec_pmdec_corr`,
#     `parallax_pmra_corr`, `parallax_pmdec_corr`, `pmra_pmdec_corr`
#   * `ref_epoch` (Julian year, 2016.0 for DR3)
#
# We assemble the full 5×5 covariance matrix Σ = diag(σ) · ρ · diag(σ)
# with the correlation matrix ρ built from the 10 published values.
#
# Position handling: `α` and `δ` are absolute angles on the sky (degrees),
# but Nereus's `GaiaDR3Data.params` stores α₀ and δ₀ as small offsets
# from a fiducial reference position (in mas) for numerical conditioning
# in the joint htof refit. Default: fiducial = the Gaia position itself,
# so `α₀ = δ₀ = 0`. Pass `ra_fiducial` / `dec_fiducial` (in degrees) to
# override — typically the Hipparcos catalog position when joint-fitting
# Hipparcos IAD + Gaia DR3.

const _GAIA_DR3_REQUIRED_KEYS = (
    :ra, :dec, :parallax, :pmra, :pmdec,
    :ra_error, :dec_error, :parallax_error,
    :pmra_error, :pmdec_error,
    :ra_dec_corr, :ra_parallax_corr, :ra_pmra_corr, :ra_pmdec_corr,
    :dec_parallax_corr, :dec_pmra_corr, :dec_pmdec_corr,
    :parallax_pmra_corr, :parallax_pmdec_corr, :pmra_pmdec_corr,
)

# Generic accessor so the loader works on either a Dict or NamedTuple.
_gaia_get(s::AbstractDict, k::Symbol) =
    haskey(s, String(k)) ? s[String(k)] :
    haskey(s, k)         ? s[k]         :
    throw(KeyError(k))
_gaia_get(s::NamedTuple, k::Symbol) = getfield(s, k)

"""
    load_gaia_dr3(spec; ra_fiducial=nothing, dec_fiducial=nothing,
                  ref_epoch=nothing) -> GaiaDR3Data

Build a `GaiaDR3Data` from a single Gaia DR3 5-parameter catalog row.
`spec` is a `NamedTuple` or `AbstractDict` keyed by the standard
`gaiadr3.gaia_source` column names. Required keys:

  - `ra`, `dec`         (deg)
  - `parallax`          (mas)
  - `pmra`, `pmdec`     (mas/yr)
  - `ra_error`, `dec_error`, `parallax_error`,
    `pmra_error`, `pmdec_error`
  - 10 correlation columns (`ra_dec_corr`, `ra_parallax_corr`, ...,
    `pmra_pmdec_corr`)

Optional:
  - `ref_epoch` — Julian year of the catalog reference epoch.
    Read from `spec[:ref_epoch]` if present; falls back to the
    `ref_epoch` keyword (default `2016.0` for DR3).

# Position fiducial

`α₀` and `δ₀` in the returned `GaiaDR3Data` are stored as offsets in mas
relative to a fiducial position. Default: fiducial = the Gaia row's
`(ra, dec)` itself, yielding `α₀ = δ₀ = 0`. Override via `ra_fiducial`
and `dec_fiducial` (in degrees) when joint-fitting Hipparcos IAD +
Gaia DR3 — typically pass the Hipparcos catalog `(α, δ)` so the joint
solver's catalog 5-vector has a single shared origin.

# Example

```julia
gaia_row = (
    ra = 21.04792, dec = 41.27516, parallax = 17.92,
    pmra = -89.30, pmdec = -57.42,
    ra_error = 0.022, dec_error = 0.018, parallax_error = 0.026,
    pmra_error = 0.030, pmdec_error = 0.024,
    ra_dec_corr        = -0.12,
    ra_parallax_corr   =  0.04, ra_pmra_corr   = -0.02,
    ra_pmdec_corr      =  0.08, dec_parallax_corr = -0.05,
    dec_pmra_corr      =  0.06, dec_pmdec_corr =  0.01,
    parallax_pmra_corr = -0.10, parallax_pmdec_corr = 0.03,
    pmra_pmdec_corr    = -0.18,
    ref_epoch          = 2016.0,
)
gaia = load_gaia_dr3(gaia_row)
```
"""
function load_gaia_dr3(spec;
    ra_fiducial::Union{Nothing, Real} = nothing,
    dec_fiducial::Union{Nothing, Real} = nothing,
    ref_epoch::Real = 2016.0,
)
    spec isa AbstractDict || spec isa NamedTuple ||
        throw(ArgumentError(
            "spec must be a NamedTuple or Dict (got $(typeof(spec)))"))

    # Validate required keys upfront — better error message than KeyError
    # mid-loop.
    if spec isa AbstractDict
        for k in _GAIA_DR3_REQUIRED_KEYS
            haskey(spec, String(k)) || haskey(spec, k) ||
                throw(ArgumentError(
                    "load_gaia_dr3: missing key `$k` in input"))
        end
    else
        ks = keys(spec)
        for k in _GAIA_DR3_REQUIRED_KEYS
            k in ks || throw(ArgumentError(
                "load_gaia_dr3: missing field `$k` in input NamedTuple"))
        end
    end

    ra      = Float64(_gaia_get(spec, :ra))
    dec     = Float64(_gaia_get(spec, :dec))
    plx     = Float64(_gaia_get(spec, :parallax))
    pmra    = Float64(_gaia_get(spec, :pmra))
    pmdec   = Float64(_gaia_get(spec, :pmdec))
    σ_ra    = Float64(_gaia_get(spec, :ra_error))         # mas
    σ_dec   = Float64(_gaia_get(spec, :dec_error))        # mas
    σ_plx   = Float64(_gaia_get(spec, :parallax_error))   # mas
    σ_pmra  = Float64(_gaia_get(spec, :pmra_error))       # mas/yr
    σ_pmdec = Float64(_gaia_get(spec, :pmdec_error))      # mas/yr

    # 10 correlation coefficients
    ρ_ra_dec    = Float64(_gaia_get(spec, :ra_dec_corr))
    ρ_ra_plx    = Float64(_gaia_get(spec, :ra_parallax_corr))
    ρ_ra_pmra   = Float64(_gaia_get(spec, :ra_pmra_corr))
    ρ_ra_pmdec  = Float64(_gaia_get(spec, :ra_pmdec_corr))
    ρ_dec_plx   = Float64(_gaia_get(spec, :dec_parallax_corr))
    ρ_dec_pmra  = Float64(_gaia_get(spec, :dec_pmra_corr))
    ρ_dec_pmdec = Float64(_gaia_get(spec, :dec_pmdec_corr))
    ρ_plx_pmra  = Float64(_gaia_get(spec, :parallax_pmra_corr))
    ρ_plx_pmdec = Float64(_gaia_get(spec, :parallax_pmdec_corr))
    ρ_pm        = Float64(_gaia_get(spec, :pmra_pmdec_corr))

    # ref_epoch from spec if present, else from kwarg
    ref_epoch_jy = if (spec isa AbstractDict &&
                       (haskey(spec, "ref_epoch") || haskey(spec, :ref_epoch))) ||
                      (spec isa NamedTuple && :ref_epoch in keys(spec))
        Float64(_gaia_get(spec, :ref_epoch))
    else
        Float64(ref_epoch)
    end

    # Position offsets from fiducial (deg → mas; account for cos δ
    # scaling on α). 1 deg = 3.6e6 mas.
    rfid = ra_fiducial  === nothing ? ra  : Float64(ra_fiducial)
    dfid = dec_fiducial === nothing ? dec : Float64(dec_fiducial)
    cosδ_fid = cos(deg2rad(dfid))
    α0_mas = (ra  - rfid) * cosδ_fid * 3.6e6
    δ0_mas = (dec - dfid) * 3.6e6

    # Assemble 5×5 covariance Σ = diag(σ) · ρ · diag(σ).
    σvec = (σ_ra, σ_dec, σ_plx, σ_pmra, σ_pmdec)
    ρmat = [
        1.0          ρ_ra_dec     ρ_ra_plx     ρ_ra_pmra    ρ_ra_pmdec
        ρ_ra_dec     1.0          ρ_dec_plx    ρ_dec_pmra   ρ_dec_pmdec
        ρ_ra_plx     ρ_dec_plx    1.0          ρ_plx_pmra   ρ_plx_pmdec
        ρ_ra_pmra    ρ_dec_pmra   ρ_plx_pmra   1.0          ρ_pm
        ρ_ra_pmdec   ρ_dec_pmdec  ρ_plx_pmdec  ρ_pm         1.0
    ]
    Σ = Matrix{Float64}(undef, 5, 5)
    @inbounds for i in 1:5, j in 1:5
        Σ[i, j] = σvec[i] * σvec[j] * ρmat[i, j]
    end

    return GaiaDR3Data(
        params = (α0_mas, δ0_mas, plx, pmra, pmdec),
        cov    = Σ,
        t_ref  = jyear_to_mjd(ref_epoch_jy),
    )
end

"""
    load_gaia_dr3(path::AbstractString;
                  source_id=nothing, delim=',', kwargs...) -> GaiaDR3Data

Load a Gaia DR3 5-parameter row from a CSV file produced by an ADQL
query against `gaiadr3.gaia_source`. The CSV must include (at minimum)
the columns listed in the NamedTuple-input form's docstring.

If the CSV has multiple rows, pass `source_id = <integer>` to disambiguate;
otherwise the loader requires a single-row file.

Other keywords (`ra_fiducial`, `dec_fiducial`, `ref_epoch`) pass through
to the underlying `load_gaia_dr3(::NamedTuple)` form.
"""
function load_gaia_dr3(path::AbstractString;
    source_id::Union{Nothing, Integer} = nothing,
    delim::Char = ',',
    kwargs...)
    isfile(path) || throw(ArgumentError("file not found: $path"))
    raw = readdlm(path, delim, Any, '\n'; header = true)
    m, h_raw = raw[1], vec(raw[2])
    h = String.(strip.(string.(h_raw)))
    n_rows = size(m, 1)
    n_rows >= 1 || throw(ArgumentError("Gaia CSV is empty: $path"))

    idx = if source_id === nothing
        n_rows == 1 || throw(ArgumentError(
            "Gaia CSV $path has $n_rows rows — pass `source_id` to disambiguate"))
        1
    else
        sid_col = findfirst(==("source_id"), h)
        sid_col === nothing && throw(ArgumentError(
            "load_gaia_dr3(path): `source_id` column not found in $path"))
        sid_target = Int(source_id)
        i = findfirst(r -> _to_int(m[r, sid_col]) == sid_target, 1:n_rows)
        i === nothing && throw(ArgumentError(
            "source_id $source_id not found in $path"))
        i
    end

    # Build a Dict from the row, parsing strings to Float64 where needed.
    row = Dict{String, Any}()
    @inbounds for c in 1:length(h)
        row[h[c]] = m[idx, c]
    end
    return load_gaia_dr3(row; kwargs...)
end

# Tolerant int parsing: ADQL CSVs can ship source_id as quoted strings,
# integer literals, or even Float64 (when the column is poorly typed).
function _to_int(x)
    x isa Integer && return Int(x)
    x isa AbstractFloat && return round(Int, x)
    s = strip(string(x))
    return parse(Int, s)
end
