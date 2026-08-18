# Data models

Nereus organises all observations through a single immutable struct
`Data` that holds the RV channel, the photometric channel, activity
indicator time series, and (optionally) one or more astrometry
products. Every sampler, likelihood, and plotting function reads from a
`Data` value.

There are two ways to build a `Data`:

1. **Julia API** — call the `Data(; ...)` keyword constructor and the
   astrometry sub-constructors directly. Covered below.
2. **`run_job` JSON/Dict config** — pass a `data` block to
   `Nereus.run_job`; the runner's `_build_data` translates it into a
   `Data` for you. This is the production path (exoautomata, Docker,
   `juliacall`). The block schema is documented in
   [The `data` config block](#The-data-config-block) and the canonical
   reference is [`docs/JOB_CONFIG.md`](https://github.com/) /
   `src/runner.jl`.

Both paths produce the *same* `Data` struct, so the field semantics
below apply regardless of how you constructed it.

## The `Data` struct

```julia
Data(;
    t_rv = Float64[], rv = Float64[], rv_err = Float64[], rv_inst = nothing,
    t_ref = nothing,
    indicators = nothing, indicator_errs = nothing,
    t_phot = Float64[], flux = Float64[], flux_err = Float64[],
    phot_inst = nothing, exposure_times = Float64[],
    relastrom = nothing, hgca = nothing, iad = nothing,
    gost = nothing, g23h = nothing, gaia_dr3 = nothing,
)
```

At least one observation channel must be non-empty (RV *or*
photometry *or* astrometry), otherwise the constructor throws
`ArgumentError`. It validates lengths, positivity of every error bar,
1-based `rv_inst`/`phot_inst` indices, and indicator/indicator-error
length consistency.

| Field | Type | Meaning |
|---|---|---|
| `t_rv`, `rv`, `rv_err` | `Vector{Float64}` | Time, value, uncertainty per RV observation. Units are the user's choice but must match the RV-prior scales (typically days / m·s⁻¹). |
| `rv_inst` | `Vector{Int}` | 1-based instrument index per RV point. Must match the ordering in `InstrumentConfig.rv`. Defaults to all-ones (single-instrument). |
| `rv_comp` | `Vector{Int}` | Stellar **component** each RV point measures, for **SB2 double-lined** fits: `1` = primary (A), `2` = secondary (B). Orthogonal to `rv_inst` (γ/σ/activity stay per-spectrograph; the barycentric γ is shared across A and B). Defaults to all-ones (the normal single-star case). Used only by the `BINARY`/`BINARY_RV` modes. |
| `t_ref` | `Float64` | Reference epoch for the `M₀`/`Tp` time anchors. Defaults to `median(t_rv)`; falls back to `median(t_phot)`, then to an astrometry epoch (IAD → GOST → relAST → HGCA midpoint → G23H DR2 epoch → Gaia DR3 `t_ref`) when RVs and photometry are absent. |
| `indicators` | `Dict{String, Vector{Float64}}` | Activity indicator time series (BIS, CCF FWHM, S-index, Hα, log R′<sub>HK</sub>, CRX, …), one entry per indicator name. Each length must match `rv`. Consumed by `ActivityDecorrelation` and `ActivityGP`. |
| `indicator_errs` | `Dict{String, Vector{Float64}}` | Per-indicator 1σ uncertainty. Only present for indicators where errors were supplied (entry key must have a matching `indicators` entry). **Required by `ActivityGP`** and any Rajpaul-style multivariate-GP indicator likelihood; ignored by `ActivityDecorrelation`. Entries must be strictly positive. See [The indicator-error convention](#The-indicator-error-convention). |
| `indicator_derivs` | `Dict{String, Vector{Float64}}` | Per-instrument finite-difference time derivatives of each indicator. **Auto-computed** by the constructor (NaN-safe central differences, one-sided at endpoints, per instrument). Used by `ActivityDecorrelation(derivative=true)` (FF′ term, [Aigrain+ 2012](https://ui.adsabs.harvard.edu/abs/2012MNRAS.419.3147A/abstract)). Not a constructor argument. |
| `t_phot`, `flux`, `flux_err` | `Vector{Float64}` | Photometry analogue of the RV channel. `flux` is normalised to ~1.0 out of transit. |
| `phot_inst` | `Vector{Int}` | 1-based photometric instrument/sector index. Defaults to all-ones. |
| `exposure_times` | `Vector{Float64}` | Per-cadence exposure time **in days** (same units as `t_phot`). Empty ⇒ instantaneous model evaluation (no finite-exposure supersampling). When present, length must match `t_phot`. |
| `relastrom` | `Union{Nothing, RelAstromData}` | Relative astrometry (RA/Dec offsets per epoch) — direct-imaging companions. |
| `hgca` | `Union{Nothing, HGCAData}` | Hipparcos–Gaia Catalogue of Accelerations row (Brandt 2018, 2021). Auto-stage with `fetch_hgca(hip_id)`. The **validated** Hipparcos–Gaia path (recovers the orvara orbit on HD 159062). |
| `iad` | `Union{Nothing, IADData}` | Along-scan Intermediate Astrometric Data. Auto-stage Hipparcos with `fetch_hip_iad(hip_id)`; **Gaia DR4 epoch astrometry** loads into the *same* channel via `read_gaia_epoch_votable` / `fetch_gaia_dr4_prerelease` (see below). Constrains the epoch abscissae; the Hip+Gaia IAD joint additionally needs `gaia_dr3` (implemented, not yet validated). |
| `gost` | `Union{Nothing, GOSTData}` | Gaia scanning law from the GOST tool (predicted scan epochs + angles). Auto-stage with `fetch_gost(ra, dec)`. **Auxiliary, not a standalone measurement** — GOST only window-averages (Mode B) the Gaia epoch of a paired `hgca`/`gaia_dr3`/`g23h`; supplied alone (or only with `iad`) it is inert and the constructor warns. |
| `g23h` | `Union{Nothing, G23HData}` | G23H five-PM Gaia DR2/DR3 + Hipparcos joint product (Thompson+ 2026, arXiv:2602.00235). |
| `gaia_dr3` | `Union{Nothing, GaiaDR3Data}` | Gaia DR3 5-parameter astrometric solution. |

Helper accessors:
- `n_rv(data)`, `n_phot(data)` (defined on `Data`); `n_relast`,
  `n_iad`, `n_gost`, `n_g23h` are defined on the respective astrometry
  structs.
- `has_astrometry(data)` — true if any of the six astrometry payloads
  is non-`nothing`.

!!! note "Thin container by design"
    `Data` carries no loaders, detrenders, or reducers (spec decision
    D5). A Python sidecar in the ecosystem handles data discovery, BJD
    conversion, detrending, outlier rejection, and instrument-specific
    systematics, then hands Nereus plain numeric arrays with integer
    instrument indices. The convenience loaders in this package
    (`load_*`, `fetch_*`) are thin parsers around already-published
    text/FITS/CSV products, not pipeline stages.

## RV format

Single-instrument fit:

```julia
data = Data(; t_rv = bjd, rv = rv, rv_err = err)
```

Multi-instrument — pass an integer index per epoch:

```julia
inst_names = ["HARPS_PRE", "HARPS_POST", "FEROS"]
inst_map   = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst    = [inst_map[s] for s in inst_str_per_epoch]

data = Data(; t_rv = bjd, rv = rv, rv_err = err, rv_inst = rv_inst)
ic   = InstrumentConfig(rv = inst_names)
```

The integer-index convention scales linearly with the number of
instruments and avoids string comparisons in the likelihood hot path.

## Indicators

For activity decorrelation, attach indicator time series whose length
matches `rv`:

```julia
data = Data(;
    t_rv = bjd, rv = rv, rv_err = err, rv_inst = rv_inst,
    indicators = Dict(
        "bisector_span" => bis_per_epoch,
        "halpha"        => halpha_per_epoch,
    ),
)
```

The constructor auto-computes `indicator_derivs` (per-instrument
central differences with one-sided endpoints), so
`ActivityDecorrelation(indicators = ["bisector_span"], derivative = true)`
works out of the box. See [Noise models](noise_models.md).

### The indicator-error convention

`ActivityDecorrelation` only needs the indicator *values*. The
multivariate-GP activity model `ActivityGP` (a single latent GP `G(t)`
driving BIS/FWHM/logR′<sub>HK</sub>/Hα through channel couplings,
Rajpaul-style) treats the indicators as a **data block with its own
noise**, so it additionally requires a per-indicator 1σ error array.
Pass them via `indicator_errs`:

```julia
data = Data(;
    t_rv = bjd, rv = rv, rv_err = err, rv_inst = rv_inst,
    indicators = Dict(
        "bis"  => bis,
        "fwhm" => fwhm,
    ),
    indicator_errs = Dict(
        "bis"  => bis_err,
        "fwhm" => fwhm_err,
    ),
)
```

Rules enforced by the constructor (`src/data.jl`):
- Every key in `indicator_errs` must have a matching key in
  `indicators` (a lone `<name>_err` with no `<name>` value is an error).
- Each error array length must equal `n_rv`.
- Every error entry must be strictly positive.

If you only supply errors for *some* indicators, that's fine — the
`indicator_errs` dict simply omits the ones you left out. `ActivityGP`
will only run over the channels for which both values **and** errors are
present.

## Photometry format

```julia
data = Data(;
    # ... RV fields ...
    t_phot         = times,
    flux           = normalized_flux,
    flux_err       = flux_uncertainty,
    phot_inst      = ones(Int, length(times)),  # 1 instrument
    exposure_times = fill(120/86_400, length(times)),  # 120 s, in days (optional)
)
```

Multi-sector / multi-instrument photometry uses the same integer-index
convention. Each photometric instrument can have its own
limb-darkening parameters (`q1`, `q2`), jitter (`jitter_<INST>`), and
offset (`offset_<INST>`) — auto-generated by [Priors](priors.md).

`exposure_times` (in **days**) enable finite-exposure supersampling of
the transit model. Leave them empty to evaluate the model
instantaneously at each cadence.

Photometry must be normalised to ~1.0 (out-of-transit baseline);
Nereus's transit model expects `flux ≈ 1` between transits.

For long, multi-sector LCs that need detrending before fitting, see
[Preprocessing](preprocessing.md).

## Rossiter–McLaughlin (RV + transit)

When a planet's mode includes an RM source (see the modes below),
Nereus adds the Rossiter–McLaughlin RV anomaly to the orbit-only RV
prediction. The model is the leading-order Hirano+ 2011 analytic form
(`RVPM_RM`/`RVPMAS_RM`) or the disk-integrated Reloaded form (Cegla+
2016, `RVPM_RM_R`/`RVPMAS_RM_R`); see `src/rm.jl`.

The RM term lives in the **RV channel** — there is no separate dataset
to attach. You enable it through the planet mode and two extra
parameters:

- **Planet modes** (`model.planet_modes`): `RVPM_RM`, `RVPMAS_RM`,
  `RVPM_RM_R`, `RVPMAS_RM_R`. All require a PM (transit) source by
  construction.
- **`v_sin_i_star`** — system-level stellar projected rotation
  V·sin(i⋆) in **m/s**. One per fit; added automatically when any
  planet has an RM mode. Default prior `LogUniform(500, 100000)` m/s
  (`src/default_priors.jl`).
- **`lambda_k<k>`** — sky-projected stellar obliquity λ for planet `k`
  in **radians**. One per RM-enabled planet. Default prior
  `Uniform(-π, π)`.

The RM contribution is now folded into `rv_predictions`
(`src/likelihood.jl`), so the posterior-predictive checks, residuals,
and every RV plot are RM-consistent. The dedicated diagnostic is the
`rm_anomaly` plot (`plot_rm`).

## Astrometry

Nereus supports six astrometric data products. Construct each with its
own keyword constructor and pass it into `Data(; ...)`. See
[Worked examples](examples.md) for end-to-end recipes. All angles are
in radians, separations/offsets in mas, PMs in mas/yr, epochs in **MJD**.

### `RelAstromData` — Direct-imaging companions

Tangent-plane offsets `(ΔRA·cos δ, Δδ)` of the companion relative to
the host, in mas. Each epoch binds to a planet index `k`.

```julia
relast = RelAstromData(;
    t          = epochs_mjd,    # MJD
    ra_off     = ra_offsets,    # ΔRA·cosδ (mas)
    dec_off    = dec_offsets,   # Δδ (mas)
    ra_err     = ra_errs,       # 1σ (mas)
    dec_err    = dec_errs,      # 1σ (mas)
    corr       = nothing,       # per-epoch RA–Dec correlation ∈ [-1,1]; default 0
    planet_idx = nothing,       # 1-based planet index per epoch; default all-ones
)
data = Data(; relastrom = relast)
```

### `HGCAData` — Hipparcos–Gaia accelerations

A single Brandt HGCA row: three calibrated proper-motion vectors at the
Hipparcos (~1991.25), Hipparcos→Gaia long-baseline, and Gaia (~2016.0)
epochs, each with a 2×2 within-epoch RA–Dec covariance.

```julia
hgca = HGCAData(;
    epochs          = mjd_epochs((1991.25, 2004.6, 2016.0)),  # decimal-year → MJD
    pmra            = (pmra_hip, pmra_hg, pmra_gaia),         # mas/yr
    pmdec           = (pmdec_hip, pmdec_hg, pmdec_gaia),      # mas/yr
    sigma_pmra      = (σ_hip, σ_hg, σ_gaia),                  # mas/yr
    sigma_pmdec     = (σ_hip, σ_hg, σ_gaia),                  # mas/yr
    corr_pmra_pmdec = (ρ_hip, ρ_hg, ρ_gaia),                 # within-epoch ρ ∈ (-1,1); default 0
    plx             = parallax_mas,
    plx_err         = parallax_err_mas,
    hip_id          = 27321,
)
data = Data(; hgca = hgca)
```

Loader for the HGCA FITS catalogue:

```julia
hgca = load_hgca_row("HGCA_vEDR3.fits", 27321)   # (fits_path, hip_id)
```

!!! note "Hipparcos–Gaia (HG) proper motion is a mean, not an instantaneous PM"
    The catalogue's cross-calibrated HG proper motion is the scaled
    positional difference `(Gaia − Hip)/baseline`, i.e. the **mean**
    reflex velocity over the ~25 yr baseline (Brandt 2021 Eq. 1). The
    likelihood models it that way —
    `(reflex_offset(Gaia) − reflex_offset(Hip))/baseline` — so the
    inferred companion mass is correct even when the orbital period is
    **comparable to the baseline** (the reflex is then strongly
    non-linear across it). Treating the HG term as an instantaneous PM
    at the midpoint biases the mass low for such short-period companions
    (e.g. HD 4747 B, P≈33 yr — validated against orvara).

### `IADData` — Hipparcos along-scan

Per-transit along-scan abscissa residuals + scan angles. Each
measurement constrains the orbit's projection onto its own scan
direction; the variety of scan angles ψ across the mission supplies the
2-D information.

```julia
iad = IADData(;
    t               = transit_epochs_mjd,
    abscissa        = along_scan_residual_mas,
    abscissa_err    = along_scan_err_mas,       # 1σ, > 0
    psi             = scan_angle_rad,           # along-scan direction angle
    parallax_factor = nothing,                  # optional; default zeros
    pm_factor       = nothing,                  # optional; default zeros
)
data = Data(; iad = iad)
```

**Auto-fetch** the van Leeuwen 2007 IAD for a HIP source — downloads
the 350 MB ESA bundle once, then caches per-source (instant on repeat):

```julia
iad = fetch_hip_iad(27321; verbose = false)
```

Cache layout (`$NEREUS_DATA_CACHE` or `~/.nereus/`):
`IAD/ResRec_JavaTool_2014.zip` (full bundle) and the extracted
`IAD/ResRec_JavaTool_2014/H273/H027321.d`. Pass
`force_redownload = true` to refresh. The `_parse_van_leeuwen_iad`
parser drops scans flagged rejected (`SRES < 0`) and sets
`psi = atan2(CPSI, SPSI)`.

There is also a `load_hip_iad(path; ...)` loader for the 6-column
synthetic IAD format (distinct from the bundle's `IORB EPOCH PARF CPSI
SPSI RES SRES` layout).

### Gaia DR4 epoch astrometry → `IADData`

Gaia DR4 (release 2 Dec 2026) publishes per-CCD **along-scan (AL) epoch
astrometry**: for each field-of-view transit, the AL centroid position
`centroid_pos_al` (mas) of the source relative to a reference point,
together with the scan position angle and the AL parallax factor. The AL
abscissa model is *identical* in structure to the Hipparcos IAD model that
`IADData` + `iad_log_likelihood` already implement (design vector
`(sinψ, cosψ, plx_factor, sinψ·Δt, cosψ·Δt)`, with the five astrometric
parameters marginalised analytically). So a DR4 source drops straight into
the existing likelihood — **no new likelihood code**; the only new piece is
a reader.

```julia
# Fetch + read the June-2026 pre-release bundle (12 illustrative sources,
# incl. the orbital systems HD 114762, Gaia-4, Gaia BH3):
path = fetch_gaia_dr4_prerelease()             # cached VOTable path
sources = read_gaia_epoch_votable(path)        # Dict{Int64, GaiaEpochSource}
src = read_gaia_epoch_votable(path, 3937211745905473024)   # one source (HD 114762)

# GaiaEpochSource fields: source_id, ra0, dec0, g_mag, iad::IADData
data = Data(; iad = src.iad)                   # → the standard IAD path
```

`read_gaia_epoch_votable` is a pure-Julia decoder for the VOTable BINARY2
payload (base64, variable-length per-CCD arrays); it keeps the
`used_by_agis_al == true` abscissae, converts times to MJD (TCB,
barycentric-corrected) and Δt to Julian years relative to the DR4 reference
epoch J2017.5, and returns an `IADData` with `psi = deg2rad(scan_pos_angle)`,
`parallax_factor = parallax_factor_al`, `pm_factor = Δt[yr]`. Validated to
sub-mas against ESA's `gaiasupdate` reference across all 12 pre-release
sources, and via orbit recovery (Gaia-4 b from epoch astrometry alone; the
HD 114762 b sin *i* break from joint RV + DR4 astrometry). For an *orbital*
source the AGIS `agis_source_excess_noise` reflects the unmodelled orbit
itself (it is the signal, not measurement noise) and is deliberately **not**
folded into `abscissa_err`.

### `GOSTData` — Gaia scanning law

Predicted scan epochs and angles from the Gaia Observation Forecast
Tool. GOST is a *prediction* (no measurement); it forward-models the
orbit's projection onto each transit's scan direction.

```julia
gost = GOSTData(;
    t               = transit_epochs_mjd,
    psi             = scan_angle_rad,
    parallax_factor = nothing,   # optional; default zeros
    along_scan_pos  = nothing,   # optional; default zeros
)
data = Data(; gost = gost)
```

**Auto-fetch** by name (Simbad-resolved) or by ICRS coordinates —
replays the GOST JSP form and caches the per-query CSV under
`GOST/<query-hash>.csv`:

```julia
gost = fetch_gost("eps Eri"; verbose = false)             # by name
gost = fetch_gost(53.232665, -9.458250; verbose = false)  # by RA, Dec (deg)
```

`from`/`to` bound the prediction window and must lie inside GOST's
queryable interval (advances each Gaia release; the fetcher surfaces
the server's bounds on an out-of-range request). `force_refetch = true`
re-runs the query. There is also a `load_gost(path; ...)` loader for an
already-downloaded GOST CSV.

### `G23HData` — Gaia DR2/DR3 + Hipparcos joint product

The Thompson+ 2026 G23H successor to HGCA EDR3: **five** PM
measurements (Hip, Hip→Gaia, DR2, DR3−DR2, DR3) with a full **10×10**
covariance (the DR2↔DR3 4×4 cross-block is populated because they share
transits).

```julia
g23h = G23HData(;
    epochs  = mjd_epochs((1991.25, 2004.6, 2015.0, 2015.5, 2016.0)),
    pmra    = (pmra_1, …, pmra_5),     # mas/yr
    pmdec   = (pmdec_1, …, pmdec_5),   # mas/yr
    cov     = cov10x10,                # symmetric pos-def, ordering
                                       # [pmra_1, pmdec_1, …, pmra_5, pmdec_5]
    plx     = parallax_mas,
    plx_err = parallax_err_mas,
    hip_id  = 18599,
)
data = Data(; g23h = g23h)
```

The caller assembles the 10×10 covariance (typically in the Python
sidecar reading the published catalog). UEVA, Hipparcos-IAD-jitter, and
Gaia-RV-variability terms of the full Thompson+ 2026 likelihood are not
in this Phase-1 type.

### `GaiaDR3Data` — DR3 5-parameter solution

```julia
g3 = GaiaDR3Data(;
    params = (α₀, δ₀, ϖ, μα*, μδ),   # offsets from catalog ref pos; (mas, mas, mas, mas/yr, mas/yr)
    cov    = cov5x5,                  # symmetric pos-def, ordered to match params
    t_ref  = jyear_to_mjd(2016.0),    # DR3 ref epoch (MJD 57388.0)
)
data = Data(; gaia_dr3 = g3)
```

`α₀`/`δ₀` are *offsets* from the catalog reference position (set to 0
for "PM + parallax only, no position-shift constraint"), which keeps
the joint refit well-conditioned. Pairs with `GOSTData` (the predicted
scan plan) and `IADData` to form the htof-equivalent joint
Hipparcos+Gaia likelihood. The `load_gaia_dr3` loader accepts either a
CSV path or a `(ra, dec, source_id)`-style spec.

### Epoch helpers

```julia
jyear_to_mjd(1991.25)                         # decimal Julian year → MJD
mjd_epochs((1991.25, 2004.6, 2016.0))         # tuple form (HGCA: 3, G23H: 5)
```

### orvara compatibility shims

```julia
rv_data     = load_orvara_rv("HIP27321.rv")
relast_data = load_orvara_relast("HIP27321.astro")
```

Parse orvara-format RV (`HIP000000.rv`) and relative-astrometry
(`HIP000000.astro`) text files.

## The `data` config block

When you drive Nereus through `Nereus.run_job(cfg)` (the production
entry point — see [`docs/JOB_CONFIG.md`]), the `cfg["data"]` block is
translated into a `Data` by `_build_data` (`src/runner.jl`). At least
one of `rv`, `transit_photometry`, `iad`, `gost`, `hgca`, `relastrom`,
`gaia_dr3` must be present.

### `rv` — inline values

```jsonc
"data": {
  "rv": {
    "values": {
      "bjd":        [...],
      "rv":         [...],
      "rv_err":     [...],
      "instrument": ["HARPS", "HARPS", "FEROS", ...],
      // --- optional activity indicators (any extra key) ---
      "bis":      [...],
      "bis_err":  [...],     // <name>_err → indicator error for "bis"
      "fwhm":     [...],
      "fwhm_err": [...]
    }
  }
}
```

`bjd`, `rv`, `rv_err`, `instrument` are required. **Any other key is an
activity indicator.** The runner splits them by the `<name>_err`
convention (`_split_indicator_errs`): a key ending in `_err` whose base
`<name>` is *also* present becomes that indicator's 1σ error; a lone
`<name>_err` (no matching `<name>`) is treated as an indicator value in
its own right. Instrument indices are derived from the sorted unique
instrument strings.

### `rv` — CSV

```jsonc
"data": {
  "rv": {
    "csv":            "/path/rv.csv",
    "time_col":       "bjd",          // default "bjd"
    "rv_col":         "rv",           // default "rv"
    "err_col":        "rv_err",       // default "rv_err"
    "inst_col":       "instrument",   // default "instrument"
    "indicator_cols": ["bis", "fwhm"] // default []  (these become indicators)
  }
}
```

For each `c` in `indicator_cols`, if the CSV also has a column
`"<c>_err"` it is loaded as that indicator's error (required by
`ActivityGP`); columns without a matching `_err` simply have no error.

### `transit_photometry` — list of light curves

```jsonc
"data": {
  "transit_photometry": [
    { "values": { "bjd": [...], "flux": [...], "flux_err": [...] },
      "instrument": "TESS_S39", "exposure_time": 120 },
    { "csv": "/path/lc.csv", "time_col": "bjd", "flux_col": "flux",
      "err_col": "flux_err", "instrument": "K2", "exposure_time": 1765 }
  ]
}
```

Each entry is one instrument/sector. `instrument` defaults to `"TESS"`;
`exposure_time` is in **seconds** (default `120` = TESS 2-min) and is
converted internally to per-cadence `exposure_times` in days. Instrument
indices are derived from the sorted unique instrument names across all
LCs.

### `iad` — auto-fetch or inline

```jsonc
"iad": { "hip_id": 27321 }            // auto-fetch via fetch_hip_iad
// OR
"iad": { "values": { "t": [...], "abscissa": [...], "abscissa_err": [...],
                     "psi": [...], "parallax_factor": [...], "pm_factor": [...] } }
```

### `gost` — auto-fetch or inline

```jsonc
"gost": { "ra_deg": 53.23, "dec_deg": -9.46,
          "from": "2014-07-26T00:00:00", "to": "2025-01-15T00:00:00" }
// OR
"gost": { "values": { "t": [...], "psi": [...], "parallax_factor": [...] } }
```

`from`/`to` default to the current GOST window.

### `hgca` — CSV or inline 3-epoch

```jsonc
"hgca": { "csv": "/path/HGCA.csv", "hip_id": 27321 }
// OR
"hgca": { "values": { "t": [e1,e2,e3], "pmra": [...], "pmdec": [...],
                      "cov_ep": [ [[...],[...]], ... ],  // three 2×2 matrices
                      "plx": ..., "plx_err": ..., "hip_id": 27321 } }
```

Inline `values` need exactly three epochs (Hipparcos, HG, Gaia).

### `relastrom` — CSV or inline

```jsonc
"relastrom": { "csv": "/path/relast.csv" }
// OR
"relastrom": { "values": { "t": [...], "ra_off": [...], "dec_off": [...],
                           "ra_err": [...], "dec_err": [...],
                           "planet_idx": [...] } }  // planet_idx optional, default 1s
```

### `gaia_dr3`

```jsonc
"gaia_dr3": { "params": [α₀, δ₀, ϖ, μα*, μδ], "cov": [[...],...], "t_ref": 57388.0 }
```

All three keys are required.

## External convenience loaders

For TESS and Vizier-published RVs, two convenience loaders bypass the
manual `readdlm` boilerplate:

```julia
# TESS LC CSV (bjd, flux, flux_err; NaN-stripped, normalised to 1) →
# NamedTuple (t, flux, flux_err). Optional `trim_window = (t_lo, t_hi)`.
lc = load_tess_lc("tess_lc.csv")
data = Data(; t_phot = lc.t, flux = lc.flux, flux_err = lc.flux_err)

# Vizier / paper-release RV CSV → NamedTuple
# (t, rv, rv_err, rv_inst, instruments). Default columns bjd,rv,rv_err,inst;
# override with `t_col`/`rv_col`/`err_col`/`inst_col` Symbol kwargs
# (pass `inst_col = nothing` for a single-instrument file).
r = load_vizier_rv("vizier_table.csv")
data = Data(; t_rv = r.t, rv = r.rv, rv_err = r.rv_err, rv_inst = r.rv_inst)
```

For pipeline-style multi-sector light-curve cleaning before fitting,
see [Preprocessing](preprocessing.md) — the production path passes raw
TESS data through `mask_transits` + `find_segments` + `detrend_gp` (or
`detrend_notch`) and lands in `save_lightcurve`'s normalised CSV
format, which the example RVPM scripts (`test/fit_HD18599_RVPM_*`) read
directly.
</content>
</invoke>
