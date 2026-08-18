# Preprocessing

Nereus includes a photometric detrending pipeline for cleaning
multi-sector TESS / Kepler / ground-based light curves before passing
them to the joint RV+phot fit. The pipeline is intentionally
modular — you mask transits, identify time segments, then choose a
detrender (Savitzky-Golay, GP, notch filter, or LOCoR).

The hard rule (per the project convention): **never bin the LC**.
Photometry stays at native cadence end-to-end; the joint fit uses the
cleaned native-cadence LC as input.

## Quick start: `clean_lightcurve`

The foolproof one-call entry point. Wrap your photometry in a `LightCurve`,
call `clean_lightcurve`, and Nereus picks the detrender (`P_rot`, segments,
method) for you. The default method is `:notch` (robust across rotation
regimes, no reliable `P_rot` needed, and it flags transit-bearing cadences).

```julia
# single sector
lc = LightCurve(t, flux, flux_err)

# or many sectors — vector position = sector number; cadences are
# concatenated and time-sorted, sector identity preserved in sector_id
lc = LightCurve([sector1, sector2, sector3])   # each a LightCurve

clean = clean_lightcurve(lc)                    # method = :notch by default
clean.flux_detrended                            # detrended flux (length n)
clean.is_transit                                # Vector{Bool}; true where notch favors a transit
                                                # (always all-false for :savgol / :locor / :gp)

per_sector = sectors(clean)                     # Vector{LightCurve}, one per sector
```

Override the detrender with `method=:savgol`, `:locor`, or `:gp` (the latter
needs `kernel=...`). On a detrending failure `clean_lightcurve` warns and
returns the raw flux with `method=:failed` rather than throwing.

## Trimming cadences: `trim_mask` / `trim_lightcurve`

Sometimes you don't want to *model* a stretch of cadences — you want them
**gone**: scattered-light ramps, flares, momentum dumps, a corrupted chunk of a
sector. Pass a `trim_mask` (per-cadence `Vector{Bool}`, `true` = trim/remove) and
the flagged cadences are dropped **entirely**, *before* any detrending or
search — they do not appear anywhere in the output, **not even in its raw form**.

```julia
bad = (flux .> 1.05) .| momentum_dumps      # cadences to drop (true = trim)

# Container-level: a LightCurve with the cadences gone from EVERY field
# (t, flux, flux_err, sector_id, and any detrend arrays).
lc_clean = trim_lightcurve(lc, bad)

# Detrend with trimming in one call — raw AND detrended output exclude them.
clean = clean_lightcurve(lc; method = :notch, trim_mask = bad)
# clean.t / clean.flux / clean.flux_detrended already exclude `bad`;
# save_lightcurve(clean) therefore writes a file with no trace of them.

# Transit search with trimming.
res = find_transits(t, flux, flux_err; trim_mask = bad, detrend = :notch)
res.flux_detrended      # spans only the kept cadences
t[res.kept]             # the matching times (`kept` is a Bool over the input)
```

`trim_mask` is **distinct from `transit_mask`**: a `transit_mask` *keeps*
in-transit cadences but shields them from the detrender, whereas `trim_mask`
*removes* cadences outright. When both are supplied, overlapping cadences are
trimmed and the surviving `transit_mask` entries stay index-aligned to the
trimmed arrays. Trimming errors if the mask length ≠ cadence count or if it
would remove every cadence.

## Sigma-clipping point-outliers: `sigma_clip`

A `trim_mask` is for cadences you can point to *in time*. But extreme
point-outliers — cosmic rays, hot pixels, single-cadence glitches — sit *inside*
otherwise-normal data, where a time window makes no sense. For those, clip them
**statistically**: within each gap-separated segment, the residuals of a robust
**running median** (window in cadences) are clipped at `sigma`·(1.4826·MAD),
iterated until stable. Flagged cadences are **removed entirely** — exactly like a
`trim_mask`, and unioned with one if you pass both.

In-transit cadences are **protected** so a deep (hot-Jupiter) dip isn't mistaken
for an outlier — pass the in-transit set:

```julia
# Standalone: clip + trim a LightCurve (protects lc.is_transit when present).
lc_clean = sigma_clip_lightcurve(lc; sigma = 5.0, window = 101)

# In the detrend call — unioned with any trim_mask, applied BEFORE detrending.
# A transit_mask in the kwargs doubles as the protect set.
clean = clean_lightcurve(lc; method = :notch, sigma_clip = 5.0,
                         transit_mask = known_transits)

# In the transit search.
res = find_transits(t, flux, flux_err; sigma_clip = 5.0, detrend = :notch)

# Just the mask, to inspect or combine yourself.
out = sigma_clip_mask(t, flux; sigma = 5.0, window = 101, protect_mask = intransit)
```

`sigma_clip` takes either a bare `Real` (the σ threshold) or a `NamedTuple`
`(; sigma, window, iters)`. Defaults: `sigma = 5.0`, `window = 101` cadences,
`iters = 5`. The running-median baseline tracks slow trends/rotation, so only
genuine point-outliers stand out; keep `window` ≳ a transit duration (and/or
pass the in-transit set) so real dips survive.

## `mask_transits(t, periods, T0s; window=0.025)`

Boolean mask for in-transit cadences.

```julia
mask = mask_transits(t,
    [4.1374, 7.85],                    # planet periods (days)
    [2458354.59, 2458401.12];          # transit centres (BJD)
    window = 0.025,                    # half-width in PHASE (±2.5% of period)
)
```

`window` is a phase half-width (fraction of period), not a duration in
days — phase is `mod((t−T0)/P + 0.5, 1) − 0.5`, so the transit sits at
phase 0 and `|φ| < window` is masked.

Returns a `Vector{Bool}` of the same length as `t`. `mask[i] = true`
means cadence `i` is **in transit** (mask it out before fitting the
stellar trend).

## `find_segments(t; gap_size=0.5)`

Split a time series into contiguous segments separated by gaps larger
than `gap_size` (in the units of `t` — for TESS day-scale data,
`gap_size = 0.5` separates sectors with the typical mid-sector
download gap).

```julia
segments = find_segments(t; gap_size = 0.5)
# returns Vector{UnitRange{Int}}, one range per contiguous block
```

For TESS multi-sector data where you already have a sector ID per
cadence, use the segmentation built from that instead — the helper
`_segments_from_sector_id(sector_id)` (internal) does the right thing
when `gap_size` heuristics are unreliable.

## `detrend_savgol(t, flux, flux_err; window_length, polyorder=3, transit_mask=nothing, gap_size=0.5)`

Savitzky-Golay smoothing of `flux` with in-transit points interpolated
across (when a `transit_mask` is supplied).

```julia
res = detrend_savgol(t, flux, flux_err;
    window_length = 301,               # odd integer, window length in *points*
    polyorder     = 3,
    transit_mask  = nothing,           # Vector{Bool}; masked points interpolated
    gap_size      = 0.5,               # segment break threshold (days)
)
```

Returns a NamedTuple `(; flux_detrended, trend, segments,
transit_mask)`, where `flux_detrended = flux ./ trend` (multiplicative,
median ≈ 1) and `segments` is the per-segment partition from
`find_segments`. Note `window_length` is given in **points** (an odd
integer), not days — for TESS 2-min cadence ~301 points ≈ 10 hr, ~1501
≈ 50 hr. Fast (~1 sec on multi-sector TESS); good for slow stellar
variability that's not rotation-modulated.

Per-segment fitting is automatic via `find_segments(t; gap_size)`;
segments shorter than `window_length` fall back to a direct polynomial
fit of order `polyorder`.

## `detrend_gp(t, flux, flux_err, kernel; transit_mask=nothing, init_params=nothing, joint_segments=true, sector_id=nothing)`

Bayesian Pathfinder GP detrending — the production path for active
stars. Fits a celerite GP (SHO, Rotation, or RotationFM17) to the LC
(transit-masked cadences down-weighted, not interpolated), drawing the
trend from the posterior predictive.

```julia
res = detrend_gp(t, flux, flux_err, CeleriteRotation();
    transit_mask   = mask,             # Vector{Bool} or nothing
    init_params    = [0.04, 8.74, 3.0, 1.0, 0.2],  # kernel-order starting values
    joint_segments = true,             # one shared hyperparameter set
    sector_id      = nothing,          # per-cadence sector tag (1-indexed)
)
```

The kernel is a **positional** argument; choices:
- `CeleriteSHO()` — broad-band correlated noise, no rotation timescale.
- `CeleriteRotation()` — rotation modulation with a clear period.
- `CeleriteRotationFM17()` — [Foreman-Mackey+ 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F/abstract) celerite-1
  rotation kernel.

Returns a NamedTuple `(; flux_detrended, trend, gp_params, offsets,
segments, transit_mask)`. The detrending is **additive**
(`flux_detrended = flux − μ − offset`, with `trend = μ + 1 + offset`),
matching the GP-as-noise-model convention used elsewhere in Nereus —
for a multiplicative convention divide `flux` by `1 + trend`.
`gp_params` is a `Vector{Vector{Float64}}` of fitted hyperparameters
(length 1 when `joint_segments=true`, one per segment otherwise) and
`offsets` holds the per-segment DC levels; both are the right values to
pin the **fit-time** GP prior on when you re-fit the joint RV+phot
model.

This is the slowest detrender (~30-90 s on multi-sector TESS) but
produces the cleanest output on rotation-dominated targets like
HD 18599.

## `detrend_notch(t, flux, flux_err; window=1.0, durations, ie_frac=0.1, delta_bic=-1.0, niter=5, refine=true, gap_size=0.5, show_progress=false)`

Notch filter ([Rizzuto+ 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..224R/abstract); Kepler/K2-style) for variable
stars where the transit signal sits inside the rotation modulation. It
is **mask-free** — no periods/T0s are supplied. A local polynomial
baseline plus a trapezoidal notch is fit in a sliding window; the
BIC-favoured model per cadence decides whether a transit-shaped feature
is present, and the chosen baseline is divided out.

The notch shape is a **trapezoid** in time (`ie_frac=0` collapses it to
Rizzuto's box), and the per-window joint poly2 + notch fit is solved by an
alternating linear LSQ (linear in `(a,b,c)` for fixed depth, linear in depth
for fixed `(a,b,c)`) rather than Levenberg–Marquardt — so there is no
nonlinear-solver dependency.

```julia
res = detrend_notch(t, flux, flux_err;
    window    = 1.0,                   # sliding window width in days
    durations = [1, 2, 3, 4, 6] ./ 24, # trial transit full-widths (days)
    ie_frac   = 0.1,                   # ingress/egress fraction of half-width
    delta_bic = -1.0,                  # accept notch if BIC_null − BIC_notch ≥ this
    niter     = 5,                     # σ-clip passes per window (3.5σ)
    refine    = true,                  # second dip-excluded trend pass (see below)
    gap_size  = 0.5,                   # segment break threshold (days)
)
```

Returns a NamedTuple `(; flux_detrended, trend, depth, delta_bic,
dur_best, outlier, segments, sector_id)`, where `flux_detrended = flux ./ trend`
(transit preserved as a residual ≈ `1 − depth`) and `delta_bic` is a
per-cadence "is there a transit-shaped feature here?" statistic
(positive ⇒ notch wins). Drop `window` to ~0.5 d for fast rotators
where the rotational gradient over 1 d breaks the local poly2
assumption. Best for K-dwarfs and earlier where rotation is fast —
Savitzky-Golay would whiten the rotation signal along with the trend,
and the GP detrender works but is slower.

`refine=true` (default) runs **two trend passes**. Pass 1 detects every dip
and accumulates its in-trapezoid cadences into a global dip mask. Pass 2
re-fits each window's trend with all pass-1 dips excluded, so a transit
sitting off-centre in a flanking window can no longer bend the local poly2
(the source of the ±0.25% detrended "shoulders" around each dip); the
per-cadence trend is then smoothly interpolated across each detected-dip run
so a transit on a steep/curved baseline doesn't ring into a V/overshoot. The
detection products (`depth`, `delta_bic`, `dur_best`, `outlier`) come **only
from pass 1** (where the transit is still in the trend fit, giving the
notch-vs-null BIC full contrast). Set `refine=false` to skip pass 2 (faster,
but the detrended shoulders return). `show_progress=true` prints a per-window
progress bar.

## `detrend_locor(t, flux, flux_err; P_rot=nothing, max_donor_cycles=nothing, niter=3, clip_sigma=3.5, gap_size=0.5, sector_id=nothing, transit_mask=nothing)`

LOCoR — Locally Optimized Combination of Rotations
([Rizzuto+ 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..224R/abstract)) — for **fast** rotators. It slices the
light curve into rotation cycles of length `P_rot` and predicts each
cycle's baseline from a least-squares linear combination of the
neighbouring ("donor") cycles, then divides it out. Transits are
protected by an iterative MAD σ-clip (`niter`, `clip_sigma`) on the
per-cycle fit. Because a transit spans only a small fraction of a fast
rotation cycle, dividing by the rotation template flattens the spot
modulation while leaving the transit dip intact.

!!! warning "Mask-free by default, and the transit-self-destruction caveat"
    `detrend_locor` is **mask-free by default** and relies solely on the
    σ-clip to keep transits out of the baseline fit. This protects the
    generic case (transit at a *unique* rotation phase each cycle), but it
    does **not** protect the commensurate case: when `P_orb` is a low-order
    rational multiple of `P_rot`, the transit recurs at the **same**
    rotation phase across donor cycles, the LSQ combination learns it as
    part of the baseline, and the transit is **attenuated or destroyed**.
    Shallow transits the σ-clip doesn't flag can also be partly absorbed.

    The real defense — mirroring the reference implementation
    (`arizzuto/Notch_and_LOCoR`, `rcomb()`'s `cleanmask`) — is the optional
    `transit_mask` kwarg: a per-cadence `Bool` over the full LC (`true` =
    in-transit). It excludes in-transit points from **both** the target
    cycle's LSQ fit (pre-seeded inactive) and the **donor** curves (donors
    interpolate their rotational baseline *across* the masked transit gap),
    so the baseline becomes the clean rotational extrapolation and the dip
    survives in `flux_detrended`. In-transit points are still reported in
    `outlier`. The intended workflow: leave it mask-free for the blind
    pre-search, then pass `transit_mask` once you have a candidate
    ephemeris (the detrend→search→re-detrend-with-mask loop). Note the
    whole-LC notch fallback ignores `transit_mask` (notch is mask-free and
    takes no mask).

```julia
res = detrend_locor(t, flux, flux_err;
    P_rot            = 0.8,            # rotation period (days); auto if nothing
    max_donor_cycles = nothing,       # cap on donor cycles per target cycle
    niter            = 3,             # per-cycle MAD σ-clip passes
    clip_sigma       = 3.5,           # σ-clip threshold
    gap_size         = 0.5,           # segment break threshold (days)
    sector_id        = nothing,       # per-cadence sector tag (passthrough)
    transit_mask     = nothing,       # Vector{Bool} (true=in-transit); cleanmask
)
```

Returns a NamedTuple `(; flux_detrended, trend, phase, cycle_id,
outlier, segments, P_rot_used, used_notch)`. `trend` is the
reconstructed raw rotation baseline, `phase`/`cycle_id` are the
within-cycle phase and integer cycle index per cadence, and
`used_notch` flags cadences where LOCoR fell back to `detrend_notch`.
When `P_rot` is omitted it is estimated via
`find_rotation_period(t, flux, flux_err).P_rot`. LOCoR needs at least
three rotation cycles; when too few cycles exist (or `P_rot` is
non-finite / longer than a third of the baseline) it falls back to
notch per-segment (or whole-LC).

## `detrend_rotation(t, flux, flux_err; method=:auto, P_rot_threshold=2.0, P_rot=nothing, kwargs...)`

Auto-selecting rotation detrender that dispatches between
`detrend_locor` (LOCoR; fast rotators, `P_rot < P_rot_threshold`) and
`detrend_notch` (the workhorse for slow rotators and the safe
fallback). `method` is one of `:auto`, `:locor`, `:notch`; `:locor`
and `:notch` force the respective detrender, `:auto` routes on the
resolved `P_rot`. Any other value throws `ArgumentError`. Extra
`kwargs...` are forwarded verbatim to the chosen detrender.

```julia
res = detrend_rotation(t, flux, flux_err;
    method          = :auto,
    P_rot_threshold = 2.0,
    P_rot           = 0.8,            # pass explicitly for reliable routing
)
```

The return schema is intentionally **not** unified — it is whatever the
chosen detrender returns; both share `flux_detrended` and `trend`, so
branch on `method`/`P_rot_used` for the method-specific fields.

!!! warning "Auto-routing caveat for fast rotators"
    `find_rotation_period`'s ACF peak-finder is tuned for **slow**
    rotators and currently returns `NaN` for many fast rotators. Since
    `:auto` requires `isfinite(P_rot)` to pick LOCoR, an auto-resolved
    `P_rot` of `NaN` routes the target to **notch** even when LOCoR
    would be the right tool. For guaranteed LOCoR on a known fast
    rotator, use `method=:locor` or pass `P_rot` explicitly.

## End-to-end recipe

A typical multi-sector TESS clean for a transit fit:

```julia
using DelimitedFiles
using Nereus

# Load raw TESS LC (multi-sector)
data = readdlm("HD18599_raw_lc.csv", ','; header=true)[1]
t        = Float64.(data[:, 1])
flux     = Float64.(data[:, 2])
flux_err = Float64.(data[:, 3])
sector_id = Int.(data[:, 4])

# Mask transits (use literature ephemeris)
mask = mask_transits(t, [4.1374], [2458354.59];
                      window = 0.025)   # phase half-width (±2.5% of period)

# GP detrend (kernel is positional; sector_id drives per-sector offsets)
res = detrend_gp(t, flux, flux_err, CeleriteRotation();
    transit_mask = mask,
    sector_id    = sector_id,
)

# Save cleaned LC + detrending diagnostics + hyperparams to NetCDF
save_detrend_result("HD18599_cleaned_lc.nc", t, flux, flux_err, res;
                    method = :gp)
```

`save_detrend_result(path, t, flux, flux_err, result; method)` bundles
the detrender's NamedTuple into a single NetCDF file — `flux`,
`flux_detrended`, `trend`, method-specific per-obs diagnostics, the
`segment_*` partition, and scalar hyperparams as global attributes. It
supports `method = :savgol | :gp | :notch | :locor`. Reload with
`load_lightcurve(path)`, which returns a NamedTuple of the saved
variables plus an `attrs` Dict.

## Detrender output contract

Every detrender returns a NamedTuple sharing at least `flux_detrended`,
`trend`, and `segments` (and `outlier` for all but savgol);
method-specific extras vary
(`gp_params`/`offsets` for GP, `depth`/`delta_bic`/`dur_best` for
notch, `phase`/`cycle_id`/`P_rot_used`/`used_notch` for LOCoR). The
detrending convention is multiplicative (`flux ./ trend`) for savgol,
notch, and LOCoR, and additive (`flux − trend`) for GP. `detrend_savgol`
omits `outlier` (it has no per-cadence clipping step).

Every detrender also accepts an optional `sector_id` (a per-cadence
integer tag) and echoes it verbatim in the output NamedTuple (`nothing`
when not supplied), so sector identity survives detrending and is
persisted by `save_detrend_result`. This is metadata only — it does not
affect the detrending math (segments stay gap-based / per-orbit) — and
lets downstream code slice a detrended multi-sector concatenation back
into per-sector light curves.

## Modifying the detrenders (for maintainers)

Source layout:

| Function | File |
|----------|------|
| `mask_transits`, `find_segments`, `detrend_savgol`, `detrend_gp` | `src/preprocessing/detrend.jl` |
| `detrend_notch` (+ `_detrend_notch_segment!`) | `src/preprocessing/notch.jl` |
| `detrend_locor`, `detrend_rotation`, internals (`_locor_cycles`, `_build_olc`, `_interp_constant`, `_rcomb_cycle`, `_rcomb_cycle_clipped`, `_locor_segment!`) | `src/preprocessing/locor.jl` |
| `save_detrend_result`, `save_lightcurve`, `load_lightcurve` | `src/io.jl` |
| Tests | `test/test_locor.jl`, `test/test_locor_io.jl` (wired into `runtests.jl`) |

Include order in `src/Nereus.jl` matters: `detrend.jl` → `notch.jl` →
`find_transits.jl` → `rotation_period.jl` → `locor.jl` (LOCoR needs
`find_segments`, `find_rotation_period`, and `detrend_notch` for its
fallback). `locor.jl` has **no `module` wrapper** — it is `include`d
into the `Nereus` module, so it calls those helpers unqualified.

**Invariants any change must preserve:**
- Output NamedTuple always contains `flux_detrended`, `trend`,
  `segments`, `sector_id` (and `outlier` for all but savgol). Add new
  per-cadence diagnostics as extra fields; never rename the shared four.
- Convention is multiplicative (`flux ./ trend`) for savgol/notch/locor,
  additive (`flux − trend`) for GP. Keep `trend` as the divided-out (or
  subtracted) baseline regardless of method.
- Order is preserved: output array `i` corresponds to input `t[i]`.
- `sector_id` is pure passthrough — echo the input verbatim (`nothing`
  if unset); do **not** let it change segmentation or the fit.
- Adding a method or output field means updating `save_detrend_result`
  (method whitelist + the per-method `extras`/`final_attrs` branch) and,
  if the field should round-trip, `load_lightcurve` already surfaces any
  `obs`-dimensioned NetCDF variable by name.

**Recently completed:**

- **Optional `transit_mask` on `detrend_locor`** (the real fix for
  commensurate-period transit destruction) — threaded through
  `_locor_segment!` → `_rcomb_cycle_clipped`: donor in-transit points are
  dropped before `_build_olc` (donors interpolate across the gap) and the
  target's in-transit points are pre-seeded inactive in the per-cycle LSQ.
- **`:locor` wired into `find_transits`** — `detrend` now accepts
  `:savgol | :gp | :notch | :locor | :none`, forwarding `detrend_kwargs`
  to `detrend_locor`.

**Concrete pending changes (known gaps, with the exact edit):**

3. **Fix `find_rotation_period` fast-rotator detection.** Its ACF
   peak-finder returns `NaN` for many fast rotators (`P_rot ≲ 2 d`),
   which is why `detrend_rotation(:auto)` can't route them to LOCoR
   without an explicit `P_rot`. Fix is in the peak-detection in
   `src/preprocessing/rotation_period.jl`, not in LOCoR.
4. **Inject-recovery validation** for LOCoR — *partially done*:
   `test/test_locor.jl` now has a commensurate-period (`P_orb = 2·P_rot`)
   inject-recovery test proving mask-free LOCoR destroys the transit while
   the masked path recovers it. The remaining work is a fuller grid of
   recovered/injected depth vs `P_orb/P_rot` to bound exactly where LOCoR
   eats transits.

## Diagnostic plot

```julia
plot_detrending(t, flux, trend, flux_clean, mask;
                output = "results/HD18599_detrend.png")
```

Two-panel: top shows raw flux + trend overlay (in-transit points
hollow), bottom shows the cleaned residual.

---

## Pre-fit signal detection

The preprocessing pipeline above feeds three top-level "find me a
signal" routines. All return plain tuples / NamedTuples; pipe their
output into the corresponding plot helpers in
[`plotting.md`](plotting.md).

### `find_transits(t, flux, flux_err; method=:bls, …)`

Detects candidate transits in a light curve, chaining detrending →
BLS or TLS → harmonic dedup → physical-quantity estimates (R_p, M_p,
K) via Chen & Kipping (2017) and Lovis & Fischer (2010).

```julia
res = find_transits(t, flux, flux_err;
    method          = :bls,           # :bls ([Kovács+ 2002](https://ui.adsabs.harvard.edu/abs/2002A&A...391..369K/abstract)) | :tls ([Hippke & Heller 2019](https://ui.adsabs.harvard.edu/abs/2019A&A...623A..39H/abstract))
    detrend         = :savgol,        # :savgol | :gp | :notch | :locor | :none
    detrend_kwargs  = (window_length = 301,),  # NamedTuple forwarded verbatim to the chosen detrender — keys must match its signature (e.g. window_length for savgol, window for notch)
    trim_mask       = nothing,        # Vector{Bool}; true = drop this cadence ENTIRELY (see "Trimming cadences")
    sigma_clip      = nothing,        # Real σ or (; sigma, window, iters); statistical point-outlier clip before search
    period_min      = 0.5, period_max = 30.0,
    n_periods       = 10_000,         # FLOOR on the grid size (see below)
    oversample      = 2.0,            # frequency-grid oversampling
    max_n_periods   = 2_000_000,      # cap on the auto-sized grid
    durations       = [0.01, 0.02, 0.04, 0.08],  # fractional-duration grid (only when scale_durations=false)
    scale_durations = true,           # set BLS box widths per-period from expected duty cycle q ∝ P^(-2/3)
    local_baseline  = false,          # measure depth against bins flanking each box, not the global mean
    flank_frac      = 1.0,            # flank width as a fraction of the box width (local_baseline only)
    n_phase_bins    = 100,            # BLS phase-bin count
    n_peaks         = 20,             # raw peaks pulled from the periodogram before dedup
    score_threshold = 7.0,
    max_candidates  = 5,
    harmonic_tol    = 0.03,           # fractional tol for marking a peak an m/n harmonic of a stronger one
    harmonic_max_order = 5,           # check m·P/n for m,n ∈ 1:order
    R_s = 1.0, M_s = 1.0,
    tls_options     = (;),            # NamedTuple forwarded into TLSOptions (method=:tls only)
)
# → NamedTuple { periods, t0s, depths, durations, snr,
#                R_p_earth, M_p_earth, K_ms,
#                detrend_result, flux_detrended, flux_err_detrended, kept }
```

Output fields worth noting:
- `t0s` — mid-transit times as **real epochs** in the same time system as
  `t` (the transit nearest the baseline midpoint), not a phase.
- `durations` — best-fit transit durations [days].
- `snr` — detection significance (BLS SNR / TLS SDE). *(Renamed from
  `scores`.)*
- `R_p_earth`, `M_p_earth`, `K_ms` — per-candidate physical estimates: radius
  from depth (`R_s`), mass from the Chen & Kipping (2017) M–R relation, and the
  RV semi-amplitude from Lovis & Fischer (2010).
- `detrend_result` — the full NamedTuple from the chosen detrender (`nothing`
  when `detrend = :none`); branch on `detrend` for its method-specific fields.
- `flux_detrended` — the detrended flux (NaN where a detrender dropped a
  cadence). With no `trim_mask`/`sigma_clip` it is at **full input length** and
  index-aligns with `t`; otherwise it spans only the kept cadences, so fold
  against `t[res.kept]` — see [`plot_transit_phasefold`](plotting.md#plot_transit_phasefold).
- `flux_err_detrended` — per-cadence fractional errors, re-levelled and aligned
  1:1 with `flux_detrended` (same kept-space), for error bars on the folded data.
- `kept` — `Vector{Bool}` of length == the original input, `true` for cadences
  that survived `trim_mask`/`sigma_clip` (all-`true` when neither given).
  `t[res.kept]` index-aligns with `flux_detrended`.
- `trim_mask`/`sigma_clip` — cadences to drop entirely before detrend + search
  (see [Trimming cadences](#trimming-cadences-trim_mask-trim_lightcurve) and
  [Sigma-clipping point-outliers](#sigma-clipping-point-outliers-sigma_clip));
  distinct from a `transit_mask`, which keeps but shields cadences. A
  `transit_mask` passed through `detrend_kwargs` is trimmed in lockstep and
  doubles as the `sigma_clip` protect set.

The BLS box widths default to a **physical** duty-cycle grid:
`scale_durations=true` sets the per-period box width from the expected duty
cycle `q ∝ P^(-2/3)` (computed from `M_s`,`R_s` → ρ⋆ → a/R⋆), which catches
wide-duty-cycle ultra-short-period transits a fixed narrow fractional grid
would alias onto harmonics. Set `scale_durations=false` to use the literal
`durations` fractional grid instead. `local_baseline=true` measures each
candidate's depth against the bins flanking its box (`flank_frac·box_bins` per
side) rather than the global mean — robust to a wandering/under-detrended
baseline at a small SNR cost on a flat baseline, so it is opt-in.

The BLS period grid is **uniform in frequency** with resolution sized to the
baseline (Ofir 2014): `df = min(durations)/(oversample·baseline)`, count
`max(n_periods, optimal)`. So `n_periods` is a *floor* — long (multi-sector)
baselines auto-refine instead of under-resolving the true period onto an
alias/harmonic, while short baselines stay at ~`n_periods`. Detrending is
applied **per gap-segment** and each segment is re-levelled to ≈1 (errors
rescaled to fractional) before stitching, so raw per-sector flux levels don't
bias the periodogram.

`detrend_kwargs` is a `NamedTuple` forwarded **verbatim** to the chosen
detrender, so its keys must match that detrender's signature (a wrong
key throws). `detrend = :notch` ([Rizzuto+ 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..224R/abstract)) is the right choice
for active stars — see `find_rotation_period` below to seed its window
width. `detrend = :locor` ([Rizzuto+ 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..224R/abstract)) is available for fast
rotators; forward `detrend_kwargs = (P_rot = ...,)` (and a `transit_mask`
once you have an ephemeris).

### `find_rotation_period(t, flux, flux_err; …)`

Estimates the stellar rotation period via the autocorrelation
function ([McQuillan+ 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.432.1203M/abstract), 2014) plus the partial autocorrelation
for AR-noise disambiguation. Use it to pre-seed
`detrend_notch(window = 1.2 * P_rot)`:

```julia
rot = find_rotation_period(t, flux, flux_err;
    min_period   = 0.5, max_period = 15.0,
    smooth_sigma = nothing,           # auto
    bin_cadence  = nothing,           # auto: coarse, ≈ max_period/100 (see below)
    method       = :both,             # :acf | :pacf | :both
    n_peaks      = 5,
)
# → NamedTuple { P_rot, P_rot_err, lags, acf, pacf,
#                peaks, bin_cadence, n_bins, threshold }
```

Notes on correctness (the ACF/PACF are a rotation-scale diagnostic):
- **Single continuous series only.** It bins `[t[1], t[end]]` onto a uniform
  grid and linearly interpolates gaps — so feeding a *stitched multi-sector*
  LC interpolates across the inter-sector gaps and corrupts the ACF. Use
  `find_rotation_period_sectors` (below) for multi-sector data.
- **Coarse internal binning.** `bin_cadence` defaults to `max(median Δt,
  max_period/100)`, *not* the raw cadence — at TESS 20-s cadence a native grid
  has ~10⁵ lags (unstemmable) and the PACF's only non-zero term sits at lag 1
  ≈ 20 s. Coarse binning is the McQuillan/statsmodels standard and leaves
  rotation (≫ bin) intact. This is the diagnostic's internal grid only.
- **Peak picking** takes the tallest ACF peak *after the first dip below
  background* (McQuillan+ 2013), skipping the high-autocorrelation shoulder
  near lag 0. `max_period` defaults to ¼ of the baseline, so set it explicitly
  if `P_rot` may exceed that (e.g. a single 27-d TESS sector → default 6.75 d
  would miss an 8.7-d rotator).

Feed the result to [`plot_acf`](plotting.md#plot_acf) for the single-sector
two-panel ACF + PACF diagnostic.

### `find_rotation_period_sectors(t, flux, flux_err; sector_id=nothing, …)`

Runs `find_rotation_period` **independently per sector** (no stitched/combined
ACF — the ACF can't be combined across gaps). Sectors come from `sector_id`
(per-cadence labels) or, if absent, contiguous `find_segments` (gap >
`gap_size`). Returns `(; names, results, P_rot)` — `results[i]` is the full
per-sector NamedTuple. Plot with
[`plot_acf_sectors`](plotting.md#plot_acf_sectors) /
[`plot_pacf_sectors`](plotting.md#plot_pacf_sectors).

```julia
rot = find_rotation_period_sectors(t, flux, flux_err;
    sector_id    = sid,                       # per-cadence sector number
    sector_names = ["S02","S03","S29","S30","S96","S97"],
    method       = :both, max_period = 15.0,
)
```

### `gls_rotation(t, flux, flux_err; sector_id=nothing, …)`

Photometric rotation **GLS** (Lomb-Scargle on each sector's flux **plus the
stitched series** + the sampling window). Unlike the ACF, LS needs no uniform
binning and handles the inter-sector gaps natively, so the stitched panel is
valid and gives the finest frequency resolution. All panels share one
frequency grid. Returns a `GLSSectorSet` for
[`plot_gls_sectors_stacked`](plotting.md#plot_gls_sectors_stacked).

```julia
gset = gls_rotation(t, flux, flux_err;
    sector_id    = sid, sector_names = ["S02", …],
    period_min   = 0.5, period_max = nothing,   # default = stitched baseline
    samples_per_peak = 20,
)
# → GLSSectorSet { names, sectors, stitched, window, n_obs, baseline_days }
```

### `find_rv_planets(t, rv, rv_err; indicators=Dict(), method=:gls, …)`

Computes a periodogram of the RV channel + every supplied activity
indicator + a sampling-pattern window function on the **same**
frequency grid. Returns a `PgramSet` ready for
[`plot_periodograms_stacked`](plotting.md#plot_periodograms_stacked)
(the paper-style RV/indicators/window stack from [Vines+ 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.2627V/abstract) Fig 7).

```julia
set = find_rv_planets(t, rv, rv_err;
    indicators = Dict(
        "BIS"     => (bis, σ_bis),
        "FWHM"    => (fwhm, σ_fwhm),
        "S-index" => (s_idx, σ_s_idx),
    ),
    method           = :gls,         # :gls | :bgls | :sbgls | :l1
    period_min       = 1.0,
    period_max       = nothing,      # default = baseline
    n_freqs          = nothing,
    samples_per_peak = 50,
    window_def       = :gls_tt,      # :gls_tt | :unit_signal
    fap_levels       = [0.1, 0.05, 0.01, 0.001],
    max_candidates   = 5,
)
# → PgramSet { rv::Periodogram, indicators::Vector{Pair},
#              window::Periodogram, method, n_obs, baseline_days }
```

Backends:
- `:gls`   — `LombScargle.jl` with Cumming 2004 analytic FAPs
- `:bgls`  — Bayesian GLS ([Mortier+ 2015](https://ui.adsabs.harvard.edu/abs/2015A&A...573A.101M/abstract))
- `:sbgls` — Stacked BGLS, 2-D map vs growing N-obs (Mortier &
  Collier Cameron 2017)
- `:l1`    — LASSO-sparse compressed-sensing periodogram ([Hara+ 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.1220H/abstract))

!!! note "Post-fit residual periodogram (diagnostics)"
    The post-fit **RV residual** periodogram computed inside the science
    summary (`src/diagnostics/ppc.jl`) is a different, bounded code path from
    the pre-fit `find_rv_planets` above. On long-baseline RV it now **caps the
    frequency grid** (`clamp(…, 64, 20_000)` freqs) and uses the **analytic**
    FAP (`fap_method = :analytic`) instead of the 1000-resample bootstrap —
    the bootstrap over a dense grid was taking hours and blowing `run_job`'s
    wall-clock. The capped run surfaces `rv_residual_top_peak_*` /
    `rv_residual_top_peak_fap` in the run's science summary.

## Related: model-level features that affect the cleaned-data fit

These live on other pages but are listed here because they consume the
photometry this pipeline produces (or change how a residual/PPC looks):

- **Rossiter–McLaughlin.** Planet modes `RVPM_RM` / `RVPMAS_RM`
  (Hirano+ 2011 leading-order) and the disk-integrated "Reloaded" variants
  `RVPM_RM_R` / `RVPMAS_RM_R` (Cegla+ 2016) add an in-transit RV anomaly from
  the params `v_sin_i_star` (m/s) and per-planet `lambda_k<k>` (sky-projected
  obliquity, rad). The model lives in `src/rm.jl`; `rv_predictions` now
  includes the RM term (`src/likelihood.jl`), so PPC, residuals, and RV plots
  are RM-consistent. See the planet-modes / job-config pages, and the
  `rm_anomaly` plot (`plot_rm`) in [`plotting.md`](plotting.md).
- **ActivityGP from `run_job`.** A `noise_models` entry
  `{kind:"ActivityGP", instruments:[...], kwargs:{channels:["bis","fwhm"],
  use_derivative:true}}` couples activity indicators into the RV GP. Indicator
  data is supplied alongside the RV channel: in an RV `values` block add
  `<name>` + `<name>_err` arrays (e.g. `bis`/`bis_err`, `fwhm`/`fwhm_err`); in
  a CSV RV block use `indicator_cols` + matching `<col>_err` columns.
  ActivityGP **requires** the indicator errors. See the noise-models /
  job-config pages.
- **New diagnostic plots.** `rm_anomaly` (`plot_rm`), `transit_overlay`
  (`plot_transit_overlay_fit` — per-transit QC gallery), and `rv_components`
  (`plot_rv_components` — RV model decomposition, distinct from
  `rv_timeseries`). `ttv_oc` also renders for single-planet fits now (data-only
  O–C, `planet_b_k=0`). See [`plotting.md`](plotting.md).

Nereus is at version **0.2.0**.
