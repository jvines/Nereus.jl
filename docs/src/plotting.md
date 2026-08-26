# Plotting

Nereus's plotting layer is built on CairoMakie with a consistent
theme (`nereus_theme()`) across all figures. Every function returns
the `Figure` object (so you can tweak in REPL) AND saves to disk if
you pass an `output` directory (or, for some plotters, a `filename`).

PNG is always written; pass `save_pdf=true` for a publication-ready
PDF companion. Nereus is on version `0.2.0`.

## Driving plots from `run_job`

In a production pipeline you almost never call the plotters directly —
you list them in `cfg.output.plots` and let
[`Nereus.run_job`](../JOB_CONFIG.md) dispatch them. The plot surface is the
set of names in `_KNOWN_PLOTS` (`src/runner.jl`); anything else is a
schema error. Every plot is **data-aware**: each one silently no-ops
(returns `nothing`, nothing written) if the data/fit it needs is
absent, so you can list a superset and only the applicable figures get
produced.

```jsonc
"output": {
  "plots": [
    "rv_timeseries", "rv_components", "rv_phasefold",
    "pm_timeseries", "pm_phasefold", "transit_overlay",
    "rm_anomaly", "ttv_oc",
    "orbit_skyplane", "relastrom_timeseries", "relastrom_residuals",
    "hgca_pm_residuals", "g23h_residuals", "iad_residuals",
    "pm_anomaly", "rv_astrom_phasefold",
    "activity_gp_latent", "activity_gp_decomposition",
    "corner", "trace", "histograms", "posteriors",
    "transdim_occupancy"
  ],
  "plot_kwargs": { "bf_cutoff": 10.0, "subtract_gp": true },
  "save_pdf": false
}
```

### The full `_KNOWN_PLOTS` surface

| name | needs | function | output (under `<out_dir>/plots/`) |
|------|-------|----------|-----------------------------------|
| `rv_timeseries`            | RV data                         | `plot_rv_timeseries`            | `models/rv_timeseries.png` |
| `rv_components`            | RV data                         | `plot_rv_components`            | `models/rv_components.png` |
| `rv_phasefold`             | RV data                         | `plot_rv_phasefold` (per planet)| `models/rv_phased_K*.png` |
| `pm_timeseries`            | photometry                      | `plot_pm_timeseries` (per inst) | `models/pm_timeseries_*.png` |
| `pm_phasefold`             | photometry                      | `plot_pm_phasefold` (per planet)| `models/pm_phased_*_K*.png` |
| `transit_overlay`          | photometry                      | `plot_transit_overlay_fit`      | `models/transit_overlay_K*.png` |
| `rm_anomaly`               | a planet with an `*_RM` mode    | `plot_rm`                       | `models/rm_anomaly_K*.png` |
| `ttv_oc`                   | photometry                      | `plot_ttv_oc`                   | `models/ttv_oc.png` |
| `orbit_skyplane`           | `data.relastrom`                | `plot_orbit_skyplane` (per planet)| `models/orbit_skyplane_K*.png` |
| `relastrom_timeseries`     | `data.relastrom`                | `plot_relastrom_timeseries`     | `models/relastrom_timeseries_K*.png` |
| `relastrom_residuals`      | `data.relastrom`                | `plot_relastrom_residuals`      | `models/relastrom_residuals_K*.png` |
| `iad_residuals`            | `data.iad`                      | `plot_iad_residuals`            | `models/iad_residuals.png` |
| `hgca_pm_residuals`        | `data.hgca`                     | `plot_pm_residuals` (per planet)| `models/hgca_pm_residuals_K*.png` |
| `g23h_residuals`           | `data.g23h`                     | `plot_g23h_residuals`           | `models/g23h_residuals.png` |
| `pm_anomaly`               | `data.gost`                     | `plot_pm_anomaly` (per planet)  | `models/pm_anomaly_K*.png` |
| `rv_astrom_phasefold`      | RV **and** astrometry           | `plot_rv_astrom_phasefold` (per planet)| `models/rv_astrom_phasefold_K*.png` |
| `activity_gp_latent`       | an `ActivityGP` noise model     | `plot_activity_gp_latent`       | `activity_gp_latent.png` |
| `activity_gp_decomposition`| an `ActivityGP` noise model     | `plot_activity_gp_decomposition`| `activity_gp_decomposition.png` |
| `corner`                   | any chain                       | `plot_corner`                   | `corner.png` |
| `trace`                    | any chain                       | `plot_trace`                    | `traces/*.png` |
| `histograms`               | any chain                       | `plot_histograms`               | `histograms/*.png` |
| `posteriors`               | any chain                       | `plot_posteriors`               | `posteriors/*.png` |
| `posteriors_raw`           | `:lp` column                    | `plot_posteriors_raw`           | `posteriors/raw/*.png` |
| `posteriors_parameters`    | `:lp` column                    | `plot_posteriors_parameters`    | `posteriors/parameters/*.png` |
| `posteriors_histograms`    | any chain                       | `plot_posteriors_histograms`    | `posteriors/histograms/*.png` |
| `traces_grouped`           | any chain                       | `plot_traces_grouped`           | `traces/*.png` |
| `transdim_occupancy`       | `:n_planets` column (trans-dim) | `plot_transdim_occupancy`       | `transdim/occupancy.png` |
| `auto`                     | —                               | data-driven selection (below)   | (union of the above) |

!!! note "Empty `plots` makes nothing"
    If `output.plots` is omitted or empty, **no figures are produced**.
    There is no implicit auto-selection — list `"auto"` (or explicit
    names) to get plots.

### `plot_kwargs` and `save_pdf`

`output.plot_kwargs` is a single dict forwarded (splatted) to *every*
plotter, so any keyword a plotter accepts is settable there
(`bf_cutoff`, `subtract_gp`, `n_draws`, `credmass`, `figsize`, …).
Kwargs a given plotter doesn't accept are simply ignored by the ones
that do — but a kwarg that *no* plotter on your list accepts will make
that plotter throw (caught and logged as a warning, the run
continues). Keep `plot_kwargs` to options common across the plots you
request, or split runs.

`output.save_pdf` (top-level) is the single toggle for PDF companions:
when `true`, every plot writes a `.pdf` next to its `.png`. An explicit
`plot_kwargs.save_pdf` overrides it per-run. Default is PNG only.

`ttv_oc`, `transdim_occupancy`, and the `posteriors_*`/`traces_grouped`
plots only honour a `save_pdf` kwarg (not the full `plot_kwargs`
splat); `ttv_oc` additionally honours an explicit `plot_kwargs.planet_b_k`.

### `"auto"` — data-driven selection

`"auto"` builds the plot set from what's actually in `data`/`params`
(each individual plot still no-ops if its slice is missing).
The selection logic (`_dispatch_plot(..., "auto", ...)`):

- Always: `corner`, `posteriors_raw`, `posteriors_parameters`,
  `posteriors_histograms`, `traces_grouped`.
- `:n_planets` column present → `transdim_occupancy`.
- RV data present → `rv_timeseries`, `rv_phasefold`.
- RV data **and** (≥ 2 planets **or** a smooth-activity noise model —
  `ActivityGP` / `CovarianceNoise` GP) → `rv_components`.
- Photometry present → `pm_timeseries`, `pm_phasefold`, `ttv_oc`,
  `transit_overlay`.
- `data.relastrom` → `orbit_skyplane`, `relastrom_timeseries`,
  `relastrom_residuals`.
- `data.iad` → `iad_residuals`; `data.hgca` → `hgca_pm_residuals`;
  `data.g23h` → `g23h_residuals`; `data.gost` → `pm_anomaly`.
- RV **and** astrometry → `rv_astrom_phasefold`.
- any `*_RM` planet mode → `rm_anomaly`.
- any `ActivityGP` noise model → `activity_gp_latent`,
  `activity_gp_decomposition`.

PPC (`ppc.png`/`ppc.json`), detection limits
(`detection_limits.png`/`.nc`), and LOO/WAIC are **separate**
`output` toggles, not entries in `plots` — see [`run_job`](../JOB_CONFIG.md).

---

## RV channel

### `plot_rv_timeseries(chains, params, data; output, fmt, save_pdf, figsize, bf_cutoff, show_keplerian, show_gp, show_total, show_activity, decompose)`

RV observations over time with the best-fit model overlay and a
data-minus-model residual panel (3:1 data/residual layout).
Multi-instrument data shown with per-instrument colours/markers.

The model is conditioned on the **winning** model: for trans-dim
chains only samples with `N_p == modal_np` and the modal noise-active
pattern are used. Within that, the EMPEROR "best-fit cluster" is kept
(samples within `ln(bf_cutoff)` of max `lp`; default `bf_cutoff=10`,
Jeffreys "strong"), and the reference curve is the max-`lp` consistent
θ. The `show_*` toggles add/hide the Keplerian, GP, activity, and Total
overlays; `decompose=true` switches to the component-decomposition view
(see `plot_rv_components`).

```julia
plot_rv_timeseries(chains, params, data; output = "results/")
```

Writes `models/rv_timeseries.png` into `output`. Pass `output=nothing`
to skip saving.

!!! note "SB2 double-lined fits"
    When the data carry a secondary component (`rv_comp == 2`),
    `plot_rv_timeseries` / `plot_rv_components` (and `run_job`'s
    `rv_timeseries` / `rv_components`) automatically route to a
    **component-coloured** rendering (primary = cyan, secondary =
    magenta on the `:cool` map) that overlays **both** total curves —
    primary (γ + K_A + planet) and secondary (γ − K_B). Saved as
    `models/rv_sb2_timeseries.png`. Likewise `plot_rv_phasefold` on the
    binary slot draws the classic **double-lined** fold (primary +K_A vs
    secondary −K_B, crossing at γ; `models/rv_sb2_binary_fold_P<k>.png`),
    and on a circumprimary planet slot folds only the primary points.

### `plot_rv_components(chains, params, data; output, fmt, save_pdf, figsize, bf_cutoff, show_keplerian, show_gp, show_total, show_activity)`

**RV component decomposition** time-series (new). Draws the raw
(γ-subtracted) RV with each Keplerian, the activity model, and their
Total as separate dense curves, plus a `data − Total` residual panel.
Distinct from `plot_rv_timeseries` (which is the
activity-decorrelated view): this one keeps the raw data and exposes
the model pieces. The activity curve is the GP/AGP latent in
identified m s⁻¹ on the real data (an `ActivityDecorrelation` model
contributes no time-only curve). Short-period Keplerians read as a
fast comb here by construction — their shape is the phase-fold's job.
Internally calls `plot_rv_timeseries(...; decompose=true)`.

Writes `models/rv_components.png`.

### `plot_rv_phasefold(chains, params, data; planet, output, fmt, save_pdf, figsize, n_draws, bf_cutoff, credmass, subtract_gp, robust_ylim)`

RV data phase-folded on one planet's period, all other Keplerians
subtracted. Points are colour-coded by observation time (`:cool`
colormap) with a colorbar; matches EMPEROR's `paint_fold_rv`
(default `n_draws=60000` model/CI draws). `credmass` (default `0.85`)
sets the credible-band mass; `subtract_gp=true` removes the GP/activity
contribution; `robust_ylim` clips outliers from the y-range.

For trans-dim chains a requested `planet > modal_np` is **not** in the
winning model, so it's skipped with a warning.

`run_job` loops `k = 1:max_kplanet`, writing `models/rv_phased_K<k>.png`.

---

## Photometry channel

### `plot_pm_timeseries(chains, params, data; output, fmt, save_pdf, figsize)`

Transit photometry over time with the median-θ transit model and a
`data − model` residual panel (in ppm), **one figure per instrument**
(`models/pm_timeseries_<inst>.png`). Density-aware marker alpha keeps
the dips visible even on dense (>20 000-point) TESS baselines. Y-axes
are snugged to the actual data range so transits don't get swallowed by
white space. The model is drawn as a small-marker "model trail" at the
data cadences (line overlays collapse to pixel-wide fence bars over a
long span of short transits).

### `plot_pm_phasefold(chains, params, data; planet, output, fmt, save_pdf, figsize, n_bins, credmass, phase_window)`

Transit data phase-folded on one planet's period, median-θ transit
model overlaid, with a residual panel. `phase_window` defaults to
`(-0.025, 0.025)` — right for short-period hot-Jupiter / sub-Neptune
transits; widen for long-duration or grazing transits. `n_bins` (auto
if `nothing`) controls the binned overlay; `credmass` (default `0.85`)
the band. `run_job` loops planets → `models/pm_phased_*_K<k>.png`.

### `plot_transit_overlay_fit(chains, params, data; planet, half_window, min_pts, output, fmt, save_pdf)`

**Per-transit light-curve gallery** (new). One panel per individual
transit of the transiting planet, each showing that transit's
photometry plus the posterior-median transit model, with x = hours
from `Tc`. This is per-transit QC: it catches a single bad transit
(spot crossing, partial/grazing, systematic) that the stacked
phase-fold averages away. `planet` defaults to the first transiting
planet (`has_pm` mode); `half_window=0.18` days sets the per-panel
zoom; transits with fewer than `min_pts=5` points are dropped.

Writes `models/transit_overlay_K<k>.png`.

---

## Rossiter-McLaughlin

### `plot_rm(chains, params, data; planet, output, fmt, save_pdf, figsize, bf_cutoff, n_draws)`

**RM in-transit RV anomaly** (new) for a fit that includes the
Rossiter-McLaughlin model (`src/rm.jl`, Hirano+ 2011). Requires a
planet whose data-source mode carries RM — one of `RVPM_RM`,
`RVPMAS_RM`, `RVPM_RM_R`, `RVPMAS_RM_R` (the `*_RM_R` variants treat
the RM/transit on a separate reduction). It picks the first
RM-enabled planet (override with `planet`) and throws if none.

The RM model adds two fit parameters:
`v_sin_i_star` (stellar projected rotation, **m s⁻¹**;
default prior `LogUniform(500, 100000)`) and one
`lambda_k<k>` per RM planet (sky-projected obliquity λ, **rad**;
default prior `Uniform(-π, π)`).

The plot:

- **Top**: RV with the Keplerian + systemic removed (= RM signal +
  noise), folded on the RM planet and zoomed to the transit window
  (x = hours from mid-transit), with the posterior RM model curve and a
  1/2/3σ band from `n_draws=400` posterior draws. The amplitude is set
  by `v·sin i_*`; the shape encodes λ.
- **Bottom**: residual after the *full* model (RM included).

The RM-free baseline is the same θ with `v_sin_i_star = 0`, so it falls
straight out of `rv_predictions`. Because `rv_predictions` now includes
the RM term (`src/likelihood.jl`), the PPC, residuals, and all RV plots
are RM-consistent for these fits.

Writes `models/rm_anomaly_K<k>.png`.

---

## Astrometry channel

### `plot_orbit_skyplane(chains, params, data; n_draws, planet_idx, output, fmt, save_pdf, figsize, bf_cutoff)`

Sky-plane `(ΔRA·cos δ, Δδ)` overlay of the relative-astrometry data
with the posterior-median orbit and an optional faint fan of `n_draws`
posterior-sampled orbits (`n_draws=0` ⇒ no fan, the default). The host
star sits at the origin (star glyph); the periastron direction is
annotated; the x-axis is reversed (north up, east left). `bf_cutoff`
default `5.0`. No-ops if `data.relastrom` is `nothing`.

### `plot_relastrom_timeseries(chains, params, data; n_draws, planet_idx, output, fmt, save_pdf, figsize, bf_cutoff)`

Two-panel **separation ρ [mas]** (top) and **PA [deg]** (bottom) vs MJD
— the canonical companion to `plot_orbit_skyplane`'s sky map for
visual-orbit papers. Observed epochs with errorbars from RA/Dec error
propagation (including correlation); posterior-median orbit as a solid
line plus `n_draws=100` faint posterior-fan draws. No-op if
`data.relastrom` is `nothing` (or no epochs for `planet_idx`).

### `plot_relastrom_residuals(chains, params, data; planet_idx, output, fmt, save_pdf, figsize, bf_cutoff)`

`(Δρ, ΔPA)` residual diagnostic — `Δρ = ρ_obs − ρ_med` [mas] (top) and
`ΔPA = PA_obs − PA_med` [deg, wrapped to (−180, 180]] (bottom) vs MJD,
using the posterior-median orbit. Errorbars from per-epoch RA/Dec
propagation; dashed zero reference per panel.

### `plot_iad_residuals(chains, params, data; output, fmt, save_pdf, figsize, bf_cutoff)`

Two-panel along-scan residual diagnostic for Hipparcos **IAD** data.
Per transit `j`, the residual is
`abscissa_j − Δη_orbit_j(θ_med) − Xⱼᵀ q_opt`, where `Δη_orbit_j` is the
orbit-induced along-scan reflex at median θ and `q_opt` is the analytic
best-fit 5-parameter catalog correction (same Cholesky factorization as
`iad_log_likelihood`). Per-transit σⱼ as errorbars. Top: residual vs ψ
(scan PA, rad); bottom: residual vs MJD. No-op (empty `Figure`) if
`data.iad` is `nothing` or has fewer than 5 transits.

### `plot_pm_residuals(chains, params, data; n_draws, planet_idx, output, fmt, save_pdf, figsize, bf_cutoff)`

**HGCA** proper-motion observed-vs-modelled at the three reference
epochs (Hipparcos / Hipparcos–Gaia / Gaia). Per panel: the observed
`(μ_α*, μ_δ)` with its 1σ within-epoch covariance ellipse, a cloud of
`n_draws=100` posterior-drawn reflex-PM points, and the median model PM
(raw reflex, barycentre not re-marginalized — visual inspection only).
A summary shows the median orbit's χ² against the three HGCA points.
Requires `data.hgca`. Dispatched as the `hgca_pm_residuals` plot,
written `models/hgca_pm_residuals_K<k>.png`.

### `plot_g23h_residuals(chains, params, data; output, fmt, save_pdf, figsize, bf_cutoff)`

Per-epoch PM residuals for **G23H** ([Thompson+ 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260200235T/abstract)) catalog data —
five epochs (Hip, HG long-baseline, Gaia DR2, DR3−DR2, DR3).
`Δμ_α* = μ_α*_obs − μ_α*_model` (top) and `Δμ_δ` (bottom) vs MJD
[mas/yr]. Model PM is the instantaneous reflex PM at each epoch
(Mode A, no window-averaging); errorbars from the diagonal of the G23H
10×10 within-epoch covariance. Requires `data.g23h`.

### `plot_pm_anomaly(chains, params, data; n_draws, planet_idx, output, fmt, save_pdf, figsize, bf_cutoff)`

Two-panel reflex proper-motion trajectory across the **GOST** scan
window: `μ_α*_reflex(t)` (top) and `μ_δ_reflex(t)` (bottom) vs MJD
[mas/yr]. Posterior-median curve in cyan, `n_draws=80` posterior fan in
orchid, HGCA / G23H observed PMs overlaid as markers with errorbars at
their tabulated epochs. Other companions' reflex contributions are
summed. Requires `data.gost`.

### `plot_rv_astrom_phasefold(chains, params, data; n_draws, planet_idx, output, fmt, save_pdf, bf_cutoff, subtract_gp, figsize)`

Joint **RV + astrometry** diagnostic for an RVAS/RVPMAS planet.
Top: RV phase-folded on `planet_idx` (other planets and per-instrument
γ subtracted) with the median Keplerian and an `n_draws=100` fan.
Bottom: sky-plane companion track over one orbital period, colour-coded
by phase (`NEREUS_CMAP`), each RV epoch overplotted at its phase
position, with relAST data as black errorbar points. Requires RV and
(relAST or HGCA) astrometry. `run_job` loops planets →
`models/rv_astrom_phasefold_K<k>.png`.

---

## Activity GP

These render only when an `ActivityGP` noise model is in the fit.
Driving `ActivityGP` from `run_job` needs both the model entry and the
indicator data with errors (see below).

### `plot_activity_gp_latent(chains, params, data; filename, save_pdf, n_draws, t_pred, figsize, rng)`

Latent activity-process figure: the posterior over `G(t)` (mean line +
1/2/3σ band) on a dense prediction grid, via `activity_gp_predict`.
`n_draws=100`, default grid auto (`t_pred=nothing`). Throws
`ArgumentError` if no `ActivityGP` is configured. `run_job` writes
`activity_gp_latent.png` (top-level under `plots/`, not `models/`).

### `plot_activity_gp_decomposition(chains, params, data; filename, save_pdf, n_draws, figsize, rng)`

Two-panel RV activity decomposition:

- **Top**: per-instrument RV scatter + the inferred activity
  contribution `Vc·G(t) + Vr·dG/dt(t)` as a 1/2/3σ band (propagated
  per draw, then quantiled — quantile of a product ≠ product of
  quantiles).
- **Bottom**: activity-corrected RV residual `rv − ⟨activity⟩` with the
  quadrature-summed measurement + activity uncertainty. What's left
  should be noise plus any Keplerian that survives activity removal.

Calls `activity_gp_decompose_rv` / `activity_gp_predict`. `run_job`
writes `activity_gp_decomposition.png`.

#### Configuring `ActivityGP` (and its indicator data)

In `cfg.model.noise_models`, add:

```jsonc
{ "kind": "ActivityGP",
  "instruments": ["HARPS"],
  "kwargs": { "channels": ["bis", "fwhm"], "use_derivative": true } }
```

`channels` is symbolised automatically; `use_derivative=true` adds the
FF′-style `dG/dt` term. `ActivityGP` **requires the indicator
errors** — supply them as:

- **inline RV `values` block**: add `<name>` and `<name>_err` arrays
  next to the RV columns, e.g. `"bis"`, `"bis_err"`, `"fwhm"`,
  `"fwhm_err"`. A lone `<name>_err` (no matching `<name>`) is treated
  as a value, not an error.
- **CSV RV block**: list the indicator columns in `indicator_cols` and
  provide a matching `<col>_err` column for each.

---

## Diagnostics

### `plot_trace(chains, params; output, fmt, save_pdf)`

One panel per unfrozen parameter, chain trace vs iteration (one trace
per walker overlaid for ensemble chains). Constant (zero-variance)
columns are skipped automatically with an info log — common for NS
top-N-by-weight chains where a parameter lands on a prior bound. Writes
`traces/*.png`.

### `plot_histograms(chains, params; output, n_bins=30, save_pdf)`

Per-parameter posterior histograms with a Gaussian-fit overlay and a
stats box (`median`, `+(84%−50%)`, `−(50%−16%)`). Writes
`histograms/*.png`.

### `plot_posteriors(chains, params; output, save_pdf)`

Composite figure: histograms + KDE overlays for every unfrozen
parameter, arranged in a grid. Writes `posteriors/*.png`.

### `plot_corner(chains, params; params_to_plot, output, fmt, save_pdf)`

Corner plot via PairPlots.jl. Pass `params_to_plot::Vector{String}` to
select parameters (omit for all unfrozen). Covariance contours + 1D
marginal histograms; constant columns (which crash Makie's histogram)
are skipped automatically. Writes `corner.png`.

```julia
plot_corner(chains, params;
    params_to_plot = ["K_k1", "sesinw_k1", "secosw_k1",
                       "gp_sigma", "gp_period", "gp_Q0"],
    output = "results/")
```

### astroEMPEROR-style posterior gallery

These four take only `output` + `save_pdf` from the dispatcher and
write into per-group subfolders (`planet`, instrument, noise, …).

- `plot_posteriors_raw` — one `lp`-vs-parameter skyline per sampled
  param: discarded draws (cyan), HPD draws (pink), a purple median
  vline and a red max-`lp` vline. **Requires an `:lp` column.** →
  `posteriors/raw/*.png`.
- `plot_posteriors_parameters` — HPD-only skyline (all points cyan)
  with median + max vlines. **Requires `:lp`.** →
  `posteriors/parameters/*.png`.
- `plot_posteriors_histograms` — HPD-region histogram per param with a
  fitted normal and distribution moments (mean, std, skew, kurtosis)
  annotated (`emperors_canvas` style). → `posteriors/histograms/*.png`.
- `plot_traces_grouped` — traces grouped into one wide multi-panel
  figure per group (per planet: P/K/sesinw/secosw/Mo stacked; instrument
  γ/jitter; noise-model params; other nuisance). De-interleaves walkers
  via `n_walkers`. → `traces/*.png`.

### `plot_transdim_occupancy(chains, params; td, noise_labels, max_kplanet, output, fmt, save_pdf, figsize)`

Three-panel trans-dimensional summary (only when `:n_planets` is in the
chain):

1. planet-count occupancy — `P(Nₚ = k | D)` bars, `k = 0 … max_kplanet`;
2. noise-model occupancy — marginal `P(model active | D)` for each
   toggleable noise model (needs `td` for the labels);
3. `Nₚ` trace — to eyeball whether the sampler actually jumps dimension
   vs sticking.

Bars coloured along `:cool`. `run_job` writes `transdim/occupancy.png`
(only `save_pdf` is forwarded from `plot_kwargs`).

!!! warning "Occupancy is model selection, not averaging"
    `transdim_occupancy` shows `P(M | D)` — model **selection**. It is
    not Bayesian model averaging; nothing is averaged over models.

---

## TTV diagrams

### `plot_ttv_oc(chains, data, params; planet_a_k, planet_b_k, tcs_predicted, filename, half_window, ngrid, n_envelope_draws, seed, kwargs...)`

End-to-end O−C diagram. Combines per-transit photometric `Tc`
measurements (profiled at the posterior median via
`measure_per_transit_tcs`) with a TTVFaster N-body envelope from
posterior draws (`ttvc_envelope`, median line + 16/84 band) in one
call.

- `planet_a_k=1` — transiting planet whose `Tc`s are measured.
- `planet_b_k=2` — perturber (non-transiting allowed). **Set to `0` to
  skip the N-body envelope and plot the data-only O−C** (measured `Tc`
  vs the linear ephemeris).
- `tcs_predicted` — linear-ephemeris grid; if `nothing`, built from the
  `P_k<a>` / `Tc_k<a>` posterior medians.

In `run_job`, `ttv_oc` defaults `planet_b_k = 2` when
`max_kplanet ≥ 2`, else `planet_b_k = 0` — so a **single-planet** fit
renders the data-only O−C rather than failing. An explicit
`plot_kwargs.planet_b_k` still wins. Writes `models/ttv_oc.png`.

```julia
plot_ttv_oc(chains, data, params;
    planet_a_k = 1, planet_b_k = 2,
    filename = "ttv_oc.png",
    half_window = 0.5, n_envelope_draws = 200)
```

### `plot_ttv_diagram` / `plot_ttv_diagram_multipanel`

Lower-level O−C builders if you already have per-transit
`(δt, σ_lo, σ_hi)` vectors.

---

## Theme

`nereus_theme()` returns the Makie theme used everywhere. Override
colours or fonts by composing on top:

```julia
with_theme(merge(nereus_theme(), Theme(fontsize = 18))) do
    plot_rv_timeseries(chains, params, data; output = "results/")
end
```

The default `pm_marker` colour is Tableau cyan (`#17becf`); the
Nereus palette (`NEREUS_COLORS`) defines `post`, `model`, `ci`,
`zero_line`, `hist_face`, and per-instrument cycling colours. The
Nereus colormap (`NEREUS_CMAP = :cool`) is used for every
time/phase-coded scatter.

---

## Preprocessing & detection diagnostics (called directly)

These are not part of the `run_job` plot surface — call them yourself
while building a pipeline. They take a `filename` (or `output`) +
`save_pdf`.

### `plot_detrending(t, flux, flux_err, result; output, …)`

Diagnostic for the [preprocessing](preprocessing.md) detrender: raw
flux + trend overlay (in-transit markers hollow) over the cleaned
residual.

### `plot_transit_phasefold(t, result; sector_id, sector_names, min_snr, xwin, output)`

Phase-folds the [`find_transits`](preprocessing.md) detrended LC at
every significant candidate period — one glued panel per candidate
(shared phase axis, zoomed to `±xwin`, transit at 0), each with a
binned-mean overlay, the duration window, an in-panel `P`/SNR label,
and robust y-zoom. Multiple sectors are colour-coded. `min_snr`
tightens the candidate set at plot time. Writes PNG + PDF by default
(dense scatter rasterized).

```julia
res = find_transits(t, flux, flux_err; detrend = :notch)
plot_transit_phasefold(t, res; sector_id = sid,
    sector_names = ["S02","S03","S29","S30","S96","S97"],
    output = "figs/hd18599_phasefold")
```

### Periodograms (pre-fit RV detection)

Paper-quality plots for `find_rv_planets` output ([Vines+ 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.2627V/abstract)
Fig 7 style: short glued panels, shared "Power" y-label, four FAP
horizontal lines with distinct B&W-safe linestyles, peak labels in
days).

- `plot_periodogram(pgram; filename, channel_label, show_faps, max_peak_labels)` —
  single-panel for any `Periodogram` subtype (`GLSPgram`, `BGLSPgram`,
  `SBGLSPgram`, `L1Pgram`); x-axis `log₁₀(Frequency(1/d))`.
- `plot_periodograms_stacked(set::PgramSet; series_order, panel_height, …)` —
  stacked RV / indicator / window-function panels with shared x-axis
  (window panel suppresses FAP lines and peak labels).
- `plot_sbgls_heatmap(pgram::SBGLSPgram; filename, max_peak_labels)` —
  `(frequency, N-obs)` heatmap of a stacked-BGLS scan; real planets
  brighten as data accumulate, aliases/activity wax and wane
  (Mortier & Collier Cameron 2017).
- `plot_gls_sectors_stacked(set::GLSSectorSet; filename, save_pdf, panel_height, …)` —
  stacked photometric-rotation GLS for [`gls_rotation`](preprocessing.md):
  one glued panel per sector, a Stitched panel, then the sampling Window.

```julia
plot_periodograms_stacked(set;
    filename       = "rv_pgrams.png",
    series_order   = ["BIS", "S-index", "FWHM"],
    panel_height   = 130,
    max_peak_labels = 3,
    fap_cutoff     = 0.1)
```

### Rotation period (ACF / PACF)

- `plot_acf(result; filename, show_pacf=true, max_peak_labels)` —
  two-panel ACF + PACF for a single-sector `find_rotation_period`
  result (statsmodels-style stems, 95% confidence band, chosen `P_rot`
  marked). `show_pacf=false` ⇒ ACF only.
- `plot_acf_sectors(res; filename, save_pdf, max_lag)` /
  `plot_pacf_sectors(res; …)` — stacked per-sector ACF (resp. PACF) for
  [`find_rotation_period_sectors`](preprocessing.md); `plot_pacf_sectors`
  needs `method=:both`/`:pacf` in the finder.

```julia
rot = find_rotation_period_sectors(t, flux, flux_err; sector_id = sid,
    sector_names = ["S02", "…"], method = :both, max_period = 15.0)
plot_acf_sectors(rot;  filename = "figs/acf_sectors.png",  save_pdf = true)
plot_pacf_sectors(rot; filename = "figs/pacf_sectors.png", save_pdf = true)
```

### Detection limits

`plot_detection_limits(res; filename, save_pdf)` renders the Bayesian
`K_lim(P)` upper-limit curve from `detection_limits(...)`. In `run_job`
this is wired through the `output.detection_limits*` toggles (default
ON for PT samplers), not the `plots` list — see [`run_job`](../JOB_CONFIG.md).

---

## Custom / composed figures

All plot functions return their `Figure`, so you can save in
non-default formats or compose multi-panel publication figures:

```julia
fig_rv = plot_rv_timeseries(chains, params, data; output = nothing)
fig_pm = plot_pm_timeseries(chains, params, data; output = nothing)
save("paper_figure.pdf", vbox(fig_rv, fig_pm))
```
