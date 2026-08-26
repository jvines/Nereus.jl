# Diagnostics & I/O

This page documents the **post-fit diagnostics** Nereus emits, both as
standalone functions and — more importantly for production pipelines —
as the automatic stages `Nereus.run_job` runs after sampling. Every
stage is **fail-soft**: a diagnostic that throws is logged with `@warn`
and recorded in `summary.json` with a `status` of `"failed"`/`"skipped"`,
but never aborts the job or alters the chains. (Nereus is at version
`0.2.0`.)

## What `run_job` runs after sampling

After the sampler returns and chains are saved, `run_job` runs five
post-fit stages in this order (`src/runner.jl`, `run_job` body):

1. **Plots** (`_make_plots`) — model, posterior, and diagnostic figures.
2. **PPC** (`_run_ppc!`) — posterior predictive checks → `ppc.png` +
   `ppc.json`, summary under the `ppc` key. Default **on**.
3. **Detection limits** (`_run_detection_limits!`) — Bayesian `K_lim(P)`
   → `detection_limits.png` + `detection_limits.nc`, summary under
   `detection_limits`. Default **on for PT samplers only**.
4. **PSIS-LOO / WAIC** (`_run_loo!`) — predictive cross-validation,
   summary under `loo`. Default **on** (skipped for GP models).
5. **Fit health** (`_run_fit_health!`) — silent-wrong structural guard,
   summary under `fit_health`. Default **on**.

All five are toggled through the top-level `output` block of the job
config. Defaults and per-stage keys are documented in each section
below.

## Convergence diagnostics

These are the *interactive* convergence tools. In `run_job`, the
R̂ / ESS gate is run automatically (and ensemble-aware) as part of the
fit-health guard — see [Fit health — the silent-wrong guard](#fit-health-the-silent-wrong-guard)
for the `convergence` check and how it lands in `summary.json`.

### `print_ess_rhat(chains, params)`

Effective sample size + Gelman-Rubin R̂ per parameter, computed via
MCMCChains. Pretty-printed table.

```julia
print_ess_rhat(chains, params)
```

Rules of thumb:
- ESS ≥ 100 per parameter for headline numbers, 1000+ for tight
  posteriors.
- R̂ ≤ 1.01 indicates convergence across multiple chains. > 1.05
  means run longer; > 1.1 suggests sampling failure.

For single-chain runs (NUTS / NS top-N fallback) R̂ is undefined and
the column shows `—`.

### `polynomial_stein_discrepancy(chains, target; degree=2)`

Polynomial Stein Discrepancy ([Srinivasan+ 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv241205135S/abstract), arXiv:2412.05135) —
gradient-free convergence diagnostic for MCMC chains. Returns a
scalar; smaller is better-converged. Useful for high-d targets where
R̂ is noisy.

```julia
psd = polynomial_stein_discrepancy(chains, target; degree = 2)
```

Bootstrap test for "is this PSD value statistically significant":

```julia
psd_bootstrap_test(chains, target; degree = 2, n_bootstrap = 100)
```

Pretty-printed via `print_psd(chains, target; degree = 2)`.

## Posterior summaries

### `summarize_fitted(chains, params)`

Returns `Dict{String, ParamStats}` where `ParamStats` carries
`median`, `q16`, `q84`, `q3lo`, `q3hi` (3σ bounds), and basic
statistics. Computed for all unfrozen parameters in the chain.

### `summarize_derived(chains, params)`

Same structure for derived parameters (m·sin i, semi-major axis a,
density ρ, T_eq, TSM, ESM, AMD stability flag, …). Computed on-the-fly
from the chain.

### `print_results(fitted, derived; out=stdout)`

Combined formatted print of both tables. Aligns medians + asymmetric
errors + 3σ ranges. Use `out=open("results.txt", "w")` to redirect to
file.

### `save_results(path, fitted, derived; metadata=Dict())`

Persistent JSON+CSV save of summary tables to `path/`.

### `print_transdim_summary(chains, params; td, M_s, R_s=1.0)`

Trans-dim-specific output: P(N\_p), joint (N\_p, noise-config)
posterior, marginal noise probabilities, and conditional posterior
per significant configuration. See [Trans-dimensional](transdim.md)
for the full format.

### `save_transdim_summary(path, chains, params; td)`

Writes the trans-dim summary to a text file. Reproducible record of
the inference output for paper appendices.

### `compute_model_stats(chains, params; td)`

Lower-level: returns `Dict` of per-config statistics for programmatic
use (e.g., feeding into a paper-figure script).

### The derived-quantity helpers

`summarize_derived` is built from standalone functions you can call directly on
scalars — useful for a quick number without a chain:

| function | gives |
|---|---|
| `planet_mass(M_s, K, P, e, inc_deg)` | companion mass |
| `planet_radius`, `planet_radius_earth`, `radius_from_depth` | radius from the radius ratio or transit depth |
| `planet_density`, `planet_density_earth` | bulk density |
| `semimajor_axis_au(P, M_s)`, `semimajor_axis_from_rho` | orbital separation |
| `equilibrium_temperature(T_eff, a_Rs; Ab)` | T_eq |
| `incident_flux`, `incident_flux_earth` | insolation |
| `esm` | emission spectroscopy metric — JWST target ranking |
| `chen_kipping_mass` | mass from radius via the Chen & Kipping relation |
| `mass_function`, `msec_from_K` | the RV mass function and its inversion |
| `stellar_mass_from_density` | host mass from `rho_s` |
| `inclination_deg`, `sky_separation`, `a_from_P` | geometry conversions |
| `kepler_solve`, `tp_to_tc`, `ew_to_sesinw`, `kipping_q_to_u` | element and parametrisation conversions |

`compute_derived` / `DerivedParams` are the chain-level versions.

> `rho_s` is in **solar** units, not g/cm³.

## Candidate vetting — the promotion gate

```julia
v = vet_candidate(t, rv, rv_err, P, K, e;
                  residual   = resid,          # OTHER planets + gamma/trend removed
                  indicators = Dict("bis" => bis, "halpha" => ha),
                  P_rot      = 12.3,
                  occupancy  = 0.94, dlnZ = 8.1)

v.decision      # :promote | :reject | :ambiguous
v.reasons       # which vetoes failed, in words
```

`CandidateVerdict` reports a decision plus the individual vetoes —
`detection`, `activity_indicator`, `rotation`, `alias`, `coherence`,
`physicality` — so you can see *which* test killed a candidate rather than
just that something did.

Two things to get right:

- **`residual` should have the other planets and the offsets/trend removed.**
  It defaults to the raw `rv`, which is only correct for a single-planet
  system; on a multi-planet fit the coherence test then runs on a series still
  containing the other signals.
- **`occ_min` and `dlnZ_min` are thresholds you calibrate to a target
  false-positive rate**, not universal constants. The defaults are a starting
  point.

`coherence_discriminant` / `CoherenceResult` are the underlying test — a
genuine Keplerian keeps its phase across the baseline; an activity signal does
not.

## Posterior predictive checks

### `posterior_predictive_check(chains, params, data; n_draws=500, rng, pgram=true, acf=true, pgram_period_min=1.0, pgram_period_max=nothing)`

Subsamples `n_draws` rows from a posterior chain, re-evaluates the
forward model per draw, and aggregates residual diagnostics:

- **Three χ²/dof statistics** addressing different failure modes:
  - `rv_red_chi2_total` — best fit found *anywhere* in the posterior
    (`min(chi2_per_draw) / dof`). Robust to degenerate or circular
    axes (e.g., uniform ω at e≈0): stays good whenever the sampler
    visited a region that fits, regardless of whether the
    per-parameter median sits at a coordinate singularity.
  - `rv_red_chi2_median_model` — χ²/dof at the per-parameter median
    Theta. Equals `_total` for well-constrained posteriors; can be
    inflated when any structural axis is degenerate at the median.
  - `rv_red_chi2_per_draw_p16/p50/p84` — distribution across draws.
    A wide spread means parameter uncertainty contributes
    meaningfully to model variance.
- Per-instrument reduced χ² at the median model (`rv_red_chi2_per_inst`
  / `phot_red_chi2_per_inst`). dof per instrument is conservative:
  `n_obs − 2` for RV (γ, σ) and `n_obs − 3` for photometry.
- Pointwise residual (median model) ± std across draws
  (`rv_resid_med` / `rv_resid_std`; same for `phot_*`). The std is the
  per-point scatter of `data − preds` across draws — a useful envelope
  around the median-model residual.
- 3σ outlier count against the median-model noise
  (`rv_outliers_3sigma` / `phot_outliers_3sigma`).
- **Residual GLS periodogram** of median-model residuals — a
  significant peak means a signal you didn't fit. RV only, requires
  ≥16 RV points. The top peak's period / power / FAP land in the
  summary (`rv_residual_top_peak_period` / `_power` / `_fap`).
- **Residual ACF** of median-model residuals — non-zero lag-1 means
  correlated noise your noise model didn't absorb. RV needs ≥16
  points, photometry ≥64. Lag-1 reported as
  `rv_residual_acf_lag1` / `phot_residual_acf_lag1`.

Returns a `PPCResult` carrying all of the above plus a `summary` dict
that ends up in `summary.json` under the `ppc` key when run from
`run_job`. Pass the result to `plot_ppc` for the standard diagnostic
figure.

**RM-consistency.** `rv_predictions` now adds the in-transit
Rossiter-McLaughlin anomaly (Hirano+ 2011 / Cegla+ 2016 "reloaded";
`src/rm.jl`, wired into `src/likelihood.jl`), so
for any RM-enabled fit the PPC residuals, per-instrument χ², outliers,
and residual GLS/ACF are computed against the **RM-corrected** model.
No-op for non-RM fits. See the [Rossiter-McLaughlin](#rossiter-mclaughlin)
section below.

**GP-detrend before χ².** When a global-covariance GP is active
(celerite or the Rajpaul `ActivityGP`), the activity lives in the
covariance, so scoring a mean-model χ² double-counts it as white
scatter. `posterior_predictive_check` subtracts the GP predictive mean
from the RV residual before χ² (via `channel_gp_mean_at`), making the
reported χ²/dof the honest white-residual goodness-of-fit. No-op for
white / decorrelation-only models.

**Residual-periodogram performance cap.** The residual GLS is a
"is-there-leftover-periodicity?" *diagnostic*, not a publication
periodogram. The full bootstrap-FAP path over a baseline-sized grid is
intractable on real long-baseline RV (a ~19-yr ε Eri set drives
≈3.5×10⁵ frequencies × 1000 bootstraps ≈ hours, blowing `run_job`'s
wall-clock). So inside PPC the grid is **capped at 20 000 frequencies**
(`clamp(…, 64, 20_000)`) and the **analytic FAP** is used
(`fap_method = :analytic`, ≈3× over-confident — fine for a flag). The
period range defaults to `[pgram_period_min=1.0 day, baseline]`.
Explicit periodogram *plots* still use the full bootstrap path.

Trans-dim chains are handled transparently — `n_planets` is read per
draw so each sample uses its own active-K. Note: the residual ACF
uses uniform binning at the median cadence, so for highly irregular
RV sampling the high-lag tail can be misleading; lag-1 is the
reliable headline statistic.

### `plot_ppc(ppc; filename=nothing, ...)`

Multi-panel figure: residual time series (per instrument), residual
GLS with FAP horizontals, residual ACF, and the same pair for the
photometry channel when present. Saves to `filename` if given.

When `run_job` is invoked, PPC runs automatically after sampling
(`_run_ppc!`) and writes `ppc.png` + `ppc.json` into the output dir,
plus the scalar `summary` under the `ppc` key of `summary.json`. PPC
runs whenever there is RV **or** photometry.

| `output` key | default | effect |
|---|---|---|
| `ppc`         | `true`  | master on/off for the PPC stage |
| `ppc_n_draws` | `500`   | posterior draws subsampled (capped at chain length) |
| `save_pdf`    | `false` | also write the figure as PDF (shared across stages) |

```json
{
  "output": { "ppc": true, "ppc_n_draws": 1000 }
}
```

The PPC draw RNG is seeded from the top-level job `seed`
(`MersenneTwister(seed)`), so the PPC is reproducible run-to-run.

## PSIS-LOO and WAIC

### `compute_loo(chains, params, data; n_draws=500, log_z=nothing)`

Computes Bayesian leave-one-out cross-validation (PSIS-LOO) and
the widely-applicable information criterion (WAIC) for the fitted
model by replaying `n_draws` posterior samples through the
per-datapoint log-likelihood and feeding the matrix to
[ParetoSmooth.jl](https://github.com/TuringLang/ParetoSmooth.jl)
([Vehtari, Gelman & Gabry 2017](https://ui.adsabs.harvard.edu/abs/2017S&C....27.1413V/abstract)).
These are the honest counterweight to the PT-based evidence stack
(TI+ / SS+ / H+) under model misspecification — log Z compares model
*priors* + likelihoods, while LOO compares predictive performance.

Returns a `LooResult` with `elpd_loo`, `se_elpd_loo`, `p_loo`,
`elpd_waic`, `se_elpd_waic`, `p_waic`, `n_obs`, the per-point Pareto k̂
diagnostic vector (`pareto_k`), and `pareto_k_max` /
`pareto_k_warn_count` (count of points with k̂ > 0.7 where PSIS-LOO is
unreliable). A high `pareto_k_warn_count` is the signal to distrust the
LOO estimate, not a bug — those points have such influential
posteriors that the importance-sampling weights are heavy-tailed.

Pass `log_z` (e.g., from the PT result) to get
`loo_compare_log_z = elpd_loo - log_z`. Useful when log Z and LOO
disagree — disagreement is a model-misspecification flag, not a bug.

**Restriction.** Pointwise log-likelihood is well-defined under
independence after the sequential noise stage: white-noise /
`ActivityDecorrelation` / `ActivityJitter` / `ARModel` / `MAModel` are
all supported (AR/MA are applied to the predictions/residuals before
the per-point Gaussian). General **GP (`CovarianceNoise`) models are
refused** with a clear `ArgumentError` — the joint Gaussian covariance
breaks pointwise factorisation, and a naive independence approximation
would silently mislead.

**One GP exception:** an `ActivityGP` configured with
`marginalize_indicators = true` *is* supported. Nereus conditions the
RV channel on the indicator channels (closed-form RV-conditional
Gaussian) and uses the Sundararajan & Keerthi (2001) leave-one-out
formula on that conditional covariance, giving an honest per-point
`log p(yᵢ | y₋ᵢ, θ)`. Any *other* `CovarianceNoise` (`CeleriteSHO`,
`CeleriteRotation`, `CeleriteRotationFM17`), or an `ActivityGP` without
`marginalize_indicators`, throws. For those, use PT log Z
(TI+ / SS+ / H+) for model comparison, or strip the GP and refit.

When invoked via `run_job` (`_run_loo!`), LOO/WAIC runs after the
sampler whenever there is RV or photometry, and writes the scalar
result into `summary.json` under the `loo` key (`elpd_loo`,
`se_elpd_loo`, `p_loo`, `elpd_waic`, `se_elpd_waic`, `p_waic`,
`pareto_k_max`, `pareto_k_warn_count`, `loo_compare_log_z`). If the
sampler result carries a finite `log_evidence`, it is passed as `log_z`
so `loo_compare_log_z = elpd_loo − log_z` is populated automatically.
When `compute_loo` throws (e.g. a GP model), the stage is recorded as
`{"status": "skipped", "error": …}` and the job continues.

| `output` key | default | effect |
|---|---|---|
| `loo`         | `true` | master on/off for the LOO/WAIC stage |
| `loo_n_draws` | `500`  | posterior draws replayed (capped at chain length) |

The LOO draw RNG is seeded from the top-level job `seed`.

## Detection limits

### `detection_limits(chains, params; planet=1, ...)`

Computes the Bayesian K upper-limit curve `K_lim(P)` from a posterior
chain. For each log-spaced bin in `P_k<planet>`, takes the
`confidence`-quantile (default 95%) of the K marginal as the
detection limit. The interpretation is:

> Given my data and priors, anything with K above this curve at
> period P would have been detected at the stated credible level.

The natural Bayesian counterpart to injection-recovery completeness:
no need to inject synthetic Keplerians and refit; the posterior
already tells you what could have been there.

**Input requirement.** The chain must span the prior on P
meaningfully for this to be honest. EMPEROR-style prior-seeded
parallel-tempering chains
([Vousden+ 2016](https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.1919V/abstract))
are the canonical input — initial walkers drawn from the prior across
the temperature ladder, so the cold chain inherits broad period
coverage even when no signal pins it to a particular mode. Other
samplers (NUTS, nested sampling, NF-coupled PT) concentrate around
modes and produce biased limits. The function warns when the chain's
P range covers less than 10% of the prior; the bin's `K_limit` is
`NaN` whenever it contains fewer than `min_samples_per_bin` draws.

Trans-dim chains are handled: draws where `n_planets < planet` are
masked out so only samples where this planet slot is active contribute.

```julia
# Recommended path: prior-seeded PT (init_strategy=:prior is the default,
# giving broad per-temperature period coverage).
result = sample_ptemcee(target, data; n_walkers=128, n_steps=10_000,
                                       n_temps=10, init_strategy=:prior)
chains = result.chains
res    = detection_limits(chains, params; n_bins=30, confidence=0.95)
plot_detection_limits(res; filename="detection_limits.png")
```

### `plot_detection_limits(res; filename, ...)`

Log-log `K_lim(P)` curve with the "detectable above" region shaded.
Sparse bins are flagged with an x-cross marker. Saves to `filename`
if given.

`run_job` (`_run_detection_limits!`) auto-invokes detection limits when
the sampler is one of `ptemcee`, `transdim_ptemcee`, `pt`, or `pt_warm`
(the prior-seeded PT set with broad period coverage). For other
samplers it **skips silently** — their chains concentrate around modes
and aren't appropriate input. The stage also requires RV data and a
fitted `P_k1` column; if either is missing it returns early.

You can force the stage on or off explicitly: setting
`output.detection_limits = true` runs it for *any* sampler (use only if
you know the chain has broad period coverage), and `false` disables it.

| `output` key | default | effect |
|---|---|---|
| `detection_limits`            | (auto: PT-only) | force on/off |
| `detection_limits_n_bins`     | `30`            | log-spaced P bins |
| `detection_limits_confidence` | `0.95`          | K-quantile / credible level |
| `detection_limits_planet`     | `1`             | planet slot for the (P, K) marginal |

Outputs: the figure `detection_limits.png`, a NetCDF sidecar
`detection_limits.nc` holding the per-bin arrays (`P_centers`,
`P_edges`, `K_limit`, `n_in_bin`), and a *scalar* `detection_limits`
block in `summary.json` (`planet_index`, `confidence`, `prior_P_lo`,
`prior_P_hi`, `n_bins`, `arrays_path`). The arrays are deliberately
kept out of `summary.json` so it stays human-skimmable — stream them
from the `.nc` sidecar.

## Fit health — the silent-wrong guard

### `assess_fit(chains; prior_bounds=nothing, ess_min=200, rhat_max=1.05, ensemble=false, …)`

`assess_fit` is Nereus's operational safety net. Its premise is blunt:
even a sampler Nereus has **not** certified must not be able to hand a
user a confident-but-wrong posterior with no flag attached. The
trans-dim/fixed-dim validation matrix caught three real failure modes
that all produced innocent-looking output:

- **`nuts`** — two chains frozen in distinct modes, merged into a
  credible interval that happens to bracket the truth. The marginal CI
  looks fine but it is the *union of two disjoint basins*, not a
  posterior.
- **`pt_whitening`** — tight CIs sitting at a wrong mode. Converged by
  every per-chain metric, just wrong.
- **`map`** — a parameter railed against a prior bound with
  `converged=true`. The optimizer walked into the wall and reported
  success.

`assess_fit` runs **four cheap structural checks** against an
`MCMCChains.Chains` and returns a `FitHealthReport` whose `overall` is
the **worst** verdict (`:ok < :warn < :fail`). None of these checks can
*prove* a fit is right — they only catch the specific ways a fit is
loudly, structurally broken while looking calm. Treat `:ok` as "no red
flags raised", **never** as certification.

The checks (`src/diagnostics/fit_health.jl`):

1. **Convergence** — rank-normalized split-R̂ and minimum ESS over the
   fitted parameters (`MCMCChains.ess_rhat(…; kind = :rank)`). `:fail`
   if any R̂ > `rhat_max` (default `1.05`) or any ESS < `ess_min`
   (default `200`). With `ensemble = true`, each walker's own trace is
   split in half and the halves stacked as the "chains" axis — so a
   single ensemble of correlated walkers (ptemcee, PA, SMC, …) is still
   diagnosable instead of being treated as N independent chains (which
   gives a misleadingly low R̂).
2. **Multimodality / non-mixing** — for every parameter, compares the
   spread (std) of the per-chain medians against the mean within-chain
   std. A ratio above `mode_ratio_max` (default `3.0`) means the chains
   sit in disjoint basins, so the merged marginal is a union of modes,
   not a posterior. `:fail`. Needs ≥2 chains (single-chain → `:ok`,
   "not assessable"). This is the `nuts` / `pt_whitening` catcher.
3. **Prior-edge rail** — with `prior_bounds` (a `Dict` mapping param
   name to `(lo, hi)`), flags any parameter whose posterior piles up
   against a finite bound: median within `edge_eps` (default `0.01`,
   fraction of the bound span) of a bound, **or** more than
   `edge_mass_frac` (default `0.05`) of draws within that margin.
   `:fail`. Without `prior_bounds` the check is skipped (`:ok`, "no
   prior_bounds given"). This is the `map` / `PA` catcher.
4. **Log-posterior sanity** — if a `:lp` / `:log_density` / `:logp`
   column exists, flags non-finite values and finite values below
   `lp_floor` (default `-1e6`) as corrupt. This is the corrupt-logZ
   catcher.

Bookkeeping columns (`:lp`, `:n_p`, `noise_active_*`, `:weights`, …)
are excluded from the per-parameter checks automatically; restrict the
checked set with `param_names`.

There is a second method that also screens a **MAP / point estimate**
for consistency with the posterior bulk:

```julia
assess_fit(map_point, chains; nsigma=5.0, kwargs...)
```

`map_point` may be a `Vector{<:Real}` aligned with the checked
parameters, a `Dict` (param ⇒ value), or a `MAPResult` (its `x_map` +
`param_names` are used). Any parameter where the MAP lands more than
`nsigma` robust σ (half the central-68% width) from the per-parameter
median `:fail`s — a sign the optimizer found a spurious mode or railed
against a bound while MCMC explored a different basin.

`FitHealthReport` has a `text/plain` show method that prints each
check's icon, name, and message, plus a loud banner when `overall` is
`:fail` ("DO NOT TRUST THIS POSTERIOR AS-IS") or `:warn`.

```julia
report = assess_fit(chains; prior_bounds = bounds_dict, ensemble = true)
report.overall      # :ok | :warn | :fail
display(report)     # full per-check breakdown
```

#### In `run_job`

`_run_fit_health!` runs the chain-only `assess_fit` after every fit.
It is **fail-soft**: it only logs a verdict and records it in the
summary; it never alters the chains or any other output. It builds
`prior_bounds` from the model layout (hard bounds of every fitted
parameter), and sets `ensemble = true` automatically when the sampler
is one of `ptemcee`, `transdim_ptemcee`, `pt_whitening`, `pa`, `smc`,
`ensemble`, or `ess`. The result lands in `summary.json` under
`fit_health`:

```json
{
  "fit_health": {
    "overall": "ok",
    "checks": {
      "convergence":   { "status": "ok",   "message": "converged: …" },
      "multimodality": { "status": "ok",   "message": "well mixed: …" },
      "prior_rail":    { "status": "ok",   "message": "no parameter railed …" },
      "logpost":       { "status": "ok",   "message": "log-posterior sane …" }
    }
  }
}
```

When `overall == "fail"`, `run_job` emits a prominent `@warn` with the
full report; the pipeline consumer should treat a `fail` as a hard stop
on trusting the posterior. Disable the stage with `output.fit_health =
false` (default `true`).

## Rossiter-McLaughlin

When a planet's `PlanetDataSources` includes an `RM` source, the RV
forward model gains an in-transit Rossiter-McLaughlin anomaly. This is
relevant to diagnostics because **`rv_predictions` includes the RM
term** (`src/likelihood.jl`, `src/rm.jl`) — so PPC residuals,
per-instrument
χ², 3σ outliers, the residual GLS/ACF, fit-health, and every RV plot
are all RM-consistent. The model is no-op for non-RM fits.

**Planet modes** (`planet_modes` entries) that enable RM:

| mode | sources | RM kernel |
|---|---|---|
| `RVPM_RM`      | RV + PM + RM        | Hirano+ 2011 (leading order) |
| `RVPMAS_RM`    | RV + PM + AS + RM   | Hirano+ 2011 |
| `RVPM_RM_R`    | RV + PM + RM(reloaded) | Cegla+ 2016 "reloaded" disk integration |
| `RVPMAS_RM_R`  | RV + PM + AS + RM(reloaded) | Cegla+ 2016 |

RM requires the transiting (PM) geometry by construction, and a stellar
`a/R_*` source — either the `rho_s` parametrization or both `M_s` and
`R_s` set in the config. If neither is available the RM term is skipped
in `rv_predictions` (it is never allowed to poison predictions).

**Parameters** (default priors from `src/default_priors.jl`):

- `v_sin_i_star` — stellar projected rotation `V·sin(i_*)` in **m/s**,
  system-level (one slot, shared across RM planets). Default prior
  `LogUniform(500, 100000)` m/s (0.5 km/s M-dwarf floor → 100 km/s
  A-star regime). Override with a `NormalPrior` when you have a
  spectroscopic `v sin i`.
- `lambda_k<k>` — sky-projected obliquity λ of planet `k` in
  **radians**, one slot per RM planet. Default prior `Uniform(-π, π)`.
  Tighten to e.g. `NormalPrior(0, π/8)` for aligned systems.

**Plot.** `rm_anomaly` (`plot_rm`) renders the RM RV-anomaly curve for
RM-enabled fits; `run_job` adds it automatically when any planet mode
has RM (`models/rm_anomaly_K*.png`).

## ActivityGP via `run_job` (indicator-GP activity model)

The Rajpaul-style `ActivityGP` (joint RV + activity-indicator GP) is now
fully driveable from a job config — relevant here because it changes how
the diagnostics behave (PPC GP-detrends the RV residual; LOO supports it
only via `marginalize_indicators = true`; see those sections above).

Declare it as a `noise_models` entry:

```json
{
  "noise_models": [
    {
      "kind": "ActivityGP",
      "instruments": ["HARPS"],
      "kwargs": {
        "channels": ["bis", "fwhm"],
        "use_derivative": true
      }
    }
  ]
}
```

`channels` is supplied as JSON strings and symbolized for you
(`Symbol.(…)`). `use_derivative = true` enables the FF′-style
derivative couplings (the Ġ term, Aigrain+ 2012 / Rajpaul+ 2015).

**Indicator data is required, with errors.** `ActivityGP` needs each
indicator channel's values *and* its 1σ uncertainties:

- **Inline RV block** — in an RV `values` block, add a `<name>` array
  and a matching `<name>_err` array (e.g. `bis` + `bis_err`, `fwhm` +
  `fwhm_err`). Any key beyond `bjd`/`rv`/`rv_err`/`instrument` is read
  as an indicator; a `<name>_err` whose base `<name>` is also present
  is taken as that indicator's error.
- **CSV RV block** — list the indicator columns in `indicator_cols`,
  and provide a matching `<col>_err` column per indicator.

If the indicator errors are missing, the `ActivityGP` likelihood and
the LOO path cannot be evaluated. (Without an `ActivityGP`/indicator-GP
model, indicator columns are simply parsed and ignored.)

## Diagnostic & model plots in `run_job`

Beyond `ppc.png` and `detection_limits.png`, `_make_plots` emits the
model and posterior figures into the `plots/` tree (logical names like
`models/…`, scanned into the `figures` manifest in `summary.json`).
Diagnostic-relevant additions:

- **`rm_anomaly`** (`plot_rm`) — RM RV-anomaly curve, per RM planet
  (`models/rm_anomaly_K*.png`). Added when any planet mode enables RM.
- **`transit_overlay`** (`plot_transit_overlay_fit`) — per-transit QC
  gallery: each individual transit window with the fitted model
  overlaid (`models/transit_overlay_K*.png`). Added when there is
  photometry. The fast visual check for which transits the model
  actually fits.
- **`rv_components`** (`plot_rv_components`) — RV model decomposition
  (Keplerian / activity / GP / trend components), distinct from
  `rv_timeseries` (`models/rv_components.png`). Added when an
  `ActivityDecorrelation`, `ActivityGP`, or `CovarianceNoise` model is
  active.
- **`ttv_oc`** (`plot_ttv_oc`) — transit-timing O−C. With ≥2 planets it
  shows the N-body O−C envelope driven by the perturber; **with a
  single transiting planet it now renders a data-only O−C** (measured
  per-transit `Tc` vs the linear ephemeris) by passing `planet_b_k=0`.
  Added when there is photometry. Override the perturber with
  `plot_kwargs.planet_b_k`.

## Chain I/O

### `save_chains(path, chains, params; data=nothing, metadata=Dict())`

Writes the `MCMCChains.Chains` object to NetCDF, with `params` layout
and (optionally) the `Data` struct alongside. Readable from Julia
(`load_chains`) AND Python (xarray / arviz). One file holds the full
fit state for reproducible analysis later.

```julia
save_chains("results/run5_chains.nc", chains, params; data = data)
```

Default size hint: ~50 MB for a 130k-sample × 47-param trans-dim run.
NetCDF compression keeps it manageable.

### `load_chains(path) -> (chains, params)`

Inverse operation. Reconstructs the `Chains` object and the `Params`
metadata.

```julia
chains, params = load_chains("results/run5_chains.nc")
```

The `params` returned carries the layout metadata necessary to slice
the chain by parameter name; the original `Data` is **not** loaded —
re-construct it from the source files if you need to recompute
likelihoods (the `data` kwarg on save is for archival only).

### Python interop

```python
import xarray as xr
import arviz as az

ds = xr.open_dataset("results/run5_chains.nc")
posterior = az.from_netcdf("results/run5_chains.nc")
```

Standard xarray operations work on the dataset; arviz can plot
posterior densities, do divergence diagnostics, etc.

## ProgressBar

`ProgressBar` (in `src/progress.jl`) is the unified progress display
used by every sampler. You won't typically construct one directly;
samplers create one internally and update it with current state
(round, acceptance, log Z, etc.). Pass `show_progress=false` to
silence on long-running batch jobs.

## Reproducibility

Every sampler accepts a `seed::Int` keyword. Re-running with the same
seed and same `target/data` produces bit-identical chains under
single-threaded execution. With multi-threading (`:static`), the
seed is propagated per-thread (`seed + 1_000_003 * tid`), so results
are deterministic conditional on the **thread count**.

For paper-grade reproducibility, fix both `JULIA_NUM_THREADS` and the
`seed`. Store both in the script that generated the run.
