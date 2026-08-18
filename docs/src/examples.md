# Worked examples

End-to-end recipes built on **`run_job(cfg)`** — the single
config-driven dispatcher that every Nereus deployment (Docker workers,
Python/`juliacall`, CLI) should call. Each example is a complete,
runnable config `Dict` with realistic priors, a sampler block, and an
output block. The configs below are distilled from the working
validation drivers in `test/validation/validate_runjob_*.jl`, which are
themselves CI-checked end-to-end fits.

There is **one example per `planet_mode`** —
`RV_ONLY`, `PM_ONLY`, `RVPM`, `RVAS`, `RVPMAS`, `RVPM_RM`, `RVPM_TTV` —
plus an ActivityGP-noise example and a trans-dimensional model-selection
example.

Nereus is at **version `0.2.0`**.

!!! note "`run_job` vs. the low-level API"
    `run_job(cfg)` builds the `Data`, `Params`, `NereusTarget`,
    runs the sampler, writes `summary.json` + `chains.nc` + the
    `plots/` tree + the science tables, and **returns the summary
    `Dict`**. If you need the in-Julia objects directly (`Params`,
    `Theta`, `chains`), call `Nereus._build_data`, `Nereus._build_model`
    and the `sample_*` functions yourself — but for production
    pipelines `run_job` is the supported, schema-validated entry point.
    The authoritative schema is `src/runner.jl` and `docs/JOB_CONFIG.md`.

## The config schema at a glance

A config is a `Dict` (or a JSON file passed to
`run_job("/path/config.json")`) with these top-level keys:

| Key | Required | Notes |
|-----|----------|-------|
| `output_dir`    | ✅ | directory for `summary.json`, `chains.nc`, `plots/`, `tables/` |
| `data`          | ✅ | at least one of `rv`, `transit_photometry`, `iad`, `gost`, `hgca`, `relastrom`, `gaia_dr3` |
| `model`         | ✅ | `max_kplanet`, `planet_modes`, `parametrization`, `stability`, `trend_order` … |
| `sampler`       | ✅ | `{name, kwargs}` |
| `priors`        | optional | per-parameter `{type, args}`; unset params get sensible defaults |
| `noise_models`  | optional | list of `{kind, instruments, channel, kwargs}` |
| `transdim`      | required for `transdim_ptemcee`/`rjmcmc`/`moms`/`moms_ns` | `{max_kplanet, birth_strategies, …}` |
| `star`          | optional | `M_s`, `R_s`, `T_eff`, `J_mag`, `K_mag`, `Ab` (needed for derived masses/radii) |
| `output`        | optional | `{plots, plot_kwargs, save_pdf, ppc, loo, detection_limits …}` |
| `version`, `seed`, `n_threads`, `timeout_sec` | optional | run metadata + safety rails |

Schema-validated string sets (`run_job` fails fast with a single
collated error message if any is wrong — see `_validate_config` in
`src/runner.jl`):

- **`planet_modes`**: `RV_ONLY`, `PM_ONLY`, `RVPM`, `RVAS`, `RVPMAS`,
  `RVPM_RM`, `RVPMAS_RM`, `RVPM_RM_R`, `RVPMAS_RM_R`, `RVPM_TTV`,
  `RVPMAS_TTV`, `RVPM_RM_TTV`, `PM_TTV`, `RVPM_TTV_NB`, `RVPMAS_TTV_NB`,
  `PM_TTV_NB`.
- **`parametrization.mass`**: `K_driven`, `M_sec_driven`, `a_driven`.
  **`.time`**: `Tp`, `Tc`, `Mo`. **`.ew`**: `sesinw`, `ew`.
  **`.geom`**: `b_rr`, `b_r`. Plus boolean `.marginalize_gamma`.
- **`stability`**: `none`, `amd`, `gladman`.
- **`priors[*].type`**: `UniformPrior`, `LogUniformPrior`,
  `ModJeffreysPrior`, `NormalPrior`, `FixedPrior`, `SinePrior`,
  `BetaPrior`. `args` follow each constructor — e.g.
  `NormalPrior(μ, σ, lo, hi)`, `UniformPrior(lo, hi)`,
  `LogUniformPrior(lo, hi)`.
- **`noise_models[*].kind`**: `CeleriteRotation`, `CeleriteSHO`,
  `CeleriteRotationFM17`, `ActivityDecorrelation`, `ARModel`, `MAModel`,
  `ActivityJitter`, `ActivityGP`.
- **`sampler.name`**: `ptemcee`, `transdim_ptemcee`, `pt`, `pt_warm`,
  `rjmcmc`, `moms`, `moms_ns`, `nested`, `nested_ins`, `nested_dynamic`,
  `pa`, `smc`, `pt_whitening`, `nuts`, `pt_hmc`, `ofti`.
- **`output.plots`**: `rv_timeseries`, `rv_components`, `rv_phasefold`,
  `pm_timeseries`, `pm_phasefold`, `rv_astrom_phasefold`,
  `orbit_skyplane`, `iad_residuals`, `hgca_pm_residuals`,
  `relastrom_timeseries`, `relastrom_residuals`, `g23h_residuals`,
  `pm_anomaly`, `ttv_oc`, `transit_overlay`, `rm_anomaly`,
  `activity_gp_latent`, `activity_gp_decomposition`, `corner`, `trace`,
  `histograms`, `posteriors`, `transdim_occupancy`, `posteriors_raw`,
  `posteriors_parameters`, `posteriors_histograms`, `traces_grouped`,
  and `auto` (picks the right plot set from what's in `data`).

!!! warning "Symbols arrive as strings; vectors arrive as JSON arrays"
    JSON has no `Symbol` type, so every enum-style `sampler.kwargs`
    value (`bounds`, `mutation_kernel`, `within_model`, `init_strategy`,
    …) is given as a **string** and coerced to a Julia `Symbol` by the
    dispatcher. `run_job` also rejects any `sampler.kwargs` key the
    chosen sampler does not declare, with an actionable error — so a
    typo never silently no-ops.

### The parameter naming convention

Per-planet parameters are suffixed `_k<k>` (1-indexed):
`P_k1`, `K_k1`, `Tc_k1`, `Tp_k1`, `Mo_k1`, `sesinw_k1`, `secosw_k1`,
`ecc_k1`/`omega_k1` (with `ew=:ew`), `b_k1`, `rr_k1`, `inc_k1`,
`Omega_k1`, `a_k1`, `M_sec_k1`, `lambda_k1` (RM obliquity),
`ttv_k1_t<i>` (per-transit offsets). Per-instrument parameters are
suffixed with the instrument name: RV `gamma_<inst>`, `sigma_<inst>`
(jitter); photometry `offset_<inst>`, `jitter_<inst>`,
`q1_<inst>`/`q2_<inst>` (Kipping 2013 limb-darkening). Shared:
`rho_s` (stellar density, when `parametrization.use_rho_s`), `dvdt`/`curv`
(RV trend), `v_sin_i_star` (RM, m/s), `plx`, `M_pri` (astrometry).
Any parameter you don't list in `priors` gets a sensible default from
`src/default_priors.jl`.

---

## 1. RV-only characterisation — `RV_ONLY`

Known single planet, fixed N\_p = 1. Inline arrays (the `juliacall`
path); swap `values` for a `csv` block to read from disk.

```julia
using Nereus

# bjd / rv / rv_err / instrument as parallel arrays (one row per epoch)
cfg = Dict(
  "version" => "0.2.0", "seed" => 42, "output_dir" => "results/51peg",
  "star"  => Dict("M_s" => 1.11, "R_s" => 1.27, "T_eff" => 5793.0),
  "data"  => Dict(
     "rv" => Dict("values" => Dict(
        "bjd"        => bjd,           # Vector{Float64}
        "rv"         => rv,
        "rv_err"     => rv_err,
        "instrument" => inst_str))),   # Vector{String}; sorted-unique → instrument index
  "model" => Dict(
     "max_kplanet"     => 1,
     "planet_modes"    => ["RV_ONLY"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tp",
                               "ew" => "sesinw", "marginalize_gamma" => true),
     "stability"       => "none",
     "trend_order"     => 0),
  "priors" => Dict(
     "P_k1" => Dict("type" => "LogUniformPrior", "args" => [3.0, 5.0]),  # ~4.23 d
     "K_k1" => Dict("type" => "UniformPrior",    "args" => [0.0, 120.0])),
  "noise_models" => [],
  # ptemcee = ensemble parallel-tempering: robust to the K/e/ω geometry,
  # prior-seeded across temps (finds weak signals), and PT-only post-fit
  # extras (detection limits) switch on automatically.
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
     "n_temps" => 8, "n_walkers" => 60, "n_steps" => 4000, "n_burnin" => 2000,
     "show_progress" => false)),
  "output" => Dict("plots" => ["rv_timeseries", "rv_phasefold", "corner",
                               "posteriors_parameters"]),
)

summary = run_job(cfg)
println(summary["fitted"]["parameters"]["K_k1"])                  # value + 1σ/3σ + unit
println(summary["derived"]["parameters"]["msini_earth_k1"])      # m·sin i in M⊕
```

`marginalize_gamma => true` analytically integrates out the per-instrument
RV zero-points (faster, fewer nuisance dimensions). `run_job` writes
`detection_limits.{png,nc}` automatically for PT samplers when an
`P_k1` column exists.

To read from a CSV instead of inline arrays:

```julia
"rv" => Dict("csv" => "data/51peg.csv",
             "time_col" => "bjd", "rv_col" => "rv",
             "err_col" => "rv_err", "inst_col" => "instrument")
```

---

## 2. Transit-only characterisation — `PM_ONLY`

A transiting planet with photometry but no RVs. `transit_photometry` is
a **list** of per-instrument light-curve blocks (one per facility /
sector group). `exposure_time` is in **seconds** (default `120.0` =
TESS 2-min) and drives finite-exposure supersampling.

```julia
using Nereus

cfg = Dict(
  "version" => "0.2.0", "seed" => 1, "output_dir" => "results/transit_only",
  "star"  => Dict("M_s" => 0.95, "R_s" => 0.92, "T_eff" => 5500.0),
  "data"  => Dict(
     "transit_photometry" => [
        Dict("values" => Dict("bjd" => t_tess, "flux" => f_tess,
                              "flux_err" => e_tess),
             "instrument" => "TESS", "exposure_time" => 120.0),
        Dict("values" => Dict("bjd" => t_cheops, "flux" => f_cheops,
                              "flux_err" => e_cheops),
             "instrument" => "CHEOPS", "exposure_time" => 60.0)]),
  "model" => Dict(
     "max_kplanet"     => 1,
     "planet_modes"    => ["PM_ONLY"],
     "parametrization" => Dict("time" => "Tc", "geom" => "b_rr"),
     "stability"       => "none",
     "phot_trend_order" => 0),
  "priors" => Dict(
     "P_k1"  => Dict("type" => "NormalPrior",  "args" => [3.52, 1e-3, 3.51, 3.53]),
     "Tc_k1" => Dict("type" => "NormalPrior",  "args" => [2459000.5, 0.01, 2459000.0, 2459001.0]),
     "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.01, 0.20]),
     "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0]),
     # Kipping (2013) limb darkening, one (q1,q2) pair per instrument:
     "q1_TESS"   => Dict("type" => "UniformPrior", "args" => [0.0, 1.0]),
     "q2_TESS"   => Dict("type" => "UniformPrior", "args" => [0.0, 1.0]),
     "q1_CHEOPS" => Dict("type" => "UniformPrior", "args" => [0.0, 1.0]),
     "q2_CHEOPS" => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
     "n_temps" => 8, "n_walkers" => 60, "n_steps" => 3500, "n_burnin" => 2000,
     "show_progress" => false)),
  "output" => Dict("plots" => ["pm_timeseries", "pm_phasefold",
                               "transit_overlay", "ttv_oc", "corner"]),
)

summary = run_job(cfg)
println(summary["derived"]["parameters"]["T14_k1"])               # transit duration (days)
println(summary["derived"]["parameters"]["rho_star_transit_k1"])  # transit-derived ρ⋆
```

`transit_overlay` renders the per-transit QC gallery
(`plot_transit_overlay_fit`); `ttv_oc` renders the data-only O−C diagram
(measured per-transit T\_c vs the linear ephemeris) even for a
single-planet fit (`planet_b_k = 0`).

---

## 3. Joint RV + transit — `RVPM`

The headline planet-fitting case: an RV-confirmed transiting planet fit
jointly. Carries both RV-derived science (`K`, `msini`) and
transit-derived science (`rr`, `T14`, `rho_star_transit`).

```julia
using Nereus

cfg = Dict(
  "version" => "0.2.0", "seed" => 1, "output_dir" => "results/rvpm",
  "star"  => Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 5800.0),
  "data"  => Dict(
     "rv" => Dict("values" => Dict("bjd" => trv, "rv" => rv_obs,
                  "rv_err" => rv_err, "instrument" => inst_str)),
     "transit_photometry" => [Dict(
        "values" => Dict("bjd" => tph, "flux" => fl_obs, "flux_err" => fl_err),
        "instrument" => "TESS", "exposure_time" => 120.0)]),
  "model" => Dict(
     "max_kplanet"     => 1,
     "planet_modes"    => ["RVPM"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tc",
                               "ew" => "sesinw", "geom" => "b_rr",
                               "marginalize_gamma" => true),
     "stability"       => "none"),
  "priors" => Dict(
     "P_k1"  => Dict("type" => "NormalPrior",  "args" => [3.0, 0.01, 2.8, 3.2]),
     "Tc_k1" => Dict("type" => "NormalPrior",  "args" => [1.0, 0.02, 0.8, 1.2]),
     "K_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 40.0]),
     "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
     "n_temps" => 8, "n_walkers" => 60, "n_steps" => 3500, "n_burnin" => 2000,
     "show_progress" => false)),
  "output" => Dict("plots" => ["rv_phasefold", "pm_phasefold",
                               "transit_overlay", "rv_timeseries", "corner"]),
)

summary = run_job(cfg)
# fitted carries both channels:
@show summary["fitted"]["parameters"]["K_k1"]["value"]        # m/s
@show summary["fitted"]["parameters"]["rr_k1"]["value"]       # Rp/R*
@show summary["derived"]["parameters"]["T14_k1"]["value"]     # days
```

For a smooth posterior with a strongly-pinned ephemeris you can also use
`"sampler" => Dict("name" => "nuts", "kwargs" => Dict("n_chains" => 4,
"n_samples" => 2000, "n_warmup" => 1000))`. ptemcee is the safer default
for the correlated K/e/b/limb-darkening geometry, which NUTS struggles to
mix.

---

## 4. Joint RV + relative astrometry — `RVAS`

A long-period companion with RV plus relative astrometry (direct-imaging
offsets). Astrometry constrains the **inclination** and the full
orientation, which RV alone cannot. `relastrom` arrays are
`t / ra_off / dec_off / ra_err / dec_err` (mas), optionally with
`planet_idx`.

```julia
using Nereus

cfg = Dict(
  "version" => "0.2.0", "seed" => 1, "output_dir" => "results/rvas",
  "star"  => Dict("M_s" => 1.0),
  "data"  => Dict(
     "rv" => Dict("values" => Dict("bjd" => t_rv, "rv" => rv_obs,
                  "rv_err" => rv_err, "instrument" => fill("HARPS", length(t_rv)))),
     "relastrom" => Dict("values" => Dict(
        "t" => t_as, "ra_off" => ra_obs, "dec_off" => dec_obs,
        "ra_err" => ra_err, "dec_err" => dec_err))),
  "model" => Dict(
     "max_kplanet"     => 1,
     "planet_modes"    => ["RVAS"],
     # Mo time-anchor + a/M_sec-driven mass are natural for long-period
     # imaged companions; K_driven also works.
     "parametrization" => Dict("mass" => "K_driven", "time" => "Mo", "ew" => "sesinw"),
     "stability"       => "none"),
  "priors" => Dict(
     "P_k1"   => Dict("type" => "NormalPrior",  "args" => [900.0, 5.0, 800.0, 1000.0]),
     "plx"    => Dict("type" => "NormalPrior",  "args" => [25.0, 0.1, 23.0, 27.0]),
     # bound inc < 90° to break the i↔180−i relastrom mirror (optional)
     "inc_k1" => Dict("type" => "UniformPrior", "args" => [0.0, π/2])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
     "n_temps" => 8, "n_walkers" => 60, "n_steps" => 3500, "n_burnin" => 2000,
     "show_progress" => false)),
  "output" => Dict("plots" => ["orbit_skyplane", "relastrom_residuals",
                               "relastrom_timeseries", "rv_astrom_phasefold",
                               "corner"]),
)

summary = run_job(cfg)
@show rad2deg(summary["fitted"]["parameters"]["inc_k1"]["value"])  # astrometry-constrained
@show summary["derived"]["parameters"]["a_au_k1"]["value"]         # semi-major axis (AU)
@show summary["derived"]["parameters"]["ecc_k1"]["value"]
```

**Real-data RVAS with HGCA.** For an HGCA proper-motion anomaly add an
`hgca` block. Inline HGCA needs exactly three epochs (Hipparcos, HG,
Gaia) with per-epoch 2×2 covariance matrices:

```julia
"hgca" => Dict("values" => Dict(
   "t"      => collect(hgca.epochs),         # 3 epochs
   "pmra"   => collect(hgca.pmra),
   "pmdec"  => collect(hgca.pmdec),
   "cov_ep" => [hgca.cov_ep[1], hgca.cov_ep[2], hgca.cov_ep[3]],  # 3×(2×2)
   "plx"    => hgca.plx, "plx_err" => hgca.plx_err, "hip_id" => hgca.hip_id))
```

This is the HD 159062 recipe (`a_driven` mass, `Mo` time anchor; orbit
recovered against the orvara/Brandt truth at a ≈ 58 AU, e ≈ 0.11,
i ≈ 62°, M\_sec ≈ 0.6 M⊙). Add `"hgca_pm_residuals"` to the plot list.
Load the HGCA row with `load_hgca_row("HGCA_vEDR3.fits", hip_id)`.

---

## 5. RV + absolute astrometry (HGCA + GOST) — `RVAS` / `RVPMAS`

The **validated** Hipparcos–Gaia path is **HGCA + GOST**: the Brandt
eDR3 HGCA gives the three cross-calibrated proper motions (Hipparcos,
Hip–Gaia, Gaia), and GOST supplies the Gaia scan geometry to
window-average the Gaia-epoch reflex (Mode B, Brandt 2018 §4). Both
blocks **auto-fetch** — `hgca` from `hip_id`, `gost` from
`{ra_deg, dec_deg}`. The Hip–Gaia proper-motion anomaly is what pins the
companion's absolute mass. This is the ε Eri (HIP 16537) recipe.

```julia
using Nereus

cfg = Dict(
  "version" => "0.2.0", "seed" => 7, "output_dir" => "results/eps_eri",
  "star"  => Dict("M_s" => 0.82),
  "data"  => Dict(
     "rv"   => Dict("values" => Dict("bjd" => bjd, "rv" => rv,
                    "rv_err" => rv_err, "instrument" => inst_str)),
     "hgca" => Dict("hip_id" => 16537),                       # auto-fetch Brandt eDR3 HGCA row
     "gost" => Dict("ra_deg" => 53.232665, "dec_deg" => -9.458250,
                    "from" => "2014-07-26T00:00:00",
                    "to"   => "2025-01-15T00:00:00")),         # auto-fetch Gaia GOST (Mode B)
  "model" => Dict(
     "max_kplanet"     => 1,
     "planet_modes"    => ["RVAS"],
     "parametrization" => Dict("mass" => "M_sec_driven", "time" => "Tp", "ew" => "sesinw"),
     "stability"       => "none"),
  "priors" => Dict(
     "P_k1"     => Dict("type" => "NormalPrior",     "args" => [2700.0, 80.0, 2300.0, 3100.0]),
     "M_sec_k1" => Dict("type" => "LogUniformPrior", "args" => [1.0e-4, 0.05]),  # M⊙
     "inc_k1"   => Dict("type" => "SinePrior",       "args" => Float64[]),
     "Omega_k1" => Dict("type" => "UniformPrior",    "args" => [0.0, 2π]),
     "plx"      => Dict("type" => "NormalPrior",     "args" => [310.94, 0.17, 305.0, 320.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
     "n_temps" => 8, "n_walkers" => 60, "n_steps" => 3000, "n_burnin" => 1800,
     "show_progress" => false)),
  "output" => Dict("plots" => ["hgca_pm_residuals", "pm_anomaly",
                               "rv_timeseries", "corner"]),
)

summary = run_job(cfg)
```

`hgca_pm_residuals` shows the 3-epoch PM residuals; `pm_anomaly` plots
the GOST-window-averaged Gaia-epoch reflex. Set `planet_modes =>
["RVPMAS"]` and add `transit_photometry` / `relastrom` to combine with
transit and relative astrometry. `SinePrior()` takes no args.

!!! warning "GOST is an auxiliary — it is inert without a Gaia-epoch product"
    `gost_log_likelihood` returns 0. GOST only window-averages the Gaia
    epoch of a **paired** `hgca` / `gaia_dr3` / `g23h`. `Data(iad, gost)`
    with no `hgca`/`gaia_dr3` is therefore **IAD-only** (Hipparcos epoch)
    — the Gaia baseline is absent, and the `Data` constructor warns.
    Use `hgca` (above) or supply a Gaia DR3 5-param solution.

!!! tip "Raw Hipparcos IAD"
    For the abscissa-level Hipparcos likelihood, use `"iad" =>
    Dict("hip_id" => …)` (auto-fetch) or inline `"values" => Dict("t",
    "abscissa", "abscissa_err", "psi", "parallax_factor", "pm_factor")`.
    IAD alone constrains the Hipparcos epoch; the full Hip+Gaia IAD joint
    additionally needs a `gaia_dr3` block (implemented, not yet validated
    — prefer HGCA).

---

## 6. Rossiter-McLaughlin — `RVPM_RM`

A misaligned transiting planet with **densely-sampled in-transit RV**
(a spectroscopic transit) carrying the RM anomaly, out-of-transit RV for
the orbit, and photometry pinning the transit geometry. The RM model is
Hirano+ 2011 (`src/rm.jl`); `RVPM_RM_R` selects the Cegla+ 2016
"Reloaded" variant on the same parameter slots.

RM adds two parameters:

- **`v_sin_i_star`** — projected stellar rotation, **m/s** (system-level;
  default `LogUniformPrior(500, 100000)`; override with a `NormalPrior`
  when spectroscopic v·sin i\_⋆ is measured).
- **`lambda_k<k>`** — sky-projected obliquity, **radians**
  (default `UniformPrior(-π, π)`; tighten to `NormalPrior(0, π/8)` for
  aligned systems).

The RM term is now folded into `rv_predictions` (`src/likelihood.jl`),
so the PPC, residuals, phase-folds, and the `rm_anomaly` plot are all
RM-consistent.

```julia
using Nereus

cfg = Dict(
  "version" => "0.2.0", "seed" => 1, "output_dir" => "results/rm",
  "star"  => Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 6200.0),
  "data"  => Dict(
     # trv must densely sample each transit window AND cover the orbit
     "rv" => Dict("values" => Dict("bjd" => trv, "rv" => rv_obs,
                  "rv_err" => rv_err, "instrument" => fill("HARPS", length(trv)))),
     "transit_photometry" => [Dict(
        "values" => Dict("bjd" => tph, "flux" => fl_obs, "flux_err" => fl_err),
        "instrument" => "TESS", "exposure_time" => 120.0)]),
  "model" => Dict(
     "max_kplanet"     => 1,
     "planet_modes"    => ["RVPM_RM"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tc",
                               "ew" => "sesinw", "geom" => "b_rr"),
     "stability"       => "none"),
  "priors" => Dict(
     "P_k1"         => Dict("type" => "NormalPrior",  "args" => [3.5, 0.01, 3.4, 3.6]),
     "Tc_k1"        => Dict("type" => "NormalPrior",  "args" => [1.0, 0.01, 0.9, 1.1]),
     "K_k1"         => Dict("type" => "UniformPrior", "args" => [0.0, 120.0]),
     "rr_k1"        => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "b_k1"         => Dict("type" => "UniformPrior", "args" => [0.0, 0.9]),
     # RM parameters (v·sin i_* in m/s; λ in radians):
     "v_sin_i_star" => Dict("type" => "NormalPrior",  "args" => [6000.0, 1500.0, 500.0, 20000.0]),
     "lambda_k1"    => Dict("type" => "UniformPrior", "args" => [-π, π])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
     "n_temps" => 8, "n_walkers" => 70, "n_steps" => 3000, "n_burnin" => 1800,
     "show_progress" => false)),
  "output" => Dict("plots" => ["rm_anomaly", "rv_phasefold", "corner"]),
)

summary = run_job(cfg)
@show summary["fitted"]["parameters"]["v_sin_i_star"]["value"]   # m/s
@show rad2deg(summary["fitted"]["parameters"]["lambda_k1"]["value"])  # ° sky-projected obliquity
```

For RM **plus** absolute astrometry use `RVPMAS_RM` (or `RVPMAS_RM_R`
for the Reloaded model).

---

## 7. Transit-timing variations — `RVPM_TTV`

Single transiting planet with a non-linear ephemeris: per-transit
timing offsets are free parameters (`ttv_k1_t<i>`). Declare the
per-planet transit count with `model.ttv_n_transits` (a map
`planet → n_transits`). The default per-offset prior is
`NormalPrior(0, 1 d, ±10 d)`.

```julia
using Nereus

cfg = Dict(
  "version" => "0.2.0", "seed" => 1, "output_dir" => "results/ttv",
  "star"  => Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 5800.0),
  "data"  => Dict(
     "rv" => Dict("values" => Dict("bjd" => trv, "rv" => rv_obs,
                  "rv_err" => rv_err, "instrument" => fill("HARPS", length(trv)))),
     "transit_photometry" => [Dict(
        "values" => Dict("bjd" => tph, "flux" => fl_obs, "flux_err" => fl_err),
        "instrument" => "TESS", "exposure_time" => 120.0)]),
  "model" => Dict(
     "max_kplanet"     => 1,
     "planet_modes"    => ["RVPM_TTV"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tc",
                               "ew" => "sesinw", "geom" => "b_rr"),
     "ttv_n_transits"  => Dict("1" => 12),      # planet 1 has 12 observed transits
     "stability"       => "none"),
  "priors" => Dict(
     "P_k1"  => Dict("type" => "NormalPrior",  "args" => [3.6, 0.01, 3.4, 3.8]),
     "Tc_k1" => Dict("type" => "NormalPrior",  "args" => [1.5, 0.02, 1.3, 1.7]),
     "K_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 30.0]),
     "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
     "n_temps" => 6, "n_walkers" => 60, "n_steps" => 2500, "n_burnin" => 1500,
     "show_progress" => false)),
  "output" => Dict("plots" => ["ttv_oc", "transit_overlay", "pm_phasefold", "corner"]),
)

summary = run_job(cfg)
```

`ttv_oc` renders the O−C diagram. For the **dynamical (N-body) TTV**
case — two or more planets whose mutual perturbations drive the timing
variations — use the `*_TTV_NB` modes (`RVPM_TTV_NB`, `RVPMAS_TTV_NB`,
`PM_TTV_NB`) and pick the integrator with
`model.ttv_backend => "ttvfaster"` (perturbative 1st-order series,
default) or `"nbody"` (full ODE integration; use for high-mass / high-e /
strongly-resonant systems). With ≥ 2 transiting planets the `ttv_oc`
plot shows the perturber-driven envelope (`planet_b_k` defaults to the
2nd planet).

---

## 8. Activity GP across multiple indicators — `ActivityGP` noise

A multi-channel activity GP that shares a latent rotation process
G(t) (and optionally its derivative) across the RV and the activity
indicators (BIS, FWHM, …), FF′-style. This is the cleanest way to fit a
planet *through* rotation activity.

`ActivityGP` is wired into the `run_job` noise registry. **It requires
the indicator errors.** Supply them in the RV `values` block by adding
`<name>` + `<name>_err` arrays (a `<name>_err` whose base `<name>` is
also present is treated as that indicator's 1σ; a lone `<name>_err`
would be a value):

```julia
using Nereus

cfg = Dict(
  "version" => "0.2.0", "seed" => 1, "output_dir" => "results/agp",
  "star"  => Dict("M_s" => 1.0),
  "data"  => Dict(
     "rv" => Dict("values" => Dict(
        "bjd"        => trv,
        "rv"         => rv_obs,   "rv_err"   => rv_err,
        "instrument" => fill("HARPS", length(trv)),
        # activity indicators + their 1σ (required by ActivityGP):
        "bis"  => bis_obs,  "bis_err"  => bis_err,
        "fwhm" => fwhm_obs, "fwhm_err" => fwhm_err))),
  "model" => Dict(
     "max_kplanet"     => 1,
     "planet_modes"    => ["RV_ONLY"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tc", "ew" => "sesinw"),
     "stability"       => "none"),
  "priors" => Dict(
     "P_k1" => Dict("type" => "NormalPrior",  "args" => [4.05, 0.02, 3.8, 4.3]),
     "K_k1" => Dict("type" => "UniformPrior", "args" => [0.0, 30.0])),
  # ActivityGP couples the rotation GP across RV + bis + fwhm.
  # use_derivative=true adds the FF′ G'(t) channel.
  "noise_models" => [Dict(
     "kind" => "ActivityGP", "instruments" => ["HARPS"],
     "kwargs" => Dict("channels" => ["bis", "fwhm"], "use_derivative" => true))],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
     "n_temps" => 8, "n_walkers" => 80, "n_steps" => 1500, "n_burnin" => 1000,
     "show_progress" => false)),
  "output" => Dict("plots" => ["activity_gp_decomposition", "activity_gp_latent",
                               "rv_components", "rv_timeseries", "corner"]),
)

summary = run_job(cfg)
@show summary["fitted"]["parameters"]["K_k1"]["value"]   # planet survives the activity GP
```

`activity_gp_latent` shows the inferred latent rotation process;
`activity_gp_decomposition` splits each channel into its activity
contribution; `rv_components` decomposes the RV model (distinct from
`rv_timeseries`, which is the decorrelated single-curve view).

For a **CSV** RV block, name the indicator columns in `indicator_cols`
and provide matching `<col>_err` columns:

```julia
"rv" => Dict("csv" => "data/star_rv.csv",
             "indicator_cols" => ["bis", "fwhm"])   # reads bis, bis_err, fwhm, fwhm_err
```

!!! note "GP and LOO"
    PSIS-LOO / WAIC are skipped (reported as `"skipped"` in the summary,
    not failed) when an active noise model is a covariance GP — the
    pointwise likelihood factorisation `compute_loo` needs doesn't hold.

---

## 9. Trans-dimensional planet search — `transdim_ptemcee`

You don't know how many planets are in the data. Nereus searches over
N\_p ∈ {0, …, `max_kplanet`} with reversible birth/death moves and
returns the **occupancy** P(N\_p = k | data) — the headline trans-dim
output. Trans-dim samplers (`transdim_ptemcee`, `rjmcmc`, `moms`,
`moms_ns`) **require** a top-level `transdim` block.

```julia
using Nereus

cfg = Dict(
  "version" => "0.2.0", "seed" => 1, "output_dir" => "results/blind_search",
  "star"  => Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 5800.0),
  "data"  => Dict(
     "rv" => Dict("values" => Dict("bjd" => trv, "rv" => rv_obs,
                  "rv_err" => rv_err, "instrument" => fill("HARPS", length(trv)))),
     "transit_photometry" => [Dict(
        "values" => Dict("bjd" => tph, "flux" => fl_obs, "flux_err" => fl_err),
        "instrument" => "TESS", "exposure_time" => 120.0)]),
  "model" => Dict(
     "max_kplanet"     => 2,                        # search N_p = 0,1,2
     "planet_modes"    => ["RVPM", "RVPM"],         # one mode entry per slot
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tc",
                               "ew" => "sesinw", "geom" => "b_rr",
                               "marginalize_gamma" => true),
     "stability"       => "none"),
  # broad-ish search priors so births must actually FIND a signal
  "priors" => Dict(
     "P_k1"  => Dict("type" => "UniformPrior", "args" => [2.5, 3.5]),
     "P_k2"  => Dict("type" => "UniformPrior", "args" => [2.5, 3.5]),
     "K_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 40.0]),
     "K_k2"  => Dict("type" => "UniformPrior", "args" => [0.0, 40.0]),
     "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "rr_k2" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "Tc_k1" => Dict("type" => "NormalPrior",  "args" => [1.0, 0.05, 0.7, 1.3]),
     "Tc_k2" => Dict("type" => "NormalPrior",  "args" => [1.0, 0.05, 0.7, 1.3]),
     "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0]),
     "b_k2"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  "transdim" => Dict(
     "max_kplanet"       => 2,
     "planets"           => true,           # toggle planet count
     "noise"             => false,
     "transdim_fraction" => 0.3,            # fraction of moves that are birth/death
     "birth_strategies"  => ["PriorBirth"], # or ["PriorBirth","InformedBirth"]
     "birth_weights"     => [1.0]),         # must match birth_strategies length
  "sampler" => Dict("name" => "transdim_ptemcee", "kwargs" => Dict(
     "n_temps" => 8, "n_walkers" => 80, "n_steps" => 500, "n_burnin" => 300,
     "informed_birth_fraction" => 0.5, "n_birth_tries" => 5,
     "show_progress" => false)),
  "output" => Dict("plots" => ["transdim_occupancy", "rv_phasefold",
                               "pm_phasefold", "corner"]),
)

summary = run_job(cfg)
@show summary["n_planets_posterior"]    # Dict "0"=>p0, "1"=>p1, "2"=>p2
```

`transdim_occupancy` renders P(N\_p = k). Available birth strategies:
`PriorBirth`, `InformedBirth`, `JointInformedBirth`, `DonorBirth`,
`MoMSBirth`. `informed_birth_fraction` / `n_birth_tries` help births land
on real signals rather than prior draws.

### Trans-dim **noise-model** selection

Set `transdim.noise => true` to additionally toggle activity models on
and off. The toggleable models must be `NoiseModel` objects, which JSON
cannot encode — so noise-model trans-dim is driven from the **in-Julia
API** rather than a JSON file (`run_job` accepts a pre-built `Dict` with
the actual `NoiseModel` objects, including
`noise_exclusion_groups` for mutually-exclusive activity models, e.g.
BIS-regression ⊥ GP-rotation). See the model-selection section in
`docs/JOB_CONFIG.md` and the `TransDimConfig` API.

---

## 10. SB2 double-lined binary + circumprimary planet — `BINARY_RV` / `BINARY`

An unresolved SB2 (double-lined CCF → two RV sets) fit as **one**
Keplerian with independent `K_A`/`K_B`, hosting a **circumprimary** RV
planet. Tag each RV point's stellar component with a reserved
`component` column (`1`/`A` = primary, `2`/`B` = secondary anti-phase);
the barycentric `gamma` is shared across A and B. Both components of one
epoch usually appear as two rows.

```julia
cfg = Dict(
  "version" => "1.0", "seed" => 3, "output_dir" => "results/sb2",
  "star"  => Dict("M_s" => 1.0),
  "data"  => Dict("rv" => Dict("values" => Dict(
     "bjd"        => bjd, "rv" => rv, "rv_err" => rv_err,
     "instrument" => inst_str,
     "component"  => comp))),                 # 1 = primary A, 2 = secondary B
  "model" => Dict(
     "max_kplanet"  => 2,
     "planet_modes" => ["BINARY_RV", "RV_ONLY"],  # binary (k1) + circumprimary planet (k2)
     "parametrization" => Dict("time" => "Tc")),
  "priors" => Dict(
     "K_A_k1" => Dict("type" => "UniformPrior", "args" => [300.0, 1500.0]),
     "K_B_k1" => Dict("type" => "UniformPrior", "args" => [300.0, 2200.0]),
     "K_k2"   => Dict("type" => "UniformPrior", "args" => [0.0, 80.0]),
     # + P_k1/Tc_k1/ecc_k1 (binary) and P_k2/Tc_k2/ecc_k2 (planet)
     ),
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
     "n_temps" => 10, "n_walkers" => 60, "n_steps" => 8000, "n_burnin" => 4000)),
  "output" => Dict("plots" => ["rv_timeseries", "rv_phasefold"]),
)
summary = run_job(cfg)
```

The `derived` table gains a **`binary_1`** block: `mass_ratio_q`
(= K_A/K_B = M_B/M_A), `M_A_sini3` / `M_B_sini3`, and — with astrometry
— absolute `M_A`/`M_B`. Plots come out SB2-aware automatically:
`rv_timeseries` is component-coloured with both the primary (K_A +
planet) and secondary (−K_B) curves, `rv_phasefold` on the binary slot
is the **double-lined** crossing fold, and on the planet slot it is
primary-only.

- **Transiting circumprimary planet:** use `["BINARY_RV", "RVPM"]` and
  add a `transit_photometry` block. The `dilution_<INST>` prior auto-
  frees (the transit is diluted by the companion's third light).
- **Absolute masses:** use `"BINARY"` (RV + astrometry) with an `hgca`
  (or `iad` + `gaia_dr3`) block; the system-level `f_light` prior (the
  secondary's astrometric-band light fraction, from the CCF flux ratio)
  scales the photocenter → absolute `M_A`/`M_B` + binary inclination.

---

## What every run produces

`run_job` writes into `output_dir`:

- **`summary.json`** — the science contract and the run's return value.
  Keys: `status`, `fitted` / `derived` (each `{conditioning, parameters}`
  where every parameter is `{value, err_lo, err_hi, ci3, unit,
  occupancy}`), `model_selection`, `run_info`, `log_z`, `n_evals`,
  `n_planets_posterior` (trans-dim only), `ppc`, `loo`,
  `detection_limits`, `fit_health`, `tables`, and `figures`
  (logical-name → PNG path).
- **`chains.nc`** — the full posterior cube (`load_chains` to reload).
- **`plots/`** — the figure tree: `models/` (data-space fits),
  `posteriors/`, `traces/`, `transdim/`, plus top-level `corner.png`,
  `ppc.png`, `detection_limits.png`.
- **`tables/`** — `fitted` and `derived` science tables in **csv,
  json, tex, ecsv, dat** (units, asymmetric 1σ + 3σ CI,
  model-conditioned).

Post-fit extras are **fail-soft** (a failure is logged and recorded in
the summary, never aborts the run):

- **PPC** (`output.ppc`, default on): posterior predictive check; the
  residual periodogram caps the grid at ≤ 20000 frequencies and uses an
  analytic FAP (so it stays fast on long-baseline RV).
- **Detection limits** (`output.detection_limits`): default on for PT
  samplers (`ptemcee`, `transdim_ptemcee`, `pt`, `pt_warm`) which keep
  the broad prior-seeded period coverage the curve needs; off otherwise.
- **PSIS-LOO / WAIC** (`output.loo`, default on; skipped under GP noise).
- **Fit-health guard** (`output.fit_health`, default on): flags
  silently-wrong posteriors (disjoint modes, railed bounds, corrupt
  log-post) without altering the chains.

Set `output.save_pdf => true` to emit a `.pdf` companion next to every
`.png`. Bound a pathological fit with top-level `timeout_sec`
(ignored under `juliacall` — enforce the timeout on the Python side
there).

---

## Preprocessing note

The configs above assume a clean light curve. Multi-sector TESS LC
cleaning (GP / notch detrending, transit masking, per-instrument 5σ MAD
outlier clipping) is a **prerequisite** to every joint RV+phot fit on an
active star, and is documented in [Preprocessing](preprocessing.md).
Photometry stays at native cadence end-to-end — never bin → fit →
interpolate.
