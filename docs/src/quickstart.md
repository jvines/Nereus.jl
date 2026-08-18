# Quick start

This page is the shortest path from raw data to a posterior. It covers
two entry points:

1. **The library API** — `build_target` (or the explicit `Params + Data
   + NereusTarget` chain) plus a sampler such as `sample_ptemcee`. Use
   this from an interactive Julia session.
2. **`run_job(cfg)`** — a single config-driven dispatcher that builds
   the model, runs the sampler, renders the full plot suite, and writes
   `summary.json` + `chains.nc` to disk. This is the production entry
   point used by Dockerised batch workers and Python (`juliacall`)
   callers.

The page then walks three end-to-end fits — single-planet RV, a
trans-dim planet search, and a joint RV + transit photometry fit with
activity — followed by the new Rossiter–McLaughlin and ActivityGP
features. Nereus is version `0.2.0`.

## Installation

Nereus requires Julia ≥ 1.11. Get it from
[julialang.org](https://julialang.org/downloads/).

Clone the repo and activate the package environment:

```julia
using Pkg
Pkg.activate("Nereus.jl")
Pkg.instantiate()
```

For multithreaded sampling (recommended for any non-trivial fit),
launch Julia with explicit threads:

```bash
julia --project=Nereus.jl -t 8
```

Nereus's PT samplers parallelise across walkers / chains; the NS and
PA families parallelise across the init pass and (for PA) across the
replica population. `run_job` accepts an optional `n_threads` hint in
the config, but the actual thread count is governed by
`JULIA_NUM_THREADS` / the `-t` flag.

## The fastest result — `build_target` + `sample_ptemcee`

`build_target` collapses the boilerplate of wiring a
`Dict{String, PriorSpec}` with `_kN` / `_NAME` suffixes, picking the
`planet_modes` tag, and configuring the mass parametrization. You pass
NamedTuples; everything else is auto-derived. It returns a
`NereusTarget` ready for any sampler.

```julia
using Nereus

# Your RV data, one NamedTuple per instrument. Each `data` object just
# needs `.t`, `.rv`, `.rv_err` fields.
harps = (t = bjd, rv = rv, rv_err = rv_err)   # Vectors{Float64}

target = build_target(
    M_s     = 1.11,                            # M_sun (used for m·sin i)
    planets = (
        b = (
            P      = LogUniformPrior(2.0, 10.0),    # days
            K      = ModJeffreysPrior(0.1, 200.0),  # m/s
            sesinw = UniformPrior(-1.0, 1.0),
            secosw = UniformPrior(-1.0, 1.0),
            Mo     = UniformPrior(0.0, 2π),
        ),
    ),
    rv = (
        HARPS = (data = harps,
                 gamma = NormalPrior(0.0, 30.0, -100.0, 100.0),
                 sigma = LogUniformPrior(0.1, 30.0)),
    ),
)

# Ensemble parallel-tempering — robust to the correlated K/e geometry of
# RV posteriors. Returns a `PTemceeResult`.
res = sample_ptemcee(target, target.data;
    n_temps   = 8,
    n_walkers = 60,
    n_steps   = 3000,
    n_burnin  = 1500,
    seed      = 42,
)
chains = res.chains                     # MCMCChains.Chains, β=1, post-burnin

print_results(summarize_fitted(chains, target.params),
              summarize_derived(chains, target.params))
```

`build_target` auto-detected `planet_modes = [RV_ONLY]` (only RV data
present) and `:K_driven` mass parametrization (planet declares `P` and
`K`). `res.log_evidence` carries the best PT evidence estimate; the full
TI / TI+ / SS+ / H+ stack is in `res.evidence`.

!!! note "What `build_target` auto-derives"
    - `planet_modes` from which observation kinds are present
      (`RV_ONLY` / `PM_ONLY` / `RVPM` / `RVAS` / `RVPMAS`).
    - The mass parametrization: `:a_driven` if a planet declares `a` +
      `M_sec`, `:M_sec_driven` if `P` + `M_sec` (no `K`), else
      `:K_driven` (`P` + `K`).
    - The time anchor from the first planet's params: `:Tc` if `Tc`
      declared, `:Tp` if `Tp`, else `:Mo`.
    - Per-key namespacing: planet params get `_kN`, instrument params
      get `_NAME`.

    Override any of these with the `parametrization`, `time_anchor`,
    `M_s`, `R_s`, `stability`, `trend_order`, `phot_trend_order`,
    `noise_models`, or `priors` keywords. For full control (custom
    layout, sharing groups), use the explicit `Params + Data +
    NereusTarget` chain shown below.

A scalar passed where a `PriorSpec` is expected becomes a `FixedPrior`
(e.g. `M_sec = 0.01` fixes that planet's mass). `M_pri` likewise takes a
`Real` (fixed) or a `PriorSpec` (sampled).

## The fastest result — `run_job(cfg)`

`run_job` consumes a config (a JSON file path **or** an in-memory
`Dict`) and runs the entire pipeline. It writes `summary.json`,
`chains.nc`, and a `plots/` tree into `output_dir`, **and** returns the
summary `Dict` (so `juliacall` callers can skip disk).

```julia
using Nereus

cfg = Dict(
  "version"    => "1.0",
  "seed"       => 1,
  "output_dir" => "results/51peg",
  "star"       => Dict("M_s" => 1.11, "R_s" => 1.22, "T_eff" => 5800.0),

  "data" => Dict(
     "rv" => Dict("values" => Dict(
         "bjd"        => bjd,            # Vector{Float64}
         "rv"         => rv,
         "rv_err"     => rv_err,
         "instrument" => inst_str))),    # Vector{String}, one per epoch

  "model" => Dict(
     "max_kplanet"  => 1,
     "planet_modes" => ["RV_ONLY"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tp",
                               "ew" => "sesinw", "geom" => "b_rr")),

  # Optional — auto-generated priors are used for any key you omit.
  "priors" => Dict(
     "P_k1" => Dict("type" => "LogUniformPrior", "args" => [2.0, 10.0]),
     "K_k1" => Dict("type" => "ModJeffreysPrior", "args" => [0.1, 200.0])),

  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
      "n_temps" => 8, "n_walkers" => 60, "n_steps" => 3000,
      "n_burnin" => 1500, "show_progress" => false)),

  "output" => Dict("plots" => ["auto"]),
)

summary = run_job(cfg)        # also writes results/51peg/{summary.json, chains.nc, plots/...}
```

The returned `summary` (and `summary.json`) carries:

- `status` (`"ok"` / `"failed"`), `elapsed_sec`, `config_version`.
- `fitted` / `derived` — unit-tagged, model-conditioned parameter
  tables with median + asymmetric 1σ / 2σ / 3σ intervals (the
  authoritative science contract; `science_summary`).
- `log_z`, `n_evals`, and `sampler` acceptance diagnostics.
- `n_planets_posterior` — `P(N_p = k)` (trans-dim runs only).
- `ppc`, `loo`, `detection_limits`, `fit_health` blocks (all
  fail-soft — a failure here is logged and recorded, never aborts the
  run). `detection_limits` only runs for PT-family samplers.
- `figures` — a manifest mapping each rendered PNG's logical name
  (path under `plots/` without extension, e.g.
  `"models/rv_timeseries"`, `"transdim/occupancy"`) to its file path.
  Nereus stays S3-agnostic; the worker uploads these and rewrites
  the paths.

Science tables are also written under `tables/` in `csv`, `json`,
`tex`, `ecsv`, and `dat` formats.

### `run_job` config schema (essentials)

Full reference: [`docs/JOB_CONFIG.md`](https://github.com/jvines/Nereus.jl/blob/main/Nereus.jl/docs/JOB_CONFIG.md).
Keys accept either string or symbol form; the validator collects **all**
schema errors and reports them in one message before any heavy work.

**Required top-level keys:** `output_dir`, `sampler`, `model`, `data`.

**`data`** — at least one of:

| block | source forms |
|---|---|
| `rv` | `values` (`bjd`, `rv`, `rv_err`, `instrument` arrays) **or** `csv` (with `time_col`/`rv_col`/`err_col`/`inst_col`/`indicator_cols`) |
| `transit_photometry` | list of per-instrument LCs, each `values` (`bjd`, `flux`, `flux_err`) **or** `csv`; per-LC `instrument` and `exposure_time` (seconds, default 120) |
| `iad` | `hip_id` (auto-fetch Hipparcos IAD) **or** inline `values` |
| `gost` | `{ra_deg, dec_deg}` (auto-fetch) **or** inline `values` |
| `hgca` | `csv` (Brandt 2018 catalog) **or** inline 3-epoch `values` |
| `relastrom` | `csv` **or** inline `values` |
| `gaia_dr3` | `params`, `cov`, `t_ref` |

**`model`** — `max_kplanet` (required), `planet_modes`,
`parametrization` (`mass` ∈ {`K_driven`, `M_sec_driven`, `a_driven`};
`time` ∈ {`Tp`, `Tc`, `Mo`}; `ew` ∈ {`sesinw`, `ew`}; `geom` ∈
{`b_rr`, `b_r`}; `marginalize_gamma` bool), `stability` ∈ {`none`,
`amd`, `gladman`}, `trend_order` / `phot_trend_order` ∈ {0, 1, 2},
`ttv_n_transits`, `ttv_backend`.

**`planet_modes`** (known values): `RV_ONLY`, `PM_ONLY`, `RVPM`,
`RVAS`, `RVPMAS`, `RVPM_RM`, `RVPMAS_RM`, `RVPM_RM_R`, `RVPMAS_RM_R`,
`RVPM_TTV`, `RVPMAS_TTV`, `RVPM_RM_TTV`, `PM_TTV`, `RVPM_TTV_NB`,
`RVPMAS_TTV_NB`, `PM_TTV_NB`.

**`priors`** — `{ "<name>": {"type": "<PriorType>", "args": [...]} }`.
Types: `UniformPrior`, `LogUniformPrior`, `ModJeffreysPrior`,
`NormalPrior`, `FixedPrior`, `SinePrior`, `BetaPrior`. Any prior you
omit is auto-generated from the data bounds.

**`sampler`** — `{ "name": "<sampler>", "kwargs": {...} }`. Names:
`ptemcee`, `transdim_ptemcee`, `pt`, `pt_warm`, `pt_hmc`, `rjmcmc`,
`moms`, `moms_ns`, `nested`, `nested_ins`, `nested_dynamic`, `pa`,
`smc`, `pt_whitening`, `nuts`, `ofti`. The trans-dim samplers
(`transdim_ptemcee`, `rjmcmc`, `moms`, `moms_ns`) require a top-level
`transdim` block. JSON strings in `kwargs` that name Julia symbols
(e.g. `"prior"` for `init_strategy`) are coerced automatically;
unknown kwargs for the chosen sampler are rejected with an actionable
error.

**`transdim`** — `max_kplanet` (required), `birth_strategies`
(`PriorBirth`, `InformedBirth`, `JointInformedBirth`, `DonorBirth`,
`MoMSBirth`), `birth_weights` (same length), `transdim_fraction`
(default 0.3), `planets`, `noise`.

**`noise_models`** — list of `{ "kind": "<Kind>", ... }`. Kinds:
`CeleriteRotation`, `CeleriteSHO`, `CeleriteRotationFM17`,
`ActivityDecorrelation`, `ARModel`, `MAModel`, `ActivityJitter`,
`ActivityGP`. Each entry takes a `channel`, `instruments`, and a
`kwargs` sub-dict (see the ActivityGP section below).

**`output`** — `plots` (list, or `["auto"]`), `plot_kwargs`,
`save_pdf` (emit a PDF companion next to every PNG), and toggles
`ppc` / `loo` / `detection_limits` / `fit_health` (all default on,
sampler permitting).

**Known plot names** (`output.plots`): `rv_timeseries`,
`rv_components`, `rv_phasefold`, `pm_timeseries`, `pm_phasefold`,
`rv_astrom_phasefold`, `orbit_skyplane`, `iad_residuals`,
`hgca_pm_residuals`, `relastrom_timeseries`, `relastrom_residuals`,
`g23h_residuals`, `pm_anomaly`, `ttv_oc`, `transit_overlay`,
`rm_anomaly`, `activity_gp_latent`, `activity_gp_decomposition`,
`corner`, `trace`, `histograms`, `posteriors`, `transdim_occupancy`,
`posteriors_raw`, `posteriors_parameters`, `posteriors_histograms`,
`traces_grouped`, and `auto`. Use `auto` to let `run_job` pick the
relevant set from what's in `data`; each plot still no-ops when its
required data is absent.

Two optional top-level keys: `seed` (default 1) and `timeout_sec` (a
wall-clock budget on the sampler call; ignored under `juliacall`,
where you should enforce timeouts on the Python side).

---

## Example 1 — Single-planet RV characterisation

You have RVs for 51 Peg from several instruments and want a posterior on
(P, K, e, ω, m·sin i). Using the explicit constructor chain (the long
form `build_target` wraps):

```julia
using Nereus
using DelimitedFiles

# Load: time, rv, rv_err, instrument
raw = readdlm("51peg.csv", ',', Any, '\n'; header=true)
bjd      = Float64.(raw[1][:, 1])
rv       = Float64.(raw[1][:, 2])
rv_err   = Float64.(raw[1][:, 3])
inst_str = String.(raw[1][:, 4])

inst_names = sort!(unique(inst_str))
inst_map   = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst    = [inst_map[s] for s in inst_str]

data = Data(; t_rv=bjd, rv=rv, rv_err=rv_err, rv_inst=rv_inst)
ic   = InstrumentConfig(rv = inst_names)

params = Params(;
    max_kplanet  = 1,
    planet_modes = [RV_ONLY],
    instruments  = ic,
    data         = data,    # auto-generates priors from data bounds
    M_s          = 1.11,
)

target = NereusTarget(params, data; unconstrained = true)

# NUTS is the right tool here — gradient-based, fast on a unimodal
# 1-planet posterior. `sample_nuts` returns the chains directly.
chains = sample_nuts(target;
    n_samples = 2000,
    n_warmup  = 500,
    seed      = 42,
)

fitted  = summarize_fitted(chains, params)
derived = summarize_derived(chains, params)
print_results(fitted, derived)
```

That's it. Nereus auto-generated all the priors (P from LogUniform
over the observing baseline, K from ModJeffreys, γ bounded by the RV
range, etc.) and printed a paper-ready summary.

Override any prior by passing a `priors::Dict` to `Params`:

```julia
params = Params(;
    max_kplanet  = 1,
    planet_modes = [RV_ONLY],
    instruments  = ic,
    data         = data,
    M_s          = 1.11,
    priors       = Dict(
        "P_k1" => LogUniformPrior(3.0, 5.0),     # know it's near 4.23 d
        "K_k1" => LogUniformPrior(10.0, 200.0),
    ),
)
```

Detailed prior reference: [Priors](priors.md).

## Example 2 — Trans-dimensional planet search

You don't know how many planets are in the data. Nereus searches over
N\_planets ∈ {0, 1, …, max\_kplanet}.

```julia
params = Params(;
    max_kplanet  = 3,
    planet_modes = [RV_ONLY, RV_ONLY, RV_ONLY],
    instruments  = ic,
    data         = data,
    M_s          = 1.11,
)

target = NereusTarget(params, data; unconstrained = true)

td = TransDimConfig(;
    max_kplanet       = 3,
    birth_strategies  = [PriorBirth(), InformedBirth()],
    birth_weights     = [0.3, 0.7],
    transdim_fraction = 0.3,
)

chains, log_evidence, n_evals = sample_pt_warm(target;
    td               = td,
    n_pathfinder_runs = 16,
    n_rounds         = 16,
    n_chains         = 16,
    seed             = 42,
    within_model     = :rwm,
)

print_transdim_summary(chains, params; td = td, M_s = 1.11)
```

`InformedBirth` proposes new planets at periodogram peaks of the
residuals (so the chain lands on real signals fast, not random prior
draws), and PT swaps break cold-chain mode trapping. The output tells
you P(N\_p = k) for each k and the conditional posterior for each
configuration.

The same search via `run_job` — note `transdim_ptemcee` requires the
`transdim` block:

```julia
cfg = Dict(
  "output_dir" => "results/search",
  "data" => Dict("rv" => Dict("values" => Dict(
      "bjd" => bjd, "rv" => rv, "rv_err" => rv_err, "instrument" => inst_str))),
  "star" => Dict("M_s" => 1.11),
  "model" => Dict("max_kplanet" => 3,
                  "planet_modes" => ["RV_ONLY", "RV_ONLY", "RV_ONLY"]),
  "sampler" => Dict("name" => "transdim_ptemcee", "kwargs" => Dict(
      "n_temps" => 10, "n_walkers" => 100, "n_steps" => 5000, "n_burnin" => 2000)),
  "transdim" => Dict("max_kplanet" => 3,
      "birth_strategies" => ["PriorBirth", "InformedBirth"],
      "birth_weights" => [0.3, 0.7], "transdim_fraction" => 0.3),
  "output" => Dict("plots" => ["transdim_occupancy", "auto"]),
)
summary = run_job(cfg)
summary["n_planets_posterior"]    # => Dict("0"=>…, "1"=>…, "2"=>…, "3"=>…)
```

For trans-dim noise selection too, see [Trans-dimensional](transdim.md).

## Example 3 — Joint RV + transit photometry

You have RVs and a multi-sector TESS light curve for a transiting
planet. You want a single posterior over Keplerian + transit parameters
under a GP activity model.

```julia
# Cleaned LC from a sidecar — see Preprocessing for the multi-sector
# detrending pipeline.
using DelimitedFiles
lc = readdlm("HD18599_cleaned_lc.csv", ',', Float64; skipstart=15)
t_phot   = lc[:, 1]
flux     = lc[:, 2]
flux_err = lc[:, 3]

# Stack BIS indicator alongside RV (used by ActivityDecorrelation)
bis = ...   # one value per RV epoch
data = Data(;
    t_rv  = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    indicators = Dict("bisector_span" => bis),
    t_phot = t_phot, flux = flux, flux_err = flux_err,
    phot_inst = ones(Int, length(t_phot)),
)
ic = InstrumentConfig(rv = inst_names, pm = ["TESS"])

# GP rotation kernel on the RV channel
gp_rv = CeleriteRotation(channel = :rv)

# Anchor the planet period and Tc from the TESS ephemeris
priors = Dict{String, PriorSpec}()
priors["P_k1"]      = NormalPrior(4.1374, 1e-3, 4.13, 4.15)
priors["Tc_k1"]     = NormalPrior(2458354.59, 0.05, 2458354.0, 2458355.0)
priors["rho_s"]     = NormalPrior(2.24, 0.48, 0.1, 10.0)
priors["gp_period"] = NormalPrior(8.74, 0.1, 8.0, 9.5)   # external constraint

parametrization = ParametrizationConfig(
    time      = :Tc,
    geom      = :b_rr,
    use_rho_s = true,
)

params = Params(;
    max_kplanet     = 1,
    planet_modes    = [RVPM],
    instruments     = ic,
    data            = data,
    M_s             = 0.807, R_s = 0.798,
    parametrization = parametrization,
    priors          = priors,
    noise_models    = [gp_rv],
    trend_order     = 1,
)
target = NereusTarget(params, data; unconstrained = true)

# Parallel tempering with Pathfinder warmstart — production sampler
# for joint RV+phot+GP. Returns (chains, log_evidence, ...).
chains, log_ev, _ = sample_pt_warm(target;
    n_pathfinder_runs = 16,
    n_rounds          = 16,
    n_chains          = 16,
    seed              = 42,
    within_model      = :rwm,
)
print_results(summarize_fitted(chains, params),
              summarize_derived(chains, params))

# Plots
plot_rv_timeseries(chains, params, data; output = "results/")
plot_rv_phasefold(chains, params, data; planet = 1, output = "results/")
plot_pm_timeseries(chains, params, data; output = "results/")
plot_corner(chains, params;
            params_to_plot = ["K_k1", "sesinw_k1", "secosw_k1",
                              "gp_sigma", "gp_period", "gp_Q0"],
            output = "results/")
```

For trans-dim noise selection on this dataset (the canonical HD 18599
recipe from the paper), see [Worked examples](examples.md).

## Rossiter–McLaughlin (RM) fits

For a transiting planet with in-transit RVs, Nereus models the
Rossiter–McLaughlin anomaly (Hirano+ 2011 leading-order form; an
optional Cegla+ 2016 "Reloaded" disk-integration variant). Enable it by
giving the planet an RM-flavoured mode:

| mode | observations | RM model |
|---|---|---|
| `RVPM_RM` | RV + photometry | Hirano analytic |
| `RVPMAS_RM` | RV + phot + astrometry | Hirano analytic |
| `RVPM_RM_R` | RV + photometry | Cegla Reloaded |
| `RVPMAS_RM_R` | RV + phot + astrometry | Cegla Reloaded |

RM activates two extra parameters:

- `v_sin_i_star` — stellar projected rotation `V·sin(i_*)` in **m/s**
  (default prior `LogUniformPrior(500.0, 100_000.0)`).
- `lambda_k<k>` — the sky-projected stellar obliquity λ of planet `k`,
  in **radians** (default prior `UniformPrior(-π, π)`).

RM requires transit geometry, so the RM planet must carry a `_PM` mode
and you must supply either `rho_s` (via `use_rho_s = true`) or both
`M_s` and `R_s` so `a/R*` can be computed. The RM term is folded into
`rv_predictions`, so the PPC, residuals, and every RV plot are
RM-consistent — the in-transit anomaly is not a separate overlay.

```julia
params = Params(;
    max_kplanet     = 1,
    planet_modes    = [RVPM_RM],
    instruments     = ic,
    data            = data,
    M_s = 0.807, R_s = 0.798,
    parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                            use_rho_s = true),
    priors = priors,
)
```

Visualise the in-transit anomaly with the `rm_anomaly` plot
(`plot_rm`), or request `"rm_anomaly"` in `output.plots`. Via
`run_job`, set `"planet_modes" => ["RVPM_RM"]` (or `RVPM_RM_R` for the
Reloaded model).

## ActivityGP via `run_job`

`ActivityGP` is a multivariate activity GP (Rajpaul+ 2015): one latent
quasi-periodic process `G(t)` drives the RV and a list of activity
indicators jointly through per-channel linear combinations of `G(t)`
and `dG/dt`. It is now drivable straight from `run_job`:

```julia
"noise_models" => [Dict(
    "kind"        => "ActivityGP",
    "instruments" => ["HARPS", "FEROS"],     # empty ⇒ global
    "kwargs"      => Dict(
        "channels"       => ["bis", "fwhm"], # plus :halpha, :logrhk
        "use_derivative" => true))],
```

The GP needs the indicator **data and their errors**. Supply them
alongside the RVs:

- **inline `values`**: add a `<name>` array and a matching `<name>_err`
  array to the `data.rv.values` block, e.g. `"bis"`, `"bis_err"`,
  `"fwhm"`, `"fwhm_err"`. A `<name>_err` key is treated as the
  uncertainty for `<name>` only when `<name>` is also present;
  otherwise it is itself an indicator value.
- **CSV `rv` block**: list the indicators in `indicator_cols` and
  provide matching `<col>_err` columns.

ActivityGP **requires** the indicator errors — a config that omits them
fails validation. `G(t)` is unit-variance by construction, so there is
no kernel-amplitude parameter; all scale lives in the per-channel
couplings (`Vc`/`Vr` for RV, `Bc`/`Br` for BIS, `Fc`/`Fr` for FWHM,
`Lc` for logR'HK, `Hc`/`Hr` for Hα). Setting `use_derivative => false`
zeros the `dG/dt` couplings globally.

```julia
"data" => Dict("rv" => Dict("values" => Dict(
    "bjd" => bjd, "rv" => rv, "rv_err" => rv_err, "instrument" => inst_str,
    "bis" => bis, "bis_err" => bis_err,
    "fwhm" => fwhm, "fwhm_err" => fwhm_err))),
```

Request `"activity_gp_latent"` and `"activity_gp_decomposition"` in
`output.plots` to inspect the inferred latent process and its
per-channel decomposition (both no-op unless an `ActivityGP` is active).

## New diagnostic plots

Three additions worth knowing for QC of joint RV + transit fits:

- **`rv_components`** (`plot_rv_components`) — decomposes the RV model
  into its pieces (per-planet Keplerians, trend, GP/activity mean).
  Distinct from `rv_timeseries`, which shows the total model over the
  data. `auto` adds it whenever there are ≥2 planets or a smooth
  (GP/AGP) activity component.
- **`transit_overlay`** (`plot_transit_overlay_fit`) — a per-transit QC
  gallery overlaying the fitted model on each individual transit event.
- **`rm_anomaly`** (`plot_rm`) — the in-transit RM RV anomaly for
  RM-enabled fits (see above).

Two `run_job` robustness notes:

- The **PPC residual periodogram** now caps the frequency grid at
  ≤ 20 000 frequencies and uses the analytic FAP, so the post-fit check
  no longer takes hours on long-baseline RV (it previously ran a
  bootstrap FAP over an unbounded grid). Explicit periodogram plots are
  unaffected.
- **`ttv_oc`** now renders for single-planet fits too: with no perturber
  it shows the data-only O–C (measured per-transit Tc vs the linear
  ephemeris, `planet_b_k = 0`). With ≥2 planets it shows the N-body O–C
  envelope.

## What's next

- Detailed reference for any specific subject: pages in the side bar.
- One-page sampler comparison + when to use each: [Samplers](samplers.md).
- Full worked recipes: [Worked examples](examples.md).
- The complete `run_job` config schema:
  [`docs/JOB_CONFIG.md`](https://github.com/jvines/Nereus.jl/blob/main/Nereus.jl/docs/JOB_CONFIG.md).
