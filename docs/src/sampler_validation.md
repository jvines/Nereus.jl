# Sampler & pipeline validation

Production-path synthetic-recovery validation for Nereus's samplers
**and** its `run_job` output surface (every plot group, the science-table
contract, and the post-fit pipeline). As of **v0.2.0** the validation
suite covers two complementary layers:

1. **Sampler recovery** — does each sampler recover injected orbits /
   the planet count, or fail loud? (`test/validation/sampler_validation.jl`,
   `transdim_validation.jl`.)
2. **`run_job` plot surface + output contract** — does the production
   dispatcher emit every requested figure, recover the injected physics
   behind each science plot, and converge?
   (`test/validation/validate_runjob_*.jl`.)

The unit tests in `test/test_*_gaussian.jl` exercise each sampler's
**bare algorithm** against an analytic Gaussian. That proves the
sampling kernel is correct but says nothing about whether the sampler
recovers a planet when driven through the **real** `NereusTarget` +
`Params` + prior machinery — which is where this codebase's
silent-wrong bugs actually live (curved `(e, ω)` ridges, `threadid()`
buffer races, transform-Jacobian MAP pulls, multi-planet label
switching).

The harness in [`test/validation/sampler_validation.jl`](https://github.com/jvines/Nereus/blob/main/Nereus.jl/test/validation/sampler_validation.jl)
closes that gap. It injects a planet with Nereus's **own** forward
model (`rv_predictions`), runs each sampler on the production path
under a smoke budget, and scores two things:

- **Recovery** — is the injected truth inside the 95% CI of every
  scored parameter (`P`, `K`, `e` per planet)?
- **Convergence** — R-hat / ESS, computed with the *correct* diagnostic
  for the sampler's family (see below).

It also raises a **silent-wrong** flag: a tight CI that confidently
**excludes** the truth. That is the dangerous failure mode — a result
that looks converged and precise but is simply wrong.

## Purpose and scope

This is **synthetic-recovery validation on the production path**. It is
the second rung of a four-rung ladder:

1. **Bare-algorithm tests** (`test_*_gaussian.jl`) — kernel correctness.
2. **Production-path synthetic recovery** (this harness) — does the
   sampler recover known truth through the real target/prior stack?
3. **`run_job` plot surface + output contract**
   (`validate_runjob_*.jl`) — does the production dispatcher recover the
   injected physics behind each science plot, emit every figure group,
   and converge? See [`run_job` output-surface validation](#run_job-output-surface-validation).
4. **SBC + real-target validation** (next) — calibration of the full
   posterior (simulation-based calibration) and agreement with
   published orbits on real data (HD 18599, 51 Peg, GJ 876, …).

Every sampler in the matrix below now clears the **recover-or-fail-loud**
bar with **zero silent-wrong cells**. The five samplers that previously
produced silent-wrong output (`nested`, `nuts`, `pt_whitening`,
`population_annealing` / `smc`) were fixed; `map` is an honest point
estimate that fails loud rather than guessing. See
[What changed](#what-changed) for the per-sampler fixes.

## How to run

The harness is importable; the self-test only fires when the file is run
as a program (`abspath(PROGRAM_FILE) == @__FILE__`).

Run the built-in self-test (ptemcee on the easy target + nested_ins on
the eccentric target):

```bash
julia --project=Nereus.jl -t 8 \
    Nereus.jl/test/validation/sampler_validation.jl
```

Drive individual cells from the REPL or a matrix script:

```julia
include(joinpath(@__DIR__, "..", "test", "validation",
                 "sampler_validation.jl"))

# one (sampler, generator) cell
cell = run_one("ptemcee", "gen_rv_easy")
print_cell(cell)

# or build the pieces and score directly
ps, data, target, truth, key = gen_rv_eccentric()
cell = score_cell("nested_ins", "gen_rv_eccentric",
                  ps, data, target, truth, key)
```

Sweep the full matrix (every sampler × every generator):

```julia
samplers = ["nested_ins", "pt", "ptemcee", "nested_dynamic",
            "population_annealing", "smc", "nuts", "pt_whitening",
            "nested", "map"]
gens = ["gen_rv_easy", "gen_rv_eccentric", "gen_rv_2planet"]
matrix = [run_one(s, g) for s in samplers, g in gens]
print_cell.(matrix)
```

Each cell is a `NamedTuple` with `truth_in_95ci`, `silent_wrong`,
`silent_wrong_keys`, `max_rhat`, `rhat_kind`, `min_ess`, `logZ`,
`runtime_s`, and a `per_key` diagnostic dict.

### Generators

| Generator | Setup | What it stresses |
|---|---|---|
| `gen_rv_easy` | 1 planet, P=12 d, K=30 m/s, e=0.10, σ=2 m/s, 40 obs | High-S/N near-circular baseline ("must work or the sampler is broken") |
| `gen_rv_eccentric` | 1 planet, P=20 d, K=40 m/s, e=0.55, σ=8 m/s, 30 obs | Curved `(e, ω)`/`(K, e)` ridge — the geometry that silently broke warm-PT |
| `gen_rv_2planet` | P1=8/K1=25/e1=0.05, P2=30/K2=15/e2=0.20, σ=2.5 m/s, 60 obs | Multi-planet decomposition / label switching |

Period priors are loosely bracketed (always ≥2× either side of truth)
to keep smoke-budget samplers out of the alias forest; `K`, `e`, `Tp`,
`γ`, `σ` use the data-driven default priors. The brackets are **not**
cheating recovery — they are wide.

## Convergence diagnostics: family-aware R-hat

Convergence scoring branches on whether the sampler moves a population
of **coupled** walkers or runs **independent** chains. The harness tags
each sampler with `is_ensemble`:

- **Ensemble** (`ptemcee`, `pt_whitening`, `population_annealing` /
  `pa`, `smc`): walkers are coupled (affine-invariant stretch, tempered
  ensembles, annealed populations). `ptemcee` in particular exposes each
  *walker* as a separate `MCMCChains` "chain". A naïve
  `MCMCChains.rhat` on that object computes a **cross-walker** R-hat,
  treating N coupled walkers as N independent chains — the **wrong**
  diagnostic, and one that spuriously flagged ptemcee's *correct*
  recovery as non-converged. For ensemble samplers the harness instead
  reports:
  - **ESS** (`MCMCChains.ess`, valid across coupled chains), and
  - a **within-walker rank-normalized split-R-hat**: each walker's own
    trace is split in half and the halves become the "chains" fed to
    `MCMCChains.rhat`. This diagnoses within-walker stationarity without
    the invalid cross-walker comparison.
- **Independent** (`pt` (in-house / Pigeons), `nested*`, `nuts`):
  cross-chain R-hat is the correct diagnostic when there are ≥2 genuine
  chains (e.g. parallel NUTS). A single resampled NS chain has no
  cross-chain R-hat, so the harness reports `rhat_kind = :none` and
  relies on recovery + ESS.

The `rhat_kind` field records which diagnostic was used
(`:within_walker_split`, `:cross_chain`, or `:none`) so a matrix reader
never mistakes one for the other.

### Silent-wrong flag at the `e ≥ 0` boundary

The silent-wrong flag fires on a **tight CI that excludes truth**. For
near-circular truth (`e ≈ 0.05–0.1`) a thin posterior tail clipped by
the hard `e ≥ 0` bound used to trip the flag spuriously. The harness now
flags silent-wrong only when **both**:

1. the CI is tight (95% width < ~15% of `|truth|`; for `e`, width <
   0.16 absolute), **and**
2. the truth is excluded by a *meaningful* margin — **>10% of the CI
   width** — **OR** the violated CI edge is **not** pressed against a
   hard prior bound.

A small exclusion sitting against a hard bound (the `e ≥ 0` floor, a
`P`/`K` prior edge) is treated as a boundary artifact, not a confident
wrong answer. This keeps the flag catching the **real** silent-wrongs
(it is exactly what caught, before they were fixed, `pt_whitening`'s
K≈89 on trivial data and `nuts` railing `P` on the 2-planet target)
while sparing the boundary clip.

## Capability matrix

Results on the three RV generators above. Budgets are modest but
sufficient to resolve these generators (e.g. `sample_nested` at its
documented `n_live = 1500`, the budget its MultiEllipsoid bound needs to
resolve the eccentric ridge and the 2-planet decomposition); this is a
recovery check, not a stress test of each sampler's scaling.

The bar is **recover-or-fail-loud**: every sampler must EITHER place the
injected truth inside its CI with honest diagnostics, OR report a loud
failure (railed / non-converged / high R-hat). The one outcome that is
never acceptable is **silent-wrong** — a tight, confident CI that
excludes truth while passing naive R-hat / ESS. The current matrix has
**zero silent-wrong cells**.

| Sampler | Status | Notes |
|---|---|---|
| `ptemcee` | **Solid** | Recovers all three; within-walker split-R-hat ≈ 1.01–1.09, good ESS. |
| `pt` (in-house / Pigeons) | **Solid** | Recovers all three; reliable fixed-dim log Z. |
| `nested` | **Solid** | Recovers all three at `n_live = 1500` (`bounds = :multi`, `proposal = :rslice`); very high ESS (~10⁴). At `n_live = 400` the eccentric ridge silently mis-resolved — closed by the higher default. |
| `nested_ins` | **Solid** | Recovers all three; INS log Z stable. |
| `nested_dynamic` | **Solid** | Recovers all three with dynamic live-point reallocation. |
| `nuts` | **Solid (1-planet); honest-fail (2-planet)** | Recovers `easy`/`eccentric` cleanly (R-hat ≈ 1.005, ESS ~300) after the mass-matrix + warm-start fix. On 2-planet it recovers the values but R-hat ≈ 1.9 / low ESS — a **loud** non-convergence (label switching) surfaced via `nuts_diagnostics`, not a silent-wrong. Run more / longer chains for multi-planet. |
| `pt_whitening` | **Solid** | Recovers all three after the ensemble-per-temperature rebuild (tempered stretch within-temp + ensemble-fit whitening-flow swaps). |
| `population_annealing` / `pa` | **Solid** | Recovers all three after the eccentricity-bias fix in the mutation / resampling step. |
| `smc` | **Solid** | Same machinery as `pa`; recovers all three. |
| `map` | **Honest point estimate (fails loud)** | A MAP / Laplace point estimate, **not** a global sampler. The RV period basin is razor-thin, so its blind multistart cannot find the global mode on a multimodal RV posterior — it then reports `converged = false` / `railed = true` rather than a wrong answer. It recovers when warm-started near the mode (it is the right tool for *polishing*, not *searching*). **Never** consume its CI / Laplace evidence as a posterior. |

### What changed

Five samplers that previously produced **silent-wrong** output now pass
the recover-or-fail-loud bar:

- **`nested`** — eccentric ridge mis-resolved at `n_live = 400`; the
  default is now `n_live = 1500` with `:multi` bounds + `:rslice`
  proposal, and it recovers.
- **`nuts`** — a degenerate one-shot diagonal mass matrix + per-chain
  prior init gave no mixing and a chain-merge that *faked* recovery.
  Fixed with a warm-started, data-scaled mass matrix and divergence /
  step-size observability (`nuts_diagnostics`). 2-planet now fails
  **loud** (R-hat ≈ 1.9) instead of silently.
- **`pt_whitening`** — frozen one-walker-per-temperature with a
  whitening flow that could not move mass (tight wrong CI on trivial
  data). Rebuilt as an ensemble-per-temperature sampler.
- **`population_annealing`** / **`smc`** — a systematic
  high-eccentricity bias in the mutation / resampling step. Both recover
  after the fix.
- **`map`** — falsely reported `railed` on legitimate recoveries (a
  parameter resting on its physical floor — jitter → 0 — always trips a
  one-sided inward bound probe). Fixed with a physical-floor exemption,
  so map now reports `converged = true` when it genuinely finds the
  mode. It still **honestly fails loud** when its multistart cannot find
  the global RV mode — expected of a point estimator, not a bug.

## Trans-dimensional (model-selection) recovery

The matrix above certifies that, GIVEN the number of planets, each
sampler recovers the orbits or fails loud — its 2-planet generator
*freezes* `n_p=2`. It says nothing about whether the trans-dim samplers
find the right NUMBER of planets. The companion harness
[`test/validation/transdim_validation.jl`](https://github.com/jvines/Nereus/blob/main/Nereus.jl/test/validation/transdim_validation.jl)
closes that, one dimension up.

It injects a KNOWN number of planets `k ∈ {0, 1, 2}` — strong,
well-separated signals, so a correct sampler MUST resolve the model — and
tabulates the marginal model posterior `P(N_p=k)` from the `:n_planets`
column every trans-dim sampler stores (via the exported
`model_probabilities`). **Recover-or-fail-loud, one dimension up:** the
modal `N_p` must equal `k`. The trans-dim analog of silent-wrong is a
CONFIDENT WRONG model — `P(modal)` high AND `modal ≠ k_true` (a
hallucinated planet on the `k=0` pure-noise control, or a missed planet).
A BROAD model posterior is honest "can't tell" — fail-loud, acceptable.

| Sampler | k=0 (noise) | k=1 | k=2 | Status |
|---|---|---|---|---|
| `rjmcmc`            | P(0)=1.00 | P(1)=1.00 | P(2)=1.00 | **Recovers** |
| `pt` (with `td`)    | P(0)=0.99 | P(1)=0.99 | P(2)=0.98 | **Recovers** |
| `transdim_ptemcee`  | P(0)=0.91 | P(1)=0.96 | P(2)=0.89 | **Recovers** |
| `moms`              | P(0)=1.00 | P(1)=1.00 | P(2)=1.00 | **Recovers** |
| `moms_ns`           | P(0)=0.94 | P(1)=0.96 | P(2)=0.98 | **Recovers** |

All five recover the true planet count (and the injected periods) with
**zero confident-wrong-model cells**. The `k=0` control is the key
result: under MoMS's `inclusion_prior=0.5` the model prior actually
favours `n_p=1` (Binomial), so recovering `n_p=0` on pure noise tests
that the Occam/evidence penalty beats the prior — every sampler passes,
none hallucinates a planet.

`moms` requires `informed_birth_fraction > 0`. Unlike the ensemble
samplers it does **not** consult `td.birth_strategies`: a single chain
births only by a random walk around the parked off-values and cannot
reach a well-separated 2nd planet's period from the prior alone. At
`informed_birth_fraction=0` it confidently under-fit the 2-planet case
(silent-wrong); at `0.7` (matching the `InformedBirth` weight the others
get) it recovers. The gate `transdim_self_test()` validates the
model-recovery scorer on synthetic `n_planets` posteriors before any
sampler is trusted.

## `run_job` output-surface validation

The matrices above certify the *samplers*. A production pipeline
(exoautomata) does not call samplers directly — it calls
[`Nereus.run_job(cfg)`](@ref) with a JSON/`Dict` config and consumes
the returned summary plus the on-disk plot tree and science tables.
That whole surface is now validated by a family of drivers under
[`test/validation/validate_runjob_*.jl`](https://github.com/jvines/Nereus/tree/main/Nereus.jl/test/validation).
Each driver builds a **synthetic dataset from Nereus's own forward
model** (so the truth is exact), runs it through `run_job` requesting a
specific plot group, and asserts two things:

- **Recovery** — the injected physics behind that plot is recovered
  from `run_job`'s returned `"fitted"` / `"derived"` parameter tables
  (and, where relevant, `assess_fit` on the saved chain cube reports
  `:convergence => :ok` and is not flagged multimodal).
- **Emission** — every requested figure is actually written to the
  `run_job` plot tree (path exists, file is non-empty, `> 1000` bytes),
  in the real on-disk layout (`plots/models/…`, `plots/corner.png`,
  `plots/traces/`, `plots/posteriors/`, `ppc.png`, `tables/…`).

These run under a smoke sampler budget (`ptemcee`, 6–8 temps,
60–80 walkers, a few thousand steps) — the goal is to certify the
**dispatch path and the science behind each figure**, not to stress
sampler scaling. All drivers default to `"sampler" => "ptemcee"`
because it is robust to the correlated `K`/`e`/`b`/LD geometry that
made `nuts` non-converge on the same targets.

### How to run

Each driver is a standalone program. From the repo root:

```bash
julia --project=Nereus.jl -t 8 \
    Nereus.jl/test/validation/validate_runjob_plots.jl
```

Several drivers honour a `*_REPLOT=1` environment variable that
re-dispatches only the plots from the saved `chains.nc` (no re-fit) —
useful when iterating on figure styling:

```bash
RM_REPLOT=1     julia --project=Nereus.jl test/validation/validate_runjob_rm_plots.jl
AGP_REPLOT=1    julia --project=Nereus.jl test/validation/validate_runjob_agp_plots.jl
TTV_REPLOT=1    julia --project=Nereus.jl test/validation/validate_runjob_ttv_plots.jl
IADGOST_REPLOT=1 julia --project=Nereus.jl test/validation/validate_runjob_iad_gost_plots.jl
```

### Driver matrix

| Driver | Likelihood / mode | What it injects & asserts | Plot group emitted |
|---|---|---|---|
| `validate_runjob_plots.jl` | RV + transit (`RVPM`) | 1 planet `P=3, K=12, rr=0.10, b=0.30, e=0.2`; asserts `K` within 3 m/s, `rr` within 0.012, `assess_fit :convergence => :ok` and not multimodal | `RV_phasefold`, `rv_timeseries`, `Transit_phasefold`, `corner`, `ppc`, per-param `traces` (≥10) + `posteriors` (≥10); `tables/fitted.*` + `derived.*` in **csv/json/tex/ecsv/dat** |
| `validate_runjob_rm_plots.jl` | RV + transit, **Rossiter-McLaughlin** (`RVPM_RM`) | misaligned hot Jupiter, 3 densely-sampled spectroscopic transits + out-of-transit RV; asserts `v·sin i_*` within 25 %, `λ` within 12° | `rm_anomaly` (+ `rv_phasefold`, `corner`) |
| `validate_runjob_ttv_plots.jl` | RV + transit (`RVPM`), injected sinusoidal TTV | linear-ephemeris fit; asserts `P` within 0.01 d, `rr` within 0.02, **measured O−C correlates with injection (corr > 0.6)** via `measure_per_transit_tcs` | `ttv_oc`, `transit_overlay` |
| `validate_runjob_agp_plots.jl` | RV + BIS + FWHM, **ActivityGP** | planet `K=6` under ≈8 m/s of FF′-projected rotation activity; asserts `K` within 3 m/s (planet survives the activity GP) | `activity_gp_decomposition`, `activity_gp_latent`, `rv_components`, `rv_timeseries`, `corner` |
| `validate_runjob_astrom_plots.jl` | RV + relative astrometry (`RVAS`) | known orbit `P=900, e=0.30, inc=63°`; asserts `P/K/ecc` recovered **and `inc` within 6°** (astrometry-only → proves astrometry is in the fit), `assess_fit :convergence => :ok` | `orbit_skyplane`, `relastrom_residuals`, `relastrom_timeseries`, `rv_astrom_phasefold`, `corner` |
| `validate_runjob_iad_gost_plots.jl` | RV + **Hipparcos IAD** + **Gaia GOST** (`RVAS`) | real ε Eri (HIP 16537) modern RV + absolute astrometry; asserts `P` near ε Eri b (within 300 d); `REPLOT` path times the full post-fit pipeline (`ppc`/`detection_limits`/`loo`/`fit_health`) on the real 968-RV / 19-yr set | `iad_residuals`, `pm_anomaly`, `rv_timeseries`, `corner` |
| `validate_runjob_transdim.jl` | RV + transit, trans-dim (`transdim_ptemcee`, `N_p` 0..2) | strong 1-planet signal; asserts occupancy concentrates on `N_p=1` (`P(N_p=1) > 0.6`, modal) | `transdim_occupancy` |

All seven drivers are tracked in the repo and pass. The injected
forward model in each is built with `build_target` / `rv_predictions` /
`phot_predictions`, so the assertions are against Nereus's own physics.

### Rossiter-McLaughlin is now first-class

The RM validation closes the loop on the new `*_RM` modes. Four planet
modes carry an RM data source ([`src/model.jl`](https://github.com/jvines/Nereus/blob/main/Nereus.jl/src/model.jl)):

| Mode string | Sources | RM model |
|---|---|---|
| `RVPM_RM` | RV + PM + RM | Hirano+ 2011 analytic |
| `RVPMAS_RM` | RV + PM + AS + RM | Hirano+ 2011 analytic |
| `RVPM_RM_R` | RV + PM + RM_R | Reloaded (Cegla+ 2016) |
| `RVPMAS_RM_R` | RV + PM + AS + RM_R | Reloaded (Cegla+ 2016) |

(`RVPM_RM_TTV` also exists for joint RM+TTV.) RM adds two kinds of free
parameter:

- **`v_sin_i_star`** — a single **system-level** stellar projected
  rotation `V·sin(i_*)`, in **m/s** (`system_vsini`, slot
  `layout.systemic.v_sin_i_star`).
- **`lambda_k<k>`** — per-planet **sky-projected obliquity** `λ_k` in
  **radians** (`planet_lambda`, name `"lambda_kN"`); only present for a
  planet whose mode carries `:RM` / `:RM_R`.

The leading-order Hirano signal is `ΔRV_RM(t) = −Δflux(t)·v_p(t)` with
`v_p = V·sin(i_*)·(x_sky·cos λ − y_sin λ)` ([`src/rm.jl`](https://github.com/jvines/Nereus/blob/main/Nereus.jl/src/rm.jl));
the Reloaded variant integrates the intensity-weighted line-of-sight
velocity over the planet's sky-projected disk (`rm_reloaded_signal`,
21×21 sub-cells). **`rv_predictions` now includes the RM term** (decoded
once via `_decode_rm_state`, added per-epoch via `rm_contribution` in
[`src/likelihood.jl`](https://github.com/jvines/Nereus/blob/main/Nereus.jl/src/likelihood.jl)),
so PPC draws, residuals, and every RV plot are RM-consistent — not just
the dedicated `rm_anomaly` figure. The driver injects
`v·sin i_* = 6000 m/s`, `λ = 35°` and recovers both, confirming the RM
term is identifiable from in-transit RV.

To drive an RM fit, request a `*_RM` mode and add the two priors:

```julia
"model" => Dict("max_kplanet" => 1, "planet_modes" => ["RVPM_RM"], …),
"priors" => Dict(
   # …P_k1, Tc_k1, K_k1, rr_k1, b_k1 as usual…
   "v_sin_i_star" => Dict("type" => "NormalPrior",  "args" => [6000.0, 1500.0, 500.0, 20000.0]),
   "lambda_k1"    => Dict("type" => "UniformPrior", "args" => [-π, π])),
"output" => Dict("plots" => ["rm_anomaly", "rv_phasefold", "corner"]),
```

### New plot groups

Three plots are new this cycle, all dispatchable by name through
`run_job`'s `"output" => Dict("plots" => […])` list (and auto-selected
by `"plots" => ["auto"]` when their data is present):

- **`rm_anomaly`** (`plot_rm`, [`src/plotting/rm_plots.jl`](https://github.com/jvines/Nereus/blob/main/Nereus.jl/src/plotting/rm_plots.jl)) —
  the in-transit RV anomaly vs orbital phase for the RM-enabled planet,
  with the model band. No-ops unless a planet has an `*_RM` mode and a
  `v_sin_i_star` slot exists. Saved as `plots/models/rm_anomaly_K<k>.png`.
- **`transit_overlay`** (`plot_transit_overlay_fit`,
  [`src/plotting/ttv_plots.jl`](https://github.com/jvines/Nereus/blob/main/Nereus.jl/src/plotting/ttv_plots.jl)) —
  a per-transit QC gallery: every transit window overplotted with the
  model, for eyeballing depth/timing consistency. Requires photometry.
  Saved as `plots/models/transit_overlay_K<k>.png`.
- **`rv_components`** (`plot_rv_components`,
  [`src/plotting/rv_plots.jl`](https://github.com/jvines/Nereus/blob/main/Nereus.jl/src/plotting/rv_plots.jl)) —
  the RV **model decomposition**: the γ-subtracted RV with each
  Keplerian, the GP/AGP activity curve, and their Total as separate
  dense curves, plus a `data − Total` residual panel. **Distinct from
  `rv_timeseries`**, which is the activity-*decorrelated* view; this one
  keeps the raw data and shows the model pieces. In `"auto"` mode it is
  added only when there is structure to decompose (`max_kplanet ≥ 2` or
  an `ActivityGP`/`CovarianceNoise` model). Saved as
  `plots/models/rv_components.png`.

`ttv_oc` (the O−C diagram) now also renders for **single-planet** fits —
a data-only O−C with `planet_b_k = 0` — so a lone transiting planet
still gets its timing diagram.

### ActivityGP is run_job-drivable

`ActivityGP` (the FF′-style multi-channel activity GP) is now wired into
the `run_job` noise registry ([`src/runner.jl`](https://github.com/jvines/Nereus/blob/main/Nereus.jl/src/runner.jl),
`_NOISE_TYPES`). Declare it in `noise_models`:

```julia
"noise_models" => [Dict(
    "kind"        => "ActivityGP",
    "instruments" => ["HARPS"],
    "kwargs"      => Dict("channels" => ["bis", "fwhm"], "use_derivative" => true))],
```

`channels` is supplied as strings and symbolized internally
(`kw_sym[:channels] = Symbol.(…)`). **ActivityGP requires the indicator
errors** — supply them alongside the indicator values:

- **`values`-block RV** (in-config arrays): add an indicator array
  `<name>` plus a matching `<name>_err` array in the same `rv` →
  `values` block, e.g. `"bis"`/`"bis_err"`, `"fwhm"`/`"fwhm_err"`. Any
  key beyond `bjd`/`rv`/`rv_err`/`instrument` is treated as an
  indicator; a `<name>_err` key whose base `<name>` is also present is
  parsed as that indicator's 1σ (`_split_indicator_errs`).
- **CSV RV block**: use `indicator_cols` to name the indicator columns
  and provide a matching `<col>_err` column per indicator; the parser
  picks up `<col>_err` automatically.

The AGP driver injects a planet (`K=6 m/s`) buried under ≈8 m/s of
rotation activity shared across RV/BIS/FWHM and confirms the planet
survives the GP (`K` within 3 m/s), then that
`activity_gp_decomposition` and `activity_gp_latent` emit.

### PPC residual periodogram: bounded grid + analytic FAP

The PPC's "is there leftover periodicity in the residuals?" diagnostic
GLS used to default to a baseline-sized frequency grid with a 1000-resample
bootstrap FAP — intractable on real long-baseline RV (a 19-yr ε Eri set
drives ~3.5×10⁵ frequencies × 1000 bootstraps ≈ hours, blowing
`run_job`'s wall-clock). It now **caps the grid at ≤ 20 000 frequencies**
and uses the **analytic** FAP (LombScargle.jl's `fap`/`fapinv`, i.e.
`fap_method = :analytic`) rather than the bootstrap
([`src/diagnostics/ppc.jl`](https://github.com/jvines/Nereus/blob/main/Nereus.jl/src/diagnostics/ppc.jl):

```julia
nf = clamp(ceil(Int, (f_max - f_min) * T * 50), 64, 20_000)
gls_periodogram(…; n_freqs = nf, fap_method = :analytic, max_peaks = 5)
```

The analytic FAP is ≈3× over-confident — fine for a residual flag.
**Explicit, publication periodogram plots still use the full bootstrap
path.** The `iad_gost` driver's `REPLOT` branch times the whole post-fit
pipeline on the real ε Eri data to confirm `ppc` / `detection_limits` /
`loo` / `fit_health` all finish in sane wall-clock.

## Caveats and next rungs

- The sampler matrices are **production-path synthetic recovery** —
  orbits given `N_p` (fixed-dim matrix) and the planet count itself
  (trans-dim, `k ≤ 2`). They are **not** a guarantee at higher `N_p`.
- The `run_job` drivers now extend production-path recovery across the
  **transit, Rossiter-McLaughlin, TTV, relative-astrometry,
  Hipparcos-IAD + Gaia-GOST, and ActivityGP** likelihoods (each behind
  its science plot). They run under a smoke sampler budget — they
  certify the **dispatch path + the physics behind each figure**, not
  high-`N_p` or activity-heavy convergence at production budgets.
- `map` is a point estimate; its CI / Laplace evidence must never be
  treated as a posterior. Use it to polish a warm start, not to search.
- Remaining rungs: **real-target validation** against published orbits
  (51 Peg, GJ 876, HD 18599, HD 4747, HD 159062, WASP-47) — the
  `iad_gost` driver already fits **real ε Eri** for ε Eri b, but as a
  plot-path check, not a publication orbit — **evidence calibration**
  (TI+/SS+/H+ vs analytic log Z), and higher-`N_p` trans-dim recovery.
  (SBC — posterior coverage — is available as an uncommitted helper but
  is not part of this recovery-focused suite.)
