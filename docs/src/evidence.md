# Evidence

## Background — what's "log Z" and why do I care?

The **Bayesian evidence** Z is the average of the likelihood over the
prior. It's the number you compare across models to decide which one
the data prefers, regardless of how many parameters each has — bigger
Z means the model explains the data better after fairly penalising
model complexity. We always work with `log Z` (natural log) because
the raw Z spans many orders of magnitude.

How to read differences:
- `Δ log Z ≈ 1` — weak preference for the higher-Z model
- `Δ log Z ≈ 2.5` — moderate (Bayes factor ~12)
- `Δ log Z ≈ 5` — strong (Bayes factor ~150)
- `Δ log Z ≳ 10` — decisive

Throughout this page, "agreement to better than 0.01 in `log Z`"
means two estimators differ by less than 1% of a single log unit —
i.e. they would never disagree on which model is preferred.

## What Nereus provides

For **parallel tempering**, Nereus runs a four-estimator stack —
classical trapezoidal thermodynamic integration (TI) plus three
modern variants (TI+, SS+, H+) from [Peña & Jenkins 2026](https://ui.adsabs.harvard.edu/abs/2025arXiv250924870P/abstract)
(arXiv:2509.24870, A&A 706 A323, "Closing the evidence gap:
reddemcee"). All four are computed jointly on the same chain and
reported as a single `EvidenceReport`
(`src/samplers/evidence.jl:396`).

For **nested samplers**, `log Z` is produced natively by the
algorithm — see [`sample_nested`](samplers.md) and friends, which
return the Skilling estimator and (for `sample_nested_ins`) the Feroz
importance estimator `log_z_ins`. Those do **not** use the β-ladder
estimators on this page.

This page documents the **PT family** estimators in
`src/samplers/evidence.jl`.

### Which samplers produce a PT `EvidenceReport`

Every parallel-tempering sampler feeds the same
[`EvidenceAccumulator`](#the-evidenceaccumulator) and returns a
`log_evidence` scalar plus the full report:

| Sampler | Trans-dim? | log Z field | Full stack field | log Z headline |
|---|---|---|---|---|
| `sample_pt` (`backend=:nereus`, default) | both | 2nd tuple element | — | TI+ (`pt.jl:798`) |
| `sample_pt` (`backend=:pigeons`) | fixed-dim only | 2nd tuple element | — | Pigeons `stepping_stone` (`pt.jl:216`) |
| `sample_pt_warm` | both | 2nd tuple element | — | delegates to `:ptemcee`/`:nereus` |
| `sample_ptemcee` → `PTemceeResult` | fixed-dim | `.log_evidence` | `.evidence::EvidenceReport` | H+ if finite, else TI+, else TI (`ptemcee.jl:576`) |
| `sample_transdim_ptemcee` → `TransDimPTemceeResult` | trans-dim | `.log_evidence` | `.evidence_report` | H+ if finite, else TI+, else TI (`transdim_ptemcee.jl:1423`) |
| `sample_pt_whitening` → `PTWhiteningResult` | fixed-dim | `.log_evidence` | `.evidence::EvidenceReport` | H+ if finite, else TI+, else TI (`pt_whitening.jl:507`) |
| `sample_pt_hmc` | fixed-dim | 2nd tuple element | 3rd tuple element (`EvidenceReport`) | TI+ (`pt_hmc.jl:264`) |

Notes for a production reader:

- **Field name differs by sampler.** `PTemceeResult` and
  `PTWhiteningResult` expose the bundled stack as `.evidence`
  (typed `EvidenceReport`), whereas `TransDimPTemceeResult` exposes it
  as `.evidence_report` (typed `Any`, `transdim_ptemcee.jl:37`). The
  tuple-returning samplers (`sample_pt`, `sample_pt_warm`,
  `sample_pt_hmc`) put the scalar `log_evidence` in the tuple;
  `sample_pt_hmc` additionally returns the full `EvidenceReport` as a
  third element.
- **Headline estimator differs.** The in-house ensemble samplers
  (`ptemcee`, `transdim_ptemcee`, `pt_whitening`) pick **H+** as the
  headline `log_evidence` (falling back TI+ → TI when H+ is non-finite).
  `sample_pt` (nereus backend) and `sample_pt_hmc` report **TI+** as
  the headline. Whichever you trust, all four numbers are always in the
  report — read the one that suits your posterior (see
  [Practical guidance](#practical-guidance)).
- **`sample_pt` return shape is history-dependent.** Fixed-dim returns
  `(chains, log_evidence)`; a user-supplied `td::TransDimConfig`
  returns `(chains, log_evidence, n_evals)` (`pt.jl:166`).
- `sample_pt` with `backend=:pigeons` does **not** use this stack — it
  returns Pigeons' own `stepping_stone(pt)` evidence and has no
  trans-dim support.

## The `EvidenceAccumulator`

A streaming accumulator updated once per (post-warmup) MCMC iteration
during PT sampling (`src/samplers/evidence.jl:44`). Memory is
`O(n_chains)`, independent of sample count.

```julia
acc = EvidenceAccumulator(n_chains)
# inside the PT loop, once per iteration after warmup:
update_evidence!(acc, log_L_per_chain, betas)
# at the end:
report = evidence_report(acc, betas)
```

`log_L_per_chain[k]` is the current log-likelihood of the replica (or
walker-averaged log-likelihood) at inverse temperature `betas[k]`.

`acc` carries (`evidence.jl:44-51`):
- `n[k]` — sample count per temperature
- `sum_logL[k]`, `sum_logL2[k]` — running first and second moments of
  `log L` per temperature (mean and variance per β)
- `ss_lse_fwd[k]`, `ss_lse_bwd[k]` — running logsumexp accumulators
  for the SS+ geometric-bridge estimator, one per adjacent β pair
  (`n_chains − 1` of them; allocated to `-Inf`)
- `started` — guards against the 0/0 of an empty accumulator

`update_evidence!` (`evidence.jl:80`) accumulates `log L` and `log L²`
per chain, and the half-Δβ-scaled logsumexp legs per pair. It is fed
internally by every sampler in the table above. All numerically
sensitive sums go through `_logaddexp` (`evidence.jl:66`), which is
`-Inf`-safe.

## The four estimators

### TI — trapezoidal

Standard thermodynamic integration (reddemcee Eq. 5):

```
log Z = ∫₀¹ ⟨log L⟩_β dβ
```

Trapezoidal quadrature over the β-ladder. Baseline reference. Biased
by quadrature error proportional to `Δβ² · d²⟨log L⟩/dβ²`.

```julia
ti_trapezoidal(mean_logL_per_temp, betas) -> log_Z
```

`src/samplers/evidence.jl:113`. It integrates **ascending in β** —
the PT ladder is stored descending (β: 1→0, cold→hot), and integrating
in array order would yield `∫₁⁰ = −log Z`. The function sorts ascending
so the sign is `+log Z` regardless of ladder direction. (This sign-flip
was a real bug — `sample_ptemcee` once reported `−log Z`, inverting
every Bayes factor; caught by the toy-Z gate vs nested sampling on
2026-06-13 and now regression-tested.)

### TI+ — PCHIP-interpolated TI

reddemcee §2 (`src/samplers/evidence.jl:232`). Interpolate
⟨log L⟩(β) with a monotonicity-preserving cubic Hermite (PCHIP;
Fritsch & Carlson 1980, implemented inline at `evidence.jl:146`) and
integrate the smooth interpolant analytically over each Hermite
segment. Each segment integrates in closed form to
`(h/2)(yₖ + yₖ₊₁) + (h²/12)(dₖ − dₖ₊₁)` (`evidence.jl:193`).
Discretisation error is estimated by Richardson extrapolation on a
coarse ladder (every other β, anchored at the endpoints; only when
`n ≥ 5`).

```julia
log_Z, σ_Z = ti_plus(acc, betas)
```

Returns `(log_Z, σ_Z)` where `σ_Z = √(σ_D² + σ_S²)`:
- σ_D — systematic (`|Z_fine − Z_coarse|`)
- σ_S — statistical: trapezoidal-weight-propagated per-β chain SE on
  ⟨log L⟩ (assumes inter-chain independence; conservative)

**Pathology**: when `⟨log L⟩(β)` has wild curvature near `β = 0` (e.g.
trans-dim integrands averaging over many model configs), the PCHIP
coarse-grid estimate disagrees with the fine-grid by hundreds of log
units → σ_D blows up. Validated on HD 18599 Run 4 where σ_D = ±2 233.
SS+ is the better estimator in this regime.

### SS+ — geometric-bridge stepping stones

reddemcee Eq. 22 (`src/samplers/evidence.jl:300`). For each adjacent
pair `(β_k, β_{k+1})` with `Δβ = β_{k+1} − β_k`:

```
log r_k = log E_{β_k}[L^{Δβ/2}] − log E_{β_{k+1}}[L^{−Δβ/2}]
log Z   = Σ_k log r_k
```

Both expectations stream via logsumexp on the accumulator
(`ss_lse_fwd`/`ss_lse_bwd`). Halving the exponent (vs the one-leg
standard SS) reduces variance when adjacent temperatures are well
separated. Carries the same descending-ladder sign flip as the TI
integrators (`evidence.jl:315`).

```julia
log_Z, σ_Z = ss_plus(acc, betas)
```

`σ_Z` is a coarse `1/√n_min` bound (`evidence.jl:317`) — the per-pair
MC error on the logsumexp estimates is hard to estimate without
retaining samples; the bound is conservative.

**Pathology**: SS+ is the most numerically stable of the four on hard
posteriors. Use this number when the others disagree.

### H+ — hybrid TI+ ⊕ SS+

reddemcee Eq. 23–24 (`src/samplers/evidence.jl:340`). Find `β*` =
first interior knot where the integrand satisfies
`2 ⟨log L⟩_k ≥ ⟨log L⟩_{k+1}` (rectangle dominates triangle — TI's
quadrature error explodes past this knot). Use TI+ on `[0, β*]` (where
the integrand is well-behaved) and SS+ on `[β*, 1]` (where it's not),
each on a sliced copy of the accumulator, and add.

```julia
log_Z, σ_Z, β_star = hybrid_evidence(acc, betas)
```

If no `β*` exists (well-behaved integrand all the way), H+ falls back
to pure TI+ and returns `β_star = 1.0`. With fewer than 3 temperatures
it also degrades to TI+ (`evidence.jl:343`).

## `evidence_report` — the bundled output

`evidence_report(acc, betas)` (`src/samplers/evidence.jl:404`) returns
an `EvidenceReport` (`evidence.jl:396`):

```julia
struct EvidenceReport
    ti               :: Tuple{Float64, Float64}   # (log_Z, σ); σ=0 for TI
    ti_plus          :: Tuple{Float64, Float64}
    ss_plus          :: Tuple{Float64, Float64}
    hybrid           :: Tuple{Float64, Float64}
    hybrid_beta_star :: Float64                    # H+ knot location
end
```

Access the numbers directly: `report.ti_plus[1]` is TI+ log Z,
`report.ti_plus[2]` its σ; likewise `report.ss_plus`, `report.hybrid`,
`report.ti`; `report.hybrid_beta_star` is the H+ switch knot.

Pretty-printed via `show(::IO, ::EvidenceReport)` (`evidence.jl:414`):

```
EvidenceReport:
  TI  (trapezoidal) =    -5984.6
  TI+ (PCHIP)       =    -5984.6 ± 0.0281
  SS+ (geom-bridge) =    -5993.7 ± 0.0020
  H+  (β* = 1.000)  =    -5984.6 ± 0.0281
```

## Reading the evidence out of a real run

Pull the full stack off the result struct (ptemcee / pt_whitening):

```julia
res = sample_ptemcee(target, data; n_temps = 12)
res.log_evidence            # headline scalar (H+, or TI+/TI fallback)
ev  = res.evidence          # the EvidenceReport
ev.ti_plus, ev.ss_plus      # (log_Z, σ) pairs to compare
ev.hybrid_beta_star         # where H+ switched TI+→SS+
```

For the trans-dim ensemble sampler the field is named differently:

```julia
res = sample_transdim_ptemcee(target, data; td = td)
res.log_evidence            # headline scalar
ev  = res.evidence_report   # EvidenceReport (field typed Any)
```

For the tuple-returning samplers:

```julia
chains, log_z          = sample_pt(target)            # TI+ headline
chains, log_z, n_evals = sample_pt(target; td = td)   # trans-dim 3-tuple
chains, log_z, report  = sample_pt_hmc(target)        # report is EvidenceReport
```

### Through `run_job`

When you drive Nereus via `Nereus.run_job` (the JSON/Dict
dispatcher — see [JOB_CONFIG.md](https://github.com/jvines/Nereus.jl/blob/main/Nereus.jl/docs/JOB_CONFIG.md)),
every sampler result is normalised so `log_evidence` is always present,
and the run summary surfaces it as the JSON key **`log_z`**
(`src/runner.jl:1621`). Samplers with no evidence at all (e.g.
`map`, `nuts`, `rjmcmc`, `moms`) return `log_evidence = NaN`
(`runner.jl:1009`, `1015`, `1060`) and the value is written only when
finite (the JSON contract forbids raw `NaN`). Note that `pa`/`smc`
(population annealing) and the nested family (`nested`, `nested_ins`,
`nested_dynamic`, `moms_ns`) carry their own `log_evidence` that is
**not** from the β-ladder stack on this page — `pa`/`smc` report a
population-annealing free-energy estimate, the NS family report
Skilling/Feroz `log Z`. The
sampler names that route through the PT evidence stack are
`"pt"`, `"pt_warm"`, `"ptemcee"`, `"transdim_ptemcee"`,
`"pt_whitening"`, and `"pt_hmc"` (`src/runner.jl:201`, `939`–`1036`).

The same `log_z` is fed into the LOO/log-Z model-comparison summary
(`loo_compare_log_z`, `runner.jl:1525`) so cross-run Bayes factors are
consistent.

## Validation status

`test/test_evidence_gaussian.jl` validates all four estimators against
an analytic log Z (2D Gaussian likelihood × uniform-box prior, `log Z
= −d · log(2L)`):

| Estimator | Log Z | Error vs truth |
|---|---|---|
| TI  | -6.098 | 0.107 |
| TI+ | -5.985 | 0.007 |
| SS+ | -5.994 | 0.002 |
| H+  | -5.985 | 0.007 |

The test PASS bar (`test_evidence_gaussian.jl:264`, `283`) is each
estimator within `TOL = 0.5` nats of the analytic value **and** a
mutual spread `< 1.0` nat. All four pass; the mutual spread is ~0.11
in log Z. The estimators are mathematically correct. The file also
carries a regression guard for the descending-β sign flip
(`test_evidence_gaussian.jl:288`).

Independent cross-validation: TI+/SS+/H+ agree with the analytic
log Z to `< 0.01` nats on the 2D-Gaussian × box, and PT's SS+ agrees
with `sample_nested_ins`'s `log_z_ins` to ~5 log units on the 22-d
HD 18599 RVPM posterior. Earlier HD 18599 disagreements were
sampling/ladder-resolution effects, **not** estimator bugs — use
`≥ 10` PT temperatures for trustworthy numbers.

## Mode-Laplace — a phase-transition-immune cross-check

The tempered estimators above all integrate `⟨log L⟩` along the temperature
ladder. On a **high-SNR / signal-locked posterior** (a strong Keplerian, a
well-detected astrometric orbit) the tempered path has a *first-order phase
transition*: `⟨log L⟩(β)` jumps almost discontinuously near `β = 1`, one pair
of adjacent temperatures is stuck at ~0.6 % swap acceptance no matter how many
rungs you add, and TI / SS+ / H+ are all biased **low by 10³–10⁴ log units**.
Worse for model selection, the bias is *model-dependent* — it lives on the
model that contains the signal, not the noise-only null — so it does **not**
cancel in a `Δ log Z` Bayes factor.

`sample_ptemcee` therefore also computes a **mode-anchored Laplace** estimate
from its own cold chain:

```
log Z ≈ log q(mode) + (d/2)·log 2π + ½·logdet Σ_local
```

anchored on the max-`lp` cold sample (the *mode*, not the mean — a multimodal
mean falls in a density valley), with `Σ_local` the sample covariance of the
closest fraction of draws in unconstrained space. It **never tempers** (immune
to the phase transition) and uses the *sample* covariance rather than the
Hessian (robust to railed MAPs, where a Hessian Laplace goes singular / NaN).
It carries an `O(100)`-log-unit non-Gaussian bias on pathological (railed,
multimodal) posteriors — negligible for a strong detection, worth a nested
cross-check for a marginal one.

The result carries both:

```julia
res = sample_ptemcee(target, data; n_temps = 24)
res.evidence.ti_plus[1]      # tempered estimate (raw)
res.log_evidence_laplace     # mode-Laplace cross-check
res.log_evidence             # the one to USE (see rule below)
mode_laplace_evidence(target, res.chains)   # standalone, any cold chain
```

**Selection rule** (`laplace_switch_tol`, default 50 log units): if the
tempered best and the Laplace value **agree**, the tempered one is validated —
keep it (it is exact when it works). If they **disagree** by more than the
tolerance, the tempered estimator has hit a phase transition — `log_evidence`
auto-switches to `log_evidence_laplace` and a warning fires (reporting the
minimum swap acceptance as the corroborating signature). This keeps
`res.log_evidence` trustworthy for model selection at ensemble-PT speed. A
tiny `minimum(res.acceptance_swap)` that will not rise with more temperatures
is the independent tell-tale of the phase transition.

## Practical guidance

- **Fixed-dim PT on a single posterior**: trust TI+ and SS+ to agree
  within their σ. H+ should match. Take H+ (the ensemble samplers'
  default headline) — or TI+ from `sample_pt`/`sample_pt_hmc` — as the
  reported value.
- **Trans-dim PT**: the TI+ Richardson term blows up (PCHIP can't
  track the ⟨log L⟩(β) curvature when the integrand averages over
  multiple configs). **Trust SS+** (`res.evidence_report.ss_plus[1]`).
  A spread of ~1 000 log units across estimators is normal on hard
  trans-dim targets; the SS+ value is the calibrated one to report.
- **NS log Z cross-check**: compare PT's SS+ to `sample_nested_ins`'s
  `log_z_ins` as an independent estimate. Agreement to a few log units
  is confidence-building; large disagreement on a curved RV posterior
  usually means NS is the biased one (its rslice can't mix the thin
  e–ω–M₀ ridge) — see [sampler validation](sampler_validation.md).
- **Ladder length matters.** Use `≥ 10` temperatures; a too-coarse
  ladder inflates TI quadrature error and destabilises the SS+ pairs.
  `sample_pt_hmc` and `sample_ptemcee`/`sample_pt_whitening` support
  `adapt_ladder` to re-grid β by thermodynamic length (√Var log L),
  which minimises discretisation error (`pt_hmc.jl:191`,
  `ptemcee.jl:61`).
- **Bayes factors across runs**: compute as `Δ log Z`, with σ combined
  in quadrature. Use the **same estimator** for both runs (e.g. both
  SS+). Through `run_job`, compare the `log_z` summary keys. **If either
  run involves a high-SNR signal** (a strong planet vs. a noise-only null),
  the tempered estimators are biased *asymmetrically* and the factor is
  wrong — use `res.log_evidence` (which auto-switches to the mode-Laplace
  cross-check) or `res.log_evidence_laplace` directly; see
  [Mode-Laplace](#mode-laplace--a-phase-transition-immune-cross-check).

For per-configuration evidence in trans-dim noise-model selection, see
[Trans-dimensional](transdim.md) — the joint (N\_p, noise-config)
posterior occupancies are interpreted as a Bayes-factor matrix
(occupancy ≈ P(M|D)), which is a *model-selection* statistic, distinct
from the single-model β-ladder log Z on this page.
