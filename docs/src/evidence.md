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

Separately, `bridge_evidence` estimates log Z from **posterior samples
alone**, never touching the prior end of the ladder. On a signal-locked
posterior — where every β-path estimator on this page is biased — it is the
one to use. See [Bridge sampling](#bridge-sampling--evidence-from-the-posterior-alone).

This page documents the **PT family** estimators in
`src/samplers/evidence.jl`, plus the two cross-checks that do not temper
(mode-Laplace and bridge sampling).

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

**`summary.json` carries an `evidence` block**, not just `log_z`: every
estimator the run produced with its standard error, plus `reported` naming
which one `log_z` came from. Read it before quoting a number — on a measured
HD 18599 run the four tempered estimators spanned 80 nats while TI+ reported
`se = 186`, none of which a bare `log_z` would have shown.

⚠ **`run_job` fits could not compute bridge at all until 2026-08-27.** The
runner builds its target with `unconstrained = false`, so `target.transform`
is `nothing` and `sample_ptemcee` skips bridge — every job silently fell
through to the tempered stack, the one estimator that can sit >100 nats low
while its four variants agree to a decimal. Bridge is now recomputed after the
fit against an unconstrained target built from the same `params` and `data`
(chains untouched), and `log_z` is promoted to it **only when the chain
carries the effective sample size its reference needs**; otherwise bridge is
recorded with `bridge_not_reported` saying why. Trans-dim chains are skipped —
a single Gaussian reference over a fixed `R^n` is undefined across mixed
dimensions.

Measured on five HD 18599 fixed configurations: four shift by ≤ 6 nats between
tempered and bridge, and ActivityGP shifts by **59.9** (−1234.6 → −1174.7) —
the one badly mixed run (worst R-hat 3.91). If you have results from a
`run_job` fit predating this, re-run or recompute the evidence.

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

## Bridge sampling — evidence from the posterior alone

```julia
br = bridge_evidence(target, chains; proposal = :student)
br.log_z, br.se, br.overlap, br.converged
```

Every β-path estimator above integrates from the prior to the posterior, so all
of them depend on ⟨log L⟩ at the **hot** end. On a joint RV+transit fit that
expectation is `E_prior[log L]`, and it is tail-dominated: prior draws on
HD 18599 give a median log L of −4.9e4 with a **minimum of −1.5e7**. No ladder
resolves a quantity with that variance — measured, TI+ sat 118 nats from the
truth with a self-reported error of 0.48 nats, and adding rungs, cooling
`beta_min` or enabling the adaptive ladder all moved it **further away**. That
is a bias that does not shrink with computation and cannot be tuned out.

Bridge sampling never touches the prior end. It needs only draws you already
have from the posterior plus a proposal you can sample and evaluate, and costs
`n_proposal` likelihood evaluations rather than the ~1.3e7 a nested run spends
on the same model.

With `p*(θ)` the unnormalised posterior, `q(θ)` a normalised proposal,
`{θ_i} ~ p` and `{φ_j} ~ q`, the optimal bridge (Meng & Wong 1996) solves the
fixed point

```
           (1/N2) Σ_j  l2_j / (s1·l2_j + s2·r)
    r  =   ───────────────────────────────────  ,   l = p*/q,  s = N/(N1+N2)
           (1/N1) Σ_i    1   / (s1·l1_i + s2·r)
```

and `r → Z`, iterated in logs. Unlike the harmonic-mean estimator it has finite
variance; unlike Laplace it assumes nothing about Gaussianity — `q` only has to
**overlap** the posterior.

**Requirements.**

- `chains` must be the β = 1 chain in the target's own parameter space.
- The target must be `unconstrained = true`. The proposal is Gaussian on ℝⁿ, so
  on a bounded parametrisation most proposal draws land outside support and are
  wasted. If your chains are in bounded space, map them through
  `transform_forward` first — an `overlap` of 0.000 is the symptom of getting
  this wrong.
- `proposal = :student` (ν = 4) is the safer default over `:gaussian`. **An
  over-narrow proposal is the one failure mode that biases bridge sampling
  badly**, because the region where `q` has no mass never gets sampled; heavier
  tails guard against it on a skewed posterior.

**Read `overlap` and `converged` before using `log_z` — but do not mistake
`overlap` for a certificate.** It counts the fraction of proposal draws that
land in the target's *support*, which is not the same as the reference covering
the posterior's *mass*. On an RV fit whose period posterior spanned its whole
prior across 23 separate modes, `overlap` read **0.99** while a single Gaussian
reference was covering almost none of it. A low value is decisive evidence of
trouble; a high value is not evidence of health.

⚠ **Mode coverage is the real precondition, and neither this estimator nor
[reference-path](#reference-path-evidence--thermodynamic-integration-without-the-prior)
can check it for you.** Both fit ONE Gaussian/Student-t to the posterior draws,
so both are valid only when a single unimodal reference can cover the mass. On a
multimodal posterior — an undetected planet, an unconstrained period, competing
aliases — use nested sampling, which explores modes natively, or supply a
mixture reference. Diagnose it directly: look at the marginal of the parameter
you suspect (period, usually) and check it is not simply reproducing its prior.

⚠ **Bridge also needs an adequate *effective* sample size, and it fails
quietly without one.** It fits `q` from the posterior draws, so when those
draws are strongly autocorrelated the reference is estimated from far fewer
independent points than the draw count suggests — a `d`-dimensional Gaussian
reference has `d + d(d+1)/2` free parameters (135 at `d = 15`), and fitting
that from ~20 effective points gives a near-singular, badly wrong `q`.
Measured on a curved 15-dimensional target with an **exactly known** log Z,
varying only the chain's autocorrelation:

| `n_eff` | bridge error | reference-path error |
|--------:|-------------:|---------------------:|
| 20 000 (i.i.d.) | −0.009 | +0.001 |
| 1 166 | −0.012 | −0.030 |
| 146 | −0.400 | −0.008 |
| 22 | **−3.217** | +0.174 |

The bias is one-sided (always *low*) and reaches several nats before anything
in the returned diagnostics looks wrong.

Two things this is **not**. It is not the estimator re-using its training
draws: a split-sample variant, which fits `q` on one half and evaluates on
the other, still errs by −2.784 at `n_eff = 22`. And it is not mere
*displacement* of the reference — bridge is robust to that, absorbing a `q`
shifted by 1σ or mis-scaled from ×0.6 to ×2.5 with no measurable error. The
failure is specifically a `q` that is **poorly estimated**, and it hits
bridge and not reference-path because reference-path anneals *away* from `q`
along a β-path and so recovers from a bad reference, while bridge consumes
`q` directly.

**Check the chain's ESS before quoting a bridge value**, and prefer
[reference-path](#reference-path-evidence--thermodynamic-integration-without-the-prior)
when `n_eff` is not comfortably above `d(d+3)/2`.

⚠ **Adequate ESS is necessary, not sufficient — and there is an open
discrepancy on real targets.** On the analytic target above the two estimators
agree to hundredths of a nat once the chain is well mixed. On the real 51 Peg
posterior (1691 RVs, 15 parameters, `n_eff` ≈ 9000 against 135 parameters in
`q`) they do **not**: bridge reads −5899.4 and reference-path −5894.8, a gap of
**4.6 nats**. Both are internally converged and neither's error bar covers it:

  - bridge is stable across proposals — Gaussian −5899.480, Student-t
    −5899.383 (0.10 nats apart), `se` ≈ 0.03, `overlap` = 1.0000;
  - reference-path is stable across resolutions — −5894.773 at 1024
    particles, −5894.909 at 2048 with more steps (0.14 nats apart), ESS
    84–92%.

Ruled out as causes: low effective sample size, restricted support
(`support_frac` = 0.990), non-finite draws being filtered (0 of 80 000),
proposal tail weight, and chain mixing. The cause is not currently known, so
on a real multi-instrument RV posterior treat **a few nats as the systematic
floor** on any single prior-free log Z, quote which estimator produced it, and
do not read a Bayes factor of that size as meaningful unless both estimators
agree on it.

**Validation.** Error of **0.003 nats** against an analytic 22-dimensional
target. On the real HD 18599 RVPM posterior, `log_z = 11466.13 ± 0.02` with
`overlap = 1.000`, from both the Student-t and Gaussian proposals
(11466.13 vs 11466.12) — against a mode-Laplace cross-check of ~11467 and a
nested value of 11470, while TI+/SS+ on the same run sat 117–245 nats low.

## Reference-path evidence — thermodynamic integration without the prior

```julia
rp = reference_path_evidence(target, chains)
rp.log_z          # the number to quote (AIS)
rp.log_z_ti       # same particles, path integral — a MIXING check
rp.ess, rp.accept, rp.support_frac
```

Bridge sampling avoids the prior by not using a path at all. This does the
opposite: it keeps thermodynamic integration and **moves the cold end**.

Let `q` be a normalised reference fitted to the posterior — the same Gaussian /
Student-t that [`bridge_evidence`](#bridge-sampling--evidence-from-the-posterior-alone)
builds. Define

```
gamma_beta(y) = q(y) * exp(beta * f(y)),     f(y) = log p*(y) - log q(y)
```

so `gamma_0 = q` with `Z_0 = 1` — and `q` can be sampled **exactly**, so there is
no equilibration problem at the cold end — while `gamma_1 = p*` with `Z_1 = Z`.
Then

```
log Z = INT_0^1 E_{p_beta}[f] d beta
```

The path is short because `q` already resembles the posterior, so there is no
phase transition to resolve and no expectation is ever taken under the prior.
This is the fix for the failure documented above: it is the *path* that breaks
TI/TI+/SS+/H+ on a signal-locked target, not the estimators, which reproduce an
analytic log Z to <0.01 nats on a well-behaved one.

**Two estimates, and their difference is a diagnostic.**

- `log_z_ais` (reported as `log_z`) is annealed importance sampling,
  `log w = SUM_k (beta_k - beta_{k-1}) f(y_{k-1})`. It is unbiased in `Z` for
  **any** kernel leaving `p_beta` invariant, so it does not care how well the
  particles mix.
- `log_z_ti` integrates `E[f]` over the same particles. Path sampling **does**
  require equilibration at each rung.

Measured on the 22-D analytic target with a deliberately mismatched Student-t
reference, TI's error runs **-3.21 / -0.79 / -0.10 / -0.02** nats at
`n_steps` = 2 / 8 / 32 / 128, while AIS stays within 0.03 throughout. So a gap
between them means *raise `n_steps`* (and watch `ess` rise with it) — it is not
a second opinion on the arithmetic.

Contrast TI+/SS+/H+, which all read one shared `mean_logL` array: a bias at any
rung appears identically in all of them, which is why they agreed to 2.6 nats on
HD 18599 while all being 147 nats wrong.

**Defaults and requirements.**

- `proposal = :gaussian` here, the **opposite** of `bridge_evidence`, and
  deliberately. Bridge only needs `q` to overlap the posterior, so heavy tails
  are free insurance. Here `q` is the cold end of a path the particles are
  annealed along, so a much broader reference lengthens the path and costs
  equilibration — the Student-t reference needed ~16x more `n_steps` to bring TI
  into agreement, at no benefit to AIS.
- As for bridge, `chains` must be the beta = 1 chain and the target must be
  `unconstrained = true`; draws are mapped through `transform_forward` before
  `q` is fitted.
- `support_frac` reports the fraction of reference draws inside the target's
  support. `log_z_ti` is returned as `NaN` when that is below 1, because
  `E_q[f]` is then `-Inf` and the path integral is genuinely undefined. AIS is
  unaffected — an out-of-support particle simply carries `w = 0`.

**Validation.**

*Controlled target.* Error **0.0001-0.0015 nats** across seeds against the
analytic 22-D target (`test/test_reference_path_evidence.jl`), AIS and TI
agreeing to better than 0.05, `ess` 511/512. Critically, it also holds under a
deliberately MISMATCHED reference -- mean shifted 1 sigma, scale x1.5, x2.5,
x0.6 give -0.030 / -0.004 / +0.090 / -0.089 nats -- which is the case that
matters, because fitting `q` to draws from the target itself tests the reference
rather than the annealing.

*Real signal-locked target.* 51 Peg b (1691 RVs, 5 instruments, 15 free
parameters, P = 4.23080 +/- 0.00000 d, K = 56.16 +/- 0.24 m/s):

| estimator | log Z |
|---|---|
| TI+ | -6073.24 |
| SS+ | -6097.41 |
| H+ | -6073.24 |
| bridge (student) | -5896.91 +/- 0.03 |
| bridge (gaussian) | -5897.04 +/- 0.03 |
| reference-path (512 particles) | -5893.26 |
| reference-path (2048 particles) | -5893.08 |

**The tempered stack is ~176 nats low**, with TI+ and H+ agreeing to the second
decimal while both being wrong -- the failure this page describes, measured on a
real target rather than asserted. The two prior-free estimators agree to 3.8
nats and reference-path is stable in particle count (ESS 1929/2048).

That residual 3.8 nats between bridge and reference-path is unexplained: both
report sigma ~ 0.03 and both are stable, so it is not sampling noise. Treat a
Bayes factor resting on less than ~5 nats from either of them with suspicion
until it is resolved.

## Practical guidance

- **Fixed-dim PT on a single posterior**: take H+ (the ensemble samplers'
  default headline) — or TI+ from `sample_pt`/`sample_pt_hmc` — as the
  reported value.

  ⚠ **But mutual agreement between TI/TI+/SS+/H+ is NOT evidence that they are
  right.** All four read the same shared `mean_logL[k]` array, so a bias in
  ⟨log L⟩ at any rung appears identically in every one of them. On a measured
  HD 18599 run the four agreed with each other to **2.6 nats while all being
  147 nats wrong**. Corroborate against something that does not temper —
  `bridge_evidence` or the mode-Laplace value — before quoting a number,
  especially for a Bayes factor.
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
