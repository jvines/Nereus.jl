# Sampler validation

What the repository actually tests about its samplers, and how a fit's
convergence is judged. There is no committed recovery-matrix harness in the
repository: earlier versions of this page described one
(`studies/validation/`), but it is not part of the published tree and its
results are not reproduced here. A sampler's rating below is therefore
what the tests in `test/` establish, nothing more.

## What the tests cover

| Layer | Test files | What they establish |
|---|---|---|
| Estimator arithmetic against analytic log Z | `test_evidence_gaussian.jl`, `test_evidence_curved.jl`, `test_bridge_evidence.jl`, `test_reference_path_evidence.jl` | TI, TI+, SS+, H+, bridge and reference-path evidence on exact draws from a Gaussian (and a curved, unit-Jacobian shear of it, 22-D for the bridge and reference-path tests). Mixing is not tested. |
| Evidence engines on the failure cases that broke them | `test_evidence_engines.jl`, `test_evidence_headline.jl` | What `sample_pt_emcee` reports as `log_evidence` and the engines (daedalus, nested family) on an RV + Hipparcos-IAD companion. |
| Sampler kernels and wrappers | `test_new_samplers.jl` | Smoke runs and argument handling of `sample_nested`, `sample_nested_ins`, `sample_nested_dynamic` (incl. the Higson allocation and the proposal kernels' invariance), `sample_smc`, `sample_pt_whitening` and the noise exclusion groups. |
| Reproducibility | `test_sampler_determinism.jl`, `test_phot_determinism.jl` | Chains are keyed on the task, never the thread, so `-t 2` and `-t 4` give the same chain at one seed. |
| Circular-angle seams | `test_seam_samplers.jl` | Every sampler moves a full-circle angle's seam off the posterior and the truth lies inside the one lobe. |
| Trans-dimensional noise swaps | `test_noise_swap_samplers.jl` | Every trans-dim sampler can cross between members of a noise exclusion group. |
| `run_job` plumbing, per engine | `test_ccf_runjob_samplers.jl`, `test_obliquity_runjob_samplers_{1,2,3}.jl` | Every engine runs a CCF or obliquity target through `run_job` briefly (plumbing, not convergence). |

None of these is a synthetic-recovery matrix at production budgets. Treat a
sampler as validated for a given problem only when you have checked its own
diagnostics on that problem (swap acceptance, R-hat, ESS, bridge against
mode-Laplace).

## Sampler status

- **`pt_emcee`** is the recommended tempered engine. Measured against
  `pt_whitening` on HD 114762, HD 159062 and HD 4747 at matched budgets it
  converged on all three (R-hat 1.001-1.006) where `pt_whitening` did not.
- **`pt_whitening`** is **deprecated** and not rated. Its whitened swap
  accepts about 8x less often than the identity swap it replaces (median
  0.027 vs 0.22), it loses to `pt_emcee` on every target and diagnostic, and
  with the flow off it *is* `pt_emcee`. It is kept for reproducibility and
  warns once when selected.
- The other engines (`pt`, the nested family, `pa`/`smc`, `nuts`, `map`,
  trans-dim) are covered by the tests above only; `map` is a point estimate
  and its CI or Laplace evidence is never a posterior.

## Convergence diagnostics: family-aware R-hat

Convergence is judged differently for a population of **coupled** walkers
and for **independent** chains.

- **Ensemble** (`pt_emcee`, `pt_whitening`, `population_annealing` / `pa`,
  `smc`): walkers are coupled (affine-invariant stretch, tempered ensembles,
  annealed populations). `pt_emcee` exposes each *walker* as a separate
  `MCMCChains` "chain", and `MCMCChains.rhat` on that object would compute a
  **cross-walker** R-hat that treats N coupled walkers as N independent
  chains, which is the wrong diagnostic. For these, fit health instead uses
  - **ESS** (`MCMCChains.ess`, valid across coupled chains), and
  - a **within-walker rank-normalized split-R-hat**: each walker's own trace
    is split in half and the halves are the "chains" fed to
    `MCMCChains.rhat`.
- **Independent** (`pt` (the in-house parallel tempering), `nested*`,
  `nuts`): cross-chain R-hat is correct when there are at least two genuine
  chains. A single resampled nested chain has no cross-chain R-hat, so
  recovery and ESS carry the verdict.

For tempered samplers also read the adjacent swap acceptance
(`res.acceptance_swap`): `sample_pt` and `sample_pt_emcee` warn when its
minimum is below 1%, which means replica exchange is effectively off across
that gap and more temperatures are needed.
