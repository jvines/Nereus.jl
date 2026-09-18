# HD 18599 (TOI-179) — trans-dimensional noise-model selection

Reproducible artifact for the real-data demonstration. Everything here runs
through `run_job`, so the analysis is a config plus a data file rather than a
study script.

## Contents

Everything needed to reproduce the analysis is **tracked in this directory**.
Sampler outputs are not: each run writes a ~300–800 MB `chains.nc`, so they go
to `results/HD18599_artifact/` (gitignored) and only the small reference
outputs are kept here under `reference/`.

| path | what it is |
|---|---|
| `input/hd18599_raw.csv` | the raw RV table, as delivered (137 rows) |
| `prepare_data.jl` | raw → `hd18599_clean.csv` (selection + clipping) |
| `hd18599_clean.csv` | the fitted data: 114 RVs, 3 instruments, 4 indicators |
| `job_transdim.json` | trans-dim noise-model selection → occupancy P(M \| D) |
| `job_fixed_<model>.json` | one fixed-configuration run per noise model → log Z |
| `run_fixed_configs.jl` | driver for the five fixed runs |
| `collect.jl` | assembles the results table from `results/HD18599_artifact/` |
| `reference/` | reference `summary.json` per run + the evidence table |

## Reproducing

Run from the repository root:

```sh
julia --project=. reproduction/hd18599/prepare_data.jl
julia --project=. -t 8 -e 'using Nereus; run_job("reproduction/hd18599/job_transdim.json")'
julia --project=. -t 8 reproduction/hd18599/run_fixed_configs.jl
julia --project=. reproduction/hd18599/collect.jl
```

Runtime: ~25 min for the trans-dim run, ~20 min per fixed configuration, on 8
threads. `prepare_data.jl` rewrites `hd18599_clean.csv` byte-identically, so a
`git diff` after running it should be empty — that is the first check that your
environment matches.

Results are reproducible from the seed in each config, and **independent of the
thread count**: sampler RNG streams are keyed per walker/replica/attempt rather
than per thread, so `-t 2` and `-t 8` give bit-identical chains. Verified on
this analysis — the trans-dim run reports the same accepted-toggle count
(`ntd_n = 3757`) across repeat runs.

## Data selection

From the raw table, in `prepare_data.jl`:

1. Drop `HARPS_POST` rows with provenance `ESO_PHASE3` — that pipeline reports
   `rv_error` in different units (a known catalogue defect). The `HARPS_POST`
   epochs survive through the other provenance.
2. Keep the three instruments the published analysis used.
3. 5σ MAD clip per instrument (robust scatter, not standard deviation).

137 raw rows → 115 after selection → 114 after clipping
(HARPS_POST 99, FEROS 8, HARPS_PRE 7).

Activity indicators are written **raw**; Nereus normalises them per instrument
internally, rescaling values and their errors together. The source table
reports no log R'HK uncertainty at all (0 of 114 rows), so indicator errors are
filled per instrument — median of the reported errors where any exist, else a
10% floor (0.1 × MAD), which is what the original analysis used after
normalising.

## Reading the two halves together

The **trans-dim run** gives occupancy P(M | D) from one chain. The **fixed
configurations** give an independent log Z per model. They answer the same
question by different routes, which is the point: neither is trusted alone.

Two cautions when reading the numbers:

- **Occupancy saturates.** When one model dominates by many nats, occupancy
  goes to 1.000 and carries no information about the margin — and the cold
  chain will show no model switches at all, because there is nothing to switch
  to. The margin lives in the fixed-config Δ log Z, not in the occupancy.
- **Compare log Z only between rows using the same estimator.** Each
  `summary.json` records an `evidence` block naming which estimator produced
  `log_z` and what the others said. The tempered stack (TI/TI+/SS+/H+) shares
  one `mean_logL` array, so those four agreeing with each other is not a
  cross-check.

Every configuration carries the always-on `IndicatorFloor`, so all of them
score the **same data**. Without it a mean-model (AD) and a joint RV+indicator
GP (ActivityGP) are fit to different likelihoods and their evidences are not
comparable.

## Reference outputs

`reference/` holds the `summary.json` from each run plus `EVIDENCE_TABLE.txt`,
produced by the code at the commit recorded in `reference/PROVENANCE.txt`.
Compare your `results/HD18599_artifact/*/summary.json` against them.

Expect exact agreement on `model_selection` (occupancy is a deterministic
function of the chain) and agreement to well within the quoted `se` on
`log_z`. If `evidence.reported` differs from the reference, you are on a
different code version — before 2026-08-27 `run_job` could not compute bridge
at all (it built its target with `unconstrained = false`), so older runs report
`ti_plus` instead, which differs here by up to 60 nats on the worst-mixed
configuration.
