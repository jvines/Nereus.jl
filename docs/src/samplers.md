# Samplers

Nereus ships a wide menu of samplers across six algorithmic families.
Most take the `(target, data; kwargs...)` (or `(target; kwargs...)`)
signature and return chains + diagnostics. This page is the reference
for which to pick, how to call each, and — crucially — the exact
`sampler.name` string `run_job` dispatches on.

Every claim below is grounded in `src/samplers/*.jl` and the
`_dispatch_sampler` map in `src/runner.jl`. Defaults shown are the
**actual function defaults**, not aspirational ones.

## run_job-dispatchable samplers

`Nereus.run_job` accepts a `sampler` block of the form
`{"name": "...", "kwargs": {...}}`. The recognised `name` strings
(`runner.jl`'s `_KNOWN_SAMPLERS`) are:

| `sampler.name` | function | fixed/trans-dim | needs `transdim` block | gives log Z |
|---|---|:---:|:---:|:---:|
| `ptemcee`          | `sample_ptemcee`          | fixed | — | yes (TI⁺/SS⁺/H⁺) |
| `transdim_ptemcee` | `sample_transdim_ptemcee` | trans | **required** | occupancy |
| `pt`               | `sample_pt`               | both  | optional (`td`) | yes (stepping-stone) |
| `pt_warm`          | `sample_pt_warm`          | both  | optional (`td`) | yes |
| `pt_hmc`           | `sample_pt_hmc`           | fixed | — | yes (TI⁺) |
| `pt_whitening`     | `sample_pt_whitening`     | fixed | — | yes |
| `rjmcmc`           | `sample_rjmcmc`           | trans | **required** | — (NaN) |
| `moms`             | `sample_moms`             | trans | **required** | — (NaN) |
| `moms_ns`          | `sample_moms_ns`          | trans | **required** | yes (NS) |
| `nested`           | `sample_nested`           | fixed | — | yes (NS) |
| `nested_ins`       | `sample_nested_ins`       | fixed | — | yes (INS) |
| `nested_dynamic`   | `sample_nested_dynamic`   | fixed | — | yes (dynamic NS) |
| `pa`               | `sample_pa`               | fixed | — | yes (PA) |
| `smc`              | `sample_smc`              | fixed | — | yes (→ PA) |
| `nuts`             | `sample_nuts`             | fixed | — | — (NaN) |
| `ofti`             | `ofti_sample`             | fixed | — | — (NaN) |

`_SAMPLERS_REQUIRE_TD = {transdim_ptemcee, rjmcmc, moms, moms_ns}` — these
error at dispatch if no `transdim` block is present. `pt`/`pt_warm` accept
trans-dim via a `transdim` block (passed as `td=`); the rest are fixed-dim.

!!! note "JSON kwarg coercion"
    `run_job` coerces **every** string in `sampler.kwargs` to a Julia
    `Symbol` (no sampler declares a `String` kwarg), so `"proposal":
    "rslice"` arrives as `:rslice`. It also validates kwargs against the
    dispatched function's signature for samplers in `_SAMPLER_FNS` and
    raises an actionable error listing the valid kwargs. (`pt_hmc` is in
    `_KNOWN_SAMPLERS` but **not** in `_SAMPLER_FNS`, so its kwargs are not
    pre-validated — they pass straight through.)

### Library-only samplers (not `run_job`-dispatchable)

These are exported and callable from Julia but have **no** `sampler.name`
and cannot be selected from a job config:

- `sample_ensemble` — affine-invariant ensemble (emcee-style).
- `sample_ess` — elliptical slice sampling.
- `sample_map` — L-BFGS MAP + Laplace evidence.
- `pathfinder_init` — Pathfinder warmstart / approximate posterior draws.

`sample_map` and `pathfinder_init` are primarily *internal* warmstart
machinery (used by `sample_pt_warm`, `sample_nuts`, `sample_ptemcee`'s
`:map_scatter`/`:pathfinder` init), but you can call them directly.

---

## Picking a sampler

| Problem | Use |
|---|---|
| Characterise a known planet, fixed N\_p, multimodal/weak signal | `sample_ptemcee` |
| Same, with gradients available, want clean TI⁺ evidence | `sample_pt_hmc` |
| Same, smooth/unimodal, gradients available | `sample_nuts` |
| Same, GP-heavy nuisance, Gaussian-ish prior | `sample_ess` |
| Same, emcee-style single-temperature ensemble | `sample_ensemble` |
| Multimodal astrometric / visual orbit (fixed-dim) | `sample_pt_warm` (`:auto` → ptemcee) |
| Trans-dim over N\_p (planet count) | `sample_transdim_ptemcee` |
| Trans-dim over noise configs (model selection) | `sample_transdim_ptemcee` (`td.noise=true` + `noise_exclusion_groups`) |
| Trans-dim variable selection without Jacobians | `sample_moms` |
| Trans-dim + per-config evidence (Bayes-factor matrix) | `sample_moms_ns` |
| Lightweight trans-dim baseline | `sample_rjmcmc` |
| Clean fixed-dim log Z (single run) | `sample_nested` |
| Evidence cross-check with INS / dynamic allocation | `sample_nested_ins` / `sample_nested_dynamic` |
| Visual-orbit fit (rel-astrom only, sparse arc, `:a_driven`) | `ofti_sample` |
| Quick MAP / Laplace evidence / seed for PT | `sample_map` / `pathfinder_init` |

For evidence on curved RV posteriors (eccentric ridges), prefer the PT/TI
stack (`sample_ptemcee`, `sample_pt_hmc`) over NS — ellipsoidal NS bounds
cannot conform to a strongly-curved likelihood ridge and the log Z bias is
structural there. See [Evidence](evidence.md).

---

## Rossiter-McLaughlin & PPC consistency

Several `planet_modes` now carry a Rossiter-McLaughlin (RM) source
(`src/model.jl`):

- `RVPM_RM`, `RVPMAS_RM` — Hirano+ 2011 analytic RM (`src/rm.jl`).
- `RVPM_RM_R`, `RVPMAS_RM_R` — the "Reloaded" RM variant.

These add two parameters per RM-enabled planet: `v_sin_i_star` (stellar
projected rotation, m/s) and `lambda_k<k>` (sky-projected spin–orbit
obliquity, rad). The RM anomaly is included **inside** `rv_predictions`
(`src/likelihood.jl`), so it is part of the likelihood every sampler
optimises — and therefore PPC, residuals, and RV plots are RM-consistent
out of the box (no special sampler handling needed). The dedicated
RM diagnostic plot is `rm_anomaly` (`plot_rm`).

This page does not re-document model/plot config — see
[Job config](JOB_CONFIG.md) — but note these sampler-adjacent additions:

- New plots: `rm_anomaly`, `transit_overlay` (`plot_transit_overlay_fit`,
  per-transit QC gallery), `rv_components` (`plot_rv_components`, RV model
  decomposition — distinct from `rv_timeseries`). `ttv_oc` now renders for
  single-planet fits (data-only O-C; `planet_b_k=0`).
- `ActivityGP` is `run_job`-drivable as a `noise_models` entry
  (`{"kind":"ActivityGP", "instruments":[...], "kwargs":{"channels":["bis","fwhm"], "use_derivative":true}}`).
  Indicator data + **errors** are required: add `<name>`/`<name>_err`
  arrays to an RV `values` block (e.g. `bis`/`bis_err`), or use
  `indicator_cols` + matching `<col>_err` columns in a CSV RV block.
- The PPC residual periodogram caps the grid at ≤ 20000 frequencies and
  uses an analytic FAP, so PPC no longer takes hours on long-baseline RV.

---

## Parallel tempering family

The PT family is Nereus's production workhorse for any multimodal
posterior. A β-ladder breaks cold-chain mode trapping, and evidence falls
out via the [Evidence](evidence.md) stack.

### `sample_ptemcee` — `name: "ptemcee"`

Parallel-tempered affine-invariant ensemble MCMC ([Vousden+ 2016](https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.1919V/abstract)).
Multiple Goodman-Weare stretch walkers per temperature. **Fixed-dim
only.** The production recoverer for fixed-N\_p multimodal / weak-signal
targets. Walkers live in **bounded** space.

```julia
res = sample_ptemcee(target, data;
    n_temps        = 5,
    n_walkers      = 100,              # auto-raised to ≥ 2·n_dim+2, made even
    n_steps        = 2000,
    n_burnin       = 1000,
    betas          = nothing,          # default: Vines+ (1/√5)^i, descending β[1]=1
    stretch_a      = 2.0,
    init           = nothing,          # bounded-space point; else init_strategy
    init_strategy  = :prior,           # :prior | :pathfinder | :map_scatter
    pf_n_runs      = 2,                # only used by :pathfinder
    seed           = 1,
    thin           = 1,
    show_progress  = true,
    adapt_ladder   = false,            # adapt β-ladder during the run
    ladder_adapt_window = 50,
    ladder_adapt_ν0     = 10.0,
    ladder_adapt_K      = 1.0,
    convergence_stop    = false,       # stop early on R̂/ESS gates
    science_params      = nothing,     # gating param set; default planet block + rho_s
    rhat_threshold      = 1.01,
    tail_ess_threshold  = 1000,
    n_converged_checks  = 3,
    min_steps           = 0,
    diag_every          = 0,
)
```

`init_strategy`:
- `:prior` **(default, astroEMPEROR-style)** — every walker at every
  temperature draws independently from the prior. Wide dispersion + swap-down
  from hot chains carries the high-likelihood region to the cold β=1 chain.
  The robust choice on multimodal / wide-prior targets.
- `:pathfinder` — `pathfinder_init` + per-walker draws (`pf_n_runs` basins).
  Tight cloud near one basin; collapses ensemble diversity when the
  Pathfinder MVN is a poor fit (Pareto k > 1).
- `:map_scatter` — `sample_map` MAP + small Gaussian scatter. Only safe
  when the posterior is approximately Gaussian around the MAP.

Returns a `PTemceeResult` with fields `chains`, `log_evidence` (best of
TI/TI⁺/SS⁺/H⁺), `evidence::EvidenceReport`, `acceptance_within`,
`acceptance_swap`, `n_evals`. Bump `n_temps` to 15+ when log-likelihood
spans more than ~2000 log units between prior bulk and posterior peak
(needed for tight evidence-estimator agreement).

**Target must be `unconstrained = false`** (bounded space).

### `sample_transdim_ptemcee` — `name: "transdim_ptemcee"` (requires `transdim`)

Trans-dim parallel-tempered ensemble with MoMS variable selection on
planet count (and, optionally, noise-model selection). This is the
production trans-dim recoverer (blind WASP-47 / K2-138 recovery target).

```julia
res = sample_transdim_ptemcee(target, data; td = td,
    inclusion_prior         = 0.5,      # Bernoulli prior P(γ_k = 1)
    moms_init_scale         = 1.0,
    informed_birth_fraction = 0.0,      # fraction of data-informed (BLS+LS) births
    target_birth_accept     = 0.234,
    n_birth_refine          = 0,        # per-slot refinement steps after birth
    n_birth_tries           = 1,        # multi-try births per move
    n_temps                 = 5,
    n_walkers               = 100,      # auto-raised to ≥ 2·n_dim+2
    n_steps                 = 2000,
    n_burnin                = 1000,
    betas                   = nothing,
    beta_min                = 1e-4,     # geometric ladder β_i = beta_min^(i/(N-1))
    stretch_a               = 2.0,
    seed                    = 1,
    thin                    = 1,
    show_progress           = true,
    noise_swap              = true,     # post-burn-in DB-correct noise-model swap
    noise_swap_rate         = 0.5,
    adapt_ladder            = false,
    ladder_adapt_window     = 50,
    ladder_adapt_ν0         = 10.0,
    ladder_adapt_K          = 1.0,
)
```

Requires `td.planets` **or** `td.noise = true`. Notes that matter in
practice:

- `informed_birth_fraction` — fraction of births using a data-informed
  proposal (`JointInformedBirth`: BLS photometry peaks + RV Lomb-Scargle,
  with depth→radius→mass→K and BLS-t0 anchoring; auto-falls back to RV-only
  `InformedBirth` when there is no photometry). **Critical for finding weak /
  arbitrary-period planets** — `0.0` gives blind RW births that rarely land
  on a real period. Set ≳ 0.5 for blind recovery.
- `n_birth_tries` / `n_birth_refine` — multi-try births and per-slot
  joint refinement (planet + jitter) to overcome the jitter-absorbs-the-dip
  ceiling that makes a true planet's conditional gain too small to lock.
- `beta_min` — densifies the **cold** end of the ladder as you add temps
  (closing the cold-rung swap gap) rather than pushing T\_max ever hotter.
- The MoMS planet move fires with probability `td.transdim_fraction` per
  (walker, temp); tempered M-H uses `β·Δlog L` so hot chains explore models
  more freely.

For honest noise model-selection, declare mutually-exclusive noise models
via `noise_exclusion_groups` on the `TransDimConfig` (occupancy collapses
to P(M\|D) only with the exclusion groups enforced).

**Target must be `unconstrained = false`.**

### `sample_pt` / `sample_pt_warm` — `name: "pt"` / `"pt_warm"`

In-house and Pigeons.jl-backed parallel tempering. `sample_pt_warm` adds a
warmstart layer and an `:auto` backend that picks the right engine per
problem. Both support fixed-dim **and** trans-dim (via a `td=`).

```julia
chains, log_ev = sample_pt(target;
    n_rounds       = 15,                # samples double each round (2^n)
    n_chains       = 10,                # β-ladder length
    seed           = 1,
    show_report    = true,
    td             = nothing,           # TransDimConfig | nothing
    init           = nothing,           # n_dim×n_chains matrix, or pathfinder_init NT
    early_stop_thresh = 0.0,
    early_stop_min_rounds = 8,
    within_model   = :slice,            # :slice | :rwm   (in-house :nereus backend)
    backend        = :nereus,          # :nereus (in-house, default) | :pigeons
    explorer       = :slice,            # :slice | :automala   (Pigeons backend only)
)
```

```julia
chains, log_ev, pf = sample_pt_warm(target;
    n_pathfinder_runs  = 16,
    n_pathfinder_draws = 0,             # 0 → max(2·n_chains, 200)
    n_rounds           = 15,
    n_chains           = 10,
    seed               = 1,
    show_report        = true,
    td                 = nothing,
    early_stop_thresh  = 0.0,
    early_stop_min_rounds = 8,
    within_model       = :slice,
    backend            = :auto,         # :auto | :ptemcee | :pigeons | :nereus
    init_strategy      = :pathfinder,   # :pathfinder | :prior
    # ptemcee-path budget (used when backend resolves to :ptemcee):
    n_temps            = 16,
    n_walkers          = 44,
    n_steps            = 8000,
    n_burnin           = 4000,
    adapt_ladder       = true,
)
```

`sample_pt_warm`'s `backend`:
- `:auto` **(default)** → `:ptemcee` for fixed-dim (`td === nothing`),
  `:nereus` for trans-dim. This is the recommended path.
- `:ptemcee` → prior-init parallel-tempered ensemble (`sample_ptemcee`),
  **no** Pathfinder. The prior-dispersed ensemble navigates sharp curved
  astrometric ridges at threaded speed and recovers the orbit robustly
  (HD 159062: a≈57, R̂<1.05, ~0.4 min). Fixed-dim only.
- `:pigeons` → seed Pigeons' adaptively-tempered PT with Pathfinder draws.
  Single-threaded for `NereusTarget` (~8× slower) and non-reproducible
  across seeds on HD 159062. Fixed-dim only.
- `:nereus` → seed the in-house RWM/slice PT with bounded-space draws.
  Required for trans-dim (`td !== nothing`); its fixed quadratic β-ladder
  and single-coordinate explorer struggle on ultra-sharp curved ridges.

!!! warning "Pathfinder false-convergence"
    On sharp curved astrometric posteriors the Pathfinder MVN warm-start
    *false-converges* the ensemble — it seeds every walker into one spurious
    basin and produces pristine-looking R̂/ESS at a wrong orbit. That is why
    `:auto`/`:ptemcee` deliberately drop Pathfinder. Use
    `init_strategy=:prior` to skip Pathfinder on the `:nereus`/`:pigeons`
    trans-dim path too.

`within_model` (`:slice`/`:rwm`) controls the in-house (`:nereus`) within-
temperature move; `explorer` (`:slice`/`:automala`) controls the Pigeons
backend explorer (`:automala` is AutoMALA, gradient-based). Returned log Z
is the Pigeons stepping-stone estimate; for TI⁺/SS⁺/H⁺ see
[Evidence](evidence.md). Run with `julia -t auto` — PT parallelises chains
over threads; single-threaded it runs the chains serially (`sample_pt`
warns).

`run_job` wraps `pt`/`pt_warm` into the standard result tuple, surfacing
only `(chains, log_evidence)` (the Pathfinder object is dropped).

### `sample_pt_hmc` — `name: "pt_hmc"`

Hamiltonian parallel tempering: NUTS-within-parallel-tempering for
fixed-dim, ForwardDiff-differentiable models. Returns the cold-chain
`Chains` (with `:lp`), the TI⁺ log-evidence, and the `EvidenceReport`.

```julia
chains, log_ev, report = sample_pt_hmc(target;
    n_temps            = 12,
    n_sweeps           = 1500,
    n_warmup           = 400,
    n_walkers_per_temp = 1,             # √M-tighter ⟨logL⟩_β at ~M× gradient cost
    swap_interval      = 1,
    target_accept      = 0.8,
    adapt_ladder       = true,          # pilot → re-grid β by √Var(logL) → final
    betas              = nothing,       # custom ladder; disables adaptation
    warm_start         = true,
    seed               = 1,
    progress           = true,
)
```

`adapt_ladder` runs a short pilot, re-grids β by thermodynamic length
(√Var logL), then the final run — minimising TI discretisation error.
**The target is internally forced to `unconstrained = true`** (it rebuilds
the target with `PackedTransforms` if needed), so you can pass either.
`run_job` surfaces `(chains, log_evidence)`.

### `sample_pt_whitening` — `name: "pt_whitening"`

NF-coupled PT with whitening (diagonal-Gaussian) swap proposals.
**Ensemble-per-temperature**: each temperature carries `n_walkers` walkers
moved by the tempered Goodman-Weare stretch move (as in `sample_ptemcee`);
adjacent-temperature swaps use a diagonal-affine *whitening* flow fit on the
per-temperature ensemble — the distinguishing feature. The affine flow has a
constant Jacobian that cancels, so the swap stays a standard tempered M-H
ratio at the transformed points.

```julia
res = sample_pt_whitening(target, data;
    n_temps        = 8,
    n_walkers      = 40,                # ≥ 2·n_dim+2, even
    n_steps        = 2000,
    n_burnin       = 500,
    betas          = nothing,           # default: (1/√5)^i geometric (same as ptemcee)
    stretch_a      = 2.0,
    proposal_scale = 0.05,              # accepted but UNUSED (ensemble stretch move)
    warmup_swaps   = 200,               # identity swap until whitening kicks in
    whiten_window  = 100,               # sliding window for the per-temp (μ,σ)
    whiten_refresh = 10,                # recompute (μ,σ) every N steps
    init_strategy  = :prior,
    seed           = 1,
    thin           = 1,
    show_progress  = true,
)
```

Returns a `WhiteningPTResult` (`chains`, `log_evidence`, `acceptance_within`,
`acceptance_swap`, `n_evals`). The whitening flow gives a 2–5× swap-
acceptance bump on multimodal posteriors with differently-scaled β
distributions. Experimental: the diagonal flow doesn't capture full
posterior correlation, so for paper-grade fits prefer `sample_ptemcee` /
`sample_pt_warm`.

**Target must be `unconstrained = false`.**

---

## Nested sampling family

### `sample_nested` — `name: "nested"`

Ellipsoidal nested sampling via NestedSamplers.jl
([Skilling 2006](https://ui.adsabs.harvard.edu/abs/2006BayAn...1..833S/abstract);
MultiEllipsoid bound, [Feroz+ 2009](https://ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F/abstract)).
The production single-run NS path for fixed-dim log Z. Importance-resampled
posterior from dead points.

```julia
chains, log_evidence = sample_nested(target, data;
    n_live     = 1500,                  # deliberately large; small n_live mis-resolves
    bounds     = :multi,                # :multi | :single | :none | :mlfriends
    proposal   = :rslice,               # :rwalk | :rstagger | :slice | :rslice | :unif | :hslice
    walk_scale = 1.0,
    n_walks    = nothing,               # nothing → max(25, 2·n_dim) for walk kernels
    slices     = nothing,               # nothing → max(5, 2·n_dim) for :rslice
    enlarge    = 1.25,
    dlogz      = 0.1,
    seed       = 1,
    parallel   = !_is_under_juliacall(),# threaded batch NS when safe
    batch_size = 0,                     # 0 → auto (nthreads when parallel, else 1)
)
```

!!! warning "Default proposal is `:rslice`, not `:rwalk`"
    On correlated RV ridges `:rwalk` takes isotropic steps inside the
    bounding ellipsoid, its acceptance collapses, the scale shrinks to a
    "stuck" floor, and you get **tight-but-wrong** credible intervals (e.g.
    e=0.47 with a tight CI when truth is 0.55; K1/K2 mis-decomposed in
    2-planet fits, surviving even `n_live=3000`). `:rslice` slices along
    random directions through the ellipsoid axes, tracks the local
    correlation, and recovers truth. Keep `:rslice` for RV fits.

`n_walks` and `slices` are dimension-scaled by default — a fixed count
under-decorrelates as dimension grows and biases log Z low. `:mlfriends`
(from the Nereus NestedSamplers fork) pairs only with `:unif`/`:rslice`/
`:slice`; the random-walk kernels are auto-redirected to `:rslice`.

With `parallel = true`, each iteration removes the `batch_size` lowest-
likelihood live points and replaces them via concurrent constrained walks
(`@threads :static`). `batch_size = 1` reproduces the serial sampler
bit-for-bit; keep `batch_size ≤ n_live/20` so the O(`batch_size`/`n_live`)
evidence bias stays well inside `logzerr`. `parallel` defaults to
`!_is_under_juliacall()` — it threads in a pure-Julia process but falls back
to serial under juliacall (Julia threads can deadlock on the Python GIL).

**Target must be `unconstrained = false`.**

### `sample_nested_ins` — `name: "nested_ins"`

Importance Nested Sampling
([Feroz+ 2019](https://ui.adsabs.harvard.edu/abs/2019OJAp....2E..10F/abstract)).
Operates in **unit-cube space**; every likelihood eval (incl. rejected)
contributes to the importance estimator `log Z_INS`, giving a second NS
log Z value beyond standard `log Z_NS`.

```julia
res = sample_nested_ins(target, data;
    n_live                = 400,
    bounds                = :multi,     # :multi | :single
    enlarge               = 1.25,
    dlogz                 = 0.01,
    max_iter              = 200_000,
    bound_update_interval = 50,
    n_walks               = nothing,    # nothing → max(25, 2·n_dim)
    walk_scale            = 1.0,
    proposal              = :rwalk,     # :rwalk | :slice | :unif
    min_X_shrinkage       = 10.0,
    seed                  = 1,
    verbose               = false,
)
```

Returns `INSResult(chains, log_z_ns, log_z_ins, n_iters, n_evals)`. These
are Nereus's own accumulator-aware kernels (NestedSamplers' proposals can't
feed the per-eval INS estimator). `run_job` surfaces `log_z_ins` as the
sampler's `log_evidence`. When `n_eff < 200` (peaked posterior), the chain
falls back to top-N dead points by weight rather than systematic resampling
— bump `n_live` to 2000+ for full posterior coverage.

**Target must be `unconstrained = false`.**

### `sample_nested_dynamic` — `name: "nested_dynamic"`

Dynamic NS
([Higson+ 2019](https://ui.adsabs.harvard.edu/abs/2019S&C....29..891H/abstract)).
Two-pass: a baseline pass with `n_live_init`, then an importance-targeted
batch pass with `n_live_batch` live points concentrated in the flagged
likelihood band.

```julia
res = sample_nested_dynamic(target, data;
    n_live_init           = 200,
    n_live_batch          = 400,
    bounds                = :multi,     # :multi | :single
    enlarge               = 1.25,
    dlogz_init            = 2.0,        # loose baseline stop
    dlogz_batch           = 0.5,        # tight batch stop
    pfrac                 = 0.8,        # importance blend (the f-knob)
    maxfrac               = 0.8,        # refine where weight > maxfrac × max
    pad                   = 1,
    max_iter              = 50_000,     # per pass
    bound_update_interval = 50,
    n_walks               = nothing,    # nothing → max(25, 2·n_dim)
    walk_scale            = 1.0,
    proposal              = :rwalk,     # :rwalk | :slice | :unif
    seed                  = 1,
    verbose               = false,
)
```

`pfrac` follows the Higson+ 2019 importance function (dynesty's
`weight_function`): a blend of posterior weight and evidence weight.
- `pfrac = 1.0` — pure posterior; concentrate at the typical set for tight
  parameter estimation.
- `pfrac = 0.0` — pure evidence; spread toward lower likelihood where the
  log Z variance lives. **Use this when the run is for evidence.**
- `pfrac = 0.8` — dynesty's posterior-leaning default.

Returns `DynamicNSResult(chains, log_z, log_z_baseline, log_z_batch,
n_iters_baseline, n_iters_batch, n_evals, target_L_lo, target_L_hi)`.
Same peaked-posterior resampling caveats as INS.

**Target must be `unconstrained = false`.**

---

## SMC / population annealing family

### `sample_pa` — `name: "pa"`

Population annealing
([Hukushima & Iba 2003](https://ui.adsabs.harvard.edu/abs/2003AIPC..690..200H/abstract)).
Tempered sequential Monte Carlo: at each β-step the population is reweighted,
resampled, and MCMC-mutated. Adaptive β-schedule via an ESS target.

```julia
res = sample_pa(target, data;
    n_replicas        = 500,
    n_mcmc            = 10,             # MINIMUM mutation sweeps per anneal step
    ess_target        = 0.9,           # adaptive β picks ESS/N = this (conservative crawl)
    max_steps         = 200,
    step_scale        = 0.05,          # :rwm only — RWM step as fraction of prior
    mutation_kernel   = :stretch,      # :stretch | :rwm | :adaptive_cov
    stretch_a         = 2.0,           # :stretch only
    mcmc_target_moves = 5.0,           # adaptive: sweep until each replica has ≥ this accepts
    max_mcmc_mult     = 12,            # cap: ≤ max_mcmc_mult × n_mcmc sweeps per β
    seed              = 1,
    show_progress     = true,
)
```

Returns `PAResult(chains, log_evidence, log_evidence_history, beta_history,
ess_history, acceptance, n_evals)`.

| `mutation_kernel` | Best for |
|---|---|
| `:stretch` (default) | Correlated low/mid-d — affine-invariant, no tuning |
| `:rwm` | Low-d sanity checks — collapses on high-d correlated posteriors |
| `:adaptive_cov` | ≥ 20-d correlated posteriors — refits population covariance each β step, Cholesky-shaped proposal, Robbins-Monro toward ~25% acceptance |

For HD 18599-scale targets use `:adaptive_cov`. Mode-jumping is still weak
(population covariance follows the dominant mode); for multimodal posteriors
prefer PT. The default `ess_target=0.9` (conservative temperature crawl) is
what fixes the old eccentricity bias and 2-planet e→1 railing — a slow crawl
lets the gradient-free mutation re-diversify the resampled population.

**Target must be `unconstrained = false`.**

### `sample_smc` — `name: "smc"`

SMC family interface
([Del Moral, Doucet & Jasra 2006](https://ui.adsabs.harvard.edu/abs/2006JRSSB..68..411D/abstract)).
Currently a thin wrapper over `sample_pa` that takes an explicit β sequence
and delegates. Only `length(betas)` is used (as a step count); the values
themselves are ignored because PA's ESS-adaptive scheduler picks the
β-spacing (the function `@info`-logs this). The `betas` must start at 0.0,
end at 1.0, and be sorted ascending.

```julia
res = sample_smc(target, data;
    betas           = range(0.0, 1.0; length = 51),  # default; only its length matters
    n_replicas      = 200,
    mutation_kernel = :adaptive_cov,
    seed            = 1,
)
```

All other kwargs forward to `sample_pa`. For ESS-adaptive scheduling, call
`sample_pa` directly — it's the SMC case Nereus uses most.

---

## MoMS family — variable selection

### `sample_moms` — `name: "moms"` (requires `transdim`)

Mixtures of Mutually Singular distributions
([van den Bergh+ 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260427791V/abstract),
arXiv:2604.27791). Trans-dim variable selection **without dimension jumps**:
each planet has an off-location in parameter space; the add move proposes via
RWM/slice near it, the delete move deterministically resets. Equivalent to
RJMCMC under an identity dimension map, with no auxiliary variables and no
Jacobian.

```julia
chains, n_evals, strategy = sample_moms(target, data; td = td,
    n_samples               = 5000,
    n_warmup                = 5000,
    seed                    = 1,
    n_chains                = 1,        # parallel chains via Threads.@spawn
    init_scale              = 0.3,      # initial proposal scale (fraction of prior width)
    target_birth_accept     = 0.234,    # Robbins-Monro warmup target
    inclusion_prior         = 0.5,      # Bernoulli P(γ_k = 1) — set deliberately for Bayes factors
    show_progress           = true,
    within_model            = :slice,   # :slice | :rwm
    informed_birth_fraction = 0.0,      # data-informed births (BLS+LS)
)
```

`strategy` (a `MoMSBirth`) carries the adapted scales for warm-starting a
follow-up. Any `birth_strategies` on `td` are ignored — `sample_moms` always
uses MoMS proposals. Best for **clean variable-selection probabilities**
(P(active = on)) when you don't need each model's evidence per se. For weak
signals raise `informed_birth_fraction` (validation needed 0.7 to lock the
planet count). `run_job` surfaces `log_evidence = NaN`.

**Target must be `unconstrained = false`.**

### `sample_moms_ns` — `name: "moms_ns"` (requires `transdim`)

MoMS on the joint (β, γ) spike-and-slab parameterisation, sampled by nested
sampling. NS bridging over the joint parameter+activation space gives
**per-configuration evidence** with global coverage (no mode-trapping by
construction).

```julia
chains, log_z, strategy = sample_moms_ns(target, data; td = td,
    n_live                  = 200,
    dlogz                   = 0.5,
    n_mcmc                  = 25,       # constrained-MCMC steps per dead-point replace
    init_scale              = 1.0,      # × IQR-stddev
    inclusion_prior         = 0.5,      # spike-and-slab γ-prior (log-Z normalisation)
    seed                    = 1,
    max_iter                = 200_000,
    show_progress           = true,
    progress_every          = 200,
    informed_birth_fraction = 0.0,
    batch_size              = 1,        # ≤ n_live/4
    warm_start_points       = nothing,  # e.g. from pathfinder_warmstart_moms_ns
)
```

Slower than `sample_moms` on a single fit (NS overhead), but the output is a
calibrated Bayes-factor matrix over candidate models. Noise toggling is
supported (RJMCMC-style prior-draw proposals). `run_job` surfaces the real
NS `log_z` as `log_evidence`. Validated on WASP-47.

`pathfinder_warmstart_moms_ns` builds the `warm_start_points` (MoMS
off-locations + scales) from Pathfinder draws to seed `sample_moms_ns` and
bypass the random-γ init that traps the chain in low-N\_p modes.

**Target must be `unconstrained = false`.**

---

## RJMCMC

### `sample_rjmcmc` — `name: "rjmcmc"` (requires `transdim`)

Classic Reversible-Jump MCMC (Green 1995). Birth/death + within-model moves;
supports parallel chains via `Threads.@spawn`. Lightweight trans-dim
baseline; for hard targets prefer `sample_transdim_ptemcee`.

```julia
chains, n_evals = sample_rjmcmc(target, data; td = td,
    n_samples     = 10000,             # total across all chains
    n_warmup      = 5000,              # per chain (adapt proposal scale)
    seed          = 1,                 # chain c uses seed + 1000·(c-1)
    n_chains      = 1,
    initial_scale = 0.01,              # initial Gaussian proposal scale
    target_accept = 0.234,             # within-model acceptance target
    within_model  = :slice,            # :slice | :rwm
    show_progress = true,
)
```

Multi-chain returns a single `Chains` with a populated `:chain` axis so R̂/ESS
work. `run_job` surfaces `log_evidence = NaN`.

**Target must be `unconstrained = false`.**

---

## Single-chain / fixed-dim MCMC

### `sample_nuts` — `name: "nuts"`

NUTS via AdvancedHMC.jl. Gradient-based. NUTS is a **local** sampler — it
cannot jump period-alias / disjoint modes — so chains are warm-started from a
short ptemcee global pre-search; on genuinely multimodal targets the chains
scatter, R̂ stays high, and `assess_fit` flags it (NUTS fails **loud**, never
silently merges).

```julia
chains = sample_nuts(target;
    n_samples     = 1000,              # post-warmup per chain
    n_warmup      = 1000,              # adaptation steps (keep ≥ ~500)
    n_chains      = 4,                 # ≥2 needed for meaningful R̂
    target_accept = 0.8,
    ad_backend    = :ForwardDiff,      # :ForwardDiff | :Enzyme | :ReverseDiff
    compile_tape  = true,              # :ReverseDiff only
    warm_start    = true,              # ptemcee pre-search (disjoint-mode fix)
    warm_temps    = 6, warm_walkers = 40, warm_steps = 400, warm_burnin = 200,
    init          = nothing,           # bounded-space point; overrides warm_start
    progress      = true,
)
```

`ad_backend`: `:ForwardDiff` (fastest ≤ 15 params), `:Enzyme` (reverse-mode,
better for many params / GP models), `:ReverseDiff` (needs `import
ReverseDiff`; `compile_tape=true` for ~2–3× speedup). Per-chain divergences /
step size / tree depth attach to `chains.info`. `run_job` surfaces
`log_evidence = NaN`. **Target must be `unconstrained = true`** (auto-rebuilt
internally if needed).

### `sample_ensemble` *(library-only)*

Affine-invariant ensemble MCMC (Goodman & Weare 2010) via
AffineInvariantMCMC.jl. Single-temperature — won't break mode trapping; for
multimodal use `sample_ptemcee`. Works in unconstrained space.

```julia
chains = sample_ensemble(target;
    n_walkers = 50,                    # auto-raised to ≥ 2·n_dim+2
    n_steps   = 5000,
    n_burnin  = 1000,
    thinning  = 1,
    init      = nothing,               # else prior-draw walkers (EMPEROR-style)
    seed      = 1,
    n_chains  = 1,                     # parallel chains via Threads.@spawn
)
```

### `sample_ess` *(library-only)*

Elliptical slice sampling
([Murray, Adams & MacKay 2010](https://ui.adsabs.harvard.edu/abs/2010arXiv1001.0175M/abstract))
via EllipticalSliceSampling.jl. Uses a Gaussian approximation to the prior
centered on `init` (or the prior midpoint). Best for characterisation near a
known mode, especially with GP noise. Works in bounded space.

```julia
chains = sample_ess(target, data;
    n_samples = 5000,
    n_burnin  = 1000,
    init      = nothing,               # center of the Gaussian prior approximation
    seed      = 1,
    n_chains  = 1,
)
```

Note the `(target, data)` signature — `data` is required.

---

## Orbit-specific

### `ofti_sample` — `name: "ofti"`

OFTI rejection sampling
([Blunt+ 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....153..229B/abstract);
orbitize!-equivalent 2-pass calibrated). For each candidate orbit drawn from
priors, "scale-and-rotate" to exactly match the first relative-astrometry
epoch, then accept ∝ exp(log-lik) for the rest. Keplerian-only,
**rel-astrom only, no joint RV**. Naturally explores all modes of a sparse
visual orbit. **Requires `parametrization.mass = :a_driven` and ≥1 relAST
epoch.**

```julia
chains = ofti_sample(target;
    n_attempts    = 100_000,
    n_calibrate   = 0,                 # pass-1 calibration draws (fixes log_lik_max ref)
    buffer        = 0.0,
    planet_idx    = 1,
    epoch_idx     = 1,
    seed          = 42,
    show_progress = false,
)
```

Returns accepted draws as `MCMCChains.Chains`; the diagnostics
(`n_in_bounds`, `n_finite`, `n_accepted`, `n_overshoot`, `log_lik_max_cal`)
are logged via `@info` and attached to the chain's `info`. `run_job`
dispatches `ofti_sample(target; kw...)` and surfaces `log_evidence = NaN`.

---

## Optimisation / warmstart *(library-only)*

### `sample_map`

L-BFGS MAP + Laplace evidence, with multi-start parallelism. An **honest
point estimator** — it never returns a credible interval, and reports
`converged=false` / `railed=true` when the optimum is untrustworthy
(bound-pinned or no dominant basin). Not a global RV search.

```julia
res = sample_map(target;
    init       = nothing,              # unconstrained initial guess
    method     = :LBFGS,               # :LBFGS | :BFGS | :NelderMead
    maxiter    = 10000,
    g_tol      = 1e-8,
    n_starts   = 32,                   # multistarts fanned across threads
    seed       = 42,
    bound_rtol = 1e-3,                 # "at a bound" tolerance (directional railing test)
    basin_rtol = 0.02,                 # two optima = same basin within this
    dom_margin = 50.0,                 # must beat runner-up basin by this to be `converged`
)
```

Returns `MAPResult(x_map, log_posterior, log_evidence_laplace, hessian,
param_names, converged, railed, railed_params, n_basins, dominance)`. The
Laplace `log_evidence_laplace` is a sanity-check evidence, not a substitute
for PT/NS. A railed parameter resting on a *physical floor* (jitter ≈ 0 with
the objective minimised there, inward gradient) is **not** flagged.
**Target must be `unconstrained = true`.**

### `pathfinder_init`

Pathfinder.jl wrapper (Bayesian L-BFGS,
[Zhang+ 2022](https://ui.adsabs.harvard.edu/abs/2021arXiv210803782Z/abstract)).
Approximate posterior draws via a sequence of normal approximations along the
L-BFGS trajectory. Used internally by `sample_pt_warm`, `sample_nuts`
warmstart, and `sample_ptemcee`'s `:pathfinder` init; call it directly for
the Pareto-k-vetted draws.

```julia
result = pathfinder_init(target;
    n_runs  = 8,                       # independent L-BFGS basins (≥16 for multimodal)
    n_draws = 100,                     # samples after ESS reweighting
    seed    = 42,
    quiet   = true,                    # one-line summary instead of chatty PSIS @warn
)
```

Returns a NamedTuple with `draws` (IS-reweighted), `per_run_draws` (one draw
per L-BFGS run — preserves basin diversity even when `pareto_shape > 0.7`),
`fit_distribution` (uniform mixture of per-run MVNs), `psis_result`
(`pareto_shape > 0.7` ⇒ poor approximation), and the raw `pathfinder_result`.
Pass `.draws` / `.per_run_draws` columns as the `init` for NUTS / PT / OFTI.
`pareto_shape > 1` ⇒ prefer prior init instead.

`pathfinder_warmstart_moms_ns` is a specialised variant that emits MoMS
off-locations + scales to seed `sample_moms_ns` (see above).
