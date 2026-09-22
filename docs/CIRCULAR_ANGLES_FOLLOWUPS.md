# Circular angles: remaining fixes

Branch `fix/circular-angles` makes full-circle angles (`Mo_k`, `Omega_k`,
`lambda_k`, `w_k` under `:ew`; Uniform prior over exactly one period) stop
treating 0 ≡ 2π as a wall. A uniform full-circle prior is rotation-invariant, so
moving the seam to the emptiest arc of the posterior is an exact relabelling.
Core helpers are in `src/circular.jl`.

## Done on the branch

- **In-sampler re-cut** at burn-in / warmup: pt_emcee, pt_whitening,
  transdim_pt_emcee, the in-house `pt`, and single-chain rjmcmc / moms.
  Trans-dim samplers share one window per same-mode slot group
  (`circular_groups`, `unify_circular_groups!`), because births and swaps copy
  raw values between slots.
- **MAP**: a circular optimum near the seam is re-charted and polished. Circular
  angles are never reported as railed. Basin distance is measured around the
  circle.
- **Outputs**: `run_job` / `_finish` call `recenter_circular!` before anything
  reads the draws, so every consumer sees a contiguous posterior with its median
  in the user's window.
  - Science tables: Ω and λ are now degrees, the `mod2pi` re-split is removed,
    and `run_info.priors` reports the user's prior with `"circular": true`.
  - Derived ω and λ are contiguous.
- **fit_health**: circular angles are measured in the window the engine sampled
  in (`circular_windows` taken before recentring). Engines that moved their
  seam pass. Engines that could not cross it get "full-circle angle cut by its
  seam".
- **Not circular**:
  - Mo under TTVs (epoch numbering uses Tp linearly).
  - Mo under `obs_prior` (the Jacobian is linear in M).
  - Non-uniform priors, arcs, and two-period priors.
- **Tests**: `test/test_circular.jl`. Reproduction: a seam-centred RV planet
  (Mo = 0.02 rad, e = 0.3), fitted with `fit_rv`.
  - `main`: rail FAIL, and the science table gives `Mo = 10.8 −6.3 +347.3 deg`.
  - Branch: OK, `7.0 −5.1 +5.5 deg`.

## Engines still cut by the seam (flagged, not fixed)

### ensemble (`src/samplers/ensemble.jl`), easy

- The move loop lives inside AffineInvariantMCMC.jl, in the logit chart, so it
  cannot be re-cut mid-run.
- Fix: split the run into a burn-in call and a production call.
  - Between them, call `recut_circular_param!` on the final walker positions,
    in bounded space.
  - Relabel them, re-transform with the moved `target.transform`, then start
    production from those positions.
- The chart is fixed for production, as in pt_emcee.

### rjmcmc / moms with `n_chains > 1`, moderate

- Chains run as parallel `Threads.@spawn` tasks over one shared `Params`. A
  per-chain re-cut would move the window under the other chains mid-sweep.
- Fix: split each chain into warmup and production phases.
  1. Spawn all warmups and join them.
  2. Pool every chain's `_CircularWarmupTrace` (active draws only).
  3. Run `recut_circular_group!` per group.
  4. Relabel every chain's `theta`, shift the MoMS off-values
     (`lo_new − lo_old`), and refresh `log_pi` / `log_L`.
  5. Spawn the production phases.
- The RNG streams must stay task-keyed (`test/test_sampler_determinism.jl`).

### NUTS / pt_hmc (`src/samplers/nuts.jl`, `pt_hmc.jl`), moderate

- The warm start already re-charts from the whole pt_emcee pre-search. That only
  works when the pre-search has converged:
  - default `warm_steps = 400`: NUTS returned 358° ± 2° against a true 7°;
  - `warm_steps = 3000`: it matched pt_emcee.
- Option A, retry. After NUTS warmup, test circular draws for a pile-up at the
  window seam, using the same test as `_check_prior_rail`.
  - If found, pool the warmup draws, re-cut, relabel the chain positions, and
    redo warmup.
  - Chains stuck in the sliver below 2π then see the rest of the mode.
  - Cheap. It fixes unimodal and cleanly separated multimodal angles.
- Option B, a wrapping transform. Add a new `PackedTransforms` type for circular
  dims: identity in y, with x = `lo + mod(y − lo, 2π)` and log-Jacobian 0.
  - The target is 2π-periodic in y, and HMC with a translation-invariant metric
    projects exactly onto the circle.
  - This needs the Enzyme path (`target.jl`, `_logdensity_for_enzyme` plus
    `EnzymeGradientConfig`) to carry the new type id.
  - Every estimator that fits a Gaussian in y (bridge, reference_path,
    mode_laplace, the MAP Laplace) must run in a one-period chart. They already
    do when built after `recenter_circular!`, but check any estimator that uses
    the sampling target directly.
  - Mass-matrix adaptation must see unwrapped increments, not the wrapped x.
- Recommendation: A first, B if A proves flaky.

### Nested samplers (`nested.jl`, `nested_dynamic.jl`, `nested_ins.jl`), hardest

- The walls are the unit cube itself.
- Periodic dims, as in dynesty's `periodic=`:
  - Wrap u mod 1 for circular dims in every proposal: vendored NestedSamplers.jl
    `unitcheck`, RWalk, RStagger and Slice / RSlice step-out and shrink, plus
    `_rwalk_within_shell!` and `_slice_probe!` in `nested_ins.jl`.
  - Stop shrinking the random-walk step size on seam "rejections".
- Bounds:
  - Ellipsoids and MLFriends fitted to live points that straddle u = 0/1 are
    centred in the empty middle.
  - Every K iterations, rotate each circular dim so its live cluster is centred:
    `u' = mod(u − c, 1)`, with the prior window shifted to match.
  - This is exact, because a uniform prior maps linearly from u.
  - Dead points are relabelled at output by `recenter_circular!`.
- Without the bound rotation, the wrap alone fixes correctness but not
  efficiency.
- Measured: nested kept only the part above 0 of a posterior centred on
  Mo = 0.02. It is now flagged, with 21% of draws at the seam.

## Other known gaps

- **In-sampler cut votes with one walker snapshot.** Gaia-4 (astrometry-only,
  pt_emcee, `examples/01`) has a minor Ω mode (~15%) at −0.09 rad, right against
  the seam.
  - At burn-in the cut kept the seam, because 100 cold walkers were not enough
    to clear the hysteresis. The seam passed through the edge of that mode:
    6.4% of draws within 0.1 rad of 2π.
  - The output recentring fixed the reporting (window (−1.66, 4.63)), and Mo was
    re-cut correctly in-sampler.
  - Fix: pool the cold-walker positions over the last ~200 burn-in steps in
    `_pt_ensemble_recut!` (and the trans-dim version) instead of one snapshot.
- **`obliquity.jl`**: `NormalPrior(Mo, σ, 0, 2π)` truncates the prior at the
  seam when the published ephemeris puts Mo near 0. Use bounds `(Mo − π, Mo + π)`
  (check `PHYSICAL_BOUNDS["Mo"]`) or a wrapped-normal prior type.
- **`chains.nc`** records neither the moved windows nor which params are
  circular. arviz users see Mo quantiles below 0 against a U(0, 2π) config.
  Add attributes `circular_params` / `circular_window_<name>`.
- **DonorBirth** (`transdim/proposals/alias.jl`): the forward proposal density
  of the newborn's values is missing from `log_q_ratio`. This is pre-existing,
  independent of the wrap.
- **`inc_k`** is registered as `("deg", false)` in `_SCI_UNITS`, so tables and
  plots show radians under a "deg" label. Pre-existing.
- **`_DERIVED_UNITS`** lacks `lambda_deg`, `lambda_abs_deg`, `psi_deg` and
  `i_star_deg`.
- **Target reuse.** Samplers mutate `params.layout` and `target.transform` in
  place. Explicit `init` vectors are relabelled, and trans-dim groups are
  unified at start. Single-member windows carry over between fits; that is
  exact, but reproducing a run bit-for-bit needs a fresh target.
