# Changelog

Changes that matter to someone fitting with Nereus. Older releases are in the
git history (`git log --first-parent develop`).

## Unreleased

### Fixed

- **The RV time series draws the trend.** With `trend_order` 1 or 2 the
  `rv_timeseries` curve was the Keplerians alone around zero, while the
  residuals took the trend out, so on a drifting star the curve missed the data.
  The curve is now the likelihood's mean minus γ: Keplerians plus
  `dvdt·(t − t_ref) + d2vdt2·(t − t_ref)²`, with the likelihood's `t_ref`, from
  one shared trend routine (`rv_trend`). It is labelled "Keplerian + trend"
  when a trend is fitted. `rv_components` draws the trend as its own curve,
  includes it in "total", and subtracts it from the residuals, which it used to
  leave in. The phase-folds already subtracted the trend from the data and are
  unchanged. A whole-model posterior band (`compute_ci_bands` without `planet`)
  includes the trend too.
- **Trend coefficients carry units in every table and figure label.** `dvdt`
  is m/s/day and `d2vdt2` m/s/day^2 (they had no unit); the CCF centre
  trends `ccf_gdot_*` / `ccf_gddot_*` are km/s/day and km/s/day^2, and
  `ccf_gamma_*`, `ccf_K_*` and `ccf_fwhm_*` km/s. The photometric baseline
  coefficients `phot_c*` stay dimensionless: they multiply a normalized time.

## 0.9.8

### Added

- **`detrend_locor` passes its `transit_mask` to the notch fallbacks.** The
  whole-LC fallback (`P_rot` non-finite or longer than a third of the baseline)
  and the per-segment fallback (fewer than 3 cycles) used to ignore the mask.
  They now call `detrend_notch` with it, so masked cadences are left out of
  every local window fit while the trend is still evaluated at them. The result
  echoes `transit_mask` (`nothing` without a mask), and the `detrend.locor`
  action accepts it as a JSON list. Without a mask every existing field is
  unchanged.

## 0.9.7

### Added

- **`detrend_notch` takes a `transit_mask`.** Same meaning as for
  `detrend_savgol`: masked cadences are left out of every local window fit, the
  trend is still evaluated at them, `flux_detrended = flux ./ trend`
  everywhere, and the returned `transit_mask` echoes the input. The BIC notch
  search runs only on unmasked cadences (not at all at masked ones). The new
  `notch_search=false` skips the search and returns the sigma-clipped local
  poly2 baseline. The `detrend.notch` action accepts `transit_mask`. Without a
  mask the result is unchanged.
- **Time trends on CCF component velocities.** A CCF component can take
  `"trend": 1` (linear) or `2` (quadratic) on its γ, static or tied:
  `γ_j + γ̇_j (t − t_ref) + ½ γ̈_j (t − t_ref)²`, with new parameters
  `ccf_gdot_<tag>_<c>` (km/s/day) and `ccf_gddot_<tag>_<c>` (km/s/day²).
  Default priors are Uniform(±span/baseline) and Uniform(±span/baseline²).
  `t_ref` is the set's `trend_t_ref`, by default the mean epoch BJD. A set with
  fewer than `trend + 2` epochs is refused. The trend is included in
  `ccf_component_velocities` and `ccf_model`, and the new parameters appear in
  chains.nc and the summary like the other `ccf_*` parameters.
- **`shared_fwhm`** (per CCF set): every component shares one width,
  `ccf_fwhm_<tag>`.
- **`pixel_correlation = "neff"`** (per CCF set): each epoch's log-likelihood is
  multiplied by `min(1, N_eff/N_pix)`, with `N_eff` = velocity span /
  resolution element; the element is `resolution_kms`, or else the smallest
  lower bound of the set's FWHM priors. This widens CCF posteriors to honest
  sizes and **changes evidences**. `ccf_epoch_weights` gives the factors.

The defaults (`trend` 0, `shared_fwhm` false, `pixel_correlation`
`"independent"`) give the 0.9.6 model bit for bit, and checkpoints written by
0.9.6 still resume.

## 0.9.6

### Added

- **CCF line components (`data.ccf`).** The CCFs of a spectroscopic blend are
  fitted directly as a sum of Gaussian components, each static (centre γ) or
  tied to a planet's orbit (centre γ + K·[cos(f + ω) + e cos ω], from that
  planet's own P, time anchor, e and ω, K of either sign). Per component:
  γ, K (tied only), FWHM, depth. Per epoch, or shared: continuum, depth scale
  (the reference epoch's is 1, which removes the depth/scale degeneracy), white
  jitter. Optional per-pixel errors; NaN pixels are dropped; velocity grids may
  differ between epochs. It is an ordinary target term: every engine, evidence
  estimator and checkpoint sees it, and it combines with RVs, photometry,
  transit times and astrometry through the shared orbit. Julia: `CCFSet`,
  `CCFComponent`, `ccf_log_likelihood`, `ccf_model`,
  `ccf_component_velocities`, `load_ccf_json`. Job config: `data.ccf`.
  Docs: docs/src/ccf.md. Checked against an independent least-squares analysis
  of NOI-101822's FEROS CCFs (`test/test_ccf_parity_noi101822.jl`).
- `jitter_names` includes the CCF jitters, so a CCF jitter at zero is not
  flagged as railed.

### Notes

- The CCF likelihood treats the pixels of a CCF as independent. CCFs are
  oversampled, so CCF-only credible intervals are lower bounds.
- `ad_backend = :Enzyme` refuses a target with CCFs (it would drop the term);
  ForwardDiff differentiates it.
- Checkpoints written before this release still resume: data without CCFs
  hashes as before.
