# Changelog

Changes that matter to someone fitting with Nereus. Older releases are in the
git history (`git log --first-parent develop`).

## Unreleased

### Added

- **The typed `fit_*` route reaches what a job config does.** `fit_rv`,
  `fit_transit`, `fit_astrometry` and `fit_joint` take `planet_modes` (one name
  per planet, e.g. `["RVPM_GD"]`, `["PM_GD"]`, `["RVPM_RM_GD"]`), `noise_models`
  (the dicts of a job config's `noise_models` block, built by run_job's own
  builder; `NoiseModel` objects still work) and `noise_menu` (a job config's
  block; with `transdim=` the menu's toggleable models and exclusion groups are
  wired in, as run_job does). `fit_transit(gravity_darkening = true)`, and a
  transit channel with it, now fits the gravity-darkened transit (`PM_GD`, or
  `RVPM_GD` with RV) where it used to throw. `build_target` gains `planet_modes`,
  `indicator_cols`, `noise_menu` and `menu_out`.
- **Activity indicators on the RV channel.** `indicator_cols` names columns of
  each instrument's data (`<name>_err` is the 1-sigma), and `indicators`
  (`{instrument: {name: values}}`) supplies them; both feed the activity noise
  models. `fit_rm`, `fit_ttv` and `fit_binary` take `indicator_cols` the same way.
- **`config_overrides` on every `fit_*`.** A dictionary in a job config's shape,
  applied last. On `fit_rm`, `fit_ttv` and `fit_binary` (which build a config) it
  is deep-merged into it, so any key of a config is reachable; they also take
  `noise_models`, `noise_menu` and `planet_modes`. On the four that build a
  target it is read block by block (`star`, `model`, `priors`, `noise_models`,
  `noise_menu`, `planet_modes`) and a block with no place there is an error.
- **Post-fit operations, no longer stubs.** `evidence`, `detection_limits`,
  `diagnostics.loo`, `diagnostics.ppc`, `diagnostics.fit_health` re-evaluate a
  finished fit from its `chains.nc` and the fit's own request (channels, planets,
  priors, model keywords), through the helpers the post-fit stage of the fit
  runs; `diagnostics.label_switching`, `select_planets` (P(N_p = k) and Bayes
  factors of a trans-dim fit's chains) and `select_noise` (noise-model occupancy)
  need only the chains. `detect.lc_periodogram` is a GLS periodogram of a light
  curve; `pre_white` is iterative GLS prewhitening. `FEATURE_NOT_IMPLEMENTED` is
  empty.

### Changed

- **A trans-dimensional engine or a planet range is trans-dim mode by itself.**
  `engine = transdim_pt_emcee` (and `rjmcmc`, `moms`, `daedalus` in a job config)
  no longer needs `transdim`: the `TransDimConfig` is built from the model's
  `max_kplanet`. `planets` as a range of two or more counts sets `transdim`.
  `transdim = false` with either is an error that says so. A job config's
  `"transdim": false` is read the same way.
- `fit_rm`, `fit_ttv` and `fit_binary` write the config they ran to
  `<output_dir>/job_config.json`; `evidence`, `diagnostics.loo`/`ppc`/`fit_health`
  and `detection_limits` rebuild the target from it the way `run_job` does.
- `RV` channel `indicators` no longer throws; it takes `{instrument: {name:
  values}}`. Anything else is still an `ArgumentError`.
- `fit_transit(gravity_darkening = true)` fits the PM_GD mode instead of throwing.
### Changed

- **HGCA / G23H without a GOST file now average the Gaia epoch over the release's nominal observing window.** The Gaia epoch used to fall back silently to the instantaneous reflex, wrong for a companion whose period is shorter than the Gaia baseline. With no `gost`, the model now takes the five-parameter fit through a synthetic stratified-jittered scan set across the window of the catalogue's release (DR3: 2014-07-25 to 2017-05-28; DR2: to 2016-05-23), one epoch drawn from a fixed hash inside each 5 d bin (`NOMINAL_GAIA_SCAN_STEP_DAYS`; deterministic, faithful for P above about 10 d; a strictly uniform grid would alias periods near 5/n d into a spurious proper motion), by the same code path as the GOST window average. `Data` logs one `@info` saying so; supply `gost` (`fetch_gost`) for the exact scan law, which is used unchanged. Hipparcos without scans still uses the instantaneous reflex (and still warns). Fits to HGCA / G23H without GOST move by the window averaging, most for short periods.

- **`pt_whitening` is deprecated** (kept, not removed): it warns once when selected. Its whitened swap accepts about 8x less often than the identity swap (median 0.027 vs 0.22) and `pt_emcee` beats it on every measured target. The comment claiming its ladder matches the validated `pt_emcee` one, and the docs rating it "Solid", were wrong and are corrected. `docs/src/sampler_validation.md` is rewritten to describe the tests that exist (the harness it described is not in the repository).

### Added

- **`sample_pt` and `sample_pt_emcee` warn once, after sampling, when the minimum adjacent swap acceptance is below 1%**, naming the rung pair and suggesting more temperatures. Warning only; draws are unchanged.

### Fixed

- **`ppc.png` and `detection_limits.png` are now in `summary["figures"]`** (keys `ppc`, `detection_limits`, plus `figures_pdf` when a PDF was saved). They are written to the output directory root, outside the `plots/` walk that builds the manifest, so clients never listed them. The files stay where they are.

## 0.9.10

### Added

- **Every derived row sent to the client carries its `unit`** (`best`,
  `unc_lo`, `unc_hi`, `ci1`-`ci3` plus `unit`, from the derived unit rules, the
  `_k<n>` suffix stripped), so a client no longer guesses `trend_P_min` as m/s.

- **Every entry of the fit summary's `params` block carries its `unit`**, and
  every variable of `chains.nc` a `units` attribute: the unit of the raw chain
  values (angles in rad, `inc_k1` -> `rad`), from `chain_param_unit`, which
  reads the same rules as the science tables, resolved per instrument channel
  (RV m/s, photometric relative flux, astrometric mas, CCF km/s). A name no rule
  knows gets no `unit`, and a client without per-entry units falls back to its
  own name map. The payload gains an optional field only; `PY_API_VERSION`
  stays 4.

- **Derived rows: ω_p, trend-implied companion bounds, sky position at an
  epoch, stellar M★ and log g★.** Per planet: `omega_p_deg` (ω★ + 180°, median
  in [0, 360)); `sep_mas` and `pa_deg` (east of north) at
  `star.separation_epoch` (default `data.t_ref`), from the relative-astrometry
  model's own orbit, for planets whose block carries astrometry. System rows
  (no planet suffix, `compute_derived_system`) with an RV trend:
  `trend_P_min` = the RV baseline T, `trend_K_min` = |γ̇|T/2π and
  `trend_Msini_min`, the minimum M sin i of a companion with P ≥ T; with
  `star.distance_pc` and `star.trend_sep_arcsec`, the Torres (1999) bound
  `trend_Mc_min_torres`. Stellar: `M_s_rho` = (4π/3)ρ★R★³ and `logg_s` from
  ρ★ (sampled `rho_s`, else the first transit's) and R★.

## 0.9.9

### Fixed

- **`fit_rv` takes an RV channel, and every RV channel option reaches the
  model.** `fit_rv` accepted only a bare instrument map, so astronereus passed
  it the channel's `data` alone and `fit_rv(rv, trend_order=1)` fitted no
  trend. It now takes the channel dict `fit_joint` takes (the instrument map
  still works). `_target_from` (fit_rv, fit_joint, replot) read only `jitter`
  and `trend_order` from an RV channel; `gamma_per_instrument = false` (one
  offset), `share` (sigma/gamma groups, merged with `sharing`) and
  `marginalize_gamma` now build the model too, and `jitter` takes a prior dict
  or a per-instrument mapping. `indicators` and unknown keys throw, as do a
  `trend_order` keyword that disagrees with the channel's, a sharing category
  given in both places, and a channel `jitter` together with
  `priors["sigma_<inst>"]`. Unknown transit-channel keys throw too.
  `PY_API_VERSION` is 4.
- `fit_joint` dropped an RM, tomography, TTV or SB channel without a word, and
  a second RV or transit channel replaced the first; each throws now.
- `build_target` takes `marginalize_gamma`.

- **A fit with no planet runs.** `fit_rv(planets = 0, trend_order = 1)` (and
  `planets = []`, which is what a JSON `[]` from astronereus is) stopped
  before sampling: `build_target` refused an empty planet set and
  `_planet_spec` took the maximum of an empty list. A radial-velocity-only
  target may now have no planet: offsets, jitters and the `dvdt` / `d2vdt2`
  trend are fitted, and the summary, tables and RV plots are written. A
  transit or astrometric target still needs a planet.
- **The notch detrender keeps transit depths.** Around every excluded run --
  a dip found by the search (`refine`, the default) or a `transit_mask` built
  from an ephemeris -- the trend was a per-window poly2 extrapolated from clean
  data on one side only, and the old repair bridged each run with a chord
  between two such extrapolated values. On TOI-4939 TESS S103 (window 0.5 d,
  no ephemeris) a false dip just after egress left a 20-min clean stretch
  whose extrapolated trend tilted the chord: a 6.5 ppt transit came out at
  8.7 ppt. Each run of excluded cadences, plus the clean cadences whose window
  is mostly excluded on one side, is now bridged by one σ-clipped poly2 fitted
  to the raw clean flux on both flanks (and inside the run), spanning about
  one window; a run at a segment edge continues the straight line of its
  one flank. Injected 6 ppt transits
  in synthetic data (rotation 0-0.8 %, 1.5 ppt noise): per-transit depth rms
  0.52-0.54 → 0.22-0.28 ppt without an ephemeris, 0.40 → 0.26 ppt with a
  `transit_mask` (white-noise floor 0.20 ppt). Transits injected into real
  TESS sectors of TOI-4939, TOI-4680 and TIC 275647095: with a `transit_mask`
  the depth-error rms fell in 13 of 16 sectors (worst single transit 86 → 16
  ppt, at a segment edge); without one it fell 2-3× for TOI-4939 (window
  0.5 d) and was mixed (±50 %) for the 3 T14 windows on the two strongly
  variable hosts, where chains of false dips remain the limit -- prefer a
  `transit_mask` there. Light curves already detrended with the notch should
  be detrended again.

- **`detrend_savgol` with a `transit_mask` anchors each masked run on averaged
  flanks.** The interpolation across a masked run started and ended at the
  single bracketing flux points, whose noise went straight into the trend
  across the transit (per-transit depth rms 0.68 → 0.30 ppt in the same test).
  Each anchor is now the mean of up to 30 nearest unmasked points.

- **The φ_c epoch is never in a gap between light curves.** The default
  `t_ref_c`, the 1/σ²-weighted mean of the photometric times, can land far
  from any data when light curves span years (TIC 275647095: 53 d from the
  nearest point). There P and φ_c are tied along narrow ridges, and
  `pt_emcee` walkers started from prior draws stalled on 7×–35× aliases of the
  17.06 d period. An epoch more than 1 d from every photometric time it was
  computed from (`PHIC_EPOCH_GAP_TOL`) now moves to the nearest one; an epoch
  already within the data is unchanged, so existing fits are too. The median
  used when no point has a usable error is moved the same way.

- **`fit_binary` with a `secondary` runs.** It sent planet mode `"SB2"`, which
  `run_job` rejects, and put the secondary RVs under a key nothing read. It now
  sends `BINARY_RV` with both components in one RV block tagged by `component`.

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
- **Every parameter has a unit, the same on every output path.** An audit of
  every name a layout can hold found more that reached the tables, summary JSON
  and figure labels with no unit, or the wrong one:
  - `i_star` (gravity darkening) is sampled in radians and was published as
    raw radians with no unit; it is now converted and labelled deg.
  - `M_pri` is M_sun; `v_sin_i_star`, `beta_p` (ARoME) m/s;
    `tomo_sigma_line_*` and `tomo_ell_v_*` km/s.
  - GP and activity hyperparameters, by the channel the model acts on (m/s for
    RVs, dimensionless for flux and tomography maps): `gp_sigma`,
    `matern_sigma`, `night_sigma`, `harm_amp` in the channel's unit; periods
    and time scales (`gp_period`, `matern_rho`, `harm_period`,
    `gp_act_period`, `gp_act_lambda_e`, `ind_floor_period`,
    `ind_floor_lambda_e`, `ma_beta`, `ar_alpha`) in d; the log-parametrised
    celerite terms as ln(m2/s2*day) (`gp_log_S0`, RV), ln(1/day)
    (`gp_log_omega0`), ln(m2/s2) (`gp_log_amp`, RV), ln(day)
    (`gp_log_timescale`, `gp_log_period`); `Vc`, `Vr`, `C_*`, `jit_base_*`,
    `jit_act_*` m/s and `Cdot_*` m/s*day (indicators are normalised to unit
    RMS by default, so indicator-side amplitudes are dimensionless).
  - A shared photometric group (`jitter_TESS+K2`) was labelled m/s, as if
    it were an RV jitter; instrument slots are now classified by the layout,
    not by the name.
  - Derived rows `lambda_deg`, `lambda_abs_deg`, `i_star_deg`, `psi_deg`
    had no unit (deg), and the SB2 masses `M_A`, `M_B`, `M_A_sini3`,
    `M_B_sini3` none (M_sun).
  The .tex tables and the figure labels now render units from one table
  (`_UNIT_FORMS`), and dimensionless parameters keep the empty unit, now as an
  explicit rule (`sci_unit_known`) rather than a fall-through.

### Added

- **More derived parameters, from what every fit already samples.**
  - The epochs the other time anchors sample: `time = :Tc` now reports `Tp`
    and `Mo_deg`, `:Tp` reports `Tc` and `Mo_deg`, and `:Mo` reports `Tp`
    and `Tc` (as `:phic` already did).
  - `Tsec` (BJD), the secondary conjunction after `Tc`, for transiting
    planets.
  - `g_p` (m/s2), the planet's surface gravity G·M_p/R_p², and `mass_ratio`,
    M_p/M★, where the transit geometry determines them.
  - `p_transit`, the geometric transit probability (R★/a)(1 + e sin ω)/(1 − e²)
    (Winn 2010), for a planet with no transit in the model and a known a/R★.
  - Astrometric planets that sample the mass (`M_sec_driven`, `a_driven`) now
    report `mass_jup`/`mass_earth`, and without a transit their `a_Rs` (hence
    `Teq`, insolation) follows from the model's `a_au` rather than from the
    minimum-mass orbit.
  - SB2 binaries report `omega_deg`, the projected relative semi-major axis
    `a_sini_au` = (K_A + K_B) P √(1 − e²)/2π, and `a_au` with the inclination.
  Fitting and defaults are unchanged; these are derived per posterior draw.

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
