# Nereus

**Planet Recovery Omni-Tool for Exoplanet Unified Sampling.**

Nereus is a Julia-native Bayesian inference toolkit for exoplanet
characterisation and detection from radial velocities, transit
photometry, and astrometry. Built for speed, flexibility, and
trans-dimensional model selection.

Unlike traditional codes where you choose the number of planets and
the noise model up front, Nereus explores those choices for you.
Trans-dimensional samplers (reversible-jump MCMC, MoMS variable
selection, parallel tempering) jump between models of different
dimensionality during the run, so the posterior naturally tells you
how many planets the data supports and which noise prescription best
fits the residual structure.

This is version `0.2.0`.

## What's in Nereus

- **Data models** for RVs, transit photometry, relative astrometry,
  HGCA (Gaia–Hipparcos proper-motion anomaly), Hipparcos IAD, Gaia
  GOST scans, Gaia DR3 five-parameter solutions, and activity
  indicator time series. The `Data` container ingests all of these
  jointly; loaders cover TESS light curves (`load_tess_lc`), Vizier
  RV CSVs (`load_vizier_rv`), orvara catalogs (`load_orvara_rv`,
  `load_orvara_relast`), HGCA rows (`load_hgca_row` / `fetch_hgca`),
  Hipparcos IAD (`load_hip_iad` / `fetch_hip_iad`), GOST (`load_gost` /
  `fetch_gost`), and Gaia DR3 (`load_gaia_dr3`). RVs may be single- or
  **double-lined** — tag each point's stellar component with
  `Data.rv_comp` for SB2 fits. →
  [Data models](data.md)
- **A unified prior system** — `UniformPrior`, `NormalPrior`,
  `LogUniformPrior`, `BetaPrior`, `ModJeffreysPrior`, `SinePrior`,
  `FixedPrior` — with auto-priors generated from the data bounds
  (following Peña & Jenkins, in prep. conventions), and an
  `ExternalPrior` mechanism for one-off scientific constraints. Every
  parameter is hard-bounded; `validate_physical` /
  `PHYSICAL_BOUNDS` guard against unphysical points. →
  [Priors](priors.md)
- **Stackable noise models** — `ARModel` (AR(p)), `MAModel` (MA(q)),
  `ActivityDecorrelation` with an optional FF′ derivative term
  ([Aigrain+ 2012](https://ui.adsabs.harvard.edu/abs/2012MNRAS.419.3147A/abstract)),
  `ActivityJitter`, three Celerite GP kernels (`CeleriteSHO`,
  `CeleriteRotation`, `CeleriteRotationFM17`), and the multivariate
  `ActivityGP` (Rajpaul+ 2015 quasi-periodic kernel jointly modelling
  RV + indicators with optional G/Ġ derivative coupling). All
  composable, all toggleable under trans-dim, all scopable to a
  subset of instruments. → [Noise models](noise_models.md)
- **Multiple parametrisations** for eccentricity (`sesinw`/`secosw`,
  `esinw`/`ecosw`, `ecc`/`w`), time anchor (`Mo` / `Tp` / `Tc`),
  transit geometry (`b`/`rr` vs `r1`/`r2`), and the stellar-density
  vs Kepler-III scale choice via `use_rho_s`. Planets carry
  per-planet `PlanetDataSources` so RV-only, transit-only, and joint
  planets coexist in one fit. → [Parametrisations](parametrizations.md)
- **Rossiter–McLaughlin** — set a planet's mode to `RVPM_RM` or
  `RVPMAS_RM` (Hirano+ 2011 leading-order model) or `RVPM_RM_R` /
  `RVPMAS_RM_R` (Reloaded, Cegla+ 2016, disk-integrated). This adds a
  system-level `v_sin_i_star` (m/s) and a per-planet `lambda_k<k>`
  (sky-projected obliquity, rad). The RM anomaly is built into
  `rv_predictions`, so PPC, residuals, and every RV plot are
  RM-consistent. → [Parametrisations](parametrizations.md)
- **Transit-timing variations** — per-transit free offsets (TTV-A,
  `RVPM_TTV` family), TTVFaster N-body predicted TTVs (TTV-C,
  `*_TTV_NB` modes), and a full-ODE backend via NbodyGradient.jl.
  Measure per-transit `T_c`'s from chains with `measure_per_transit_tcs`
  and the `ttvc_envelope`. → [Parametrisations](parametrizations.md)
- **SB2 double-lined binaries** — the `BINARY` / `BINARY_RV` planet
  modes fit an unresolved spectroscopic binary as one Keplerian with
  *independent* `K_A`/`K_B` (mass ratio `q = M_B/M_A = K_A/K_B`),
  optionally hosting a **circumprimary transiting planet** (a separate
  slot, visible only in the primary's lines). RV points carry a stellar
  component tag (`Data.rv_comp`: 1 = primary, 2 = secondary anti-phase).
  Transit dilution from the companion's third light auto-frees; with
  absolute astrometry a system-level `f_light = L_B/(L_A+L_B)`
  photocenter term scales the reflex by `(1 − f_light/f_mass)`, giving
  absolute component masses. Derived: `mass_ratio_q`, `M_A/M_B sin³i`,
  and absolute `M_A/M_B`. → [Parametrisations](parametrizations.md)
- **Astrometry** — relative astrometry, HGCA proper-motion anomaly,
  Hipparcos IAD along-scan likelihood, Gaia GOST scan-angle modelling,
  and Gaia DR3 solutions, all with PlanetOrbits-backed sky projection
  (`along_scan_projection`, `astrom_log_likelihood`). Hipparcos IAD,
  Gaia GOST, and the Brandt eDR3 HGCA auto-stage by HIP id /
  coordinates. **GOST is an auxiliary, not a standalone measurement** —
  it only window-averages (Mode B) the Gaia epoch of a paired HGCA /
  Gaia DR3 / G23H; supplied alone it is inert (the `Data` constructor
  warns). The validated Hipparcos–Gaia path is **HGCA + GOST**. →
  [Data models](data.md)
- **Trans-dim machinery** — `TransDimConfig`, five birth strategies
  (`PriorBirth`, `InformedBirth`, `JointInformedBirth`, `DonorBirth`,
  `MoMSBirth`), spike-and-slab priors, and user-declared
  `noise_exclusion_groups` for honest activity-model selection.
  → [Trans-dimensional](transdim.md)
- **Fifteen-plus samplers** spanning gradient HMC (`sample_nuts`),
  parallel tempering (`sample_pt` / `sample_pt_warm`,
  `sample_ptemcee` Vousden ensemble, `sample_pt_hmc` NUTS-in-PT,
  `sample_pt_whitening` NF-coupled), nested sampling (`sample_nested`,
  `sample_nested_ins` importance NS, `sample_nested_dynamic` dynamic
  NS), population annealing / SMC (`sample_pa`, `sample_smc`), the
  ensemble slice sampler (`sample_ess`), the affine-invariant
  ensemble (`sample_ensemble`), OFTI (`ofti_sample`), MAP point
  estimation (`sample_map`), Pathfinder warm-starts (`pathfinder_init`),
  and the trans-dim engines (`sample_rjmcmc`, `sample_moms`,
  `sample_moms_ns`, `sample_transdim_ptemcee`). →
  [Samplers](samplers.md)
- **Evidence stack** — four estimators of the Bayesian evidence
  (`log Z`, used to compare models) computed from the PT ladder:
  trapezoidal TI (`ti_trapezoidal`), TI+ (`ti_plus`), SS+ (`ss_plus`),
  and the hybrid H+ (`hybrid_evidence`) from
  [Peña & Jenkins 2026](https://ui.adsabs.harvard.edu/abs/2025arXiv250924870P/abstract),
  validated to better than 0.01 in log-evidence on analytic test
  problems. `model_probabilities` / `bayes_factors` turn trans-dim
  occupancy into model selection. → [Evidence](evidence.md)
- **Preprocessing** — multi-sector photometry detrending with
  Savitzky–Golay, GP, notch, LOCoR, and rotation filters
  (`clean_lightcurve`, `detrend_savgol`, `detrend_gp`, `detrend_notch`,
  `detrend_locor`, `detrend_rotation`); per-instrument 5σ MAD
  clipping; native-cadence transit masking (`find_transits`,
  `transit_mask`, `window_to_transits`); rotation-period search
  (`find_rotation_period`); and GLS/BGLS/SBGLS/L1 periodograms
  (`gls_periodogram`, `find_rv_planets`). Photometry stays at native
  cadence end-to-end — no binning. → [Preprocessing](preprocessing.md)
- **Publication-quality plots** for RV (`plot_rv_timeseries`,
  `plot_rv_phasefold`, `plot_rv_components` — model decomposition; SB2
  fits render component-coloured with both K_A and −K_B curves, and the
  binary phase-fold is the classic double-lined crossing),
  proper motion (`plot_pm_timeseries`, `plot_pm_phasefold`),
  transit (`plot_transit_phasefold`, `plot_transit_overlay_fit` —
  per-transit QC gallery), Rossiter–McLaughlin (`plot_rm`),
  astrometry (`plot_orbit_skyplane`, `plot_iad_residuals`,
  `plot_relastrom_timeseries`, `plot_pm_anomaly`), TTV
  (`plot_ttv_oc`, `plot_ttv_diagram`), periodograms, posteriors
  (`plot_corner`, `plot_trace`, `plot_posteriors`), trans-dim
  occupancy, detection limits, and the ActivityGP latent. →
  [Plotting](plotting.md)
- **Diagnostics & evidence-of-fit** — Polynomial Stein Discrepancy
  (`polynomial_stein_discrepancy`), ESS/R̂ (`print_ess_rhat`),
  posterior predictive checks (`posterior_predictive_check` /
  `plot_ppc`), Bayesian detection limits (`detection_limits`),
  PSIS-LOO / WAIC (`compute_loo`), a post-fit "silent-wrong" guard
  (`assess_fit`), model-conditioned science tables
  (`summarize_fitted`, `summarize_derived`), and NetCDF chain I/O
  (`save_chains` / `load_chains`) readable from Julia and Python. →
  [Diagnostics & I/O](diagnostics.md)
- **One JSON/dict entry point** — `run_job` is the single
  config-driven dispatcher used from Docker, Python (juliacall), and
  the CLI. It returns a JSON contract (science numbers + figure
  paths) for every run.

## Where to start

If you've never used Nereus, read [Quick start](quickstart.md) first
— it walks through a complete RV characterisation and a trans-dim
planet search end-to-end. Then dip into whichever subject is relevant
to your problem.

If you want one place to look up "what does kwarg X do on
`sample_pt_warm`?", every sampler has a kwarg reference in
[Samplers](samplers.md). Same for noise models, priors, and data
loaders.

If you want an end-to-end recipe for a specific dataset (RV+phot,
RV+astrometry, RM, trans-dim noise selection), see [Worked
examples](examples.md).

## Driving Nereus from a pipeline

Every workload can be expressed as a single `run_job` config (the
production path used by exoautomata). The shape mirrors the
in-Julia API. A minimal RV + activity-indicator GP job:

```julia
cfg = Dict(
    "model" => Dict(
        "planet_modes" => ["RV_ONLY"],   # see _KNOWN_PLANET_MODES for the full set
    ),
    "data" => Dict(
        "rv" => Dict(
            # inline "values" block: reserved keys bjd/rv/rv_err/instrument,
            # anything else is an activity indicator; a "<name>_err" array is
            # its 1σ (REQUIRED by ActivityGP).
            "values" => Dict(
                "bjd"        => bjd,
                "rv"         => rv,
                "rv_err"     => rv_err,
                "instrument" => inst,
                "bis"        => bis,  "bis_err"  => bis_err,
                "fwhm"       => fwhm, "fwhm_err" => fwhm_err,
            ),
        ),
    ),
    "noise_models" => [
        Dict("kind" => "ActivityGP",
             "instruments" => ["HARPS"],
             "kwargs" => Dict("channels" => ["bis", "fwhm"],
                              "use_derivative" => true)),
    ],
    "sampler" => Dict("kind" => "pt"),
)
run_job(cfg)
```

For a CSV RV block, name the activity columns with `indicator_cols`
and provide a matching `<col>_err` column per indicator (ActivityGP
requires the indicator errors). The valid `planet_modes`,
`noise_models[*].kind`, and `sampler.kind` strings are enumerated in
`src/runner.jl`; mode helpers like `RVPM_RM` / `RVPMAS_RM_R` enable
the Rossiter–McLaughlin terms.

## Citation

If you use Nereus in your research, please cite Vines et al. (in prep.).
