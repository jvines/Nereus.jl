# Nereus job config

A single JSON file (or an in-memory `Dict`) fully describes a Nereus fit.
`Nereus.run_job` is the one canonical entry point — the Dockerised batch
workers, the Python (`juliacall`) callers, and the CLI all go through it.
This page is the authoritative schema reference; it is meant to be enough to
wire Nereus into a production pipeline (exoautomata) without reading the
Julia source. Every field, option, sampler, noise model, and plot named here
exists in `src/runner.jl` (and the modules it dispatches into).

Run via:

```bash
julia --project=/path/to/Nereus.jl \
      -e 'using Nereus; Nereus.run_job(ARGS[1])' \
      /work/in/config.json
```

Two call forms (`src/runner.jl:29-32`):

```julia
run_job("/path/to/config.json")   # reads + parses JSON from disk
run_job(config::AbstractDict)      # accepts a pre-built dict, arrays inline
```

Both write `summary.json`, `chains.nc`, and the plot tree into `output_dir`
**and** return the `summary` dict, so `juliacall` callers can skip disk
entirely. Keys may be JSON strings or Julia `Symbol`s interchangeably
(`_has`/`_get`, `src/runner.jl:143-154`).

Outputs land in `output_dir/`:

```
output_dir/
├── chains.nc                  # MCMCChains NetCDF (always)
├── summary.json               # machine-readable contract (always)
├── ppc.png / ppc.json         # posterior predictive check (default on)
├── detection_limits.png       # K_lim(P) curve (PT samplers, default on)
├── detection_limits.nc        # per-bin arrays sidecar
├── tables/                    # science tables (fitted/derived, multi-format)
│   ├── fitted.{json,csv,ecsv,dat,tex}
│   └── derived.{json,csv,ecsv,dat,tex}
└── plots/                     # all figures (configurable)
    ├── models/                # rv_timeseries, pm_phasefold, rm_anomaly, …
    ├── posteriors/            # raw/, parameters/, histograms/
    ├── traces/
    └── transdim/occupancy.png
```

`run_job` exits 0 on success. Any uncaught exception during the fit writes
`{"status":"failed","error":…,"traceback":…}` to `summary.json` **and**
rethrows (non-zero exit) so the dispatcher can pick it up
(`src/runner.jl:122-130`). The auxiliary stages (PPC, detection limits, LOO,
fit-health) are **fail-soft**: they record a `"failed"`/`"skipped"` entry in
the summary and never abort the job.

> **Version**: the current schema version is **`0.2.0`** (see `Project.toml`).
> `version` is echoed back as `summary.config_version` but is **not**
> validated against a whitelist; supply `"0.2.0"`.

---

## Job kinds

By default a config describes a **fit**: `run_job` builds the model from
`data`/`model`/`priors` and runs the sampler in `sampler`. That is everything
documented below.

One alternative kind exists:

### `"kind": "tomography"`

Doppler tomography — recover the sky-projected obliquity λ from the planet's
shadow in the stellar line profile. It is dispatched **before** the normal
config validation, because it is not a sampler job and has none of the fields
that validation requires. There is no TOMO source flag and no tomographic
likelihood: the estimator is a matched filter over λ against the shadow track
predicted by the transit geometry, with significance from a null distribution
built by scrambling the in-transit frames. It therefore takes **no `sampler`
block** — passing an engine is a category error.

```jsonc
{
  "kind": "tomography",
  "output_dir": "/work/out",               // REQUIRED
  "orbit": {                               // REQUIRED; all but vsys are mandatory
    "P":     2.827969,                     // days
    "a_Rs":  6.8103,
    "inc":   1.4598,                       // RADIANS
    "vsini": 25.862,                       // km/s
    "T14":   0.10917,                      // days; sets which frames are in transit
    "vsys":  18.93                         // km/s (default 0.0)
  },
  "nights": [                              // REQUIRED; one entry per transit night
    {
      "profiles": [[...], [...]],          // n_time x n_velocity, list of rows
      "vgrid":    [...],                   // km/s, length n_velocity
      "times":    [...],                   // length n_time
      "Tc":       2460500.50000,           // THIS night's own measured mid-transit
      "bervs":    [...]                    // km/s; required for pooling across weeks
    }
  ],
  "options": { "n_null": 300, "n_lambda": 721 }
}
```

Two traps worth stating plainly, both of which silently destroy the signal
rather than erroring:

* **`Tc` is per night, not per system.** Pooling several transits folded on a
  single ephemeris smears the shadow away.
* **Every night's profiles must share a sign convention.** A DRS CCF is a dip
  (`ccf/continuum`) while a mask-CCF built by accumulating line depth peaks;
  mixing the two makes one night subtract from the others. Convert dips with
  `(continuum - ccf)/continuum` before pooling.

Returns the usual `summary.json` in `output_dir`, with `lambda_rad`,
`lambda_deg`, `score`, `p_value`, `n_nights`, `n_null`, and the full
`lambda_scan_deg` / `lambda_scan_score` arrays so the landscape can be plotted
rather than a single number trusted.

---

## Schema

```jsonc
{
  "version": "0.2.0",                      // echoed to summary.config_version (default "1.0")
  "seed":   42,                            // RNG seed; controls every sampler (default 1)
  "output_dir": "/work/out",               // REQUIRED
  "n_threads": null,                       // logged only; actual threading honours
                                           //   JULIA_NUM_THREADS (src/runner.jl:47-49)
  "timeout_sec": 3600,                     // optional wall-clock watchdog (see below)

  // ---- Star (NamedTuple; src/runner.jl:671-680) ---------------------
  "star": {
    "M_s": 1.02,                           // M_sun. Needed for stability != none,
                                           //   astrometry, RM modes, and derived masses
    "R_s": null,                           // R_sun. Needed for transit fits + RM modes
    "T_eff": null,                         // K. Drives T_eq / TSM / ESM in `derived`
    "J_mag": null, "K_mag": null,          // mags; optional
    "Ab": 0.3                              // Bond albedo (default 0.3)
  },

  // ---- Data (at least ONE of these; src/runner.jl:235-268) ----------
  "data": {
    "rv": {                                // omit for transit-only / astrometry-only
      // EITHER a CSV file …
      "csv": "/work/in/rvs.csv",
      "time_col":  "bjd",                  // default "bjd"
      "rv_col":    "rv",                   // default "rv"
      "err_col":   "rv_err",               // default "rv_err"
      "inst_col":  "instrument",           // default "instrument"
      "indicator_cols": ["bis", "fwhm"]    // optional activity indicators; a matching
                                           //   "<col>_err" column (e.g. bis_err) is read
                                           //   as that indicator's 1σ (ActivityGP needs it)
      // … OR an inline `values` block (mutually exclusive with `csv`):
      // "values": {
      //   "bjd": [...], "rv": [...], "rv_err": [...], "instrument": ["HARPS", ...],
      //   "bis": [...], "bis_err": [...],   // any extra key is an indicator;
      //   "fwhm": [...], "fwhm_err": [...]  //   "<name>_err" w/ matching "<name>" = its error
      // }
    },

    "transit_photometry": [                // list of per-instrument light curves
      {
        "csv": "/work/in/sector33.csv",
        "time_col":   "bjd",               // default "bjd"
        "flux_col":   "flux",              // default "flux"
        "err_col":    "flux_err",          // default "flux_err"
        "instrument": "TESS_S33",          // default "TESS"
        "exposure_time": 120.0             // SECONDS (default 120 = TESS 2-min); used for
                                           //   finite-exposure supersampling
        // OR inline: "values": { "bjd": [...], "flux": [...], "flux_err": [...] }
      }
    ],

    "iad":  { "hip_id": 24205 },           // Hipparcos IAD: hip_id auto-fetch OR `values`
    "gost": {                              // Gaia GOST scan plan: ra/dec auto-fetch OR `values`
      "ra_deg":  86.5735,
      "dec_deg":  1.2117,
      "from": "2014-07-26T00:00:00",       // default 2014-07-26 (Gaia start)
      "to":   "2025-01-15T00:00:00"
    },
    "hgca":      { "csv": "/work/in/hgca.csv", "hip_id": 24205 },  // Brandt 2018 HGCA, OR `values`
    "relastrom": { "csv": "/work/in/relastrom.csv" },             // direct-imaging, OR `values`
    "gaia_dr3":  { "params": [86.5, 1.2, 35.2, 12.1, -5.4],       // [α, δ, ϖ, μα*, μδ]
                   "cov": [[/* 5×5 */]], "t_ref": 2016.0 }
  },

  // ---- Model (src/runner.jl:701-755) --------------------------------
  "model": {
    "max_kplanet": 2,                      // REQUIRED. Max planet slots
    // planet_modes: one entry per slot (default = max_kplanet × "RV_ONLY").
    // VALID modes (src/runner.jl:180-186, 690-699):
    //   RV_ONLY        — RV only
    //   PM_ONLY        — transit photometry only
    //   RVPM           — RV + photometry
    //   RVAS           — RV + astrometry
    //   RVPMAS         — RV + photometry + astrometry
    //   RVPM_RM        — RVPM + Rossiter-McLaughlin (Hirano+ 2011)   [adds v_sin_i_star, lambda_k<k>]
    //   RVPMAS_RM      — RVPMAS + RM (Hirano)
    //   RVPM_RM_R      — RVPM + Reloaded RM (Cegla+ 2016)            [same params; accurate at high rr]
    //   RVPMAS_RM_R    — RVPMAS + Reloaded RM
    //   RVPM_TTV       — RVPM + free per-transit TTV offsets (ttv_k<k>_t<i>)
    //   RVPMAS_TTV     — RVPMAS + free TTV offsets
    //   PM_TTV         — photometry-only + free TTV offsets
    //   RVPM_RM_TTV    — RVPM + RM + free TTV offsets
    //   RVPM_TTV_NB    — RVPM + N-body-predicted TTVs (no free offsets; ≥2 NB planets)
    //   RVPMAS_TTV_NB  — RVPMAS + N-body-predicted TTVs
    //   PM_TTV_NB      — photometry-only + N-body-predicted TTVs
    //   BINARY_RV      — SB2 double-lined binary orbit, RV only              [slots K_A_k<k>, K_B_k<k>]
    //   BINARY         — SB2 double-lined binary + astrometry (absolute masses) [+ inc_k<k>, Omega_k<k>, f_light]
    //   See "SB2 double-lined binaries" below for the component RV column + derived masses.
    "planet_modes": ["RV_ONLY", "RVAS"],

    "parametrization": {                   // src/runner.jl:708-715, _KNOWN_PARAMETRIZATIONS
      "mass": "K_driven",                  // K_driven | M_sec_driven | a_driven
      "time": "Tp",                        // Tp | Tc | Mo
      "ew":   "sesinw",                    // sesinw | ew
      "geom": "b_rr",                      // b_rr | b_r
      "marginalize_gamma": false,          // analytic γ marginalisation (default false)
      "use_rho_s": false                   // sample stellar density ρ⋆ instead of deriving
                                           //   a/R★ from M_s×R_s (default false). ρ⋆ is what
                                           //   transits actually constrain — the standard
                                           //   transit parametrization. Enabling auto-adds a
                                           //   LogUniform(0.001,100) ρ⋆ prior; override with
                                           //   an informative one via external_priors
                                           //   ExternalPrior(:rho_s, NormalPrior(...), false)
    },

    "trend_order": 0,                      // RV systemic γ trend: 0 | 1 | 2 (src/runner.jl:294-298)
    "phot_trend_order": 0,                 // per-instrument photometric baseline: 0 | 1 | 2
                                           //   → params phot_c<p>_<inst> (src/runner.jl:299-303)
    "stability": "none",                   // none | amd | gladman (src/runner.jl:193, 689)

    // ttv_backend: consulted only for *_TTV_NB modes (src/runner.jl:747-750)
    "ttv_backend": "ttvfaster",            // ttvfaster (default) | nbody
                                           //   ttvfaster: Agol & Deck 2016 1st-order, autodiff-clean
                                           //   nbody:     NbodyGradient.jl (AHL21); strips Duals →
                                           //              use a gradient-free sampler
    // ttv_n_transits: free per-transit offset counts for *_TTV modes.
    //   Keys are 1-based slot indices as strings (src/runner.jl:742-746).
    "ttv_n_transits": { "1": 30 },

    // ---- external_priors: priors on DERIVED (non-sampled) quantities ----
    // `priors` above keys on sampled slots; these act on values computed FROM
    // them, so they cannot be expressed there.
    //   quantity: "ecc"    — eccentricity, per planet
    //             "rho_s"  — stellar density, global
    //   per_planet defaults from the quantity (true for ecc, false for rho_s).
    // The ecc prior is not cosmetic on a sparse, activity-contaminated RV:
    // uniform sesinw/secosw implies p(e) ∝ e, which rails e → 1.
    "external_priors": [
      { "quantity": "ecc",
        "prior": { "type": "NormalPrior", "args": [0.0, 0.3] },
        "per_planet": true }
    ]
  },

  // ---- Priors — per-parameter overrides (src/runner.jl:768-776) -----
  // Keys are Nereus parameter names; anything not listed gets its default prior.
  //   Per-planet:  P_k1, K_k1, sesinw_k1/secosw_k1 (or e_k1/w_k1), Tp_k1/Tc_k1/Mo_k1,
  //                b_k1, rr_k1, inc_k1, Omega_k1, M_sec_k1, a_k1, …
  //   RM:          v_sin_i_star (m/s; default LogUniformPrior(500,100000)),
  //                lambda_k1 (rad; default UniformPrior(-π, π))   — src/default_priors.jl:299-304
  //   Systemic:    plx, M_pri, trend_1, trend_2
  //   Per-instrument: gamma_<inst>, jitter_<inst>, offset_<inst>, … (suffix = instrument name)
  // VALID prior types (src/runner.jl:194-196, 758-766):
  //   UniformPrior, LogUniformPrior, ModJeffreysPrior, NormalPrior,
  //   FixedPrior, SinePrior, BetaPrior
  "priors": {
    "P_k1":         { "type": "LogUniformPrior",  "args": [1.0, 1000.0] },
    "K_k1":         { "type": "ModJeffreysPrior", "args": [1.0, 1000.0] },
    "plx":          { "type": "NormalPrior",      "args": [35.25, 1.02, 5.0, 70.0] },
    "v_sin_i_star": { "type": "LogUniformPrior",  "args": [500.0, 50000.0] },
    "M_pri":        { "type": "FixedPrior",       "args": [1.02] }
  },

  // ---- Noise models (src/runner.jl:779-844) -------------------------
  // VALID kinds (src/runner.jl:197-200, 779-788):
  //   CeleriteRotation, CeleriteSHO, CeleriteRotationFM17  — celerite GPs (channel + instruments)
  //   ActivityDecorrelation                                — linear indicator decorrelation (MeanModifier)
  //   ARModel, MAModel                                     — autoregressive / moving-average noise (channel)
  //   ActivityJitter                                       — extra activity jitter term (MeanModifier)
  //   ActivityGP                                           — joint RV+indicator GP (channels)
  //   IndicatorFloor                                       — always-on floor on the indicator
  //     channels (channels + kernel: "white" | "qp"). REQUIRED for a fair fixed-config
  //     comparison between a mean model (AD) and ActivityGP: the GP scores the indicator
  //     channels and a mean model does not, so without a floor the two log Z values are
  //     computed on different data and are not comparable.
  //   ErrorScale, NightlyOffset, HarmonicBlock, StudentT, MaternGP
  // `channel`/`instruments` are injected ONLY into constructors that accept them.
  "noise_models": [
    { "kind": "CeleriteRotation", "channel": "rv", "instruments": [], "kwargs": {} },
    { "kind": "ActivityDecorrelation", "channel": "rv",
      "instruments": ["HARPS"], "kwargs": { "indicators": ["bis"] } },
    // ActivityGP: needs per-indicator errors in the rv data (bis_err/fwhm_err).
    { "kind": "ActivityGP", "instruments": ["HARPS", "FEROS"],
      "kwargs": { "channels": ["bis", "fwhm"], "use_derivative": true } }
  ],

  // ---- Noise menu (ALTERNATIVE to noise_models; src/runner.jl) -------
  // Server-built default noise menu for trans-dim NOISE-MODEL SELECTION.
  // Mutually exclusive with `noise_models`: the menu BUILDS the model list
  // (ErrorScale + CeleriteRotation + ActivityDecorrelation(lin) + ActivityGP
  // + always-on IndicatorFloor) AND the single exclusion group (white XOR
  // correlated XOR decorrelation — one winner), and wires
  // `transdim.toggleable` / `transdim.noise_exclusion_groups` automatically.
  // JSON cannot express those two fields itself (they hold NoiseModel
  // objects), so this block is the ONLY way to run menu-based noise
  // selection through run_job.
  //
  // `indicators` — WHICH activity channels the menu's indicator-driven
  //   members (AD / ActivityGP / IndicatorFloor) use. Default: every
  //   indicator channel present in the RV data block. Not every indicator
  //   correlates with the RVs on every target, and a non-correlating
  //   channel is pure leak surface — a near-commensurate indicator (e.g.
  //   BIS at P_rot/2 ≈ P_orb) can absorb planet amplitude through its
  //   regression coefficient (measured on HD 18599: 4 m/s of planet-phase
  //   power in the AD term, K biased 11 → 8.8). FRONTEND: render as a
  //   per-channel multi-select over the channels uploaded with the RV data;
  //   send the checked subset here. Channels must be a subset of the data's
  //   indicator columns (hard error otherwise).
  //
  // Optional include_* toggles (default false) admit extra menu members:
  // include_matern, include_studentt, include_harmonic, include_ad_ffprime,
  // include_nightly. `include_activity_gp` (default TRUE) drops the joint
  // RV+indicator GP when false — REQUIRED for single-channel selections
  // where the channel carries power near the orbital period: the AGP latent
  // is only as constrained as the channel set, and on one contaminated
  // channel it absorbs the planet (HD 18599 bis-only: AGP occupancy 1.0,
  // K 11 -> 4.2). FRONTEND: when exactly one indicator is checked, surface
  // this toggle (or default it off) with that warning.
  //
  // Requires: a `transdim` block (max_kplanet etc.) and a trans-dim
  // sampler. `transdim.noise` defaults to true when this block is present.
  // `transdim.toggleable`/`noise_exclusion_groups` must NOT be set.
  "noise_menu": {
    "indicators": ["bis"]              // e.g. BIS-only decorrelation
    // "include_studentt": false, ...
  },

  // ---- Sampler (src/runner.jl:956-1074) -----------------------------
  // VALID names (src/runner.jl:201-205):
  //   ptemcee, transdim_ptemcee, pt, pt_warm, rjmcmc, moms, moms_ns,
  //   nested, nested_ins, nested_dynamic, pa, smc, pt_whitening,
  //   nuts, pt_hmc, ofti
  // Trans-dim model selection (transdim_ptemcee/rjmcmc/moms/moms_ns) REQUIRES
  // a top-level `transdim` block (src/runner.jl:206-207, 336-338).
  // kwargs that the sampler doesn't declare → hard error (src/runner.jl:918-932).
  // String kwarg values are coerced to Julia Symbols (bounds/proposal/… ; 970-972).
  "sampler": {
    "name": "transdim_ptemcee",
    "kwargs": {
      "n_temps":   12,
      "n_walkers": 100,
      "n_steps":   5000,
      "n_burnin":  1500,
      "adapt_ladder": true
    }
  },

  // ---- Trans-dim (REQUIRED iff sampler ∈ {transdim_ptemcee, rjmcmc,
  //      moms, moms_ns}; src/runner.jl:1105-1124) ---------------------
  "transdim": {
    "max_kplanet": 2,                      // REQUIRED
    // birth_strategies (src/runner.jl:208-210, 1076-1082):
    //   PriorBirth | InformedBirth | JointInformedBirth | DonorBirth | MoMSBirth
    "birth_strategies": ["PriorBirth"],
    "birth_weights":    [1.0],             // must match birth_strategies length
    "transdim_fraction": 0.3,              // fraction of steps that jump dimension (default 0.3)
    "planets": true,                       // toggle planet-count dimension (default true)
    "noise":   false,                      // toggle noise-model dimension (default false)
    "toggleable": [],                      // MUST be empty via run_job (JSON can't encode
                                           //   NoiseModel objects; src/runner.jl:1091-1098)
    "noise_exclusion_groups": []           // likewise empty via run_job
  },

  // ---- Output (src/runner.jl:1130-1401, 1407-1614) ------------------
  "output": {
    "save_chains":  true,                  // chains.nc
    "save_summary": true,                  // summary.json
    "save_pdf":     false,                 // emit a .pdf next to every .png (global toggle)

    // VALID plot names (src/runner.jl:211-223, _KNOWN_PLOTS). Each no-ops when
    // its required data/model is absent.
    "plots": [
      // RV
      "rv_timeseries",        // full RV model over time
      "rv_components",        // RV model DECOMPOSITION (Keplerian/GP/trend pieces)
      "rv_phasefold",         // per-planet phase fold
      // Photometry
      "pm_timeseries",
      "pm_phasefold",
      "transit_overlay",      // per-transit QC gallery (plot_transit_overlay_fit)
      "ttv_oc",               // O−C; data-only (planet_b_k=0) for single-planet fits
      // Rossiter-McLaughlin
      "rm_anomaly",           // in-transit RM RV anomaly (plot_rm); only for *_RM modes
      // Astrometry
      "orbit_skyplane", "relastrom_timeseries", "relastrom_residuals",
      "hgca_pm_residuals", "g23h_residuals", "iad_residuals", "pm_anomaly",
      "rv_astrom_phasefold",
      // ActivityGP
      "activity_gp_latent", "activity_gp_decomposition",   // only when an ActivityGP is configured
      // Diagnostics
      "corner", "trace", "histograms", "posteriors",
      "transdim_occupancy",   // only when chain has :n_planets
      "posteriors_raw", "posteriors_parameters", "posteriors_histograms",
      "traces_grouped",
      // OR a single "auto" — picks every applicable plot from the data + model present
      "auto"
    ],
    "plot_kwargs": {                       // forwarded to each plotter (Symbol-keyed)
      "bf_cutoff": 10.0,                   // best-fit cluster filter: lp within ln(bf_cutoff) of max
      "subtract_gp": true
    },

    // Posterior predictive checks — default ON; emits ppc.png + ppc.json. RV/phot only.
    "ppc":           true,
    "ppc_n_draws":   500,

    // Bayesian K upper-limit curve K_lim(P). Default ON for PT samplers
    // (ptemcee/transdim_ptemcee/pt/pt_warm), OFF otherwise. Set true/false to
    // force. Needs RV + a P_k1 column. Per-bin arrays go to detection_limits.nc.
    "detection_limits":            true,
    "detection_limits_n_bins":     30,
    "detection_limits_confidence": 0.95,
    "detection_limits_planet":     1,

    // PSIS-LOO + WAIC via chain replay. Default ON; sampler-agnostic.
    // Auto-skipped for GP models (joint covariance breaks pointwise factorisation).
    "loo":          true,
    "loo_n_draws":  500,

    // Post-fit "silently-wrong" guard (disjoint modes / railed bounds /
    // corrupt log-post). Default ON; never aborts the run.
    "fit_health":   true
  }
}
```

---

## Field reference notes

### `version`, `seed`, `output_dir`, `n_threads`, `timeout_sec`
- `output_dir` is the only top-level **required** key (besides the `data` /
  `model` / `sampler` blocks). `mkpath`'d on entry (`src/runner.jl:36-37`).
- `seed` (default `1`) seeds every sampler **and** the PPC / LOO RNGs, so the
  same config + seed reproduces the run.
- `n_threads` is **advisory only** — it is logged but actual threading honours
  `JULIA_NUM_THREADS` (`src/runner.jl:47-49`).
- `timeout_sec` arms a wall-clock watchdog: the sampler runs on a spawned task
  polled every second; on budget overrun a `JobTimeoutError` is thrown and the
  job ends with `status="failed"` (`src/runner.jl:864-911`). **Ignored under
  `juliacall`** (the GIL deadlocks the watchdog) — enforce the timeout on the
  Python side there.

### `data`
The `data` block must contain at least one of `rv`, `transit_photometry`,
`iad`, `gost`, `hgca`, `relastrom`, `gaia_dr3` (`src/runner.jl:237-241`).

- **`rv`** — `csv` (column-named) or inline `values`. Inline `values` requires
  `bjd`, `rv`, `rv_err`, `instrument`; any other key is an activity indicator.
  A `<name>_err` key whose base `<name>` is also present is read as that
  indicator's 1σ (`_split_indicator_errs`, `src/runner.jl:480-493`); a lone
  `<name>_err` is itself an indicator value. For the CSV form, indicator errors
  come from a `<col>_err` column when present (`src/runner.jl:529-533`).
  **ActivityGP requires these indicator errors.**
  An optional reserved **`component`** (alias **`star`**) column tags each RV by
  stellar component for SB2 double-lined fits — `1`/`A`/`primary` (default) or
  `2`/`B`/`secondary` (`_rv_component_code`). It is NOT an indicator. Absent ⇒
  all-primary (the normal single-star case). CSV form: `comp_col` (default
  `"component"`). See "SB2 double-lined binaries" below.
- **`transit_photometry`** — a list; each LC has its own `instrument` and
  `exposure_time` (seconds; stored per-cadence in days, `src/runner.jl:563`).
  Instrument names are sorted-unique → integer indices.
- **`iad`** — `hip_id` (auto-fetches the Hipparcos IAD bundle) or `values`
  (`t`, `abscissa`, `abscissa_err`, `psi`, `parallax_factor`, `pm_factor`).
- **`gost`** — `{ra_deg, dec_deg, from?, to?}` (auto-fetches the Gaia GOST scan
  plan) or `values` (`t`, `psi`, `parallax_factor`). **GOST is an AUXILIARY,
  not a standalone measurement** — `gost_log_likelihood` returns 0; GOST only
  does anything when it window-averages (Mode B) the Gaia epoch of a paired
  `hgca` / `gaia_dr3` / `g23h`. Supplied alone — or only with `iad` (whose Gaia
  joint requires `gaia_dr3`) — it is INERT, and the `Data` constructor warns.
- **`hgca`** — `hip_id` (auto-fetches + caches the Brandt eDR3 `HGCA_vEDR3.fits`
  and pulls the row — the VALIDATED Hipparcos–Gaia path, recovers the orvara
  orbit on HD 159062), or `{csv, hip_id}` for a local catalog, or `values`
  (exactly 3 epochs: `t`, `pmra`, `pmdec`, `cov_ep`, plus `plx`, `plx_err`,
  `hip_id`). Pair with `gost` for the Mode-B Gaia epoch.
- **`relastrom`** — `{csv}` or `values` (`t`, `ra_off`, `dec_off`, `ra_err`,
  `dec_err`, `planet_idx?`).
- **`gaia_dr3`** — `{params:[α,δ,ϖ,μα*,μδ], cov:5×5, t_ref}`.

### `model.planet_modes` and Rossiter-McLaughlin
- One mode per planet slot. The `*_RM` / `*_RM_R` modes add the in-transit RM
  RV anomaly to **`rv_predictions`** itself (`src/likelihood.jl:1368-1369,1439`),
  so PPC, residuals, and all RV plots are RM-consistent. They introduce two
  parameters: **`v_sin_i_star`** (stellar V·sin i*, m/s) and **`lambda_k<k>`**
  (sky-projected obliquity λ, rad) — defaults `LogUniformPrior(500, 100000)`
  and `UniformPrior(-π, π)` (`src/default_priors.jl:299-304`). RM modes require
  `M_s` and `R_s`; a missing stellar pair silently skips the RM term rather than
  poisoning predictions (`src/likelihood.jl:165, 1368-1369`).
- `*_RM` is Hirano+ 2011 (analytic, fast); `*_RM_R` is the Reloaded RM of
  Cegla+ 2016 (numerical disk integration, accurate at large `rr`).

### SB2 double-lined binaries (`BINARY` / `BINARY_RV`)
An unresolved SB2 (double-lined CCF → two RV sets) is fit as ONE Keplerian
carrying **two amplitudes**: `K_A_k<k>` (primary, applied to `component==1`
points) and `K_B_k<k>` (secondary, applied ANTI-PHASE to `component==2`). Set
one planet slot's mode to `BINARY_RV` (RV only) or `BINARY` (+ absolute
astrometry). A **circumprimary planet** is a SEPARATE slot (e.g. `RV_ONLY` /
`RVPM`); its signal is present only in the primary's lines (`component==1`).

- **Data:** tag each RV with the `component` column (`1`/`A` vs `2`/`B`). The
  barycentric `gamma_<inst>` is SHARED across A and B (component is orthogonal
  to instrument). Both components of one epoch usually appear as two rows.
- **Priors auto-generated:** `K_A_k<k>`, `K_B_k<k>` (broad `ModJeffreys`); for
  `BINARY`, `inc_k<k>` (`Sine`), `Omega_k<k>` (`Uniform 0..2π`), and the
  system-level **`f_light`** = L_B/(L_A+L_B) in the astrometric band
  (`Uniform 0..0.5`) — the luminous-companion photocenter correction, which
  scales the reflex by `(1 − f_light/f_mass)`, `f_mass = K_A/(K_A+K_B)`.
  Constrain `f_light` from the CCF flux ratio when known.
- **Transit dilution:** for a transiting circumprimary planet (`RVPM`), the
  per-instrument `dilution_<inst>` prior auto-frees to `Uniform(0, 0.9)` when
  an SB2 binary is present (else it stays fixed at 0) — the transit is diluted
  by star B's constant light. This is the transit-band twin of `f_light`
  (separate params because L_B/L_A is wavelength-dependent).
- **Derived masses** (in the `derived` table, block `binary_<k>`): mass ratio
  `mass_ratio_q = K_A/K_B = M_B/M_A`, minimum masses `M_A_sini3`/`M_B_sini3`,
  and — for `BINARY` — absolute `M_A`/`M_B` and `inc_deg`.
- **Plots:** `rv_timeseries`/`rv_components` render component-colored with both
  the primary and secondary model curves; `rv_phasefold` on the binary slot is
  the classic double-lined fold (primary +K_A vs secondary −K_B), and on a
  planet slot is primary-only. `M_pri` is vestigial for a pure SB2 (M_A is
  derived from the two amplitudes).

### `model.stability`
`none | amd | gladman` (`src/runner.jl:193, 689`). AMD/Gladman both need `M_s`.

### Activity indicators — scaling

Indicators supplied through `data.rv.indicators` are **normalised at `Data`
construction**: per instrument, the median is subtracted and the result divided
by its RMS. This is the convention of Vines et al. 2023 (MNRAS 518, 2627),
Table 7 footnote — *"Activity indices were mean subtracted and normalized to
their RMS"* — and it is the default. Pass raw values; no pre-scaling is needed
or wanted. `normalize_indicators=false` on the Julia `Data` constructor disables
it for a caller who has already scaled them.

It matters because `ActivityDecorrelation` adds `C * indicator` to the RV
prediction using the raw value, so without normalisation `C` carries each
channel's units and one coefficient prior means a different thing on every
instrument. Measured on HD 18599, scaling each indicator to its instrument's RV
amplitude instead of its own RMS moved the recovered semi-amplitude from
11.9 m/s to 5.9 — it let a 7-point instrument's regression absorb the planet.
With unit-RMS regressors `C` is simply the m/s amplitude of the activity term
and is comparable across instruments and channels.

**The coefficient bound is a science choice.** Where an indicator is nearly
degenerate with the orbital period — BIS sits at P_rot/2 ~ P_orb on HD 18599 —
a coefficient free to reach the measured indicator-RV slope will absorb the
planet rather than the activity. The default prior is deliberately permissive
(see `_default_noise_priors!`); supply an explicit `C_<indicator>_<instrument>`
prior when that degeneracy is present.

### `noise_models`
- `channel` (default `"rv"`) and `instruments` (default `[]`) are injected
  **only** into constructors that accept them: celerite GPs take both;
  `ARModel`/`MAModel` take `channel`; `ActivityGP` takes `channels`;
  `ActivityDecorrelation`/`ActivityJitter` (mean modifiers) take neither
  (`src/runner.jl:824-836`).
- `kwargs.channels` (for `ActivityGP`) are symbolised, and any vector-valued
  kwarg is narrowed from a `JSON3.Array` (`src/runner.jl:818-825`).
- `validate_noise_models` runs a cross-model consistency check (per-instrument
  exclusivity, AR/MA conflicts, channel validity); violations surface as
  `status="failed"` (`src/runner.jl:842`).

### `sampler`
- Trans-dim model-selection samplers (`transdim_ptemcee`, `rjmcmc`, `moms`,
  `moms_ns`) require a top-level `transdim` block. `pt`/`pt_warm` also build a
  `transdim` block if present (`src/runner.jl:850-851`) but do not require one.
- Evidence is surfaced as `summary.log_z` per sampler: PT-family and the nested
  family return real `log_evidence`; `rjmcmc`/`moms`/`nuts`/`ofti` write
  `NaN` (no evidence estimator). `nested_ins` reports the INS estimate,
  `nested_dynamic` the dynamic-NS estimate, `moms_ns` its nested evidence
  (`src/runner.jl:1006-1069`).
- Unknown kwargs are rejected with an actionable error listing the valid set
  (`src/runner.jl:918-932`); every JSON-string kwarg value is coerced to a Julia
  `Symbol` so `"multi"`, `"rslice"`, `"stretch"`, etc. work as expected.

### `output` plots
- `auto` expands to the applicable set from the data + model present
  (`src/runner.jl:1360-1396`): always the posterior/trace/corner family; RV →
  `rv_timeseries`+`rv_phasefold` (+`rv_components` when ≥2 planets or a GP/AGP);
  photometry → `pm_*`+`ttv_oc`+`transit_overlay`; astrometry → the astrometry
  plots; any `*_RM` mode → `rm_anomaly`; an ActivityGP → `activity_gp_latent`
  + `activity_gp_decomposition`; a `:n_planets` chain → `transdim_occupancy`.
- Individual plots are fail-soft (a failing plotter logs a warning and is
  skipped, `src/runner.jl:1150-1156`).
- `ttv_oc` renders for single-planet fits too — it shows the data-only O−C
  (measured per-transit `Tc` vs the linear ephemeris) via `planet_b_k=0`; with
  ≥2 planets it defaults to the perturber-driven envelope (`planet_b_k=2`,
  `src/runner.jl:1264-1281`).

---

## summary.json (output contract)

`run_job` always returns / writes a `summary` dict. The legacy unconditioned
`params` block is **removed** before writing (`src/runner.jl:108`) — the
authoritative, model-conditioned numbers live under `fitted` / `derived`
(from `science_summary`, `src/science_tables.jl:550-601`).

```jsonc
{
  "status":         "ok",                  // "ok" | "failed"  (failed adds error+traceback)
  "config_version": "0.2.0",
  "chains_path":    "chains.nc",
  "config_path":    "/work/in/config.json",
  "elapsed_sec":    612.3,

  // ---- Evidence + sampler metrics (_populate_summary!) ----
  "log_z":   12345.67,                      // NaN for rjmcmc/moms/nuts/ofti
  "n_evals": 4501500,
  "sampler": {
    "acceptance_within":   [0.32, 0.28, /*…*/],
    "acceptance_swap":     [0.21, 0.19, /*…*/],
    "acceptance_transdim": [0.18, 0.05]     // present only for trans-dim results
  },

  // ---- Trans-dim planet-count posterior (when chain has :n_planets) ----
  "n_planets_posterior": { "0": 0.05, "1": 0.82, "2": 0.13 },

  // ---- Authoritative science contract (science_summary) ----
  "fitted": {                               // model-conditioned fitted parameters
    "conditioning": { "n_planets": 1, "stellar": { /*…*/ } },
    "parameters": {
      // entry schema (sci_entry_dict, src/science_tables.jl:110-121):
      "P_k1": { "value": 14.31,             // best (median)
                "err_lo": 0.02, "err_hi": 0.02,   // asymmetric 1σ (med−16th, 84th−med)
                "ci3": [14.24, 14.40],      // 3σ credible interval [lo, hi]
                "unit": "d",
                "occupancy": 1.0,           // P(component | D)
                "n_used": 12000 }
      // … one entry per active parameter
    }
  },
  "derived": {                              // physical quantities; skipped if stellar params unset
    "conditioning": { "n_planets": 1, "stellar": { /*…*/ } },
    "parameters": {
      "msini_earth_k1": { /*…*/ }, "a_au_k1": { /*…*/ }, "T_eq_k1": { /*…*/ },
      "rho_p_cgs_k1": { /*…*/ }, "tsm_k1": { /*…*/ }, "esm_k1": { /*…*/ }
    }
  },
  "model_selection": {                      // occupancy = P(component | D)
    "n_planets": { "0": 0.05, "1": 0.82, "2": 0.13 },
    "noise_models": { /* per-model occupancy + CIs, when toggled */ }
  },
  "run_info": {
    "git_hash":        "abcd123",
    "convergence":     { "assessed": true, "pass": true, "worst_rhat": 1.01,
                         "worst_rhat_param": "P_k1", "min_ess": 812.4,
                         "min_ess_param": "K_k1", "n_fail": 0 },
    "priors":          { /* prior block per param */ },
    "data_provenance": { /* per-instrument provenance */ },
    "stellar":         { /* M_s, R_s, T_eff, … */ },
    "log_evidence":    12345.67,
    "n_evals":         4501500
  },
  "tables": {                               // multi-format science tables on disk
    "fitted":  { "json": "tables/fitted.json",  "csv": "…", "ecsv": "…",
                 "dat": "…", "tex": "…" },
    "derived": { "json": "tables/derived.json", "csv": "…", /*…*/ }
  },

  // ---- Figures manifest: logical_name → absolute path (scanned from plots/) ----
  // logical_name = path relative to plots/ without extension
  //   e.g. "models/rv_timeseries", "posteriors/raw/raw_K_k1", "transdim/occupancy"
  "figures": {
    "models/rv_timeseries": "/work/out/plots/models/rv_timeseries.png",
    "transdim/occupancy":   "/work/out/plots/transdim/occupancy.png"
  },
  // legacy flat list of generated plot paths (PPC/detection-limits pushed here too)
  "plots": [ "ppc.png", "detection_limits.png" ],

  // ---- Posterior predictive check (output.ppc, default on) ----
  "ppc": {
    "n_draws":                    500,
    "rv_red_chi2_total":          0.98,     // best fit found in the posterior
    "rv_red_chi2_median_model":   1.10,     // χ²/dof at the per-param median Θ
    "rv_red_chi2_per_inst":       { "1": 1.04, "2": 1.12 },
    "rv_red_chi2_per_draw_p16":   0.97,
    "rv_red_chi2_per_draw_p50":   1.11,
    "rv_red_chi2_per_draw_p84":   1.42,
    "rv_outliers_3sigma":         2,
    "rv_residual_top_peak_period": 38.4,    // top GLS-of-residuals peak (missed signal);
    "rv_residual_top_peak_power":  0.18,    //   grid capped ≤20000 freqs, analytic FAP
    "rv_residual_top_peak_fap":    0.41,
    "rv_residual_acf_lag1":        0.05     // > 0 → correlated unmodelled noise
  },

  // ---- PSIS-LOO + WAIC (output.loo, default on; "skipped" for GP models) ----
  "loo": {
    "n_draws": 500, "n_obs": 170,
    "elpd_loo": -245.31, "se_elpd_loo": 12.4, "p_loo": 7.2,
    "elpd_waic": -245.18, "se_elpd_waic": 12.4, "p_waic": 7.1,
    "pareto_k_max": 0.51,                   // > 0.7 → PSIS-LOO unreliable
    "pareto_k_warn_count": 0,
    "loo_compare_log_z": 8.7                // elpd_loo − PT log Z; sign disagreement = misspec.
  },

  // ---- Detection limits (PT samplers, default on) ----
  // scalar metadata only; per-bin arrays in detection_limits.nc
  "detection_limits": {
    "planet_index": 1, "confidence": 0.95,
    "prior_P_lo": 0.5, "prior_P_hi": 365.0,
    "n_bins": 30,
    "arrays_path": "detection_limits.nc"    // K_limit[], P_centers[], P_edges[], n_in_bin[]
  },

  // ---- Post-fit health guard (output.fit_health, default on) ----
  "fit_health": {
    "overall": "ok",                        // "ok" | "warn" | "fail"
    "checks": { "disjoint_modes": { "status": "ok", "message": "…" }, /*…*/ }
  }
}
```

A `"failed"` summary additionally carries `"error"` (the message) and
`"traceback"`; the auxiliary stages instead write a nested
`{"status":"failed"|"skipped","error":…}` under their own key and the job
continues (`src/runner.jl:122-130, 1434-1438, 1501-1505, 1548-1552, 1609-1613`).

---

## Minimal example

The smallest valid RV-only config (a single CSV, one planet, fixed-dim PT):

```json
{
  "version": "0.2.0",
  "seed": 42,
  "output_dir": "/work/out",
  "star": { "M_s": 1.02 },
  "data": {
    "rv": {
      "csv": "/work/in/rvs.csv",
      "time_col": "bjd", "rv_col": "rv", "err_col": "rv_err",
      "inst_col": "instrument"
    }
  },
  "model": {
    "max_kplanet": 1,
    "planet_modes": ["RV_ONLY"]
  },
  "sampler": {
    "name": "ptemcee",
    "kwargs": { "n_temps": 10, "n_walkers": 100, "n_steps": 5000, "n_burnin": 1500 }
  },
  "output": { "plots": ["auto"] }
}
```

A trans-dim model-selection run only differs by the sampler + a `transdim`
block:

```json
{
  "version": "0.2.0",
  "output_dir": "/work/out",
  "star": { "M_s": 1.02 },
  "data": { "rv": { "csv": "/work/in/rvs.csv" } },
  "model": { "max_kplanet": 3, "planet_modes": ["RV_ONLY", "RV_ONLY", "RV_ONLY"] },
  "sampler": {
    "name": "transdim_ptemcee",
    "kwargs": { "n_temps": 12, "n_walkers": 100, "n_steps": 5000, "n_burnin": 1500 }
  },
  "transdim": {
    "max_kplanet": 3,
    "birth_strategies": ["PriorBirth", "InformedBirth"],
    "birth_weights":    [0.5, 0.5],
    "transdim_fraction": 0.3,
    "planets": true
  },
  "output": { "plots": ["auto"], "detection_limits": true }
}
```

---

## Behaviour spec

- **Validation is fail-fast** — `_validate_config` collects *all* schema errors
  and throws one `ArgumentError` listing them, before any heavy allocation
  (`src/runner.jl:227-378`).
- **Threading** — `n_threads` is advisory; honour `JULIA_NUM_THREADS`.
- **Determinism** — `seed` controls every RNG (sampler, PPC, LOO).
- **Errors** — any uncaught exception writes
  `{"status":"failed","error":…,"traceback":…}` and rethrows (non-zero exit).
- **Auxiliary stages are fail-soft** — PPC, detection limits, LOO/WAIC, and the
  fit-health guard each record a `"failed"`/`"skipped"` summary entry and never
  abort the job.
- **Auto-fetch** — HIP IAD bundles and GOST scan plans are fetched on demand
  from `hip_id` / `ra_deg,dec_deg`.
```
