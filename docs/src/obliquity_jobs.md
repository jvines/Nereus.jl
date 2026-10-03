# Obliquity fits as Nereus targets

An obliquity fit — the Rossiter–McLaughlin (RM) anomaly in the velocities, the
Doppler shadow in the line profiles, or both at once — is an ordinary Nereus
target: a `Params`, a `Data` and a `NereusTarget`, run by any Nereus sampler.
It is configured through `run_job` like every other fit, with two data blocks
(`rm_nights`, `tomography`) and a `model.obliquity` block, and it writes
`chains.nc` and `summary.json` like every other fit. From Julia the same model
comes from `obliquity_data` + `obliquity_params`; `run_job` calls exactly
those.

The defaults below are the **standard configuration**. Every one of them is an
option.

## The model

**Velocities.** One RM night is one RV instrument, with its own offset γ_n,
jitter σ_n and noise:

```
RV(t) = Kep(K, P, Tc) + γ_n + RM(t; λ, v sin i, b, a/R★, R_p/R★, u1, u2, σ0_n) + GP_n
```

The anomaly is the ARoME kernel (Boué et al. 2013, Eq. 15), the RM shift a
Gaussian fit to a CCF actually returns. Its two widths are not free: σ0_n is
the MEASURED dispersion of the night's out-of-transit CCF, and the sub-planet
width follows from it at every evaluation,

```
β_p² = σ0_n² − (0.5503 v sin i)²        (floored at `beta_p_floor`, default 0)
```

so the amplitude is fixed by v sin i, R_p/R★ and the limb darkening, with
nothing free to rescale it. A free σ0/β_p pair (the behaviour without per-night
σ0) lets the ARoME prefactor scale the anomaly anywhere in (0, 1.27]× — a free
RM amplitude — which is why the standard configuration has none.

The occulted flux fraction f is the exact overlap of the planet's disc with the
limb-darkened star (`occultation = "disc"`, the true fraction the ARoME formula
calls for) or the point-planet approximation f = (R_p/R★)² I(μ)/⟨I⟩ at the
planet's centre, zero unless the centre is on the disc (`"point"`).

`GP_n` is, per night, a damped harmonic oscillator (celerite SHO:
`gp_log_S0_<tag>`, `gp_log_Q_<tag>`, `gp_log_omega0_<tag>`, natural logs, ω0 in
rad/day). Per night because the pulsation phase and the amplitude a night
happens to sample are not shared between epochs a year apart.

**Line profiles.** Each residual map n is a shadow plus Gaussian noise of
covariance

```
Σ_n = K_t(A_n, ℓ_t,n) ⊗ K_v(ℓ_v,n) + σ_n² I
```

with Matérn-3/2 factors in time and velocity. The shadow is a Gaussian of width
σ_line,n at the sub-planet velocity, scaled by α_n × the occulted flux
(point planet, limb-darkened). Per night: α_n (`tomo_alpha_<tag>`, or one
shared `tomo_alpha` with `shared_alpha`), σ_line,n (`tomo_sigma_line_<tag>`,
km/s), ℓ_v,n (`tomo_ell_v_<tag>`, km/s), σ_n (`tomo_jit_<tag>`, map units),
and the temporal kernel from the noise menu: `matern_sigma_tomo_<tag>` (the
amplitude A_n) and `matern_rho_tomo_<tag>` (ℓ_t,n, in DAYS). The likelihood is
exact and cheap: the Kronecker structure is diagonalised by the two factors'
eigendecompositions.

**Shared.** λ (`lambda_k1`), v sin i (`v_sin_i_star`, m/s), b (`b_k1`), a/R★
(`a_Rs_k1`), R_p/R★ (`rr_k1`), P (`P_k1`) and the transit time (`Tc_k1`) are
single slots read by both observables. Every RM night and every map is placed
on the transit of the model ephemeris nearest it, Tc_n = Tc + E·P, so the two
can never see different ephemerides — and when P or Tc are free, both follow.
A circular orbit is the default; with `ecc = "free"` both observables use the
eccentric sky position.

**λ is wrapped, never bounded.** Its prior is `WrappedUniformPrior`: uniform on
the circle, density 1/2π at every value, charted on [−π, π) (or wherever its
bounds say). No sampler sees a wall at the seam: ensemble moves cross it, and
the unconstrained samplers (NUTS, `pt_hmc`) use an identity transform for it
instead of a logit. Draws are relabelled into the chart when the run is
summarised.

## Priors of the standard configuration

| parameter | prior | |
|---|---|---|
| `lambda_k1` | WrappedUniform(−π, π) | the full circle |
| `v_sin_i_star` | N(μ, σ) from `vsini` (m/s) | |
| `P_k1`, `Tc_k1`, `rr_k1` | fixed, or N(μ, σ) | a number fixes, `[μ, σ]` is Gaussian |
| `b_k1`, `a_Rs_k1` | fixed, or N(μ, σ) | a/R★ itself, not a density proxy |
| `K_k1` | N(μ, σ) truncated at 0 | with velocities |
| `sesinw_k1`, `secosw_k1` | fixed at 0 | `ecc = "free"` samples them |
| `gamma_<tag>` | U(μ_n ± s) | μ_n the night's mean RV; s the Nereus γ rule: max(3 × largest per-night MAD, 3 × the night's std, 100 m/s) |
| `sigma_<tag>` | logU(0.1, 3162) m/s | |
| `gp_log_S0_<tag>` | U(ln 1e-2, ln 1e12) | |
| `gp_log_Q_<tag>` | U(ln 0.2, ln 100) | |
| `gp_log_omega0_<tag>` | U(ln 0.2, ln 60) | rad/day |
| `tomo_alpha_<tag>` | U(0, 20) | in units of the naive shadow |
| `tomo_sigma_line_<tag>` | U(2, 25) km/s | |
| `tomo_ell_v_<tag>` | logU(1, 60) km/s | |
| `tomo_jit_<tag>` | logU(1e-6, 1) | maps in unit line depth* |
| `matern_sigma_tomo_<tag>` | logU(1e-6, 1) | the Kronecker amplitude A* |
| `matern_rho_tomo_<tag>` | logU(0.05/24, 12/24) d | 0.05–12 h |
| `u1_spec`, `u2_spec` | fixed at `limb_darkening` | |

\* `tomogram_residuals` scales maps to unit line depth. For a map in other
units (scatter ≥ 1) the upper bound is ten times its scatter.

Any of them can be overridden by name through the job's `priors` block (or the
`priors` keyword), including `WrappedUniformPrior`.

## The job config

```jsonc
{
  "output_dir": "/work/out/lambda_joint",
  "seed": 20260813,
  "data": {
    "rm_nights": [                       // one entry per spectroscopic transit
      { "tag": "CORALIE",                // the night's instrument name (unique)
        "file": "/data/rm/CORALIE.dat",  // columns: BJD, RV, RV error (m/s); '#' comments
        "columns": [1, 2, 3],            // optional, 1-based
        "rv_unit": "m/s",                // optional: "m/s" (default) or "km/s"
        "sigma0": 15702.0,               // REQUIRED: measured out-of-transit CCF dispersion, m/s
        "instrument": "CORALIE",         // recorded in summary.json
        "pipeline": "DRS CCF centroid" },
      { "tag": "HARPS", "file": "/data/rm/HARPS.dat", "sigma0": 15912.0,
        "instrument": "HARPS", "pipeline": "SERVAL" }
      // or "values": {"bjd": [...], "rv": [...], "rv_err": [...]} instead of "file"
    ],
    "tomography": [                      // one entry per line-profile stack
      { "tag": "CORALIE",
        "profiles": "/data/tomo/CORALIE_prof.dat",   // n_exposure x n_v, one profile per row
        "vgrid":    "/data/tomo/CORALIE_vgrid.dat",  // km/s
        "times":    "/data/tomo/CORALIE_t.dat",      // BJD
        "berv":     "/data/tomo/CORALIE_berv.dat",   // km/s; required with several stacks
        "vsys": 18.93,                               // km/s
        "grid": [-42, 42, 57],                       // residual-map velocity grid (km/s)
        "t14_hours": 2.634 },                        // or a shared model.obliquity.t14_hours
      { "tag": "HARPS", "profiles": "...", "vgrid": "...", "times": "...", "berv": "...",
        "vsys": 18.93, "grid": [-42, 42, 57], "t14_hours": 2.634 }
    ]
    // "rv" and "transit_photometry" blocks may be added as in any fit
  },
  "model": {
    "obliquity": {
      "fit": "joint",                    // "velocities" | "shadow" | "joint": which data blocks are read
      "P":  2.82796938,                  // number = fixed; [mean, sd] = Gaussian prior
      "Tc": 2459986.40824,
      "b":    [0.7519, 0.0100],
      "a_Rs": [6.822, 0.086],
      "rr":   0.11577,
      "vsini": [25900, 1500],            // m/s
      "K":    [367.9, 26.9],             // m/s (with velocities)
      "limb_darkening": [0.32, 0.30],    // quadratic u1, u2 of the spectroscopic band
      "occultation": "disc",             // "disc" (default) | "point"
      "beta_p_floor": 0.0,               // m/s
      "shared_alpha": false,
      "ecc": 0,                          // 0 (circular, default) | "free"
      "lambda": "wrapped",               // "wrapped" (default) | "bounded"
      "a_Rs_param": "a_Rs",              // "a_Rs" (default) | "rho_s" | "kepler"
      "rv_noise": "sho",                 // per RM night: "sho" (default) | "matern" | "white"
      "tomo_noise": "matern",            // per map: "matern" (default) | "sho" | "white"
      "noise_menu": false,               // true: select per-night noise trans-dimensionally
      "t14_hours": 2.634
    }
  },
  "priors": {                            // optional; overrides by name
    "lambda_k1": { "type": "WrappedUniformPrior", "args": [0, 6.283185307179586] }
  },
  "sampler": {
    "name": "pt_emcee",
    "kwargs": { "n_temps": 1, "n_walkers": 300, "n_steps": 40000,
                "n_burnin": 20000, "thin": 5 }
  },
  "output": { "plots": ["rm_anomaly", "rv_timeseries", "corner", "traces_grouped"] }
}
```

`model.obliquity` replaces `model.max_kplanet` / `planet_modes` (the model is
one planet: `RVPM_RM_A` with velocities, `PM_DT` — λ and v sin i, no RV — for
the shadow alone). The fit kind decides which blocks are read, so one data
section serves all three. `sigma0` may also be given for other RV instruments
in `model.obliquity.sigma0` (`{"FEROS": 15000}`) — every RV instrument needs
one, since an in-transit point of any instrument carries the anomaly.

Schema errors are collected and reported before anything is read: an unknown
fit kind or option, a missing `sigma0`, a stack without `t14_hours` or (with
several stacks) without `berv`, a geometry value that is neither a number nor
`[mean, sd > 0]`, a file that does not exist, `noise_menu` without a
`transdim` block.

**Checkpoint / resume.** `pt_emcee` (and every sampler that checkpoints)
writes `pt_emcee_state.jls` into `output_dir` every 15 min and at the end.
Re-run the same job with `"resume": true` in `sampler.kwargs` and a larger
`n_steps` to continue it.

**Outputs.** `chains.nc` carries every sampled parameter under the names
above; `summary.json` the usual tables plus an `obliquity` block (fit kind,
each RM night's tag, instrument, pipeline, σ0 and point count, each map's
size and transit time, the model options). The posterior predictive check,
PSIS-LOO and the detection-limit curve are RV-planet diagnostics and are off
unless `output` asks for them. Useful plots: `rm_anomaly`, `rv_timeseries`,
`corner`, `traces_grouped`, `posteriors_histograms`.

## From Julia

```julia
data, names = obliquity_data(rm_nights; tomo_nights = maps)   # RMNight / TomoNight
params = obliquity_params(data, names;
             P = 2.82796938, Tc = 2459986.40824,
             b = (0.7519, 0.0100), a_Rs = (6.822, 0.086), rr = 0.11577,
             vsini = (25_900.0, 1_500.0), K = (367.9, 26.9),
             sigma0 = rm_nights, ld = (0.32, 0.30))
target = NereusTarget(params, data)
chains = sample_pt_emcee(target, data; n_temps = 1, n_walkers = 300, n_steps = 40_000,
                         n_burnin = 20_000, thin = 5).chains
```

`obliquity_noise_models(rv_tags, map_tags)` builds the standard per-night
noise; `obliquity_noise_menu(rv_tags, map_tags)` the trans-dimensional menu
(per night: white, oscillator or Matérn, at most one) for
`transdim_pt_emcee`, `rjmcmc`, `moms` or `daedalus`.

## Samplers

Every Nereus sampler runs the target: `pt_emcee` (any number of temperatures),
`pt`, `pt_whitening`, `ensemble`, `ess`, the nested family (`nested`,
`nested_ins`, `nested_dynamic`), `pa` / `smc`, and the trans-dimensional
samplers with `noise_menu`. Evidence estimators and resume work as for any
target.

The likelihood is differentiable with ForwardDiff, including the Kronecker
term (its derivative is analytic, from the same two eigendecompositions), so
`nuts` and `pt_hmc` run it too. `ad_backend = :Enzyme` is refused with an
error: that path differentiates only the white-noise RV likelihood and would
silently drop the maps and the noise models.

## Changes to earlier behaviour

The options above are additions; where the old behaviour was a bug it was
fixed instead, everywhere:

- **Samplers ignored the maps.** Only `NereusTarget`'s log-density included the
  Doppler-tomography term; `pt_emcee`, `transdim_pt_emcee`, `pt_whitening`,
  `pa`/`smc`, `rjmcmc` (and with it `pt`, `moms`, `daedalus`), `ess` and the
  nested samplers assembled their own likelihood without it.
- **The shadow used each map's fixed `Tc`**, while the velocities used the
  fitted ephemeris. It now follows the fitted ephemeris too.
- **The white term of a map was fixed at its scatter** while a correlated
  kernel was active, so the variance was counted twice. It is now the free
  `tomo_jit_<tag>`.
- **A `:tomo` noise model scoped to a night was ignored** (its parameters were
  looked up without their suffix, so the map was silently white). Scoped
  models now cover the maps they name; a name that is no map is an error.
- **The `:tomo` SHO kernel decoded its parameters as log10** while its priors
  are natural-log; it now decodes natural logs. The `:tomo` Matérn and SHO
  default priors are in map units and hours instead of m/s and days.
- **Circular orbits had NaN gradients** (`atan(0, 0)`), so NUTS and `pt_hmc`
  failed on any fixed-circular fit.
- **`pt_emcee` with `n_temps = 1`** threw from its progress readout.
- **Residual maps alone** needed a dummy RV point; `Data` now accepts them.
