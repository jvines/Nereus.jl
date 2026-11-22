# CCF line components

A spectroscopic blend shows up in the cross-correlation function as more than
one line. `data.ccf` fits the CCFs themselves as a sum of Gaussian components,
each either **static** or **tied to the orbit of one of the model's planets**,
so a blend is an ordinary Nereus target: it runs under every engine, evidence
estimator and checkpoint, and combines with RVs, photometry, transit times and
astrometry through the shared orbit.

Source: `src/ccf_data.jl` (container, layout slots), `src/ccf_likelihood.jl`
(likelihood, model), `_parse_ccf_blocks` in `src/runner.jl` (job config).

## Model

For the CCF of epoch `n` (time `t_n`) of a set, at velocity `v_i`:

```
m_ni = c_n − s_n Σ_j d_j exp(−(v_i − v_j(t_n))² / (2 σ_j²)),   σ_j = FWHM_j / (2√(2 ln 2))

v_j(t) = γ_j                     static component
       = γ_j + K_j g_k(t)        component tied to planet k
g_k(t) = cos(f + ω) + e cos ω
```

`g_k` is planet k's unit-amplitude Keplerian, computed from the planet's own
`P`, time anchor, `e` and `ω` exactly as the RV model computes it: ω is the
argument of periastron of the reflex the RVs measure and the transit is at
`f + ω = π/2`. So a tied component follows the transit ephemeris, and
`K_j > 0` moves it in phase with the star's reflex (`K_j < 0`: antiphase, the
companion's own lines). `K_j` is per component and in km/s; the planet's RV
`K_k` (m/s) is a different parameter. A component's `planet` must exist in the
model; on a trans-dimensional fit, ln L = −Inf while that planet is inactive.

The likelihood is Gaussian and independent per pixel:

```
ln L = −½ Σ_n Σ_i [ (y_ni − m_ni)² / (e_ni² + j_n²) + ln 2π (e_ni² + j_n²) ]
```

`e_ni` are optional per-pixel errors (zero when not given, so the jitter `j_n`
is the whole noise). It is added to the posterior by `rv_log_likelihood`,
beside the astrometry and the transit times, so every engine sees it; it is
ForwardDiff-differentiable (NUTS, pt_hmc, Pathfinder, MAP). `ad_backend =
:Enzyme` is refused on a target with CCFs.

**Correlated pixels.** A CCF is oversampled, so neighbouring pixels are
correlated. The likelihood treats them as independent: point estimates are not
biased by it, but posterior widths are too narrow by roughly the square root of
the oversampling factor. Treat CCF-only credible intervals as lower bounds, or
compare them with an epoch bootstrap.

## Parameters

Per set (tag `<tag>`), per component `<c>`:

| name | meaning | default prior |
|---|---|---|
| `ccf_gamma_<tag>_<c>` | centre (static) or systemic velocity (tied), km/s | Uniform over the velocity range of the CCFs |
| `ccf_K_<tag>_<c>` | semi-amplitude, either sign, km/s (tied only) | Uniform(−span, span), span the width of that range |
| `ccf_fwhm_<tag>_<c>` | FWHM, km/s | Uniform(2 grid steps, span) |
| `ccf_depth_<tag>_<c>` | depth at the reference epoch | Uniform(0, 1) normalised; else Uniform(0, 2 × peak-to-peak) |

Per epoch `<label>` (or one per set when shared):

| name | meaning | default prior |
|---|---|---|
| `ccf_cont_<tag>_<label>` / `ccf_cont_<tag>` | continuum level | Uniform(0.95, 1.05) × the range of the edge levels |
| `ccf_scale_<tag>_<label>` | depth scale, multiplies every depth | Uniform(0.2, 3) |
| `ccf_jit_<tag>_<label>` / `ccf_jit_<tag>` | white noise, normalised units | LogUniform(10⁻⁶ h, h), h = max(1, 10 × CCF scatter) |

**Depth normalisation.** A free depth scale at every epoch would be exactly
degenerate with the depths (one common factor). Under `depth_scale =
:per_epoch` the reference epoch (`ref_epoch`, default the first) has no scale
parameter: its scale is 1, and `ccf_depth_*` are the depths at that epoch. The
products `d_j s_n` are what the data identify. `depth_scale = :none` uses the
same depths at every epoch.

**Label switching.** Two components of the same type with overlapping priors
can swap. Give them different types (static vs tied), or priors that separate
them (FWHM or γ ranges).

**Normalisation.** By default each CCF (and its errors) is divided by the median
of its 5 first and 5 last points (`normalize_edges`), so the continuum is ≈ 1
and depths are fractions of it. `normalize_edges = 0` uses the values as given.

Pixels whose velocity, value or error is not finite (or `null` in JSON) are
dropped. Velocity grids may differ between epochs.

## Julia

```julia
comps = [CCFComponent("A"), CCFComponent("B"; planet = 1)]   # static, tied to planet 1
s = CCFSet(; tag = "FEROS", t = bjd, v = velocity_grids, ccf = ccfs,
           labels = string.(ids), components = comps)           # options: see ?CCFSet
data = Data(; ccf = [s])            # alone, or with t_rv/rv/..., t_phot/flux/...
```

`ccf_model(theta, data)` gives the model, residuals and per-pixel σ of every
epoch; `ccf_component_velocities(theta, data)` the component centres;
`load_ccf_json(path)` reads a database CCF dump (`velocities`, `ccf_values`,
`id`).

## Job config (`data.ccf`)

One block, or a list of blocks (one per instrument):

```json
"ccf": [{
  "tag": "FEROS",
  "components": [{"name": "A"}, {"name": "B", "planet": 1}],
  "epochs": [
    {"bjd": 2459267.59447, "file": "ccf/27930.json"},
    {"bjd": 2459269.59447, "velocities": [39.0, 39.2], "ccf": [0.99, 0.98],
     "err": [0.01, 0.01], "label": "n2"}
  ],
  "normalize_edges": 5,
  "continuum": "per_epoch",
  "depth_scale": "per_epoch",
  "ref_epoch": 1,
  "jitter": "per_epoch"
}]
```

- `components` — `planet` absent or 0: static; `k`: tied to planet k.
- `epochs` — `bjd` (BJD_TDB, on the time base of the other data) and either a
  CCF dump `file` (JSON with `velocities`, `ccf_values`, optional `id`) or
  inline `velocities` + `ccf` (+ optional `err`). `label` defaults to the
  dump's `id`, else the position.
- `ref_epoch` — 1-based position or a label.
- `continuum`, `jitter` — `"per_epoch"` or `"shared"`; `depth_scale` —
  `"per_epoch"` or `"none"`.

A planet a component is tied to needs a mode with an orbit; with CCFs only, use
`RV_ONLY` and fix what the CCFs cannot constrain (the planet's `K_k1`, and
usually the orbit from an earlier fit).

### Example: NOI-101822, CCFs only

Two components on the transit ephemeris of a previous transit+RV fit (P, Tc, e,
ω held fixed), per-epoch continuum, depth scale and jitter:

```json
{
  "output_dir": "out/noi101822_ccf",
  "data": {"ccf": [{
    "tag": "FEROS",
    "components": [{"name": "A"}, {"name": "B", "planet": 1}],
    "epochs": [{"bjd": 2459267.59447, "file": "ccf/27930.json"},
               {"bjd": 2459269.59447, "file": "ccf/27874.json"}]
  }]},
  "model": {"max_kplanet": 1, "planet_modes": ["RV_ONLY"],
            "parametrization": {"time": "Tc"}},
  "priors": {
    "P_k1":      {"type": "fixed", "args": [9.483157326]},
    "Tc_k1":     {"type": "fixed", "args": [2460097.5261792196]},
    "sesinw_k1": {"type": "fixed", "args": [0.5229]},
    "secosw_k1": {"type": "fixed", "args": [-0.1092]},
    "K_k1":      {"type": "fixed", "args": [3021.0]},
    "ccf_gamma_FEROS_A": {"type": "uniform", "args": [40, 80]},
    "ccf_gamma_FEROS_B": {"type": "uniform", "args": [40, 80]},
    "ccf_K_FEROS_B":     {"type": "uniform", "args": [-40, 40]}
  },
  "sampler": {"name": "pt_emcee",
              "kwargs": {"n_temps": 4, "n_walkers": 160, "n_steps": 3000, "n_burnin": 3000}}
}
```

(All 19 epochs go in `epochs`.) Component A is static here; to test whether it
moves, make it `{"name": "A", "planet": 1}` and compare the evidences.

### Example: NOI-101822, joint with the transit and the RVs

The same block next to the run's existing `rv` and `transit_photometry`, with
the planet in `RVPM` and the orbit free: P, Tc, e and ω are then constrained by
the transit, the RVs and both CCF components at once.

```json
"data": {
  "rv": {"csv": "rv.csv"},
  "transit_photometry": [{"instrument": "TESS", "csv": "tess.csv"}],
  "ccf": [{"tag": "FEROS",
           "components": [{"name": "A"}, {"name": "B", "planet": 1}],
           "epochs": [{"bjd": 2459267.59447, "file": "ccf/27930.json"}]}]
},
"model": {"max_kplanet": 1, "planet_modes": ["RVPM"]}
```

## What is and is not covered

- PPC, PSIS-LOO and the detection limits read the RVs and photometry only; the
  CCF term is in the posterior but not in their pointwise likelihoods.
- There are no CCF plots yet; `ccf_model` returns everything needed to draw
  them.
- Components are independent between sets: two instruments' CCFs fit separate
  γ, K, FWHM and depth per component.
