# Parametrisations

Nereus lets you choose how planet parameters are parametrised
without changing the underlying physics. Different choices condition
the posterior geometry differently — pick the one that gives the
sampler the smoothest landscape.

The choices are bundled in `ParametrizationConfig` and `PlanetDataSources`.

!!! note "Two ways in"
    Everything below is the Julia `Params(...)` / `ParametrizationConfig(...)`
    API. When you drive Nereus through `Nereus.run_job(cfg)` (the JSON/Dict
    entry point — see [`docs/JOB_CONFIG.md`](https://github.com/) and
    `src/runner.jl`), the same options live in the `model` block, but the
    run_job schema currently exposes a **strict subset** of them. Each section
    flags what is reachable from run_job and what is Julia-API-only.

## `PlanetDataSources` — which observables a planet contributes to

A planet is described by the **set** of data sources that see it
(`src/model.jl:71`). It is a composable `Set{Symbol}`, not an enum, so
RV / photometry / astrometry / Rossiter–McLaughlin / TTV compose
cleanly. Nereus exports the common combinations as named constants
(`src/model.jl:91`–`121`):

| Constant | Sources | Use case |
|---|---|---|
| `RV_ONLY` | RV | RV planet, no transit, no astrometric signature (Keplerian-only) |
| `PM_ONLY` | photometry | Transit-only candidate (no RV data / no RV detection yet) |
| `RVPM` | RV + photometry | Standard transiting planet with confirmation RV |
| `RVAS` | RV + astrometry | Long-period companion seen in both RV and astrometry (HGCA/IAD/imaging) |
| `RVPMAS` | RV + photometry + astrometry | Joint RV + transit + astrometric (rare but supported) |
| `RVPM_RM` | RV + photometry + RM | Transiting planet with Rossiter–McLaughlin (Hirano+ 2011) |
| `RVPMAS_RM` | RV + photometry + astrometry + RM | Above, plus astrometry |
| `RVPM_RM_R` | RV + photometry + Reloaded-RM | RM via Cegla+ 2016 disk integration |
| `RVPMAS_RM_R` | … + astrometry | Above, plus astrometry |
| `RVPM_TTV` | RV + photometry + TTV | Transiting planet with free per-transit timing offsets (TTV-A) |
| `RVPMAS_TTV` | … + astrometry | Above, plus astrometry |
| `RVPM_RM_TTV` | RV + photometry + RM + TTV | RM + free TTVs |
| `PM_TTV` | photometry + TTV | Transit-only with free TTVs |
| `RVPM_TTV_NB` | RV + photometry + N-body TTV | TTVs **predicted** from mutual gravity (TTVFaster / N-body) |
| `RVPMAS_TTV_NB` | … + astrometry | Above, plus astrometry |
| `PM_TTV_NB` | photometry + N-body TTV | Transit-only N-body TTVs |
| `BINARY_RV` | RV (double-lined) | SB2 spectroscopic binary — one Keplerian, **two** amplitudes `K_A`/`K_B` |
| `BINARY` | RV + astrometry (double-lined) | SB2 binary with absolute astrometry → absolute component masses |

The two `BINARY*` modes fit an unresolved **SB2 double-lined binary**:
one Keplerian carrying independent `K_A_k<k>` (primary, applied to
`rv_comp == 1` points) and `K_B_k<k>` (secondary, applied anti-phase to
`rv_comp == 2`), so the mass ratio `q = M_B/M_A = K_A/K_B` is data-driven.
A **circumprimary planet** is a SEPARATE slot (e.g. `RV_ONLY` / `RVPM`) —
its signal is present only in the primary's lines. `BINARY` adds
`inc_k<k>`, `Omega_k<k>`, and a system-level `f_light` photocenter term
(→ absolute `M_A`/`M_B`). Tag each RV point with `Data.rv_comp`. See the
SB2 example in [Worked examples](examples.md) and the derived masses in
[Diagnostics](diagnostics.md).

You can also build custom combinations directly:

```julia
PlanetDataSources(:RV)            # ≡ RV_ONLY
PlanetDataSources(:RV, :PM)       # ≡ RVPM
PlanetDataSources(:PM)            # ≡ PM_ONLY
```

Membership tests use `in` (`src/model.jl:77`); predicate helpers
`has_rv`, `has_pm`, `has_as`, `has_rm`, `has_rm_r`, `has_any_rm`,
`has_ttv`, `has_ttv_nb` are exported (`src/model.jl:131`–`138`).

You pass one mode per planet to `Params`:

```julia
params = Params(;
    max_kplanet  = 3,
    planet_modes = [RVPM, RV_ONLY, RV_ONLY],   # planet 1 is transiting, 2 & 3 are RV-only
    instruments  = ic,
    data         = data,
    M_s          = 1.0,
)
```

The mode determines which parameter slots a planet has (RV planets
have `K`, PM planets have `b`, `rr`, AS planets have `inc`/`Omega`, RM
planets add `lambda`, …). See [Priors → Naming convention](priors.md#naming-convention)
for the per-mode parameter list, and the `PlanetBlock` subtypes
(`RVOnlyBlock`, `PMOnlyBlock`, `RVPMBlock`, `RVASBlock`,
`RVPMASBlock`) in `src/model.jl:316`–`434` for the exact slot order.

!!! warning "Astrometry requires RV"
    `_make_block` (`src/model.jl:781`) rejects pure-AS and PM+AS
    planets — every astrometric planet must also carry `:RV`. RM and
    TTV both require `:PM` (you need a transit window / transit
    observations), enforced by the constant definitions.

### run_job equivalent

In the `model.planet_modes` array (`src/runner.jl:690`), pass the
constant **names** as strings. The schema validator
(`src/runner.jl:180`) recognises exactly:

```
RV_ONLY, PM_ONLY, RVPM, RVAS, RVPMAS,
RVPM_RM, RVPMAS_RM, RVPM_RM_R, RVPMAS_RM_R,
RVPM_TTV, RVPMAS_TTV, RVPM_RM_TTV, PM_TTV,
RVPM_TTV_NB, RVPMAS_TTV_NB, PM_TTV_NB
```

```json
"model": { "max_kplanet": 2, "planet_modes": ["RVPM_RM", "RV_ONLY"] }
```

## `ParametrizationConfig`

```julia
ParametrizationConfig(;
    ew                = :sesinw,
    time              = :Mo,
    geom              = :b_rr,
    use_rho_s         = false,
    mass              = :K_driven,
    obs_prior         = false,
    marginalize_gamma = false,
)
```

Defined in `src/model.jl:175`–`208`. The keyword constructor validates
`ew`, `time`, `geom`, and `mass` and throws an `ArgumentError` on any
unknown value, so a typo fails fast at construction.

### `ew::Symbol` — eccentricity parametrisation

The two eccentricity slots (`e1`, `e2`) are named and decoded
according to this choice (`src/model.jl:717`–`726`, decoded in
`src/parameters.jl` `planet_e_w`; conversions in `src/orbit.jl`):

- `:sesinw` (default) — sample `(√e·sin ω, √e·cos ω)`, named
  `sesinw_kN` / `secosw_kN`. Implicit prior on `e` is `U(0, 1)`; on `ω`
  uniform. The recommended default (Eastman+ 2013 geometry; Peña &
  Jenkins convention). Conversion `sesinw_to_ew` (`src/orbit.jl:108`).
- `:esinw` — sample `(e·sin ω, e·cos ω)`, named `esinw_kN` /
  `ecosw_kN`. Implicit prior `e ∝ 1` (linear) → biased toward higher
  `e`. Conversion `esinw_to_ew` (`src/orbit.jl:120`).
- `:ew` — sample `e` and `ω` directly, named `ecc_kN` / `w_kN`. Use
  when you want an explicit prior on `e` (e.g. `BetaPrior(0.867, 3.03)`
  for Kipping 2013, or a `NormalPrior` from an external constraint).
  Poorly conditioned at low `e` (ω unconstrained), so prefer `:sesinw`
  unless you specifically need a named `e` prior.

`true_anomaly` (`src/orbit.jl:31`) and the conversions clamp `e` into
`[0, 0.9999]` defensively because the unconstrained sampler can
transiently propose `s²+c² > 1`.

!!! note "External `e` prior"
    You don't have to switch to `:ew` to impose an `e` prior. The
    `external_priors` hook (see below) carries an `ExternalPrior(:ecc, …)`
    that is evaluated on the **derived** eccentricity regardless of the
    `ew` parametrisation — keep the well-conditioned `:sesinw`
    geometry and still get a `BetaPrior` on `e`.

### `time::Symbol` — orbital time anchor

The single time slot (`t`) is named and decoded per this choice
(`src/model.jl:729`; conversions `src/orbit.jl:160`–`208`):

- `:Mo` (default in the Julia API) — sample mean anomaly `M₀` at the
  reference epoch `t_ref` (= `data.t_ref`, the data median). Best
  conditioning for poorly-constrained periods (mid-baseline anchor
  reduces P–Tp covariance). `mo_to_tp` (`src/orbit.jl:165`).
- `:Tp` — sample time of periastron passage. Natural for visual orbits
  where Tp has direct geometric meaning. `tp_to_mo` (`src/orbit.jl:174`).
- `:Tc` — sample time of transit centre (inferior conjunction). Right
  for `RVPM`/`PM_ONLY`/transit fits: the transit ephemeris constrains
  `Tc` directly, and `Tp` becomes derived via `tc_to_tp`
  (`src/orbit.jl:184`, transit at `f = π/2 − ω`). Required for the
  `ttv_oc` plot (it reads `Tc_kN` by name).

### `geom::Symbol` — transit geometry parametrisation

The two transit-geometry slots (present only for PM planets,
`src/model.jl:732`) are named and decoded per this choice:

- `:b_rr` (default) — `b` (impact parameter) and `rr = R_p/R_⋆`, named
  `b_kN` / `rr_kN`. Standard.
- `:r1r2` — Espinoza (2018) two-parameter form, named `r1_kN` /
  `r2_kN`. `(r1, r2)` map the unit square `[0,1]²` onto the physically
  allowed `(b, rr)` region with uniform density, explicitly excluding
  the `b > 1 + rr` prohibited region from the prior box. Cleaner for
  transit fits where `b` is near 1 (grazing). Decoded in
  `planet_b_rr` (`src/parameters.jl:360`); the RM path converts
  `(r1, r2) → (b, rr)` internally (`src/rm.jl:193`).

### `use_rho_s::Bool` — stellar-density parametrisation

- `false` (default) — each transiting planet's `a/R_⋆` is derived from
  `P + M_⋆ + R_⋆` via Kepler III (so `M_s` and `R_s` must be set on
  `Params`).
- `true` — a single **shared** stellar density `rho_s` slot is sampled
  (`src/model.jl:1006`) and `a/R_⋆` is derived from `rho_s + P` for
  every transiting planet (`rho_s_to_a_Rs`). Use for multi-planet
  transiting systems: `rho_s` is one physical quantity shared across
  the system, and sampling it directly trades N per-planet
  density-equivalent parameters for one. Standard for TESS-style
  transit fits ([Vines+ 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.2627V/abstract)
  Table 7 convention). The RM `a/R_⋆` also prefers the `rho_s` path
  when present (`src/rm.jl:204`).

```julia
parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr, use_rho_s = true)
priors["rho_s"] = NormalPrior(2.241, 0.479, 0.1, 10.0)   # ARIADNE-derived
```

### `mass::Symbol` — mass / first-two-slot parametrisation

Controls how the first one or two planet slots are named and what they
mean (`src/model.jl:700`–`714`; mass-function machinery in
`src/parameters.jl`):

- `:K_driven` (default) — sample RV `K`; derive `M_sec` from the mass
  function. Right for RV-only and `RVPM` (the Keplerian reflex is what
  the data constrain).
- `:M_sec_driven` — sample `M_sec` directly; derive `K`. Recommended
  for joint RV + astrometry — the astrometric channel constrains
  `(M_sec, inc, Omega, a)` more directly than `(K, P, e, ω)`.
- `:a_driven` — sample `(a, M_sec)`; derive `P` (Kepler III) and `K`
  (mass function). orvara's convention; best for visual-orbit
  companions where `a` is the direct geometric quantity.

The first slot is named `P_kN` (days) under `:K_driven`/`:M_sec_driven`
and `a_kN` (AU) under `:a_driven`. The second RV slot is named `K_kN`
(m/s) under `:K_driven` and `M_sec_kN` (M_⊙) under the other two.

| mode | `block.P` slot | `block.K` slot |
|---|---|---|
| `:K_driven` | `P` (days) | `K` (m/s) |
| `:M_sec_driven` | `P` (days) | `M_sec` (M_⊙) |
| `:a_driven` | `a` (AU) | `M_sec` (M_⊙) |

Accessors `planet_P`, `planet_K`, `planet_M_sec`, `planet_a` decode
all three modes and derive missing quantities on the fly. All are
ForwardDiff-clean.

### `obs_prior::Bool` — O'Neil 2019 observation-based prior

When `true`, applies the O'Neil 2019 (AJ 158, 4) observation-based
prior on `(P, e, Tp)` for direct-imaging astrometric orbits. Encodes
the geometric truth that long-period orbits with poorly-sampled phase
coverage have correlated `(P, e, Tp)` posteriors and gives the sampler
a prior that respects that geometry. Only relevant for planets carrying
`:AS`.

### `marginalize_gamma::Bool` — analytic γ marginalisation

(`src/model.jl:182`–`190`.) When `true`, the per-instrument RV systemic
offset `γ` is **analytically marginalised** (orvara's approach). Because
`γ` enters the RV model linearly (`pred = γ + Keplerian + trend`), the
likelihood is Gaussian in `γ` and each instrument's `γ` integrates in
closed form at fixed jitter `σ`. The `gamma_*` slots are then **frozen
out** of the sampled set (fewer free dimensions, cleaner mixing and
evidence) and the RV likelihood uses the γ-marginalised path
(`_rv_ll_gamma_marginalized`).

- White-noise RV only — it does **not** compose with a covariance
  (GP/celerite) RV model.
- Most useful for short-arc RV + astrometry, where the Keplerian reflex
  over the RV baseline is a ~km/s offset that `γ` must absorb and
  marginalising it removes a stiff nuisance direction.

### run_job coverage of `ParametrizationConfig`

The `model.parametrization` block in run_job (`src/runner.jl:708`–`715`)
maps only these keys, with these accepted string values:

| run_job key | accepted strings | maps to | default |
|---|---|---|---|
| `mass` | `K_driven`, `M_sec_driven`, `a_driven` | `:K_driven` / `:M_sec_driven` / `:a_driven` | `K_driven` |
| `time` | `Tp`, `Tc`, `Mo` | `:Tp` / `:Tc` / `:Mo` | **`Tp`** |
| `ew` | `sesinw`, `ew` | `:sesinw` / `:ew` | `sesinw` |
| `geom` | `b_rr` | `:b_rr` | `b_rr` |
| `marginalize_gamma` | `true` / `false` | bool | `false` |

```json
"model": {
  "max_kplanet": 1,
  "planet_modes": ["RVPM"],
  "parametrization": {
    "mass": "K_driven", "time": "Tc", "ew": "sesinw",
    "geom": "b_rr", "marginalize_gamma": false
  }
}
```

!!! warning "run_job gaps vs the Julia API"
    Several Julia-API options are **not** reachable through run_job
    today (`src/runner.jl:709`):
    - `use_rho_s` and `obs_prior` are never read from the config — set
      them by constructing `Params`/`ParametrizationConfig` directly.
    - `ew = :esinw` and `geom = :r1r2` are valid in Julia but **absent
      from the run_job string maps** (`_EW_PARAM`, `_GEOM_PARAM`,
      `src/runner.jl:687`–`688`).
    - The run_job `time` default is `Tp`, whereas the Julia
      `ParametrizationConfig` default is `:Mo`.
    - The schema validator advertises `geom = "b_r"`
      (`src/runner.jl:191`) and the map yields `:b_r`, but
      `ParametrizationConfig` only accepts `:b_rr`/`:r1r2` — passing
      `geom: "b_r"` therefore throws at model construction. Use
      `b_rr`.

## Rossiter–McLaughlin parameters

Adding `:RM` (Hirano+ 2011) or `:RM_R` (Cegla+ 2016 "Reloaded") to a
planet's data sources turns on the in-transit RV anomaly model
(`src/rm.jl`). Both require `:RV` and `:PM` (a transit window plus
in-transit RV epochs).

Slots added to the layout:

- **`v_sin_i_star`** — system-level stellar projected rotation
  `V·sin(i_*)`, in **m/s**, shared across every RM planet (added once,
  `src/model.jl:1035`). Default prior `LogUniformPrior(500, 100_000)`
  m/s (`src/default_priors.jl:299`); override with a `NormalPrior` when
  spectroscopic `v sin i` is measured. Read via `system_vsini`
  (`src/parameters.jl:541`).
- **`lambda_kN`** — per-planet sky-projected obliquity `λ`, in
  **radians** (`src/model.jl:756`). Default prior `UniformPrior(-π, π)`
  (`src/default_priors.jl:304`); tighten to `NormalPrior(0, π/8)` for
  aligned systems. Read via `planet_lambda` (`src/parameters.jl:554`).

The leading-order Hirano model is `ΔRV_RM(t) = −Δflux(t) · v_p(t)` with
`v_p = V·sin(i_*)·(x_sky cos λ − y_sky sin λ)` (`rm_signal`,
`src/rm.jl:65`). The `:RM_R` Reloaded variant
(`rm_reloaded_signal`, `src/rm.jl:289`) instead integrates the
intensity-weighted line-of-sight velocity over the planet's
sky-projected disk (21×21 sub-cells), more accurate at high `rr` /
high `b`, reducing to Hirano in the small-planet limit.

The RM term is now part of `rv_predictions` (`src/likelihood.jl:1370`),
so PPC, residuals, fit-health diagnostics and **all** RV plots are
RM-consistent — not just the likelihood. `a/R_⋆` for the RM geometry
comes from `rho_s` when `use_rho_s = true`, otherwise from
`M_s + R_s + P` (`src/rm.jl:204`); if neither is available the RM term
is skipped (predictions) or the fit fails loud (likelihood).

```julia
params = Params(;
    max_kplanet  = 1,
    planet_modes = [RVPM_RM],          # or RVPM_RM_R for the Reloaded model
    instruments  = ic,
    data         = data,
    M_s = 1.0, R_s = 1.0,
)
# priors v_sin_i_star (m/s) and lambda_k1 (rad) auto-populated; override e.g.:
priors["v_sin_i_star"] = NormalPrior(4500.0, 500.0, 500.0, 100_000.0)
priors["lambda_k1"]    = NormalPrior(0.0, π/8, -π, π)
```

Visualise the fit with the `rm_anomaly` plot (`plot_rm`,
`src/plotting/rm_plots.jl:51`) — see [Plotting](plotting.md).

## Activity-indicator data (for `ActivityGP` and indicator GPs)

The multivariate `ActivityGP` (Rajpaul+ 2015) and other indicator-GP
noise models need the activity-indicator time series **and their
1σ errors**. This is data wiring rather than a parametrisation choice,
but it is required to make the relevant `planet_modes`/`noise_models`
fit run, so it is documented here; the noise model itself lives in
[Noise models → `ActivityGP`](noise_models.md).

In run_job, declare the noise model in `noise_models`
(`src/runner.jl:787`, `_build_noise_models` `src/runner.jl:801`):

```json
"noise_models": [
  { "kind": "ActivityGP",
    "instruments": ["HARPS"],
    "kwargs": { "channels": ["bis", "fwhm"], "use_derivative": true } }
]
```

`channels` is symbolised for you (`src/runner.jl:824`); valid channels
are `bis`, `fwhm`, `logrhk` (no derivative term), `halpha`
(`src/noise/types.jl:280`–`303`). `ActivityGP` deliberately has **no**
kernel-amplitude parameter (unit-variance latent `G(t)`); all scale
lives in the per-channel couplings (`Vc/Vr`, `Bc/Br`, …).

Supply the indicator data alongside the RV data:

- **`values` RV block** — add `<name>` and `<name>_err` arrays next to
  `bjd`/`rv`/`rv_err`/`instrument` (`src/runner.jl:464`–`492`). A
  `<name>_err` key is treated as the error for `<name>` **only when
  `<name>` is also present**; a lone `<name>_err` is itself an
  indicator value:

  ```json
  "rv": { "values": {
    "bjd": [...], "rv": [...], "rv_err": [...], "instrument": [...],
    "bis": [...], "bis_err": [...],
    "fwhm": [...], "fwhm_err": [...]
  } }
  ```

- **`csv` RV block** — list the indicator columns in `indicator_cols`
  and provide a matching `<col>_err` column for each
  (`src/runner.jl:501`–`533`):

  ```json
  "rv": { "csv": "rvs.csv",
          "indicator_cols": ["bis", "fwhm"] }
  ```
  with `bis`, `bis_err`, `fwhm`, `fwhm_err` columns in `rvs.csv`.

!!! warning "ActivityGP requires indicator errors"
    `ActivityGP` (and the indicator GPs) need the per-epoch indicator
    uncertainties. Provide the `<name>_err` / `<col>_err` arrays — a
    missing error column means the indicator is not usable by the GP.

Diagnose the latent process with the `activity_gp_latent` /
`activity_gp_decomposition` plots (see [Plotting](plotting.md)).

## `InstrumentConfig`

```julia
InstrumentConfig(; rv = String[], pm = String[])
```

Instrument names (`src/model.jl:223`), used only to build parameter
names like `gamma_HARPS_PRE`. The likelihood operates on integer
indices (`rv_inst`, `phot_inst` in `Data`).

```julia
ic = InstrumentConfig(rv = ["HARPS_PRE", "HARPS_POST", "FEROS"],
                       pm = ["TESS"])
```

`n_rv_instruments(ic)` and `n_pm_instruments(ic)` return the
respective counts.

## `Params` — the full model description

`Params(...)` glues everything together:

```julia
Params(;
    max_kplanet,                                 # required
    planet_modes::Vector{PlanetDataSources},     # required
    instruments::InstrumentConfig,               # required
    data::Data,                                  # required
    M_s::Real,                                   # required for stability + derived a/R*
    R_s::Real = 1.0,                             # required for transits (when use_rho_s=false)
    parametrization::ParametrizationConfig = ParametrizationConfig(),
    priors::Dict{String, PriorSpec} = Dict(),    # overrides for auto-priors
    noise_models::Vector{NoiseModel} = NoiseModel[],
    trend_order::Int = 0,                        # RV γ trend: 0 = none, 1 = dvdt, 2 = +d²v/dt²
    phot_trend_order::Int = 0,                   # per-PM-instrument baseline polynomial order
    external_priors::Vector{ExternalPrior} = ExternalPrior[],
    sharing::Dict{Symbol, Vector{Vector{String}}} = Dict(),  # see below
    stability::Symbol = :amd,                    # :none, :amd, :gladman
)
```

Returns a `Params` value with two fields (`src/model.jl:672`):

- `params.config` — the immutable `ParamsConfig` (data sources,
  instruments, parametrisation, priors, stability, M_s/R_s, …)
- `params.layout` — the derived `ParamsLayout` (`names`,
  `name_to_idx`, `n_total`, `n_p_idx`, `unfrozen_idx`,
  `unfrozen_names`, `unfrozen_priors`, `frozen_idx`/`frozen_values`,
  `planet_blocks`, `systemic`, `packed_priors`)

Everything for the hot path is precomputed by `_build_layout`
(`src/model.jl:883`), which also **validates** that every layout slot
has a prior, rejects unknown prior keys (typo detection), enforces
globally-unique parameter names, and runs `validate_physical` on each
prior.

Useful accessors (`src/model.jl:1147`–`1169`):

```julia
n_total(params)       # total parameter count
n_unfrozen(params)    # only the sampled ones
n_frozen(params)      # FixedPrior parameters
param_index(params, "K_k1")   # index into the theta vector
```

### `trend_order` / `phot_trend_order` — global trends

- `trend_order` adds a long-term RV trend on the systemic velocity:
  `1` adds a `dvdt` slot (linear), `2` also adds `d2vdt2` (quadratic)
  (`src/model.jl:1012`–`1021`).
- `phot_trend_order` adds a per-PM-instrument photometric baseline
  polynomial `1 + offset + Σ c_p·xᵖ` in normalised time, sharing the
  `offset` grouping; coefficients are named `phot_cP_<label>` with a
  default `NormalPrior(0, 0.05, -0.5, 0.5)` (`src/model.jl:960`–`980`).

### `stability` — multi-planet stability prior

A hard prior rejecting dynamically unstable multi-planet configs.
One of `:none`, `:amd` (AMD criterion), or `:gladman` (Gladman 1993
Hill-stability), validated in `src/params_constructor.jl:69`. The
Julia `Params` default is `:amd` (`src/params_constructor.jl:55`); the
run_job default is `none` (`src/runner.jl:734`). A non-`:none` choice
requires a finite `M_s` for `max_kplanet ≥ 2`
(`src/params_constructor.jl:71`).

### `sharing` — parameter group sharing

For multi-pipeline datasets where the same physical instrument is
reduced by multiple pipelines (e.g. HARPS-DRS vs HARPS-TERRA), some
parameters should be shared. Pass a dict mapping a **category** to
groups of instrument names (`_instrument_groups`, `src/model.jl:835`):

```julia
sharing = Dict(
    :sigma => [["HARPS_DRS", "HARPS_TERRA"]],   # share RV jitter
    :ld    => [["TESS", "K2"]],                 # share limb darkening
)
```

Valid categories (`src/model.jl:602`): RV `:sigma`, `:gamma`; PM
`:pm_jitter`, `:pm_offset`, `:pm_dilution`; and `:ld` (the `q1`/`q2`
pair). Instruments listed in the same inner vector share one slot;
instruments in no group keep their own. Group labels join names with
`+` (`gamma_HARPS_DRS+HARPS_TERRA`).

GP/celerite hyperparameter sharing across instruments is handled by the
noise-model `instruments` field rather than this dict — see
[Noise models](noise_models.md).

### `external_priors` — priors on derived quantities

`ExternalPrior(quantity, prior, per_planet)` (`src/model.jl:531`)
applies a prior to a value **derived** from sampled parameters, not a
sampled slot itself. Supported quantities (`src/model.jl:537`):

- `:ecc` — eccentricity, `per_planet = true` (evaluated per active planet)
- `:rho_s` — stellar density, `per_planet = false` (global, once)

```julia
external_priors = [
    ExternalPrior(:ecc,   BetaPrior(0.867, 3.03), true),   # Kipping 2013, keeps :sesinw geometry
    ExternalPrior(:rho_s, NormalPrior(1.4, 0.1),  false),
]
```

## `NereusTarget`

The final wrapper bound to a likelihood:

```julia
target = NereusTarget(params, data; unconstrained = true)
```

`unconstrained = true` returns a `LogDensityProblems`-compatible target
on the unconstrained Bijectors space (for NUTS / Pathfinder / PT-warm /
PT-HMC), `false` returns a target on the bounded prior space (for
nested samplers / ensemble PT in bounded space). Pick the one matching
your sampler's expectations — every sampler doc page says which it
expects ([Samplers](samplers.md)).
