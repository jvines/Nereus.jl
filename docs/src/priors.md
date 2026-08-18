# Priors

Every unfrozen parameter in Nereus has a `PriorSpec` — a distribution
plus hard bounds `[lo, hi]`. The bounds matter even for nominally
unbounded distributions: they truncate the prior and are used by
nested samplers (unit-cube quantile transform), MCMC samplers
(rejection at the wall), and parallel tempering (proposal scaling).

If you don't specify priors, Nereus **auto-generates** them from the
data bounds, following EMPEROR / Peña & Jenkins conventions
(see [Auto-priors](#auto-priors)).

This page is the definitive reference: every prior type, the
per-parameter naming scheme, the data-driven defaults (including the
Rossiter-McLaughlin block, ActivityGP, and the per-instrument noise
models), and how to override any prior from a `run_job` JSON/Dict
config.

## Built-in prior types

All constructors live in `src/priors.jl`, return a `PriorSpec`, and are
exported from Nereus. Internally every `PriorSpec` stores
`(dist, lo::Float64, hi::Float64)`; the math (`logpdf`, `quantile`,
`cdf`, `rand`) is delegated to Distributions.jl wherever possible. Only
`ModJeffreys`, `Sine`, and `Fixed` are implemented in-house.

### `UniformPrior(lo, hi)`

Uniform on `[lo, hi]`. Requires `lo < hi`.

```julia
priors["sesinw_k1"] = UniformPrior(-1.0, 1.0)
```

### `NormalPrior(μ, σ, lo, hi)` / `NormalPrior(μ, σ)`

Normal with mean `μ`, std `σ` (`σ > 0` required). The four-argument
form is **truncated** to `[lo, hi]` via `truncated(Normal(μ,σ), lo, hi)`.
The two-argument form returns a raw `Normal` with `(-Inf, Inf)` bounds —
valid for unconstrained quantities (systemic velocity, offsets), but the
sampler stack must then handle infinite endpoints (nested samplers clip
`u` away from 0/1 before `quantile`, since `quantile(Normal, 0) = -Inf`).

```julia
priors["P_k1"]  = NormalPrior(4.137, 1e-3, 4.13, 4.14)
priors["rho_s"] = NormalPrior(2.241, 0.479, 0.1, 10.0)
```

### `LogUniformPrior(lo, hi)`

Uniform in log-space on `[lo, hi]` (Jeffreys prior on positive-real
scales), density `∝ 1/x`. Requires `lo > 0` and `lo < hi`. Use for
strictly-positive scales (periods, amplitudes, jitter) where you want
prior mass per decade rather than per unit.

```julia
priors["P_k1"]      = LogUniformPrior(0.1, 200.0)
priors["gp_period"] = LogUniformPrior(1.0, 365.0)
```

### `ModJeffreysPrior(knee, upper)`

Modified Jeffreys (Gregory 2005), density `∝ 1/(x + knee)` on
`[0, upper]`. Requires `knee > 0` and `upper > knee`. Flat for
`x ≪ knee`, Jeffreys `1/x` for `x ≫ knee`. Ideal for parameters that
**can legitimately be zero** but otherwise want a 1/x-style prior: RV
jitter, activity amplitudes, photometric extra scatter.

```julia
priors["jitter_TESS"] = ModJeffreysPrior(1e-4, 1e-1)
```

### `BetaPrior(a, b; lo=0.0, hi=1.0)`

Beta(`a`, `b`) on `[0, 1]`, optionally affine-rescaled to `[lo, hi]`.
Requires `a > 0`, `b > 0`, `lo < hi`. The canonical use is the
Kipping (2013) eccentricity prior `Beta(0.867, 3.03)`.

```julia
priors["ecc_k1"] = BetaPrior(0.867, 3.03)
```

!!! note "`BetaPrior` from a `run_job` config"
    `run_job` splats the config `args` array **positionally** into the
    constructor (see [Overriding priors](#overriding-priors-in-a-run_job-config)).
    Because `lo`/`hi` are keyword arguments, a config-supplied
    `BetaPrior` can only pass `a` and `b` (so `lo=0, hi=1`). Rescaled
    Beta priors require the Julia API.

### `SinePrior()`

`p(i) = sin(i)/2` on `[0, π]` — equivalent to uniform-in-`cos(i)`, the
isotropic-orbit prior for inclination. Required for any astrometric /
direct-imaging fit where inclination is a sampled slot; without it the
chain escapes into face-on (`i ≈ 0` or `π`) modes where huge `M_sec`
× tiny `sin i` degenerately satisfies the RV mass function.

```julia
priors["inc_k1"] = SinePrior()
```

### `FixedPrior(value)`

Parameter held at a constant. Not sampled, not in the sampler vector,
contributes `0` to the log-prior. Useful for fixing a nuisance to a
literature value, or freezing a slot (`dilution`, `M_pri`, a
γ-marginalized `gamma`).

```julia
priors["P_k1"] = FixedPrior(4.1374685)
```

### Programmatic API

```julia
ps = UniformPrior(0.0, 50.0)
lo, hi = bounds(ps)        # (0.0, 50.0)
is_fixed(ps)               # false
fixed_value(ps)            # errors unless is_fixed(ps)
in_support(ps, x)          # ps.lo <= x <= ps.hi
prior_transform(u, ps)     # unit-cube u∈[0,1] → param (alias for quantile)
logpdf(ps, x)              # -Inf outside [lo,hi]; 0.0 for FixedPrior
```

Collection helpers used by the sampler stack: `logpdf_sum(priors, values)`
(short-circuits to `-Inf` on the first out-of-support value) and
`prior_transform!(dst, us, priors)` (in-place unit-cube transform).

Serialization (for NetCDF metadata / inter-process interchange):
`prior_to_dict(ps)` → a `Dict` with a `"type"` key plus hyperparameters
and `lo`/`hi`; `prior_from_dict(d)` reconstructs it. Recognised types:
`Fixed`, `Uniform`, `LogUniform`, `Normal`, `Beta`, `ModJeffreys`. An
affine-rescaled `BetaPrior(lo≠0 || hi≠1)` round-trips as `"Unknown"`.

## Naming convention

Per-parameter prior keys follow a predictable pattern. The names are
generated by the model layer from `(planet_modes, instruments,
noise_models, parametrization)`. You override any individual prior by
its key. The hot path never reads your `priors` dict directly — at
construction the priors are baked into `params.layout.unfrozen_priors`
(see [Auto-priors](#auto-priors)).

### Planet parameters

`<param>_k<planet_index>` for planets `k1`, `k2`, … The exact slots
depend on the `parametrization` (see [Parametrizations](parametrizations.md)).

| Parameter | Meaning | Default prior |
|---|---|---|
| `P_k1` | Orbital period (days) — `:K_driven` / `:M_sec_driven` | `LogUniformPrior(0.1, 3·baseline)` |
| `a_k1` | Semi-major axis (AU) — `:a_driven` (orvara) | `LogUniformPrior(0.5, 1000.0)` |
| `K_k1` | RV semi-amplitude (m/s) — `:K_driven` | `ModJeffreysPrior(min(100, K_upper/2), K_upper)`, `K_upper = max(3·rv_scatter, 200)` |
| `M_sec_k1` | Companion mass (M_sun) — `:M_sec_driven` / `:a_driven` | `LogUniformPrior(3e-6, 0.5)` (≈1 M⊕ → 0.5 M_sun) |
| `K_A_k1`, `K_B_k1` | SB2 binary primary/secondary amplitudes (m/s) — `BINARY*` modes | `ModJeffreysPrior(min(100, K_upper/2), K_upper)` each, `K_upper = max(3·rv_scatter, 1000)` |
| `sesinw_k1`, `secosw_k1` | √e·sin/cos ω (`ew = :sesinw`, default) | `UniformPrior(-1, 1)` |
| `esinw_k1`, `ecosw_k1` | e·sin/cos ω (`ew = :esinw`) | `UniformPrior(-1, 1)` |
| `ecc_k1`, `w_k1` | direct e, ω (`ew = :ew`) | `NormalPrior(0, 0.3, 0, 1)`, `UniformPrior(-π, π)` |
| `Mo_k1` | Mean anomaly at `t_ref` (`time = :Mo`, default) | `UniformPrior(0, 2π)` |
| `Tp_k1` | Time of periastron (`time = :Tp`) | `UniformPrior(t_min, t_max)` |
| `Tc_k1` | Time of transit centre (`time = :Tc`) | `UniformPrior(t_min, t_max)` |
| `rr_k1`, `b_k1` | radius ratio, impact param (`geom = :b_rr`) | `UniformPrior(0.001, 1)`, `UniformPrior(0, 1)` |
| `r1_k1`, `r2_k1` | Espinoza (2018) reparam (`geom = :b_r`) | `UniformPrior(0, 1)` each |
| `inc_k1` | Inclination — AS planet, not RVPM+AS | `SinePrior()` |
| `Omega_k1` | Longitude of ascending node — AS planet | `UniformPrior(0, 2π)` |
| `lambda_k1` | Sky-projected obliquity (rad) — RM planet | `UniformPrior(-π, π)` |

`baseline` is the longest data span (RV or photometry), floored at 100
days; `t_min`/`t_max` span all data; `rv_scatter` is the max
per-instrument RV std floored at 10 m/s (`_rv_activity_scatter`). For a
combined RVPM+AS planet (e.g. `RVPMAS`) inclination is *derived* from
the transit impact parameter and is **not** a sampled `inc` slot.

### Rossiter-McLaughlin (RM)

When any planet's `planet_mode` carries an RM source — `RVPM_RM`,
`RVPMAS_RM`, `RVPM_RM_R`, `RVPMAS_RM_R` (`*_R` = Reloaded RM, Cegla+
2016; the others = Hirano+ 2011) — two kinds of parameter are added
(`default_priors`, `src/default_priors.jl`):

| Parameter | Meaning | Default prior |
|---|---|---|
| `v_sin_i_star` | system-level stellar projected rotation V·sin(i⋆), **m/s** | `LogUniformPrior(500.0, 100000.0)` |
| `lambda_k<k>` | per-planet sky-projected obliquity λ (**rad**) | `UniformPrior(-π, π)` |
| `f_light` | SB2 secondary light fraction L_B/(L_A+L_B), astrometric band (`BINARY` + astrometry only) | `UniformPrior(0.0, 0.5)` — constrain from the CCF flux ratio |

`v_sin_i_star` is a single system-level slot shared by all RM planets;
`lambda_k<k>` is added once per RM-enabled planet. Override
`v_sin_i_star` with a `NormalPrior` when you have a spectroscopic
V·sin(i⋆) (remember the units are m/s — 5 km/s ⇒ `5000.0`), and tighten
`lambda_k<k>` to e.g. `NormalPrior(0, π/8, -π, π)` for an aligned system.

The RM model (`src/rm.jl`) is the leading-order Hirano+ 2011 anomaly
`ΔRV = −Δflux·V·sin(i⋆)·x_p`, or, for the `*_R` modes, the Reloaded
intensity-weighted disk integral. RM requires transit geometry plus a
scale for `a/R⋆`: either `use_rho_s = true` (preferred) or both `M_s`
and `R_s` set on the model. The RM term is included in
`rv_predictions` (`src/likelihood.jl`), so PPC, residuals, and all RV
plots are RM-consistent. Render the in-transit anomaly with the
`rm_anomaly` plot.

### RV per-instrument

`<INST>` is the instrument label (or a shared-group label when
`sharing[:gamma]` etc. is set; see the `sharing` keyword to `Params`).

| Parameter | Meaning | Default |
|---|---|---|
| `gamma_<INST>` | RV zero-point offset | `UniformPrior(μ_inst ± spread)`, `spread = max(3·std_inst, 3·rv_scatter, 100)`; widened to **≥ ±3×10⁴ m/s for joint RV+astrometry** (see note); frozen to `FixedPrior(0.0)` under `marginalize_gamma` |
| `sigma_<INST>` | extra Gaussian RV jitter (m/s) | `NormalPrior(5.0, 5.0, 0.0, 50.0)` (Vines+ 2023 Table 7) |
| `dvdt` | linear RV trend (if `trend_order ≥ 1`) | `UniformPrior(-1, 1)` |
| `d2vdt2` | quadratic RV trend (if `trend_order = 2`) | `UniformPrior(-0.01, 0.01)` |

!!! note "Wide γ for joint RV + astrometry"
    For RV-only fits the systemic velocity sits near the data mean, so
    the default `γ` prior is the per-instrument `mean ± 3·scatter`. In a
    **joint RV + astrometry** fit (any of `relastrom`, `hgca`, `iad`,
    `gost`, `g23h`, `gaia_dr3` present), the Keplerian RV over a short
    observed arc of a long-period orbit is a large near-constant offset
    (can reach ~km/s) that `γ` must absorb on top of the relative
    zero-point. Nereus therefore widens the `γ` spread to at least
    ±3×10⁴ m/s when astrometry is present. (Without this a too-narrow
    `γ` forced HD 159062 B into a spurious short-period orbit; the wide
    prior recovers the orvara solution.) Override per instrument if your
    companion's reflex exceeds this.

### Photometry per-instrument

| Parameter | Meaning | Default |
|---|---|---|
| `offset_<INST>` | OOT baseline offset (near 0) | `NormalPrior(0.0, 0.005, -1.0, 1.0)` |
| `jitter_<INST>` | extra photometric scatter (flux units) | `ModJeffreysPrior(1e-4, 1e-1)` |
| `dilution_<INST>` | transit dilution / third light | `FixedPrior(0.0)` (override to free it) — **auto-frees to `UniformPrior(0, 0.9)`** when an SB2 `BINARY*` binary is present (the transit is diluted by the companion's light) |
| `q1_<INST>`, `q2_<INST>` | Kipping (2013) limb-darkening reparam | `UniformPrior(0, 1)` |
| `rho_s` | stellar density (when `use_rho_s = true`) | `LogUniformPrior(0.001, 100.0)` |

### Astrometry systemic (AS planets present)

| Parameter | Meaning | Default |
|---|---|---|
| `plx` | parallax (mas) | `NormalPrior(μ, σ, lo, hi)` from the HGCA parallax (±10σ, floored at 0.001) if `data.hgca` is set; else `LogUniformPrior(0.1, 1000.0)` |
| `M_pri` | primary mass (M_sun) | `FixedPrior(config.M_s)` if `M_s` set; else `LogUniformPrior(0.05, 5.0)`. Override with a `NormalPrior` to propagate stellar-mass uncertainty into the companion mass |

### TTV per-transit offsets

For a planet with a TTV source and `ttv_n_transits[k] = n`, slots
`ttv_k<k>_t<i>` (`i = 1..n`) default to `NormalPrior(0.0, 1.0, -10.0, 10.0)`
(days). Tighten with `NormalPrior(0, minutes_in_days, ±tight)` when you
have a-priori TTV-amplitude knowledge. The `ttv_oc` (observed−computed)
plot also renders for single-planet fits as a data-only O−C diagram
(`planet_b_k = 0`).

### Noise-model parameters

Each noise model contributes its own named slots (`_default_noise_priors!`,
`src/default_priors.jl`). Most carry a channel suffix — `_rv` / `_phot`
(`_channel_suffix`) — and a GP suffix when more than one of a kind
coexists (`_gp_suffix`). Per-instrument models append `_<INST>`.

| Model | Slots | Default |
|---|---|---|
| `ARModel(order = p, channel)` | `ar_phi_<j>_<INST><s>`, `ar_alpha_<j>_<INST><s>`, `j = 1..p` | φ: `UniformPrior(-1, 1)`; α (timescale): `LogUniformPrior(min_cadence, 365)` |
| `MAModel(order = q, channel)` | `ma_omega_<j>_<INST><s>`, `ma_beta_<j>_<INST><s>`, `j = 1..q` | ω: `UniformPrior(-1, 1)`; β (timescale): `LogUniformPrior(min_cadence, 365)` |
| `ActivityDecorrelation(indicators)` | `C_<ind>_<INST>` per indicator that has finite values for that instrument | `UniformPrior(-1, 1)` |
| `ActivityDecorrelation(..., derivative=true)` | adds `Cdot_<ind>_<INST>` (FF′, Aigrain+ 2012) | `UniformPrior(-1, 1)` |
| `ActivityJitter(indicator)` | `jit_base_<ind>_<INST>`, `jit_act_<ind>_<INST>` | base: `UniformPrior(0, 3·rv_max)`; act: `UniformPrior(-50, 50)` |
| `IndicatorFloor(channels; kernel=:white)` | `ind_floor_<ch>` | `LogUniformPrior(0.05·scale, 20·scale)`, `scale` = channel MAD |
| `IndicatorFloor(channels; kernel=:qp)` | `ind_floor_period`, `ind_floor_lambda_e`, `ind_floor_lambda_p`, `ind_floor_<ch>_amp`, `ind_floor_<ch>_jit` | LogUniform; same kernel scales as `ActivityGP` |
| `CeleriteSHO(channel)` | `gp_log_S0<s>`, `gp_log_Q<s>`, `gp_log_omega0<s>` | `UniformPrior(-5,15)`, `UniformPrior(-2,5)`, `UniformPrior(-3,5)` |
| `CeleriteRotation(channel)` | `gp_sigma<s>`, `gp_period<s>`, `gp_Q0<s>`, `gp_dQ<s>`, `gp_f<s>` | σ: `LogUniformPrior(1e-4, σ_max)` (σ_max = `1.0` phot, `3·rv_scatter` RV); period: `LogUniformPrior(1, 365)`; Q0: `LogUniformPrior(0.1, 10)`; dQ: `UniformPrior(0, 5)`; f: `UniformPrior(0.01, 0.99)` |
| `CeleriteRotationFM17(channel)` | `gp_log_amp<s>`, `gp_log_timescale<s>`, `gp_log_period<s>`, `gp_log_factor<s>` | log-amp upper = `log(0.1)` phot / `log(3·rv_scatter)` RV; the rest as in the source |
| `ActivityGP(channels, use_derivative, …)` | see below | see below |

`min_cadence` is the minimum positive sampling gap of the relevant
channel; `rv_max = max|RV|` (the systemic scale); `rv_scatter` is the
activity-scatter scale (`_rv_activity_scatter`), which amplitude priors
use instead of `rv_max` to keep their ceilings constraining on
absolute-RV data.

#### `ActivityGP` (Rajpaul+ 2015)

A single latent quasi-periodic GP `G(t)` drives the RV and the activity
indicators. There is deliberately **no** kernel amplitude (G is
unit-variance; scale lives in the channel couplings).

| Slot | Meaning | Default |
|---|---|---|
| `gp_act_period<s>` | rotation period (days) | `LogUniformPrior(1, 365)` |
| `gp_act_lambda_e<s>` | exponential-decay timescale λ_e (days) | `LogUniformPrior(1, 1000)` |
| `gp_act_lambda_p<s>` | periodic length λ_p (dimensionless) | `LogUniformPrior(0.25, 10)` |
| `Vc<s>` | RV coupling to G (only if RV channel) | `UniformPrior(±(5·rv_max + 1))` |
| `Vr<s>` | RV coupling to dG/dt (RV channel, `use_derivative`) | `UniformPrior(±(5·rv_max + 1))` |
| per-channel `cg`, `cd` | indicator couplings to G and dG/dt | `UniformPrior(±3·MAD_ch)` |
| `gp_act_jit_<ch><s>` | per-indicator jitter floor on Σ_II | `LogUniformPrior(0.05·med_err, 30·med_err)` |

The λ_p floor of `0.25` is load-bearing: at λ_p ≈ 0.02 the sin² kernel
becomes a free interpolator that can absorb a planet (measured on HD
18599). The derivative couplings are only added when `use_derivative =
true` and the channel has a defined dG/dt coefficient (`logRʹHK` has
none). The per-indicator jitter floor uses the reported indicator
errors when available, so `ActivityGP` **requires** indicator error
arrays for every modelled non-RV channel.

## Auto-priors

Pass `data` to `Params` and omit any prior — Nereus generates it from
the data bounds (`default_priors`):

```julia
params = Params(;
    max_kplanet  = 2,
    planet_modes = [RV_ONLY, RV_ONLY],
    instruments  = ic,
    data         = data,
    M_s          = 1.0,
)   # no priors arg → all auto
```

Auto-priors are computed once at construction; thereafter the priors
live in `params.layout.unfrozen_priors` (parallel to
`params.layout.unfrozen_names`). To inspect:

```julia
for (n, ps) in zip(params.layout.unfrozen_names, params.layout.unfrozen_priors)
    println(rpad(n, 25), "  ", ps)
end
```

Mix auto and manual — keys you supply override; everything else is
auto-generated:

```julia
params = Params(;
    max_kplanet  = 2,
    planet_modes = [RV_ONLY, RV_ONLY],
    instruments  = ic,
    data         = data,
    M_s          = 1.0,
    priors       = Dict("P_k1" => NormalPrior(4.137, 1e-3, 4.13, 4.14)),
)
```

## `ExternalPrior` — constraints on derived quantities

Some priors act on quantities that are *derived* from the sampled
parameters rather than sampled directly — eccentricity (when you sample
in `sesinw`/`secosw`) or stellar density. `ExternalPrior` attaches a
`PriorSpec` to such a quantity; its contribution is summed into the log
prior (`src/model.jl`):

```julia
ExternalPrior(quantity::Symbol, prior::PriorSpec, per_planet::Bool)
```

- `quantity` — one of `:ecc`, `:rho_s` (`_VALID_EXTERNAL_QUANTITIES`).
- `prior` — any `PriorSpec`.
- `per_planet` — `true` evaluates the prior on each planet's value
  (e.g. `:ecc`); `false` on the single global value (e.g. `:rho_s`).

```julia
params = Params(;
    # …
    external_priors = [
        ExternalPrior(:ecc,   BetaPrior(0.867, 3.03), true),   # Kipping 2013
        ExternalPrior(:rho_s, NormalPrior(1.4, 0.1),  false),  # ρ⋆ constraint
    ],
)
```

This is the right home for an external-data constraint on eccentricity
(on top of the implicit Uniform from `sesinw`/`secosw`) or an
asteroseismic / isochrone ρ⋆ prior. `external_priors` is a Julia-API
keyword to `Params`; it is **not** part of the `run_job` JSON schema.

## Overriding priors in a `run_job` config

`run_job` (the JSON/Dict entry point, `src/runner.jl`) accepts a
top-level `priors` object: a map from parameter name to
`{ "type": <constructor>, "args": [ … ] }`. `_build_priors` looks up the
constructor and **splats `args` positionally** into it:

```json
{
  "priors": {
    "P_k1":         { "type": "NormalPrior",      "args": [4.137, 1e-3, 4.13, 4.14] },
    "K_k1":         { "type": "ModJeffreysPrior", "args": [1.0, 200.0] },
    "ecc_k1":       { "type": "BetaPrior",        "args": [0.867, 3.03] },
    "inc_k1":       { "type": "SinePrior",        "args": [] },
    "gamma_HARPS":  { "type": "UniformPrior",     "args": [-50.0, 50.0] },
    "v_sin_i_star": { "type": "NormalPrior",      "args": [5000.0, 300.0, 0.0, 20000.0] },
    "lambda_k1":    { "type": "NormalPrior",      "args": [0.0, 0.3927, -3.14159, 3.14159] }
  }
}
```

Recognised `type` values (`_KNOWN_PRIORS`): `UniformPrior`,
`LogUniformPrior`, `ModJeffreysPrior`, `NormalPrior`, `FixedPrior`,
`SinePrior`, `BetaPrior`. The config validator rejects any other
`type` before the run starts; an empty `args` (`SinePrior`) is fine.
Because the splat is positional, `BetaPrior`'s `lo`/`hi` keywords are
not reachable from a config (see the note under
[`BetaPrior`](#betapriora-b-lo00-hi10)).

The parameter **key** is the exact slot name from the
[naming convention](#naming-convention) above (`P_k1`, `gamma_HARPS`,
`Cdot_bis_FEROS`, `gp_act_period`, …). Any key you don't list keeps its
[auto-prior](#auto-priors).

### Driving `ActivityGP` (and other activity models) from a config

Activity models are `run_job`-drivable via `noise_models`. The kwargs
mirror the Julia constructor:

```json
{
  "noise_models": [
    { "kind": "ActivityGP",
      "instruments": ["HARPS", "FEROS"],
      "kwargs": { "channels": ["bis", "fwhm"], "use_derivative": true } }
  ]
}
```

`channels` strings are symbolized for you; `instruments`/`channel` are
injected only into constructors that accept them. The indicator data
itself rides in the data block:

- **Inline RV block** — add `<name>` and `<name>_err` arrays in the rv
  `values` (e.g. `bis` + `bis_err`, `fwhm` + `fwhm_err`). A `<name>_err`
  column is treated as an error only when its base `<name>` also exists;
  a lone `<name>_err` is itself an indicator value.
- **CSV RV block** — list the indicator columns in `indicator_cols`, and
  add a matching `<col>_err` column for each.

`ActivityGP` requires the indicator errors (it floors Σ_II from them);
omit them and the model has nothing to anchor the per-channel jitter.

## Physical bounds and validation

Independently of Bayesian priors, Nereus enforces a global table of
hard physical bounds per parameter base name — `PHYSICAL_BOUNDS` in
`src/priors.jl` (e.g. `e ∈ [0,1]`, `b ∈ [0,2]` to allow grazing,
`P > 0`, `rho_s > 0`, ARMA `phi ∈ [-1,1]`). `validate_physical(name, ps)`
asserts that a user-supplied prior's bounds (or a `FixedPrior`'s value)
lie inside the physical range, throwing `ArgumentError` otherwise; it
strips the `rv-`/`pm-` mode prefix and the instrument/planet/order
suffix via `extract_base_name` to find the base key, and passes unknown
names through silently. These encode physics, not choices, and are not
configurable per fit.

For multi-planet **dynamical stability** (`stability = :amd` /
`:gladman`), see the `stability` keyword to `Params` — covered under
[Trans-dimensional](transdim.md). The same stability check gates both
fixed-dim and trans-dim transitions.
