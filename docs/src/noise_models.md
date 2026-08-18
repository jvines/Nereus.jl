# Noise models

*(Nereus v0.2.0)*

Nereus organises noise/activity models into three stages that
compose under well-defined rules:

| Stage | Abstract type | What it does | Compose? |
|---|---|---|---|
| 1 | `MeanModifier` | Additive correction to the mean RV (activity decorrelation, activity jitter) | Composes freely with Stage 2 and 3 |
| 2 | `SequentialNoise` | Sequential correlation in the residuals (AR, MA, ARMA) | Mutually exclusive with a global Stage 3 GP on the same channel |
| 3 | `CovarianceNoise` | Full covariance structure (Celerite GPs, `ActivityGP`) | One global per channel **or** multiple per-instrument with disjoint instrument sets |

`IndicatorFloor` is a special fourth kind (a bare `NoiseModel`, not in any
of the three stages) — it scores the **activity-indicator data block** so a
trans-dim selection that includes `ActivityGP` is well-posed (see below).

There are two ways to configure noise models.

**In-process (Julia API).** Pass a `Vector{NoiseModel}` to `Params`:

```julia
params = Params(;
    # ...
    noise_models = [
        ActivityDecorrelation(indicators = ["bisector_span"], derivative = true),
        ARModel(order = 1, per_instrument = true),
        MAModel(order = 1, per_instrument = true),
        CeleriteRotation(channel = :rv),
    ],
)
```

**Via `run_job` (JSON/Dict config).** Each entry is
`{kind, channel, instruments, kwargs}`:

```json
"noise_models": [
  { "kind": "CeleriteRotation", "channel": "rv", "instruments": [], "kwargs": {} },
  { "kind": "ActivityDecorrelation", "channel": "rv", "instruments": ["HARPS"],
    "kwargs": { "indicators": ["bisector_span"], "derivative": true } }
]
```

`run_job` recognises these `kind`s (`_NOISE_TYPES` in `src/runner.jl:779`):
`CeleriteRotation`, `CeleriteSHO`, `CeleriteRotationFM17`,
`ActivityDecorrelation`, `ARModel`, `MAModel`, `ActivityJitter`,
`ActivityGP`. The top-level `channel`/`instruments` are injected only into
constructors that accept them (`src/runner.jl:826-835`): the Celerite GPs
take both, `ARModel`/`MAModel` take `channel` only, `ActivityGP` takes
`channels` (note the plural — supply it in `kwargs`), and the
`MeanModifier`s take neither. `IndicatorFloor` is **not** in `_NOISE_TYPES`
and so is only constructible through the Julia API.

For trans-dim noise selection (turning models on/off during the run),
see [Trans-dimensional](transdim.md) — specifically
`TransDimConfig(noise=true, toggleable=[...])` and
`noise_exclusion_groups`. The `Params` kwarg `transdim_noise = true`
relaxes the construction-time Stage 2 / Stage 3 mutual-exclusion so both
can live in the layout as toggleable components.

## Stage 1 — `MeanModifier`

Mean-modifier models contribute an additive term to the predicted
mean RV (or photometric baseline). They are zero-cost to compose and
are always allowed under trans-dim unless a `noise_exclusion_groups`
constraint forbids them.

### `ActivityDecorrelation`

Linear regression of RV against one or more activity indicators
(bisector span, FWHM, S-index, Hα, log R'<sub>HK</sub>, CRX, …), with
an optional FF′-style time-derivative term (Dumusque+ 2012).

```julia
ActivityDecorrelation(;
    indicators::Vector{String},      # required
    per_instrument::Bool = true,
    derivative::Bool     = false,
)
```

- `indicators` — names matching keys in `data.indicators`.
- `per_instrument = true` (default) — each instrument gets its own
  coefficient `C_<ind>_<INST>`. Cross-instrument indicator zero points
  differ, so this is almost always the right choice.
- `derivative = true` — adds a `Cdot_<ind>_<INST>` coefficient on
  `∂indicator/∂t`, the FF′ term ([Aigrain, Pont & Zucker 2012](https://ui.adsabs.harvard.edu/abs/2012MNRAS.419.3147A/abstract);
  [Rajpaul+ 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.2269R/abstract)). Captures rotation-phase-shifted activity-RV coupling
  that a single coefficient on the instantaneous indicator misses.
  The per-instrument central-difference derivative `data.indicator_derivs`
  is auto-computed at `Data` construction; `NaN` at endpoints/gaps and
  non-finite indicator values are skipped in the sum
  (`src/noise/activity.jl:44,57`).

Parameter names (`src/noise/param_names.jl:11`):
- `C_<ind>_<INST>` per (indicator, instrument) — coefficient on `Ind(t)`.
  When `per_instrument = true`, instruments with no finite indicator
  values get no slot.
- `Cdot_<ind>_<INST>` per (indicator, instrument), iff `derivative = true`
  — coefficient on `∂Ind/∂t`.
- With `per_instrument = false`: bare `C_<ind>` / `Cdot_<ind>`.

run_job form: `{"kind": "ActivityDecorrelation", "kwargs":
{"indicators": ["bisector_span"], "per_instrument": true, "derivative":
false}}`. `channel`/`instruments` are ignored here (this is a
`MeanModifier`).

### `ActivityJitter`

Activity-dependent **jitter** ([Diaz+ 2016](https://ui.adsabs.harvard.edu/abs/2016A&A...585A.134D/abstract); [Feng+ 2016](https://ui.adsabs.harvard.edu/abs/2016MNRAS.461.2440F/abstract) Eq. 16). Replaces
the constant per-instrument jitter with a time-varying
`σ_J(t) = σ_base + σ_act · indicator(t)`, added in quadrature to the
measurement error (`src/noise/activity.jl:17`):

```julia
ActivityJitter(; indicator::String = "log_rhk")
```

Parameter names (per RV instrument, `src/noise/param_names.jl:78`):
`jit_base_<ind>_<INST>` and `jit_act_<ind>_<INST>`. The indicator name
is baked into both so that multiple `ActivityJitter` models targeting
different indicators on the same instrument don't collide.

When `ActivityJitter` is active for an instrument, the constant
per-instrument `σ_<INST>` jitter is **not** added — the base jitter
replaces it. Set that constant-jitter prior to a small fixed value (or
remove it) accordingly.

run_job form: `{"kind": "ActivityJitter", "kwargs": {"indicator":
"log_rhk"}}`.

## Stage 2 — `SequentialNoise`

Sequential AR/MA processes on RV (or photometry) residuals with
exponential decay ([Tuomi+ 2013](https://ui.adsabs.harvard.edu/abs/2013A&A...551A..79T/abstract)). Both act on the **Stage-1
residual** (the deterministic-model residual), snapshotting it before
the loop so the lag reads the pre-correction value — matching the
astroEMPEROR `moav`/`ar` convention (`src/noise/arma.jl:39,113`).
Evaluation is sequential O(n), no matrix inversion. Orders are fixed at
construction (not trans-dimensional — trans-dim toggles the whole
component on/off, not its order).

### `ARModel`

Autoregressive of order `q`, applied to the model predictions:
`pred[i] += Σ_j φ_j · exp(−Δt/α_j) · pred_orig[i−j]`. `φ` sets
correlation strength, `α` the decay timescale (days).

```julia
ARModel(;
    order::Int          = 1,
    per_instrument::Bool = false,   # confine correlation within instrument
    channel::Symbol      = :rv,     # :rv or :phot
)
```

Parameter names (`src/noise/param_names.jl:58`): `ar_phi_<j>_<INST>$s`,
`ar_alpha_<j>_<INST>$s` per (order, instrument) when
`per_instrument = true`; bare `ar_phi_<j>$s`, `ar_alpha_<j>$s` otherwise.
`$s` is the channel suffix — empty for `:rv`, `_phot` for `:phot`.

### `MAModel`

Moving-average of order `p`, applied to the residuals:
`res[i] −= Σ_j ω_j · exp(−Δt/β_j) · res_orig[i−j]`. `ω` controls
correlation strength, `β` the timescale (days).

```julia
MAModel(;
    order::Int          = 1,
    per_instrument::Bool = false,
    channel::Symbol      = :rv,
)
```

Parameter names (`src/noise/param_names.jl:38`): `ma_omega_<j>_<INST>$s`,
`ma_beta_<j>_<INST>$s` (per-instrument) or `ma_omega_<j>$s`,
`ma_beta_<j>$s` (global).

run_job form: `{"kind": "ARModel", "channel": "rv", "kwargs": {"order":
1, "per_instrument": true}}` (likewise `MAModel`). `instruments` is
ignored — AR/MA scope per-instrument via the `per_instrument` flag, not
an instrument list.

### Combining AR and MA → ARMA

AR + MA are explicitly composable in Nereus (real ARMA = AR + MA, not
"either/or"):

```julia
noise_models = [
    ARModel(order=1, per_instrument=true),
    MAModel(order=1, per_instrument=true),
]
```

The two are mathematically distinct (different lag structures) and are
never mutually exclusive at the typing layer. Each can be toggled
independently under trans-dim.

### When **not** to use AR/MA

If a *global* Stage 3 GP is active on the same channel, AR/MA are
blocked (the algebra doesn't compose cleanly — GP residuals are already
correlated through the kernel). `validate_noise_models`
(`src/noise/param_names.jl:266`) enforces this at construction (unless
`transdim=true`), and the trans-dim proposal layer enforces it at
runtime. A *restricted* (per-instrument) GP does **not** block AR/MA on
the same channel, because it only covers a subset of observations.

## Stage 3 — `CovarianceNoise`

GP kernels via the celerite ([Foreman-Mackey+ 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F/abstract), 2018)
semi-separable formulation — O(n) likelihood evaluation regardless of
data length (`celerite_loglike` in `src/noise/gp.jl:104`). The
multivariate `ActivityGP` is also a `CovarianceNoise` but is built
dense (O(n³)); it is documented separately below.

All Stage 3 models carry two scoping fields:

- `channel::Symbol` — `:rv` or `:phot` (which observation block).
- `instruments::Vector{String}` — empty (default) means **global** on
  the channel (acts on all instruments); non-empty restricts the GP to
  the listed instruments, with white noise covering the rest. Multiple
  per-instrument GPs with pairwise-disjoint `instruments` sets can
  coexist on the same channel (`src/noise/gp.jl:714`,
  `_eval_channel_likelihood`).

Parameter names get a suffix `$s` built by `_gp_suffix`
(`src/noise/param_names.jl:127`): empty for global `:rv`; `_phot` for a
global `:phot` GP; `_HARPS` / `_HARPS+FEROS` / `_phot_TESS` for
per-instrument scoping.

### `CeleriteSHO`

Simple Harmonic Oscillator kernel ([Foreman-Mackey+ 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F/abstract)).
Overdamped (`Q < 0.5`) → two real terms; underdamped → one complex term
(`sho_coefficients`, `src/noise/gp.jl:24`).

```julia
CeleriteSHO(; channel::Symbol = :rv, instruments::Vector{String} = String[])
```

Three **log-space** fitted parameters (`src/noise/param_names.jl:141`,
`src/noise/gp.jl:427`):

| Param | Meaning |
|---|---|
| `gp_log_S0$s` | log of the power-spectral-density normalisation `S0` |
| `gp_log_Q$s` | log of the quality factor `Q` |
| `gp_log_omega0$s` | log of the undamped angular frequency `ω0` |

(`S0 = exp(gp_log_S0)`, etc.) Use for broad-band correlated noise
without a specific rotation structure. Good first GP to try.

run_job form: `{"kind": "CeleriteSHO", "channel": "rv", "instruments":
[]}`.

### `CeleriteRotation`

celerite2 rotation kernel ([Foreman-Mackey 2018, RNAAS 2, 31](https://ui.adsabs.harvard.edu/abs/2018RNAAS...2...31F/abstract)), the
**physically motivated** rotation model. Two SHO terms at `P_rot` and
`P_rot/2` (fundamental + first harmonic), each with its own quality
factor (`rotation_coefficients`, `src/noise/gp.jl:71`).

```julia
CeleriteRotation(; channel::Symbol = :rv, instruments::Vector{String} = String[])
```

Five fitted parameters (`src/noise/param_names.jl:152`,
`src/noise/gp.jl:444`):

| Param | Meaning | Default prior |
|---|---|---|
| `gp_sigma$s`  | overall GP amplitude (m/s or normalised flux units) | `LogUniform(1e-4, σ_max)` |
| `gp_period$s` | primary rotation period (days) | `LogUniform(1, 365)` |
| `gp_Q0$s`     | quality factor of the secondary mode at P/2 | `LogUniform(0.1, 10)` |
| `gp_dQ$s`     | `Q1 − Q2`, difference in quality factors | `Uniform(0, 5)` |
| `gp_f$s`      | fractional amplitude of the secondary mode, `0 ≤ f ≤ 1` | `Uniform(0.01, 0.99)` |

(Priors from `src/default_priors.jl:505`.) Use when the star has clear
rotation modulation and you want each harmonic's coherence to vary
independently. Default for active stars.

run_job form: `{"kind": "CeleriteRotation", "channel": "rv",
"instruments": []}`.

### `CeleriteRotationFM17`

Original [Foreman-Mackey+ 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F/abstract) celerite rotation kernel, as used
in astroEMPEROR. One real exponential + one damped cosine sharing a
single decay timescale (`rotation_fm17_coefficients`,
`src/noise/gp.jl:53`):

```
k(τ) = a (1+f)/(2+f) exp(−τ/τ_decay) + a/(2+f) exp(−τ/τ_decay) cos(2π τ / P)
```

```julia
CeleriteRotationFM17(; channel::Symbol = :rv, instruments::Vector{String} = String[])
```

Four **log-space** fitted parameters (`src/noise/param_names.jl:146`,
`src/noise/gp.jl:462`):

| Param | Meaning |
|---|---|
| `gp_log_amp$s`       | log of overall amplitude `a` |
| `gp_log_timescale$s` | log of decay timescale `τ_decay` (days) |
| `gp_log_period$s`    | log of rotation period `P` (days) |
| `gp_log_factor$s`    | log of mixture factor `f ≥ 0` |

Mathematically distinct from `CeleriteRotation` — only **one** decay
timescale shared between both modes, no separate harmonic quality
factor. Use when comparing against published [Foreman-Mackey+ 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F/abstract) /
astroEMPEROR fits.

run_job form: `{"kind": "CeleriteRotationFM17", "channel": "rv",
"instruments": []}`.

### `ActivityGP` — multivariate-GP framework ([Rajpaul+ 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.2269R/abstract))

A single latent process `G(t)` (quasi-periodic kernel) drives RV and a
configurable list of activity indicators jointly via per-channel linear
combinations of `G(t)` and its time derivative `dG/dt`
(`src/noise/activity_gp.jl`):

    ΔRV(t)   = Vc · G(t) + Vr · dG/dt
    BIS(t)   = Bc · G(t) + Br · dG/dt        (when :bis ∈ channels)
    FWHM(t)  = Fc · G(t) + Fr · dG/dt        (when :fwhm ∈ channels)
    logR'HK  = Lc · G(t)                      (when :logrhk ∈ channels; no dG/dt)
    Hα(t)    = Hc · G(t) + Hr · dG/dt        (when :halpha ∈ channels)

The joint data covariance is assembled from four analytic kernel blocks
`k_GG`, `k_GdotG`, `k_dotGG`, `k_dotGdotG` (closed-form derivatives of
the quasi-periodic kernel `k(τ) = σ² exp(−τ²/2λ_e² − sin²(πτ/P)/2λ_p²)`,
`activity_kernel_blocks`, `src/noise/activity_gp.jl:50`) weighted by the
per-observation channel coefficients. This is the principled framework
for activity decorrelation on noisy K-dwarfs — it captures
rotation-phase-shifted activity-RV coupling that single-channel BIS
decorrelation (`ActivityDecorrelation`) and the FF′ derivative shortcut
([Aigrain+ 2012](https://ui.adsabs.harvard.edu/abs/2012MNRAS.419.3147A/abstract)) cannot reach, because both reduce to one regressor
at a time.

```julia
ActivityGP(;
    channels::Vector{Symbol}     = [:bis, :fwhm],  # which indicators to couple
    use_derivative::Bool         = true,            # global dG/dt switch
    instruments::Vector{String}  = String[],        # RV-instrument scoping
    marginalize_indicators::Bool = false,           # DIAGNOSTIC ONLY (see below)
    indicators_only::Bool        = false,           # score only the indicator block
)
```

Valid indicator channel symbols (`_ACTIVITY_GP_COEFFS`,
`src/noise/types.jl:354`): `:bis`, `:fwhm`, `:logrhk`, `:halpha`. `:rv`
is always an active channel implicitly and is **not** listed in
`channels`.

#### Fitted parameters

`G(t)` is **unit-variance** per Rajpaul+ 2015 — there is deliberately
**no kernel amplitude parameter** (a free amplitude is an exact
non-identifiability against the channel couplings; it destabilised log Z
by ~80 nats run-to-run on HD 18599). All scale lives in the couplings.
Names from `src/noise/param_names.jl:160`, priors from
`src/default_priors.jl:533`:

| Parameter | Meaning | Default prior |
|---|---|---|
| `gp_act_period$s`   | rotation period `P_rot` (days) | `LogUniform(1, 365)` |
| `gp_act_lambda_e$s` | exponential decay length scale (days) | `LogUniform(1, 1000)` |
| `gp_act_lambda_p$s` | periodic length scale (dimensionless) | `LogUniform(0.25, 10)` |
| `Vc$s`, `Vr$s`      | RV coefficients on `G` and `dG/dt` | `Uniform(±(5·rv_max+1))` |
| `Bc$s`, `Br$s`      | BIS coefficients (when `:bis` active) | `Uniform(±3·MAD)` |
| `Fc$s`, `Fr$s`      | FWHM coefficients (when `:fwhm` active) | `Uniform(±3·MAD)` |
| `Lc$s`              | logR'HK coefficient (when `:logrhk` active; no derivative) | `Uniform(±3·MAD)` |
| `Hc$s`, `Hr$s`      | Hα coefficients (when `:halpha` active) | `Uniform(±3·MAD)` |
| `gp_act_jit_<ch>$s` | per-indicator-channel jitter floor on Σ_II's diagonal | `LogUniform(0.05·med_err, 30·med_err)` |

Notes on the couplings:
- `use_derivative = false` drops all `*r` (derivative) coefficients from
  the layout — useful for matching the FF′ simplification under a
  like-for-like comparison.
- The derivative coefficients are sampled as **amplitudes** in the
  channel's data units; the likelihood divides by `std(Ġ) = sqrt(1/λ_e²
  + π²/(P²λ_p²))` at evaluation (`src/noise/gp.jl:600`). This avoids the
  unpenalised `(λ_e, λ_p, Vr)` ridge where `Var(Ġ) → 0` and the raw
  coefficient rails.
- The `λ_p` floor of `0.25` keeps the kernel out of the
  free-interpolator regime (at `λ_p ≈ 0.02` the `sin²` kernel supports
  hour-wide structure that aliases a planet).
- The per-channel `gp_act_jit_<ch>` floors the indicator block's
  diagonal — reported BIS/FWHM/index errors are routinely
  underestimated, and a near-singular `Σ_II` lets the conditional mean
  over-track the indicators.

#### Indicator data requirement

`ActivityGP` requires both the indicator **values** and their **1σ
errors** for every non-`:rv` channel. Errors are mandatory (the
`activity_gp_predict`/likelihood paths throw / return `nothing` without
`data.indicator_errs`, `src/noise/activity_gp.jl:614,637`).

- **Julia API:** `Data(; indicators = Dict("bis" => …, "fwhm" => …),
  indicator_errs = Dict("bis" => …, "fwhm" => …), …)` with string keys
  matching the channel symbols.
- **run_job, inline `values` RV block:** add `<name>` and `<name>_err`
  arrays alongside `bjd`/`rv`/`rv_err`/`instrument`. Any extra key is
  treated as an indicator value; a `<name>_err` key whose base `<name>`
  is also present becomes that indicator's error
  (`_split_indicator_errs`, `src/runner.jl:480`):

  ```json
  "rv": { "values": {
    "bjd": [...], "rv": [...], "rv_err": [...], "instrument": [...],
    "bis": [...], "bis_err": [...],
    "fwhm": [...], "fwhm_err": [...]
  }}
  ```

- **run_job, CSV RV block:** list the indicator columns in
  `indicator_cols`; a matching `<col>_err` column supplies the error
  (`src/runner.jl:524`):

  ```json
  "rv": { "csv": "rvs.csv", "indicator_cols": ["bis", "fwhm"] }
  ```
  with columns `bis`, `bis_err`, `fwhm`, `fwhm_err` present.

#### run_job form

`ActivityGP` is **run_job-drivable** (`_NOISE_TYPES`,
`src/runner.jl:787`). Supply `channels` as a kwarg (strings are
symbolised, `src/runner.jl:824`):

```json
{ "kind": "ActivityGP", "instruments": [],
  "kwargs": { "channels": ["bis", "fwhm"], "use_derivative": true } }
```

`Params(...)` auto-generates priors for every kernel hyperparameter and
channel coefficient, and `rv_log_likelihood` routes through the joint
Rajpaul path when an `ActivityGP` is active — no per-call wiring
required.

#### `marginalize_indicators` and `indicators_only`

`marginalize_indicators = true` switches the likelihood from the joint
`log p(RV, indicators | θ)` to the conditional Gaussian `log p(RV |
indicators, θ)`. It is **DIAGNOSTIC ONLY** — never use it as a
model-selection / log-Z objective (`src/noise/types.jl:316`). The
conditional drops the indicator marginal `log p(y_I | θ)`, the only term
anchoring the kernel + couplings to the indicator data; the model can
then buy unbounded conditional sharpness for free (couplings rail, `λ_p`
rails, the planet is absorbed into the conditional mean, "evidence"
inflates by hundreds of nats — HD 18599 post-mortem 2026-06-12).

For an honest log-Z comparison against `ActivityDecorrelation`, use the
**chain rule** instead:

    log Z_cond = log Z(joint run) − log Z(indicators_only run)

which equals `log p(y_R | y_I)` with hyperparameters anchored by
`p(θ | y_I)`. The second term is produced by an `ActivityGP` with
`indicators_only = true` (`src/noise/types.jl:342`): the likelihood
scores **only** the indicator block `log p(y_I | θ)` and the RV path
falls through the standard white/celerite machinery untouched. In
`indicators_only` mode the RV couplings `Vc`/`Vr` are not sampled
(`src/noise/param_names.jl:174`).

#### Latent-G recovery + RV decomposition

- [`activity_gp_predict(chains, params, data)`](@ref)
  (`src/noise/activity_gp.jl:517`) returns the posterior over the
  inferred activity process `G(t)` **and** its derivative `dG/dt` at a
  dense prediction grid (default 200 points spanning the RV baseline),
  conditioning the joint covariance per posterior draw. Returns
  `t_pred`, `G_samples`, `G_mean/G_lo/G_hi` (16/84 band),
  `dG_dt_*`, and `Vc_samples`/`Vr_samples`. Under `transdim_noise = true`
  it first filters to draws where the AGP is active **and** whose kernel
  hyperparameters fall inside their prior bounds (guards against
  hot-chain pollution).
- [`activity_gp_decompose_rv(chains, params, data)`](@ref)
  (`src/noise/activity_gp.jl:772`) multiplies that posterior by the
  per-draw `Vc`/`Vr` and returns the inferred RV activity contribution
  at each observation plus the activity-corrected residual
  `rv − ⟨activity⟩` with its inflated error.
- [`plot_activity_gp_latent`](@ref) and
  [`plot_activity_gp_decomposition`](@ref) draw the paper figures.
  `run_job` emits them automatically as `activity_gp_latent.png` and
  `activity_gp_decomposition.png` whenever an `ActivityGP` is configured
  (`src/runner.jl:1337,1349`).

#### Trans-dim head-to-head with `ActivityDecorrelation`

Declare both models, set `transdim_noise = true` on `Params`, and add
them to the same `noise_exclusion_groups` entry in `TransDimConfig`
(also listed in `toggleable`). The trans-dim sampler flips the active
mask between the two; the data picks the winner on log Z:

```julia
ad  = ActivityDecorrelation(indicators = ["bisector_span"])
agp = ActivityGP(channels = [:bis])
params = Params(...; noise_models = [ad, agp], transdim_noise = true)
td_cfg = TransDimConfig(; max_kplanet = K,
                         noise = true,
                         toggleable = NoiseModel[ad, agp],
                         noise_exclusion_groups = [[ad, agp]])
chains = sample_rjmcmc(target, data, td_cfg; ...)
```

Because the AGP scores the **joint** `p(RV, indicators)` while every
other noise model scores only `p(RV)`, an honest occupancy needs an
`IndicatorFloor` (next section) so both states score the *same* data.
The HD 18599 K-gap scenario is the canonical use case.

#### Cost

The joint covariance is built dense — `O(n_total³)` Cholesky per
likelihood call, where `n_total = n_rv + Σᵢ n_indᵢ`. Tractable up to
~500 total observations. A Woodbury / low-rank route
(`activity_gp_joint_logpdf_lowrank`, `src/noise/activity_gp.jl:827`) and
a block-factored builder (`activity_gp_covariance_blocked`,
`src/noise/activity_gp.jl:425`) reduce the dominant factorisation from
`(C·N)²` to `(2N)²` for the common same-epoch case. The celerite-block
O(n) route is a dead-end for the QP joint covariance (its kernel is not
a finite sum of damped exponentials, and the FM17 celerite form carries
a `|τ|` kink that makes `Var(dG/dt)` formally infinite — see the
warning at `src/noise/activity_gp.jl:75`).

### `IndicatorFloor`

Null model for the activity-indicator channels (`src/noise/types.jl:71`).
It scores the indicator data block so a trans-dim noise selection that
includes `ActivityGP` is **well-posed**: AGP scores the joint
`p(RV, indicators)` while every other noise model scores only `p(RV)`,
so without a floor the trans-dim chain compares *different datasets* and
its occupancy is not a model posterior. With the floor always active:

- AGP-**inactive** state scores `p(RV | model) · p(y_I | floor)`
- AGP-**active** state scores `p(RV, y_I | GP)`

— the **same** data either way, so the occupancy is a valid `P(M | D)`.
The floor self-gates: it skips any channel currently covered by an
active (joint) `ActivityGP`. It has no effect on any config without an
`ActivityGP` toggle.

```julia
IndicatorFloor(;
    channels::Vector{Symbol} = [:bis, :fwhm, :halpha, :logrhk],
    kernel::Symbol           = :white,
)
```

`kernel` sets the indicator model:

- `:white` — iid Gaussian about zero,
  `y_{I,c,i} ~ N(0, σ_floor_c² + err_i²)`. One param `ind_floor_<ch>`
  per channel (`src/noise/param_names.jl:101`).
- `:qp` — a per-channel quasi-periodic GP (the same kernel family AGP
  uses for its indicator block), with **shared** `ind_floor_period`,
  `ind_floor_lambda_e`, `ind_floor_lambda_p` and per-channel
  `ind_floor_<ch>_amp`, `ind_floor_<ch>_jit`
  (`src/noise/param_names.jl:91`).

!!! warning
    `:white` is a **strawman** null for activity indicators, which are
    temporally correlated. AGP models that correlation and the white
    floor cannot, so AGP banks the indicator-correlation evidence
    (HD 18599: ≈ +82 nats) for free — irrelevant to the planet science
    but it leaks into the AD↔AGP occupancy (drags AGP up to a spurious
    ~15%). `:qp` gives the floor the same correlation-capturing power so
    the indicator-evidence terms ≈ cancel and the occupancy reflects the
    RV-conditional comparison (AD wins). **Use `:qp` for any AD-vs-AGP
    selection on real, correlated indicators.**

`IndicatorFloor` is **not** in `_NOISE_TYPES` and is therefore not
constructible through `run_job` — add it via the Julia API.

### Per-instrument GP example

A single rotation GP shared across HARPS\_PRE + HARPS\_POST + FEROS
versus per-instrument GPs:

```julia
# Global rotation GP — shared hyperparameters across all RV instruments
noise_models = [CeleriteRotation(channel = :rv)]

# Per-instrument — each gets its own (σ, P_rot, Q0, dQ, f)
noise_models = [
    CeleriteRotation(channel = :rv, instruments = ["HARPS_PRE"]),
    CeleriteRotation(channel = :rv, instruments = ["HARPS_POST"]),
    CeleriteRotation(channel = :rv, instruments = ["FEROS"]),
]
```

The per-instrument form is useful when activity manifests differently
across spectrographs (different wavelength coverage → different
sensitivity to facular vs spot regions). Uncovered instruments fall
back to white noise.

## Composition rules

These are enforced automatically by `validate_noise_models`
(`src/noise/param_names.jl:214`) and the trans-dim proposal layer:

1. **Stage 1 + anything** — always allowed.
2. **AR + MA** — always allowed (real ARMA = AR + MA, not "either/or").
3. **Stage 2 (AR/MA) + global Stage 3 GP on same channel** — forbidden
   (unless `transdim_noise = true`).
4. **Stage 3 + Stage 3 on same channel** — at most one *global* GP
   (`instruments=[]`), **or** any number of *restricted* GPs with
   pairwise-disjoint `instruments` sets — never both, never overlapping.
   A global on `:rv` and a global on `:phot` coexist fine.
5. **`ActivityGP`** is handled by the Rajpaul routing in
   `rv_log_likelihood`; it is skipped by the celerite channel path
   (`src/noise/gp.jl:735`). Treat it as one global `CovarianceNoise` on
   `:rv`.

Under `transdim_noise = true`, rule 3's mutual exclusion is relaxed at
construction so both categories can live in the layout as toggleable
components; the trans-dim birth proposals enforce single-category
activation at runtime, and `noise_exclusion_groups` adds any further
user-declared physical exclusions.

## Trans-dim and `noise_exclusion_groups`

When you want trans-dim to **also** enforce *physical* exclusions beyond
the typing-layer rules — e.g., BIS regression and GP rotation both model
stellar rotation modulation, so allowing both inflates activity-related
fit power and leaks planet signal — declare an exclusion group:

```julia
act   = ActivityDecorrelation(indicators = ["bisector_span"])
gp_rv = CeleriteRotation(channel = :rv)

td = TransDimConfig(;
    max_kplanet            = 1,
    noise                  = true,
    toggleable             = NoiseModel[act, gp_rv],
    noise_exclusion_groups = [[act, gp_rv]],    # at most one of these on
)
```

Every model in every exclusion group must also be in `toggleable`
(`src/transdim/config.jl:148`). AR + MA stay composable (no group),
AR/MA + global GP still blocked by the Stage 2 vs Stage 3 typing rule,
only the BIS↔GP exclusion is added.

See [Trans-dimensional](transdim.md) for the full API.

## Related: the Rossiter-McLaughlin term

Not a noise model, but it changes the RV mean and therefore the
residuals every noise model sees. When a planet's `planet_modes`
includes RM, an in-transit RV anomaly is added to the RV prediction.

- Modes: `RVPM_RM`, `RVPMAS_RM` (Hirano leading-order),
  `RVPM_RM_R`, `RVPMAS_RM_R` (Reloaded RM, Cegla+ 2016), plus
  `RVPM_RM_TTV` (`src/model.jl:98`; run_job map at `src/runner.jl:693`).
- Physics: `ΔRV_RM(t) = −Δflux(t) · v_p(t)` with
  `v_p = V·sin(i_*)·(x_sky cos λ − y_sky sin λ)`
  ([Hirano+ 2011](https://ui.adsabs.harvard.edu/abs/2011ApJ...742...69H/abstract), `src/rm.jl`); the Reloaded variant integrates the
  intensity-weighted line-of-sight velocity over the planet's disk
  (`rm_reloaded_signal`, `src/rm.jl:289`).
- Parameters: a system-level `v_sin_i_star` (m/s,
  `LogUniform(500, 1e5)`) and a per-planet sky-projected obliquity
  `lambda_k<k>` (rad, `Uniform(−π, π)`); priors at
  `src/default_priors.jl:299`.
- The RM contribution is included in `rv_predictions`
  (`src/likelihood.jl:1400`), so PPC residuals, the RV time-series
  overlay and phase folds are all RM-consistent.
- Diagnostic plot: `rm_anomaly` (`plot_rm`, `src/plotting/rm_plots.jl`),
  emitted by `run_job` when any planet mode has RM.

## Related run_job plots and diagnostics

- `rv_components` (`plot_rv_components`, `src/plotting/rv_plots.jl:243`)
  — RV model decomposition into its pieces (Keplerian(s), GP/AGP
  activity, total). Distinct from `rv_timeseries`. Auto-emitted when
  there are ≥2 planets or any `CovarianceNoise`/`ActivityGP` is active
  (`src/runner.jl:1370`).
- `transit_overlay` (`plot_transit_overlay_fit`,
  `src/plotting/ttv_plots.jl:328`) — per-transit QC gallery.
- `ttv_oc` (`plot_ttv_oc`, `src/plotting/ttv_plots.jl:144`) — O-C
  transit-timing diagram. For a single-planet fit it renders the
  data-only O-C (no N-body envelope: `planet_b_k = 0` skips the
  perturber path, `src/plotting/ttv_plots.jl:185`).
- The PPC residual GLS periodogram is a *leftover-periodicity flag*, not
  a publication periodogram: its frequency grid is capped at 20 000 and
  it uses the analytic FAP (`fap_method = :analytic`) so it stays fast
  on long-baseline RV (`src/diagnostics/ppc.jl:278`). Explicit
  periodogram plots still use the full bootstrap path.
