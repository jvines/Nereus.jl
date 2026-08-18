# Trans-dimensional sampling

Nereus supports trans-dimensional inference over **planet count**
(N\_p ∈ {0, 1, …, max\_kplanet}) and **noise-model configuration**
(which models are active). The same `TransDimConfig` controls both.

The deliverable of a trans-dim run is an **occupancy** posterior:
P(N\_p = k | D) over planet counts and P(model active | D) over each
toggleable noise model. Under a uniform model prior the occupancy *is*
the posterior model probability P(M | D) — see the
**Occupancy = P(M | D)** section below. This is model
**selection**, never model averaging.

## When to use it

| Goal | Use |
|---|---|
| Characterise a *known* planet | Fixed-dim run, **don't** use trans-dim |
| Detect planets in a fresh RV time series | Trans-dim over N\_p (`planets = true`) |
| Compare noise prescriptions (model selection across activity models) | Trans-dim over noise (`noise = true`) |
| Both | Both — but expect much longer runs and harder convergence |

### The four trans-dim samplers

`run_job` requires a top-level `transdim` block for exactly four
sampler names (`runner.jl:206`, `_SAMPLERS_REQUIRE_TD`):

| Sampler | `run_job` name | Engine | Evidence | Notes |
|---|---|---|---|---|
| `sample_transdim_ptemcee` | `transdim_ptemcee` | PT + affine-invariant ensemble, MoMS birth/death | TI⁺/SS⁺/H⁺ from PT ladder | Planets **and** noise toggling. The blind-recovery workhorse. |
| `sample_rjmcmc` | `rjmcmc` | Classic reversible-jump MCMC | none (`NaN`) | Single-chain or PT; full Jacobian bookkeeping. |
| `sample_moms` | `moms` | Mixtures of Mutually Singular distributions (van den Bergh+ 2026) | none (`NaN`) | Variable selection with no dimension-jumping. |
| `sample_moms_ns` | `moms_ns` | MoMS within nested sampling | nested-sampling log Z | Joint trans-dim posterior + evidence in one run. |

`sample_pt` and `sample_pt_warm` are *also* trans-dim capable: passing
`td=TransDimConfig(...)` routes them through the in-house RJMCMC-PT
engine (`backend=:nereus`, which `:auto` selects whenever `td !==
nothing`; `pt.jl:317-329`). The Pigeons backend (`backend=:pigeons`)
**cannot** do trans-dim and throws if given a `td`.

`sample_ptemcee` is **fixed-dim only** (no `td` keyword).

## `TransDimConfig`

```julia
TransDimConfig(;
    max_kplanet,                              # required, Int
    planets::Bool         = true,             # enable planet birth/death
    noise::Bool           = false,            # enable noise toggling
    toggleable::Vector{<:NoiseModel} = NoiseModel[],
    birth_strategies::Vector{<:BirthStrategy} = [PriorBirth(), InformedBirth()],
    birth_weights::Vector{Float64}           = [0.3, 0.7],
    transdim_fraction::Float64               = 0.2,
    noise_exclusion_groups::Vector{<:Vector{<:NoiseModel}} = Vector{NoiseModel}[],
)
```

Definition: `src/transdim/config.jl:116`.

### Field-by-field

- `max_kplanet` — upper bound on N\_p. The chain explores
  `{0, 1, …, max_kplanet}` of planets. Must match the `max_kplanet`
  passed to `Params`. Validated `≥ 0` (`config.jl:137`).
- `planets` — set `false` for **noise-only trans-dim** with N\_p fixed
  (e.g., HD 18599: characterise the known planet AND select across
  activity models). When `false`, every planet slot starts active and
  the birth/death machinery only toggles noise. Default `true`.
- `noise` — enable noise birth/death. Requires `toggleable` non-empty
  (else `ArgumentError`, `config.jl:145`).
- `toggleable` — subset of `params.noise_models` that the chain can
  birth/kill. Models in `params.noise_models` but **not** in
  `toggleable` are always-on (their hyperparameters stay live but the
  model is never deactivated).
- `birth_strategies` — strategies for planet births. See below. Must be
  the same length as `birth_weights` (`config.jl:138`).
- `birth_weights` — mixing weights over strategies. Must sum to 1
  (tolerance `1e-10`, `config.jl:141`).
- `transdim_fraction` — fraction of MCMC moves that are birth/death
  rather than within-model. Validated to `[0, 1]` (`config.jl:143`).
  **Default `0.2`** in the constructor; note `run_job`'s
  `_build_transdim` uses a different default of **`0.3`**
  (`runner.jl:1113`).
- `noise_exclusion_groups` — user-declared mutually-exclusive noise
  model sets. At most one model per group is active simultaneously.
  See the **`noise_exclusion_groups`** section below.

## Birth strategies

Concrete `BirthStrategy` subtypes (`src/transdim/config.jl:8-83`). All
five are accepted by `run_job` (`runner.jl:1076`, `_BIRTH_TYPES`).

### `PriorBirth()`

Draw new planet parameters from the prior. Baseline strategy. Low
acceptance on weak signals (prior support is huge; landing on a real
peak is rare), but always valid.

### `InformedBirth()`

Propose period from a Lomb-Scargle periodogram of the **residuals**,
other parameters from the prior. Each iteration recomputes the
periodogram against the current chain state's residuals, so the
proposal distribution adapts as planets are added or removed. Much
higher acceptance on real RV signals.

### `JointInformedBirth()`

Combined RV-Lomb-Scargle + photometry-BLS periodogram peaks. Joint
proposal mixture weighted by `max(LS_norm, BLS_norm)` so both
RV-only signals (transit-undetectable, e.g. WASP-47 c) and
transit-only signals (RV-undetectable, e.g. WASP-47 e) surface. BLS
peaks within ±3 % of `m·P/n` (m, n ∈ 1..5) for any active planet are
filtered as harmonics. Falls back to `InformedBirth` if photometry is
absent (`config.jl:27`).

### `DonorBirth()`

Clone a planet from another particle in the population (PT chain or
NS live point) with small mutation. Only available in samplers with a
population; silently falls back to `PriorBirth`/`InformedBirth` in
standalone single-chain RJMCMC (`config.jl:43`).

### `MoMSBirth(off_values, scales, slot_indices)` / `MoMSBirth(params::Params)`

Mixtures of Mutually Singular distributions (van den Bergh, Clyde,
Raftery, Marsman 2026; arXiv:2604.27791) flavour. Each inactive planet
sits at a designated off-location `β_k_off` in parameter space; the
*add* move proposes new params via a Gaussian random walk centred on
`β_k_off`; the *delete* move deterministically resets to `β_k_off`
(`config.jl:53`). The mutable struct carries `off_values`, `scales`,
and `slot_indices` (the unfrozen-layout indices owned by each planet
block).

Mathematically equivalent to RJMCMC under an identity dimension map,
but with **no auxiliary variables and no Jacobian** — bookkeeping
reduces to standard fixed-dim Metropolis-Hastings. This is the native
move used by `sample_moms`, `sample_moms_ns`, and
`sample_transdim_ptemcee`.

Convenience constructor:

```julia
strategy = MoMSBirth(params; init_scale = 0.3, rng = MersenneTwister(42))
```

builds `off_values` (prior mid-range), `scales` (prior-width-derived),
and `slot_indices` for every planet slot automatically. The scales
are adapted during warmup toward a 23–44 % acceptance band (default
target `0.234`).

### Mixing strategies

```julia
TransDimConfig(;
    max_kplanet      = 3,
    birth_strategies = [PriorBirth(), InformedBirth(), DonorBirth()],
    birth_weights    = [0.1, 0.6, 0.3],
)
```

Each birth move samples a strategy according to `birth_weights`; the
death move always uses the inverse-of-birth (no separate kill
strategy), preserving detailed balance.

## `noise_exclusion_groups`

User-declared mutually-exclusive noise model sets. At most one model
per group active at any time. The proposal machinery checks the
constraint at birth time, and walkers are rejection-resampled at
init so no walker starts in a forbidden config
(`transdim_ptemcee.jl:319-333`).

The hard typing-level exclusion (`SequentialNoise` AR/MA vs
`CovarianceNoise` GP on the same channel) is **always** applied
regardless — it is a mathematical constraint, not a modelling choice.
AR + MA stay composable (ARMA), so they never belong in an exclusion
group. Exclusion groups add a second, user-controlled layer.

```julia
act    = ActivityDecorrelation(indicators = ["bisector_span"], derivative = true)
gp_rv  = CeleriteRotation(channel = :rv)
ar     = ARModel(order = 1, per_instrument = true)
ma     = MAModel(order = 1, per_instrument = true)

td = TransDimConfig(;
    max_kplanet            = 1,
    planets                = false,         # N_p fixed at 1
    noise                  = true,
    toggleable             = [ar, ma, act, gp_rv],
    noise_exclusion_groups = [
        [act, gp_rv],     # rotation tracker: BIS regression OR GP, not both
        [act, ma],        # short-lag activity: BIS regression OR MA, not both
    ],
)
```

Validation (`config.jl:151-160`):

- every member of every group must be in `toggleable`
- groups must have ≥ 2 members

Why it matters: BIS regression and a rotation GP both absorb the same
stellar-rotation modulation. Letting them coexist leaks planet signal
into the joint activity fit (the HD 18599 K-gap). Without the group,
trans-dim occupancy is *entrenchment-biased*, not P(M | D); the group
restores occupancy = evidence. See
[Noise models — exclusion groups](noise_models.md).

## `sample_transdim_ptemcee` — the workhorse

`sample_transdim_ptemcee` (`src/samplers/transdim_ptemcee.jl:166`)
extends `sample_ptemcee` with MoMS variable-selection moves. Each
walker carries its own `TransDimState` (planet activation bits + the
toggleable-noise bits). The within-step ordering per (walker, temp)
is: stretch ensemble move on continuous params → MoMS planet/noise
move (fires with probability `td.transdim_fraction`) → swap between
temperatures. Acceptance is tempered (`β · ΔlogL`) so hot chains
explore the model space freely.

Despite the stale `TODO` comment at the head of the file, **noise
toggling IS wired in** (`do_noise`, `transdim_ptemcee.jl:285`):
birth/death of noise models, an OLS-informed AD birth, a post-burn-in
DB-correct noise swap, and per-model post-birth refinement are all
present.

### Key keywords

| Keyword | Default | Meaning |
|---|---|---|
| `td::TransDimConfig` | required | trans-dim config; needs `planets` or `noise` true (`transdim_ptemcee.jl:202`) |
| `inclusion_prior::Real` | `0.5` | Bernoulli prior P(γ\_k = 1); must be in (0, 1) |
| `moms_init_scale::Real` | `1.0` | initial scale multiplier for the MoMS Gaussian-RW birth |
| `informed_birth_fraction::Real` | `0.0` | fraction of births using `JointInformedBirth` (BLS + LS, depth→radius→mass→K, BLS-t0 anchoring; auto-falls back to RV-only `InformedBirth` with no photometry). Forward/reverse use the same strategy family so detailed balance holds. **Critical for finding weak / arbitrary-period planets** — `0.0` gives blind RW births that rarely land. |
| `n_birth_tries::Int` | `1` | multi-try birth candidates per move |
| `n_birth_refine::Int` | `0` | post-birth RWM refinement steps (jointly refines jitter with the newborn) |
| `target_birth_accept::Real` | `0.234` | adaptation target for the MoMS scales |
| `n_temps::Int` | `5` | PT ladder size |
| `n_walkers::Int` | `100` | ensemble size (bumped to `≥ 2·n_dim + 2`, made even) |
| `n_steps` / `n_burnin` | `2000` / `1000` | post-burn-in steps / burn-in |
| `beta_min::Real` | `1e-4` | hottest β; geometric ladder `β_i = beta_min^(i/(n_temps-1))` over `[beta_min, 1]` — adding temps **densifies the cold end** rather than pushing T\_max hotter |
| `betas` | `nothing` | explicit ladder (overrides `beta_min`) |
| `noise_swap` / `noise_swap_rate` | `true` / `0.5` | post-burn-in DB-correct within-exclusion-group noise swap |
| `adapt_ladder` | `false` | thermodynamic-length ladder re-grid (window `ladder_adapt_window`, ν₀ `ladder_adapt_ν0`, K `ladder_adapt_K`) |
| `seed` / `thin` / `show_progress` | `1` / `1` / `true` | |

The result `TransDimPTemceeResult` carries `chains`, `log_evidence`
(hybrid → TI⁺ → TI fallback), `evidence_report`, within/swap/transdim
acceptance, the β ladder, `n_evals`, plus **raw** per-planet
(`td_proposed`, `td_accepted`) and per-temp noise
(`noise_td_proposed`, `noise_td_accepted`) move counts — raw counts
because a rounded rate of `0.000` reads as "frozen" even while the
chain is transitioning (HD 18599 post-mortem,
`transdim_ptemcee.jl:43`).

## `Params` — must match `TransDimConfig`

For trans-dim runs, `Params` needs to be aware of the maximum
configuration:

```julia
params = Params(;
    max_kplanet     = 3,                              # match TransDimConfig
    planet_modes    = [RV_ONLY, RV_ONLY, RV_ONLY],    # one per slot
    instruments     = ic,
    data            = data,
    M_s             = 1.0,
    noise_models    = [ar, ma, act, gp_rv],           # all declared
    transdim_noise  = true,                           # noise toggles enabled
)
```

`transdim_noise = true` on `Params` allocates parameter slots for
**all** declared noise models even when they're inactive; the
trans-dim chain then toggles activation bits, not array shape. This
gives a fixed-shape parameter vector — `TransDimState` flips bits
rather than resizing the vector (`src/transdim/state.jl:1-30`),
preserving ForwardDiff compatibility, pre-computed indices, and layout
stability.

## Occupancy = P(M | D)

A trans-dim chain spends a fraction of its samples in each
configuration. That fraction is the (model-prior-weighted) posterior
model probability:

```math
P(\mathrm{N_p} = k \mid D) \;=\; \frac{1}{N}\sum_{s} \mathbb{1}[\,n_{\mathrm{planets}}^{(s)} = k\,]
```

and likewise P(model\_i active | D) is the fraction of samples with
`noise_active_i == 1`. Under a uniform model prior this *is* P(M | D);
for a non-uniform prior reweight with `bayes_factors(...;
model_prior = k -> p(k))`. The same identity is what the validation
suite checks: occupancy must agree with an independently computed log Z
to ≲ 0.2 nats once the ladder is dense and recovery aids are on.

## Reading trans-dim output

The chain gets these extra columns
(`transdim_ptemcee.jl:1428-1434`):

- `:n_planets` — number of active planets per sample.
- `:planet_active_<k>` — one boolean column per planet slot
  `k ∈ 1..max_kplanet`.
- `:noise_active_<i>` — one boolean column per **toggleable** noise
  model, where `<i>` is the model's index into
  `params.config.noise_models` (so the column index can skip
  always-on models).

Standard summary helpers (`src/reporting.jl`):

```julia
print_transdim_summary(chains, params; td = td, M_s = 1.0)
save_transdim_summary(chains, params, "results/transdim_summary.txt";
                      starname = "HD18599", td = td, M_s = 1.0)
```

Note the argument order: `print_transdim_summary(chains, params; …)`
(`reporting.jl:298`) and `save_transdim_summary(chains, params, path;
starname, …)` (`reporting.jl:420`) — the path is positional, and
`starname` is required.

`print_transdim_summary` reports:

1. **Marginal N\_p posterior** — P(N\_p = k) for each k.
2. **Joint (N\_p, noise-config) posterior** — top configurations
   sorted by mass, e.g.

   ```
   1   AR(1)=0  MA(1)=0  Activity=0  GP-Rot=1   0.998   130826
   1   AR(1)=0  MA(1)=0  Activity=0  GP-Rot=0   0.002      244
   ```

3. **Marginal noise-model probabilities** — P(model\_i = on) globally.
4. **Conditional posteriors** — for each significant (N\_p,
   noise-config), the within-config posterior medians for fitted +
   derived parameters.

To slice the chain to a specific configuration:

```julia
gp_only_mask = vec(Bool.(Int.(Array(chains[:noise_active_4]))))
gp_only_K    = vec(Array(chains[:K_k1]))[gp_only_mask]
```

For Bayes factors / model probabilities computed from the
`:n_planets` column (`src/samplers/moms_ns.jl:720,746`):

```julia
model_probabilities(chains)         # → Dict(0 => 0.0, 1 => 0.998, 2 => 0.002)
bayes_factors(chains)               # → Dict of all pairwise BFs
bayes_factors(chains; reference=0)  # → Dict(k => BF(k vs 0))
```

## The `transdim_occupancy` plot

`plot_transdim_occupancy(chains, params; td=nothing, …)`
(`src/plotting/transdim_plots.jl:19`) renders a 2×2 trans-dim summary:

1. **[1,1] N\_p occupancy bars** — P(N\_p = k | D), k = 0 … max\_kplanet.
2. **[1,2] noise-model occupancy bars** — marginal P(model active | D)
   for each toggleable model (needs `td` or `noise_labels` for the
   labels; falls back to generic `model i`). Drawn only when there ARE
   toggleable models; the figure narrows to a single column otherwise.
3. **[2,1] N\_p trace** — n\_planets vs sample, to eyeball whether the
   sampler actually jumps dimensions (vs sticking).
4. **[2,2] noise-model raster** — rows = models, white = off, magenta =
   on, over the run.

Bars are coloured along the `:cool` colormap (`NEREUS_CMAP`) by their
value. No titles (info in axes/labels). It errors if the chain has no
`:n_planets` column. It can plot straight from a saved chain (pass
`noise_labels` or `max_kplanet` directly) without rebuilding
params/td.

## Driving trans-dim from `run_job`

The standard production entry point is the JSON/Dict-driven
`run_job`. A trans-dim job needs a top-level `transdim` block; the
validator enforces it for the four samplers above
(`runner.jl:336`). Birth strategies are named by string
(`runner.jl:208`).

```json
{
  "sampler": { "name": "transdim_ptemcee",
               "kwargs": { "n_temps": 10, "n_walkers": 120,
                           "n_steps": 4000, "n_burnin": 2000,
                           "informed_birth_fraction": 0.5 } },
  "model": {
    "max_kplanet": 3,
    "planet_modes": ["RV_ONLY", "RV_ONLY", "RV_ONLY"]
  },
  "transdim": {
    "max_kplanet": 3,
    "planets": true,
    "noise": false,
    "birth_strategies": ["PriorBirth", "JointInformedBirth"],
    "birth_weights": [0.3, 0.7],
    "transdim_fraction": 0.3
  },
  "output": { "plots": ["auto"] }
}
```

`_build_transdim` (`runner.jl:1105`) parses the block. `auto` plots add
`transdim_occupancy` automatically whenever the chain has a
`:n_planets` column (`runner.jl:1366`).

!!! note "Toggleable noise models can't come from JSON"
    `transdim.toggleable` and `noise_exclusion_groups` must hold
    `NoiseModel` *objects*, which JSON can't encode. `_coerce_noise_models`
    rejects a non-empty JSON `toggleable`/`noise_exclusion_groups` with an
    explicit error (`runner.jl:1091`). Noise-only trans-dim therefore needs
    the Julia API (build the `TransDimConfig` in-process), not a raw JSON job.
    Schema reference: `Nereus.jl/docs/JOB_CONFIG.md`.

## Rossiter-McLaughlin in joint fits

The `*_RM` planet modes add the in-transit Rossiter-McLaughlin RV
anomaly to a joint RV + transit fit
(`src/model.jl:98-106`). Both the leading-order Hirano+ 2011 analytical
form (`RM_SOURCE`, `src/rm.jl`) and the Cegla+ 2016 "reloaded",
intensity-weighted-integration variant (`RM_R_SOURCE`) share the same
two extra parameters:

| Mode | Sources | Engine |
|---|---|---|
| `RVPM_RM` | RV + PM + RM | Hirano+ 2011 |
| `RVPMAS_RM` | RV + PM + AS + RM | Hirano+ 2011 |
| `RVPM_RM_R` | RV + PM + RM-R | Cegla+ 2016 reloaded |
| `RVPMAS_RM_R` | RV + PM + AS + RM-R | Cegla+ 2016 reloaded |

RM requires both RV and PM (a transit window plus in-transit RV).
Parameters:

- `v_sin_i_star` — stellar projected rotation V·sin(i\_⋆) in **m/s**,
  one global slot (`model.jl:1037`); default prior
  `LogUniformPrior(500, 100_000)` m/s (`default_priors.jl:299`). Read
  with `system_vsini(theta)` (`parameters.jl:541`).
- `lambda_k<k>` — sky-projected stellar obliquity λ\_k of planet `k` in
  **radians** (`parameters.jl:555`); default prior
  `UniformPrior(-π, π)` (`default_priors.jl:304`). Read with
  `planet_lambda(theta, k)`.

The RM term is now folded into `rv_predictions`
(`src/likelihood.jl:1400`), so the predicted RV used by the PPC,
residuals, fit-health metrics, and every RV plot is RM-consistent
(no-op for non-RM fits). Use the dedicated `rm_anomaly` plot
(`plot_rm`, `src/plotting/rm_plots.jl:51`) to inspect the in-transit
anomaly; `run_job` emits it automatically for any RM-enabled fit
(`runner.jl:1383`).

## Newer plots relevant to trans-dim QC

Beyond `transdim_occupancy`, three plots help vet a trans-dim
solution (all dispatchable via `output.plots`, `runner.jl:211`):

- `rm_anomaly` — per-planet RM anomaly with predictive band
  (`plot_rm`); only for `*_RM` fits.
- `transit_overlay` — per-transit QC gallery, the median transit model
  overlaid on each individual transit window
  (`plot_transit_overlay_fit`, `runner.jl:1282`).
- `rv_components` — RV model decomposition (Keplerian / activity /
  trend pieces), distinct from `rv_timeseries`
  (`plot_rv_components`, `runner.jl:1166`); auto-emitted when there are
  ≥ 2 planets or a smooth (GP/AGP) activity component.
- `ttv_oc` — O−C transit-timing diagram. Now renders for
  **single-planet** fits too: with no perturber it shows the data-only
  O−C (measured per-transit T\_c vs the linear ephemeris) by passing
  `planet_b_k = 0` (`runner.jl:1264-1272`).

## ActivityGP from `run_job`

`ActivityGP` (a latent activity GP shared across RV + indicator
channels) is now run_job-drivable as a `noise_models` entry:

```json
{ "noise_models": [
    { "kind": "ActivityGP",
      "instruments": ["HARPS", "FEROS"],
      "kwargs": { "channels": ["bis", "fwhm"], "use_derivative": true } }
] }
```

`channels` (default `[:bis, :fwhm]`) and `use_derivative` (default
`true`, the Aigrain+ 2012 FF′-style dG/dt coupling) map onto the
`ActivityGP` constructor (`src/noise/types.jl:345`). JSON strings are
symbolised before construction (`runner.jl:824`).

**ActivityGP needs the indicator errors.** Provide them via the data
block:

- In an `rv` `values` block, add `<name>` and `<name>_err` arrays
  (e.g. `bis`/`bis_err`, `fwhm`/`fwhm_err`). The
  `<name>_err`-paired-with-`<name>` convention splits values from
  errors (`runner.jl:480`, `_split_indicator_errs`).
- In a CSV `rv` block, set `indicator_cols` and provide a matching
  `<col>_err` column per indicator (`runner.jl:501,530`).

A lone `<name>_err` with no matching `<name>` is treated as an
indicator *value*, not an error.

## Practical workflow

A trans-dim run takes longer than fixed-dim because the chain has to
visit lower-evidence configurations to integrate over them. Rough
budgets on a 22-d posterior like HD 18599:

| Budget | What you get |
|---|---|
| `n_temps=8 × n_walkers≈100`, `n_steps≈2000` (~3 min) | Sanity check, top-config posterior |
| `n_temps=10 × n_walkers≈120`, `n_steps≈4000` (~6 min) | Convergent N\_p posterior |
| `n_temps=12+ × n_walkers≈150`, `n_steps≈8000` (~12 min) | Production posterior + calibrated occupancy/evidence |

Use **≥ 10 PT temps** for trustworthy evidence (the TI⁺/SS⁺/H⁺
estimators need a dense ladder), set `informed_birth_fraction > 0` to
find weak / arbitrary-period planets, and thread with
`JULIA_NUM_THREADS=8` (the per-thread workspaces are indexed by
`threadid()` under `@threads :static`). For the production
noise-model-selection recipe see the HD 18599 worked example in
[Worked examples](examples.md).
