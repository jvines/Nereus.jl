# Trans-dimensional sampling

Nereus supports trans-dimensional inference over **planet count**
(N\_p ∈ {0, 1, …, max\_kplanet}) and **noise-model configuration**
(which models are active). The same `TransDimConfig` controls both.

The deliverable of a trans-dim run is an **occupancy** posterior:
P(N\_p = k | D) over planet counts and P(model active | D) over each
toggleable noise model. The occupancy *is* the posterior model
probability P(M | D) under the run's prior on the models (the prior on k is
written out below, and recorded in `run_info`) — see the
**Occupancy = P(M | D)** section. This is model
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
sampler names (`runner.jl:316`, `_SAMPLERS_REQUIRE_TD`):

| Sampler | `run_job` name | Engine | Evidence | Notes |
|---|---|---|---|---|
| `sample_transdim_pt_emcee` | `transdim_pt_emcee` | PT + affine-invariant ensemble, MoMS birth/death | TI⁺/SS⁺/H⁺ from PT ladder | Planets **and** noise toggling. The blind-recovery workhorse. |
| `sample_rjmcmc` | `rjmcmc` | Classic reversible-jump MCMC | none (`NaN`) | Single-chain or PT; full Jacobian bookkeeping. |
| `sample_moms` | `moms` | Mixtures of Mutually Singular distributions (van den Bergh+ 2026) | none (`NaN`) | Variable selection with no dimension-jumping. |
| `sample_daedalus` | `daedalus` | MoMS within nested sampling | nested-sampling log Z | Joint trans-dim posterior + evidence in one run. |

`sample_pt` is *also* trans-dim capable: passing
`td=TransDimConfig(...)` routes it through the in-house RJMCMC-PT
engine, which is the only PT implementation `sample_pt` runs
(`pt.jl:195-223`). There is no `backend` keyword any more — the engine
name is the sole algorithm selector. The Pigeons path survives as the
unexported `_sample_pt_pigeons`, for cross-checks only: it is not an
engine, and it **cannot** do trans-dim — it throws if given a `td`
(`pt.jl:1106`).

`sample_pt`'s `init_strategy` (`:prior`, the default, or
`:pathfinder`) chooses how the replicas are *initialised*, not which
algorithm runs. `:pathfinder` seeds each replica from an independent
multi-run Pathfinder L-BFGS basin (`n_pathfinder_runs = 16`,
`n_pathfinder_draws = 0` → `max(2·n_chains, 200)`), which is what the
former `pt_warm` engine did; that engine is gone and this keyword
replaces it (`pt.jl:170-172`). The trans-dim machinery is unaffected
by the choice.

`sample_pt_emcee` is **fixed-dim only** (no `td` keyword), and
deliberately does **not** offer a `:pathfinder` init: Pathfinder's MVN
approximation is poor on sharp curved ridges (Pareto k ≫ 0.5) and
seeds every walker into one spurious basin, producing pristine-looking
R̂ / ESS at a wrong orbit (HD 159062: a = 34.5 against a true ≈ 58,
where prior-dispersed pt_emcee recovers a ≈ 57).

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
  passed to `Params`. Validated `≥ 0` (`config.jl:144`).
- `planets` — set `false` for **noise-only trans-dim** with N\_p fixed
  (e.g., HD 18599: characterise the known planet AND select across
  activity models). When `false`, every planet slot starts active and
  the birth/death machinery only toggles noise. Default `true`.
- `noise` — enable noise birth/death. Requires `toggleable` non-empty
  (else `ArgumentError`, `config.jl:154`).
- `toggleable` — subset of `params.noise_models` that the chain can
  birth/kill. Models in `params.noise_models` but **not** in
  `toggleable` are always-on (their hyperparameters stay live but the
  model is never deactivated).
- `birth_strategies` — strategies for planet births. See below. Must be
  the same length as `birth_weights` (`config.jl:145`).
- `birth_weights` — mixing weights over strategies. Must sum to 1
  (tolerance `1e-10`, `config.jl:148`).
- `transdim_fraction` — fraction of MCMC moves that are birth/death
  rather than within-model. Validated to `[0, 1]` (`config.jl:150`).
  **Default `0.2`** in the constructor; note `run_job`'s
  `_build_transdim` uses a different default of **`0.3`**
  (`runner.jl:1655`).
- `noise_exclusion_groups` — user-declared mutually-exclusive noise
  model sets. At most one model per group is active simultaneously.
  See the **`noise_exclusion_groups`** section below.

## Birth strategies

Concrete `BirthStrategy` subtypes (`src/transdim/config.jl:8-83`). All
five are accepted by `run_job` (`runner.jl:1608`, `_BIRTH_TYPES`).

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
absent (`config.jl:39`).

### `DonorBirth()`

Clone a planet from another particle in the population (PT chain or
NS live point) with small mutation. Only available in samplers with a
population; silently falls back to `PriorBirth`/`InformedBirth` in
standalone single-chain RJMCMC (`config.jl:48`).

### `MoMSBirth(off_values, scales, slot_indices)` / `MoMSBirth(params::Params)`

Mixtures of Mutually Singular distributions (van den Bergh, Clyde,
Raftery, Marsman 2026; arXiv:2604.27791) flavour. Each inactive planet
sits at a designated off-location `β_k_off` in parameter space; the
*add* move proposes new params via a Gaussian random walk centred on
`β_k_off`; the *delete* move deterministically resets to `β_k_off`
(`config.jl:58`). The mutable struct carries `off_values`, `scales`,
and `slot_indices` (the unfrozen-layout indices owned by each planet
block).

Mathematically equivalent to RJMCMC under an identity dimension map,
but with **no auxiliary variables and no Jacobian** — bookkeeping
reduces to standard fixed-dim Metropolis-Hastings. This is the native
move used by `sample_moms`, `sample_daedalus`, and
`sample_transdim_pt_emcee`.

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
(`transdim_pt_emcee.jl:686-700`).

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

Validation (`config.jl:157-169`):

- every member of every group must be in `toggleable`
- groups must have ≥ 2 members

Why it matters: BIS regression and a rotation GP both absorb the same
stellar-rotation modulation. Letting them coexist leaks planet signal
into the joint activity fit (the HD 18599 K-gap). Without the group,
trans-dim occupancy is *entrenchment-biased*, not P(M | D); the group
restores occupancy = evidence. See
[Noise models — exclusion groups](noise_models.md).

## `sample_transdim_pt_emcee` — the workhorse

`sample_transdim_pt_emcee` (`src/samplers/transdim_pt_emcee.jl:425`)
extends `sample_pt_emcee` with MoMS variable-selection moves. Each
walker carries its own `TransDimState` (planet activation bits + the
toggleable-noise bits). The within-step ordering per (walker, temp)
is: stretch ensemble move on continuous params → MoMS planet/noise
move (fires with probability `td.transdim_fraction`) → swap between
temperatures. Acceptance is tempered (`β · ΔlogL`) so hot chains
explore the model space freely.

Despite the stale `TODO` comment at the head of the file, **noise
toggling IS wired in** (`do_noise`, `transdim_pt_emcee.jl:652`):
birth/death of noise models, an OLS-informed AD birth, a post-burn-in
DB-correct noise swap, and per-model post-birth refinement are all
present.

### Key keywords

| Keyword | Default | Meaning |
|---|---|---|
| `td::TransDimConfig` | required | trans-dim config; needs `planets` or `noise` true (`transdim_pt_emcee.jl:511`) |
| `inclusion_prior::Real` | `0.5` | Bernoulli prior P(γ\_k = 1); must be in (0, 1) |
| `moms_init_scale::Real` | `1.0` | initial scale multiplier for the MoMS Gaussian-RW birth |
| `informed_birth_fraction::Real` | `0.5` | fraction of births using `JointInformedBirth` (BLS + LS, depth→radius→mass→K, BLS-t0 anchoring; auto-falls back to RV-only `InformedBirth` with no photometry). Forward/reverse use the same strategy family so detailed balance holds. **Critical for finding weak / arbitrary-period planets** — `0.0` gives blind RW births that rarely land. |
| `n_birth_tries::Int` | `1` | multi-try birth candidates per move |
| `n_birth_refine::Int` | `0` | post-birth RWM refinement steps (jointly refines jitter with the newborn) |
| `target_birth_accept::Real` | `0.234` | adaptation target for the MoMS scales |
| `n_temps::Int` | `16` | PT ladder size |
| `n_walkers::Int` | `100` | ensemble size (bumped to `≥ 2·n_dim + 2`, made even) |
| `n_steps` / `n_burnin` | `3000` / `2000` | post-burn-in steps / burn-in |
| `beta_min::Real` | `1e-4` | hottest β; geometric ladder `β_i = beta_min^(i/(n_temps-1))` over `[beta_min, 1]` — adding temps **densifies the cold end** rather than pushing T\_max hotter |
| `betas` | `nothing` | explicit ladder (overrides `beta_min`) |
| `noise_swap` / `noise_swap_rate` | `true` / `0.5` | post-burn-in DB-correct within-exclusion-group noise swap |
| `adapt_ladder` | `true` | equalise swap acceptance across the ladder during burn-in, holding β = 1 and `beta_min` fixed, then freeze it (window `ladder_adapt_window`, ν₀ `ladder_adapt_ν0`, K `ladder_adapt_K`) |
| `de_move::Real` | `0.5` | per walker and step, the probability of a differential-evolution move in place of the stretch move (as in `sample_pt_emcee`) |
| `prune_stranded::Bool` | `true` | burn-in only: walkers stranded far below their rung's mode, in the joint (model, parameter) space, are drawn about a mode walker's parameters and take its model |
| `node_flip::Real` | `0.1` | per walker and step, the probability of the (Ω, ω) → (Ω+π, ω+π) mirror of each astrometry-only planet; `0` disables it |
| `swap_min::Real` | `0.05` | least adjacent swap rate for the tempered `log_evidence` to be quoted |
| `seed` / `thin` / `show_progress` | `1` / `1` / `true` | |

The result `TransDimPTemceeResult` carries `chains`, `log_evidence`
(H⁺ → TI⁺ → TI, quoted only when the ladder reaches β = 0 and every adjacent
pair swapped at least `swap_min`, 5%, of the time — the default ladder stops at
`beta_min`, so it is `NaN` unless `betas` has a 0 rung; the occupancies do not
depend on it), `evidence_report` (raw), within/swap/transdim
acceptance, the β ladder, `n_evals`, the per-step `ladder::LadderHistory`
(drawn by `plot_ladder_rates` / `plot_beta_ladder`), plus **raw** per-planet
(`td_proposed`, `td_accepted`) and per-temp noise
(`noise_td_proposed`, `noise_td_accepted`) move counts — raw counts
because a rounded rate of `0.000` reads as "frozen" even while the
chain is transitioning (HD 18599 post-mortem,
`transdim_pt_emcee.jl:51`).

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
`noise_active_i == 1`. The occupancy of the planet count is P(k | D) under the
prior on k of the next section, which is not uniform on k (it is uniform over
the period-ordered evidences at the default `inclusion_prior = 0.5`); noise models
have the Bernoulli prior alone. To read a Bayes factor between planet counts off
the occupancy, divide the prior out: `bayes_factors(...; model_prior = k -> prior[k + 1])`
with `prior = transdim_k_prior(max_kplanet; inclusion_prior)`, or take
`run_info["transdim"]["ln_bayes_factor_adjacent"]`, which does. The same identity is what the validation
suite checks: occupancy must agree with an independently computed log Z
to ≲ 0.2 in ln Z once the ladder is dense and recovery aids are on.

## The prior on the planet count k

The prior a trans-dim run puts on the number of planets is **not** uniform on
k, and not binomial. Every planet slot carries an independent
Bernoulli(`inclusion_prior`) activation bit (`sample_transdim_pt_emcee`,
`sample_moms`, `sample_daedalus`; `rjmcmc` has none, which is the same as
`inclusion_prior = 0.5`), the planets of one planet mode are kept in period
order (next section), and which slot holds a planet carries no information. The
prior mass of the k-planet model, against its evidence ``Z_k`` under the
normalised slab prior, is

```math
P(k) \;\propto\; \Big(\frac{p}{1-p}\Big)^{k} \frac{1}{k!},
\qquad k = 0, \dots, k_\mathrm{max},\quad p = \texttt{inclusion\_prior},
```

where `c` is the share of one planet's slab prior that the model allows (below)
truncated at `k_max` (`max_kplanet`) and normalised, with one `1/k_g!` per planet
mode when the slot menu mixes modes (`transdim_k_prior`). The `1/k!` is the mass
of the period-ordered region of the k-planet parameter space, which is where the
chain lives and which the sampler does not renormalise. Then

```math
P(k \mid D) \;\propto\; P(k)\, Z_k ,
```

and the same statement in the fixed-k convention: at the default `p = 0.5`,
P(k | D) is *uniform-prior* over ``Z_k/k!``, the evidences of the period-ordered
model that a fixed-k fit with the ordering gate integrates. For `max_kplanet = 3`
at `p = 0.5` the prior is ``(0.375, 0.375, 0.1875, 0.0625)``. (with `c = 1`; see below). A likelihood that
carries no information (all errors 10⁶ m/s) returns the occupancy ratios
k+1 : k in the proportions c : c/2 : c/3 at `p = 0.5` and multiplied by
`p/(1-p)` at `p = 0.3`, which is this formula.

**The slab fraction `c`.** The default (√e sin ω, √e cos ω) parametrisation draws
both uniformly on [-1, 1] and the likelihood rejects e ≥ 1, so a new planet's
prior keeps π/4 ≈ 0.785 of its box: `c = π/4`. This is a hard constraint of the
prior, not a bug: the prior-only run's factor of about 0.75 per planet was this
(0.76, 0.75 and 0.69 of the 1 : 1/2 : 1/3 ratios; π/4 = 0.785). The evidence of the
k-planet model under the slab prior normalised over its allowed region is the
box-prior evidence divided by `c^k`, so `P(k)` carries `c^k`, and the ln B between
adjacent k is the evidence ratio under that normalised prior. `c` is computed by
Monte Carlo over the first slot's priors (`transdim_slab_fraction`) and reported
as `prior.slab_fraction`; other hard constraints (dynamical stability, `M_s`-dependent)
involve several planets at once, are not a per-planet factor, and are not in `c`.

The prior is recorded in `run_info["transdim"]["prior"]` (`p_k`,
`inclusion_prior`, `group_sizes`, the form and its meaning) and in
`model_selection["n_planets_prior"]`.

## Period ordering convention

Planets are labelled by period. Within each planet mode, the **active** slots
are kept in increasing period in slot order, `P_k1 < P_k2 < …` among the active
slots of that mode:

- a hard gate in the likelihood (`rv_log_likelihood`) rejects any within-model
  move that would put two active same-mode planets out of order;
- a birth draws the new planet, then inserts it into period order by swapping
  parameter blocks between active slots (`_sort_group_periods!`), with the
  activation bits untouched; a death cannot disorder a sorted state;
- a slot of another mode (an RV-only slot among RV+astrometry slots) is ordered
  only against slots of its own mode, and the SB2 companion is not a planet and
  is not ordered at all.

So slot `k1` is the shortest-period *active* planet of its mode, `k2` the next,
and so on, per draw: **a slot index is a rank by period, not an identity**.
Slots that are inactive in a draw hold parked values, and an inactive slot can sit
between two active ones, so `planet_active_<k>` is not "has at least k planets".
A planet that is the shortest-period one in one draw and not in another changes
slot between draws; across draws the columns `P_k<j>`, `K_k<j>`, … of one slot are a
mixture over planets, and a quantity of one *planet* (the 12.3 d one) is read by
selecting draws where some active slot lies near that period, or through the
relabelled columns that `convergence_report`, the fit-health check and
`label_switching` build (active planets in the lowest slots, period-sorted,
per draw). Conditioning on a slot being active, as the summary tables do, is
conditioning on the slot's rank being occupied. The ordering is what removes the
`k!` label-permutation degeneracy of the posterior; it is also the `1/k!` of the
prior on k above.

## Reporting the planet count

Every trans-dim run (any engine that has `n_planets` in its chain) writes, under
`run_info["transdim"]`:

| key | content |
|---|---|
| `prior` | the prior on k above: `p_k`, `inclusion_prior`, `group_sizes`, form |
| `occupancy`, `per_k` | P(k \| D); per k the draws, effective draws (`n_eff`), walkers that visited it, separate visits |
| `ln_bayes_factor_adjacent` | per pair `k_hi = k+1`, `k_lo = k`: `ln_bayes_factor`, `se`, `ln_prior_ratio`, `estimable`, `reason` |
| `mixture_evidence` | `available`, `log_z`, `method`, `how`, `evidence_reliable`, `ln_z_k` |
| `k_max_truncation` | `p_k_max`, `warn`, `advice` |
| `occupancy_health` | `ok`, `problems`, `rule` |
| `k_transitions` | accepted k transitions per move type (below) |

(also `diagnostics["k_transitions"]`, and the prior, the adjacent ln B and the
mixture evidence in `model_selection`.)

**ln B between adjacent k.**
`ln B(k+1 vs k) = ln[P(k+1 | D) / P(k | D)] − ln[P(k+1) / P(k)]`, the posterior
occupancy ratio less the prior ratio of the previous section. This is the Bayes
factor of the evidences ``Z_{k+1}/Z_k`` under the normalised slab prior. `se` is
the delta-method standard error from the effective size of the two occupancy
indicators (autocorrelation-corrected across the walkers' traces, with the
multinomial covariance). `per_k[k].n_eff` is the effective number of draws in k
(its occupancy times that size). A pair is `estimable = false`, with a `reason` and
a warning, when either k has fewer than 25 effective draws; when a count was
never drawn, `ln_bayes_factor` is `NaN` (null in JSON), not infinity.

**Mixture evidence.** `mixture_evidence.log_z` is ``\ln Z = \ln \sum_k P(k)\,Z_k``,
the evidence of the whole trans-dim model marginalised over k with the prior
above. It is the engine's own log evidence: a tempered or nested estimator
integrates the likelihood against the normalised prior of the trans-dim model
(`method` names the estimator `log_z` came from: `ti_plus`, `hybrid`, `nested`,
…). It is available only when the engine reports one (the tempered stack needs a
ladder that reaches β = 0 and swaps ≥ 5%; `rjmcmc` and `moms` have no estimator);
otherwise `available = false` and `reason` says so, and the adjacent ln B still
hold. `ln_z_k` then gives each visited k's evidence,
``\ln Z_k = \ln Z + \ln P(k \mid D) − \ln P(k)``.

**Truncation by `k_max`.** When `P(k = k_max | D) > 0.5` the run warns that the
prior's upper bound on k is likely truncating the posterior, and says to raise
`k_max`: `max_kplanet` in the `transdim` block (or `TransDimConfig`) **and** in
the model, then refit. Not raised for a run that fixes the count
(`planets = false`).

**Occupancy health.** Every k with P(k | D) ≥ 0.01 must have been visited by at
least `min(3, n_walkers)` walkers and, unless it holds ≥ 90% of the posterior, by
at least 10 separate visits (entries into k, summed over walkers); otherwise its
mass rests on one or two excursions and is not an estimate. `occupancy_health.ok =
false` fails the run's convergence verdict (next section).

**Accepted k transitions.** `k_transitions` counts, per move type, the proposed and
accepted moves summed over the ladder, and at the β = 1 rung (`cold_*`): `birth`,
`death`, `noise_toggle_swap` (noise-model birth, death and within-group swap, which
the engine counts together) and, for `transdim_pt_emcee`, `planet_noise_swap` (the
planet ↔ activity-model swap, with both directions). Engines with no move counters (`rjmcmc`, `moms`, `daedalus`) report `up` and `down`, the k changes seen between consecutive recorded draws of a walker (also `transitions_observed` for every engine; a lower bound on the accepted births and deaths). A run whose cold rung accepts
no birth or death did not move in k, whatever its occupancy says.

## One convergence verdict, and the evidence of an unconverged fit

A summary used to carry two convergence answers: `run_info["convergence"]["pass"]`
(split R̂ ≤ 1.1, ESS ≥ 400 gated on well-occupied components) and
`fit_health["checks"]["convergence"]` (R̂ ≤ 1.05, ESS ≥ 200), and they could
disagree. Both are now written from one rule (`converged` at the top of the summary,
with `convergence_verdict` saying which checks looked and which failed): a run has
converged when **none** of these failed: the R̂/ESS gate, the fit-health convergence
check, the trans-dim occupancy health, a MAP fit's optimiser, and the run ending on
its own (not `interrupted`, `requested`, `max_seconds`, `max_evals`).
`run_info.convergence.pass`/`converged`, `fit_health.converged` and the status of
`fit_health.checks.convergence` all follow it; what the gate and the check each said
on their own is kept as `rhat_ess_pass` and `own_status`.

The evidence of a run that has not converged is **reported, not withheld**, and
flagged: `evidence_reliable = false` sits beside it (`summary`, `summary["evidence"]`,
`run_info`, the trans-dim mixture evidence), `evidence_unreliable_reasons` lists the
checks that failed and, if `log_z` is the mode-Laplace fallback, says so, and a
warning prints the same. It applies to every engine and every evidence path (TI,
TI⁺, SS⁺, H⁺, bridge, nested, mode-Laplace, a MAP fit's Laplace). When nothing assessed the run
`evidence_reliable` is `null`.

## Reading trans-dim output

The chain gets these extra columns
(`transdim_pt_emcee.jl:2173-2179`):

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
(`reporting.jl:307`) and `save_transdim_summary(chains, params, path;
starname, …)` (`reporting.jl:429`) — the path is positional, and
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
`:n_planets` column (`src/samplers/daedalus.jl:736,762`):

```julia
model_probabilities(chains)         # → Dict(0 => 0.0, 1 => 0.998, 2 => 0.002)
bayes_factors(chains)               # → Dict of all pairwise BFs
bayes_factors(chains; reference=0)  # → Dict(k => BF(k vs 0))
```

## The `transdim_occupancy` plot

`plot_transdim_occupancy(chains, params; td=nothing, …)`
(`src/plotting/transdim_plots.jl:24`) renders a 2×2 trans-dim summary:

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
(`runner.jl:526`). Birth strategies are named by string
(`runner.jl:318`).

```json
{
  "sampler": { "name": "transdim_pt_emcee",
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

`_build_transdim` (`runner.jl:1637`) parses the block. `auto` plots add
`transdim_occupancy` automatically whenever the chain has a
`:n_planets` column (`runner.jl:1764`).

!!! note "Toggleable noise models can't come from JSON"
    `transdim.toggleable` and `noise_exclusion_groups` must hold
    `NoiseModel` *objects*, which JSON can't encode. `_coerce_noise_models`
    rejects a non-empty JSON `toggleable`/`noise_exclusion_groups` with an
    explicit error (`runner.jl:1626`). Noise-only trans-dim therefore needs
    the Julia API (build the `TransDimConfig` in-process), not a raw JSON job.
    Schema reference: `Nereus.jl/docs/JOB_CONFIG.md`.

## Rossiter-McLaughlin in joint fits

The `*_RM` planet modes add the in-transit Rossiter-McLaughlin RV
anomaly to a joint RV + transit fit
(`src/model.jl:183-261`). Both the leading-order Hirano+ 2011 analytical
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
  one global slot (`model.jl:1872`); default prior
  `LogUniformPrior(500, 100_000)` m/s (`default_priors.jl:447`). Read
  with `system_vsini(theta)` (`parameters.jl:584`).
- `lambda_k<k>` — sky-projected stellar obliquity λ\_k of planet `k` in
  **radians** (`parameters.jl:637`); default prior
  `UniformPrior(-π, π)` (`default_priors.jl:487`). Read with
  `planet_lambda(theta, k)`.

The RM term is now folded into `rv_predictions`
(`src/likelihood.jl:1771`), so the predicted RV used by the PPC,
residuals, fit-health metrics, and every RV plot is RM-consistent
(no-op for non-RM fits). Use the dedicated `rm_anomaly` plot
(`plot_rm`, `src/plotting/rm_plots.jl:48`) to inspect the in-transit
anomaly; `run_job` emits it automatically for any RM-enabled fit
(`runner.jl:1785`).

## Newer plots relevant to trans-dim QC

Beyond `transdim_occupancy`, three plots help vet a trans-dim
solution (all dispatchable via `output.plots`, `runner.jl:321`):

- `rm_anomaly` — per-planet RM anomaly with predictive band
  (`plot_rm`); only for `*_RM` fits.
- `transit_overlay` — per-transit QC gallery, the median transit model
  overlaid on each individual transit window
  (`plot_transit_overlay_fit`, `runner.jl:2103`).
- `rv_components` — RV model decomposition (Keplerian / activity /
  trend pieces), distinct from `rv_timeseries`
  (`plot_rv_components`, `runner.jl:1967`); auto-emitted when there are
  ≥ 2 planets or a smooth (GP/AGP) activity component.
- `ttv_oc` — O−C transit-timing diagram. Now renders for
  **single-planet** fits too: with no perturber it shows the data-only
  O−C (measured per-transit T\_c vs the linear ephemeris) by passing
  `planet_b_k = 0` (`runner.jl:2092-2097`).

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
`ActivityGP` constructor (`src/noise/types.jl:475`). JSON strings are
symbolised before construction (`runner.jl:1328`).

**ActivityGP needs the indicator errors.** Provide them via the data
block:

- In an `rv` `values` block, add `<name>` and `<name>_err` arrays
  (e.g. `bis`/`bis_err`, `fwhm`/`fwhm_err`). The
  `<name>_err`-paired-with-`<name>` convention splits values from
  errors (`runner.jl:725`, `_split_indicator_errs`).
- In a CSV `rv` block, set `indicator_cols` and provide a matching
  `<col>_err` column per indicator (`runner.jl:718,793`).

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
