# Parametrisations

Nereus lets you choose how planet parameters are parametrised
without changing the underlying physics. Different choices condition
the posterior geometry differently — pick the one that gives the
sampler the smoothest landscape.

The choices are bundled in `ParametrizationConfig` and `PlanetDataSources`.

!!! note "Two ways in"
    Everything below is the Julia `Params(...)` / `ParametrizationConfig(...)`
    API. When you drive Nereus through `Nereus.run_job(cfg)` (the JSON/Dict
    entry point — see [`docs/JOB_CONFIG.md`](../JOB_CONFIG.md) and
    `src/runner.jl`), the same options live in the `model` block, but the
    run_job schema currently exposes a **strict subset** of them. Each section
    flags what is reachable from run_job and what is Julia-API-only.

## `PlanetDataSources` — which observables a planet contributes to

A planet is described by the **set** of data sources that see it
(`src/model.jl:90`). It is a composable `Set{Symbol}`, not an enum, so
RV / photometry / astrometry / Rossiter–McLaughlin / TTV compose
cleanly. Nereus exports the common combinations as named constants
(`src/model.jl:120`–`462`):

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

Membership tests use `in` (`src/model.jl:96`); predicate helpers
`has_rv`, `has_pm`, `has_as`, `has_rm`, `has_rm_r`, `has_any_rm`,
`has_ttv`, `has_ttv_nb` are exported (`src/model.jl:472`–`562`).

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
`RVPMASBlock`) in `src/model.jl:813`–`948` for the exact slot order.

!!! warning "Astrometry requires RV"
    `_make_block` (`src/model.jl:1559`) rejects pure-AS and PM+AS
    planets — every astrometric planet must also carry `:RV`. RM and
    TTV both require `:PM` (you need a transit window / transit
    observations), enforced by the constant definitions.

### run_job equivalent

In the `model.planet_modes` array, pass the
constant **names** as strings. The schema validator
(`_KNOWN_PLANET_MODES` in `src/runner.jl`) recognises exactly:

```
RV_ONLY, PM_ONLY, RVPM, RVAS, RVPMAS,
RVPM_RM, RVPMAS_RM, RVPM_RM_R, RVPMAS_RM_R,
RVPM_TTV, RVPMAS_TTV, RVPM_RM_TTV, PM_TTV,
RVPM_TTV_NB, RVPMAS_TTV_NB, PM_TTV_NB,
RVPM_RM_A, RVPMAS_RM_A, RVPM_GD, RVPM_RM_GD, PM_GD,
BINARY, BINARY_RV, PM_DT
```

```json
"model": { "max_kplanet": 2, "planet_modes": ["RVPM_RM", "RV_ONLY"] }
```

## `ParametrizationConfig`

```julia
ParametrizationConfig(;
    ew                = :sesinw,
    time              = :auto,       # resolved by Params: see below
    geom              = :b_rr,
    use_rho_s         = false,
    mass              = :K_driven,
    obs_prior         = false,
    marginalize_gamma = false,
    sample_a_Rs       = false,       # sample a/R★ per transiting planet
    t_ref_c           = NaN,         # phic epoch; NaN = taken from the data
    density           = :rho_circ,   # :rho_circ | :rho_star: see below
)
```

Defined in `src/model.jl` (`struct ParametrizationConfig`). The keyword
constructor validates `ew`, `time`, `geom`, `mass` and `density` and throws an
`ArgumentError` on any unknown value, so a typo fails fast at construction.
`use_rho_s` and `sample_a_Rs` are mutually exclusive: a/R★ is derived from the
stellar density or sampled directly, not both. `t_ref_c` is read only under
`time = :phic`.

### `ew::Symbol` — eccentricity parametrisation

The two eccentricity slots (`e1`, `e2`) are named and decoded
according to this choice (`src/model.jl:1493`–`1503`, decoded in
`src/parameters.jl` `planet_e_w`; conversions in `src/orbit.jl`):

- `:sesinw` (default) — sample `(√e·sin ω, √e·cos ω)`, named
  `sesinw_kN` / `secosw_kN`. Implicit prior on `e` is `U(0, 1)`; on `ω`
  uniform. The recommended default (Eastman+ 2013 geometry; Peña &
  Jenkins convention). Conversion `sesinw_to_ew` (`src/orbit.jl:126`).
- `:esinw` — sample `(e·sin ω, e·cos ω)`, named `esinw_kN` /
  `ecosw_kN`. Implicit prior `e ∝ 1` (linear) → biased toward higher
  `e`. Conversion `esinw_to_ew` (`src/orbit.jl:146`).
- `:ew` — sample `e` and `ω` directly, named `ecc_kN` / `w_kN`. Use
  when you want an explicit prior on `e` (e.g. `BetaPrior(0.867, 3.03)`
  for Kipping 2013, or a `NormalPrior` from an external constraint).
  Poorly conditioned at low `e` (ω unconstrained), so prefer `:sesinw`
  unless you specifically need a named `e` prior.

`true_anomaly` (`src/orbit.jl`) and the astrometric orbit clamp `e` into
`[0, 1)` defensively because the unconstrained sampler can transiently
propose `s²+c² > 1`. Every likelihood rejects `e ∉ [0, 1)` first, so the
clamp never changes the `e` a likelihood evaluates.

!!! note "External `e` prior"
    You don't have to switch to `:ew` to impose an `e` prior. The
    `external_priors` hook (see below) carries an `ExternalPrior(:ecc, …)`
    that is evaluated on the **derived** eccentricity regardless of the
    `ew` parametrisation — keep the well-conditioned `:sesinw`
    geometry and still get a `BetaPrior` on `e`.

### `time::Symbol` — orbital time anchor

The single time slot (`t`) is named and decoded per this choice
(`src/model.jl:1505`; conversions `src/orbit.jl:175`–`236`). With no
anchor given, `ParametrizationConfig` holds `:auto`, and `Params` resolves it
when it is built (`resolve_time_anchor`), the same way in `build_target`, the
`fit_*` functions and run_job: the anchor the priors name (`Mo_k1`, `Tp_k1`,
...; in `build_target` also the planet blocks), else `:phic`, the default
(`:Mo` under `obs_prior`). So a setup written for the old default with an
`Mo_k1` prior still samples `Mo`, with or without a `ParametrizationConfig`
of its own. Priors naming two anchors are an error: a model samples one. A
`Params` never holds `:auto`: it stores the resolved anchor, so a checkpoint
written by a fit whose anchor was resolved to `:Mo` resumes into one given
`time = :Mo` explicitly, and refuses a `:phic` one. With an explicit anchor, a
prior left on another anchor's slot is reported as such.

!!! note "Why the transit phase is the default"
    At low `e` the data pin the mean longitude `λ = M₀ + ω`, not `M₀` and
    `ω` apart. Sampled as `(M₀, √e sin ω, √e cos ω)` that is a thin ridge,
    a helix in the sampler's coordinates; chains mix badly along it, each
    holding its own `ω`, and that biases `e`. On an RV + Hipparcos IAD target
    with a circular orbit, `pt_emcee` in `M₀` gave a median `e` of 0.0097
    against 0.0081 in `φ_c` at `i = 45°`, and `rjmcmc` 0.0019 against
    0.0061 at 89.5°. `φ_c` has no such ridge: at `e = 0` it is `π/2 − λ`.
    For fixed `(P, e, ω)` the map `M₀ → φ_c` is a shift, so a uniform `φ_c`
    is the same prior as a uniform `M₀` and the posterior is unchanged; only
    the geometry the sampler sees differs. It is also the natural anchor of a
    transit fit. `M₀`, `Tp` and `Tc` are reported per draw in the derived
    table (`Mo_deg`, `Tp`, `Tc`). The one exception is the O'Neil `obs_prior`
    (below), which keeps `:Mo`.

- `:phic` (the default) — sample the transit phase `φ_c = 2π (Tc − t_ref_c) / P`
  (`phic_k<k>`, rad, default prior `U(−π, π)`). It is `Tc` measured in
  phase: it has no `e` or `ω` in it, so the transit pins it directly, as it
  does `Tc`, and it lives on a circle, so its window is one period at every
  `P` (`Tc ∈ [t_ref_c − P/2, t_ref_c + P/2)`). A uniform `φ_c` is uniform
  transit phase, `p(Tc | P) = 1/P`. Without a transit it is still defined
  (`Tc` is the inferior conjunction, `f = π/2 − ω`) and is the mean
  longitude in disguise, which is why it suits RV and astrometric orbits too.
  `Tc` is reported per draw as the derived `Tc` (`phic_to_tc`), so its
  covariance with `P` is exact, and `Tp` and `M₀` (`Mo_deg`, at `data.t_ref`)
  alongside it.
  `t_ref_c` (`ParametrizationConfig(t_ref_c = …)`, run_job
  `parametrization.t_ref_c`) defaults to the median photometric time, where
  `Tc` is least correlated with `P`; without photometry it is `data.t_ref`.
  `Params` fixes it with the parametrization, so every draw means the same
  `Tc` whatever subset of the data a likelihood later sees; a `Params` built
  without `data` needs `t_ref_c` (or another anchor) and says so.
  `chains.nc` records the anchor and this epoch (attributes `time_anchor`
  and `t_ref_c`), so a sampled `phic_k<k>` decodes without the data.
  Like `Mo`, `φ_c` is not treated as circular when TTVs or the O'Neil
  `obs_prior` are on: a 2π shift relabels the transit epochs there.
- `:Mo` — sample mean anomaly `M₀` at the
  reference epoch `t_ref` (= `data.t_ref`, the data median). The default
  until the transit phase replaced it, and still the default under
  `obs_prior = true`: the O'Neil prior is a density over one fixed period of
  `Tp`, which `M₀ ∈ [0, 2π)` spans (`Tp ∈ (t_ref − P, t_ref]`) and the
  `φ_c` window, which moves with `(e, ω)`, does not. `mo_to_tp`
  (`src/orbit.jl:193`).
- `:Tp` — sample time of periastron passage. Natural for visual orbits
  where Tp has direct geometric meaning. `tp_to_mo` (`src/orbit.jl:202`).
- `:Tc` — sample time of transit centre (inferior conjunction). Right
  for `RVPM`/`PM_ONLY`/transit fits: the transit ephemeris constrains
  `Tc` directly, and `Tp` becomes derived via `tc_to_tp`
  (`src/orbit.jl:212`, transit at `f = π/2 − ω`). The `ttv_oc` plot and
  the per-transit Tc measurements need `:Tc` or `:phic`.

!!! warning "Keep a Tp or Tc prior within one period"
    The likelihood repeats every period, so a `Tp`/`Tc` prior wider than
    `P` holds several identical copies of the orbit, and the default one
    spans the whole data baseline. Ensemble samplers cannot move between
    copies: on a 12.3 d RV target with a 929 d window the walkers spread
    over 46 copies and R-hat stayed at 1.9–5 with every walker in the mode.
    `Params` warns when a window is wider than the shortest period the `P`
    prior allows. Use `phic` (the default) or `Mo`, or a prior no
    wider than one period around a known epoch. The `Tc` window also fails
    the other way: when `P` is longer than the baseline, `U(t_min, t_max)`
    forbids transit times outside the data, which `phic` does not.

### `geom::Symbol` — transit geometry parametrisation

The two transit-geometry slots (present only for PM planets,
`src/model.jl:1508`–`1517`) are named and decoded per this choice:

- `:b_rr` (default) — `b` (impact parameter) and `rr = R_p/R_⋆`, named
  `b_kN` / `rr_kN`. Standard.
- `:r1r2` — Espinoza (2018) two-parameter form, named `r1_kN` /
  `r2_kN`. `(r1, r2)` map the unit square `[0,1]²` onto the physically
  allowed `(b, rr)` region with uniform density, explicitly excluding
  the `b > 1 + rr` prohibited region from the prior box. Cleaner for
  transit fits where `b` is near 1 (grazing). Decoded in
  `planet_b_rr` (`src/parameters.jl:397`); the RM path converts
  `(r1, r2) → (b, rr)` internally (`src/rm.jl:350`).

### `use_rho_s::Bool` — stellar-density parametrisation

- `false` (default) — each transiting planet's `a/R_⋆` is derived from
  `P + M_⋆ + R_⋆` via Kepler III with the total mass `M_⋆ + M_p` (so `M_s` and
  `R_s` must be set on `Params`).
- `true` — a single **shared** stellar density `rho_s` slot is declared
  and `a/R_⋆` is derived from `rho_s + P` for
  every transiting planet (`rho_s_to_a_Rs`, with the total mass:
  `(a/R_⋆)³ = G ρ_⋆ P² (1 + q)/(3π)`, `q = M_p/M_⋆`). Use for multi-planet
  transiting systems: `rho_s` is one physical quantity shared across
  the system, and sampling it directly trades N per-planet
  density-equivalent parameters for one. Standard for TESS-style
  transit fits ([Vines+ 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.2627V/abstract)
  Table 7 convention). The RM `a/R_⋆` also prefers the `rho_s` path
  when present (`src/rm.jl:360`).

  `rho_s` is a *sampled* slot only when the density reparametrization below
  does not take it over. Under the default `density = :rho_circ`, a transiting
  planet with a free eccentricity samples `rho_circ_k<k>` instead, and `rho_s`
  becomes a derived quantity (still in the chain, the tables and the corner). It
  stays sampled under `density = :rho_star`, for a model with no such planet,
  and for every engine that runs in ρ★ and (e, ω) (see below). A `rho_s` prior
  that is fixed makes ρ★ *given*, which is the case where every eligible planet
  samples `eta_k<k>`.

```julia
parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr, use_rho_s = true)
priors["rho_s"] = NormalPrior(2.241, 0.479, 0.1, 10.0)   # ARIADNE-derived
```

### `density::Symbol` — the photo-eccentric (ρ_circ) reparametrization

A transit measures its duration, and the duration fixes the
circular-equivalent density

```
ρ_circ = ρ★ · g³,     g = (1 + e sin ω) / √(1 − e²)
```

(Kipping 2010; Dawson & Johnson 2012). With a free eccentricity and no density
prior, the posterior in `(ρ★, e, ω)` is a thin curved sheet along constant
`ρ_circ` that reaches high `e`; an ensemble crosses it over thousands of steps
without R-hat, ESS or the drift check noticing. Sampling `ρ_circ` lays the sheet
along an axis. The posterior is the same distribution: the prior the model states
on `(ρ★, e, ω)` is carried through the Jacobian (`src/density.jl`), so a prior on
`rho_s`, `sesinw_k<k>` or `secosw_k<k>` still means what it said.

- `:rho_circ` (default). **Eligible** planets are those with transit photometry,
  transit geometry and both eccentricity coordinates free (not fixed). A model
  with none, or with `sample_a_Rs = true`, or with no usable ρ★ (no `rho_s`
  prior under `use_rho_s`, or no `M_s` and `R_s`), is not reparametrized: the
  option is inert.
  - **Anchor case.** When ρ★ is sampled (`use_rho_s = true` and a `rho_s` prior
    that is not fixed), the *first* eligible planet samples `rho_circ_k<k>` in
    place of `rho_s`; its `(e, ω)` coordinates stay sampled, and ρ★ = ρ_circ/g³ is
    derived.
  - **η case.** Every other eligible planet samples `rho_circ_k<k>` **and**
    `eta_k<k>`, and its `e` and `ω` are derived. This is every eligible planet
    when ρ★ is *given*: a fixed `rho_s`, or `M_s`/`R_s` (the host mass, sampled
    `M_pri` or not, over the fixed radius). At the model's ρ★, `g = (ρ_circ/ρ★)^{1/3}`
    and the orbits of that `g` fill an ellipse in the `(e cos ω, e sin ω)` disc,
    which `η ∈ (−π/2, 3π/2)` parametrizes (a bijection onto `e < 1`; where the
    `e` prior stops short of 1, `η` is sampled on the arc that holds it).
  - **The cap.** `DENSITY_G = 100` bounds `g` to `[1/100, 100]`, so the box of
    `ρ_circ` is `[ρ★_min/10⁶, ρ★_max·10⁶]`. That excludes orbits with
    `e > 1 − 2/G² = 0.9998`, whose periastron lies inside the star (about
    3/(4G²) ≈ 7.5×10⁻⁵ of the prior mass under a uniform `e`). A wider box costs
    nested sampling: at `G = 10⁶` the two-planet log Z ran 1.3 nats low.
  - **Priors on the new slots.** None are yours to give: `rho_circ_k<k>` and
    `eta_k<k>` carry the prior on ρ★ and `(e, ω)`. They are not in
    `summary.provenance.priors`, which lists the priors you set.
  - **Output.** The derived coordinates (`rho_s`, `e`/`ω` of the η planets) are
    added back to the chain, so `chains.nc`, the tables and the corner show ρ★
    and `(e, ω)` as before, plus the new `rho_circ_k<k>` / `eta_k<k>` columns.
- `:rho_star` — sample ρ★ (or `a/R★`) and `(e, ω)` as before.

**Engines that keep the old coordinates.** `pt`, `transdim_pt_emcee`, `moms`,
`daedalus`, `rjmcmc` and `ess` run the model in ρ★ and `(e, ω)` whatever
`density` says (`without_density_reparam`; `src/runner.jl` swaps the model and
its parameters before the sampler runs), and so does the optimisation inside
`sample_map`, which returns its point in the layout's own coordinates. Their
chains therefore have no `rho_circ_k<k>` or `eta_k<k>` columns. Every other
engine run through `run_job` (`pt_emcee`, `pt_hmc`, the nested family, `nuts`,
…) samples `ρ_circ`.

```julia
ParametrizationConfig(use_rho_s = true)                     # default: anchor + η planets
ParametrizationConfig(use_rho_s = true, density = :rho_star) # the old coordinates
```

### `mass::Symbol` — mass / first-two-slot parametrisation

Controls how the first one or two planet slots are named and what they
mean (`src/model.jl:1467`–`1491`; mass-function machinery in
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

What is not sampled is still reported, in the **derived** table (never in the
fitted one, so `K_k1` of the two tables cannot be confused):

| derived key | unit | reported when |
|---|---|---|
| `P_days` (and `P_yr` in the science table) | d (yr) | `P` is not a slot: `:a_driven` |
| `K` | m/s | `K` is not a slot and the planet has RV: `:M_sec_driven`, `:a_driven` |
| `a_mas` | mas | the planet carries astrometry (`RVAS`, `RVPMAS`): `a·ϖ`, the relative orbit as an angle |
| `alpha_star_uas` | μas | same: `a·ϖ·M_sec/(M_pri+M_sec)`, the host's reflex orbit (astrometric signature) |

Each is evaluated per draw with `planet_P` / `planet_K` / `planet_a` /
`planet_M_sec` / `astrom_M_pri` / `astrom_plx`, so all three modes agree with
the likelihood; under `:K_driven` the companion mass behind `alpha_star_uas`
comes from the mass function with the sampled inclination.

None of the derived masses or separations uses the `M_p << M_★` approximation:

| derived key | how it is computed |
|---|---|
| `msini_*` | companion mass at i = 90° from the exact mass function `(M_p sin i)³/(M_★+M_p)² = f(K,P,e)`, solved with `msec_from_K` and the table's per-draw M_★ |
| `mass_*` | the same inversion at the draw's `sin i` (transit planets); for astrometric planets with a sampled K, the model's own `planet_M_sec` per draw |
| `a_au` | Kepler III with the total mass `M_★ + M_p` (M_p the true mass, or the exact minimum mass when no inclination exists); for astrometric planets the model's `planet_a`, so `a_au·ϖ = a_mas`; where a/R★ comes from the transit model, `a_au = (a/R★)·R★` |

A brown dwarf (0.08 M_⊙ about 0.5 M_⊙) was 9 % off in `msini` with the closed
form; a Jupiter about a Sun-like star moves by 0.07 %.

The mass function is solved analytically. In `x = M_p/M_⋆` and
`F = f/(M_⋆ sin³i)` it is `x³ − F x² − 2F x − F = 0`; its discriminant,
`−F²(4F + 27)`, is negative for every `F > 0`, so there is exactly one real
root and Cardano's formula gives it (`mass_ratio_from_F`) with no cancellation
anywhere, to ≈ 4·10⁻¹⁶ relative over `F = 10⁻¹⁶ … 10⁴`.

**The transit and the companion's mass.** The planet is in the transit through
Kepler III, so a light curve measures `ρ_⋆(1 + q)`, not `ρ_⋆`. Where the planet's
mass is determinable (a transiting planet with a sampled `K` or `M_sec`:
`RVPM`, `RVPMAS`) the model solves `q`, `a/R_⋆` and `cos i` together
(`planet_geometry`, `coupled_geometry`):

    a/R_⋆ = a₀ (1 + q)^(1/3),   cos i = b(1 + e sin ω)/((a/R_⋆)(1 − e²)),
    q = x( f / (M_⋆ sin³i) ),

a fixed point started at `q = 0` that contracts at about `q/3` per pass (twenty
passes at `q = 0.5`; two or three for a planet). `a₀` is the value for a massless
companion, from `ρ_⋆` or from `(M_⋆, R_⋆)`. `M_⋆` is `astrom_M_pri` (the sampled
`M_pri`, else `M_s`; for a `rho_s` model with neither, `ρ_⋆ R_⋆³`), the same star
the RV mass function uses. A **transit-only** planet has no mass to put in: `q = 0`,
and the fitted `rho_s` is then, strictly, `ρ_⋆(1 + q)` (invisible for a
planet, 10⁻³ for a Jupiter; not for a brown dwarf or a stellar companion).
`a_Rs`, `a_au`, `inc_deg`-dependent quantities (`T14`, `Teq`, `S_earth`, density,
TSM/ESM) follow from the coupled `a/R_⋆`, and `rho_star_transit` is
`ρ_⋆` once `1 + q` is divided out where `q` is known.

### `obs_prior::Bool` — O'Neil 2019 observation-based prior

When `true`, applies the O'Neil 2019 (AJ 158, 4) observation-based
prior on `(P, e, Tp)` for direct-imaging astrometric orbits. Encodes
the geometric truth that long-period orbits with poorly-sampled phase
coverage have correlated `(P, e, Tp)` posteriors and gives the sampler
a prior that respects that geometry. Only relevant for planets carrying
`:AS`.

### `marginalize_gamma::Bool` — analytic γ marginalisation

(`src/model.jl:630`–`631`.) When `true`, the per-instrument RV systemic
offset `γ` is **analytically marginalised** (orvara's approach). Because
`γ` enters the RV model linearly (`pred = γ + Keplerian + trend`), the
likelihood is Gaussian in `γ` and each instrument's `γ` integrates in
closed form at fixed jitter `σ`. The `gamma_*` slots are then **frozen
out** of the sampled set (fewer free dimensions, cleaner mixing and
evidence) and the RV likelihood uses the γ-marginalised path
(`_rv_ll_gamma_marginalized`).

- White-noise RV only — it does **not** compose with a covariance
  (GP/celerite) RV model, and any RV-channel noise model is rejected at
  construction. Photometry-channel noise models (a GP on the light curve,
  say) are fine: the RV likelihood stays γ-marginalised alongside them.
- `γ` is integrated against a **flat, unnormalised measure**, not against the
  `gamma_<INST>` prior. The marginal likelihood therefore carries an arbitrary
  constant per instrument. It is the same constant for every model fitted to the
  same instruments, so comparing those models is unaffected; comparing against
  an evidence computed with `γ` sampled is not meaningful.
- For the same reason a prior supplied for `gamma_<INST>` cannot be applied, and
  is rejected at construction rather than accepted and ignored.
- Most useful for short-arc RV + astrometry, where the Keplerian reflex
  over the RV baseline is a ~km/s offset that `γ` must absorb and
  marginalising it removes a stiff nuisance direction.

### run_job coverage of `ParametrizationConfig`

The `model.parametrization` block in run_job (`_build_model` in
`src/runner.jl`) maps only these keys, with these accepted string values:

| run_job key | accepted strings | maps to | default |
|---|---|---|---|
| `mass` | `K_driven`, `M_sec_driven`, `a_driven` | `:K_driven` / `:M_sec_driven` / `:a_driven` | `K_driven` |
| `time` | `Tp`, `Tc`, `Mo`, `phic` | `:Tp` / `:Tc` / `:Mo` / `:phic` | `phic` |
| `t_ref_c` | a time (`phic` only) | `t_ref_c` | median photometric time |
| `ew` | `sesinw`, `ew` | `:sesinw` / `:ew` | `sesinw` |
| `geom` | `b_rr` | `:b_rr` | `b_rr` |
| `marginalize_gamma` | `true` / `false` | bool | `false` |
| `use_rho_s` | `true` / `false` | bool | `false` |
| `density` | `rho_circ`, `rho_star` | `:rho_circ` / `:rho_star` | `rho_circ` |

```json
"model": {
  "max_kplanet": 1,
  "planet_modes": ["RVPM"],
  "parametrization": {
    "mass": "K_driven", "time": "Tc", "ew": "sesinw",
    "geom": "b_rr", "marginalize_gamma": false,
    "use_rho_s": true, "density": "rho_circ"
  }
}
```

!!! warning "run_job gaps vs the Julia API"
    Several Julia-API options are **not** reachable through run_job
    today (`_build_model`, `src/runner.jl`):
    - `obs_prior` and `sample_a_Rs` are never read from the config — set them by
      constructing `Params`/`ParametrizationConfig` directly. (`use_rho_s` and
      `density` ARE read, from `model.parametrization`.)
    - `ew = :esinw` and `geom = :r1r2` are valid in Julia but **absent
      from the run_job string maps** (`_EW_PARAM`, `_GEOM_PARAM`).
    - The schema validator advertises `geom = "b_r"`
      (`_KNOWN_PARAMETRIZATIONS`) and the map yields `:b_r`, but
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
  `src/model.jl:1872`). Default prior `LogUniformPrior(500, 100_000)`
  m/s (`src/default_priors.jl:447`); override with a `NormalPrior` when
  spectroscopic `v sin i` is measured. Read via `system_vsini`
  (`src/parameters.jl:584`).
- **`lambda_kN`** — per-planet sky-projected obliquity `λ`, in
  **radians** (`src/model.jl:1534`). Default prior `UniformPrior(-π, π)`
  (`src/default_priors.jl:487`); tighten to `NormalPrior(0, π/8)` for
  aligned systems. Read via `planet_lambda` (`src/parameters.jl:637`).

The leading-order Hirano model is `ΔRV_RM(t) = −Δflux(t) · v_p(t)` with
`v_p = V·sin(i_*)·(x_sky cos λ − y_sky sin λ)` (`rm_signal`,
`src/rm.jl:71`). The `:RM_R` Reloaded variant
(`rm_reloaded_signal`, `src/rm.jl:611`) instead integrates the
intensity-weighted line-of-sight velocity over the planet's
sky-projected disk (21×21 sub-cells), more accurate at high `rr` /
high `b`, reducing to Hirano in the small-planet limit.

The RM term is now part of `rv_predictions` (`src/likelihood.jl:1771`),
so PPC, residuals, fit-health diagnostics and **all** RV plots are
RM-consistent — not just the likelihood. `a/R_⋆` for the RM geometry
comes from `rho_s` when `use_rho_s = true`, otherwise from
`M_s + R_s + P` (`src/rm.jl:360`); if neither is available the RM term
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
`src/plotting/rm_plots.jl:48`) — see [Plotting](plotting.md).

## Activity-indicator data (for `ActivityGP` and indicator GPs)

The multivariate `ActivityGP` (Rajpaul+ 2015) and other indicator-GP
noise models need the activity-indicator time series **and their
1σ errors**. This is data wiring rather than a parametrisation choice,
but it is required to make the relevant `planet_modes`/`noise_models`
fit run, so it is documented here; the noise model itself lives in
[Noise models → `ActivityGP`](noise_models.md).

In run_job, declare the noise model in `noise_models`
(`src/runner.jl:1119`, `_build_noise_models` `src/runner.jl:1299`):

```json
"noise_models": [
  { "kind": "ActivityGP",
    "instruments": ["HARPS"],
    "kwargs": { "channels": ["bis", "fwhm"], "use_derivative": true } }
]
```

`channels` is symbolised for you (`src/runner.jl:1328`); valid channels
are `bis`, `fwhm`, `logrhk` (no derivative term), `halpha`
(`src/noise/types.jl:386`–`398`). `ActivityGP` deliberately has **no**
kernel-amplitude parameter (unit-variance latent `G(t)`); all scale
lives in the per-channel couplings (`Vc/Vr`, `Bc/Br`, …).

Supply the indicator data alongside the RV data:

- **`values` RV block** — add `<name>` and `<name>_err` arrays next to
  `bjd`/`rv`/`rv_err`/`instrument` (`src/runner.jl:694`–`722`). A
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
  (`src/runner.jl:740`–`810`):

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

Instrument names (`src/model.jl:712`), used only to build parameter
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

Returns a `Params` value with two fields (`src/model.jl:1400`):

- `params.config` — the immutable `ParamsConfig` (data sources,
  instruments, parametrisation, priors, stability, M_s/R_s, …)
- `params.layout` — the derived `ParamsLayout` (`names`,
  `name_to_idx`, `n_total`, `n_p_idx`, `unfrozen_idx`,
  `unfrozen_names`, `unfrozen_priors`, `frozen_idx`/`frozen_values`,
  `planet_blocks`, `systemic`, `packed_priors`)

Everything for the hot path is precomputed by `_build_layout`
(`src/model.jl:1683`), which also **validates** that every layout slot
has a prior, rejects unknown prior keys (typo detection), enforces
globally-unique parameter names, and runs `validate_physical` on each
prior.

Useful accessors (`src/model.jl:2088`–`2119`):

```julia
n_total(params)       # total parameter count
n_unfrozen(params)    # only the sampled ones
n_frozen(params)      # FixedPrior parameters
param_index(params, "K_k1")   # index into the theta vector
```

### `trend_order` / `phot_trend_order` — global trends

- `trend_order` adds a long-term RV trend on the systemic velocity:
  `1` adds a `dvdt` slot (linear), `2` also adds `d2vdt2` (quadratic)
  (`src/model.jl:1850`–`1857`).
- `phot_trend_order` adds a per-PM-instrument photometric baseline
  polynomial `1 + offset + Σ c_p·xᵖ` in normalised time, sharing the
  `offset` grouping; coefficients are named `phot_cP_<label>` with a
  default `NormalPrior(0, 0.05, -0.5, 0.5)` (`src/model.jl:1794`–`1798`).

### `stability` — multi-planet stability prior

A hard prior rejecting dynamically unstable multi-planet configs.
One of `:none`, `:amd` (AMD criterion), or `:gladman` (Gladman 1993
Hill-stability), validated in `src/params_constructor.jl:100`. The
Julia `Params` default is `:amd` (`src/params_constructor.jl:85`); the
run_job default is `none` (`src/runner.jl:1169`). A non-`:none` choice
requires a finite `M_s` for `max_kplanet ≥ 2`
(`src/params_constructor.jl:102`).

### `sharing` — parameter group sharing

For multi-pipeline datasets where the same physical instrument is
reduced by multiple pipelines (e.g. HARPS-DRS vs HARPS-TERRA), some
parameters should be shared. Pass a dict mapping a **category** to
groups of instrument names (`_instrument_groups`, `src/model.jl:1635`):

```julia
sharing = Dict(
    :sigma => [["HARPS_DRS", "HARPS_TERRA"]],   # share RV jitter
    :ld    => [["TESS", "K2"]],                 # share limb darkening
)
```

Valid categories (`src/model.jl:1328`): RV `:sigma`, `:gamma`; PM
`:pm_jitter`, `:pm_offset`, `:pm_dilution`; and `:ld` (the `q1`/`q2`
pair). Instruments listed in the same inner vector share one slot;
instruments in no group keep their own. Group labels join names with
`+` (`gamma_HARPS_DRS+HARPS_TERRA`).

GP/celerite hyperparameter sharing across instruments is handled by the
noise-model `instruments` field rather than this dict — see
[Noise models](noise_models.md).

### `external_priors` — priors on derived quantities

`ExternalPrior(quantity, prior, per_planet)` (`src/model.jl:1179`)
applies a prior to a value **derived** from sampled parameters, not a
sampled slot itself. Supported quantities (`src/model.jl:1185`):

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
