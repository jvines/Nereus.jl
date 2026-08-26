# The `fit_*` API

The highest-level way to use Nereus: **one function per technique**, each
taking data in its natural shape and returning the same result triple. No
`Data` construction, no `Params`, no `build_target`, no sampler selection
unless you want it.

```julia
using Nereus

res = fit_rv(Dict("HARPS" => (t = bjd, rv = rv, rv_err = err)); planets = 2)

res.chains                  # MCMCChains.Chains (or nothing, for point estimators)
res.log_z                   # marginal likelihood
res.summary["params"]       # per-parameter medians and credible intervals
res.summary["derived"]      # masses, radii, separations — physical units
```

Everything below composes down onto the same machinery documented in
[Parametrisations](parametrizations.md) and [Samplers](samplers.md). Use this
page when you want a fit; use those when you want to control the model.

The other two routes into Nereus are [`build_target`](quickstart.md) (build the
target yourself, call a sampler directly) and [`run_job`](examples.md) (a single
JSON/Dict config, for pipelines and non-Julia callers). They are three
interfaces to one engine, not three engines.

---

## The eight entry points

| function | technique | needs |
|---|---|---|
| `fit_rv` | radial velocities alone | RV |
| `fit_transit` | transit photometry alone | photometry |
| `fit_astrometry` | absolute and/or relative astrometry, no RV | IAD / HGCA / GOST / relative |
| `fit_joint` | several techniques simultaneously | two or more channels |
| `fit_rm` | Rossiter–McLaughlin | RV **and** photometry |
| `fit_ttv` | transit timing variations | photometry or measured times |
| `fit_binary` | spectroscopic binary (SB1 / SB2) | RV |
| `fit_tomography` | Doppler tomography | line profiles |

`fit_rm`, `fit_ttv` and `fit_binary` route through the config layer rather than
`build_target`, because RM, TTV and SB2 are not expressible as `build_target`
keywords. That is an implementation detail: the signatures stay
technique-shaped and you never see a config.

`fit_tomography` is not a sampler at all — see [Obliquity](obliquity.md).

---

## Data goes in per instrument, always

Every entry point takes a **per-instrument mapping**, never flat columns:

```julia
fit_rv(Dict("HARPS" => (t = bjd, rv = rv, rv_err = err)))
fit_rv(Dict("HARPS" => (...), "HIRES" => (...)); planets = 2)

fit_transit(Dict("TESS" => (t = bjd, flux = f, flux_err = ferr)))
```

This is not ceremony. The instrument name is what the offset (`gamma_<ins>`)
and jitter (`sigma_<ins>`) terms are keyed on, so there is no unnamed form —
a flat vector of RVs from three spectrographs is not a well-posed fit. Passing
bare vectors is rejected with a message naming the fields it wanted.

Required fields: `(:t, :rv, :rv_err)` for RV, `(:t, :flux, :flux_err)` for
photometry. `NamedTuple` or `Dict` both work.

> **Split by provenance, not just by instrument.** The same spectrograph
> reduced by two pipelines should be two entries. Offsets and jitter differ
> between reductions, and merging them silently averages that away.

---

## `planets` — an integer, or a range

```julia
fit_rv(rv; planets = 2)            # fixed dimension: exactly 2 planets
fit_rv(rv; planets = 0:4)          # trans-dimensional: model selection over count
```

An integer builds that many identical default blocks. A range hands the
problem to the trans-dimensional layer, which sets `max_kplanet` and the
occupancy prior — see [Trans-dimensional](transdim.md). Note a range needs a
trans-dim engine (`transdim_ptemcee`, `rjmcmc`, `moms`) to mean anything.

**The default priors come from your data, not from constants.** For RV,
`default_rv_planet` takes the period from twice the median cadence up to three
baselines, and the semi-amplitude from 0.1 m/s to ten times the RV scatter:

```julia
default_rv_planet(t, rv)      # -> (P, K, sesinw, secosw, Mo) priors
default_astrom_planet()       # -> (a, M_sec, sesinw, secosw, Mo, inc, Omega)
```

You should still look at them. A data-derived prior is a sane starting point,
not a considered one — and on a short baseline the upper period bound is
doing real work.

To override, pass a `NamedTuple` of blocks directly instead of an integer, in
exactly the form [`build_target`](quickstart.md) takes:

```julia
fit_rv(rv; planets = (k1 = (P = NormalPrior(4.1375, 0.001),
                            K = LogUniformPrior(1.0, 30.0),
                            sesinw = UniformPrior(-1, 1),
                            secosw = UniformPrior(-1, 1),
                            Mo     = UniformPrior(0, 2pi)),))
```

---

## Choosing an engine

```julia
fit_rv(rv; engine = "pt")                                   # by name
fit_rv(rv; engine = Dict("engine" => "ptemcee",
                         "options" => Dict("n_temps" => 16,
                                           "n_walkers" => 40)))
```

`ENGINES` is the whole table — eighteen samplers, one line each:

```
pt              pt_warm        pt_hmc       pt_whitening   ptemcee
transdim_ptemcee               nested       nested_ins     nested_dynamic
moms            moms_ns        rjmcmc       nuts           map
smc             ensemble       ess          pa
```

Two behaviours worth relying on:

- **An unknown engine name is an error that lists the valid ones.**
- **An option the sampler does not accept is an error, not a silent no-op.**
  A typo'd `n_round` that quietly does nothing is how you lose a day, so
  `run_engine` checks the option names against the sampler's actual keyword
  list and refuses unknown ones, naming what it does accept.

`run_engine(target, spec)` is the same dispatcher if you already have a target.

Which sampler to pick is [Samplers](samplers.md); the short version is that
`pt` is the right default, `transdim_ptemcee` is the workhorse for model
selection, and nested sampling wants good priors.

## `Stopping`

Uniform across engines, separate from sampler-specific knobs:

```julia
Stopping(; max_seconds = nothing, max_evals = nothing,
           ess_min = nothing, rhat_max = nothing,
           logz_tol = nothing, check_every = 500)
```

---

## What comes back

Every entry point returns `(; chains, log_z, summary)`.

`summary` is a plain `Dict{String,Any}`, JSON-able, with:

| key | contents |
|---|---|
| `"op"` | which entry point ran |
| `"status"` | `"ok"`, or `"railed"` — see below |
| `"log_z"` | marginal likelihood (`NaN` where the engine gives none) |
| `"elapsed_sec"` | wall time |
| `"params"` | per-parameter median and credible intervals |
| `"derived"` | physical quantities — masses, radii, separations, equilibrium temperatures |
| `"output_dir"` | present when you passed one |

`chains` is `nothing` for point estimators (`map`) and for `fit_tomography`.
Engine-specific extras (acceptance rates, occupancy, evidence breakdowns) are
merged in on top.

**The railed-MAP guard.** If the MAP lands against a prior bound, `status`
becomes `"railed"` and `"error"` names the offending parameters. That is a
**failed fit reported as a failure**, not a result with a caveat — a posterior
pressed against a bound is telling you the bound is wrong, and the median it
reports is meaningless. Do not read past it.

**Saving.** Pass `output_dir` and the chains are written to `chains.nc`
alongside the data and evidence, ready to reload for plotting without
refitting. See [Diagnostics & I/O](diagnostics.md).

---

## Astrometry references

`IADData` does not cross a socket well — it is a struct of parallel arrays plus
scan geometry, and shipping it as JSON is both enormous and a second copy of
the reader's conventions waiting to drift. So `fit_astrometry` accepts a
**reference** and resolves it server-side with the same code path a Julia user
would call:

```julia
fit_astrometry(iad = Dict("catalogue" => "gaia_dr4",
                          "source_id" => 1457486023639239296),
               parallax = Dict("dist" => "Normal", "mu" => 13.628, "sigma" => 0.021),
               m_pri = 0.644)

fit_astrometry(iad = Dict("catalogue" => "hipparcos", "hip" => 27321))
```

Supported catalogues are `gaia_dr4` (also `gaia_epoch`, `dr4`) and
`hipparcos` (also `hip`, `iad`). A `"path"` key, or `NEREUS_GAIA_DR4_XML`,
short-circuits the download. `resolve_astrometry(spec)` does this standalone,
and passing an `IADData` straight through works too.

> **Always give `fit_astrometry` an informative parallax.** The abscissae
> constrain the product a₀ ∝ M_sec·ϖ and are blind to its factorisation, so
> the parallax prior *is* the mass scale. The default without one is four
> decades wide, which is no scale at all. See
> [Data models](data.md) and [the solution ladder](solution_ladder.md).

---

## Worked shapes

**Multi-instrument RV, two planets, explicit sampler:**

```julia
res = fit_rv(Dict("HARPS" => (t = t1, rv = v1, rv_err = e1),
                  "HIRES" => (t = t2, rv = v2, rv_err = e2));
             planets = 2, trend_order = 1,
             engine = Dict("engine" => "pt", "options" => Dict("n_rounds" => 12)),
             output_dir = "out/two_planet")
```

**Trans-dimensional search with a noise menu:**

```julia
res = fit_rv(rv; planets = 0:4,
             noise = default_noise_menu(data).noise_models,
             engine = "transdim_ptemcee")
```

**Joint RV + astrometry — the `M sin i` → `M` break:**

```julia
res = fit_joint(Dict("source" => "RV", "data" => rv),
                Dict("source" => "AS", "iad" => iad,
                     "parallax" => NormalPrior(25.36, 0.30), "m_pri" => 0.83);
                planets = 1)
```

`fit_joint` requires two or more channels on purpose: for a single technique
the dedicated entry point has a clearer signature and better defaults.

**Rossiter–McLaughlin, ARoME kernel:**

```julia
res = fit_rm(rv = rv, phot = phot, flavour = :arome,
             star = Dict("M_s" => 1.1, "R_s" => 1.3))
```

**Doppler tomography:**

```julia
r = fit_tomography(nights; P = 2.83, a_Rs = 5.6, inc = 1.48,
                   vsini = 95.0, T14 = 0.16)
r.summary["lambda_deg"], r.summary["p_value"]
```

---

## Where to go next

- [Quick start](quickstart.md) — `build_target` + a sampler, the level below this one
- [Worked examples](examples.md) — the `run_job` config route, ten complete fits
- [Samplers](samplers.md) — what the eighteen engines are and when to use each
- [Obliquity](obliquity.md) — RM flavours and Doppler tomography in detail
- [Data models](data.md) — the astrometry containers and the Gaia DR4 reader
- [Solution ladder](solution_ladder.md) — 5p/7p/9p model selection on epoch astrometry
