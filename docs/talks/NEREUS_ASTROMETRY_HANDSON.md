# Hands-on: fitting Gaia DR4 epoch astrometry with Nereus

Instructor's companion to the two notebooks in [`examples/`](../../examples/). It carries
what a notebook cannot: what to say while a cell runs, what will break in a room full of
laptops, and what to do when it does.

The physics, the data format and the models are in the companion document
[`GAIA_DR4_EPOCH_DATA.md`](GAIA_DR4_EPOCH_DATA.md), which is tool-agnostic. **This one assumes
that material and does not repeat it.**

| notebook | system | channels | the point |
|---|---|---|---|
| `01_gaia4_astrometry_only.ipynb` | Gaia-4 b | DR4 epoch astrometry | an orbit from along-scan abscissae alone, no RVs |
| `02_hd114762_joint_rv_astrometry.ipynb` | HD 114762 b | RV + DR4 epoch astrometry | breaking sin(i) to get a true mass |

Run them in that order; the second assumes the first.

---

# 0. Before the session

## 0.1 Install

Julia:

```julia
using Pkg; Pkg.add(url="https://github.com/jvines/Nereus.jl")
```

Python, for anyone who does not have Julia and does not want it — a prebuilt runtime is
fetched on first use:

```sh
pip install astronereus
python -c "import astronereus; astronereus.install()"
```

The PyPI distribution is `astronereus` because `nereus` belongs to an unrelated geophysics
package. The Julia package, the repository and the runtime cache all keep the name Nereus.

## 0.2 Have everyone do this the day before

Two things dominate the first ten minutes of any hands-on session, and both are avoidable:

```julia
using Nereus                       # precompilation: minutes, once
fetch_gaia_dr4_prerelease()        # ~1.2 MB download, cached under ~/.nereus/GaiaDR4
```

If the venue's network is hostile, distribute the VOTable on a USB stick and have people set

```sh
export NEREUS_GAIA_DR4_XML=/path/to/GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml
```

Both notebooks check that variable before attempting any download.

**Do not have thirty laptops precompile Nereus simultaneously over conference wifi.** Say this
in the announcement email, not on the day.

## 0.3 To run the notebooks

```julia
using Pkg; Pkg.add("IJulia")
```

then `jupyter lab` from `examples/`. For people who would rather not touch Jupyter, the same
cells run as scripts:

```sh
julia --project -e 'using Pkg; Pkg.add("NBInclude")'
julia --project -t auto -e 'using NBInclude; @nbinclude("01_gaia4_astrometry_only.ipynb")'
```

## 0.4 Threads

Everything below assumes `-t 4` or better. `julia -t auto` is fine. A single-threaded run of
notebook 1 at production settings is not a session-length activity.

---

# 1. Orientation: the shape of the API

Five things, in order. Everything in both notebooks is one of these.

```
  read the data      ->  read_gaia_epoch_votable(xml, source_id)  ->  src.iad :: IADData
  declare the model  ->  build_target(; planets, iad, rv, plx, M_pri, ...)  ->  target
  sample             ->  chains, log_Z = sample_pt(target; n_rounds, n_chains, seed)
  read the posterior ->  chains[:, :a_k1, :]  (an MCMCChains.Chains)
  plot and save      ->  plot_*(chains, target.params, target.data; output = dir)
                         save_chains(path, chains, target.params; data, log_evidence)
```

**`IADData` is the whole interface to absolute astrometry.** It holds the six vectors of the
data document's section 1.8, one element per surviving CCD crossing:

```julia
struct IADData
    t::Vector{Float64}                # barycentric MJD (TCB)
    abscissa::Vector{Float64}         # mas
    abscissa_err::Vector{Float64}     # mas
    psi::Vector{Float64}              # rad, North through East
    parallax_factor::Vector{Float64}
    pm_factor::Vector{Float64}        # Julian yr from the catalogue reference epoch
end
```

**The single most useful thing to tell an audience in this session:** Hipparcos IAD and Gaia
DR4 epoch data enter through the *same* container and the *same* likelihood. There is no
"DR4 likelihood". The DR4 reader is a format adapter, and the physics document explains why
that is provably sufficient rather than merely convenient.

Every plot function takes `(chains, params, data)` plus `output = dir` and names the file
itself; model figures land under `models/`.

---

# 2. Session 1 — Gaia-4 b, astrometry alone

Notebook `01_gaia4_astrometry_only.ipynb`, 16 cells. Budget ~20 minutes of talking plus the fit.

## 2.1 The data

```julia
const GAIA4_SID = 1457486023639239296

xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
if isempty(xml) || !isfile(xml)
    xml = fetch_gaia_dr4_prerelease()
end
src = read_gaia_epoch_votable(xml, GAIA4_SID)
```

Expected output:

```
Gaia-4: 824 along-scan abscissae, G = 11.91
reference position (ra0, dec0) = (209.5063, 31.6955) deg
baseline: 4.94 yr
```

**What to say while this runs.** 824 is not 824 independent measurements — it is 115
field-of-view transits, each contributing up to nine CCD crossings 4.86 s apart. Parsing the
whole 12-source VOTable takes 0.38 s; the reader keeps only rows AGIS itself used
(`used_by_agis_al`), which drops 10,080 CCD slots to 7,467 file-wide.

Worth showing live, because it costs nothing and makes the geometry concrete:

```julia
using Statistics
println("distinct scan angles: ", length(unique(round.(rad2deg.(src.iad.psi); digits=2))))
println("median sigma: ", median(src.iad.abscissa_err), " mas")
println("abscissa RMS: ", std(src.iad.abscissa), " mas")
```

Gaia-4 gives 101 distinct angles with full 360-degree coverage, median error 0.082 mas, and an
abscissa RMS of 81 mas — **the last number is the one that surprises people.** The abscissae
are absolute positions dominated by proper motion, not residuals. Contrast HD 114762 in
session 2: 558 abscissae but only 70 distinct angles and a genuine 60-degree hole.

## 2.2 The model

```julia
const M_PRI    = 0.644     # M_sun
const PLX_GAIA = 13.628    # mas
const PLX_ERR  = 0.021

target = build_target(
    M_pri = M_PRI,
    planets = (b = (
        a      = LogUniformPrior(0.3, 4.0),     # AU
        M_sec  = LogUniformPrior(0.001, 0.05),  # M_sun (11.8 M_Jup = 0.0113)
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        inc    = SinePrior(),
        Omega  = UniformPrior(0.0, 2pi),
        Mo     = UniformPrior(0.0, 2pi),
    ),),
    iad = src.iad,
    plx = NormalPrior(PLX_GAIA, PLX_ERR),
    M_s = M_PRI,
)
```

```
free parameters: 8
a_k1, M_sec_k1, sesinw_k1, secosw_k1, Mo_k1, inc_k1, Omega_k1, plx
```

**Four things to point at in this cell.**

**(a) Declaring `a` and `M_sec` chose the parametrisation.** Nereus auto-detects: `a` + `M_sec`
gives `:a_driven`, `M_sec` + `P` gives `:M_sec_driven`, otherwise `:K_driven` (the default,
sampling `P` and `K`). Under `:a_driven` the period is *derived* from Kepler III at every
evaluation, so the prior is log-flat on the **true mass** rather than on `K`. For joint
RV+astrometry that is what you want, and it is why there is no `P` and no `K` in the free
parameter list.

**(b) `plx` is a free parameter with a tight prior, and that prior is doing the work.** The
abscissae constrain only the product a0 = a_rel * M_sec/M_tot * plx. Demonstrate it — the
demonstration is more convincing than the assertion:

```julia
# with the reflex switched off, the abscissa likelihood ignores the parallax entirely
```

A 1000x change in the assumed parallax moves the log-likelihood by 4e-6 nats. The default when
no parallax information is supplied is `LogUniformPrior(0.1, 1000.0)` — four decades, i.e.
effectively no mass scale at all. **If you take one habit away from this session: on an
astrometry-only fit, always set the parallax prior deliberately, and quote it beside the
mass.**

**(c) `M_pri` is FROZEN.** Passing a bare number makes it a `FixedPrior`. A quoted `M_sec`
interval therefore does **not** propagate host-mass uncertainty. To include it, pass
`M_pri = NormalPrior(0.644, 0.03)` and watch the mass interval widen.

**(d) `SinePrior()` on inclination is p(i) = sin(i)/2 on [0, pi]** — isotropic orbital
orientation, equivalently uniform in cos(i). The code samples `i`, not `cos i`; the isotropy
lives in the prior. Without it a sampler escapes into face-on configurations with huge `M_sec`
and tiny `sin i`.

> **There is no astrometry-only mode**, and that is deliberate rather than an oversight. A fit
> with `iad` and no `rv` is still built as `RVAS`; the block keeps a velocity slot it never
> uses and the RV likelihood runs over zero observations. It costs nothing and it means one
> code path.

## 2.3 The fit

```julia
nrounds = parse(Int, get(ENV, "GAIA4_ROUNDS", "4"))   # raise to 12 for production
chains, log_Z = sample_pt(target; n_rounds = nrounds, n_chains = 8,
                          seed = 42, show_report = false)
```

`n_rounds` **doubles the sample count each round**, so the cost roughly doubles per round.

| rounds | wall time (4 threads) | what you get |
|---|---|---|
| 3-4 | ~12 s - 1 min | smoke test: does it run |
| 12 | ~6.4 min | a result you would quote |

**Run the short setting live and the long one during the coffee break.** The short setting is
pedagogically useful: at 3 rounds Gaia-4 gets the period roughly right
(568.8 [+49.5, -275.5] d) and fails the mass check. **That is what not-yet-converged looks
like**, and showing it is worth more than another converged plot.

## 2.4 The posterior

```julia
a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
ses   = vec(Array(chains[:, :sesinw_k1, :]))
sec   = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))

e_v = ses.^2 .+ sec.^2
M_J = M_sec .* 1047.57
P_d = [365.25 * sqrt(a_v[j]^3 / (M_PRI + M_sec[j])) for j in eachindex(a_v)]
```

At 12 rounds:

```
                 median  [+1sigma, -1sigma]      published
  P (d)           578.6  [+3.0, -3.0]            571.3
  M (MJup)        10.83  [+0.38, -0.34]           11.8
  e               0.406  [+0.062, -0.061]        0.338
  i (deg)         121.2  [+2.3, -2.3]            116.9
```

**A published Gaia+RV orbit, recovered from astrometry alone in six and a half minutes.** That
is the headline of session 1.

Note the eccentricity is 2 sigma high and the period 2.4 sigma long against the published
values. Do not paper over it — this is an 8-parameter fit to one channel with a fixed host
mass, and the published solution used more information.

> **Parallel tempering under multiple threads is not bit-reproducible even at fixed seed.** A
> different session recorded P = 573.3 d, i = 124.2 deg from this exact script. **If you re-run
> live, quote the recovery — "within 3% of the published period" — not a specific median.**
> Someone in the room will re-run it and get a different last digit, and you want to have said
> so first.

## 2.5 Plots

```julia
outdir = joinpath(pwd(), "gaia4_figures")
mkpath(outdir)

plot_iad_residuals(chains, target.params, target.data; output = outdir)
plot_epoch_astrometry_orbit(chains, target.params, target.data; output = outdir)
plot_orbit_skyplane(chains, target.params, target.data; output = outdir)
plot_corner(chains, target.params; output = outdir)
```

**`plot_iad_residuals` is the one to spend time on.** It shows the along-scan residuals before
and after the Keplerian:

```
5-param single star (orbit NOT modelled)   RMS 0.1543 mas   chi2/dof 3.50
5-param + Keplerian photocentre orbit      RMS 0.1017 mas   chi2/dof 1.42
```

and the point is that the second row is **1.24x the median per-point error, not 1.0**.
Something is still unmodelled at the ~0.06 mas level. Say so out loud.

`plot_epoch_astrometry_orbit` bins to 24 phase bins by default. It has to: a0 is about 0.25 mas
against 0.08 mas errors, so the unbinned cloud swamps the ellipse. If someone asks why the
points look "too good", that is the answer.

## 2.6 Save

```julia
save_chains(joinpath(outdir, "chains.nc"), chains, target.params;
            data = target.data, log_evidence = log_Z)
```

NetCDF, and it round-trips — the figures can be regenerated later without refitting. Worth
demonstrating, because the natural next question after a 6-minute fit is "do I have to do that
again to change a plot".

---

# 3. Session 2 — HD 114762, the joint fit

Notebook `02_hd114762_joint_rv_astrometry.ipynb`, 17 cells.

**Why this system.** HD 114762 b was the first exoplanet candidate ever announced (Latham et
al. 1989). RVs give M sin(i) ~ 11 M_Jup — a giant planet. The joint fit shows the orbit is
nearly face-on and the companion is a ~0.1 M_sun **star**. Thirty-five years of literature
resolved by one channel added to another; it is the best advertisement joint fitting has.

## 3.1 The two channels

RVs ship with the package (`test/data/hd114762_rv.dat`, California Legacy Survey, Rosenthal et
al. 2021): 24 HIRES + 35 Lick over 29.3 yr. Astrometry is 558 DR4 abscissae from 90 transits.

**Point out the baseline mismatch**: 29.3 yr of RV against 4.9 yr of astrometry, and the
astrometry still supplies the inclination. Long baselines are not what the astrometry is for
here — the reflex amplitude is.

## 3.2 The joint model

```julia
target = build_target(
    M_pri = M_PRI,                          # 0.83 M_sun
    planets = (b = (
        a      = LogUniformPrior(0.30, 0.45),   # AU  => P ~ 60-105 d
        M_sec  = LogUniformPrior(0.003, 0.5),   # M_sun -- reaches STELLAR
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        Mo     = UniformPrior(0.0, 2pi),
        inc    = SinePrior(),
        Omega  = UniformPrior(0.0, 2pi),
    ),),
    rv = (
        HIRES = (data = hires, sigma = LogUniformPrior(0.5, 50.0)),
        Lick  = (data = lick,  sigma = LogUniformPrior(0.5, 50.0)),
    ),
    iad = src.iad,
    plx = NormalPrior(PLX, PLX_ERR),        # 25.36 +/- 0.30 mas
    M_s = M_PRI,
    trend_order = 1,
)
```

13 free parameters. **Three deliberate choices worth defending out loud, because someone will
ask about each:**

**(a) The prior on `a` is tight, bracketing P = 83.92 d.** This is a targeted mass and
inclination fit, not a blind period search. The period has been known since 1989; re-searching
it would only add multimodality to a session with a fixed time budget. Say that explicitly
rather than letting it look like prior tuning.

**(b) `M_sec` reaches 0.5 M_sun — stellar.** If the prior stopped at the planetary regime the
fit could not find the answer. **This is the cell where the result is actually decided**, and
it is a good moment to make the general point: a mass prior that excludes the truth returns a
confident wrong answer with no diagnostic.

**(c) `trend_order = 1`, a linear trend for the wide outer M dwarf HD 114762 B.** Without it the
RVs cannot be fitted. A *quadratic* term over the 29-yr baseline is a near-unconstrained
degenerate direction that rails the posterior and breaks the evidence — the right model for B
is a second Keplerian, which is a different exercise.

## 3.3 What comes out

At 12 rounds, ~8 minutes:

```
posterior-median per-channel logL:  RV = -200.2 (N=59)   ASTROM = -2146.1 (N=558)
  -> RV chi2/N = 0.97

                 median  [+1sigma, -1sigma]      Kiefer+ 2019
  P (d)           83.92  [+0.00, -0.00]          83.92
  e               0.343  [+0.001, -0.002]        0.335
  M sin i (MJup)  12.27  [+0.03, -0.04]          ~11 (the RV-only value)
  i (deg)          4.84  [+0.08, -0.08]          6.2 +1.9 -1.3
  M_true (Msun)   0.139  [+0.002, -0.003]        0.103 +0.030 -0.025
```

**The slide is the last two rows.** RV alone: M sin i = 12.3 M_Jup, a giant planet. RV + DR4
astrometry: i = 4.8 degrees, M = 0.139 M_sun, a low-mass star. +1.2 sigma from the published
joint value.

**Be honest about the fit quality.** The astrometric chi2/dof goes 106 -> 10.1, not -> 1. Real
structure remains — the known wide outer companion, or a photocentre effect the point-mass
reflex does not capture. The mass measurement survives (RV chi2/N = 0.97, and the inclination
is pinned by a 1.75 mas photocentre orbit against 0.125 mas errors), but this is not a "the
model fits" slide. An audience that catches you overselling this loses the previous twenty
minutes too.

Also worth 30 seconds: fitting the orbit moves the parallax by 0.33 mas and pmDec by
1.6 mas/yr relative to DR3, on a source whose DR3 RUWE is 3.16. **The catalogue's own
single-star solution is measurably corrupted by the companion.**

## 3.4 The demonstration that lands

If there is time for one interactive exercise, it is the ridge walk: pin K and P, re-derive
`M_sec` at each inclination, evaluate each channel separately. The RV log-likelihood is
identical to every printed digit across a 54x range in true mass; the astrometry
log-likelihood swings by 5e4 nats with a sharp peak at 4.84 degrees. Numbers and method are in
the physics document, section 3.9.

It takes about fifteen lines and it makes the degeneracy break physical rather than asserted.

---

# 4. What will bite you in a live session

1. **Julia's soft-scope rule.** A top-level `for` loop accumulating into a global
   (`tot += d`, `ok &= c`) throws `UndefVarError` at the REPL and in a script, though *not*
   inside a function or a notebook cell that defines one. Wrap accumulators in a `let` block.
   This catches everyone once, always at the worst moment.
2. **PT is not bit-reproducible across threads at fixed seed** (2.4). Say it before you re-run.
3. **Short round counts produce plausible-looking wrong answers.** On HD 114762 at 4 rounds the
   inclination rails toward zero and the RV chi-squared per point sits near 59 — while the
   headline mass still lands near the published value. **A short run that agrees with the
   answer you expected is not evidence of anything.** Both notebooks default to production
   settings and expose the round count through `GAIA4_ROUNDS` / `HD114762_ROUNDS` so a smoke
   test is a deliberate act.
4. **`rv_log_likelihood` is the TOTAL, not the RV channel.** It includes astrometry. Using it as
   "the RV term" double-counts and inflated a reported RV statistic by ~10x until it was caught.
   The RV-only term is `Nereus._rv_log_likelihood_core`. If a participant reports an
   implausible per-channel number, this is the first thing to check.
5. **Two different parallaxes.** The sampled `plx` scales the reflex from AU to mas; the
   parallactic signal *in the abscissae* is one of five nuisances marginalised under a flat
   prior. They are not tied together. A participant expecting the fit to "measure the parallax"
   from epoch data will be confused — it does not, and the physics document explains why that is
   arguably the honest choice.
6. **Inclination comes back in radians on [0, pi]**, both senses reachable. For reporting, fold
   with `i > 90 ? 180 - i : i`. The (Omega+pi, omega+pi) pair is an *exact* likelihood
   degeneracy in astrometry alone, so expect two modes there in an astrometry-only corner plot;
   the joint fit breaks it, because RV is sensitive to omega.
7. **Building an `IADData` by hand without `parallax_factor` and `pm_factor`** leaves them at
   zero defaults and the design matrix goes rank-deficient. Anyone writing their own reader
   during the session will hit it, and **it does not announce itself**: the orbit likelihood's
   Cholesky fails and it falls through to an *unmarginalised* Gaussian on the raw abscissa — a
   finite number, no warning, catastrophically wrong. Measured on Gaia-4, **+622 becomes
   -437,831,126**. The ladder path fails more quietly still: with a proper prior,
   `M = X'WX + S^-1` is positive definite even when `X` has rank 2, so nothing errors at all
   and you get parameters sitting exactly at the prior with `prob = [0.33, 0.33, 0.33]`.
   **That uniform-thirds signature is the tell.** `IADData` does reject non-finite entries,
   which catches a NaN epoch — but zeros are finite, so it does not catch this.
8. **`rho_s` is in solar units, not g/cm^3** — only relevant if someone brings a transit into
   the session, but it is a silent factor of 1.411.

---

# 5. API reference for the session

## 5.1 Reading

```julia
fetch_gaia_dr4_prerelease(; force = false)      # caches under ~/.nereus/GaiaDR4
read_gaia_epoch_votable(xml)             # -> Dict{Int64, GaiaEpochSource}, all sources
read_gaia_epoch_votable(xml, source_id)  # -> GaiaEpochSource
```

`GaiaEpochSource` carries `source_id`, `ra0`, `dec0`, `g_mag` and `iad::IADData`.
`NEREUS_GAIA_DR4_XML` short-circuits the fetch.

Hipparcos IAD comes through `fetch_hgca` / the van Leeuwen parser and lands in the *same*
`IADData`.

## 5.2 Declaring the model

`build_target` infers **one** mode for all planets from which data arguments are non-`nothing`:

| data supplied | mode | inclination |
|---|---|---|
| rv | `RV_ONLY` | not sampled |
| phot | `PM_ONLY` | from `b` |
| rv + phot | `RVPM` | from `b` |
| **rv + astrom** | **`RVAS`** | **sampled** |
| **astrom only** | **`RVAS`** | **sampled** |
| rv + phot + astrom | `RVPMAS` | **derived from `b`**, not sampled |

`RVPMAS` deriving `cos i = b(1 + e sin w)/((a/R*)(1 - e^2))` is what stops transit and
astrometry pulling inclination to inconsistent values through two independent slots.

For **mixed** modes — one companion astrometric, another RV-only — drop to the low-level
constructor:

```julia
Params(; max_kplanet = 2, planet_modes = [RVAS, RV_ONLY], ...)
```

That is the right model for HD 114762 with its wide outer M dwarf. **Note the consequence:** an
`RV_ONLY` companion contributes **zero** reflex to the abscissae. That is a modelling
assumption, not a physical statement, and it should be a deliberate one.

## 5.3 Parametrisation

| declared | mode | sampled |
|---|---|---|
| `P` + `K` | `:K_driven` (default) | period and velocity semi-amplitude |
| `P` + `M_sec` | `:M_sec_driven` | period and companion mass |
| `a` + `M_sec` | `:a_driven` | semi-major axis and companion mass; P derived |

Under `:a_driven` and `:M_sec_driven` the prior is log-flat on the **true mass**; under
`:K_driven` with a sine prior on inclination it is log-flat on `K` and the implied mass prior
is not log-flat. Same posterior geometry, different prior. Prefer `:M_sec_driven` or
`:a_driven` for joint RV+astrometry.

Two traps: under `:a_driven`, changing `M_sec` at fixed `a` also changes `P`, so any hand-built
ridge scan must re-solve `a`. And for a **non-astrometric** block under `:M_sec_driven`,
`sin i` falls back to 1 — so the sampled `M_sec` is really **M sin i**.

## 5.4 Sampling

`sample_pt(target; n_rounds, n_chains, seed, show_report)` is what both notebooks use, and it
is the right default for these problems: robust on RV+astrometry, and it returns `log_Z`.

Nested sampling is available and is **not** recommended here without good priors — it is slow
in the modes that would help and struggles on the multimodal geometry.

## 5.5 The solution ladder

The 5p/7p/9p machinery is a library API — closed-form, ~22.5 microseconds per rung, no sampler:

```julia
astrom_design(iad, order)                       # n x order design matrix
astrom_solution(iad, order; prior_sigma)        # (names, q, sigma, cov, chi2_reduced, log_z, order)
astrom_logZ(iad, order; prior_sigma, residual)  # exact marginal likelihood
astrom_chi2_reduced(iad, order)
default_ladder_prior(iad, order; a0_max = 5.0, accel_yr = 10.0)
ladder_probabilities(iad; orders = (5,7,9), prior, model_prior, residual)
  # -> (orders, log_z, prob, best, chi2_reduced, adequate)
```

**Always read `chi2_reduced` and `adequate` beside `prob`** — the reason is in the physics
document, section 3.3, and Gaia BH3 is the example that makes it stick.

`residual` is the hook for "is the leftover curvature Keplerian or generic": subtract a
Keplerian along-scan model and re-run the ladder on what is left. That is how the +858 nat
Gaia-4 demonstration is produced. There is no one-line wrapper for it; you compose
`star_reflex_offset` and `along_scan_projection` yourself.

The ladder is **not** wired into `run_job` — it is a library API only.

## 5.6 Plots and output

```julia
plot_iad_residuals(chains, params, data; output = dir)
plot_epoch_astrometry_orbit(chains, params, data; output = dir)
plot_orbit_skyplane(chains, params, data; output = dir)
plot_rv_timeseries(chains, params, data; output = dir)
plot_corner(chains, params; output = dir)

save_chains(path, chains, params; data, log_evidence)
```

---

# 6. Capability boundaries — say these up front

Not caveats about correctness; boundaries of what exists. Stating them at the start saves
participants from spending the session trying.

- **No joint orbit + acceleration fit.** The absolute-astrometry likelihood always marginalises
  exactly five nuisance parameters, never seven or nine. "Keplerian plus a long-period
  companion's acceleration" is not expressible today.
- **No acceleration-to-mass conversion.** The M_c ~ P^(4/3) relation of the physics document is
  arithmetic you do yourself; there is no function for it.
- **No reader for a published 7p/9p catalogue row.** You can *fit* a 7p or 9p solution from
  epoch data, but there is no likelihood term that consumes a published (acc_ra, acc_dec) plus
  covariance as a constraint. The nearest available path is the long-baseline
  proper-motion-difference channel (Hipparcos-Gaia), which encodes acceleration-like
  information over 26 yr rather than within the mission.
- **Along-scan only, and no chromatic term.** That is what the DR4 pre-release contains — the
  across-scan columns are empty — but the colour columns *are* present in the file and are not
  used. Whether that matters at the 0.06 mas level was not established either way.
- **Excess noise is not folded into the per-point errors.** Deliberate: for an orbital source
  the excess noise *is* the orbit, and folding it in absorbs the signal being fitted. The cost
  is that formal parameter errors are optimistic relative to published Gaia uncertainties.
- **Per-CCD abscissae are treated as independent.** Consistent with the data; a ~20% correlated
  per-transit component is not excluded, and there is no machinery to relax the assumption.

---

# 7. Timings, for planning the session

| step | cost |
|---|---|
| `using Nereus`, first time | minutes (precompilation) |
| `fetch_gaia_dr4_prerelease()` | ~1.2 MB, seconds |
| parse the VOTable, all 12 sources | 0.38 s |
| one orbit log-likelihood, N = 824 | 230 microseconds |
| Gaia-4 fit, 3 rounds | ~12 s |
| **Gaia-4 fit, 12 rounds** | **6.4 min** |
| HD 114762 joint, 9 rounds | ~1.2 min |
| **HD 114762 joint, 12 rounds** | **8.0 min** |
| one closed-form ladder rung | 22.5 microseconds |

All measured on four threads. **Plan the two production fits to run over a break**, and use the
short settings live — deliberately, and while saying what is wrong with them.
