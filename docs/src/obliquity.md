# Obliquity: RM, Doppler tomography, gravity darkening

Three independent ways to measure how a planet's orbit is tilted against its
star's spin, all in one package. They constrain different things, fail in
different ways, and can be fitted jointly.

| technique | measures | needs | entry point |
|---|---|---|---|
| Rossiter–McLaughlin | sky-projected obliquity λ | RV through transit + photometry | `fit_rm` |
| Doppler tomography | λ, from the planet's shadow in the line profile | time series of line profiles / CCFs | `fit_tomography`, `tomogram_bayes` |
| Gravity darkening | λ **and** stellar inclination i★ | high-precision transit photometry of an oblate star | `fit_transit(gravity_darkening = true)` |

**Only gravity darkening reaches the true 3-D obliquity ψ from one dataset.**
RM and tomography both give the sky-projected angle; the stellar inclination
they cannot see is exactly what a symmetric transit hides. That is the reason
the gravity-darkened model exists despite being harder.

---

## Rossiter–McLaughlin

RM is an RV effect: the planet occults part of the rotating stellar disc, so
the disc-integrated line is distorted and a Gaussian fit to it moves. Nereus
carries three kernels, and **the choice matters more than people expect**.

```julia
res = fit_rm(rv = rv, phot = phot, flavour = :arome,
             star = Dict("M_s" => 1.1, "R_s" => 1.3))
```

| `flavour` | mode | what it computes |
|---|---|---|
| `:arome` *(default)* | `RVPM_RM_A` | Boué et al. 2013 CCF model — what a Gaussian fit to a CCF actually measures |
| `:reloaded` | `RVPM_RM_R` | Cegla et al. 2016, disk-integrated local profiles |
| `:rm` | `RVPM_RM` | legacy flux-weighted subplanet mean (Ohta/Winn form) |

**Use `:arome` unless you have a specific reason not to.** The legacy `:rm`
kernel computes the flux-weighted mean subplanet velocity, which is *not* the
quantity a pipeline reports: pipelines fit a Gaussian to a CCF, and the two
diverge in amplitude and shape as `v sin i` rises and as the local line width β
becomes comparable to the rotational broadening. It is kept for method
comparison, not for measuring anything.

The sub-planet line width is not a free parameter in the ARoME kernel — it
follows from the CCF width. A rotationally broadened profile fitted by a
Gaussian has σ ≈ 0.5503 · v sin i convolved with the local line, so
β_p² = σ₀² − (0.5503 v sin i)², recomputed every call because `v sin i` is
sampled.

**Every RM mode is RV + photometry + RM.** There is no RM-only mode, because
the kernel needs the transit geometry — impact parameter, scaled semi-major
axis, radius ratio — and inventing it defeats the point. `fit_rm` therefore
requires both channels and will refuse without them.

Parameters added: a system-level `v_sin_i_star` (m/s) and a per-planet
`lambda_k<k>` (rad). See [Parametrisations](parametrizations.md).

---

## Doppler tomography

Instead of collapsing the line to one velocity, keep the profile. The planet
occults a specific velocity slice of the rotating disc, leaving a travelling
bump — the *shadow* — that walks across the line profile during transit. Its
track is set by the geometry, so fitting the track gives λ.

**For a rapid rotator this is strictly more information than RM.** The RM
signal in RV is a single number per epoch, swamped by pulsations on an A or F
star; the line profile keeps the spatial structure that distinguishes a planet
shadow from a pulsation mode.

### Data: one `TomoNight` per transit

```julia
struct TomoNight
    tag::String              # night label
    t::Vector{Float64}       # timestamps
    R::Matrix{Float64}       # residual profiles, n_time × n_velocity
    grid::Vector{Float64}    # velocity grid (km/s)
    Tc::Float64              # THIS night's measured mid-transit time
end
```

`ccf_profile(λ_obs, flux, ...)` builds profiles from spectra if you do not
already have CCFs.

> **`Tc` is per night, not per system.** Pooling several transits means each one
> carries its own measured mid-time. Folding them all on a single ephemeris is
> exactly the error that smears the shadow away — and it does so quietly, by
> reducing the significance rather than by failing.

### Two silent killers

Both learned the hard way; neither announces itself.

1. **Barycentric velocities are not optional when pooling.** Nights separated
   by weeks differ in barycentric velocity by tens of km/s — 13.4 km/s between
   two nights of one campaign, half the `v sin i`. Without `bervs` the tracks
   sit at different offsets and the pooled peak is meaningless.
2. **Every night's profiles must share a sign convention.** A DRS CCF is a
   *dip* (`ccf/continuum`); a mask-CCF built by accumulating line depth
   *peaks*. Mix the two and one night subtracts from the others. Convert dips
   with `(continuum − ccf)/continuum` before pooling.

### Why pool at all

A single night of an A star does not have the signal. On one γ Dor host, three
separate nights peaked tens of degrees apart and none of them meant anything.
Pooling works because **the planet's track depends only on the geometry and is
the same angle every night, while the pulsations dominating each night's
residuals are at a different phase each time** — so the shadow adds coherently
and the noise does not.

### The Bayesian path — prefer this

```julia
post = tomogram_bayes(nights; P = 2.83,
                      vsini = (95.0, 3.0),      # (mean, sd) — propagated as priors
                      b     = (0.30, 0.05),
                      a_Rs  = (5.6,  0.2))
bf = shadow_bayes_factor(post.chain[:, α_index])
post.λ            # degrees, wrapped to (−180, 180]
```

Geometry uncertainties go in as `(mean, sd)` pairs and are **propagated into λ
rather than asserted away**. Walkers start with λ spread uniformly around the
full circle on purpose: the shadow track for λ and for some reflections of it
can look similar when the transit is short compared with the crossing time, and
an ensemble started in one basin reports a confident answer without ever having
visited the other.

Significance comes from `shadow_bayes_factor` — a Savage–Dickey ratio on the
shadow amplitude against α = 0. The models are nested (no shadow *is* α = 0),
so the Bayes factor is the prior density at zero over the posterior density
there, estimated with a **reflected** kernel (an ordinary kernel puts half its
mass outside the support at a boundary and would inflate the factor ~2×).

> **Check convergence before believing the Bayes factor.** It depends entirely
> on the posterior density at one edge of the support, so it is set by whichever
> draws sit in the extreme tail — precisely where walkers stuck in a distant
> mode end up. On real data it has returned ln B ≈ 0 for an amplitude posterior
> well separated from zero, because a few of 260 walkers were trapped in a
> distant λ mode where the amplitude collapses, and were still leaking out at
> the end of the chain. Verify the fraction near zero is stationary across the
> chain and that no walker spends most of its life there.

### The matched-filter path — legacy, still the `fit_*` route

```julia
r = fit_tomography(nights; P = 2.83, a_Rs = 5.6, inc = 1.48,
                   vsini = 95.0, T14 = 0.16)
r.summary["lambda_deg"], r.summary["p_value"]
```

`fit_tomography` and the `run_job` `"tomography"` kind route through
`tomogram_pooled`: a matched filter over λ against the predicted shadow track,
with significance from a **permutation null** built by scrambling the
in-transit frames.

**This is the legacy statistic.** Prefer `tomogram_bayes` + `shadow_bayes_factor`
where you can. A permutation null answers a weaker question — whether the map
has *any* coherent structure — and on a pulsating star it answers that far too
enthusiastically, because pulsations are themselves coherent structure. The
Bayesian version is a statement made by the same posterior that reports λ,
under the same noise model.

`fit_tomography` takes `nights` as NamedTuples/Dicts with `profiles`, `vgrid`,
`times`, that night's `Tc`, and `bervs`. Which frames are in transit is derived
from `Tc` and `T14`, not passed in. It is **not a sampler** — passing `engine`
warns and is ignored.

`filter_pulsations = true` is the default: for a rapid rotator the pulsation
power sits right where the shadow does and inflates the significance. It costs
signal on data that has none, so it is exposed rather than hardwired
(`tomogram_pulsation_filter`).

### The pieces underneath

| function | what it does |
|---|---|
| `shadow_track` | predicted velocity of the shadow vs time, for a geometry |
| `shadow_map` | the full model shadow, n_time × n_velocity |
| `tomogram_residuals` | out-of-transit mean removed, returns the residual map |
| `tomogram_pulsation_filter` | suppress coherent pulsation power |
| `tomogram_matched_filter` | score a residual map against a track |
| `tomogram_null_distribution` | permutation null for that score |
| `kron_gp_loglike` / `white_map_loglike` | correlated / white likelihoods for the map |
| `tomogram_injection_test` | inject a known shadow and recover it |

Because `fit_tomography` calls `shadow_map`, the simulator and the fitted model
cannot drift apart.

---

## Gravity darkening

A rapidly rotating star is oblate and its poles are hotter than its equator
(von Zeipel; Barnes 2009). A planet crossing such a disc produces an
**asymmetric** transit whose shape depends on λ *and* on the stellar
inclination i★ separately — which is why this is the only one of the three that
reaches the true 3-D obliquity from a single dataset.

```julia
res = fit_transit(phot; gravity_darkening = true)
```

Modes: `PM_GD`, `RVPM_GD`, `RVPM_RM_GD` (`GD_SOURCE`). It reuses the RM slots —
`lambda_k<k>` and `v_sin_i_star` mean the same thing — and adds exactly **one**
parameter, `i_star`, because the rotation rate is not free once `v sin i` and
the stellar radius are known.

It needs genuinely good photometry. The asymmetry is a subtle distortion of the
transit shape, not a feature you will see by eye.

---

## Building the inputs

`joint_obliquity_fit` takes nights directly, but the framework path — a `Data`
plus a `Params` you can hand to any sampler — goes through two functions.

```julia
data, inst_names = obliquity_data(rm_nights; tomo_nights = tomo_nights)
params = obliquity_params(data, inst_names;
                          P = 2.83, Tc = 2459000.123,
                          b = (0.30, 0.05), a_Rs = (5.6, 0.2),
                          rr = (0.10, 0.02), vsini = (95_000.0, 3_000.0))
```

**`obliquity_data(rm_nights; tomo_nights, t_ref)`** assembles RM velocity
nights and tomographic maps into one `Data`, with **one night mapped to one
instrument**. That is structural, not cosmetic: each night gets its own offset
and jitter, because a spectroscopic transit taken on a different night has its
own velocity zero point and its own pulsation realisation. Night tags become
the instrument names and must therefore be unique — a duplicate tag is an
error, not a silent merge, since the two nights' offsets and jitters would
otherwise collide.

**`obliquity_params(data, inst_names; P, Tc, b, a_Rs, rr, vsini, ...)`** builds
the parameter set. The transit-solution arguments are `(mean, sd)` tuples
rather than scalars, and they enter as **priors, not fixed values** — an
obliquity fit usually has no light curve of its own, so the geometry comes from
a published solution and carries that solution's uncertainty into λ. `P` is
pinned hard (a `NormalPrior` of fractional width `1e-6`); the ephemeris enters
through `Mo_k1`, not a transit-time slot, so a published `sigma_Tc` is
converted to a width `2π·sigma_Tc/P` in mean anomaly.

!!! warning "`vsini` units differ between the two paths"
    `obliquity_params` takes `vsini` in **m/s**, because it writes the
    framework's `v_sin_i_star` slot, which lives on the RV channel (its hard
    bounds are `100.0` to `300_000.0`). So 95 km/s is `(95_000.0, 3_000.0)`.

    `joint_obliquity_fit` and `tomogram_bayes` take it in **km/s**, matching
    the tomographic `vgrid`, and convert internally for the RV kernel
    (`obliquity_joint.jl:163`). There 95 km/s is `(95.0, 3.0)`.

    Both are self-consistent; they are simply different interfaces. Passing a
    km/s value to `obliquity_params` lands under its 100 m/s floor and will
    rail, rather than fail loudly.

Set `arome = true` to use the ARoME CCF formulation (`RVPM_RM_A`) instead of
Hirano; prefer it whenever `v sin i ≳ β_p`.

!!! note "rm_velocity_fit is superseded"
    `rm_velocity_fit` still exists and emits a deprecation warning. RM
    velocities already flow through `rv_log_likelihood`, so a `Params` built
    by `obliquity_params` covers the velocity-only case — use
    `joint_obliquity_fit(...; use_tomogram = false)`, which keeps the
    parameter space identical to the joint fit and so stays comparable to it.

## Joint fits

```julia
post = joint_obliquity_fit(tomo_nights, rm_nights;
                           P = 2.83, vsini = (95.0, 3.0),
                           b = (0.30, 0.05), a_Rs = (5.6, 0.2))
```

Tomography nights and RM nights in one posterior, sharing λ and the geometry.
Pass `rm_nights = RMNight[]` for the tomogram alone, or `use_tomogram = false`
for the velocities alone — **the parameter space is identical in all three
cases, so the three results are directly comparable.** That is the design: it
lets you show what each channel contributes rather than asserting it.

```julia
struct RMNight
    tag::String
    t::Vector{Float64}
    rv::Vector{Float64}
    err::Vector{Float64}
    σ0::Float64            # CCF width, m/s
    Tc::Float64            # this night's mid-transit
end
```

`use_rv_gp` toggles a damped-oscillator (SHO) term on the velocities against a
white jitter-only null. The null is genuinely the term **absent**, not merely
shrunk — a Bayes factor between them is otherwise meaningless.

---

## Simulating a night before you propose for it

Both simulators call the same forward model the fitters use, so a feasibility
estimate cannot drift from what the fit would actually do.

```julia
sim = simulate_tomogram(; P = 2.83, Tc = 2459000.5, a_Rs = 5.6, λ = deg2rad(-47),
                        vsini = 95.0, rr = 0.116, T14 = 0.16,
                        sigma_pixel = 0.002, b = 0.30)
# -> (; tag, profiles, vgrid, times, bervs, Tc, in_transit, truth)

night = simulate_rm_night(; P = 2.83, Tc = 2459000.5, a_Rs = 5.6, λ = deg2rad(-47),
                          vsini = 95.0, rr = 0.116, T14 = 0.16, sigma_rv = 8.0)
# -> (; night::RMNight, truth)
```

`sigma_pixel` is in units of the line depth. Feed the output straight into
`tomogram_bayes` or `joint_obliquity_fit` to answer "would this exposure time
and this many nights actually detect the shadow" — which is the question a
telescope proposal is really asking.

---

## Practical guidance

- **Rapid rotator, pulsating host** → tomography, pooled over nights, Bayesian
  path, pulsation filter on. RM in RV will fight the pulsations and lose.
- **Slow rotator, clean RVs** → RM with `:arome`. Cheaper and sufficient.
- **You need ψ, not λ** → gravity darkening, or combine λ with an independent
  i★ (asteroseismology, `v sin i` plus a rotation period).
- **You have both RM and tomography** → `joint_obliquity_fit`, and report the
  three comparable runs.
- **Never quote a single night's λ on an A star** without showing that the
  other nights agree.
