# The astrometric solution ladder — 5p / 7p / 9p

Given epoch astrometry (Hipparcos IAD or Gaia DR4), how many parameters does a
source actually need? Five for a single star, seven if there is measurable
curvature, nine if there is a jerk — or none of them, if the reflex is a full
Keplerian.

Nereus answers that in **closed form**, with exact evidences and proper priors,
at about 22 µs per rung. No sampler, no estimator error bar.

```julia
lad = ladder_probabilities(iad)
lad.prob            # [P(5p), P(7p), P(9p)]
lad.best            # argmax
lad.chi2_reduced    # per rung
lad.adequate        # DOES ANY OF THEM FIT? — read this every time
```

---

## The model

One abscissa model with a growing number of free linear terms:

```
w_i = dra* sin(psi) + ddec cos(psi) + plx p_i                     (5p, with proper motion)
      + pmra* sin(psi) dt + pmdec cos(psi) dt
      + acc_ra* sin(psi) dt^2/2 + acc_dec cos(psi) dt^2/2         (7p)
      + jrk_ra* sin(psi) dt^3/6 + jrk_dec cos(psi) dt^3/6         (9p)
```

`w` is the along-scan abscissa (mas), `psi` the scan position angle, `p` the
along-scan parallax factor, `dt` the time from the catalogue reference epoch in
Julian years. The rungs are strictly **nested prefixes** — the first five
columns of the 9p design matrix *are* the 5p design matrix.

`astrom_design(iad, order)` builds it; `order ∈ (5, 7, 9)`. Six-parameter
(pseudo-colour) and two-parameter solutions are catalogue products that do not
arise here and are rejected by name.

Column names are in `ASTROM_PARAM_NAMES`: `dra_cosdec`, `ddec`, `parallax`,
`pmra_cosdec`, `pmdec`, `accel_ra`, `accel_dec`, `jerk_ra`, `jerk_dec`.

Because `dt` is set by the reader, **the same ladder runs unchanged on
Hipparcos IAD and on Gaia DR4 epoch data** — same `IADData` container. See
[Data models](data.md).

### The ½ and ⅙ are Taylor coefficients, not fudge factors

Expanding a bounded reflex displacement about the reference epoch,

```
x(dt) = x0 + xdot dt + xddot dt^2/2 + xdddot dt^3/6 + ...
```

Putting 1/2! and 1/3! **into the basis function** makes the fitted coefficient
the literal derivative: `accel_ra` is d²α*/dt² in mas/yr², `jerk_ra` is d³α*/dt³
in mas/yr³. Without it the coefficients are in arbitrary units and no prior on
them means anything.

Conditioning improves too, but that is a side benefit and a modest one — on
real DR4 data (N = 637, T = 4.75 yr) the order-9 condition number goes 8.8e2 →
2.5e2, a factor 3.5.

> **Comparing against published columns:** since the ⅙ is folded in, the `jerk_*`
> coefficient here **is** the third derivative. If Gaia's published
> `deriv_accel_*` columns use a different normalisation, a naive comparison is
> off by a factor of 6. Check the NSS data model first.

---

## Closed-form evidence, and why the priors must be proper

Every rung is linear-Gaussian in **all** its parameters, so its marginal
likelihood is exact. For `w = Xq + eps`, `eps ~ N(0, W⁻¹)`, prior `q ~ N(0, S)`:

```
log Z = -n/2 log(2 pi) + 1/2 log|W|
        - 1/2 (w'Ww - v'M^-1 v)
        - 1/2 log|S| - 1/2 log|M|      with M = X'WX + S^-1,  v = X'Ww
```

`-½log|S| - ½log|M|` is the Occam factor.

**This is why the ladder does not reuse `iad_log_likelihood`.** That function
marginalises the catalogue correction under a **flat improper** prior, which is
fine for orbit fitting — the resulting constant cancels between orbit samples —
and **fatal for model comparison**, because it does not cancel between models
with different column counts. Improper priors give undefined Bayes factors;
Lindley's paradox, arriving on schedule.

### `default_ladder_prior` — a prior stated in milliarcseconds

```julia
default_ladder_prior(iad, order; a0_max = 5.0, accel_yr = 10.0)
# order 5: [5.0, 5.0, 5.0, 5.0, 5.0]
# order 7: [..., 0.1, 0.1]
# order 9: [..., 0.1, 0.1, 0.03, 0.03]
```

Demand that a column's contribution equal `a0_max` at `dt = accel_yr`, and the
2 and 6 fall out as the inverses of the ½ and ⅙:

```
sigma_acc * accel_yr^2/2 = a0_max   =>   sigma_acc = 2 a0_max / accel_yr^2
sigma_jrk * accel_yr^3/6 = a0_max   =>   sigma_jrk = 6 a0_max / accel_yr^3
```

So the prior means "how curved could a real long-period companion make this",
which is the hypothesis being tested — rather than an arbitrary number attached
to an arbitrary basis.

**This is the single most consequential choice in the comparison.** Widening
the acceleration prior by a decade costs ~log(10) per added parameter — about
**4.6 nats for the two acceleration terms, a factor of 100 in odds per decade**.
Measured across five decades of prior width on the twelve DR4 pre-release
sources, one of twelve changes its preferred rung.

It does **not** bias the 5p block: default versus an effectively-flat 1e6 mas
prior shifts parallax by at most 1e-3 mas and pmra by 3e-3 mas/yr. The prior
matters only through the Occam factor on the acceleration and jerk columns.

> `default_ladder_prior` takes an `iad` argument it does not use — the widths
> are data-independent despite the signature.

---

## The API

| function | returns |
|---|---|
| `astrom_design(iad, order)` | the `n × order` design matrix |
| `astrom_solution(iad, order; prior_sigma, residual)` | `(names, q, sigma, cov, chi2_reduced, log_z, order)` |
| `astrom_logZ(iad, order; prior_sigma, residual)` | exact marginal likelihood |
| `astrom_chi2_reduced(iad, order; ...)` | χ²/(n − order) |
| `default_ladder_prior(iad, order; a0_max, accel_yr)` | per-column prior σ |
| `ladder_probabilities(iad; orders, prior, model_prior, residual)` | `(orders, log_z, prob, best, chi2_reduced, adequate)` |

`astrom_solution` returns the **complete** posterior from one Cholesky —
mean = MAP = global maximum, plus the full covariance. There is no optimiser,
no starting guess and no convergence to check. All the multimodality that makes
astrometry hard lives in the *orbital* rung, not here.

```
Gaia-4, order 7:
  dra_cosdec   =     +0.02075 ± 0.00552      pmdec       =    +17.94173 ± 0.00303
  ddec         =     +0.05294 ± 0.00748      accel_ra    =     -0.01910 ± 0.00435
  parallax     =    +13.60576 ± 0.00646      accel_dec   =     -0.04149 ± 0.00484
  pmra_cosdec  =    -75.54930 ± 0.00266      chi2_reduced = 3.394   log Z = -259.86
```

---

## Read `adequate` and `chi2_reduced`, every single time

`ladder_probabilities` normalises over the rungs **you supply**. It therefore
**cannot tell you whether any of them fits.** `adequate` is
`minimum(chi2_reduced) < 2.0`.

The example that makes this stick — Gaia BH3 on real DR4 data:

```
P(5p) = 0.000   P(7p) = 0.000   P(9p) = 1.000
log Z:  -1,839,105.7  ->  -495,854.5  ->  -334,302.6
chi2/dof:      6652.7        1788.3        1140.1      <-- adequate = FALSE
```

A crushing 9p "detection" with a 1.5-million-nat evidence improvement, and
χ²/dof = 1140. The ladder is faithfully picking the least hopeless of three
hopeless models, because a 33 M☉ companion produces an orbit no polynomial in
time describes.

**Never read `best` without `prob`, either.** Source 60730287810150016 reports
`best = 9p` with `prob = [0.31, 0.34, 0.34]` — three near-identical evidences
where the argmax is noise.

And the converse: source 4181040337841125632 has
`accel_ra = -0.0211 ± 0.0086` — a 2.1σ per-component "detection" — and the
ladder still says 5p at P = 0.89. That is the Occam factor doing its job, in
contrast to a hard frequentist significance threshold.

---

## Which regime gives which rung

With `T` the mission baseline (~5 yr for DR4), for a photocentre reflex:

- **P ≲ T** — not a polynomial over the window; every rung has χ²/dof ≫ 1 →
  an **orbital** solution is required.
- **T ≲ P ≲ 3T** — cubic term detectable → **9p**.
- **P ≈ 3T** — quadratic only → **7p**.
- **P ≳ 5T** — curvature below the noise, reflex absorbed into position and
  proper motion → **5p**, and the companion is invisible to astrometry alone.

The 7p/9p boundary follows from the basis: at |dt| = T/2 the jerk-to-
acceleration ratio is πT/(3P), so **9p is chosen precisely as you approach the
orbital regime** — which is why 9p sources are the ones most likely to deserve
a Keplerian.

The upper boundary is signal-to-noise dependent, not purely P/T: a 20 mas
reflex is still 7p at P = 100 yr.

Measured per-source acceleration precision on real DR4 pre-release data:
**σ_acc ≈ 0.005–0.01 mas/yr²** for well-observed sources.

---

## Interpreting an acceleration

A 7p solution gives (acc_α*, acc_δ) and nothing else about the companion. What
it measures is `GM_c/r²` — one number, two unknowns:

```
|accel| [mas/yr^2] = 4 pi^2 * plx[mas] * M_c / (M_tot^(2/3) * P^(4/3))
```

so the locus is **M_c ∝ P^(4/3)**. The same measured acceleration is a Saturn
at 10 yr and a 0.15 M☉ M dwarf at 1000 yr. A 7p solution supports "there is a
companion with M_c/r² = X", **not** "there is a planet". Any single mass quoted
from an acceleration imported a period from somewhere else — say which.

A **9p jerk is qualitatively stronger**, not merely bigger: jerk/accel ~ 2π/P
gives an independent handle on the period, hence on the mass.

Nereus has no acceleration-to-mass function; that relation is arithmetic you do
yourself.

---

## Is the leftover curvature Keplerian or generic?

Both `astrom_logZ` and `ladder_probabilities` take a `residual` vector that
replaces the abscissae. Subtract a Keplerian along-scan model and re-run the
ladder on what is left:

```julia
# per-epoch reflex of the fitted orbit, projected onto each scan direction
orb, M_sec = Nereus._planet_orbit(theta, 1, M_pri, plx, data.t_ref)
dη = [ (dra, ddec = star_reflex_offset(orb, iad.t[j], M_sec);
        along_scan_projection(dra, ddec, iad.psi[j])) for j in eachindex(iad.t) ]

before = ladder_probabilities(iad)
after  = ladder_probabilities(iad; residual = iad.abscissa .- dη)
```

On Gaia-4 that gives:

```
raw abscissae : P = [0.000, 0.014, 0.986], best 9p, chi2/dof = [3.50, 3.39, 3.38], adequate = FALSE
after Kepler  : P = [0.993, 0.006, 0.001], best 5p, chi2/dof = [1.41, 1.41, 1.41], adequate = TRUE
delta logZ(5p) = +858.0 nats
```

"9p, and none of them fit" becomes "5p, P = 0.993, adequate". The orbit
likelihood independently gives +857.9 nats for the same comparison —
**agreement to 0.1 nat by a completely different route**, one a closed-form
Gaussian evidence and one an MCMC-marginalised likelihood. That is a real
correctness check, and it is a self-contained argument that the curvature is
Keplerian rather than generic polynomial drift.

This is a hook, not a workflow — there is no wrapper that does it for you.

---

## Boundaries

- **No joint orbit + acceleration fit.** `iad_log_likelihood` always
  marginalises exactly five nuisance parameters, never seven or nine.
  "Keplerian plus a long-period companion's acceleration" is not expressible.
- **No reader for a published 7p/9p catalogue row.** You can *fit* one from
  epoch data, but no likelihood term consumes a published (acc_ra, acc_dec)
  plus covariance as a constraint. The nearest path is the long-baseline
  proper-motion-difference channel (HGCA / G23H), which encodes
  acceleration-like information over 26 yr rather than within the mission.
- **Not wired into `run_job`.** The ladder is a library API only.
- **Rank-deficient input fails quietly here.** With a proper prior,
  `M = X'WX + S⁻¹` is positive definite even when `X` has rank 2, so an
  `IADData` built without `parallax_factor`/`pm_factor` produces parameters
  sitting exactly at the prior with `prob ≈ [0.33, 0.33, 0.33]`. **That
  uniform-thirds signature is the tell.**
