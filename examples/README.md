# Nereus examples

Four worked tutorials. Notebooks 01 and 02 are Julia tutorials on Gaia DR4
(pre-release) epoch astrometry; run them in order, the second assumes the
first. Notebooks 03 and 04 are Python tutorials (`astronereus`), described
below.

| notebook | system | channels | what it shows |
|---|---|---|---|
| `01_gaia4_astrometry_only.ipynb` | Gaia-4 b | DR4 epoch astrometry | an orbit from along-scan abscissae alone, no RVs |
| `02_hd114762_joint_rv_astrometry.ipynb` | HD 114762 b | RV + DR4 epoch astrometry | breaking the `sin i` degeneracy to get a true mass |

HD 114762 b was the first exoplanet candidate ever announced (Latham et al.
1989). Radial velocities give `M sin i ~ 11 M_Jup`; the joint fit shows the
orbit is nearly face-on and the companion is a ~0.1 M_sun star.

## Running notebooks 01 and 02

Both need only Nereus. Data is handled for you: the DR4 VOTable is fetched to
the Nereus cache on first use, and the HD 114762 radial velocities ship with the
package (`test/data/hd114762_rv.dat`, California Legacy Survey, Rosenthal et al.
2021).

To run as notebooks you need a Julia kernel:

```julia
using Pkg; Pkg.add("IJulia")
```

then `jupyter lab` from this directory.

To run them as plain scripts instead, without Jupyter:

```sh
julia --project -e 'using Pkg; Pkg.add("NBInclude")'
julia --project -t auto -e 'using NBInclude; @nbinclude("01_gaia4_astrometry_only.ipynb")'
```

## Cost

Both fit with `sample_pt_emcee` and run `n_steps` steps (3000 by default, burn-in
included; burn-in is `min(1500, n_steps ÷ 2)`), so the cost is roughly linear in
the step count. Both notebooks default to production settings and expose the step
count through an environment variable (`GAIA4_STEPS`, `HD114762_STEPS`) so you
can smoke-test deliberately rather than by accident.

**A short run does not give a usable posterior.** Short chains can leave the
inclination railing, the RV chi-squared per point far above 1 and the credible
intervals grossly asymmetric, while the headline mass still lands near the
published value. That combination is the trap: a short run that agrees with the
answer you expected is not evidence of anything.

## Figures

Every plot takes `(chains, params, data)` and a directory as `output`; Nereus
names the files and puts model figures under `models/`. The two notebooks
produce along-scan residuals, the epoch-astrometry orbit, the sky-plane orbit,
RV time series and phase folds, and the posterior corner plots (one per
companion, and one of the noise model when it has two or more parameters).

## 03 — Python tutorial (`astronereus`)

`03_python_tutorial.ipynb` covers the same science from Python, for people who do
not have Julia and do not want it. A short tour of what Nereus samples over, then
straight into two systems:

- **Gaia-4 b** — an orbit from Gaia DR4 epoch astrometry alone (824 abscissae,
  no RVs).
- **HD 114762** — the same system fitted three ways: RV only (gives M sin i),
  astrometry only (Hipparcos + Gaia DR4 as two numbered instruments), and joint.
  The joint fit is the one that breaks sin i and turns a 35-year-old minimum mass
  into an actual mass.

Runs on a stock Python install: `pip install astronereus numpy`. The runtime is
fetched on first use.

## 04 — Python tutorial: RVs, transits and astrometry

`04_tutorial_rv_transits_astrometry.ipynb` starts from installation and fits one
system per combination of channels, every fit with `PTEmcee`:

- **NGTS-6 b**: CORALIE and FEROS velocities alone (Vines et al. 2019), then
  jointly with five sectors of TESS 2-min photometry.
- **Gaia-4 b**: Gaia DR4 epoch astrometry alone, then jointly with the HPF, NEID
  and FIES velocities.
- **HD 114762 b**: velocities alone, then with DR4. The joint fit puts the orbit
  at i = 3.5° and the companion at 0.2 M_sun.

Needs `pip install "astronereus[diagnostics,lightcurves]"` (xarray to read the
chains, lightkurve for TESS) and `unzip` on the PATH for the DR4 download. A full
run takes about 15 minutes with 16 threads. `NEREUS_TUTORIAL_QUICK=1` cuts every
chain to 600 steps, to check that the notebook runs end to end; what it prints
then is not a result.
