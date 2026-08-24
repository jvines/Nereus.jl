# Nereus examples

Two worked tutorials on Gaia DR4 (pre-release) epoch astrometry. Run them in
order — the second assumes the first.

| notebook | system | channels | what it shows |
|---|---|---|---|
| `01_gaia4_astrometry_only.ipynb` | Gaia-4 b | DR4 epoch astrometry | an orbit from along-scan abscissae alone, no RVs |
| `02_hd114762_joint_rv_astrometry.ipynb` | HD 114762 b | RV + DR4 epoch astrometry | breaking the `sin i` degeneracy to get a true mass |

HD 114762 b was the first exoplanet candidate ever announced (Latham et al.
1989). Radial velocities give `M sin i ~ 11 M_Jup`; the joint fit shows the
orbit is nearly face-on and the companion is a ~0.1 M_sun star.

## Running them

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

Both fit with parallel tempering, where `n_rounds` doubles the sample count each
round — so the cost roughly doubles per round too.

| rounds | Gaia-4 | use |
|---|---|---|
| 3–4 | ~1 min | smoke test: does it run |
| 12 | ~1 h | a result you would quote |

**The short settings do not give a usable posterior.** On HD 114762 at 4 rounds
the inclination rails toward zero, the RV chi-squared per point sits near 59,
and the credible intervals come out grossly asymmetric — while the headline mass
still lands near the published value. That combination is the trap: a short run
that agrees with the answer you expected is not evidence of anything. Both
notebooks default to production settings and expose the round count through an
environment variable (`GAIA4_ROUNDS`, `HD114762_ROUNDS`) so you can smoke-test
deliberately rather than by accident.

## Figures

Every plot takes `(chains, params, data)` and a directory as `output`; Nereus
names the files and puts model figures under `models/`. The two notebooks
produce along-scan residuals, the epoch-astrometry orbit, the sky-plane orbit,
RV time series and phase folds, and the posterior corner plot.
