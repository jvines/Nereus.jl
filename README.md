# Nereus.jl

tra**N**s-dimensional **E**vidence and **R**ecovery for **E**xoplanets by m**U**lti-observable **S**ampling.

Radial velocities, transits, absolute and relative astrometry,
Rossiter–McLaughlin, Doppler tomography and transit timing under a single
likelihood — with trans-dimensional model selection over the number of
planets, the noise description, and which observable constrains which
companion.

## What it is for

Most orbit codes fit a model you specify. Nereus samples over the model:

- **How many companions** — the classic trans-dimensional problem.
- **Which noise description** — a menu of correlated-noise models
  (celerite rotation and SHO, Matérn, harmonic, activity GP, linear
  decorrelation) competing under role-based exclusion groups, so two
  descriptions of the same stellar signal cannot both be active.
- **Which observable constrains which companion** — per planet. A companion
  the RV establishes beyond doubt can be astrometrically undetected; forcing
  the coupling on makes every fit report an inclination, and therefore a
  "dynamical mass", whether or not the astrometry constrained one.

Eighteen samplers share one target interface: parallel tempering (plain,
ensemble, Hamiltonian, normalising-flow coupled), nested sampling (static,
importance, dynamic), RJMCMC, MoMS, SMC, MAP, Pathfinder, OFTI — with
thermodynamic-integration, stepping-stone and hybrid evidence estimators.

## Install

Julia:

```julia
using Pkg; Pkg.add(url="https://github.com/jvines/Nereus.jl")
```

Python, with no Julia installed and none required — a prebuilt runtime is
fetched on first use:

```sh
pip install astronereus
python -c "import astronereus; astronereus.install()"
```

The Python distribution is called `astronereus` because `nereus` on PyPI
belongs to an unrelated geophysics package. The Julia package, this repository
and the runtime cache all keep the name Nereus.

## Quick start

```julia
using Nereus

data   = Data(t_rv=t, rv=rv, rv_err=err, rv_inst=inst)
menu   = default_noise_menu(data)
params = Params(max_kplanet = 3, planet_modes = [RV_ONLY, RV_ONLY, RV_ONLY],
                instruments = InstrumentConfig(rv = ["HARPS"]), data = data,
                noise_models = menu.noise_models, transdim_noise = true)

td = TransDimConfig(max_kplanet = 3, noise = true,
                    toggleable = menu.toggleable,
                    noise_exclusion_groups = menu.exclusion_groups)

result = sample_transdim_ptemcee(NereusTarget(params, data), data; td = td)
model_probabilities(result.chains)      # P(N_planets | data)
```

`run_job(config)` is the batch entry point, driven by a JSON/dict schema
documented in `docs/JOB_CONFIG.md`.

## Layout

The include list in `src/Nereus.jl` is a topologically sorted dependency
graph — Julia requires definition before use, so reading it top to bottom is
reading the architecture.

| directory | what it holds |
|---|---|
| `src/samplers/` | the eighteen samplers and the evidence estimators |
| `src/transdim/` | masks, birth/death proposals, swaps, annealed births |
| `src/noise/` | the noise menu and its likelihoods |
| `src/astrometry/` | IAD, HGCA, GOST, Gaia DR4 epoch data, solution ladder |
| `src/diagnostics/` | PPC, LOO, detection limits, fit-health guards |
| `test/validation/` | falsifiable science gates, separate from unit tests |

## Testing

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

`test/` holds unit and correctness tests. `test/validation/` holds the science
gates — occupancy against independently computed evidence, recovery of injected
signals, and the reference-free symmetry benchmarks. Those are deliberately
outside the unit suite: they take minutes to hours and answer "is the inference
right", not "does the code run".

## Vendored dependencies

`vendor/NestedSamplers/` is an in-tree copy of a fork carrying `Bounds.MLFriends`
and `Proposals.HSlice`. Upstream is unmaintained; see `vendor/NestedSamplers/VENDORED.md`
for the exact commit and rationale. MIT, copyright retained.

## Licence

MIT — see `LICENSE`.
