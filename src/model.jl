# Model configuration: Params, parametrization, planet data sources,
# and the derived parameter layout.
#
# Architectural notes (see plan doc `soft-herding-torvalds.md`):
#
# 1. Planets are described by a `PlanetDataSources` set, not a fixed
#    enum, so RV / PM / (future) TTV / astrometry compose cleanly.
#    Factory constants `RV_ONLY`, `PM_ONLY`, `RVPM` preserve the
#    ergonomic short names the user was already writing.
#
# 2. `PlanetBlock` is now an abstract type with one concrete subtype
#    per meaningful slot shape (`RVOnlyBlock`, `PMOnlyBlock`,
#    `RVPMBlock`). No more `0`-sentinel fields — absent slots are
#    absent from the type. Hot-path accessors dispatch on the subtype
#    and Union-split cleanly (only 3 concrete types).
#
# 3. `Params` is split into `ParamsConfig` (user-facing inputs,
#    immutable intent) and `ParamsLayout` (derived indices, precomputed
#    parallel prior vector, frozen/unfrozen partition). `Params` is a
#    thin wrapper over both. This makes layout logic testable in
#    isolation and enforces the config/layout separation in the type.
#
# 4. `ParamsLayout.unfrozen_priors::Vector{PriorSpec}` is a parallel
#    vector to `unfrozen_idx`. The hot-path `log_prior` iterates over
#    both and eliminates the former dict lookup, which was boxing every
#    `PriorSpec{D}` into an `Any` and killing autodiff specialization.
#
# 5. `n_p_idx::Int` makes the slot-1 convention a typed field. Callers
#    read `layout.n_p_idx` instead of hardcoding `1`.
#
# Scope note: supports planets + RV/PM instruments + shared `rho_s`.
# GP, ARMA, activity indicators, LD groups, acceleration, trends — all
# still deferred. Hooks in `SystemicIndices` are ready for them.

# =====================================================================
# PlanetDataSources — composable set of data types a planet contributes
# =====================================================================

# Canonical source symbols. Future extensions (TTV, astrometry, RV line
# profile variation, …) are added as new symbols without any file-wide
# refactor.
const RV_SOURCE     = :RV
const PM_SOURCE     = :PM
const AS_SOURCE     = :AS    # absolute + relative astrometry (HGCA, RelAstromData, …)
const RM_SOURCE     = :RM    # Rossiter-McLaughlin in-transit RV anomaly (Hirano+ 2011)
const RM_R_SOURCE   = :RM_R  # Reloaded Rossiter-McLaughlin (Cegla+ 2016)
const RM_A_SOURCE   = :RM_A  # ARoME CCF-Gaussian-fit RM (Boué+ 2013, Eq. 15)
const GD_SOURCE     = :GD    # gravity-darkened transit (von Zeipel; Barnes 2009)
const TTV_SOURCE    = :TTV   # per-transit free time offsets (TTV-A)
const TTV_NB_SOURCE = :TTV_NB # N-body-predicted TTVs (TTV-C, TTVFaster)
"""
    SB_SOURCE

Data-source flag for an **SB2 double-lined binary orbit**: one Keplerian
carrying two RV semi-amplitudes (`K_A` for the primary, anti-phase `K_B`
for the secondary) plus luminous-companion photocenter astrometry.

Kept distinct from a planet on purpose -- period ordering, stability
checks and RM counting all skip an SB2 component. Use it through
[`BINARY`](@ref) or [`BINARY_RV`](@ref).
"""
const SB_SOURCE     = :SB    # SB2 binary orbit — two RV amplitudes (K_A primary,
                             # anti-phase K_B secondary) + luminous-companion
                             # photocenter astrometry. Distinct from a planet so
                             # period-ordering/stability/RM-counting skip it.

"""
    PlanetDataSources(sources::Symbol...)

Set of data types a planet contributes to. Replaces the old
`PlanetMode` enum — composable, so adding TTV / astrometry later is
additive with no refactor.

Use the factory constants `RV_ONLY`, `PM_ONLY`, `RVPM` for the common
cases, or construct directly:

```
PlanetDataSources(:RV, :PM)     # ≡ RVPM
PlanetDataSources(:RV)          # ≡ RV_ONLY
```

Membership tests use `in`:

```
:RV in planet_modes[k]          # true if planet k has RV data
```
"""
struct PlanetDataSources
    sources::Set{Symbol}
end

PlanetDataSources(sources::Symbol...) = PlanetDataSources(Set{Symbol}(sources))

Base.in(s::Symbol, p::PlanetDataSources) = s in p.sources
Base.:(==)(a::PlanetDataSources, b::PlanetDataSources) = a.sources == b.sources
Base.hash(p::PlanetDataSources, h::UInt) = hash(p.sources, h)
Base.length(p::PlanetDataSources) = length(p.sources)
Base.isempty(p::PlanetDataSources) = isempty(p.sources)

function Base.show(io::IO, p::PlanetDataSources)
    s = sort!(collect(p.sources))
    print(io, "PlanetDataSources(", join((":" * string(x) for x in s), ", "), ")")
end

# Factory constants. These are PlanetDataSources, NOT enum values, but
# they satisfy the same user-facing role — callers that wrote
# `planet_modes=[RVPM, RV_ONLY]` keep working unchanged.
"""
    RV_ONLY :: PlanetDataSources

Planet model driven by:

  - radial velocities

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RV_ONLY = PlanetDataSources(RV_SOURCE)
"""
    PM_ONLY :: PlanetDataSources

Planet model driven by:

  - transit photometry

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const PM_ONLY = PlanetDataSources(PM_SOURCE)
"""
    RVPM :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPM    = PlanetDataSources(RV_SOURCE, PM_SOURCE)
"""
    RVAS :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - astrometry (absolute + relative)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVAS    = PlanetDataSources(RV_SOURCE, AS_SOURCE)
"""
    RVPMAS :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - astrometry (absolute + relative)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPMAS  = PlanetDataSources(RV_SOURCE, PM_SOURCE, AS_SOURCE)
# RM variants — Rossiter-McLaughlin requires both RV and PM (need a
# transit window + in-transit RV observations).
"""
    RVPM_RM :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - the Rossiter-McLaughlin anomaly (Hirano+ 2011)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPM_RM   = PlanetDataSources(RV_SOURCE, PM_SOURCE, RM_SOURCE)
"""
    RVPMAS_RM :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - astrometry (absolute + relative)
  - the Rossiter-McLaughlin anomaly (Hirano+ 2011)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPMAS_RM = PlanetDataSources(RV_SOURCE, PM_SOURCE, AS_SOURCE, RM_SOURCE)
# Reloaded RM (Cegla+ 2016) — same lambda/v_sin_i_star slots as Hirano,
# but the in-transit RV anomaly is computed by intensity-weighted
# integration over the planet's sky-plane disk (more accurate at high
# rr or high impact parameter; reduces to Hirano in the small-planet
# limit).
"""
    RVPM_RM_R :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - the reloaded Rossiter-McLaughlin anomaly (Cegla+ 2016)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPM_RM_R   = PlanetDataSources(RV_SOURCE, PM_SOURCE, RM_R_SOURCE)
"""
    RVPMAS_RM_R :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - astrometry (absolute + relative)
  - the reloaded Rossiter-McLaughlin anomaly (Cegla+ 2016)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPMAS_RM_R = PlanetDataSources(RV_SOURCE, PM_SOURCE, AS_SOURCE, RM_R_SOURCE)

# ARoME CCF formulation (Boué+ 2013 Eq. 15) — the RM anomaly a Gaussian fit to a
# CCF actually measures. Adds two system-level line-width parameters,
# `sigma_ccf` (out-of-transit CCF Gaussian dispersion) and `beta_p` (sub-planet
# line width). Prefer this over RVPM_RM whenever v·sin i ≳ β_p.
"""
    RVPM_RM_A :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - the ARoME CCF Rossiter-McLaughlin anomaly (Boue+ 2013)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPM_RM_A   = PlanetDataSources(RV_SOURCE, PM_SOURCE, RM_A_SOURCE)
"""
    RVPMAS_RM_A :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - astrometry (absolute + relative)
  - the ARoME CCF Rossiter-McLaughlin anomaly (Boue+ 2013)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPMAS_RM_A = PlanetDataSources(RV_SOURCE, PM_SOURCE, AS_SOURCE, RM_A_SOURCE)

# Gravity-darkened transit (von Zeipel 1924; Barnes 2009). A rotating star is
# oblate and hotter at the pole, so a transiting planet crosses regions of
# different surface brightness and the light curve becomes ASYMMETRIC. The
# asymmetry depends on lambda AND on the stellar inclination i_star SEPARATELY --
# which is the point: an RM or a Doppler tomogram gives only lambda, so the TRUE
# 3-D obliquity psi is otherwise unreachable without a rotation period.
#
# Reuses the RM slots: `lambda_k$k` and `v_sin_i_star` mean the same thing here,
# so RM (or tomography) and gravity darkening constrain the SAME lambda jointly.
# Adds exactly one parameter, `i_star`, because the rotation rate is not free:
#     v_eq = v_sin_i_star / sin(i_star)  ->  omega = v_eq / v_crit
# Low i_star therefore means a fast rotator seen pole-on and STRONG darkening,
# which is what makes this a measurement of i_star rather than a degeneracy.
"""
    RVPM_GD :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - a gravity-darkened transit (Barnes 2009)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPM_GD    = PlanetDataSources(RV_SOURCE, PM_SOURCE, GD_SOURCE)
"""
    RVPM_RM_GD :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - the Rossiter-McLaughlin anomaly (Hirano+ 2011)
  - a gravity-darkened transit (Barnes 2009)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPM_RM_GD = PlanetDataSources(RV_SOURCE, PM_SOURCE, RM_SOURCE, GD_SOURCE)
"""
    PM_GD :: PlanetDataSources

Planet model driven by:

  - transit photometry
  - a gravity-darkened transit (Barnes 2009)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const PM_GD      = PlanetDataSources(PM_SOURCE, GD_SOURCE)
# TTV variants — per-transit free time offsets. Requires PM (need
# transit observations to fit transit-time offsets against).
"""
    RVPM_TTV :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - per-transit free timing offsets (TTV-A)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPM_TTV     = PlanetDataSources(RV_SOURCE, PM_SOURCE, TTV_SOURCE)
"""
    RVPMAS_TTV :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - astrometry (absolute + relative)
  - per-transit free timing offsets (TTV-A)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPMAS_TTV   = PlanetDataSources(RV_SOURCE, PM_SOURCE, AS_SOURCE, TTV_SOURCE)
"""
    RVPM_RM_TTV :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - the Rossiter-McLaughlin anomaly (Hirano+ 2011)
  - per-transit free timing offsets (TTV-A)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPM_RM_TTV  = PlanetDataSources(RV_SOURCE, PM_SOURCE, RM_SOURCE, TTV_SOURCE)
"""
    PM_TTV :: PlanetDataSources

Planet model driven by:

  - transit photometry
  - per-transit free timing offsets (TTV-A)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const PM_TTV       = PlanetDataSources(PM_SOURCE, TTV_SOURCE)
# TTV-C (N-body via TTVFaster). Same shape as TTV_SOURCE downstream,
# but δts are PREDICTED at each likelihood eval from the mutual
# gravitational interaction (Agol & Deck 2016, 1st order in e).
# No per-transit free parameters — `ttv_n_transits[k]` only sizes the
# predicted-Tc grid. Needs ≥ 2 planets carrying :TTV_NB for any signal;
# otherwise the prediction collapses to zero.
"""
    RVPM_TTV_NB :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - N-body-predicted TTVs (TTV-C, TTVFaster)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPM_TTV_NB    = PlanetDataSources(RV_SOURCE, PM_SOURCE, TTV_NB_SOURCE)
"""
    RVPMAS_TTV_NB :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - transit photometry
  - astrometry (absolute + relative)
  - N-body-predicted TTVs (TTV-C, TTVFaster)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const RVPMAS_TTV_NB  = PlanetDataSources(RV_SOURCE, PM_SOURCE, AS_SOURCE, TTV_NB_SOURCE)
"""
    PM_TTV_NB :: PlanetDataSources

Planet model driven by:

  - transit photometry
  - N-body-predicted TTVs (TTV-C, TTVFaster)

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const PM_TTV_NB      = PlanetDataSources(PM_SOURCE, TTV_NB_SOURCE)

# SB2 double-lined binary orbit (companion star). RV in BOTH components + the
# photocenter astrometry; no transit (the binary doesn't eclipse — only the
# circumprimary planet transits, as its own RVPM*/RM block).
"""
    BINARY :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - astrometry (absolute + relative)
  - an SB2 double-lined binary orbit

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const BINARY    = PlanetDataSources(RV_SOURCE, AS_SOURCE, SB_SOURCE)
"""
    BINARY_RV :: PlanetDataSources

Planet model driven by:

  - radial velocities
  - an SB2 double-lined binary orbit

Pass through `planet_modes=` to state which observables constrain a
given planet; the set decides which parameter block is built for it.
"""
const BINARY_RV = PlanetDataSources(RV_SOURCE, SB_SOURCE)   # RV-only SB2 (no astrometry)

"""
    has_rv(modes::PlanetDataSources) -> Bool
    has_pm(modes::PlanetDataSources) -> Bool
    has_as(modes::PlanetDataSources) -> Bool

Convenience predicates. Equivalent to `:RV in modes` / `:PM in modes`
/ `:AS in modes`.
"""
@inline has_rv(modes::PlanetDataSources) = RV_SOURCE in modes
"""
    has_pm(modes::PlanetDataSources) -> Bool

Whether this planet's data-source set includes transit photometry.

See [`PlanetDataSources`](@ref) for the set itself and the named
combinations (`RVPM`, `RVPMAS_RM`, ...) that build one.
"""
@inline has_pm(modes::PlanetDataSources) = PM_SOURCE in modes
"""
    has_as(modes::PlanetDataSources) -> Bool

Whether this planet's data-source set includes astrometry.

See [`PlanetDataSources`](@ref) for the set itself and the named
combinations (`RVPM`, `RVPMAS_RM`, ...) that build one.
"""
@inline has_as(modes::PlanetDataSources) = AS_SOURCE in modes
"""
    has_rm(modes::PlanetDataSources) -> Bool

Whether this planet's data-source set includes the Hirano+ 2011 Rossiter-McLaughlin anomaly.

See [`PlanetDataSources`](@ref) for the set itself and the named
combinations (`RVPM`, `RVPMAS_RM`, ...) that build one.
"""
@inline has_rm(modes::PlanetDataSources) = RM_SOURCE in modes
"""
    has_rm_r(modes::PlanetDataSources) -> Bool

Whether this planet's data-source set includes the reloaded (Cegla+ 2016) Rossiter-McLaughlin anomaly.

See [`PlanetDataSources`](@ref) for the set itself and the named
combinations (`RVPM`, `RVPMAS_RM`, ...) that build one.
"""
@inline has_rm_r(modes::PlanetDataSources) = RM_R_SOURCE in modes
"""
    has_rm_a(modes::PlanetDataSources) -> Bool

Whether this planet's data-source set includes the ARoME (Boue+ 2013) Rossiter-McLaughlin anomaly.

See [`PlanetDataSources`](@ref) for the set itself and the named
combinations (`RVPM`, `RVPMAS_RM`, ...) that build one.
"""
@inline has_rm_a(modes::PlanetDataSources) = RM_A_SOURCE in modes
"""
    has_any_rm(modes::PlanetDataSources) -> Bool

Whether ANY Rossiter-McLaughlin formulation is active -- Hirano,
reloaded (Cegla+ 2016) or ARoME (Boue+ 2013).

Use this rather than testing the three separately when the question is
"does this planet carry RM parameters at all".
"""
@inline has_any_rm(modes::PlanetDataSources) =
    has_rm(modes) || has_rm_r(modes) || has_rm_a(modes)
@inline has_gd(modes::PlanetDataSources) = GD_SOURCE in modes
"""
    has_ttv(modes::PlanetDataSources) -> Bool

Whether this planet's data-source set includes per-transit free timing offsets (TTV-A).

See [`PlanetDataSources`](@ref) for the set itself and the named
combinations (`RVPM`, `RVPMAS_RM`, ...) that build one.
"""
@inline has_ttv(modes::PlanetDataSources) = TTV_SOURCE in modes
"""
    has_ttv_nb(modes::PlanetDataSources) -> Bool

Whether this planet's data-source set includes N-body-predicted TTVs (TTV-C).

See [`PlanetDataSources`](@ref) for the set itself and the named
combinations (`RVPM`, `RVPMAS_RM`, ...) that build one.
"""
@inline has_ttv_nb(modes::PlanetDataSources) = TTV_NB_SOURCE in modes
"""
    has_sb(modes::PlanetDataSources) -> Bool

Whether this planet's data-source set includes an SB2 double-lined binary orbit.

See [`PlanetDataSources`](@ref) for the set itself and the named
combinations (`RVPM`, `RVPMAS_RM`, ...) that build one.
"""
@inline has_sb(modes::PlanetDataSources) = SB_SOURCE in modes   # SB2 binary orbit

# =====================================================================
# ParametrizationConfig
# =====================================================================

"""
    ParametrizationConfig(; ew=:sesinw, time=:Mo, geom=:b_rr, use_rho_s=false,
                            mass=:K_driven)

Choices for how orbital parameters are parametrized in the sampler.

# Fields
- `ew::Symbol`        : one of `:sesinw`, `:esinw`, `:ew`
- `time::Symbol`      : one of `:Mo`, `:Tp`, `:Tc`
- `geom::Symbol`      : one of `:b_rr`, `:r1r2`
- `use_rho_s::Bool`   : if `true`, a shared stellar density parameter
                         replaces per-planet `a_kN`
- `mass::Symbol`      : one of `:K_driven` (default — sample RV semi-
                         amplitude K, derive M_sec via mass function),
                         `:M_sec_driven` (sample M_sec directly, derive
                         K — recommended for joint RV + astrom), or
                         `:a_driven` (sample (a, M_sec), derive P via
                         Kepler's 3rd law and K via mass function —
                         orvara's convention, best for visual orbits).

| mode             | `block.P` slot | `block.K` slot |
|------------------|----------------|----------------|
| `:K_driven`      | P (days)       | K (m/s)        |
| `:M_sec_driven`  | P (days)       | M_sec (M_sun)  |
| `:a_driven`      | a (AU)         | M_sec (M_sun)  |

The accessors `planet_P`, `planet_K`, `planet_M_sec`, `planet_a` decode
the slots regardless of mode and derive missing quantities on the fly
(Kepler's third law and the RV mass function). All decoders are
ForwardDiff-clean.
"""
struct ParametrizationConfig
    ew::Symbol
    time::Symbol
    geom::Symbol
    use_rho_s::Bool
    mass::Symbol
    obs_prior::Bool   # O'Neil 2019 observation-based prior on (P, e, t_p)
    # Analytically marginalize the per-instrument RV systemic offset γ
    # (orvara's approach). Because γ enters the RV model linearly
    # (pred = γ + Keplerian + trend), the likelihood is Gaussian in γ
    # and the per-instrument γ can be integrated in closed form at FIXED
    # jitter σ. When `true`, the `gamma_*` slots are frozen out of the
    # sampled set (fewer free dims, cleaner mixing/evidence) and the RV
    # likelihood uses the γ-marginalized path. White-noise only — see
    # `_rv_ll_gamma_marginalized`.
    marginalize_gamma::Bool
end

function ParametrizationConfig(; ew::Symbol=:sesinw, time::Symbol=:Mo,
                                geom::Symbol=:b_rr, use_rho_s::Bool=false,
                                mass::Symbol=:K_driven,
                                obs_prior::Bool=false,
                                marginalize_gamma::Bool=false)
    ew in (:sesinw, :esinw, :ew) ||
        throw(ArgumentError("ew must be :sesinw, :esinw, or :ew; got :$ew"))
    time in (:Mo, :Tp, :Tc) ||
        throw(ArgumentError("time must be :Mo, :Tp, or :Tc; got :$time"))
    geom in (:b_rr, :r1r2) ||
        throw(ArgumentError("geom must be :b_rr or :r1r2; got :$geom"))
    mass in (:K_driven, :M_sec_driven, :a_driven) ||
        throw(ArgumentError("mass must be :K_driven, :M_sec_driven, or :a_driven; got :$mass"))
    return ParametrizationConfig(ew, time, geom, use_rho_s, mass, obs_prior,
                                 marginalize_gamma)
end

# =====================================================================
# InstrumentConfig
# =====================================================================

"""
    InstrumentConfig(rv_names, pm_names)
    InstrumentConfig(; rv=String[], pm=String[])

Names of RV and photometric instruments. Instrument indices throughout
the code refer to positions in these vectors (1-based). The names are
used only for building parameter names like `gamma_HARPS` — the
physics operates purely on integer indices.
"""
struct InstrumentConfig
    rv_names::Vector{String}
    pm_names::Vector{String}
end

InstrumentConfig(; rv=String[], pm=String[]) = InstrumentConfig(rv, pm)

"""
    n_rv_instruments(ic::InstrumentConfig) -> Int

Number of distinct RV instruments in the configuration -- the count of
per-instrument offset/jitter slot pairs the layout allocates.
"""
n_rv_instruments(ic::InstrumentConfig) = length(ic.rv_names)
"""
    n_pm_instruments(ic::InstrumentConfig) -> Int

Number of distinct photometry instruments in the configuration -- the
count of per-instrument baseline/jitter/dilution slots allocated.
"""
n_pm_instruments(ic::InstrumentConfig) = length(ic.pm_names)

# =====================================================================
# PlanetBlock — abstract + concrete per-mode subtypes
# =====================================================================
#
# The old one-size `PlanetBlock` used `0` as a sentinel for "absent slot"
# (e.g., `K = 0` for PM-only planets). The mode-specific subtypes below
# encode presence/absence in the type itself, so:
#
#   - Hot-path accessors dispatch on subtype — zero runtime cost after
#     Union-splitting (3 concrete types, well within the compiler's
#     Union-splitting budget).
#   - Reading an absent slot is a type error caught at compile time
#     (`(::PMOnlyBlock).K` doesn't exist), not a runtime `== 0` check.
#   - New subtypes for new data sources (TTV, astrometry) drop in
#     additively.

"""
    PlanetBlock

Abstract supertype for per-planet parameter index blocks. Concrete
subtypes are `RVOnlyBlock`, `PMOnlyBlock`, and `RVPMBlock`. Each stores
integer indices into `Theta.values` for the parameters that exist
under that planet's data-source set. Absent parameters are absent from
the type, not set to a sentinel.

# Field vocabulary

All subtypes share a common vocabulary for field names. The fields are
parametrization-agnostic *slot indices* — they tell you **where** to
find each parameter in `theta.values`, not **what** that parameter
means. The meaning is resolved by reading `params.config.parametrization`
and applying the appropriate conversion (see the hot-path accessors
`planet_P`, `planet_K`, `planet_e_w`, `planet_time_anchor`,
`planet_b_rr` in `parameters.jl`).

- **`P`**  — orbital period slot. Always present on every block subtype.
- **`K`**  — RV semi-amplitude slot. Present on `RVOnlyBlock` and
             `RVPMBlock`; absent on `PMOnlyBlock` (the planet does not
             contribute to RV, so there is nothing to constrain K).
- **`e1`** — **first eccentricity slot.** Its meaning depends on
             `parametrization.ew`:
             - `:sesinw` → `√e sin ω`
             - `:esinw`  → `e sin ω`
             - `:ew`     → `e` (eccentricity directly)
- **`e2`** — **second eccentricity slot.** Paired with `e1`:
             - `:sesinw` → `√e cos ω`
             - `:esinw`  → `e cos ω`
             - `:ew`     → `ω` (argument of periastron)
- **`t`**  — **time-anchor slot.** Its meaning depends on
             `parametrization.time`:
             - `:Mo` → `M₀`, mean anomaly at the reference epoch (rad)
             - `:Tp` → time of periastron passage (same units as data)
             - `:Tc` → time of transit center
- **`b`**  — **transit geometry slot 1.** Present on `PMOnlyBlock` and
             `RVPMBlock`; absent on `RVOnlyBlock` (no transit to fit).
             Meaning depends on `parametrization.geom`:
             - `:b_rr` → `b` (impact parameter, 0 = central transit)
             - `:r1r2` → `r₁` (Espinoza 2018 reparametrization)
- **`r`**  — **transit geometry slot 2.** Paired with `b`:
             - `:b_rr` → `Rp/Rs` (radius ratio)
             - `:r1r2` → `r₂` (Espinoza 2018)

The hot-path accessors decode these slots automatically. Use them
instead of reading fields directly unless you know exactly what
parametrization is in effect.
"""
abstract type PlanetBlock end

"""
    RVOnlyBlock(P, K, e1, e2, t)

Index block for a planet that contributes only to the RV likelihood.
No transit geometry slots — this planet has no photometric data.

# Fields
- `P::Int`  : orbital period slot
- `K::Int`  : RV semi-amplitude slot
- `e1::Int` : first eccentricity slot (`√e sin ω`, `e sin ω`, or `e`)
- `e2::Int` : second eccentricity slot (`√e cos ω`, `e cos ω`, or `ω`)
- `t::Int`  : time anchor slot (`M₀`, `Tp`, or `Tc`)

See `PlanetBlock` for the full field vocabulary.
"""
struct RVOnlyBlock <: PlanetBlock
    P::Int
    K::Int
    e1::Int
    e2::Int
    t::Int
end

"""
    PMOnlyBlock(P, e1, e2, t, b, r)

Index block for a planet that contributes only to the photometric
(transit) likelihood. No RV semi-amplitude — this planet has no radial
velocity data.

# Fields
- `P::Int`  : orbital period slot
- `e1::Int` : first eccentricity slot (`√e sin ω`, `e sin ω`, or `e`)
- `e2::Int` : second eccentricity slot (`√e cos ω`, `e cos ω`, or `ω`)
- `t::Int`  : time anchor slot (`M₀`, `Tp`, or `Tc`)
- `b::Int`  : transit geometry slot 1 (`b` or `r₁`)
- `r::Int`  : transit geometry slot 2 (`Rp/Rs` or `r₂`)

See `PlanetBlock` for the full field vocabulary.
"""
struct PMOnlyBlock <: PlanetBlock
    P::Int
    e1::Int
    e2::Int
    t::Int
    b::Int
    r::Int
end

"""
    RVPMBlock(P, K, e1, e2, t, b, r)

Index block for a planet that contributes to both RV and transit
likelihoods. Has all seven slots.

# Fields
- `P::Int`  : orbital period slot
- `K::Int`  : RV semi-amplitude slot
- `e1::Int` : first eccentricity slot (`√e sin ω`, `e sin ω`, or `e`)
- `e2::Int` : second eccentricity slot (`√e cos ω`, `e cos ω`, or `ω`)
- `t::Int`  : time anchor slot (`M₀`, `Tp`, or `Tc`)
- `b::Int`  : transit geometry slot 1 (`b` or `r₁`)
- `r::Int`  : transit geometry slot 2 (`Rp/Rs` or `r₂`)

See `PlanetBlock` for the full field vocabulary.
"""
struct RVPMBlock <: PlanetBlock
    P::Int
    K::Int
    e1::Int
    e2::Int
    t::Int
    b::Int
    r::Int
end

"""
    RVASBlock(P, K, e1, e2, t, inc, Omega)

Index block for an RV + astrometry planet (no transit). Same as
`RVOnlyBlock` plus two astrometry slots:

- `inc::Int`   : orbit inclination slot (rad)
- `Omega::Int` : longitude of ascending node slot (rad)

Inclination is sampled with a `Sine(0, π)` prior (isotropic in
cos i) by default, matching EMPEROR II. Omega is uniform on [0, 2π).

The companion mass `M_sec` is *derived* from `(K, P, e, sin i, M_pri)`
via the standard RV mass function — there is no `M_sec` slot in this
block. (The `:M_sec_driven` parametrization, where M_sec is sampled
directly and K is derived, is a future extension that introduces a
separate block subtype.)
"""
struct RVASBlock <: PlanetBlock
    P::Int
    K::Int
    e1::Int
    e2::Int
    t::Int
    inc::Int
    Omega::Int
end

"""
    RVPMASBlock(P, K, e1, e2, t, b, r, Omega)

Index block for the full triple-data planet (RV + transit + astrometry).
Same as `RVPMBlock` plus the astrometric `Omega` slot.

Inclination is **not** sampled for this block: it is derived from the
sampled transit impact parameter `b` via

    cos i = b · (1 + e sin ω) / ((a/R★) · (1 − e²))

with `a/R★` obtained from `rho_s` (when
`parametrization.use_rho_s == true`) or from `M_s, R_s, P` via
Kepler's third law otherwise. This eliminates the spurious likelihood
ridge that arises when transit and astrometry pull `i` to inconsistent
values through independently sampled `b` and `inc`.

The inclination is read out via `planet_inc(theta, k)` — the accessor
returns the derived value transparently.
"""
struct RVPMASBlock <: PlanetBlock
    P::Int
    K::Int
    e1::Int
    e2::Int
    t::Int
    b::Int
    r::Int
    Omega::Int
end

"""
    SB2Block(P, KA, KB, e1, e2, t, inc, Omega)

Index block for the **SB2 double-lined binary orbit** — a single
Keplerian carrying *two* RV semi-amplitudes: `K_A` (primary, applied to
`rv_comp == 1` points) and `K_B` (secondary, applied *anti-phase* to
`rv_comp == 2` points). The mass ratio `q = K_A / K_B = M_B / M_A` is a
*derived* quantity, never a sampled slot.

# Fields
- `P::Int`     : orbital period slot
- `KA::Int`    : primary RV semi-amplitude `K_A` slot (m/s)
- `KB::Int`    : secondary RV semi-amplitude `K_B` slot (m/s)
- `e1::Int`    : first eccentricity slot (`√e sin ω`, `e sin ω`, or `e`)
- `e2::Int`    : second eccentricity slot (`√e cos ω`, `e cos ω`, or `ω`)
- `t::Int`     : time anchor slot (`M₀`, `Tp`, or `Tc`)
- `inc::Int`   : binary inclination slot (rad); `0` when RV-only (no astrometry)
- `Omega::Int` : longitude of ascending node slot (rad); `0` when RV-only

For the RV-only binary (`BINARY_RV = (:RV, :SB)`) `inc` and `Omega` are
`0` sentinels — no astrometry geometry is allocated. For the full
`BINARY = (:RV, :AS, :SB)` they index the two absolute-astrometry
orientation slots, reused from the RVAS companion-orbit geometry. A
luminous secondary's photocenter is handled separately via the
system-level `f_light` term (see astrometry projection).
"""
struct SB2Block <: PlanetBlock
    P::Int
    KA::Int
    KB::Int
    e1::Int
    e2::Int
    t::Int
    inc::Int
    Omega::Int
end

# Predicate helpers for hot-path dispatch.
"""
    has_K(block) -> Bool

Whether this planet's parameter block carries an RV semi-amplitude `K`.

False for a photometry-only block, which constrains geometry but no RV
reflex. Dispatches on the block type, so it answers what was actually
allocated rather than what the data-source set requested.
"""
@inline has_K(::RVOnlyBlock)  = true
@inline has_K(::RVPMBlock)    = true
@inline has_K(::PMOnlyBlock)  = false
@inline has_K(::RVASBlock)    = true
@inline has_K(::RVPMASBlock)  = true
@inline has_K(::SB2Block)     = true

"""
    has_geometry(block) -> Bool

Whether this planet's parameter block carries transit geometry --
radius ratio, impact parameter, and the rest of the transit shape.

True only for blocks fed by photometry; an RV-only or RV+astrometry
block has an orbit but no transit, so there is no geometry to fit.
"""
@inline has_geometry(::PMOnlyBlock) = true
@inline has_geometry(::RVPMBlock)   = true
@inline has_geometry(::RVOnlyBlock) = false
@inline has_geometry(::RVASBlock)   = false
@inline has_geometry(::RVPMASBlock) = true
@inline has_geometry(::SB2Block)    = false

"""
    is_sb2(block::PlanetBlock) -> Bool

True for the SB2 binary orbit block. Used to (a) gate the K_A/K_B
two-channel RV contribution and (b) EXCLUDE the binary from the
planet-only hard priors (period ordering, dynamical stability, the
per-planet eccentricity prior) — a stellar binary is not
interchangeable with the circumprimary planets.
"""
@inline is_sb2(::PlanetBlock) = false
@inline is_sb2(::SB2Block)    = true

"""
    has_AS(block::PlanetBlock) -> Bool

True if the planet contributes astrometric data (relative or absolute).
Used by `astrom_log_likelihood` to skip purely-RV/PM companions.
"""
@inline has_AS(::RVOnlyBlock)  = false
@inline has_AS(::PMOnlyBlock)  = false
@inline has_AS(::RVPMBlock)    = false
@inline has_AS(::RVASBlock)    = true
@inline has_AS(::RVPMASBlock)  = true
# Value-based: the RV-only binary (BINARY_RV) carries `Omega == 0`
# sentinels and contributes no astrometry; the full BINARY allocates
# real inc/Omega slots (> 0) and does.
@inline has_AS(b::SB2Block)    = b.Omega != 0

# =====================================================================
# SystemicIndices — hooks for instrument/noise parameter indices
# =====================================================================

"""
    SystemicIndices

Precomputed indices for systemic (non-planet) parameters. Built once
at layout construction; read in the hot path to access gamma/sigma/
offset/jitter/dilution/rho_s slots.

# Fields
- `rv_gamma::Vector{Int}`  : one entry per RV instrument
- `rv_sigma::Vector{Int}`  : one entry per RV instrument
- `pm_offset::Vector{Int}` : one entry per PM instrument
- `pm_jitter::Vector{Int}` : one entry per PM instrument
- `pm_dilution::Vector{Int}` : one entry per PM instrument
- `rho_s::Int`             : shared stellar-density slot, or `0` if unused

Future extensions (GP kernels, ARMA, activity, LD groups, acceleration,
trends) add further fields here.
"""
struct SystemicIndices
    rv_gamma::Vector{Int}
    rv_sigma::Vector{Int}
    pm_offset::Vector{Int}
    pm_jitter::Vector{Int}
    pm_dilution::Vector{Int}
    pm_trend::Vector{Vector{Int}}  # per PM instrument: [c1_slot, …, c_order_slot]
                                    # for the photometric baseline polynomial;
                                    # empty when phot_trend_order == 0
    ld_q1::Vector{Int}     # one per PM instrument (shared groups → same slot)
    ld_q2::Vector{Int}     # one per PM instrument (shared groups → same slot)
    rho_s::Int      # 0 if not used
    dvdt::Int       # 0 if not used (linear RV trend)
    d2vdt2::Int     # 0 if not used (quadratic RV trend)
    plx::Int        # 0 if no astrometry; system parallax (mas)
    m_pri::Int      # 0 if no astrometry; primary (host) mass (M_sun)
    v_sin_i_star::Int  # 0 if no planet has RM or GD; stellar V·sin(i_*) [m/s]
    i_star::Int        # 0 unless a planet uses :GD; stellar inclination [rad].
                       # Shared with nothing else. Together with v_sin_i_star it
                       # fixes v_eq = v_sin_i_star/sin(i_star) and hence the
                       # oblateness, so the gravity-darkening amplitude is NOT a
                       # free scale -- it is set by i_star.
    gd_beta::Vector{Int}  # one per PM instrument; all 0 unless a planet uses :GD.
                       # EFFECTIVE gravity-darkening exponent for that BANDPASS, in
                       # I ∝ g^{4β_eff}. Per band because gravity darkening is
                       # chromatic: the surface temperature contrast is bolometric
                       # (T ∝ g^β, β = 0.25 radiative), but a filter samples
                       # dlnB_λ/dlnT, which for a 7400 K star is ~2.7 in the TESS
                       # band and ~3.6 in g' -- not 4. Treating every band as
                       # bolometric overstates the TESS signal by ~33% and would
                       # bias i_star high. `gd_beta_band` computes the right value;
                       # the default is bolometric (0.25), i.e. the old behaviour.
                       # FIXED, not sampled: it is a property of the envelope and
                       # the filter, and a free one soaks up the i_star signal.
    # Doppler tomography, one entry per residual map in `data.tomo`. Empty
    # when no tomography is supplied.
    #
    # alpha is the shadow amplitude and is PER NIGHT, not shared: it absorbs
    # the per-instrument CCF normalisation, which differs between a DRS CCF and
    # a mask-CCF and cannot be assumed common. sigma_line is the local line
    # width, also per night, because it depends on the instrumental profile.
    #
    # Both are nuisance parameters of the MEASUREMENT, not of the star — the
    # obliquity lives in `lambda_k` and the rotation in `v_sin_i_star`, shared
    # with the RM velocities. That sharing is the whole point of a joint fit.
    tomo_alpha::Vector{Int}
    tomo_sigma_line::Vector{Int}
    # Velocity-axis correlation length of the residual map [km/s]. INSTRUMENTAL:
    # the CCF is oversampled and smoothed by the instrumental profile, so
    # neighbouring velocity bins are correlated whatever the star is doing.
    # Not a competing physical hypothesis, which is why it stays a fitted
    # nuisance while the TIME axis is the menu's to choose.
    tomo_ell_v::Vector{Int}
    rm_sigma_ccf::Int  # 0 unless a planet uses the ARoME CCF kernel (:RM_A);
                       # dispersion of the Gaussian fitted to the OUT-OF-TRANSIT
                       # CCF [m/s]. Measure it from the data — do not derive it
                       # from v·sin i (Boué+2013 Eq. 15).
    rm_beta_p::Int     # 0 unless :RM_A; sub-planet (local) line-profile
                       # dispersion [m/s] = intrinsic width ⊗ instrumental profile
    f_light::Int       # 0 unless an SB2 binary has astrometry; secondary
                       # light fraction L_B/(L_A+L_B) in the ASTROMETRIC band,
                       # for the luminous-companion photocenter correction.
end

# =====================================================================
# ExternalPrior — priors on derived (non-sampled) quantities
# =====================================================================

"""
    ExternalPrior

A prior on a derived quantity (not a direct sampled parameter). Evaluated
in the likelihood after orbital parameters are decoded.

Implements the "external prior" concept from priors.jl design note #6:
the prior acts on a value computed from sampled parameters (e.g.,
eccentricity derived from √e sin ω / √e cos ω).

# Supported quantities
- `:ecc`   — eccentricity (per-planet, evaluated for each active planet)
- `:rho_s` — stellar density (global, evaluated once)

# Fields
- `quantity::Symbol`  — which derived quantity
- `prior::PriorSpec`  — the prior distribution (reuses existing prior system)
- `per_planet::Bool`  — `true` for per-planet quantities (:ecc), `false` for global (:rho_s)

# Examples
```julia
ExternalPrior(:ecc,   BetaPrior(0.867, 3.03), true)   # Kipping 2013
ExternalPrior(:rho_s, NormalPrior(1.4, 0.1),  false)   # stellar density constraint
```
"""
struct ExternalPrior
    quantity::Symbol
    prior::PriorSpec
    per_planet::Bool
end

const _VALID_EXTERNAL_QUANTITIES = (:ecc, :rho_s)

# =====================================================================
# ParamsConfig — immutable user-facing inputs
# =====================================================================

"""
    ParamsConfig

User-facing configuration for a Nereus model. Constructed once by
the user (typically via the `Params(; …)` constructor, which wraps
this) and treated as immutable afterward.

# Fields
- `max_kplanet::Int`
- `parametrization::ParametrizationConfig`
- `planet_modes::Vector{PlanetDataSources}`
- `instruments::InstrumentConfig`
- `priors::Dict{String, PriorSpec}`
- `stability::Symbol` — `:none`, `:amd`, or `:gladman`; hard prior rejecting unstable multi-planet configs
- `M_s::Float64` — stellar mass (M_sun); required for stability checks and derived quantities; `NaN` = unknown
- `R_s::Float64` — stellar radius (R_sun); required for transit a/R* when not using rho_s; `NaN` = unknown
- `external_priors::Vector{ExternalPrior}` — priors on derived quantities (ecc, rho_s, etc.)
- `sharing::Dict{Symbol, Vector{Vector{String}}}` — instrument parameter sharing groups

The `priors` dict is the friendly user API. The hot-path sampler code
does not read from it — see `ParamsLayout.unfrozen_priors`.

The `sharing` dict maps parameter categories to instrument groups:
```julia
sharing = Dict(
    :sigma => [["HARPS_DRS", "HARPS_SERVAL"]],  # share jitter
    :ld    => [["TESS", "K2"]],                  # share LD
)
```
Valid categories: `:sigma`, `:gamma`, `:pm_jitter`, `:pm_offset`, `:pm_dilution`, `:ld`.
Instruments not in any group keep their own parameter slot.
"""
struct ParamsConfig
    max_kplanet::Int
    parametrization::ParametrizationConfig
    planet_modes::Vector{PlanetDataSources}
    instruments::InstrumentConfig
    priors::Dict{String, PriorSpec}
    trend_order::Int  # 0=none, 1=linear (dvdt), 2=+quadratic (d2vdt2) — RV γ trend
    phot_trend_order::Int  # 0=none, 1=linear, 2=quadratic — per-instrument
                           # photometric baseline polynomial (in normalized time)
    noise_models::Vector{NoiseModel}
    stability::Symbol  # :none, :amd, :gladman
    M_s::Float64       # stellar mass (M_sun); NaN = unknown (stability skipped)
    R_s::Float64       # stellar radius (R_sun); NaN = unknown
    external_priors::Vector{ExternalPrior}
    sharing::Dict{Symbol, Vector{Vector{String}}}
    # Per-planet TTV transit count: planet_k → N (number of free
    # per-transit time offsets to allocate). Only consulted for planets
    # with :TTV in their PlanetDataSources. Defaults to empty.
    ttv_n_transits::Dict{Int, Int}
    # TTV-NB backend selector — `:ttvfaster` (perturbative 1st-order
    # series, default) or `:nbody` (full ODE integration via
    # NbodyGradient.jl). Only consulted when `:TTV_NB` planet modes are
    # active. Use `:nbody` for high-mass / high-e / strongly-resonant
    # systems where TTVFaster breaks down.
    ttv_backend::Symbol
end

const _VALID_SHARING_CATEGORIES = (
    :sigma, :gamma,                          # RV
    :pm_jitter, :pm_offset, :pm_dilution,    # PM
    :ld,                                      # LD (q1/q2 pair)
)

const _RV_SHARING_CATEGORIES = (:sigma, :gamma)
const _PM_SHARING_CATEGORIES = (:pm_jitter, :pm_offset, :pm_dilution, :ld)

# =====================================================================
# ParamsLayout — derived layout, built once from ParamsConfig
# =====================================================================

"""
    ParamsLayout

Derived parameter layout: the full ordered name list, the name→index
map, per-planet block indices, per-instrument systemic indices, the
frozen/unfrozen partition, and a parallel vector of priors for the
unfrozen slots.

Built by `_build_layout(config::ParamsConfig)` — pure function of the
config, testable in isolation.

# Fields
- `names::Vector{String}`            : ordered list of all theta slots
- `name_to_idx::Dict{String,Int}`    : name → slot index
- `n_total::Int`                     : length of the theta vector
- `n_p_idx::Int`                     : slot index of the `n_p` parameter
- `unfrozen_idx::Vector{Int}`        : slot indices that are sampled
- `unfrozen_names::Vector{String}`   : names of unfrozen params (parallel)
- `unfrozen_priors::Vector{PriorSpec}` : priors of unfrozen params (parallel)
- `frozen_idx::Vector{Int}`          : slot indices held fixed
- `frozen_values::Vector{Float64}`   : fixed values (parallel to frozen_idx)
- `planet_blocks::Vector{PlanetBlock}` : per-planet block, concrete subtype
- `systemic::SystemicIndices`        : instrument + rho_s indices
"""
struct ParamsLayout
    names::Vector{String}
    name_to_idx::Dict{String, Int}
    n_total::Int
    n_p_idx::Int

    unfrozen_idx::Vector{Int}
    unfrozen_names::Vector{String}
    unfrozen_priors::Vector{PriorSpec}
    frozen_idx::Vector{Int}
    frozen_values::Vector{Float64}

    planet_blocks::Vector{PlanetBlock}
    systemic::SystemicIndices
    packed_priors::PackedPriors   # type-stable prior eval for Enzyme
end

# =====================================================================
# Params — thin wrapper over config + layout
# =====================================================================

"""
    Params

Aggregate model object: immutable `ParamsConfig` plus derived
`ParamsLayout`. All hot-path code reads from `params.layout`.
The config is available for serialization, introspection, and
reconstruction.

Construct via the keyword constructor `Params(; priors, max_kplanet,
…)` — it validates inputs, builds the layout, and returns a fully
populated `Params`.
"""
struct Params
    config::ParamsConfig
    layout::ParamsLayout
end

# ---------------------------------------------------------------------
# Layout construction (pure function of the config)
# ---------------------------------------------------------------------

"""
    _planet_param_names(pc, modes, k) -> Vector{String}

Build the ordered list of parameter names for planet `k` given the
parametrization and data-source set. Called by `_build_layout` — the
block subtype is chosen from `modes` after the names are appended to
the global list so the block's field indices can be read back from the
offset.

Internal helper.
"""
function _planet_param_names(pc::ParametrizationConfig,
                              modes::PlanetDataSources, k::Int;
                              n_ttv::Int = 0)
    suffix = "_k$k"
    names = String[]

    # First slot: P (days) under :K_driven and :M_sec_driven, or a (AU)
    # under :a_driven. Stored in `block.P` regardless.
    if pc.mass === :a_driven
        push!(names, "a" * suffix)
    else
        push!(names, "P" * suffix)
    end

    # Second slot (if RV planet): K (m/s) under :K_driven, M_sec (M_sun)
    # under :M_sec_driven and :a_driven. Stored in `block.K` regardless.
    #
    # SB2 binary: TWO amplitude slots K_A, K_B (m/s) — always sampled
    # directly (K_driven-style), independent of the global mass
    # parametrization, since the two amplitudes ARE the observables that
    # pin the mass ratio q = K_A/K_B.
    if has_sb(modes)
        push!(names, "K_A" * suffix)
        push!(names, "K_B" * suffix)
    elseif has_rv(modes)
        if pc.mass === :K_driven
            push!(names, "K" * suffix)
        else
            push!(names, "M_sec" * suffix)
        end
    end

    # Eccentricity pair (always two slots — parametrization names them).
    if pc.ew === :sesinw
        push!(names, "sesinw" * suffix)
        push!(names, "secosw" * suffix)
    elseif pc.ew === :esinw
        push!(names, "esinw" * suffix)
        push!(names, "ecosw" * suffix)
    else  # :ew
        push!(names, "ecc" * suffix)
        push!(names, "w" * suffix)
    end

    # Time anchor (always one slot).
    push!(names, String(pc.time) * suffix)

    # Transit geometry, only if the planet contributes to PM.
    if has_pm(modes)
        if pc.geom === :b_rr
            push!(names, "b" * suffix)
            push!(names, "rr" * suffix)
        else  # :r1r2
            push!(names, "r1" * suffix)
            push!(names, "r2" * suffix)
        end
    end

    # Astrometry: orbit inclination and longitude of ascending node.
    # For RVPMAS planets (RV + PM + AS) inclination is derived from the
    # transit impact parameter `b` — only `Omega` enters the layout.
    if has_as(modes)
        if !(has_rv(modes) && has_pm(modes))
            push!(names, "inc" * suffix)
        end
        push!(names, "Omega" * suffix)
    end

    # Sky-projected obliquity λ, per planet. Needed by the Rossiter-McLaughlin
    # models (Hirano+ 2011 / Cegla+ 2016 Reloaded / ARoME) AND by gravity
    # darkening, which measures the same angle from the light-curve asymmetry --
    # so a :GD planet must get the slot even with no RM data, or the two could
    # not be fitted against a shared λ.
    if has_any_rm(modes) || has_gd(modes)
        push!(names, "lambda" * suffix)
    end

    # TTV: per-transit free time offsets. Number of slots set by
    # `n_ttv` (allocated from `config.ttv_n_transits[k]`).
    if has_ttv(modes) && n_ttv > 0
        for i in 1:n_ttv
            push!(names, "ttv" * suffix * "_t$i")
        end
    end

    return names
end

"""
    _make_block(modes, start) -> PlanetBlock

Build the concrete `PlanetBlock` subtype for a planet given its data
sources and the starting slot offset (the index of its P slot in the
global names vector). Slot order matches `_planet_param_names`.

Internal helper. Throws if `modes` is empty (a planet with no data is
a user error).
"""
function _make_block(modes::PlanetDataSources, start::Int)
    isempty(modes) && throw(ArgumentError(
        "planet has empty PlanetDataSources — at least one source required"))

    # SB2 binary FIRST — must precede the rv/pm/as chain below. A
    # BINARY = (:RV, :AS, :SB) set has rv && as, which would otherwise
    # fall through to the `rv && as` branch and build an RVASBlock that
    # silently misreads the K_B slot as `inc` (wrong physics, no error).
    if has_sb(modes)
        has_rv(modes) || throw(ArgumentError(
            "SB2 (:SB) requires RV — got modes=$(modes)"))
        has_pm(modes) && throw(ArgumentError(
            "SB2 + transiting-binary (:PM on the binary) is not supported — " *
            "the circumprimary transiting planet is a SEPARATE block; got modes=$(modes)"))
        if has_as(modes)
            # BINARY: [P, K_A, K_B, e1, e2, t, inc, Omega]
            return SB2Block(start, start+1, start+2, start+3, start+4,
                            start+5, start+6, start+7)
        else
            # BINARY_RV: [P, K_A, K_B, e1, e2, t] — no astrometry slots
            # (inc/Omega = 0 sentinels).
            return SB2Block(start, start+1, start+2, start+3, start+4,
                            start+5, 0, 0)
        end
    end

    rv = has_rv(modes)
    pm = has_pm(modes)
    as = has_as(modes)

    # Phase 1 supports the four configurations the user can produce in
    # Nereus today: RV-only, PM-only, RV+PM, RV+AS, RV+PM+AS. Pure
    # AS-only and PM+AS configurations are uncommon for an RV-first
    # fitter — guard against them with a clear error.
    if as && !rv
        throw(ArgumentError(
            "Phase 1 astrometry requires RV — got modes=$(modes). " *
            "Pure AS-only or PM+AS support is a future extension."))
    end

    if rv && pm && as
        # RVPMAS: [P, K, e1, e2, t, b, r, Omega]
        # Inclination is *derived* from b (transit impact parameter), not
        # sampled — see RVPMASBlock docstring for the relation.
        return RVPMASBlock(start, start+1, start+2, start+3, start+4,
                           start+5, start+6, start+7)
    elseif rv && as
        # RVAS: [P, K, e1, e2, t, inc, Omega]
        return RVASBlock(start, start+1, start+2, start+3, start+4,
                         start+5, start+6)
    elseif rv && pm
        # RVPM: [P, K, e1, e2, t, b, r]
        return RVPMBlock(start, start + 1, start + 2, start + 3,
                         start + 4, start + 5, start + 6)
    elseif rv
        # RV only: [P, K, e1, e2, t]
        return RVOnlyBlock(start, start + 1, start + 2, start + 3, start + 4)
    else  # pm only
        # PM only: [P, e1, e2, t, b, r]
        return PMOnlyBlock(start, start + 1, start + 2, start + 3,
                           start + 4, start + 5)
    end
end

"""
    _instrument_groups(ins_names, category_groups) -> Vector{Tuple{String, Vector{Int}}}

Resolve sharing groups for a parameter category. Returns `(group_label, [ins_indices])`
pairs covering all instruments exactly once.

When `category_groups` is `nothing` (no sharing), each instrument is its own group.
Otherwise, instruments listed in the same inner vector share one parameter slot.
Unlisted instruments get singleton groups.

Group labels join instrument names with `+`: `["HARPS_DRS", "SERVAL"]` → `"HARPS_DRS+SERVAL"`.
"""
function _instrument_groups(ins_names::Vector{String},
                             category_groups::Union{Nothing, Vector{Vector{String}}})
    n = length(ins_names)
    if category_groups === nothing || isempty(category_groups)
        return [(ins_names[i], [i]) for i in 1:n]
    end

    name_to_idx = Dict(ins_names[i] => i for i in 1:n)
    assigned = falses(n)
    result = Tuple{String, Vector{Int}}[]

    for group in category_groups
        indices = Int[]
        for gname in group
            idx = get(name_to_idx, gname, nothing)
            idx === nothing && throw(ArgumentError(
                "unknown instrument '$gname' in sharing group"))
            assigned[idx] && throw(ArgumentError(
                "instrument '$gname' appears in multiple sharing groups " *
                "for the same category"))
            assigned[idx] = true
            push!(indices, idx)
        end
        push!(result, (join(group, "+"), indices))
    end

    # Unassigned instruments become singleton groups
    for i in 1:n
        assigned[i] && continue
        push!(result, (ins_names[i], [i]))
    end

    return result
end


"""
    _build_layout(config::ParamsConfig) -> ParamsLayout

Compute the full parameter layout from an immutable `ParamsConfig`.
Pure — same input always produces the same output. Validates that
every slot in the layout has a prior, no extra prior keys are present
(typo detection), and every prior passes `validate_physical`.

This function is the one-stop layout builder. All the indexing
invariants (slot 1 = `n_p`, planet blocks in order, systemic after)
live here.
"""
function _build_layout(config::ParamsConfig; data::Union{Data, Nothing}=nothing)
    names = String[]

    # --- Slot 1: n_p -------------------------------------------------
    push!(names, "n_p")
    n_p_idx = 1  # typed invariant; consumers read this field

    # --- Planet blocks -----------------------------------------------
    planet_blocks = Vector{PlanetBlock}(undef, config.max_kplanet)
    for k in 1:config.max_kplanet
        modes = config.planet_modes[k]
        n_ttv_k = has_ttv(modes) ? get(config.ttv_n_transits, k, 0) : 0
        pnames = _planet_param_names(config.parametrization, modes, k;
                                       n_ttv = n_ttv_k)
        start = length(names) + 1
        append!(names, pnames)
        planet_blocks[k] = _make_block(modes, start)
    end

    # --- Systemic block (with sharing support) -------------------------
    sh = config.sharing
    rv_names = config.instruments.rv_names
    n_rv = length(rv_names)

    rv_gamma = zeros(Int, n_rv)
    gamma_groups = _instrument_groups(rv_names, get(sh, :gamma, nothing))
    for (label, idxs) in gamma_groups
        push!(names, "gamma_$label")
        slot = length(names)
        for idx in idxs
            rv_gamma[idx] = slot
        end
    end

    rv_sigma = zeros(Int, n_rv)
    sigma_groups = _instrument_groups(rv_names, get(sh, :sigma, nothing))
    for (label, idxs) in sigma_groups
        push!(names, "sigma_$label")
        slot = length(names)
        for idx in idxs
            rv_sigma[idx] = slot
        end
    end

    pm_names = config.instruments.pm_names
    n_pm = length(pm_names)

    pm_offset = zeros(Int, n_pm)
    offset_groups = _instrument_groups(pm_names, get(sh, :pm_offset, nothing))
    for (label, idxs) in offset_groups
        push!(names, "offset_$label")
        slot = length(names)
        for idx in idxs
            pm_offset[idx] = slot
        end
    end

    pm_jitter = zeros(Int, n_pm)
    jitter_groups = _instrument_groups(pm_names, get(sh, :pm_jitter, nothing))
    for (label, idxs) in jitter_groups
        push!(names, "jitter_$label")
        slot = length(names)
        for idx in idxs
            pm_jitter[idx] = slot
        end
    end

    pm_dilution = zeros(Int, n_pm)
    dilution_groups = _instrument_groups(pm_names, get(sh, :pm_dilution, nothing))
    for (label, idxs) in dilution_groups
        push!(names, "dilution_$label")
        slot = length(names)
        for idx in idxs
            pm_dilution[idx] = slot
        end
    end

    # Per-instrument photometric baseline trend: a polynomial continuum
    # 1 + offset + Σ_{p=1}^order c_p·x^p (x = normalized time, see transit
    # likelihood). Coefficients share the OFFSET grouping (one continuum per
    # instrument group). order==0 ⇒ pm_trend stays empty ⇒ identity continuum.
    pm_trend = [Int[] for _ in 1:n_pm]
    if config.phot_trend_order >= 1
        for (label, idxs) in offset_groups
            slots = Int[]
            for p in 1:config.phot_trend_order
                push!(names, "phot_c$(p)_$label")
                push!(slots, length(names))
                if !haskey(config.priors, "phot_c$(p)_$label")
                    config.priors["phot_c$(p)_$label"] =
                        NormalPrior(0.0, 0.05, -0.5, 0.5)
                end
            end
            for idx in idxs
                pm_trend[idx] = slots
            end
        end
    end

    # Limb darkening: one (q1, q2) pair per LD group
    ld_q1 = zeros(Int, n_pm)
    ld_q2 = zeros(Int, n_pm)
    has_transit_planets = any(has_pm(m) for m in config.planet_modes)
    if has_transit_planets
        ld_groups = _instrument_groups(pm_names, get(sh, :ld, nothing))
        for (label, idxs) in ld_groups
            push!(names, "q1_$label")
            q1_slot = length(names)
            push!(names, "q2_$label")
            q2_slot = length(names)
            for idx in idxs
                ld_q1[idx] = q1_slot
                ld_q2[idx] = q2_slot
            end
            if !haskey(config.priors, "q1_$label")
                config.priors["q1_$label"] = UniformPrior(0.0, 1.0)
            end
            if !haskey(config.priors, "q2_$label")
                config.priors["q2_$label"] = UniformPrior(0.0, 1.0)
            end
        end
    end

    rho_s_idx = 0
    if config.parametrization.use_rho_s
        push!(names, "rho_s")
        rho_s_idx = length(names)
    end

    dvdt_idx = 0
    d2vdt2_idx = 0
    if config.trend_order >= 1
        push!(names, "dvdt")
        dvdt_idx = length(names)
    end
    if config.trend_order >= 2
        push!(names, "d2vdt2")
        d2vdt2_idx = length(names)
    end

    # Parallax + primary-mass slots, only when at least one planet has AS.
    plx_idx = 0
    m_pri_idx = 0
    if any(has_as(m) for m in config.planet_modes)
        push!(names, "plx")
        plx_idx = length(names)
        push!(names, "M_pri")
        m_pri_idx = length(names)
    end

    # Rossiter-McLaughlin: system-level V·sin(i_*), shared across any
    # planet with RM enabled. Added once.
    v_sin_i_star_idx = 0
    if any(has_any_rm(m) || has_gd(m) for m in config.planet_modes)
        push!(names, "v_sin_i_star")
        v_sin_i_star_idx = length(names)
    end

    # Gravity darkening: one stellar inclination for the system, and one
    # effective exponent per photometric band (the effect is chromatic).
    i_star_idx = 0
    gd_beta_idx = zeros(Int, n_pm)
    if any(has_gd(m) for m in config.planet_modes)
        push!(names, "i_star")
        i_star_idx = length(names)
        for (ix, nm_pm) in enumerate(pm_names)
            push!(names, "gd_beta_$nm_pm")
            gd_beta_idx[ix] = length(names)
            if !haskey(config.priors, "gd_beta_$nm_pm")
                config.priors["gd_beta_$nm_pm"] = FixedPrior(0.25)
            end
        end
    end

    # ARoME CCF kernel (Boué+2013 Eq. 15) needs the two line widths that decide
    # how much of the flux-weighted anomaly a Gaussian CCF fit actually recovers.
    # Only allocated when some planet asks for :RM_A.
    rm_sigma_ccf_idx = 0
    rm_beta_p_idx = 0
    if any(has_rm_a(m) for m in config.planet_modes)
        push!(names, "sigma_ccf")
        rm_sigma_ccf_idx = length(names)
        push!(names, "beta_p")
        rm_beta_p_idx = length(names)
    end

    # Doppler tomography: per-night shadow amplitude and local line width.
    # Named by the night's tag so a fit with CORALIE + HARPS + PFS maps is
    # readable in the chain rather than tomo_alpha_1..3.
    n_tomo = data === nothing ? 0 : length(data.tomo)
    tomo_alpha_idx = zeros(Int, n_tomo)
    tomo_sigma_line_idx = zeros(Int, n_tomo)
    tomo_ell_v_idx = zeros(Int, n_tomo)
    for j in 1:n_tomo
        tag = data.tomo[j].tag
        push!(names, "tomo_alpha_$(tag)");      tomo_alpha_idx[j] = length(names)
        push!(names, "tomo_sigma_line_$(tag)"); tomo_sigma_line_idx[j] = length(names)
        push!(names, "tomo_ell_v_$(tag)");      tomo_ell_v_idx[j] = length(names)
    end

    # Luminous-companion photocenter: system-level secondary light
    # fraction f_light = L_B/(L_A+L_B) in the ASTROMETRIC band. Only when
    # an SB2 binary ALSO has astrometry (dark-companion astrometry needs
    # no correction; RV-only SB2 has no photocenter).
    f_light_idx = 0
    if any(m -> has_sb(m) && has_as(m), config.planet_modes)
        push!(names, "f_light")
        f_light_idx = length(names)
    end

    systemic = SystemicIndices(rv_gamma, rv_sigma,
                               pm_offset, pm_jitter, pm_dilution, pm_trend,
                               ld_q1, ld_q2,
                               rho_s_idx, dvdt_idx, d2vdt2_idx,
                               plx_idx, m_pri_idx, v_sin_i_star_idx, i_star_idx,
                               gd_beta_idx,
                               tomo_alpha_idx, tomo_sigma_line_idx, tomo_ell_v_idx,
                               rm_sigma_ccf_idx, rm_beta_p_idx, f_light_idx)

    # --- Noise model parameters ---------------------------------------
    for nm in config.noise_models
        nm_names = noise_param_names(nm, config.instruments; data=data)
        append!(names, nm_names)
    end

    # Layout names must be globally unique. A duplicate silently collapses in
    # `name_to_idx` below (overwrite) and only surfaces much later as a NetCDF
    # `defVar` -42 ("name in use") when writing chains.nc — by then the whole
    # fit is lost. Fail here with an actionable message. Realistic source: two
    # noise models generating the same param (e.g. two ActivityJitter with the
    # same `indicator` on one instrument).
    if length(unique(names)) != length(names)
        seen = Set{String}(); dups = String[]
        for n in names
            (n in seen) ? push!(dups, n) : push!(seen, n)
        end
        throw(ArgumentError(
            "Duplicate parameter name(s) in the model layout: " *
            join(sort(unique(dups)), ", ") * ". Two noise models likely " *
            "generate the same parameter — e.g. two ActivityJitter models " *
            "with the same `indicator` on the same instrument. Give them " *
            "distinct indicators, scope them to different instruments, or " *
            "drop the duplicate."))
    end

    # --- Name → index map --------------------------------------------
    name_to_idx = Dict{String, Int}()
    for (i, n) in enumerate(names)
        name_to_idx[n] = i
    end
    n_total_ = length(names)

    # --- Validate priors cover the layout ----------------------------
    missing_priors = String[]
    for n in names
        haskey(config.priors, n) || push!(missing_priors, n)
    end
    if !isempty(missing_priors)
        throw(ArgumentError(
            "Missing priors for parameters: " * join(missing_priors, ", ")))
    end

    extra_priors = String[]
    for k in keys(config.priors)
        haskey(name_to_idx, k) || push!(extra_priors, k)
    end
    if !isempty(extra_priors)
        throw(ArgumentError(
            "Unknown parameter names in priors dict (typos?): " *
            join(extra_priors, ", ")))
    end

    # --- Physical validation for each prior --------------------------
    for (name, ps) in config.priors
        validate_physical(name, ps)
    end

    # --- Frozen / unfrozen partition + parallel priors vector --------
    unfrozen_idx    = Int[]
    unfrozen_names  = String[]
    unfrozen_priors = PriorSpec[]
    frozen_idx      = Int[]
    frozen_values   = Float64[]

    for (i, name) in enumerate(names)
        ps = config.priors[name]
        if is_fixed(ps)
            push!(frozen_idx, i)
            push!(frozen_values, fixed_value(ps))
        else
            push!(unfrozen_idx, i)
            push!(unfrozen_names, name)
            push!(unfrozen_priors, ps)
        end
    end

    # Build type-stable packed priors for Enzyme-compatible evaluation.
    packed = pack_priors(unfrozen_priors)

    return ParamsLayout(
        names, name_to_idx, n_total_, n_p_idx,
        unfrozen_idx, unfrozen_names, unfrozen_priors,
        frozen_idx, frozen_values,
        planet_blocks, systemic, packed,
    )
end

# Public Params constructor is in params_constructor.jl (loaded after
# data.jl and default_priors.jl to resolve dependencies).

# ---------------------------------------------------------------------
# Introspection helpers — all delegate to the layout
# ---------------------------------------------------------------------

"""
    n_total(params) -> Int

Total number of parameter slots (frozen + unfrozen).
"""
n_total(params::Params) = params.layout.n_total

"""
    n_unfrozen(params) -> Int

Number of parameters actively sampled.
"""
n_unfrozen(params::Params) = length(params.layout.unfrozen_idx)

"""
    n_frozen(params) -> Int

Number of parameters held fixed.
"""
n_frozen(params::Params) = length(params.layout.frozen_idx)

"""
    param_index(params, name) -> Int

Slot index of a named parameter. Throws `KeyError` if unknown.
"""
param_index(params::Params, name::AbstractString) =
    params.layout.name_to_idx[name]
