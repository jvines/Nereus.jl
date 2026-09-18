# Prior distributions for Nereus.
#
# Design principles (hard rules, not negotiable):
#
#   1. EVERY PRIOR STORES EXPLICIT (lo, hi) BOUNDS.
#      `PriorSpec` always carries `(lo, hi)` fields so downstream code can
#      query the hard support without introspecting wrapper types like
#      `Truncated{Normal{Float64}}`. For genuinely-unbounded priors
#      (e.g. a plain Normal) the bounds are `(-Inf, Inf)`; the caller
#      must handle infinite endpoints (clip u before `quantile`, etc.).
#
#      Bounded forms of Normal are built with `NormalPrior(μ, σ, lo, hi)`
#      which uses `truncated(Normal, lo, hi)` internally. Unbounded form
#      `NormalPrior(μ, σ)` is permitted and returns a raw Normal with
#      `(-Inf, Inf)` bounds.
#
#   2. BUILT ON Distributions.jl.
#      The actual math (logpdf, quantile, cdf, rand) is delegated to
#      Distributions.jl wherever possible. Nereus wraps with a thin
#      `PriorSpec{D}` struct that adds explicit bounds storage and the
#      `is_fixed` marker. Only genuinely non-standard distributions
#      (`ModJeffreys`, `Fixed`) are implemented in this file.
#
#   3. CONFIGURABLE PER PARAMETER.
#      Each sampled or derived parameter can have its own `PriorSpec`.
#      The association from parameter name to prior lives in the model
#      layer (`model.jl` / `parameters.jl`), not here. This file defines
#      the distribution types and their arithmetic only.
#
#   4. FIXED PARAMETERS ARE PRIORS TOO.
#      A parameter held fixed at a value is represented as a
#      `PriorSpec{Fixed}`. It contributes 0 to the log-prior and returns
#      its fixed value trivially from `quantile()` and `rand()`. The model
#      layer checks `is_fixed(ps)` to decide whether the parameter enters
#      the sampler vector.
#
#   5. AUTODIFF COMPATIBLE.
#      `logpdf(ps, x)` must work when `x` is a `ForwardDiff.Dual`,
#      `Enzyme.Duplicated`, etc. Distributions.jl handles this for
#      standard distributions via ChainRules. Our custom `ModJeffreys` is
#      generic over `Real` so Dual types trace through.
#
#   6. INTERNAL vs EXTERNAL IS A MODEL-LEVEL CONCEPT.
#      astroEMPEROR distinguishes "internal" priors (acting on sampled
#      parameters) from "external" priors (acting on derived quantities
#      like eccentricity when sampling in sesinw). Both are evaluated
#      with `logpdf(ps, value)` — the only difference is *where*
#      `value` comes from (direct sample vs model computation). The
#      distinction lives in `model.jl`, not here.
#
# Public API:
#   - `AbstractPrior` — abstract supertype
#   - `PriorSpec{D}` — wrapper: (distribution, lo, hi)
#   - `Fixed{T}` — marker for non-sampled parameters
#   - `ModJeffreys{T}` — Gregory 2005 modified Jeffreys distribution
#   - Constructors: `UniformPrior`, `NormalPrior`, `LogUniformPrior`,
#     `BetaPrior`, `ModJeffreysPrior`, `FixedPrior`
#   - Methods: `logpdf`, `quantile`, `cdf`, `rand`, `bounds`,
#     `is_fixed`, `fixed_value`, `prior_transform`, `in_support`
#   - Serialization: `prior_to_dict`, `prior_from_dict`

using Distributions
using Random
import Distributions: logpdf, quantile, cdf, pdf, rand, minimum, maximum,
                      insupport, params
import Base: rand

# =====================================================================
# Abstract supertype
# =====================================================================

"""
    AbstractPrior

Root type for all Nereus priors. Every concrete subtype stores whatever
it needs internally and implements the interface methods: `logpdf`,
`quantile`, `cdf`, `rand`, `bounds`, `is_fixed`, `prior_transform`.
"""
abstract type AbstractPrior end

# =====================================================================
# ModJeffreys — Gregory 2005 modified Jeffreys (proper Distribution)
# =====================================================================

"""
    ModJeffreys(knee, upper)

Modified Jeffreys distribution (Gregory 2005). Probability density on
`[0, upper]`:

    p(x) ∝ 1 / (x + knee)

Reduces to a flat prior for `x ≪ knee` and Jeffreys `1/x` for
`x ≫ knee`, with smooth interpolation across the knee. Commonly used
for hyperparameters that can be zero (e.g. RV jitter where "no extra
jitter" is a legitimate posterior mode).

# Fields
- `knee::T` : knee location, must be `> 0`.
- `upper::T`: upper bound, must be `> knee`.
"""
struct ModJeffreys{T<:Real} <: ContinuousUnivariateDistribution
    knee::T
    upper::T

    function ModJeffreys{T}(knee::T, upper::T) where {T<:Real}
        knee > 0 || throw(ArgumentError("ModJeffreys requires knee > 0"))
        upper > knee || throw(ArgumentError("ModJeffreys requires upper > knee"))
        return new{T}(knee, upper)
    end
end

ModJeffreys(knee::Real, upper::Real) = begin
    T = promote_type(typeof(knee), typeof(upper), Float64)
    ModJeffreys{T}(T(knee), T(upper))
end

Distributions.minimum(d::ModJeffreys{T}) where {T} = zero(T)
Distributions.maximum(d::ModJeffreys{T}) where {T} = d.upper
Distributions.params(d::ModJeffreys) = (d.knee, d.upper)
Base.eltype(::Type{ModJeffreys{T}}) where {T} = T

# Normalisation constant: ∫₀^upper 1/(x+knee) dx = log((knee+upper)/knee)
# _modjeffreys_log_norm: log of the normalisation constant ∫₀^upper 1/(x+k) dx.
# norm = log((k+upper)/k), so log(norm) = log(log((k+upper)/k)).
@inline _modjeffreys_norm(d::ModJeffreys) = log((d.knee + d.upper) / d.knee)
@inline _modjeffreys_log_norm(d::ModJeffreys) = log(_modjeffreys_norm(d))

function Distributions.logpdf(d::ModJeffreys, x::Real)
    if x < 0 || x > d.upper
        return oftype(float(x), -Inf)
    end
    # p(x) = 1/((x+k) * norm) where norm = log((k+upper)/k)
    # logpdf = -log(x+k) - log(norm)
    return -log(x + d.knee) - _modjeffreys_log_norm(d)
end

function Distributions.cdf(d::ModJeffreys, x::Real)
    if x <= 0
        return zero(float(x))
    elseif x >= d.upper
        return one(float(x))
    end
    return log((x + d.knee) / d.knee) / _modjeffreys_norm(d)
end

function Distributions.quantile(d::ModJeffreys, u::Real)
    # CDF: u = log((x+k)/k) / norm  where norm = log((k+upper)/k)
    # ⇒  (x+k)/k = exp(u * norm)
    # ⇒  x = k * (exp(u * norm) - 1)
    if u <= 0
        return zero(float(u))
    elseif u >= 1
        return oftype(float(u), d.upper)
    end
    norm = _modjeffreys_norm(d)
    return d.knee * (exp(u * norm) - 1)
end

Base.rand(rng::AbstractRNG, d::ModJeffreys) = quantile(d, rand(rng))

Distributions.insupport(d::ModJeffreys, x::Real) = (x >= 0) && (x <= d.upper)

# =====================================================================
# Sine — isotropic-orbit inclination prior on [0, π]
# =====================================================================

"""
    Sine()

Sine distribution on `[0, π]` with PDF `p(i) = sin(i)/2`. Equivalent
to a uniform prior on `cos(i) ∈ [-1, 1]` — i.e. the assumption that
orbital orientations are isotropic in 3-D space.

Use this prior for inclination in any astrometric or direct-imaging
fit. Without it (e.g. `Uniform(0, π)` on i directly), the chain can
escape into face-on (i ≈ 0 or π) modes where M_sec sin i is satisfied
by huge M_sec × tiny sin i — physically degenerate but numerically
allowed. EMPEROR II uses this prior; orvara uses the equivalent
sin(i) penalty in its log-prior.
"""
struct Sine <: ContinuousUnivariateDistribution end

Distributions.minimum(::Sine) = 0.0
Distributions.maximum(::Sine) = π
Distributions.params(::Sine) = ()
Base.eltype(::Type{Sine}) = Float64

function Distributions.logpdf(::Sine, x::Real)
    if x < 0 || x > π
        return oftype(float(x), -Inf)
    end
    return log(sin(x) / 2)
end

# CDF: F(i) = ∫₀ⁱ sin(t)/2 dt = (1 − cos i)/2
function Distributions.cdf(::Sine, x::Real)
    if x <= 0
        return zero(float(x))
    elseif x >= π
        return one(float(x))
    end
    return (1 - cos(x)) / 2
end

# Inverse CDF: i = acos(1 − 2u)
function Distributions.quantile(::Sine, u::Real)
    if u <= 0
        return zero(float(u))
    elseif u >= 1
        return oftype(float(u), π)
    end
    return acos(1 - 2 * u)
end

Base.rand(rng::AbstractRNG, d::Sine) = quantile(d, rand(rng))
Distributions.insupport(::Sine, x::Real) = (x >= 0) && (x <= π)

# =====================================================================
# Fixed — marker for non-sampled parameters
# =====================================================================

"""
    Fixed(value)

Marker type for a parameter held fixed at `value`. Not a distribution —
wrapped in `PriorSpec{Fixed{T}}` for API uniformity with sampled priors.
Fixed parameters contribute 0 to the total log-prior (they are not
random) and are not passed to the sampler.
"""
struct Fixed{T<:Real}
    value::T
end

# =====================================================================
# PriorSpec — the Nereus-level wrapper
# =====================================================================

"""
    PriorSpec{D}

Wraps a distribution or `Fixed` marker together with explicit hard
bounds `(lo, hi)`. All Nereus priors are of this type.

# Fields
- `dist::D` : underlying distribution or `Fixed` marker.
- `lo::Float64` : hard lower bound.
- `hi::Float64` : hard upper bound.

Store bounds explicitly (rather than introspecting `D`) so the sampler
layer can query `bounds(ps)` without having to case-split on wrapper
types like `Truncated{Normal{Float64}}`.
"""
struct PriorSpec{D} <: AbstractPrior
    dist::D
    lo::Float64
    hi::Float64

    function PriorSpec(dist::D, lo::Real, hi::Real) where {D}
        lo <= hi || throw(ArgumentError(
            "PriorSpec requires lo ≤ hi (got lo=$lo, hi=$hi)"))
        return new{D}(dist, Float64(lo), Float64(hi))
    end
end

# ---------------------------------------------------------------------
# Constructors for standard priors
# ---------------------------------------------------------------------

"""
    UniformPrior(lo, hi)

Uniform prior on `[lo, hi]`. Requires `lo < hi`.
"""
function UniformPrior(lo::Real, hi::Real)
    lo < hi || throw(ArgumentError("UniformPrior requires lo < hi"))
    return PriorSpec(Uniform(lo, hi), lo, hi)
end

"""
    NormalPrior(μ, σ, lo, hi)
    NormalPrior(μ, σ)

Normal prior with mean `μ` and standard deviation `σ`.

The four-argument form truncates to `[lo, hi]` via
`truncated(Normal(μ, σ), lo, hi)`. The two-argument form returns an
unbounded Normal with `(-Inf, Inf)` bounds — perfectly valid for priors
on unconstrained quantities (e.g. systemic velocity, offsets) but
requires the sampler stack to handle infinite endpoints. In particular,
nested samplers must clip `u` away from `0` and `1` before calling
`quantile`, since `quantile(Normal(μ,σ), 0) = -Inf`.
"""
function NormalPrior(μ::Real, σ::Real, lo::Real, hi::Real)
    σ > 0 || throw(ArgumentError("NormalPrior requires σ > 0"))
    lo < hi || throw(ArgumentError("NormalPrior requires lo < hi"))
    return PriorSpec(truncated(Normal(μ, σ), lo, hi), lo, hi)
end

function NormalPrior(μ::Real, σ::Real)
    σ > 0 || throw(ArgumentError("NormalPrior requires σ > 0"))
    return PriorSpec(Normal(μ, σ), -Inf, Inf)
end

"""
    SinePrior()

Isotropic inclination prior on `[0, π]`: `p(i) = sin(i)/2`. Equivalent
to uniform-in-cos(i). Required for any orbit fit where inclination is
a sampled parameter — without it the chain can escape into face-on
(i ≈ 0) modes where huge M_sec × tiny sin i degenerately satisfies
the RV mass function.
"""
SinePrior() = PriorSpec(Sine(), 0.0, π)

"""
    LogUniformPrior(lo, hi)

Log-uniform (Jeffreys) prior on `[lo, hi]`. Requires `lo > 0`.
Density `p(x) ∝ 1/x`. Use this for strictly-positive scale parameters
(periods, amplitudes, jitter) where you want prior mass per decade
rather than per unit.
"""
function LogUniformPrior(lo::Real, hi::Real)
    lo > 0 || throw(ArgumentError("LogUniformPrior requires lo > 0"))
    lo < hi || throw(ArgumentError("LogUniformPrior requires lo < hi"))
    return PriorSpec(LogUniform(lo, hi), lo, hi)
end

"""
    BetaPrior(a, b; lo=0.0, hi=1.0)

Beta(`a`, `b`) distribution, optionally affine-scaled to `[lo, hi]`.
Useful for eccentricity priors following Kipping 2013 (Beta(0.867, 3.03)
on `[0, 1]`) and for any parameter with natural bounded support.
"""
function BetaPrior(a::Real, b::Real; lo::Real=0.0, hi::Real=1.0)
    a > 0 || throw(ArgumentError("BetaPrior requires a > 0"))
    b > 0 || throw(ArgumentError("BetaPrior requires b > 0"))
    lo < hi || throw(ArgumentError("BetaPrior requires lo < hi"))
    base = Beta(a, b)
    dist = (lo == 0.0 && hi == 1.0) ? base : (lo + (hi - lo) * base)
    return PriorSpec(dist, lo, hi)
end

"""
    ModJeffreysPrior(knee, upper)

Modified Jeffreys prior (Gregory 2005) on `[0, upper]`, density
`∝ 1/(x + knee)`. For `x ≪ knee` behaves as flat, for `x ≫ knee` as
Jeffreys `1/x`. Good choice for hyperparameters that can legitimately
be zero (jitter, extra scatter, activity amplitudes).
"""
function ModJeffreysPrior(knee::Real, upper::Real)
    return PriorSpec(ModJeffreys(knee, upper), 0.0, upper)
end

"""
    FixedPrior(value)

Parameter held fixed at `value`. Not sampled — contributes 0 to the
total log-prior.
"""
function FixedPrior(value::Real)
    return PriorSpec(Fixed(Float64(value)), Float64(value), Float64(value))
end

# ---------------------------------------------------------------------
# Interface methods
# ---------------------------------------------------------------------

"""
    bounds(ps) -> (lo, hi)

Hard lower and upper bounds of the prior.
"""
bounds(ps::PriorSpec) = (ps.lo, ps.hi)

"""
    is_fixed(ps) -> Bool

`true` iff the parameter is held fixed (wraps `Fixed`).
"""
is_fixed(::PriorSpec{<:Fixed}) = true
is_fixed(::PriorSpec) = false

"""
    fixed_value(ps) -> value

Return the fixed value. Errors if `ps` is not a fixed prior.
"""
fixed_value(ps::PriorSpec{<:Fixed}) = ps.dist.value
fixed_value(ps::PriorSpec) = throw(ArgumentError("fixed_value called on non-fixed prior"))

"""
    in_support(ps, x) -> Bool

`true` iff `x` is within the hard bounds of the prior.
"""
in_support(ps::PriorSpec, x::Real) = ps.lo <= x <= ps.hi

# --- logpdf ----------------------------------------------------------

"""
    logpdf(ps::PriorSpec, x) -> log p(x)

Log prior density at `x`. Returns `-Inf` if `x` is outside the hard
bounds. For fixed priors, returns `0.0` (the fixed parameter contributes
nothing to the log-prior sum).
"""
function Distributions.logpdf(ps::PriorSpec{<:Fixed}, x::Real)
    # Fixed params contribute nothing to the log-prior sum. We don't
    # check `x == ps.dist.value` because in practice `logpdf` is never
    # called on a fixed prior by the sampler (fixed params are not in
    # the sampling vector). If it is called diagnostically, returning
    # 0.0 is the correct "no contribution" answer.
    return zero(float(x))
end

function Distributions.logpdf(ps::PriorSpec, x::Real)
    if !in_support(ps, x)
        return oftype(float(x), -Inf)
    end
    return logpdf(ps.dist, x)
end

# --- quantile --------------------------------------------------------

"""
    quantile(ps::PriorSpec, u)

Inverse CDF at `u ∈ [0, 1]`. Used for nested-sampling / Pigeons prior
transforms. For fixed priors always returns the fixed value regardless
of `u`.
"""
function Distributions.quantile(ps::PriorSpec{<:Fixed}, u::Real)
    return oftype(float(u), ps.dist.value)
end

function Distributions.quantile(ps::PriorSpec, u::Real)
    return quantile(ps.dist, u)
end

# Alias matching the nested-sampling convention.
"""
    prior_transform(u, ps) -> x

Transform a uniform sample `u ∈ [0, 1]` to a prior sample `x`. Alias for
`quantile(ps, u)` under the nested-sampling convention.
"""
prior_transform(u::Real, ps::PriorSpec) = quantile(ps, u)

# --- cdf -------------------------------------------------------------

function Distributions.cdf(ps::PriorSpec{<:Fixed}, x::Real)
    return x < ps.dist.value ? zero(float(x)) : one(float(x))
end

function Distributions.cdf(ps::PriorSpec, x::Real)
    if x <= ps.lo
        return zero(float(x))
    elseif x >= ps.hi
        return one(float(x))
    end
    return cdf(ps.dist, x)
end

# --- rand ------------------------------------------------------------

function Base.rand(rng::AbstractRNG, ps::PriorSpec{<:Fixed})
    return ps.dist.value
end

function Base.rand(rng::AbstractRNG, ps::PriorSpec)
    return rand(rng, ps.dist)
end

# Default RNG variant
Base.rand(ps::PriorSpec) = rand(Random.default_rng(), ps)

# =====================================================================
# Batch / collection operations
# =====================================================================

"""
    logpdf_sum(priors, values) -> Float64

Sum of `logpdf(p, v)` over paired `priors` and `values`. Both arguments
must support `eachindex` and have matching lengths. Short-circuits to
`-Inf` on the first out-of-support value for speed.
"""
function logpdf_sum(priors, values)
    n = length(priors)
    n == length(values) || throw(ArgumentError(
        "priors length ($n) must match values length ($(length(values)))"))
    total = zero(eltype(values))
    @inbounds for i in 1:n
        lp = logpdf(priors[i], values[i])
        if lp == -Inf
            return oftype(total, -Inf)
        end
        total += lp
    end
    return total
end

"""
    prior_transform!(dst, us, priors)

In-place prior transform: `dst[i] = quantile(priors[i], us[i])`.
Used by nested samplers that work in the unit hypercube and need to
map each coordinate through its parameter's prior.
"""
function prior_transform!(dst::AbstractVector, us::AbstractVector, priors)
    @assert length(dst) == length(us) == length(priors)
    @inbounds for i in eachindex(dst)
        dst[i] = quantile(priors[i], us[i])
    end
    return dst
end

# =====================================================================
# Physical validation
# =====================================================================
#
# A prior attached to a specific parameter must not permit unphysical
# values. For example, an eccentricity prior must not allow e < 0, and
# a fixed eccentricity of -1 is absurd. This section provides:
#
#   - `PHYSICAL_BOUNDS` : registry mapping parameter base names to their
#     physical ranges (a hard requirement from the physics of the
#     model, not a choice).
#   - `extract_base_name(param_name)` : strip mode prefix and instrument
#     / planet / order suffix from a full parameter name to find its
#     base-name key in `PHYSICAL_BOUNDS`.
#   - `physical_bounds(param_name)` : look up the physical range, or
#     `(-Inf, Inf)` for unknown parameters.
#   - `validate_physical(param_name, ps)` : assert that a prior is
#     physically consistent; throws `ArgumentError` on violation.
#
# Callers (typically the model layer at prior-registration time) are
# expected to invoke `validate_physical` before accepting a user-
# supplied prior. Unknown parameter names pass through silently — this
# function is strict only for parameters whose physics we know.
#
# Parameter-name conventions (astroEMPEROR-compatible):
#   - Planet params:  `P_k1`, `K_k2`, `ecc_k3`, `sesinw_k1`, `b_k2`, …
#   - RV instrument:  `gamma_HARPS`, `sigma_HARPS`
#   - PM instrument:  `offset_TESS`, `jitter_TESS`, `dilution_TESS`
#   - GP:             `rv-GP-tau_HARPS`, `pm-GP-S0`, `pm-GP-Q_TESS`
#   - ARMA:           `rv-phi_1,HARPS`, `pm-omega_2,TESS`
#   - LD groups:      `q1_TESS,TESS2`
#   - Activity:       `C_0,HARPS`
#   - Acceleration:   `acc_1`, `acc_2`
#   - Trends:         `trend_1,TESS`
#
# The mode prefix `rv-` or `pm-` is stripped first; then we split on
# the first `_` and take the head as the base name. This gives us
# `GP-tau`, `phi`, `q1`, etc.

"""
    PHYSICAL_BOUNDS :: Dict{String, Tuple{Float64,Float64}}

Registry of hard physical bounds per parameter base name. Priors on a
parameter must not extend beyond these bounds. Values are `(min, max)`
inclusive.

The physics encoded here is not negotiable — these bounds come from
the definitions of the quantities themselves (eccentricity ∈ [0, 1),
impact parameter ≥ 0, stellar density > 0, etc.). Parameters not in
this registry are treated as unrestricted and pass validation silently.
"""
const PHYSICAL_BOUNDS = Dict{String, Tuple{Float64, Float64}}(
    # --- Orbital ---
    "P"       => (0.0, Inf),       # Period [days]
    "K"       => (0.0, Inf),       # RV semi-amplitude [m/s]
    "e"       => (0.0, 1.0),       # Eccentricity (open upper end in practice)
    "ecc"     => (0.0, 1.0),       # astroEMPEROR alias for e
    "w"       => (-2π, 2π),        # Argument of periastron [rad]; periodic
    "Mo"      => (-2π, 2π),        # Mean anomaly [rad]
    "sesinw"  => (-1.0, 1.0),      # √e sin ω
    "secosw"  => (-1.0, 1.0),      # √e cos ω
    "esinw"   => (-1.0, 1.0),      # e sin ω
    "ecosw"   => (-1.0, 1.0),      # e cos ω
    "Tp"      => (-Inf, Inf),      # Time of periastron [JD]
    "Tc"      => (-Inf, Inf),      # Time of transit center [JD]
    "TM"      => (-Inf, Inf),      # astroEMPEROR time parameter [JD]
    # --- Transit geometry ---
    "b"       => (0.0, 2.0),       # Impact parameter (allow grazing)
    "rr"      => (0.0, 1.0),       # Rp/Rs (user can narrow further)
    "r1"      => (0.0, 1.0),       # Espinoza 2018 r1
    "r2"      => (0.0, 1.0),       # Espinoza 2018 r2
    "a"       => (0.0, Inf),       # Semi-major axis (a/Rs or AU)
    "rho_s"   => (0.0, Inf),       # Stellar density [g/cm³] (multi-seg base)
    "inc"     => (0.0, 180.0),     # Inclination [deg]
    "q1"      => (0.0, 1.0),       # Kipping LD
    "q2"      => (0.0, 1.0),       # Kipping LD
    # --- RV instrumental ---
    "gamma"   => (-Inf, Inf),      # Systemic velocity
    "sigma"   => (0.0, Inf),       # RV white-noise jitter
    "acc"     => (-Inf, Inf),      # Acceleration coefficient
    # --- Photometry instrumental ---
    "offset"  => (-Inf, Inf),      # Flux offset
    "jitter"  => (0.0, Inf),       # Flux white-noise jitter
    "dilution" => (0.0, 1.0),      # Transit dilution factor
    "trend"   => (-Inf, Inf),      # Polynomial trend coefficient
    # --- Activity ---
    "C"       => (-Inf, Inf),      # Activity-indicator coefficient
    # --- GP hyperparameters (amplitudes / timescales strictly positive) ---
    "GP-a"      => (0.0, Inf),     # Rotation / Matern / Exp amplitude
    "GP-tau"    => (0.0, Inf),     # Decay timescale
    "GP-p"      => (0.0, Inf),     # Rotation period
    "GP-f"      => (-Inf, Inf),    # Factor (log)
    "GP-taue"   => (0.0, Inf),     # Exp timescale
    "GP-taum"   => (0.0, Inf),     # Matern timescale
    "GP-S0"     => (0.0, Inf),     # SHO power
    "GP-omega0" => (0.0, Inf),     # SHO frequency
    "GP-Q"      => (0.0, Inf),     # SHO quality factor
    # --- ARMA (permissive; real constraints are stationarity) ---
    "phi"     => (-1.0, 1.0),
    "alpha"   => (-Inf, Inf),
    "omega"   => (-1.0, 1.0),
    "beta"    => (-Inf, Inf),
)

# Multi-segment base names — parameter names whose "base" contains an
# underscore and therefore should NOT be split at the first `_`. Any
# parameter name that equals one of these, or starts with it followed by
# a suffix separator (`_`, `,`, `k1`...), is recognised as that base.
#
# Keep this list tiny. Adding a new multi-segment base should be a
# deliberate architectural choice, not a dumping ground.
const _MULTI_SEGMENT_BASES = ("rho_s",)

"""
    extract_base_name(param_name::AbstractString) -> String

Extract the physics base name from a full parameter name.

Strips a leading `"rv-"` / `"pm-"` mode prefix if present, recognises
a small set of multi-segment base names (currently `"rho_s"`), and
otherwise returns everything before the first underscore.

# Examples
```
julia> extract_base_name("P_k1")
"P"

julia> extract_base_name("ecc_k3")
"ecc"

julia> extract_base_name("gamma_HARPS")
"gamma"

julia> extract_base_name("rv-GP-tau_HARPS")
"GP-tau"

julia> extract_base_name("pm-GP-p")
"GP-p"

julia> extract_base_name("C_0,HARPS")
"C"

julia> extract_base_name("rho_s")
"rho_s"
```
"""
function extract_base_name(param_name::AbstractString)
    s = param_name
    # Strip mode prefix `rv-` or `pm-`.
    if length(s) >= 3 && (SubString(s, 1, 3) == "rv-" || SubString(s, 1, 3) == "pm-")
        s = SubString(s, 4)
    end
    # Check multi-segment bases first — they take precedence over the
    # "split at first underscore" rule.
    for base in _MULTI_SEGMENT_BASES
        if s == base
            return base
        end
        # Match the base followed by a legal suffix separator so that
        # `rho_s_HARPS` (if it ever existed) matches but `rho_star` does
        # not.
        n = length(base)
        if length(s) > n + 1 && SubString(s, 1, n) == base
            sep = s[n+1]
            if sep == '_' || sep == ','
                return base
            end
        end
    end
    # Default: split on first underscore.
    idx = findfirst('_', s)
    if idx === nothing
        return String(s)
    end
    return String(SubString(s, 1, prevind(s, idx)))
end

"""
    physical_bounds(param_name) -> (min, max)

Look up the physical bounds for a parameter. Returns `(-Inf, Inf)` for
unknown parameters (no restriction).
"""
function physical_bounds(param_name::AbstractString)
    base = extract_base_name(param_name)
    return get(PHYSICAL_BOUNDS, base, (-Inf, Inf))
end

"""
    validate_physical(param_name::AbstractString, ps::PriorSpec)

Assert that `ps` is physically consistent with the named parameter.
Throws `ArgumentError` if:

  - The prior is a `FixedPrior` whose fixed value lies outside the
    physical range.
  - The prior's hard lower bound is below the physical minimum, or
    its upper bound is above the physical maximum.

Unknown parameter names pass through without error. Returns `nothing`
on success.

Call this at prior-registration time in the model layer — typically
just after the user hands us a `Dict{String,PriorSpec}` mapping
parameter names to priors.
"""
function validate_physical(param_name::AbstractString, ps::PriorSpec)
    base = extract_base_name(param_name)
    haskey(PHYSICAL_BOUNDS, base) || return nothing   # unknown: pass through

    phys_lo, phys_hi = PHYSICAL_BOUNDS[base]

    if is_fixed(ps)
        v = fixed_value(ps)
        if v < phys_lo || v > phys_hi
            throw(ArgumentError(
                "Fixed prior value $v for parameter '$param_name' " *
                "(base '$base') is outside physical range " *
                "[$phys_lo, $phys_hi]"))
        end
        return nothing
    end

    # For non-fixed priors, require the stored bounds to be a subset
    # of the physical range. Allow a tiny numerical slop at the
    # boundary so that e.g. `UniformPrior(0, 1)` passes for `e` even
    # though `1.0` is technically at the open upper end.
    slop = 16 * eps(Float64)

    if ps.lo < phys_lo - slop
        throw(ArgumentError(
            "Prior lower bound $(ps.lo) for parameter '$param_name' " *
            "(base '$base') is below physical minimum $phys_lo"))
    end
    if ps.hi > phys_hi + slop
        throw(ArgumentError(
            "Prior upper bound $(ps.hi) for parameter '$param_name' " *
            "(base '$base') is above physical maximum $phys_hi"))
    end
    return nothing
end

# =====================================================================
# Serialization (for NetCDF metadata / Python worker interchange)
# =====================================================================

"""
    prior_to_dict(ps::PriorSpec) -> Dict{String,Any}

Serialize a `PriorSpec` to a plain dict suitable for NetCDF metadata or
JSON interchange with the Python worker. The dict always contains a
`"type"` key identifying the prior class plus that class's
hyperparameters and the bounds.
"""
function prior_to_dict(ps::PriorSpec)
    d = ps.dist
    out = Dict{String,Any}("lo" => ps.lo, "hi" => ps.hi)

    if d isa Fixed
        out["type"] = "Fixed"
        out["value"] = d.value
    elseif d isa Uniform
        out["type"] = "Uniform"
    elseif d isa LogUniform
        out["type"] = "LogUniform"
    elseif d isa Truncated{<:Normal}
        out["type"] = "Normal"
        out["mu"] = d.untruncated.μ
        out["sigma"] = d.untruncated.σ
    elseif d isa Beta
        out["type"] = "Beta"
        out["a"] = d.α
        out["b"] = d.β
    elseif d isa ModJeffreys
        out["type"] = "ModJeffreys"
        out["knee"] = d.knee
        out["upper"] = d.upper
    else
        # Affine-scaled Beta or similar — try to detect
        out["type"] = "Unknown"
        out["julia_type"] = string(typeof(d))
    end
    return out
end

"""
    prior_from_dict(d::AbstractDict) -> PriorSpec

Reconstruct a `PriorSpec` from a dict produced by `prior_to_dict`.
Dispatches on the `"type"` key.
"""
function prior_from_dict(d::AbstractDict)
    t = d["type"]
    if t == "Fixed"
        return FixedPrior(d["value"])
    elseif t == "Uniform"
        return UniformPrior(d["lo"], d["hi"])
    elseif t == "LogUniform"
        return LogUniformPrior(d["lo"], d["hi"])
    elseif t == "Normal"
        return NormalPrior(d["mu"], d["sigma"], d["lo"], d["hi"])
    elseif t == "Beta"
        lo = get(d, "lo", 0.0)
        hi = get(d, "hi", 1.0)
        return BetaPrior(d["a"], d["b"]; lo=lo, hi=hi)
    elseif t == "ModJeffreys"
        return ModJeffreysPrior(d["knee"], d["upper"])
    else
        throw(ArgumentError("Unknown prior type in dict: $t"))
    end
end
