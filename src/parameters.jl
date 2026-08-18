# Theta — runtime parameter state.
#
# Holds a mutable `Vector{T}` of parameter values, where `T` is the
# element type (`Float64` for vanilla sampling, a `ForwardDiff.Dual`
# or `Enzyme.Duplicated` element for autodiff-based samplers). The
# back-reference to `Params` gives accessors O(1) access to the
# precomputed layout metadata (`PlanetBlock` subtype per planet, the
# `SystemicIndices` for instrument parameters, the parallel
# `unfrozen_priors` for `log_prior` evaluation).
#
# Design decisions locked in here (see `soft-herding-torvalds.md`):
#
# 1. **Parametric on element type `T`.** `Theta{T<:Real}` means the
#    same struct definition and accessors serve both Float64 sampling
#    and autodiff gradient passes. Constructing a `Theta{Dual{…}}` and
#    passing a Dual vector through `set_unfrozen!` now preserves the
#    Dual through to `log_prior` and into the downstream physics —
#    HMC/NUTS will flow gradients correctly.
#
# 2. **`set_unfrozen!` requires exact element type match.** No silent
#    narrowing from Dual to Float64. A dispatch error is infinitely
#    better than a silently wrong gradient. The autodiff wrapper
#    constructs `Theta{T}(params)` with the matching sampler vector
#    type up front.
#
# 3. **`log_prior` reads `params.layout.unfrozen_priors` directly.**
#    No dict lookup, no per-iteration name resolution. The parallel
#    `unfrozen_priors` vector was precomputed at `Params` construction.
#    The handful of concrete `PriorSpec{D}` subtypes inside the vector
#    Union-split cleanly.
#
# 4. **Hot-path accessors dispatch on `PlanetBlock` subtype.**
#    `planet_K` on an `RVOnlyBlock` or `RVPMBlock` reads the `.K`
#    field; on a `PMOnlyBlock` the block has no `K` field, so the
#    method is not defined for it — a type error is raised instead of
#    a runtime sentinel check. Same for `planet_b_rr` on RV-only
#    planets.
#
# 5. **`n_p` reads `params.layout.n_p_idx`.** No magic slot-1 hardcoded
#    in the accessor.

# =====================================================================
# Theta struct
# =====================================================================

"""
    Theta{T<:Real}

Runtime parameter state for a Nereus model. Parametric on element
type `T`:

- `Theta{Float64}` for vanilla (non-autodiff) sampling
- `Theta{ForwardDiff.Dual{Tag,V,N}}` for forward-mode autodiff gradients
- Any other `T <: Real` as long as Distributions.jl's `logpdf` and
  the orbit physics functions accept it

Construct via:

- `Theta(params)` — defaults to `Theta{Float64}`
- `Theta{T}(params)` — allocates with element type `T`
- `Theta{T}(params, values)` — explicit initial values, checked length
- `Theta(params, values)` — infers `T` from `eltype(values)`

Frozen slots are initialized from `params.layout.frozen_values` at
construction (cast to `T` via `convert`). Unfrozen slots start at
zero.
"""
mutable struct Theta{T<:Real}
    values::Vector{T}
    params::Params
    td::Union{TransDimState, Nothing}
end

# Construct a Theta{T} with allocated values vector, frozen slots
# populated, unfrozen slots zero.
function Theta{T}(params::Params; td::Union{TransDimState, Nothing}=nothing) where {T<:Real}
    n = n_total(params)
    values = zeros(T, n)
    layout = params.layout
    @inbounds for (idx, v) in zip(layout.frozen_idx, layout.frozen_values)
        values[idx] = convert(T, v)
    end
    return Theta{T}(values, params, td)
end

# Default element type is Float64.
Theta(params::Params; td::Union{TransDimState, Nothing}=nothing) =
    Theta{Float64}(params; td=td)

# Construct from an explicit values vector. Length is checked.
# Frozen slots are overwritten from the values arg (useful for loading
# a saved state; caller's responsibility to ensure consistency).
function Theta{T}(params::Params, values::AbstractVector;
                  td::Union{TransDimState, Nothing}=nothing) where {T<:Real}
    length(values) == n_total(params) || throw(ArgumentError(
        "values length $(length(values)) does not match params n_total " *
        "$(n_total(params))"))
    v = Vector{T}(undef, length(values))
    @inbounds for i in eachindex(values)
        v[i] = convert(T, values[i])
    end
    return Theta{T}(v, params, td)
end

# Infer T from the values vector's eltype.
Theta(params::Params, values::AbstractVector{T};
      td::Union{TransDimState, Nothing}=nothing) where {T<:Real} =
    Theta{T}(params, values; td=td)

# =====================================================================
# Size / introspection
# =====================================================================

Base.length(theta::Theta)  = length(theta.values)
Base.eltype(::Theta{T}) where {T} = T
n_total(theta::Theta)      = length(theta.values)
n_unfrozen(theta::Theta)   = length(theta.params.layout.unfrozen_idx)
n_frozen(theta::Theta)     = length(theta.params.layout.frozen_idx)

# =====================================================================
# Sampler interface: set/get unfrozen subset
# =====================================================================

"""
    set_unfrozen!(theta::Theta{T}, x::AbstractVector{T}) where {T}

Write the unfrozen subset of `theta.values` from the sampler vector
`x`. `x` must have length `n_unfrozen(theta)` **and exact element type
`T`** — the method signature requires it. The exact-type requirement
is deliberate: silently narrowing `Dual` to `Float64` would destroy
gradient information. If you want to pass a different element type,
construct a `Theta{eltype(x)}` first.

Returns `theta` for chaining.
"""
function set_unfrozen!(theta::Theta{T}, x::AbstractVector) where {T}
    unfrozen = theta.params.layout.unfrozen_idx
    length(x) == length(unfrozen) || throw(ArgumentError(
        "x length $(length(x)) does not match n_unfrozen " *
        "$(length(unfrozen))"))
    @inbounds for i in eachindex(unfrozen)
        theta.values[unfrozen[i]] = convert(T, x[i])
    end
    return theta
end

"""
    unfrozen_values(theta) -> Vector{T}

Return a fresh vector containing the current values of the unfrozen
slots, in the same order as `params.layout.unfrozen_names`. Allocates.
Use for diagnostics and warm-start, not the hot path.
"""
function unfrozen_values(theta::Theta{T}) where {T}
    unfrozen = theta.params.layout.unfrozen_idx
    out = Vector{T}(undef, length(unfrozen))
    @inbounds for i in eachindex(unfrozen)
        out[i] = theta.values[unfrozen[i]]
    end
    return out
end

"""
    unfrozen_values!(dst, theta)

In-place version: write current unfrozen values into `dst`. `dst`
must have length `n_unfrozen(theta)`.
"""
function unfrozen_values!(dst::AbstractVector, theta::Theta)
    unfrozen = theta.params.layout.unfrozen_idx
    length(dst) == length(unfrozen) || throw(ArgumentError(
        "dst length does not match n_unfrozen"))
    @inbounds for i in eachindex(unfrozen)
        dst[i] = theta.values[unfrozen[i]]
    end
    return dst
end

# =====================================================================
# Name-based access (config / diagnostic path, NOT hot path)
# =====================================================================

"""
    get_param(theta, name) -> value

Read a parameter by full name. Dict lookup. Use for config,
diagnostics, or one-off access — the hot path uses block accessors.
"""
function get_param(theta::Theta, name::AbstractString)
    idx = theta.params.layout.name_to_idx[name]
    return theta.values[idx]
end

"""
    set_param!(theta, name, value)

Write a parameter by full name. Dict lookup. For config and
diagnostics only.
"""
function set_param!(theta::Theta, name::AbstractString, value::Real)
    idx = theta.params.layout.name_to_idx[name]
    theta.values[idx] = value
    return theta
end

# =====================================================================
# Hot-path block accessors
# =====================================================================
#
# These functions dispatch on the concrete `PlanetBlock` subtype,
# which is resolved via Union-splitting (only 3 concrete types).
# Accessing an absent parameter is a method-error at runtime — there
# is no field on the block to read, so the call cannot compile.

"""
    planet_P(theta, k) -> value

Orbital period of planet `k`, in days. Valid for any planet block.

Under `:K_driven` and `:M_sec_driven`, reads the sampled `P_kN` slot
directly. Under `:a_driven`, the slot holds `a` (AU); P is derived
from Kepler's third law `P²[yr] = a³[AU] / M_total[M_sun]` using the
sampled or fixed M_pri and the per-planet M_sec.
"""
@inline function planet_P(theta::Theta, k::Int)
    block = theta.params.layout.planet_blocks[k]
    if theta.params.config.parametrization.mass === :a_driven
        a = theta.values[block.P]    # slot holds a, in AU
        M_pri = astrom_M_pri(theta)
        M_sec = theta.values[block.K]  # slot holds M_sec under :a_driven
        # Kepler 3rd: P² = a³ / M_total (in solar/AU/yr units)
        P_yr = sqrt(a^3 / max(M_pri + M_sec, oftype(a, 1e-12)))
        return P_yr * oftype(P_yr, 365.25)
    else
        return theta.values[block.P]
    end
end

"""
    planet_a(theta, k) -> a (AU)

Orbital semi-major axis (relative separation) of planet `k`, in AU.
Under `:a_driven`, returns the sampled slot directly. Otherwise
derives via Kepler's third law from `planet_P` and the total mass.
"""
@inline function planet_a(theta::Theta, k::Int)
    block = theta.params.layout.planet_blocks[k]
    if theta.params.config.parametrization.mass === :a_driven
        return theta.values[block.P]   # slot holds a
    else
        P_d   = theta.values[block.P]
        M_pri = astrom_M_pri(theta)
        M_sec = planet_M_sec(theta, k)
        P_yr  = P_d / oftype(P_d, 365.25)
        return cbrt((M_pri + M_sec) * P_yr * P_yr)
    end
end

"""
    planet_K(theta, k) -> value

RV semi-amplitude of planet `k`, in m/s.

In `:K_driven` mass parametrization (default), reads the sampled
`K_kN` slot directly. In `:M_sec_driven` mode the slot holds `M_sec`
in M_sun; K is then derived from the standard RV mass function:

    K^3 = (2πG / P) · (M_sec sin i)^3 / (M_pri + M_sec)^2 / (1 - e²)^(3/2)

via the bidirectional helper `_K_from_msec`. ForwardDiff propagates
through the derivation cleanly.
"""
@inline planet_K(theta::Theta, k::Int) =
    _planet_K(theta.params.layout.planet_blocks[k], theta, k)

@inline _planet_K(::PMOnlyBlock, ::Theta, ::Int) =
    throw(ArgumentError("planet has no K — PMOnlyBlock"))
@inline _planet_K(::SB2Block, ::Theta, ::Int) =
    throw(ArgumentError("SB2Block has two amplitudes K_A and K_B, not a " *
                        "single K — use planet_K_A / planet_K_B"))
@inline function _planet_K(block::Union{RVOnlyBlock, RVPMBlock, RVASBlock, RVPMASBlock},
                            theta::Theta, k::Int)
    if theta.params.config.parametrization.mass === :K_driven
        return theta.values[block.K]
    else  # :M_sec_driven or :a_driven — slot holds M_sec, derive K
        M_sec = theta.values[block.K]
        return _planet_K_from_msec(theta, k, M_sec)
    end
end

# Internal: K from (M_sec, P, e, sin i, M_pri). Pure math, no Theta
# accessor recursion. Used by planet_K under :M_sec_driven and by RV
# likelihood paths that need K.
@inline function _planet_K_from_msec(theta::Theta, k::Int, M_sec)
    P    = planet_P(theta, k)
    e, _ = planet_e_w(theta, k)
    M_pri = astrom_M_pri(theta)
    block = theta.params.layout.planet_blocks[k]
    # If the block doesn't carry inclination, fall back to edge-on
    # (sin i = 1) — corresponds to RVOnlyBlock / RVPMBlock without
    # astrometry. In those cases there is no astrometric constraint
    # on i, and K samples M_sec sin i; we report K as if sin i = 1
    # (i.e. the minimum mass form).
    sin_i = if block isa Union{RVASBlock, RVPMASBlock}
        sin(_planet_inc(block, theta))
    else
        one(M_sec)
    end
    one_minus_e2 = max(1 - e * e, zero(e))
    F = oftype(M_sec, _F_M_FACTOR)
    f_M = M_sec^3 * sin_i^3 / (M_pri + M_sec)^2
    K3 = f_M / (P * one_minus_e2^(3/2) * F)
    return cbrt(K3)
end

# Mass-function constant in Nereus units: K^3 [m³/s³] · P [days] ·
# (1−e²)^(3/2) → f_M [M_sun]. Defined in astrometry/projection.jl;
# we re-import the value here for use in planet_K under :M_sec_driven.
const _F_M_FACTOR = 1.0367e-16

"""
    planet_K_A(theta, k) -> value

Primary RV semi-amplitude `K_A` (m/s) of the SB2 binary block `k`.
Applied to `rv_comp == 1` (primary) points. Sampled directly,
independent of the global mass parametrization — the SB2 binary always
carries two explicit amplitudes so the mass ratio `q = K_A/K_B` is
constrained by the data, not a mass-function assumption.
"""
@inline function planet_K_A(theta::Theta, k::Int)
    block = theta.params.layout.planet_blocks[k]
    block isa SB2Block || throw(ArgumentError(
        "planet_K_A is only defined for the SB2 binary block (got $(typeof(block)))"))
    return theta.values[block.KA]
end

"""
    planet_K_B(theta, k) -> value

Secondary RV semi-amplitude `K_B` (m/s) of the SB2 binary block `k`.
Applied *anti-phase* (`−K_B`) to `rv_comp == 2` (secondary) points.
See `planet_K_A`.
"""
@inline function planet_K_B(theta::Theta, k::Int)
    block = theta.params.layout.planet_blocks[k]
    block isa SB2Block || throw(ArgumentError(
        "planet_K_B is only defined for the SB2 binary block (got $(typeof(block)))"))
    return theta.values[block.KB]
end

"""
    planet_e_w(theta, k) -> (e, ω)

Eccentricity and argument of periastron, decoded from whatever
parametrization is in use (`:sesinw`, `:esinw`, or `:ew`).
"""
@inline function planet_e_w(theta::Theta, k::Int)
    block = theta.params.layout.planet_blocks[k]
    x1 = theta.values[block.e1]
    x2 = theta.values[block.e2]
    ew = theta.params.config.parametrization.ew
    if ew === :sesinw
        return sesinw_to_ew(x1, x2)
    elseif ew === :esinw
        return esinw_to_ew(x1, x2)
    else  # :ew
        return (x1, x2)
    end
end

"""
    planet_time_anchor(theta, k) -> value

The raw time-anchor parameter for planet `k`. Meaning depends on
`parametrization.time` (`:Mo`, `:Tp`, or `:Tc`).
"""
@inline function planet_time_anchor(theta::Theta, k::Int)
    block = theta.params.layout.planet_blocks[k]
    return theta.values[block.t]
end

"""
    planet_b_rr(theta, k) -> (b_or_r1, rr_or_r2)

Transit geometry parameters for planet `k`. Only valid for
`PMOnlyBlock` and `RVPMBlock`; calling on an `RVOnlyBlock` raises
`ArgumentError`.

Under `parametrization.geom = :b_rr` the values are `(b, rr)`;
under `:r1r2` they are `(r1, r2)`. Conversion helpers live in
`orbit.jl` / `transit.jl`.
"""
@inline planet_b_rr(theta::Theta, k::Int) =
    _planet_b_rr(theta.params.layout.planet_blocks[k], theta)

@inline _planet_b_rr(block::PMOnlyBlock, theta::Theta) =
    (theta.values[block.b], theta.values[block.r])
@inline _planet_b_rr(block::RVPMBlock, theta::Theta) =
    (theta.values[block.b], theta.values[block.r])
@inline _planet_b_rr(block::RVPMASBlock, theta::Theta) =
    (theta.values[block.b], theta.values[block.r])
@inline _planet_b_rr(::RVOnlyBlock, ::Theta) =
    throw(ArgumentError("planet has no transit geometry — RVOnlyBlock"))
@inline _planet_b_rr(::RVASBlock, ::Theta) =
    throw(ArgumentError("planet has no transit geometry — RVASBlock"))

# =====================================================================
# Astrometry accessors (RVASBlock, RVPMASBlock)
# =====================================================================

"""
    planet_inc(theta, k) -> i

Orbital inclination of planet `k` in radians.

- For `RVASBlock`: read directly from the sampled `inc` slot.
- For `RVPMASBlock`: **derived** from the transit impact parameter `b`
  via the geometric relation
  `cos i = b · (1 + e sin ω) / ((a/R★) · (1 − e²))`. The arccos
  argument is clamped to `[-1, 1]` to avoid domain errors at the
  numerical edge.

Raises `ArgumentError` for the non-AS block types.
"""
@inline planet_inc(theta::Theta, k::Int) =
    _planet_inc(theta.params.layout.planet_blocks[k], theta)

@inline _planet_inc(block::RVASBlock, theta::Theta) = theta.values[block.inc]
@inline _planet_inc(block::SB2Block, theta::Theta) = theta.values[block.inc]

# RVPMASBlock: i is derived from b, not sampled.
@inline function _planet_inc(block::RVPMASBlock, theta::Theta)
    parametrization = theta.params.config.parametrization

    # Decode b from the sampled (b, rr) or (r1, r2) pair — mirrors
    # the decoding in transit_likelihood.jl.
    b_raw = theta.values[block.b]
    rr_raw = theta.values[block.r]
    b = parametrization.geom === :r1r2 ?
        b_raw * (1 - rr_raw) / 2 :
        b_raw

    # Eccentricity / argument of periastron.
    P = theta.values[block.P]
    e = _ecc_from_block(block, theta)
    sinw = _sinw_from_block(block, theta)

    # a/R*: prefer rho_s when available; otherwise fall back to
    # M_s, R_s (config) via Kepler's third law. Mirrors the logic in
    # transit_likelihood.jl.
    a_Rs = _a_Rs_for_inc(theta, P)

    # Geometric relation: cos i = b * (1 + e sin ω) / (a_Rs * (1 - e²))
    one_minus_e2 = 1 - e * e
    denom = a_Rs * one_minus_e2
    # Guard against pathological numerical states (e≥1 should already
    # be rejected upstream, but be defensive here).
    cos_i = b * (1 + e * sinw) / denom
    cos_i = clamp(cos_i, -one(cos_i), one(cos_i))
    return acos(cos_i)
end

# Internal: pull e from the planet block without going through the
# generic `planet_e_w(theta, k)` (which would re-resolve the block via
# `theta.params.layout.planet_blocks[k]`). We already have the block.
@inline function _ecc_from_block(block::PlanetBlock, theta::Theta)
    x1 = theta.values[block.e1]
    x2 = theta.values[block.e2]
    ew = theta.params.config.parametrization.ew
    if ew === :sesinw
        return x1 * x1 + x2 * x2          # e = se² + sc²
    elseif ew === :esinw
        return sqrt(x1 * x1 + x2 * x2)    # e = √(es² + ec²)
    else  # :ew
        return x1
    end
end

@inline function _sinw_from_block(block::PlanetBlock, theta::Theta)
    x1 = theta.values[block.e1]
    x2 = theta.values[block.e2]
    ew = theta.params.config.parametrization.ew
    if ew === :sesinw
        # sesinw = √e sin ω, secosw = √e cos ω → sin ω = se / √(se²+sc²)
        denom = sqrt(x1 * x1 + x2 * x2)
        return iszero(denom) ? zero(x1) : x1 / denom
    elseif ew === :esinw
        # esinw = e sin ω, ecosw = e cos ω
        denom = sqrt(x1 * x1 + x2 * x2)
        return iszero(denom) ? zero(x1) : x1 / denom
    else  # :ew — x2 is ω directly
        return sin(x2)
    end
end

# Internal: a/R* for the planet, used by the inclination derivation.
# Pulls from rho_s when active, else uses (M_s, R_s) via Kepler 3.
@inline function _a_Rs_for_inc(theta::Theta{T}, P) where {T}
    parametrization = theta.params.config.parametrization
    if parametrization.use_rho_s
        return rho_s_to_a_Rs(rho_s(theta), P)
    end
    M_s = theta.params.config.M_s
    R_s = theta.params.config.R_s
    if isnan(M_s) || isnan(R_s) || R_s <= 0
        throw(ArgumentError(
            "RVPMASBlock requires either parametrization.use_rho_s=true " *
            "or finite (M_s, R_s) in config to derive inclination from b"))
    end
    # Mirror the formula in transit_likelihood.jl.
    P_s = P * T(86400.0)
    GM = T(1.3271244e26) * M_s         # GM_sun * M_s [cm³/s²]
    a_cm = cbrt(GM * P_s * P_s / (4 * T(π)^2))
    return a_cm / (R_s * T(6.9570e10))
end

@inline _planet_inc(::RVOnlyBlock, ::Theta) =
    throw(ArgumentError("planet has no inc — RVOnlyBlock (no astrometry)"))
@inline _planet_inc(::PMOnlyBlock, ::Theta) =
    throw(ArgumentError("planet has no inc — PMOnlyBlock (no astrometry)"))
@inline _planet_inc(::RVPMBlock, ::Theta) =
    throw(ArgumentError("planet has no inc — RVPMBlock (no astrometry)"))

"""
    planet_Omega(theta, k) -> Ω

Longitude of ascending node of planet `k` in radians. Same domain as
`planet_inc`.
"""
@inline planet_Omega(theta::Theta, k::Int) =
    _planet_Omega(theta.params.layout.planet_blocks[k], theta)

@inline _planet_Omega(block::RVASBlock,   theta::Theta) = theta.values[block.Omega]
@inline _planet_Omega(block::RVPMASBlock, theta::Theta) = theta.values[block.Omega]
@inline _planet_Omega(block::SB2Block,    theta::Theta) = theta.values[block.Omega]
@inline _planet_Omega(::RVOnlyBlock, ::Theta) =
    throw(ArgumentError("planet has no Omega — RVOnlyBlock (no astrometry)"))
@inline _planet_Omega(::PMOnlyBlock, ::Theta) =
    throw(ArgumentError("planet has no Omega — PMOnlyBlock (no astrometry)"))
@inline _planet_Omega(::RVPMBlock, ::Theta) =
    throw(ArgumentError("planet has no Omega — RVPMBlock (no astrometry)"))

"""
    astrom_plx(theta) -> plx (mas)

System parallax slot. Returns `0.0` if astrometry is not configured
(systemic.plx slot index is 0). Hot-path callers should check for the
0-slot via `theta.params.layout.systemic.plx == 0` before invoking.
"""
@inline function astrom_plx(theta::Theta{T}) where {T}
    idx = theta.params.layout.systemic.plx
    idx == 0 && return zero(T)
    return theta.values[idx]
end

"""
    astrom_M_pri(theta) -> M_pri (M_sun)

Primary (host) mass slot. Returns `theta.params.config.M_s` if
astrometry is not configured (so existing fixed-mass code paths still
work). When astrometry is on, the slot exists in the layout and can
be either sampled (Gaussian prior) or frozen (FixedPrior).
"""
@inline function astrom_M_pri(theta::Theta{T}) where {T}
    idx = theta.params.layout.systemic.m_pri
    idx == 0 && return convert(T, theta.params.config.M_s)
    return theta.values[idx]
end

"""
    system_vsini(theta) -> V·sin(i_*) (m/s)

Stellar projected rotation. Returns `0.0` if no planet has RM enabled
(slot index is 0). Used by the Hirano+ 2011 RM model.
"""
@inline function system_vsini(theta::Theta{T}) where {T}
    idx = theta.params.layout.systemic.v_sin_i_star
    idx == 0 && return zero(T)
    return theta.values[idx]
end

"""
    system_i_star(theta) -> i_* (rad)

Stellar inclination, 90° = equator-on. Returns `π/2` when the slot is
absent, which makes the gravity-darkened model reduce to the standard one
for any code path that asks without a `:GD` planet present.
"""
@inline function system_i_star(theta::Theta{T}) where {T}
    idx = theta.params.layout.systemic.i_star
    idx == 0 && return T(π / 2)
    return theta.values[idx]
end

"""
    system_gd_beta(theta, ins_idx) -> β_eff

Effective gravity-darkening exponent for photometric instrument `ins_idx`, in
I ∝ g^{4β_eff}. Returns `0.25` (bolometric von Zeipel) when the slot is absent.
Per instrument because gravity darkening is chromatic — see `gd_beta_band`.
"""
@inline function system_gd_beta(theta::Theta{T}, ins_idx::Int) where {T}
    gb = theta.params.layout.systemic.gd_beta
    (isempty(gb) || ins_idx > length(gb) || gb[ins_idx] == 0) && return T(0.25)
    return theta.values[gb[ins_idx]]
end

"""
    system_f_light(theta) -> f_light

Secondary light fraction `f_light = L_B/(L_A+L_B)` in the ASTROMETRIC
band, for the luminous-companion photocenter correction of an SB2
binary. Returns `0.0` (dark companion — the reflex needs no correction)
when the slot is absent (no astrometric SB2 binary).
"""
@inline function system_f_light(theta::Theta{T}) where {T}
    idx = theta.params.layout.systemic.f_light
    idx == 0 && return zero(T)
    return theta.values[idx]
end

"""
    planet_lambda(theta, k) -> λ_k (rad)

Sky-projected obliquity of planet `k`, angle between the projected
stellar spin axis and the projected orbit normal. Looked up by name
from the layout — only present when planet `k` has the `RM` source.
"""
@inline function planet_lambda(theta::Theta{T}, k::Int) where {T}
    name = "lambda_k$k"
    idx = get(theta.params.layout.name_to_idx, name, 0)
    idx == 0 && throw(ArgumentError(
        "planet $k has no λ slot (RM not enabled). Add :RM to its " *
        "PlanetDataSources, e.g. RVPM_RM."))
    return theta.values[idx]
end

"""
    planet_M_sec(theta, k) -> M_sec (M_sun)

Companion mass of planet `k`, in solar masses.

In `:M_sec_driven` mass parametrization, reads the sampled `M_sec_kN`
slot directly. In `:K_driven` mode (default) the slot holds K (m/s);
M_sec is derived by inverting the RV mass function via the
`msec_from_K` helper in `astrometry/projection.jl`.

ForwardDiff-clean.
"""
@inline planet_M_sec(theta::Theta, k::Int) =
    _planet_M_sec(theta.params.layout.planet_blocks[k], theta, k)

@inline _planet_M_sec(::PMOnlyBlock, ::Theta, ::Int) =
    throw(ArgumentError("planet has no M_sec — PMOnlyBlock"))
@inline function _planet_M_sec(block::Union{RVOnlyBlock, RVPMBlock, RVASBlock, RVPMASBlock},
                                theta::Theta, k::Int)
    mass_mode = theta.params.config.parametrization.mass
    if mass_mode === :M_sec_driven || mass_mode === :a_driven
        return theta.values[block.K]   # slot holds M_sec
    else  # :K_driven — slot holds K, derive M_sec
        K     = theta.values[block.K]
        P     = planet_P(theta, k)
        e, _  = planet_e_w(theta, k)
        M_pri = astrom_M_pri(theta)
        sin_i = if block isa Union{RVASBlock, RVPMASBlock}
            sin(_planet_inc(block, theta))
        else
            one(K)   # edge-on assumption (no astrometry → no i constraint)
        end
        return msec_from_K(K, P, e, sin_i, M_pri)
    end
end

# SB2 double-lined binary: BOTH masses are fully determined by the two
# amplitudes + inclination — M_pri is NOT an independent input (using the
# SB1 mass function `msec_from_K(K_A, …, M_pri)` gives an inconsistent
# reflex). From the standard SB2 relations (F = 1/(2πG) in Nereus units):
#   M_total sin³i = F (K_A+K_B)³ P (1−e²)^1.5
#   M_A = M_total · K_B/(K_A+K_B),   M_B = M_total · K_A/(K_A+K_B)
# so M_B/M_total = K_A/(K_A+K_B) = f_mass reproduces the RV amplitudes.
# Edge-on fallback for the RV-only binary (inc slot == 0), which never
# reaches the astrometry path.
@inline function _sb2_masses(theta::Theta, k::Int)
    block = theta.params.layout.planet_blocks[k]
    K_A = theta.values[block.KA]; K_B = theta.values[block.KB]
    P   = planet_P(theta, k)
    e, _ = planet_e_w(theta, k)
    sin_i = block.inc == 0 ? one(K_A) : sin(_planet_inc(block, theta))
    Ksum = K_A + K_B
    omE  = max(1 - e * e, zero(e))^(3/2)
    Mtot_sini3 = _F_M_FACTOR * Ksum^3 * P * omE
    s3   = max(sin_i^3, eps(typeof(sin_i)))
    Mtot = Mtot_sini3 / s3
    return (Mtot * K_B / Ksum, Mtot * K_A / Ksum)   # (M_A, M_B)
end

@inline function _planet_M_sec(block::SB2Block, theta::Theta, k::Int)
    _, M_B = _sb2_masses(theta, k)
    return M_B
end

# =====================================================================
# n_p (active planet count) accessors
# =====================================================================

"""
    n_p(theta) -> Int

Number of currently active planets. Reads the slot at
`params.layout.n_p_idx` and floors to an integer, clamping to
`[0, max_kplanet]`. Works for both fixed-N (`n_p` is frozen) and
trans-dim (`n_p` is sampled as a continuous value).
"""
@inline function n_p(theta::Theta)
    # Trans-dim: active count from mask
    if theta.td !== nothing
        return theta.td.n_planets_active
    end
    # Fixed-dim: read from slot
    idx = theta.params.layout.n_p_idx
    raw = theta.values[idx]
    k = floor(Int, raw)
    if k < 0
        return 0
    elseif k > theta.params.config.max_kplanet
        return theta.params.config.max_kplanet
    end
    return k
end

"""
    planet_indices(theta) -> iterator

Return an iterator over active planet indices. In fixed-dim mode,
returns `1:n_p(theta)`. In trans-dim mode, returns the active mask
indices (may be non-contiguous).
"""
@inline function planet_indices(theta::Theta)
    if theta.td !== nothing
        return active_planets(theta.td)
    end
    return 1:n_p(theta)
end

"""
    set_n_p!(theta, k::Integer)

Set the active planet count directly. Clamps to `[0, max_kplanet]`.
"""
function set_n_p!(theta::Theta, k::Integer)
    mkp = theta.params.config.max_kplanet
    clamped = clamp(k, 0, mkp)
    theta.values[theta.params.layout.n_p_idx] =
        convert(eltype(theta), clamped)
    return theta
end

"""
    is_noise_model_active(theta, idx) -> Bool

Check if noise model at index `idx` is active. Always true in
fixed-dim mode. In trans-dim mode, checks the noise_active mask.
"""
@inline function is_noise_model_active(theta::Theta, idx::Int)
    theta.td === nothing && return true
    # Trans-dim noise sampling may be off (td.noise_active is empty even
    # when noise_models are present). Treat untracked indices as active.
    idx > length(theta.td.noise_active) && return true
    return is_noise_active(theta.td, idx)
end

# =====================================================================
# RV systemic accessors
# =====================================================================

@inline rv_gamma(theta::Theta, ins_idx::Int) =
    theta.values[theta.params.layout.systemic.rv_gamma[ins_idx]]

@inline rv_sigma(theta::Theta, ins_idx::Int) =
    theta.values[theta.params.layout.systemic.rv_sigma[ins_idx]]

# =====================================================================
# RV trend accessors
# =====================================================================

@inline function rv_dvdt(theta::Theta)
    idx = theta.params.layout.systemic.dvdt
    return theta.values[idx]
end

@inline function rv_d2vdt2(theta::Theta)
    idx = theta.params.layout.systemic.d2vdt2
    return theta.values[idx]
end

# =====================================================================
# PM systemic accessors
# =====================================================================

@inline pm_offset(theta::Theta, ins_idx::Int) =
    theta.values[theta.params.layout.systemic.pm_offset[ins_idx]]

@inline pm_jitter(theta::Theta, ins_idx::Int) =
    theta.values[theta.params.layout.systemic.pm_jitter[ins_idx]]

@inline pm_dilution(theta::Theta, ins_idx::Int) =
    theta.values[theta.params.layout.systemic.pm_dilution[ins_idx]]

# Photometric baseline-trend coefficients for instrument `ins_idx` — the slot
# index vector [c1, …, c_order] (empty when phot_trend_order == 0).
@inline pm_trend_slots(theta::Theta, ins_idx::Int) =
    theta.params.layout.systemic.pm_trend[ins_idx]

# p-th trend coefficient value (1-based); zero past the configured order.
@inline function pm_trend_coef(theta::Theta{T}, ins_idx::Int, p::Int) where {T}
    slots = theta.params.layout.systemic.pm_trend[ins_idx]
    p <= length(slots) ? theta.values[slots[p]] : zero(T)
end

# =====================================================================
# rho_s accessor
# =====================================================================

"""
    rho_s(theta) -> value

Shared stellar density. Errors if `parametrization.use_rho_s` is
false (in which case the slot does not exist in the layout).
"""
@inline function rho_s(theta::Theta)
    idx = theta.params.layout.systemic.rho_s
    idx == 0 && throw(ArgumentError(
        "rho_s is not in use (parametrization.use_rho_s == false)"))
    return theta.values[idx]
end

# =====================================================================
# Log-prior evaluation
# =====================================================================

"""
    log_prior(theta) -> value

Sum of log-priors over all unfrozen slots. Frozen slots contribute
zero. Returns `-Inf` (appropriately typed) as soon as any unfrozen
value is out of its prior's support.

Uses the precomputed `params.layout.unfrozen_priors` vector — no
dict lookup, no name resolution in the loop. This is the type-stable
hot path that autodiff gradients flow through.

The return type tracks `eltype(theta)` — for a `Theta{Dual{…}}`
the output is a `Dual`, for `Theta{Float64}` it's a `Float64`.
"""
function log_prior(theta::Theta{T}) where {T}
    if theta.td === nothing
        # Fixed-dim: evaluate all unfrozen priors (hot path, Enzyme-compatible)
        return log_prior_packed(
            theta.values,
            theta.params.layout.unfrozen_idx,
            theta.params.layout.packed_priors,
        )
    else
        # Trans-dim: skip priors for inactive planet/noise slots
        return _log_prior_transdim(theta)
    end
end

"""
    _log_prior_transdim(theta) -> T

Log-prior that skips parameters belonging to inactive planets and
noise models. Builds a skip-set from the TransDimState, then
evaluates packed priors only for active slots.
"""
function _log_prior_transdim(theta::Theta{T}) where {T}
    layout = theta.params.layout
    td = theta.td
    packed = layout.packed_priors

    # Build set of theta-slot indices to skip (inactive planets)
    skip_slots = Set{Int}()
    for k in 1:length(td.planet_active)
        if !td.planet_active[k]
            for idx in planet_slot_indices(layout.planet_blocks[k])
                push!(skip_slots, idx)
            end
        end
    end

    # Build set of noise param slot indices to skip (inactive noise)
    # Noise params are named and indexed via the layout. We need to
    # identify which unfrozen indices belong to inactive noise models.
    noise_models = theta.params.config.noise_models
    if !isempty(noise_models)
        nm_instruments = theta.params.config.instruments
        slot_offset = 0
        # When trans-dim noise sampling is OFF (td.noise == false),
        # `td.noise_active` is an empty BitVector — we must treat
        # ALL noise models as active and not skip any of their slots.
        n_noise_tracked = length(td.noise_active)
        for (nm_idx, nm) in enumerate(noise_models)
            nm_names = noise_param_names(nm, nm_instruments)
            inactive = nm_idx <= n_noise_tracked && !is_noise_active(td, nm_idx)
            if inactive
                for name in nm_names
                    if haskey(layout.name_to_idx, name)
                        push!(skip_slots, layout.name_to_idx[name])
                    end
                end
            end
        end
    end

    # Evaluate priors, skipping inactive slots
    total = zero(T)
    unfrozen_idx = layout.unfrozen_idx
    @inbounds for i in eachindex(unfrozen_idx)
        slot = unfrozen_idx[i]
        slot in skip_slots && continue
        x = theta.values[slot]
        tid = packed.type_ids[i]
        p1 = packed.params[i, 1]
        p2 = packed.params[i, 2]
        lo = packed.lowers[i]
        hi = packed.uppers[i]
        lp = eval_packed_logpdf(x, tid, p1, p2, lo, hi)
        if !isfinite(lp)
            return convert(T, -Inf)
        end
        total += lp
    end
    return total
end
