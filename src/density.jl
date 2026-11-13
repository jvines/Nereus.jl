# The photo-eccentric reparametrization (`density = :rho_circ`).
#
# A transit measures its duration, which fixes the circular-equivalent density
# ρ_circ = ρ★ g³, g = (1 + e sin ω)/√(1 − e²) (Kipping 2010; Dawson & Johnson
# 2012). With e free and no density prior that settles it, the posterior in
# (ρ★, e, ω) is a thin curved sheet along constant ρ_circ reaching high e, and
# the ensembles cross it over thousands of steps without R-hat, ESS or the drift
# check noticing: on a transit-only fit pt_emcee put 0.7-4% of its draws above
# e = 0.5 against nested sampling's 6.6-9%. Sampling ρ_circ lays the sheet along
# an axis. `DensityReparam` (src/model.jl) has the coordinates and the map; this
# file builds the plan, evaluates the map, its Jacobian and the prior it carries,
# and supplies the accessors that read the derived coordinates.
#
# Everything a likelihood reads goes through `planet_e_w` and `_rho_star`, so no
# model code sees the new coordinates: at matching physical points the likelihood
# is the same function. The prior is the model's prior on the derived
# coordinates times |∂old/∂new|, so the posterior is the same distribution.

using MCMCChains

# g ∈ [1/G, G]: the sampled box of ρ_circ is [ρ★_min/G³, ρ★_max·G³]. The orbits
# it leaves out have e within ~2/G² of 1 (old-prior mass ≈ 3/(4G²) under a
# uniform e, 7.5e-13 here), where the planet passes through the star.
const DENSITY_G = 1.0e6

# η's interval: both ends are the point e = 1, ω = -90° of every ellipse, where
# the density of the map vanishes, so the walls are never felt.
const _ETA_LO = -π / 2
const _ETA_HI = 3π / 2

# The part of that interval an e-prior that stops at e_max < 1 needs. Along an
# ellipse of constant g, e² = (1 + g² + g⁴ − 2g² sin η − g² sin²η)/(1 + g²)²
# falls as sin η rises, so e grows with |η − π/2|, from |1 − g²|/(1 + g²) at
# η = π/2, and e ≤ e_max is the arc sin η ≥ (g + 1/g)·√(1 − e_max²) − 1. That arc
# is widest at g = 1, so π/2 ± (π/2 − asin(2√(1 − e_max²) − 1)) holds the orbits
# of every g. Sampling the whole interval instead would put most of η's box
# outside the prior: with |sesinw|, |secosw| ≤ 0.3, a point 0.1 from the middle
# of the unconstrained space (e = 0) was already at e = 0.1, secosw = 0.31.
function _density_eta_range(e_max::Float64)
    e_max < 1 || return (Float64(_ETA_LO), Float64(_ETA_HI))
    s = 2 * sqrt(1 - e_max^2) - 1
    Δ = π / 2 - asin(clamp(s, -1.0, 1.0))
    Δ > 0 || return (Float64(_ETA_LO), Float64(_ETA_HI))
    return (π / 2 - Δ, π / 2 + Δ)
end

# ρ★ = M★ / R★³ in the solar units of `rho_s`: `RHO_SUN_CGS` is the mean density
# of a 1 M☉, 1 R☉ star, so this is the density whose a/R★ (`rho_s_to_a_Rs`) is
# the (M★, R★) route's (`planet_geometry`).
@inline _rho_from_mass_radius(M, R) = M / R^3

# The finite range a prior spans, for the sampled box: its bounds, or where a
# bound is open or at zero (a density has none below 0), the quantile 1e-15
# from that end.
function _density_range(ps::PriorSpec)
    lo, hi = bounds(ps)
    isfinite(hi) || (hi = Float64(quantile(ps, 1 - 1e-15)))
    if !(isfinite(lo) && lo > 0)
        lo = Float64(quantile(ps, 1e-15))
        lo > 0 || (lo = 1e-12 * hi)
    end
    return (Float64(lo), Float64(hi))
end

# The largest e the e-coordinates' priors allow (capped at 1).
function _density_e_max(ew::Symbol, p1::PriorSpec, p2::PriorSpec)
    lo1, hi1 = bounds(p1)
    lo2, hi2 = bounds(p2)
    if ew === :ew
        return isfinite(hi1) ? min(Float64(hi1), 1.0) : 1.0
    end
    m1 = max(lo1^2, hi1^2)
    m2 = max(lo2^2, hi2^2)
    s = ew === :sesinw ? m1 + m2 : sqrt(m1 + m2)
    return isfinite(s) ? min(Float64(s), 1.0) : 1.0
end

# g bound of one planet: DENSITY_G, or tighter where its e-prior stops short of 1.
_density_G(e_max::Float64) =
    e_max < 1 ? min(DENSITY_G, sqrt((1 + e_max) / (1 - e_max))) : DENSITY_G

"""
    density_reparam_names(k; anchor = false) -> Vector{String}

The sampled slot names planet `k` gets under `density = :rho_circ`:
`rho_circ_k<k>` alone for the anchor, `rho_circ_k<k>` and `eta_k<k>` otherwise.
"""
density_reparam_names(k::Integer; anchor::Bool = false) =
    anchor ? ["rho_circ_k$k"] : ["rho_circ_k$k", "eta_k$k"]

const _DENSITY_NAME = r"^(rho_circ|eta)_k\d+$"

"""
    is_density_reparam_name(name) -> Bool

Whether `name` is a slot the photo-eccentric reparametrization adds
(`rho_circ_k<k>`, `eta_k<k>`). Their priors are not the user's to give: they
carry the prior on ρ★ and (e, ω) through the Jacobian.
"""
is_density_reparam_name(name) = occursin(_DENSITY_NAME, String(name))

# Decide the plan and push its sampled slots onto `names` (called by
# `_build_layout`, beside the density slot). Returns the slot indices; the
# derived slots' priors are packed by `_density_plan` once the layout has them.
function _density_slots!(names::Vector{String}, config, planet_blocks, rho_s_idx::Int)
    K = config.max_kplanet
    none = (mode = :none, anchor = 0, rho_circ = Int[], eta = Int[], derived = Int[],
            old_e = NTuple{2,Int}[], old_rho = 0, rho_slot = rho_s_idx,
            lnorm = Float64[])
    pc = config.parametrization
    pc.density === :rho_circ || return none
    # a/R★ sampled per planet: no shared density to carry; not reparametrized.
    pc.sample_a_Rs && return none
    pr = config.priors

    # A planet is eligible when it has transit photometry, transit geometry, and
    # both its e-coordinates free.
    elig = Int[]
    for k in 1:K
        has_pm(config.planet_modes[k]) || continue
        b = planet_blocks[k]
        has_geometry(b) || continue
        n1, n2 = names[b.e1], names[b.e2]
        (haskey(pr, n1) && haskey(pr, n2)) || continue
        (is_fixed(pr[n1]) || is_fixed(pr[n2])) && continue
        push!(elig, k)
    end
    isempty(elig) && return none

    # Where ρ★ comes from, and the range it spans.
    if pc.use_rho_s
        haskey(pr, "rho_s") || return none
        if is_fixed(pr["rho_s"])
            mode = :given
            v = Float64(fixed_value(pr["rho_s"]))
            rho_lo = rho_hi = v
        else
            mode = :anchor
            rho_lo, rho_hi = _density_range(pr["rho_s"])
        end
    else
        # ρ★ = M★ / R★³: the host mass (sampled `M_pri` or the configured M_s)
        # and the fixed radius. Without both there is no a/R★, and no density.
        R = config.R_s
        (isfinite(R) && R > 0) || return none
        has_mpri = any(has_as, config.planet_modes) && haskey(pr, "M_pri")
        if has_mpri && !is_fixed(pr["M_pri"])
            M_lo, M_hi = _density_range(pr["M_pri"])
        else
            M = has_mpri ? Float64(fixed_value(pr["M_pri"])) : config.M_s
            (isfinite(M) && M > 0) || return none
            M_lo = M_hi = M
        end
        mode = :given
        rho_lo, rho_hi = _rho_from_mass_radius(M_lo, R), _rho_from_mass_radius(M_hi, R)
    end
    anchor = mode === :anchor ? first(elig) : 0

    rho_circ = zeros(Int, K)
    eta = zeros(Int, K)
    lnorm = zeros(K)
    for k in elig
        b = planet_blocks[k]
        e_max = _density_e_max(pc.ew, pr[names[b.e1]], pr[names[b.e2]])
        G = _density_G(e_max)
        lo, hi = rho_lo / G^3, rho_hi * G^3
        push!(names, "rho_circ_k$k")
        rho_circ[k] = length(names)
        _density_surrogate!(pr, "rho_circ_k$k", LogUniformPrior(lo, hi))
        lnorm[k] = log(log(hi / lo))
        k == anchor && continue
        push!(names, "eta_k$k")
        eta[k] = length(names)
        η_lo, η_hi = _density_eta_range(e_max)
        _density_surrogate!(pr, "eta_k$k", UniformPrior(η_lo, η_hi))
        lnorm[k] += log(η_hi - η_lo)
    end

    derived = Int[]
    mode === :anchor && push!(derived, rho_s_idx)
    for k in elig
        eta[k] > 0 || continue
        b = planet_blocks[k]
        push!(derived, b.e1, b.e2)
    end
    sort!(derived)
    pos(i) = findfirst(==(i), derived)
    old_e = NTuple{2,Int}[eta[k] > 0 ? (pos(planet_blocks[k].e1), pos(planet_blocks[k].e2)) :
                          (0, 0) for k in 1:K]
    old_rho = mode === :anchor ? pos(rho_s_idx) : 0
    return (mode = mode, anchor = anchor, rho_circ = rho_circ, eta = eta,
            derived = derived, old_e = old_e, old_rho = old_rho,
            rho_slot = rho_s_idx, lnorm = lnorm)
end

# Put the surrogate prior of a sampled slot into the config's priors. A prior
# already there is refused unless it is this same surrogate (a layout rebuilt
# from its own config): the prior of ρ_circ and η is the one carried from ρ★
# and (e, ω), so a prior meant for them belongs on those.
function _density_surrogate!(pr, name::String, ps::PriorSpec)
    if haskey(pr, name) && pr[name] != ps
        throw(ArgumentError(
            "$name is sampled by the photo-eccentric reparametrization " *
            "(density = :rho_circ), and its prior is the one the model puts on the " *
            "stellar density and the eccentricity, carried through the change of " *
            "variables. Put the prior on rho_s (or M_pri) and the e-coordinates, or " *
            "set the parametrization's density = :rho_star to sample ρ★ and (e, ω) " *
            "directly."))
    end
    pr[name] = ps
    return pr
end

# The plan, with the derived slots' priors (as the layout samples them) packed.
function _density_plan(ds, derived_priors::Vector{PriorSpec})
    ds.mode === :none && return NO_DENSITY_REPARAM
    return DensityReparam(ds.mode, ds.anchor, ds.rho_circ, ds.eta, ds.derived,
                          pack_priors(derived_priors), derived_priors, ds.old_e,
                          ds.old_rho, ds.rho_slot, ds.lnorm, DENSITY_G)
end

"""
    density_reparam(params) -> DensityReparam

The photo-eccentric reparametrization of `params` (`NO_DENSITY_REPARAM` when
none: `density = :rho_star`, or no transiting planet with a free e).
"""
density_reparam(params::Params) = params.layout.density

"""
    density_active(params) -> Bool

Whether `params` samples ρ_circ (see `DensityReparam`).
"""
density_active(params::Params) = params.layout.density.mode !== :none

"""
    density_derived_names(params) -> Vector{String}

The coordinates the reparametrization derives (`rho_s` under `:anchor`, the
e-coordinates of every planet sampling `eta_k`), in layout order. Their values
are in every chain the samplers return, beside the sampled slots.
"""
density_derived_names(params::Params) =
    String[params.layout.names[i] for i in params.layout.density.derived]
# Anything else standing in for a Params (a layout of names, as the summary
# helpers accept) carries no reparametrization: nothing is derived.
density_derived_names(_) = String[]

"""
    reported_param_names(params) -> Vector{String}

The parameters a fit reports per draw: the sampled ones (`unfrozen_names`) and
then the coordinates the photo-eccentric reparametrization derives from them
(`density_derived_names`), so ρ★ and the e-coordinates are tabulated and plotted
where they were when they were sampled. `unfrozen_names` without it.
"""
reported_param_names(params::Params) =
    density_active(params) ? vcat(params.layout.unfrozen_names, density_derived_names(params)) :
                             params.layout.unfrozen_names

"""
    density_planet_names(params) -> Dict{String, Int}

The planet each reparametrization name belongs to: `rho_circ_k<k>`, `eta_k<k>`
and the e-coordinates an η planet derives. ρ★ belongs to none.
"""
function density_planet_names(params::Params)
    out = Dict{String, Int}()
    d = params.layout.density
    d.mode === :none && return out
    nm = params.layout.names
    for k in eachindex(d.rho_circ)
        d.rho_circ[k] > 0 && (out[nm[d.rho_circ[k]]] = k)
        if d.eta[k] > 0
            out[nm[d.eta[k]]] = k
            b = params.layout.planet_blocks[k]
            out[nm[b.e1]] = k; out[nm[b.e2]] = k
        end
    end
    return out
end

"""
    density_shared_names(params) -> Vector{String}

The sampled coordinates every planet's orbit reads through the
reparametrization: under `:anchor`, the anchor's `rho_circ_k<j>` and its two
e-coordinates, from which ρ★ follows (`_anchor_rho`). Empty otherwise (under
`:given`, ρ★ is fixed or M★/R★³ with the host mass a slot of its own).
"""
function density_shared_names(params::Params)
    d = params.layout.density
    d.mode === :anchor || return String[]
    j = d.anchor; nm = params.layout.names; b = params.layout.planet_blocks[j]
    return String[nm[i] for i in (d.rho_circ[j], b.e1, b.e2) if nm[i] in params.layout.unfrozen_names]
end

# =====================================================================
# The map
# =====================================================================

# g = (1 + e sin ω)/√(1 − e²): what the transit duration divides.
@inline _duration_factor(e, sinw) = (1 + e * sinw) / sqrt(1 - e * e)

# The point of the g-ellipse at η, in the (e cos ω, e sin ω) disc.
@inline function _ellipse_xy(c, η)
    c2 = c * c
    s, co = sincos(η)
    A = c / sqrt(1 + c2)
    return A * co, (c2 * s - 1) / (1 + c2)
end

# log |∂(x, y)/∂(g, η)| = log[g² (1 + sin η)² / (1 + g²)^{5/2}].
@inline _ellipse_logjac(c, η) = 2 * log(c) + 2 * log1p(sin(η)) - 2.5 * log1p(c * c)

# (g, η) of a disc point (x, y), e < 1: the inverse of `_ellipse_xy`.
@inline function _ellipse_c_eta(x, y)
    c = (1 + y) / sqrt(1 - x * x - y * y)
    c2 = c * c
    A = c / sqrt(1 + c2)
    B = c2 / (1 + c2)
    η = atan((y + 1 / (1 + c2)) / B, x / A)
    η < _ETA_LO && (η += 2π)
    return c, η
end

# The e-coordinates of `ew` at (x, y), and log |∂(o₁, o₂)/∂(x, y)|. ω is put in
# [wlo, wlo + 2π) for `:ew`. At e = 0 exactly (a point) the density is returned
# as zero rather than its integrable infinity.
@inline function _old_from_xy(ew::Symbol, x, y, wlo::Float64)
    e = sqrt(x * x + y * y)
    if iszero(e)
        z = zero(e)
        return z, z, oftype(e, -Inf)
    end
    if ew === :sesinw
        r = sqrt(e)
        return y / r, x / r, -log(2 * e)
    elseif ew === :esinw
        return y, x, zero(e)
    else
        w = atan(y, x)
        w = wlo + mod(w - wlo, oftype(w, 2π))
        return e, w, -log(e)
    end
end

# (x, y) from the e-coordinates of `ew`.
@inline function _xy_from_old(ew::Symbol, o1, o2)
    if ew === :sesinw
        r = sqrt(o1 * o1 + o2 * o2)       # √e
        return r * o2, r * o1
    elseif ew === :esinw
        return o2, o1
    else
        s, c = sincos(o2)
        return o1 * c, o1 * s
    end
end

# Lower edge of the window a derived ω is reported and evaluated in (`:ew`).
@inline _dplan_wlo(d::DensityReparam, j::Int) = d.derived_priors.lowers[j]

# The ρ★ the model holds fixed or derives from other coordinates (`:given`):
# the frozen `rho_s`, else M★/R★³.
@inline function _given_rho(theta::Theta{T}) where {T}
    d = theta.params.layout.density
    d.rho_slot > 0 && return theta.values[d.rho_slot]
    return _rho_from_mass_radius(astrom_M_pri(theta), convert(T, theta.params.config.R_s))
end

# ρ★ under `:anchor`: ρ_circ of the anchor over its g³.
@inline function _anchor_rho(theta::Theta)
    d = theta.params.layout.density
    j = d.anchor
    blk = theta.params.layout.planet_blocks[j]
    e = _ecc_from_block(blk, theta)
    e < 1 || return oftype(e, NaN)
    g = _duration_factor(e, _sinw_from_block(blk, theta))
    return theta.values[d.rho_circ[j]] / g^3
end

"""
    _rho_star(theta) -> ρ★

The stellar density of a `use_rho_s` model in solar units: the `rho_s` slot, or
under the photo-eccentric reparametrization's `:anchor` mode the anchor's
ρ_circ / g³.
"""
@inline function _rho_star(theta::Theta)
    d = theta.params.layout.density
    d.mode === :anchor && return _anchor_rho(theta)
    return theta.values[theta.params.layout.systemic.rho_s]
end

# The ρ★ an η planet's g is measured against.
@inline _eta_ref_rho(theta::Theta) =
    theta.params.layout.density.mode === :anchor ? _anchor_rho(theta) : _given_rho(theta)

# (x, y, g, η) of an η planet.
@inline function _eta_planet_xy(theta::Theta, k::Int, rho_ref)
    d = theta.params.layout.density
    c = cbrt(theta.values[d.rho_circ[k]] / rho_ref)
    η = theta.values[d.eta[k]]
    x, y = _ellipse_xy(c, η)
    return x, y, c, η
end

# (e, ω) of an η planet, ω in (-π, π] as `sesinw_to_ew` gives it.
@inline function _eta_planet_e_w(theta::Theta, k::Int)
    x, y, _, _ = _eta_planet_xy(theta, k, _eta_ref_rho(theta))
    return sqrt(x * x + y * y), atan(y, x)
end

# (e, sin ω) of an η planet.
@inline function _eta_planet_e_sinw(theta::Theta, k::Int)
    x, y, _, _ = _eta_planet_xy(theta, k, _eta_ref_rho(theta))
    e = sqrt(x * x + y * y)
    return e, iszero(e) ? zero(e) : y / e
end

# Whether the sampled slots hold a point of the map (positive finite ρ_circ, η
# on its interval, a finite positive ρ★): not so in a Theta just built, whose
# unfrozen slots are zero.
function _density_sampled_ok(theta::Theta)
    d = theta.params.layout.density
    d.mode === :none && return true
    v = theta.values
    for k in eachindex(d.rho_circ)
        d.rho_circ[k] > 0 || continue
        r = v[d.rho_circ[k]]
        (isfinite(r) && r > 0) || return false
        if d.eta[k] > 0
            η = v[d.eta[k]]
            (isfinite(η) && _ETA_LO < η < _ETA_HI) || return false
        end
    end
    ρ = d.mode === :anchor ? _anchor_rho(theta) : _given_rho(theta)
    return isfinite(ρ) && ρ > 0
end

@inline _is_eta_planet(d::DensityReparam, k::Int) =
    d.mode !== :none && 1 <= k <= length(d.eta) && d.eta[k] > 0

# =====================================================================
# The prior
# =====================================================================

"""
    density_logprior_correction(theta) -> T

What `log_prior` adds to the sampled slots' packed (surrogate) priors so that the
prior on the sampled coordinates is the model's prior on the derived ones carried
through the map:

    log π_new = log π_old(old(new)) + log |∂old/∂new|.

The surrogate is log-uniform on each `rho_circ_k` box and uniform on η's
interval, so per planet (A_k = log log(hi/lo) of its box; `lnorm[k]` holds A_k,
plus log(η_hi − η_lo) for an η planet):

  * the anchor: log π(ρ★) + log ρ★ + A_k, with ρ★ = ρ_circ/g³ (|∂ρ★/∂ρ_circ| = g⁻³);
  * an η planet: log π(o₁) + log π(o₂) + log|∂o/∂(x, y)| +
    log|∂(x, y)/∂(g, η)| + log(g/3) + A_k + log(η_hi − η_lo), at the model's ρ★
    (|∂g/∂ρ_circ| = g/(3 ρ_circ)), with o the planet's e-coordinates.

Under `:anchor` the map is triangular (ρ★ from the anchor, then each η planet's
orbit at that ρ★), so the determinant is the product of these. Zero for a model
without the reparametrization; `-Inf` outside the old prior's support.
"""
function density_logprior_correction(theta::Theta{T}) where {T}
    d = theta.params.layout.density
    d.mode === :none && return zero(T)
    ninf = convert(T, -Inf)
    pp = d.derived_priors
    lp = zero(T)
    if d.mode === :anchor
        rho = _anchor_rho(theta)
        (isfinite(rho) && rho > 0) || return ninf
        lr = eval_packed_logpdf(rho, pp, d.old_rho)
        isfinite(lr) || return ninf
        lp += lr + log(rho) + d.lnorm[d.anchor]
    end
    any_eta = false
    for k in eachindex(d.eta)
        d.eta[k] > 0 || continue
        any_eta = true
        break
    end
    any_eta || return lp
    rho_ref = _eta_ref_rho(theta)
    (isfinite(rho_ref) && rho_ref > 0) || return ninf
    ew = theta.params.config.parametrization.ew
    for k in eachindex(d.eta)
        d.eta[k] > 0 || continue
        x, y, c, η = _eta_planet_xy(theta, k, rho_ref)
        j1, j2 = d.old_e[k]
        o1, o2, lj = _old_from_xy(ew, x, y, _dplan_wlo(d, j2))
        isfinite(lj) || return ninf
        x * x + y * y < 1 || return ninf
        l1 = eval_packed_logpdf(o1, pp, j1)
        l2 = eval_packed_logpdf(o2, pp, j2)
        (isfinite(l1) && isfinite(l2)) || return ninf
        lp += l1 + l2 + lj + _ellipse_logjac(c, η) + log(c / 3) + d.lnorm[k]
    end
    return isfinite(lp) ? lp : ninf
end

# =====================================================================
# Physical values in, sampled values out, and back
# =====================================================================

"""
    density_physical!(theta) -> theta

Write the value of every derived coordinate (`density_derived_names`) into its
slot, from the sampled ones. Nothing reads those slots in a likelihood; this is
for reporting and for code that reads a slot by name.
"""
function density_physical!(theta::Theta{T}) where {T}
    d = theta.params.layout.density
    d.mode === :none && return theta
    v = theta.values
    d.mode === :anchor && (v[d.derived[d.old_rho]] = _anchor_rho(theta))
    ew = theta.params.config.parametrization.ew
    rho_ref = _eta_ref_rho(theta)
    for k in eachindex(d.eta)
        d.eta[k] > 0 || continue
        x, y, _, _ = _eta_planet_xy(theta, k, rho_ref)
        j1, j2 = d.old_e[k]
        o1, o2, _ = _old_from_xy(ew, x, y, _dplan_wlo(d, j2))
        v[d.derived[j1]] = o1
        v[d.derived[j2]] = o2
    end
    return theta
end

"""
    density_from_physical!(theta) -> theta

The inverse of `density_physical!`: with the derived slots holding a physical
point (ρ★, the e-coordinates), set the sampled `rho_circ_k` and `eta_k` slots to
the same point. A point with e ≥ 1 sets them to NaN, which every prior rejects.
"""
function density_from_physical!(theta::Theta{T}) where {T}
    d = theta.params.layout.density
    d.mode === :none && return theta
    v = theta.values
    ew = theta.params.config.parametrization.ew
    rho = d.mode === :anchor ? v[d.derived[d.old_rho]] : _given_rho(theta)
    if d.mode === :anchor
        j = d.anchor
        blk = theta.params.layout.planet_blocks[j]
        e = _ecc_from_block(blk, theta)
        v[d.rho_circ[j]] = e < 1 ? rho * _duration_factor(e, _sinw_from_block(blk, theta))^3 :
                                   convert(T, NaN)
    end
    for k in eachindex(d.eta)
        d.eta[k] > 0 || continue
        j1, j2 = d.old_e[k]
        x, y = _xy_from_old(ew, v[d.derived[j1]], v[d.derived[j2]])
        if x * x + y * y < 1
            c, η = _ellipse_c_eta(x, y)
            v[d.rho_circ[k]] = rho * c^3
            v[d.eta[k]] = η
        else
            v[d.rho_circ[k]] = convert(T, NaN)
            v[d.eta[k]] = convert(T, NaN)
        end
    end
    return theta
end

"""
    density_prior_draw!(theta, rng) -> theta

Draw the derived coordinates from their priors (the model's prior on ρ★ and the
e-coordinates) and set the sampled slots from them (`density_from_physical!`):
a draw from the prior in the sampled coordinates, as the surrogate box is not.
"""
function density_prior_draw!(theta::Theta, rng::AbstractRNG)
    d = theta.params.layout.density
    d.mode === :none && return theta
    for (j, i) in enumerate(d.derived)
        theta.values[i] = quantile(d.derived_specs[j], rand(rng))
    end
    return density_from_physical!(theta)
end

"""
    density_prior_draw_x!(x, params, rng) -> x

`x`, a point of the sampled (unfrozen) coordinates, with its `rho_circ_k` and
`eta_k` entries redrawn from the prior they carry (`density_prior_draw!`) at the
other coordinates of `x`. `x` unchanged without the reparametrization.
"""
function density_prior_draw_x!(x::AbstractVector, params::Params, rng::AbstractRNG)
    density_active(params) || return x
    th = Theta{Float64}(params)
    set_unfrozen!(th, x)
    density_prior_draw!(th, rng)
    @inbounds for (j, idx) in enumerate(params.layout.unfrozen_idx)
        x[j] = th.values[idx]
    end
    return x
end

# The derived coordinates' draws for a chain: `vals[:, j]` is the j-th derived
# name at each draw, `X` the sampled values (draws × unfrozen).
function _density_derived_draws(params::Params, X::AbstractMatrix{<:Real})
    d = params.layout.density
    uf = params.layout.unfrozen_idx
    n = size(X, 1)
    out = Matrix{Float64}(undef, n, length(d.derived))
    nslot = _nthread_slots()
    thetas = [Theta{Float64}(params) for _ in 1:nslot]
    Threads.@threads :static for i in 1:n
        th = thetas[Threads.threadid()]
        @inbounds for (j, idx) in enumerate(uf)
            th.values[idx] = X[i, j]
        end
        density_physical!(th)
        @inbounds for (j, idx) in enumerate(d.derived)
            out[i, j] = th.values[idx]
        end
    end
    return out
end

"""
    with_density_columns(chains, params) -> chains

`chains` with a parameter column added for every coordinate the
photo-eccentric reparametrization derives (`density_derived_names`: `rho_s`,
the e-coordinates of the planets sampling `eta_k`) that it lacks, computed per
draw from the sampled columns. Every fixed-dimension sampler passes its chain
through this, so ρ★ and (e, ω) stay in the chain, `chains.nc`, the tables and
the corner as they were before ρ_circ was sampled. `chains` itself without the
reparametrization, or when a sampled column is missing.
"""
function with_density_columns(ch::MCMCChains.Chains, params::Params)
    density_active(params) || return ch
    pn = MCMCChains.names(ch, :parameters)
    pnames = String.(pn)
    dn = density_derived_names(params)
    add = [nm for nm in dn if !(nm in pnames)]
    isempty(add) && return ch
    cols = Int[]
    allv = MCMCChains.names(ch)
    for nm in params.layout.unfrozen_names
        j = findfirst(==(Symbol(nm)), allv)
        j === nothing && return ch
        push!(cols, j)
    end
    V = ch.value.data
    ni, _, nc = size(V)
    X = Matrix{Float64}(undef, ni * nc, length(cols))
    for c in 1:nc, (jj, j) in enumerate(cols)
        @views X[(c - 1) * ni + 1:c * ni, jj] .= V[:, j, c]
    end
    D = _density_derived_draws(params, X)
    keep = [findfirst(==(nm), dn) for nm in add]
    E = Array{Float64}(undef, ni, length(add), nc)
    for c in 1:nc, (jj, j) in enumerate(keep)
        @views E[:, jj, c] .= D[(c - 1) * ni + 1:c * ni, j]
    end
    newv = cat(Array{Float64}(V), E; dims = 2)
    newnames = vcat(allv, Symbol.(add))
    nmap = Dict{Symbol, Vector{Symbol}}(k => copy(v) for (k, v) in pairs(ch.name_map))
    append!(nmap[:parameters], Symbol.(add))
    return MCMCChains.Chains(newv, newnames, (; nmap...); iterations = collect(range(ch)),
                             evidence = ch.logevidence, info = ch.info)
end

"""
    without_density_reparam(params, data = nothing) -> Params

`params` with `density = :rho_star`: the model sampled in ρ★ and (e, ω). The
same model; for the engines that run in those coordinates (the trans-dimensional
samplers, MAP). `params` itself when it has no reparametrization.
"""
function without_density_reparam(params::Params, data = nothing)
    density_active(params) || return params
    c = params.config
    pr = Dict{String, PriorSpec}(k => v for (k, v) in c.priors
                                 if !is_density_reparam_name(k))
    cfg = ParamsConfig(c.max_kplanet, _with_density(c.parametrization, :rho_star),
                       c.planet_modes, c.instruments, pr, c.trend_order,
                       c.phot_trend_order, c.noise_models, c.stability, c.M_s, c.R_s,
                       c.external_priors, c.sharing, c.ttv_n_transits,
                       c.ttv_backend, c.obliquity)
    return Params(cfg, _build_layout(cfg; data = data))
end

"""
    density_old_x(x, params, p0) -> Vector{Float64}

A point `x` of `params`' sampled coordinates as a point of `p0 =
without_density_reparam(params)`'s: the same orbit in ρ★ and (e, ω).
"""
function density_old_x(x::AbstractVector, params::Params, p0::Params)
    th = Theta{Float64}(params)
    set_unfrozen!(th, x)
    density_physical!(th)
    idx = params.layout.name_to_idx
    return Float64[th.values[idx[nm]] for nm in p0.layout.unfrozen_names]
end

"""
    density_new_x(x0, p0, params) -> Vector{Float64}

The inverse of `density_old_x`: a point `x0` of `p0`'s coordinates (ρ★, (e, ω))
as a point of `params`' sampled ones (ρ_circ, η).
"""
function density_new_x(x0::AbstractVector, p0::Params, params::Params)
    th = Theta{Float64}(params)
    idx = params.layout.name_to_idx
    for (j, nm) in enumerate(p0.layout.unfrozen_names)
        th.values[idx[nm]] = x0[j]
    end
    density_from_physical!(th)
    return unfrozen_values(th)
end

"""
    density_chart_key(params) -> NamedTuple

The checkpoint-fingerprint entry of a model that samples ρ_circ: a checkpoint
written before holds its walkers in ρ★ and (e, ω), which the sampler would read
as ρ_circ and η. Empty for a model without the reparametrization, whose
checkpoints stay valid.
"""
density_chart_key(params::Params) =
    density_active(params) ? (; density = :rho_circ) : (;)
