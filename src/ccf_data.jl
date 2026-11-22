# Cross-correlation-function profiles fitted as a sum of Gaussian line
# components (`data.ccf`).
#
# A `CCFSet` holds the CCFs of one instrument (one per epoch, each on its own
# velocity grid) and states the line components they are modelled with. Each
# component is either STATIC (its centre is a constant γ_j) or tied to the
# orbit of one of the model's planets (its centre is γ_j + K_j·g_k(t), with
# g_k = cos(f + ω) + e cos ω computed from that planet's P, time anchor, e and
# ω -- the same orbit the RVs, the transit and the transit times read). The
# likelihood is in src/ccf_likelihood.jl; docs/src/ccf.md has the model and the
# job-config form.
#
# Like the residual maps of Doppler tomography, the set carries the choices
# that decide which nuisance parameters exist (the components, how the
# continuum, depth scale and jitter are shared), and `_build_layout` names the
# parameters from it.

using JSON3
using Statistics: median

"""
    CCFComponent(name; planet = 0, trend = 0)

One Gaussian line component of a [`CCFSet`](@ref). `planet = 0` makes it
static (centre γ); `planet = k ≥ 1` ties its centre to planet k's orbit,
γ + K·[cos(f + ω) + e cos ω], with K of either sign (km/s). `name` labels its
parameters (`ccf_gamma_<tag>_<name>`, ...).

`trend` adds a polynomial in time to the centre, static or tied:
`trend = 1` adds γ̇·(t − t_ref) (`ccf_gdot_<tag>_<name>`, km/s/day),
`trend = 2` also ½γ̈·(t − t_ref)² (`ccf_gddot_<tag>_<name>`, km/s/day²), with
t_ref the set's `trend_t_ref`. 0 (the default) adds nothing.
"""
struct CCFComponent
    name::String
    planet::Int
    trend::Int
    function CCFComponent(name::AbstractString, planet::Integer, trend::Integer = 0)
        isempty(name) && throw(ArgumentError("a CCF component needs a non-empty name"))
        occursin(r"^[A-Za-z0-9.\-]+$", name) || throw(ArgumentError(
            "CCF component name \"$name\" may hold only letters, digits, '.' and '-': " *
            "it is part of parameter names separated by '_'"))
        planet >= 0 || throw(ArgumentError(
            "CCF component \"$name\": planet = $planet; use 0 for a static component " *
            "or the 1-based index of the planet whose orbit it follows"))
        trend in (0, 1, 2) || throw(ArgumentError(
            "CCF component \"$name\": trend = $trend; use 0 (none), 1 (linear) or " *
            "2 (quadratic)"))
        return new(String(name), Int(planet), Int(trend))
    end
end
CCFComponent(name::AbstractString; planet::Integer = 0, trend::Integer = 0) =
    CCFComponent(name, planet, trend)

is_static(c::CCFComponent) = c.planet == 0

"""
    CCFSet(; t, v, ccf, components, tag = "CCF", err = nothing, labels = nothing,
           normalize_edges = 5, continuum = :per_epoch, depth_scale = :per_epoch,
           ref_epoch = 1, jitter = :per_epoch, trend_t_ref = nothing,
           shared_fwhm = false, pixel_correlation = :independent,
           resolution_kms = nothing)

The CCFs of one instrument, passed to `Data(; ccf = [...])`, and the line
components they are fitted with.

- `t` -- epoch of each CCF, BJD_TDB (days), on the time base of the rest of the
  data.
- `v`, `ccf` -- one vector per epoch: the velocity grid (km/s) and the CCF on
  it. Grids may differ between epochs. A pixel whose velocity, value or error
  is not finite (or `nothing`) is dropped.
- `err` -- optional per-pixel 1σ of each CCF, same shapes; the likelihood adds
  the jitter in quadrature. Absent: zero, so the jitter is the whole noise.
- `components` -- the [`CCFComponent`](@ref)s, at least one, names unique.
- `labels` -- one string per epoch naming its nuisance parameters; default
  `"1"`, `"2"`, ...
- `normalize_edges` -- each CCF (and its `err`) is divided by the median of its
  `normalize_edges` first and `normalize_edges` last points, so the continuum is
  ≈ 1 and depths are fractions of it. 0 leaves the values as given.
- `continuum` -- `:per_epoch` (a `ccf_cont_<tag>_<label>` per epoch) or
  `:shared` (one `ccf_cont_<tag>`).
- `depth_scale` -- `:per_epoch`: every epoch except `ref_epoch` gets a
  `ccf_scale_<tag>_<label>` multiplying all component depths, and the depths
  `ccf_depth_<tag>_<name>` are those of epoch `ref_epoch` (its scale is 1 by
  definition; a free scale there would be exactly degenerate with the depths).
  `:none`: the depths are the same at every epoch.
- `jitter` -- white noise added in quadrature to `err`: `:per_epoch`
  (`ccf_jit_<tag>_<label>`) or `:shared` (`ccf_jit_<tag>`), in the units of the
  normalised CCF.
- `trend_t_ref` -- reference time (BJD_TDB) of the components' γ trends
  (`CCFComponent(...; trend)`). Default: the mean epoch of the set, so γ_j is
  the centre at the middle of the data and is uncorrelated with γ̇_j.
- `shared_fwhm` -- `true` ties every component to one width `ccf_fwhm_<tag>`
  instead of a `ccf_fwhm_<tag>_<name>` each.
- `pixel_correlation` -- `:independent` (every pixel a datum) or `:neff`: each
  epoch's log-likelihood is multiplied by `min(1, N_eff / N_pix)`, with
  `N_eff = (velocity span of that CCF) / (resolution element)`. The resolution
  element is `resolution_kms` (km/s) when given, else the smallest lower bound
  of the set's FWHM priors. This changes posterior widths and evidences.
- `resolution_kms` -- the resolution element for `pixel_correlation = :neff`;
  refused otherwise.

The model of pixel i of epoch n is

    m_ni = c_n − s_n Σ_j d_j exp(−(v_i − v_j(t_n))² / (2 σ_j²)),
    σ_j = FWHM_j / (2√(2 ln 2)),

with v_j = γ_j(t_n) for a static component and γ_j(t_n) + K_j g_k(t_n) for one
tied to planet k, γ_j(t) = γ_j + γ̇_j (t − t_ref) + ½ γ̈_j (t − t_ref)² (the
trend terms only when the component has them). Throws an `ArgumentError` for
mismatched lengths, a non-finite epoch time, fewer than 3 finite pixels in a
CCF, non-increasing velocity grids, duplicate labels or component names, a
component with `trend` > 0 in a set with fewer than `trend + 2` epochs (or
epochs all at one time), or an unknown option.
"""
struct CCFSet
    tag::String
    t::Vector{Float64}
    labels::Vector{String}
    v::Vector{Vector{Float64}}
    y::Vector{Vector{Float64}}
    err::Vector{Vector{Float64}}      # zeros when not given
    components::Vector{CCFComponent}
    continuum::Symbol                 # :per_epoch | :shared
    depth_scale::Symbol               # :per_epoch | :none
    ref_epoch::Int                    # epoch whose depth scale is 1
    jitter::Symbol                    # :per_epoch | :shared
    norm::Vector{Float64}             # the factor each CCF was divided by (1 if none)
    normalize_edges::Int              # 0: the CCFs are as given
    trend_t_ref::Float64              # t_ref of the components' γ trends
    shared_fwhm::Bool                 # one ccf_fwhm_<tag> for every component
    pixel_correlation::Symbol         # :independent | :neff
    resolution_kms::Float64           # :neff resolution element; NaN: from the FWHM priors
end

has_trend(s::CCFSet) = any(c -> c.trend > 0, s.components)

function CCFSet(; t::AbstractVector{<:Real},
                v::AbstractVector,
                ccf::AbstractVector,
                components::AbstractVector{CCFComponent},
                tag::AbstractString = "CCF",
                err::Union{Nothing, AbstractVector} = nothing,
                labels::Union{Nothing, AbstractVector} = nothing,
                normalize_edges::Integer = 5,
                continuum::Symbol = :per_epoch,
                depth_scale::Symbol = :per_epoch,
                ref_epoch::Integer = 1,
                jitter::Symbol = :per_epoch,
                trend_t_ref::Union{Nothing, Real} = nothing,
                shared_fwhm::Bool = false,
                pixel_correlation::Symbol = :independent,
                resolution_kms::Union{Nothing, Real} = nothing)
    n = length(t)
    n > 0 || throw(ArgumentError("CCFSet \"$tag\": no epochs"))
    occursin(r"^[A-Za-z0-9.\-]+$", tag) || throw(ArgumentError(
        "CCFSet tag \"$tag\" may hold only letters, digits, '.' and '-': it is part " *
        "of parameter names separated by '_'"))
    all(isfinite, t) || throw(ArgumentError("CCFSet \"$tag\": non-finite epoch time"))
    length(v) == n || throw(ArgumentError(
        "CCFSet \"$tag\": $(length(v)) velocity grids for $n epochs"))
    length(ccf) == n || throw(ArgumentError(
        "CCFSet \"$tag\": $(length(ccf)) CCFs for $n epochs"))
    err === nothing || length(err) == n || throw(ArgumentError(
        "CCFSet \"$tag\": $(length(err)) error vectors for $n epochs"))
    isempty(components) && throw(ArgumentError(
        "CCFSet \"$tag\": at least one line component is needed"))
    cnames = [c.name for c in components]
    length(unique(cnames)) == length(cnames) || throw(ArgumentError(
        "CCFSet \"$tag\": component names must be unique; got $(cnames)"))
    continuum in (:per_epoch, :shared) || throw(ArgumentError(
        "CCFSet \"$tag\": continuum must be :per_epoch or :shared; got :$continuum"))
    depth_scale in (:per_epoch, :none) || throw(ArgumentError(
        "CCFSet \"$tag\": depth_scale must be :per_epoch or :none; got :$depth_scale"))
    jitter in (:per_epoch, :shared) || throw(ArgumentError(
        "CCFSet \"$tag\": jitter must be :per_epoch or :shared; got :$jitter"))
    1 <= ref_epoch <= n || throw(ArgumentError(
        "CCFSet \"$tag\": ref_epoch = $ref_epoch is not an epoch (1..$n)"))
    normalize_edges >= 0 || throw(ArgumentError(
        "CCFSet \"$tag\": normalize_edges must be ≥ 0; got $normalize_edges"))
    for c in components
        c.trend == 0 && continue
        n >= c.trend + 2 || throw(ArgumentError(
            "CCFSet \"$tag\": component \"$(c.name)\" has trend = $(c.trend), which " *
            "needs at least $(c.trend + 2) epochs; the set has $n"))
        maximum(t) > minimum(t) || throw(ArgumentError(
            "CCFSet \"$tag\": component \"$(c.name)\" has trend = $(c.trend), but " *
            "every epoch is at the same time"))
    end
    t_ref = trend_t_ref === nothing ? sum(t) / n : Float64(trend_t_ref)
    isfinite(t_ref) || throw(ArgumentError(
        "CCFSet \"$tag\": trend_t_ref must be finite; got $trend_t_ref"))
    pixel_correlation in (:independent, :neff) || throw(ArgumentError(
        "CCFSet \"$tag\": pixel_correlation must be :independent or :neff; got " *
        ":$pixel_correlation"))
    if resolution_kms !== nothing
        pixel_correlation === :neff || throw(ArgumentError(
            "CCFSet \"$tag\": resolution_kms is the resolution element of " *
            "pixel_correlation = :neff; it has no use under :$pixel_correlation"))
        (isfinite(resolution_kms) && resolution_kms > 0) || throw(ArgumentError(
            "CCFSet \"$tag\": resolution_kms must be finite and > 0; got $resolution_kms"))
    end
    res = resolution_kms === nothing ? NaN : Float64(resolution_kms)
    labs = labels === nothing ? [string(i) for i in 1:n] : [string(l) for l in labels]
    length(labs) == n || throw(ArgumentError(
        "CCFSet \"$tag\": $(length(labs)) labels for $n epochs"))
    length(unique(labs)) == n || throw(ArgumentError(
        "CCFSet \"$tag\": epoch labels must be unique; they name per-epoch parameters"))
    for l in labs
        occursin(r"^[A-Za-z0-9.\-]+$", l) || throw(ArgumentError(
            "CCFSet \"$tag\": epoch label \"$l\" may hold only letters, digits, '.' " *
            "and '-'"))
    end

    vs = Vector{Vector{Float64}}(undef, n)
    ys = Vector{Vector{Float64}}(undef, n)
    es = Vector{Vector{Float64}}(undef, n)
    norm = ones(n)
    for i in 1:n
        vi = _ccf_float_vec(v[i]); yi = _ccf_float_vec(ccf[i])
        m0 = length(vi)
        length(yi) == m0 || throw(ArgumentError(
            "CCFSet \"$tag\", epoch $(labs[i]): $(length(yi)) CCF values on a grid of $m0"))
        ei = err === nothing ? zeros(m0) : _ccf_float_vec(err[i])
        length(ei) == m0 || throw(ArgumentError(
            "CCFSet \"$tag\", epoch $(labs[i]): $(length(ei)) errors for $m0 points"))
        # A pixel with a non-finite velocity, value or error (a NaN a pipeline
        # wrote for a masked order, say) is dropped, not refused: the rest of
        # the CCF is still data.
        keep = [isfinite(vi[k]) && isfinite(yi[k]) && isfinite(ei[k]) for k in 1:m0]
        vi = vi[keep]; yi = yi[keep]; ei = ei[keep]
        m = length(vi)
        m >= 3 || throw(ArgumentError(
            "CCFSet \"$tag\", epoch $(labs[i]): a CCF needs at least 3 finite points; " *
            "it has $m"))
        all(>(0), diff(vi)) || throw(ArgumentError(
            "CCFSet \"$tag\", epoch $(labs[i]): the velocity grid must be strictly " *
            "increasing"))
        all(>=(0), ei) || throw(ArgumentError(
            "CCFSet \"$tag\", epoch $(labs[i]): errors must be ≥ 0"))
        if normalize_edges > 0
            2normalize_edges <= m || throw(ArgumentError(
                "CCFSet \"$tag\", epoch $(labs[i]): normalize_edges = " *
                "$normalize_edges needs at least $(2normalize_edges) points"))
            k = normalize_edges
            c = median(vcat(yi[1:k], yi[end-k+1:end]))
            (isfinite(c) && c > 0) || throw(ArgumentError(
                "CCFSet \"$tag\", epoch $(labs[i]): the edge median is $c; cannot " *
                "normalise by it (set normalize_edges = 0 for CCFs not positive)"))
            yi = yi ./ c; ei = ei ./ c; norm[i] = c
        end
        vs[i] = vi; ys[i] = yi; es[i] = ei
    end
    return CCFSet(String(tag), Vector{Float64}(t), labs, vs, ys, es,
                  Vector{CCFComponent}(components), continuum, depth_scale,
                  Int(ref_epoch), jitter, norm, Int(normalize_edges), t_ref,
                  shared_fwhm, pixel_correlation, res)
end

# Float64 vector of a JSON array that may hold `nothing` for a masked pixel.
_ccf_float_vec(x) = Float64[y === nothing ? NaN : Float64(y) for y in x]

n_ccf_epochs(s::CCFSet) = length(s.t)
n_ccf_points(s::CCFSet) = sum(length, s.y)

"""
    CCFIndices

Layout slots of one `CCFSet` (`SystemicIndices.ccf`). Per component:
`gamma`, `gdot` and `gddot` (0 without that trend term), `K` (0 for a static
component), `fwhm` (one slot repeated under `shared_fwhm`), `depth`. Per
epoch: `cont`, `scale` (0 for the reference epoch and under `depth_scale =
:none`), `jit`; shared parameters repeat one slot. Also per epoch, `weight`:
the factor its log-likelihood is multiplied by (1 under `pixel_correlation =
:independent`, `min(1, N_eff/N_pix)` under `:neff`), and `resolution`, the
resolution element (km/s) `:neff` used (NaN under `:independent`).
"""
struct CCFIndices
    gamma::Vector{Int}
    gdot::Vector{Int}
    gddot::Vector{Int}
    K::Vector{Int}
    fwhm::Vector{Int}
    depth::Vector{Int}
    cont::Vector{Int}
    scale::Vector{Int}
    jit::Vector{Int}
    weight::Vector{Float64}
    resolution::Float64
end

"""
    ccf_param_names(s::CCFSet) -> Vector{String}

The parameter names `s` adds to the layout, in layout order.
"""
function ccf_param_names(s::CCFSet)
    out = String[]
    _push_ccf_slots!(out, s)
    return out
end

# Allocate the slots of `s` at the end of `names` (pushing them) and return its
# `CCFIndices`. `priors` (name => PriorSpec, the final priors of the model)
# give the `:neff` resolution element when `resolution_kms` is not set; without
# them the weights are left at 1 (`ccf_param_names` only needs the names).
function _push_ccf_slots!(names::Vector{String}, s::CCFSet, priors = nothing)
    nc = length(s.components); ne = n_ccf_epochs(s); tg = s.tag
    gamma = zeros(Int, nc); gdot = zeros(Int, nc); gddot = zeros(Int, nc)
    K = zeros(Int, nc); fwhm = zeros(Int, nc); depth = zeros(Int, nc)
    for (j, c) in enumerate(s.components)
        push!(names, "ccf_gamma_$(tg)_$(c.name)"); gamma[j] = length(names)
        if c.trend >= 1
            push!(names, "ccf_gdot_$(tg)_$(c.name)"); gdot[j] = length(names)
        end
        if c.trend >= 2
            push!(names, "ccf_gddot_$(tg)_$(c.name)"); gddot[j] = length(names)
        end
        if !is_static(c)
            push!(names, "ccf_K_$(tg)_$(c.name)"); K[j] = length(names)
        end
        if !s.shared_fwhm
            push!(names, "ccf_fwhm_$(tg)_$(c.name)"); fwhm[j] = length(names)
        end
        push!(names, "ccf_depth_$(tg)_$(c.name)"); depth[j] = length(names)
    end
    if s.shared_fwhm
        push!(names, "ccf_fwhm_$(tg)"); fill!(fwhm, length(names))
    end
    cont = zeros(Int, ne); scale = zeros(Int, ne); jit = zeros(Int, ne)
    if s.continuum === :shared
        push!(names, "ccf_cont_$(tg)"); fill!(cont, length(names))
    else
        for (i, l) in enumerate(s.labels)
            push!(names, "ccf_cont_$(tg)_$(l)"); cont[i] = length(names)
        end
    end
    if s.depth_scale === :per_epoch
        for (i, l) in enumerate(s.labels)
            i == s.ref_epoch && continue
            push!(names, "ccf_scale_$(tg)_$(l)"); scale[i] = length(names)
        end
    end
    if s.jitter === :shared
        push!(names, "ccf_jit_$(tg)"); fill!(jit, length(names))
    else
        for (i, l) in enumerate(s.labels)
            push!(names, "ccf_jit_$(tg)_$(l)"); jit[i] = length(names)
        end
    end
    res = NaN; weight = ones(ne)
    if s.pixel_correlation === :neff && priors !== nothing
        res = _ccf_resolution(s, priors, unique(names[fwhm]))
        weight = ccf_epoch_weights(s, res)
    end
    return CCFIndices(gamma, gdot, gddot, K, fwhm, depth, cont, scale, jit, weight, res)
end

# The `:neff` resolution element of `s`: `resolution_kms` when given, else the
# smallest lower bound of the priors of its FWHM parameters `fwhm_names`.
function _ccf_resolution(s::CCFSet, priors, fwhm_names)
    isnan(s.resolution_kms) || return s.resolution_kms
    lo = minimum(n -> priors[n].lo, fwhm_names)
    (isfinite(lo) && lo > 0) || throw(ArgumentError(
        "CCFSet \"$(s.tag)\": pixel_correlation = :neff takes its resolution element " *
        "from the narrowest FWHM prior's lower bound, which is $lo; give " *
        "`resolution_kms`, or FWHM priors bounded below by a positive width"))
    return lo
end

"""
    ccf_epoch_weights(s::CCFSet, resolution) -> Vector{Float64}

Per epoch of `s`, the factor `pixel_correlation = :neff` multiplies its
log-likelihood by: `min(1, N_eff / N_pix)`, with `N_eff` the velocity span of
that CCF's (finite) pixels divided by `resolution` (km/s) and `N_pix` their
number.
"""
function ccf_epoch_weights(s::CCFSet, resolution::Real)
    (isfinite(resolution) && resolution > 0) || throw(ArgumentError(
        "resolution must be finite and > 0; got $resolution"))
    return [min(1.0, ((v[end] - v[1]) / resolution) / length(v)) for v in s.v]
end

"""
    load_ccf_json(path) -> (id, v, ccf)

Read one CCF dump of the form the ExoAutomata database writes: a JSON object
with `velocities` and `ccf_values` (arrays, or JSON strings holding arrays) and
an optional `id`. `id` is `nothing` when absent.
"""
function load_ccf_json(path::AbstractString)
    d = JSON3.read(read(path, String))
    _arr(x) = _ccf_float_vec(x isa AbstractString ? JSON3.read(x) : x)
    haskey(d, :velocities) || throw(ArgumentError("$path: no `velocities` field"))
    haskey(d, :ccf_values) || throw(ArgumentError("$path: no `ccf_values` field"))
    id = haskey(d, :id) ? d[:id] : nothing
    return id, _arr(d[:velocities]), _arr(d[:ccf_values])
end
