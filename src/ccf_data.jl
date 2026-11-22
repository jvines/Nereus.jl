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
    CCFComponent(name; planet = 0)

One Gaussian line component of a [`CCFSet`](@ref). `planet = 0` makes it
static (centre γ); `planet = k ≥ 1` ties its centre to planet k's orbit,
γ + K·[cos(f + ω) + e cos ω], with K of either sign (km/s). `name` labels its
parameters (`ccf_gamma_<tag>_<name>`, ...).
"""
struct CCFComponent
    name::String
    planet::Int
    function CCFComponent(name::AbstractString, planet::Integer)
        isempty(name) && throw(ArgumentError("a CCF component needs a non-empty name"))
        occursin(r"^[A-Za-z0-9.\-]+$", name) || throw(ArgumentError(
            "CCF component name \"$name\" may hold only letters, digits, '.' and '-': " *
            "it is part of parameter names separated by '_'"))
        planet >= 0 || throw(ArgumentError(
            "CCF component \"$name\": planet = $planet; use 0 for a static component " *
            "or the 1-based index of the planet whose orbit it follows"))
        return new(String(name), Int(planet))
    end
end
CCFComponent(name::AbstractString; planet::Integer = 0) = CCFComponent(name, planet)

is_static(c::CCFComponent) = c.planet == 0

"""
    CCFSet(; t, v, ccf, components, tag = "CCF", err = nothing, labels = nothing,
           normalize_edges = 5, continuum = :per_epoch, depth_scale = :per_epoch,
           ref_epoch = 1, jitter = :per_epoch)

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

The model of pixel i of epoch n is

    m_ni = c_n − s_n Σ_j d_j exp(−(v_i − v_j(t_n))² / (2 σ_j²)),
    σ_j = FWHM_j / (2√(2 ln 2)),

with v_j = γ_j for a static component and γ_j + K_j g_k(t_n) for one tied to
planet k. Throws an `ArgumentError` for mismatched lengths, a non-finite epoch time,
fewer than 3 finite pixels in a CCF, non-increasing velocity grids, duplicate labels or component names, or an
unknown option.
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
end

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
                jitter::Symbol = :per_epoch)
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
                  Int(ref_epoch), jitter, norm, Int(normalize_edges))
end

# Float64 vector of a JSON array that may hold `nothing` for a masked pixel.
_ccf_float_vec(x) = Float64[y === nothing ? NaN : Float64(y) for y in x]

n_ccf_epochs(s::CCFSet) = length(s.t)
n_ccf_points(s::CCFSet) = sum(length, s.y)

"""
    CCFIndices

Layout slots of one `CCFSet` (`SystemicIndices.ccf`). Per component:
`gamma`, `K` (0 for a static component), `fwhm`, `depth`. Per epoch: `cont`,
`scale` (0 for the reference epoch and under `depth_scale = :none`), `jit`;
shared parameters repeat one slot.
"""
struct CCFIndices
    gamma::Vector{Int}
    K::Vector{Int}
    fwhm::Vector{Int}
    depth::Vector{Int}
    cont::Vector{Int}
    scale::Vector{Int}
    jit::Vector{Int}
end

"""
    ccf_param_names(s::CCFSet) -> Vector{String}

The parameter names `s` adds to the layout, in layout order.
"""
function ccf_param_names(s::CCFSet)
    out = String[]
    tg = s.tag
    for c in s.components
        push!(out, "ccf_gamma_$(tg)_$(c.name)")
        is_static(c) || push!(out, "ccf_K_$(tg)_$(c.name)")
        push!(out, "ccf_fwhm_$(tg)_$(c.name)")
        push!(out, "ccf_depth_$(tg)_$(c.name)")
    end
    if s.continuum === :shared
        push!(out, "ccf_cont_$(tg)")
    else
        append!(out, ("ccf_cont_$(tg)_$(l)" for l in s.labels))
    end
    if s.depth_scale === :per_epoch
        append!(out, ("ccf_scale_$(tg)_$(l)" for (i, l) in enumerate(s.labels)
                      if i != s.ref_epoch))
    end
    if s.jitter === :shared
        push!(out, "ccf_jit_$(tg)")
    else
        append!(out, ("ccf_jit_$(tg)_$(l)" for l in s.labels))
    end
    return out
end

# Allocate the slots of `s` at the end of `names` (pushing them) and return its
# `CCFIndices`. Order matches `ccf_param_names`.
function _push_ccf_slots!(names::Vector{String}, s::CCFSet)
    nc = length(s.components); ne = n_ccf_epochs(s); tg = s.tag
    gamma = zeros(Int, nc); K = zeros(Int, nc); fwhm = zeros(Int, nc); depth = zeros(Int, nc)
    for (j, c) in enumerate(s.components)
        push!(names, "ccf_gamma_$(tg)_$(c.name)"); gamma[j] = length(names)
        if !is_static(c)
            push!(names, "ccf_K_$(tg)_$(c.name)"); K[j] = length(names)
        end
        push!(names, "ccf_fwhm_$(tg)_$(c.name)"); fwhm[j] = length(names)
        push!(names, "ccf_depth_$(tg)_$(c.name)"); depth[j] = length(names)
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
    return CCFIndices(gamma, K, fwhm, depth, cont, scale, jit)
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
