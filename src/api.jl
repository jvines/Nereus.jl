# Nereus public API — one entry point per technique.
#
# This replaces `run_job(cfg)` as the thing users and clients call. `run_job`
# was a single polymorphic entry whose behaviour was entirely determined by an
# opaque config: you could not tell from a call site what would run, and every
# option for every technique shared one namespace.
#
# Here each technique is its own function with only its own parameters, and
# joint fitting is its own operation composing explicit channels. This mirrors
# `nereus-py` 1:1 — the Python client is a thin transport over these names, so
# the two surfaces cannot drift.
#
# Why not one function per data combination: `PlanetDataSources` is a
# composable set of source flags (RV_SOURCE, PM_SOURCE, AS_SOURCE, RM_SOURCE,
# RM_R_SOURCE, RM_A_SOURCE, GD_SOURCE, TTV_SOURCE, TTV_NB_SOURCE, SB_SOURCE),
# not an enum — 10 flags, 1023 combinations, 23 of which have convenience
# names. Enumerating them in the API would recreate exactly what the model
# layer deliberately stopped doing.

using Statistics: median, std, quantile
using Random
using JSON3
import MCMCChains

# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
# One table, so adding a sampler means adding one line here and one dataclass
# in engines.py. Anything not in this table is rejected by name, with the list.

const ENGINES = Dict{String, Function}(
    "pt"                => sample_pt,
    "pt_warm"           => sample_pt_warm,
    "pt_hmc"            => sample_pt_hmc,
    "pt_whitening"      => sample_pt_whitening,
    "ptemcee"           => sample_ptemcee,
    "transdim_ptemcee"  => sample_transdim_ptemcee,
    "nested"            => sample_nested,
    "nested_ins"        => sample_nested_ins,
    "nested_dynamic"    => sample_nested_dynamic,
    "moms"              => sample_moms,
    "moms_ns"           => sample_moms_ns,
    "rjmcmc"            => sample_rjmcmc,
    "nuts"              => sample_nuts,
    "map"               => sample_map,
    "smc"               => sample_smc,
    "ensemble"          => sample_ensemble,
    "ess"               => sample_ess,
    "pa"                => sample_pa,
)

"""
    run_engine(target, spec) -> (chains, log_evidence)

Dispatch to a sampler by name, passing only the options it declares. Options
the sampler does not accept are an ERROR, not a silent no-op: a typo'd
`n_round` that quietly does nothing is how you lose a day.
"""
function run_engine(target, spec::AbstractDict)
    name = String(get(spec, "engine", "pt"))
    fn = get(ENGINES, name, nothing)
    fn === nothing && throw(ArgumentError(
        "unknown engine $(repr(name)). Available: " *
        join(sort!(collect(keys(ENGINES))), ", ")))

    # JSON has no Symbol. These options are enum-like on the Julia side
    # (they mirror the Literal[...] fields in engines.py), so a String from
    # the wire must become a Symbol or the call is a TypeError.
    opts = Dict{Symbol,Any}()
    for (k, v) in pairs(get(spec, "options", Dict()))
        sk = Symbol(k)
        opts[sk] = (sk in SYMBOL_OPTIONS && v isa AbstractString) ? Symbol(v) : v
    end
    accepted = _kwarg_names(fn)
    unknown = setdiff(keys(opts), accepted)
    isempty(unknown) || throw(ArgumentError(
        "engine $(name) does not accept $(join(sort!(collect(unknown)), ", ")). " *
        "It accepts: $(join(sort!(collect(accepted)), ", "))"))

    # Positional arity differs across samplers: 6 take (target), 12 take
    # (target, data). Detect it rather than hardcoding a list, so a new
    # sampler works without editing this function.
    return _n_positional(fn) >= 2 ? fn(target, target.data; opts...) :
                                    fn(target; opts...)
end

_n_positional(fn) = length(first(methods(fn)).sig.parameters) - 1

_kwarg_names(fn) = Set(Base.kwarg_decl(first(methods(fn))))

"Engine options typed ::Symbol in Julia; JSON delivers them as strings."
const SYMBOL_OPTIONS = Set{Symbol}([
    :within_model, :explorer, :backend, :init_strategy, :bounds, :proposal,
    :method, :ad_backend, :mutation_kernel, :parametrization, :time_anchor,
    :stability,
])

# ---------------------------------------------------------------------------
# Stopping — uniform across engines, separate from sampler-specific knobs
# ---------------------------------------------------------------------------

Base.@kwdef struct Stopping
    max_seconds::Union{Nothing,Float64} = nothing
    max_evals::Union{Nothing,Int}       = nothing
    ess_min::Union{Nothing,Float64}     = nothing
    rhat_max::Union{Nothing,Float64}    = nothing
    logz_tol::Union{Nothing,Float64}    = nothing
    check_every::Int                    = 500
end

# ---------------------------------------------------------------------------
# Channels -> build_target keywords
# ---------------------------------------------------------------------------

"""
    _target_from(channels, planets; kwargs...) -> NereusTarget

Assemble a target from explicit channel specs. `build_target` already infers
the source-flag set from which keywords are non-empty, so this is a translation
layer, not a second model.
"""
function _target_from(channels::AbstractVector, planets; M_pri = nothing,
                      plx = nothing, trend_order = 0, noise_models = nothing,
                      priors = nothing, stability = :none)
    rv    = NamedTuple()
    phot  = NamedTuple()
    iad = hgca = gost = relast = nothing
    for ch in channels
        src = String(get(ch, "source", ""))
        if src == "RV"
            rv = _rv_namedtuple(ch)
            trend_order = get(ch, "trend_order", trend_order)
        elseif src == "PM"
            phot = _phot_namedtuple(ch)
        elseif src == "AS"
            iad    = get(ch, "iad", nothing)
            hgca   = get(ch, "hgca", nothing)
            gost   = get(ch, "gost", nothing)
            relast = get(ch, "relast", nothing)
            plx    = something(get(ch, "parallax", nothing), plx)
            M_pri  = something(get(ch, "m_pri", nothing), M_pri)
        elseif src in ("RM", "RM_R", "RM_A", "TOMO", "TTV", "TTV_NB", "SB")
            # handled by the technique-specific entry points below
        else
            throw(ArgumentError("unknown channel source $(repr(src))"))
        end
    end
    # Only pass stellar mass when we actually have one: build_target types
    # M_s::Real, and an RV-only fit legitimately has no stellar mass (K-driven
    # parametrisation needs none). Passing nothing is a TypeError, not a default.
    kw = Dict{Symbol,Any}(:planets => _planet_spec(planets), :rv => rv,
                          :phot => phot, :iad => iad, :hgca => hgca,
                          :gost => gost, :relAST => relast, :plx => plx,
                          :trend_order => trend_order,
                          :noise_models => noise_models, :priors => priors,
                          :stability => stability)
    if M_pri !== nothing
        kw[:M_pri] = M_pri
        kw[:M_s]   = M_pri
    end
    return build_target(; kw...)
end

# ---------------------------------------------------------------------------
# Entry points — one per technique
# ---------------------------------------------------------------------------

"""
    fit_rv(rv; planets, engine, stopping, output_dir, ...)

Fit radial velocities alone. `planets` is an integer for a fixed-dimension fit
or a range for trans-dimensional model selection over planet count.
"""
function fit_rv(rv; planets = 1, engine = Dict("engine" => "pt"),
                stopping = nothing, output_dir = nothing, noise = nothing,
                trend_order = 0, kwargs...)
    ch = _as_channel(rv, "RV")
    if !(planets isa NamedTuple)
        d = collect(values(get(ch, "data", Dict())))
        isempty(d) && throw(ArgumentError("fit_rv: no RV data given"))
        allt  = reduce(vcat, [collect(x.t)  for x in d])
        allrv = reduce(vcat, [collect(x.rv) for x in d])
        planets = _planet_spec(planets; block = default_rv_planet(allt, allrv))
    end
    tgt = _target_from([ch], planets;
                       trend_order, noise_models = noise, kwargs...)
    return _finish(tgt, engine, stopping, output_dir; op = "fit_rv")
end

"""
    fit_transit(phot; planets, limb_darkening, rho_star, gravity_darkening, ...)

Fit transit photometry alone. `gravity_darkening=true` selects the oblate
von Zeipel/Barnes model (GD_SOURCE), which constrains stellar inclination i*
separately from λ — unreachable from a symmetric transit.
"""
function fit_transit(phot; planets = 1, limb_darkening = :quadratic,
                     rho_star = nothing, gravity_darkening = false,
                     engine = Dict("engine" => "pt"), stopping = nothing,
                     output_dir = nothing, kwargs...)
    tgt = _target_from([_as_channel(phot, "PM")], planets; kwargs...)
    return _finish(tgt, engine, stopping, output_dir; op = "fit_transit")
end

"""
    resolve_astrometry(spec) -> IADData | nothing

Turn a JSON-able astrometry REFERENCE into the real object, server-side.

`IADData` cannot cross a socket: it is a struct of parallel arrays plus scan
geometry, and shipping it as raw JSON is both enormous and a second copy of the
reader's conventions waiting to drift. So the client sends a reference — which
catalogue, which source — and the server loads it with the same code path a
Julia user would call.

    Dict("catalogue" => "gaia_dr4", "source_id" => 1457486023639239296)
    Dict("catalogue" => "gaia_dr4", "source_id" => ..., "path" => "/local.xml")
    Dict("catalogue" => "hipparcos", "hip" => 27321, ...)
"""
function resolve_astrometry(spec)
    spec === nothing && return nothing
    spec isa AbstractDict || return spec          # already an IADData
    cat = lowercase(String(get(spec, "catalogue", "")))
    if cat in ("gaia_dr4", "gaia_epoch", "dr4")
        sid = get(spec, "source_id", nothing)
        sid === nothing && error("gaia_dr4 astrometry needs a source_id")
        path = get(spec, "path", get(ENV, "NEREUS_GAIA_DR4_XML", ""))
        xml = (path == "" || !isfile(String(path))) ?
              fetch_gaia_dr4_prerelease() : String(path)
        return read_gaia_epoch_votable(xml, Int(sid)).iad
    elseif cat in ("hipparcos", "hip", "iad")
        hip = get(spec, "hip", nothing)
        hip === nothing && error("hipparcos astrometry needs a hip number")
        return fetch_hip_iad(Int(hip);
                             catalogue_pos = get(spec, "catalogue_pos", nothing),
                             catalogue_pm  = get(spec, "catalogue_pm", nothing),
                             parallax_mas  = get(spec, "parallax_mas", nothing))
    end
    error("unknown astrometry catalogue $(repr(cat)). " *
          "Supported: gaia_dr4, hipparcos")
end

"""
    fit_astrometry(; iad, hgca, gost, relast, parallax, m_pri, planets, ...)

Fit astrometry alone, no RV. `parallax` should be an informative prior: the
abscissae constrain a0 ∝ M_sec·ϖ, degenerate without it.
"""
function fit_astrometry(; iad = nothing, hgca = nothing, gost = nothing,
                        relast = nothing, parallax = nothing, m_pri = nothing,
                        planets = 1, engine = Dict("engine" => "pt"),
                        stopping = nothing, output_dir = nothing, kwargs...)
    iad = resolve_astrometry(iad)
    parallax = _as_prior(parallax)
    ch = Dict("source" => "AS", "iad" => iad, "hgca" => hgca, "gost" => gost,
              "relast" => relast, "parallax" => parallax, "m_pri" => m_pri)
    planets isa NamedTuple ||
        (planets = _planet_spec(planets; block = default_astrom_planet()))
    tgt = _target_from([ch], planets; kwargs...)
    return _finish(tgt, engine, stopping, output_dir; op = "fit_astrometry")
end





"""
    fit_joint(channels...; planets, engine, stopping, ...)

Fit several techniques simultaneously. One operation for "these together",
rather than a named function per combination.
"""
function fit_joint(channels...; planets = 1, engine = Dict("engine" => "pt"),
                   stopping = nothing, output_dir = nothing, kwargs...)
    length(channels) >= 2 || throw(ArgumentError(
        "fit_joint needs at least two channels; for one technique use its " *
        "dedicated entry point, which has a clearer signature"))
    tgt = _target_from(collect(channels), planets; kwargs...)
    return _finish(tgt, engine, stopping, output_dir; op = "fit_joint")
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_as_channel(x::AbstractDict, src) = haskey(x, "source") ? x :
    Dict("source" => src, "data" => x)
_as_channel(x, src) = Dict("source" => src, "data" => x)

"""Per-instrument RV -> the NamedTuple `build_target` expects."""
function _rv_namedtuple(ch::AbstractDict)
    data = get(ch, "data", Dict())
    jit  = get(ch, "jitter", "default")
    names = Tuple(Symbol.(collect(keys(data))))
    vals  = Tuple(begin
        d = data[String(n)]
        jit == "default" ? (data = d,) : (data = d, sigma = jit)
    end for n in names)
    return NamedTuple{names}(vals)
end

"""Per-instrument photometry -> NamedTuple."""
function _phot_namedtuple(ch::AbstractDict)
    data = get(ch, "data", Dict())
    names = Tuple(Symbol.(collect(keys(data))))
    return NamedTuple{names}(Tuple((data = data[String(n)],) for n in names))
end

"""
`planets` is an Int (fixed dimension) or a range (trans-dimensional).

An Int builds that many identical default planet blocks; the caller overrides
per-planet priors via `priors`. A range is passed through so the trans-dim
layer sets max_kplanet and the occupancy prior.
"""
_planet_spec(nt::NamedTuple) = nt
_planet_spec(n::Integer; block = NamedTuple()) =
    NamedTuple{Tuple(Symbol("k$i") for i in 1:n)}(Tuple(block for _ in 1:n))
_planet_spec(r::AbstractVector; block = NamedTuple()) =
    _planet_spec(maximum(r); block)

"""
    default_rv_planet(t, rv) -> NamedTuple

Default P+K block for an RV planet, with bounds taken from the data rather
than hard-coded: period from twice the median cadence to three baselines,
semi-amplitude from 0.1 m/s to ten times the RV scatter. A user who wants
something else passes `priors`; a user who just wants a fit should not have to
invent numbers to get one.
"""
function default_rv_planet(t::AbstractVector, rv::AbstractVector)
    base   = maximum(t) - minimum(t)
    cad    = length(t) > 1 ? median(diff(sort(collect(t)))) : 1.0
    P_lo   = max(2 * cad, 0.5)
    P_hi   = max(3 * base, 10 * P_lo)
    K_hi   = max(10 * std(rv), 10.0)
    return (P      = LogUniformPrior(P_lo, P_hi),
            K      = LogUniformPrior(0.1, K_hi),
            sesinw = UniformPrior(-1.0, 1.0),
            secosw = UniformPrior(-1.0, 1.0),
            Mo     = UniformPrior(0.0, 2pi))
end

"""Default a+M_sec block for an astrometric companion."""
default_astrom_planet(; a_lo = 0.1, a_hi = 100.0, m_lo = 1e-4, m_hi = 1.0) =
    (a      = LogUniformPrior(a_lo, a_hi),
     M_sec  = LogUniformPrior(m_lo, m_hi),
     sesinw = UniformPrior(-1.0, 1.0),
     secosw = UniformPrior(-1.0, 1.0),
     Mo     = UniformPrior(0.0, 2pi),
     inc    = SinePrior(),
     Omega  = UniformPrior(0.0, 2pi))

"""
    _normalize(raw) -> (chains, log_z, extra)

Samplers return different shapes: `sample_pt`/`sample_nested` a
`(chains, log_Z)` tuple, `sample_ptemcee` a `PTemceeResult`, `sample_map` a
`MAPResult` with no chains at all. Unifying that here is the whole point of a
facade — callers should not branch on which engine they picked.
"""
function _normalize(raw)
    extra = Dict{String,Any}()
    if raw isa Tuple && length(raw) >= 2
        return raw[1], raw[2], extra
    end
    # MAP: a point estimate, not a posterior. Surface its honesty flags —
    # a railed MAP is a failed fit, not a fit.
    if hasproperty(raw, :x_map)
        extra["map"] = Dict("converged" => raw.converged,
                            "railed" => raw.railed,
                            "railed_params" => raw.railed_params,
                            "n_basins" => raw.n_basins,
                            "log_posterior" => raw.log_posterior)
        return nothing, getproperty(raw, :log_evidence_laplace), extra
    end
    ch = hasproperty(raw, :chains) ? raw.chains : nothing
    lz = hasproperty(raw, :log_evidence) ? raw.log_evidence : NaN
    for f in (:acceptance_within, :acceptance_swap, :betas, :n_evals)
        hasproperty(raw, f) && (extra[String(f)] = getproperty(raw, f))
    end
    return ch, lz, extra
end


"""
    _summarise_params(chains, params) -> Dict

Per-parameter posterior summary: median plus asymmetric 1/2/3-sigma bounds.
Matches what runner.jl writes into summary["params"] so the two agree and a
client cannot tell which produced a given result.
"""
function _summarise_params(chains, params)
    out = Dict{String,Any}()
    chains === nothing && return out
    names_in = Set(MCMCChains.names(chains, :parameters))
    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        sym in names_in || continue
        v = vec(Array(chains[sym]))
        out[name] = Dict{String,Float64}(
            "median" => median(v),
            "lo16"   => quantile(v, 0.16),   "hi84"   => quantile(v, 0.84),
            "lo2sig" => quantile(v, 0.025),  "hi2sig" => quantile(v, 0.975),
            "lo3sig" => quantile(v, 0.0015), "hi3sig" => quantile(v, 0.9985))
    end
    return out
end

"""
    _as_prior(x) -> PriorSpec

Priors cross the wire as `{"type": "normal", "mu":…, "sigma":…}` because a
PriorSpec is a Julia type. Accept both that and an already-constructed prior.
"""
_as_prior(x::PriorSpec) = x
_as_prior(::Nothing) = nothing
_as_prior(x::Real) = FixedPrior(float(x))
function _as_prior(d::AbstractDict)
    t = lowercase(String(get(d, "type", "")))
    g(k, dflt=nothing) = get(d, k, dflt)
    t == "normal"      && return NormalPrior(float(g("mu")), float(g("sigma")))
    t == "uniform"     && return UniformPrior(float(g("lo")), float(g("hi")))
    t == "loguniform"  && return LogUniformPrior(float(g("lo")), float(g("hi")))
    t == "fixed"       && return FixedPrior(float(g("value")))
    t == "sine"        && return SinePrior()
    error("unknown prior type $(repr(t)); use normal|uniform|loguniform|fixed|sine")
end

"Stellar mass from the target config, or nothing if the fit never needed one."
function _M_s_of(target)
    ms = try target.params.config.M_s catch; nothing end
    (ms === nothing || (ms isa Real && isnan(ms))) ? nothing : ms
end

"""Run the engine, then assemble the standard result."""
function _finish(target, engine, stopping, output_dir; op::String)
    t0 = time()
    raw = run_engine(target, engine isa AbstractDict ? engine :
                     Dict("engine" => String(engine)))
    chains, log_z, extra = _normalize(raw)
    summary = Dict{String,Any}(
        "op" => op,
        "status" => "ok",
        "log_z" => log_z,
        "elapsed_sec" => time() - t0,
    )
    merge!(summary, extra)
    summary["params"] = _summarise_params(chains, target.params)
    try
        # compute_derived(chains, params; M_s, R_s, ...) — stellar quantities
        # are keywords, not positional. Only pass what we actually know; the
        # derived block then contains whatever is computable.
        # summarize_derived(chains, params; M_s, ...) does the compute itself.
        summary["derived"] = chains === nothing ? Dict() :
            summarize_derived(chains, target.params; M_s = _M_s_of(target))
    catch err
        summary["derived_error"] = sprint(showerror, err)
    end
    if get(get(summary, "map", Dict()), "railed", false)
        summary["status"] = "railed"
        summary["error"] = "MAP railed against a prior bound on: " *
            join(get(summary["map"], "railed_params", String[]), ", ") *
            " — treat this as a failed fit, not a result"
    end
    if output_dir !== nothing && chains !== nothing
        mkpath(output_dir)
        save_chains(joinpath(output_dir, "chains.nc"), chains, target.params;
                    data = target.data, log_evidence = log_z)
        summary["output_dir"] = output_dir
    end
    return (; chains, log_z, summary)
end

# ---------------------------------------------------------------------------
# Config-route entry points
# ---------------------------------------------------------------------------
# RM, TTV and SB2 are NOT expressible through `build_target`: it plumbs no rm /
# transit_times / secondary keywords, and there is no RM-only or TTV-only
# `PlanetDataSources` — every RM mode is RV+PM+RM because the kernel needs the
# transit geometry. These fits therefore go through the JOB_CONFIG route, which
# does support them. The signature stays technique-shaped; the config is an
# implementation detail the caller never sees.

const _RM_MODE = Dict(:rm => "RVPM_RM", :reloaded => "RVPM_RM_R", :arome => "RVPM_RM_A")

"""Assemble the JOB_CONFIG `data` block from RV + transit photometry."""
function _data_block(; rv = nothing, phot = nothing)
    d = Dict{String,Any}()
    if rv !== nothing
        bjd = Float64[]; v = Float64[]; e = Float64[]; inst = String[]
        for (name, blob) in pairs(rv)
            b = blob isa NamedTuple ? blob : NamedTuple(Symbol(k) => x for (k,x) in blob)
            append!(bjd, Float64.(b.t)); append!(v, Float64.(b.rv))
            append!(e, Float64.(b.rv_err)); append!(inst, fill(String(name), length(b.t)))
        end
        d["rv"] = Dict("values" => Dict("bjd" => bjd, "rv" => v,
                                        "rv_err" => e, "instrument" => inst))
    end
    if phot !== nothing
        blocks = Any[]
        for (name, blob) in pairs(phot)
            b = blob isa NamedTuple ? blob : NamedTuple(Symbol(k) => x for (k,x) in blob)
            entry = Dict{String,Any}("instrument" => String(name),
                "values" => Dict("bjd" => Float64.(b.t), "flux" => Float64.(b.flux),
                                 "flux_err" => Float64.(b.flux_err)))
            hasproperty(b, :exposure_time) && (entry["exposure_time"] = b.exposure_time)
            push!(blocks, entry)
        end
        d["transit_photometry"] = blocks
    end
    return d
end

"""Run a JOB_CONFIG and return the same shape the other entry points do."""
function _run_config(cfg::AbstractDict; op::String)
    t0 = time()
    summary = run_job(cfg)
    s = Dict{String,Any}(String(k) => v for (k, v) in pairs(summary))
    s["op"] = op
    haskey(s, "elapsed_sec") || (s["elapsed_sec"] = time() - t0)
    return (; chains = nothing, log_z = get(s, "log_z", NaN), summary = s)
end

"""
    fit_rm(; rv, phot, star, flavour, priors, engine, output_dir, ...)

Rossiter–McLaughlin. Requires BOTH radial velocities covering the transit AND
transit photometry: the RM kernel needs the transit geometry, which is why no
RM-only mode exists.

`flavour` picks the kernel — `:arome` (Boué+2013 CCF, what a Gaussian fit to a
CCF actually measures), `:reloaded`, or `:rm` for the legacy flux-weighted-mean
form, which carries known amplitude and shape error at high vsini/β and should
only be used for methods comparison.
"""
function fit_rm(; rv, phot, star::AbstractDict = Dict(), flavour::Symbol = :arome,
                priors::AbstractDict = Dict(), planets::Int = 1,
                engine = Dict("engine" => "pt"), output_dir = nothing,
                parametrization = Dict("mass" => "K_driven", "time" => "Tc",
                                       "ew" => "sesinw", "geom" => "b_rr",
                                       "use_rho_s" => true),
                extra::AbstractDict = Dict())
    haskey(_RM_MODE, flavour) || throw(ArgumentError(
        "flavour must be one of :rm, :reloaded, :arome; got $(repr(flavour))"))
    cfg = Dict{String,Any}("version" => "1.0",
        "output_dir" => output_dir === nothing ? mktempdir() : String(output_dir),
        "star" => Dict{String,Any}(star),
        "data" => _data_block(; rv, phot),
        "model" => Dict{String,Any}("max_kplanet" => planets,
                                    "planet_modes" => [_RM_MODE[flavour]],
                                    "parametrization" => parametrization,
                                    "stability" => "none"),
        "priors" => Dict{String,Any}(priors),
        "sampler" => _sampler_block(engine))
    merge!(cfg, Dict{String,Any}(extra))
    return _run_config(cfg; op = "fit_rm")
end

"""
    fit_ttv(; phot, transit_times, rv, nbody, ...)

Transit timing variations. Needs photometry (or measured transit times);
`nbody=true` selects the N-body model (TTV_NB_SOURCE) over the analytic one.
"""
function fit_ttv(; phot = nothing, transit_times = nothing, rv = nothing,
                 nbody::Bool = false, planets::Int = 2,
                 star::AbstractDict = Dict(), priors::AbstractDict = Dict(),
                 engine = Dict("engine" => "pt"), output_dir = nothing,
                 extra::AbstractDict = Dict())
    (phot === nothing && transit_times === nothing) && throw(ArgumentError(
        "fit_ttv needs photometry or measured transit_times"))
    mode = rv === nothing ? (nbody ? "PM_TTV_NB" : "PM_TTV") :
                            (nbody ? "RVPM_TTV_NB" : "RVPM_TTV")
    cfg = Dict{String,Any}("version" => "1.0",
        "output_dir" => output_dir === nothing ? mktempdir() : String(output_dir),
        "star" => Dict{String,Any}(star),
        "data" => _data_block(; rv, phot),
        "model" => Dict{String,Any}("max_kplanet" => planets,
                                    "planet_modes" => [mode],
                                    "stability" => "none"),
        "priors" => Dict{String,Any}(priors),
        "sampler" => _sampler_block(engine))
    transit_times === nothing || (cfg["data"]["transit_times"] = transit_times)
    merge!(cfg, Dict{String,Any}(extra))
    return _run_config(cfg; op = "fit_ttv")
end

"""
    fit_binary(; rv, secondary, ...)

Spectroscopic binary. SB1 when `secondary` is nothing; SB2 otherwise, in which
case the component masses come from BOTH amplitudes rather than the SB1 mass
function.
"""
function fit_binary(; rv, secondary = nothing, star::AbstractDict = Dict(),
                    priors::AbstractDict = Dict(), engine = Dict("engine" => "pt"),
                    output_dir = nothing, extra::AbstractDict = Dict())
    mode = secondary === nothing ? "BINARY_RV" : "SB2"
    data = _data_block(; rv)
    secondary === nothing || (data["rv_secondary"] =
        _data_block(; rv = secondary)["rv"])
    cfg = Dict{String,Any}("version" => "1.0",
        "output_dir" => output_dir === nothing ? mktempdir() : String(output_dir),
        "star" => Dict{String,Any}(star), "data" => data,
        "model" => Dict{String,Any}("max_kplanet" => 1,
                                    "planet_modes" => [mode],
                                    "stability" => "none"),
        "priors" => Dict{String,Any}(priors),
        "sampler" => _sampler_block(engine))
    merge!(cfg, Dict{String,Any}(extra))
    return _run_config(cfg; op = "fit_binary")
end

"""JOB_CONFIG sampler block from an engine spec."""
function _sampler_block(engine)
    e = engine isa AbstractDict ? engine : Dict("engine" => String(engine))
    b = Dict{String,Any}("name" => String(get(e, "engine", "pt")))
    for (k, v) in pairs(get(e, "options", Dict()))
        b[String(k)] = v
    end
    return b
end

"""
    fit_tomography(nights; P, a_Rs, inc, vsini, T14, vsys, λs, n_null, ...)

Doppler tomography — recover the sky-projected obliquity λ from the planet's
shadow in the stellar line profile.

NOT a sampler. There is no TOMO source flag and no tomographic likelihood: the
estimator is a matched filter over λ against the predicted shadow track, with
significance from a null distribution built by scrambling the in-transit
frames. That is deliberate — for a rapid rotator the RM signal in RV is a
single number swamped by pulsations, whereas the line profile keeps the spatial
information, and a matched filter on it needs no noise model to be honest about
significance.

`nights` is a vector of NamedTuples/Dicts, one per transit night, each with
`profiles` (n_time × n_velocity), `vgrid`, `times`, that night's own measured
`Tc`, and — for anything pooled across weeks — `bervs`. Which frames are in
transit is derived from `Tc` and `T14`, not passed in. Returns λ, its score,
the p-value from the null, and the full λ scan so a caller can plot the
landscape rather than trust a single number.

Every night's profiles must share a sign convention: a DRS CCF is a dip while a
mask-CCF peaks, and mixing them makes one night subtract from the others.

    r = fit_tomography(nights; P=2.83, a_Rs=5.6, inc=1.48, vsini=95.0, T14=0.16)
    r.summary["lambda_deg"], r.summary["p_value"]
"""
function fit_tomography(nights; P::Real, a_Rs::Real, inc::Real, vsini::Real,
                        T14::Real, vsys::Real = 0.0,
                        λs = range(-π, π; length = 721), n_null::Int = 300,
                        filter_pulsations::Bool = true,
                        rng = Random.default_rng(), output_dir = nothing,
                        engine = nothing, stopping = nothing)
    engine === nothing || @warn "fit_tomography ignores `engine`: the estimator " *
        "is a matched filter, not a sampler. See the docstring."
    t0 = time()
    ns = _tomo_nights(nights)
    isempty(ns) && throw(ArgumentError("fit_tomography: no nights given"))
    # filter_pulsations defaults ON: for a rapid rotator the pulsation power
    # sits right where the shadow does and inflates the significance. It costs
    # signal on data that has none, so it is exposed rather than hardwired.
    λ_best, score, pval, λv, tot = tomogram_pooled(ns; P, a_Rs, inc, vsini,
                                                   vsys, T14, λs, n_null, rng,
                                                   filter_pulsations)
    summary = Dict{String,Any}(
        "op" => "fit_tomography",
        "status" => "ok",
        "lambda_rad" => λ_best,
        "lambda_deg" => rad2deg(λ_best),
        "score" => score,
        "p_value" => pval,
        "n_nights" => length(ns),
        "n_null" => n_null,
        "filter_pulsations" => filter_pulsations,
        "lambda_scan_deg" => rad2deg.(collect(λv)),
        "lambda_scan_score" => collect(tot),
        "elapsed_sec" => time() - t0,
    )
    if output_dir !== nothing
        mkpath(output_dir)
        open(joinpath(output_dir, "summary.json"), "w") do io
            JSON3.write(io, summary)
        end
        summary["output_dir"] = String(output_dir)
    end
    return (; chains = nothing, log_z = NaN, summary)
end

"Normalise night specs (Dict from the wire, or NamedTuple from Julia)."
function _tomo_nights(nights)
    out = Any[]
    for (i, nt) in enumerate(nights)
        d = nt isa AbstractDict ?
            NamedTuple(Symbol(k) => v for (k, v) in nt) : nt
        get1(names...) = begin
            for n in names; hasproperty(d, n) && return getproperty(d, n); end
            error("tomography night $i is missing `$(first(names))` " *
                  "(have: $(join(String.(collect(propertynames(d))), ", ")))")
        end
        prof = get1(:profiles)
        prof = prof isa AbstractMatrix ? Matrix{Float64}(prof) :
               _rows_to_matrix(prof)
        times = Float64.(collect(get1(:times, :t)))
        size(prof, 1) == length(times) || error("tomography night $i: " *
            "profiles has $(size(prof,1)) frames but $(length(times)) times")
        bervs = (hasproperty(d, :bervs) && d.bervs !== nothing) ?
                Float64.(collect(d.bervs)) : nothing
        bervs === nothing || length(bervs) == length(times) ||
            error("tomography night $i: $(length(bervs)) bervs for " *
                  "$(length(times)) frames")
        # `Tc` is per night, not per system: pooling several transits means each
        # one has its own measured mid-time, and folding them on a single
        # ephemeris is exactly the error that smears the shadow away.
        push!(out, (profiles = prof, vgrid = Float64.(collect(get1(:vgrid))),
                    times = times, Tc = Float64(get1(:Tc, :tc)), bervs = bervs))
    end
    return out
end

"JSON gives a matrix as a vector of rows."
function _rows_to_matrix(rows)
    r = collect(rows)
    isempty(r) && return Matrix{Float64}(undef, 0, 0)
    M = Matrix{Float64}(undef, length(r), length(first(r)))
    for (i, row) in enumerate(r); M[i, :] = Float64.(collect(row)); end
    return M
end
