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

"""
    ENGINES :: Dict{String, Function}

Name-to-sampler table backing the `engine=` string accepted by
[`run_job`](@ref) and the higher-level fit entry points. Keys are the
strings a JSON job config may use; values are the `sample_*` functions
themselves, so `ENGINES["pt_emcee"]` is `sample_pt_emcee`.

`keys(ENGINES)` is the authoritative list of what this build supports —
prefer it over a hardcoded list when validating user input.
"""

"""
    run_engine(target, spec) -> (chains, log_evidence)

Dispatch to a sampler by name, passing only the options it declares. Options
the sampler does not accept are an ERROR, not a silent no-op: a typo'd
`n_round` that quietly does nothing is how you lose a day.
"""
function run_engine(target, spec::AbstractDict)
    name = String(get(spec, "engine", "pt_emcee"))
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
    :within_model, :init_strategy, :bounds, :proposal,
    :method, :ad_backend, :mutation_kernel, :parametrization, :time_anchor,
    :stability,
])

# ---------------------------------------------------------------------------
# Stopping — uniform across engines, separate from sampler-specific knobs
# ---------------------------------------------------------------------------

"""
    Stopping(; max_seconds=nothing, max_evals=nothing, ess_min=nothing,
               rhat_max=nothing, logz_tol=nothing, check_every=nothing)

Engine-independent stopping rules, handed to the `fit_*` functions as
`stopping=`. Every field defaults to `nothing`, meaning "do not stop on this
criterion"; the run then ends on the sampler's own step count. Each field is
translated into the engine's own option, so the same budget reads the same
across engines, and a field an engine cannot honour is an `ArgumentError` that
names the engines that can, never a silent no-op. Giving a field AND the engine
option it maps to is an error unless they agree.

| field         | engine option                          | engines                                   |
|---------------|----------------------------------------|-------------------------------------------|
| `max_seconds` | `max_seconds`                          | `pt_emcee`, `transdim_pt_emcee`           |
| `max_evals`   | `max_evals`                            | `pt_emcee`, `transdim_pt_emcee`           |
| `rhat_max`    | `rhat_threshold` (+ `convergence_stop`) | `pt_emcee`, `transdim_pt_emcee`          |
| `ess_min`     | `tail_ess_threshold` (+ `convergence_stop`) | `pt_emcee`, `transdim_pt_emcee`      |
| `check_every` | `diag_every` (steps between diagnostics) | `pt_emcee`, `transdim_pt_emcee`         |
| `logz_tol`    | `dlogz` (`dlogz_batch` for `nested_dynamic`) | `nested`, `nested_ins`, `daedalus`, `nested_dynamic` |

The PT ensembles end production at the last recorded step and finish normally;
the result and `summary.json` say how (`"stopped"`: `"converged"`,
`"max_seconds"` or `"max_evals"`, with `"stopped_at_step"` and
`"n_draws_kept"`). A limit reached during burn-in, when there is nothing to
keep, is an error. `rhat_max` and `ess_min` gate on the worst gating parameter
(`sample_pt_emcee`, `sample_transdim_pt_emcee` say which).

To stop a run from outside (a worker, a web front end), `fit_*` and `run_job`
also take `stop_file`: a path whose appearance ends the run the same way, with
`"stopped": "requested"`. It defaults to `<output_dir>/STOP`.
"""
Base.@kwdef struct Stopping
    max_seconds::Union{Nothing,Float64} = nothing
    max_evals::Union{Nothing,Int}       = nothing
    ess_min::Union{Nothing,Float64}     = nothing
    rhat_max::Union{Nothing,Float64}    = nothing
    logz_tol::Union{Nothing,Float64}    = nothing
    check_every::Union{Nothing,Int}     = nothing
end

# What each Stopping field becomes in an engine. A field an engine cannot honour
# is an ArgumentError naming the engines that can, never a silent no-op.
# `logz_tol` is the remaining-evidence tolerance of the nested family; the
# dynamic sampler has two stages and the field is the tolerance of the final one.
const _STOPPING_KW = (
    max_seconds = :max_seconds,
    max_evals   = :max_evals,
    ess_min     = :tail_ess_threshold,
    rhat_max    = :rhat_threshold,
    check_every = :diag_every,
    logz_tol    = :dlogz,
)

"""
    _stopping_options(engine_name, stopping, options) -> Dict{Symbol,Any}

Fold a [`Stopping`](@ref) into an engine's options. `options` are the engine's
own keywords as the caller gave them; the result is what `run_engine` should
get. A Stopping field the engine cannot honour, or one that disagrees with an
engine keyword given as well, is an `ArgumentError`.
"""
function _stopping_options(name::AbstractString, stopping, options)
    opts = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(options))
    stopping === nothing && return opts
    st = stopping isa Stopping ? stopping : begin
        d = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(stopping))
        bad = setdiff(keys(d), fieldnames(Stopping))
        isempty(bad) || throw(ArgumentError(
            "unknown Stopping field(s) $(sort!(collect(bad))); the fields are " *
            join(fieldnames(Stopping), ", ")))
        Stopping(; (k => (v === nothing ? nothing : v) for (k, v) in d)...)
    end
    fn = get(ENGINES, String(name), nothing)
    fn === nothing && return opts
    accepted = _kwarg_names(fn)
    for f in fieldnames(Stopping)
        v = getfield(st, f)
        v === nothing && continue
        kw = _STOPPING_KW[f]
        f === :logz_tol && String(name) == "nested_dynamic" && (kw = :dlogz_batch)
        if !(kw in accepted)
            throw(ArgumentError(
                "Stopping($f = $v) cannot be honoured by engine \"$name\": " *
                "it is supported by " *
                join(sort!(String[n for (n, g) in ENGINES
                                  if _STOPPING_KW[f] in _kwarg_names(g) ||
                                     (f === :logz_tol && :dlogz_batch in _kwarg_names(g))]),
                     ", ") *
                ". Drop the field, or pick one of those engines."))
        end
        val = f === :check_every ? Int(v) : f === :max_evals ? Float64(v) : Float64(v)
        if haskey(opts, kw) && !(opts[kw] == val)
            throw(ArgumentError(
                "Stopping($f = $v) conflicts with the engine option " *
                "$kw = $(opts[kw]); give one of them."))
        end
        opts[kw] = val
        if f in (:rhat_max, :ess_min)
            haskey(opts, :convergence_stop) && opts[:convergence_stop] == false &&
                throw(ArgumentError(
                    "Stopping($f = $v) conflicts with the engine option " *
                    "convergence_stop = false; give one of them."))
            opts[:convergence_stop] = true
        end
    end
    return opts
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
function _target_from(channels::AbstractVector, planets; M_s = NaN,
                      trend_order = 0, noise_models = nothing,
                      priors = nothing, stability = :none,
                      external_priors = nothing,
                      # Everything below build_target already supported and
                      # _target_from simply did not forward, so it was
                      # reachable from Julia and not from the fit_* API.
                      parametrization = nothing, time_anchor = nothing,
                      use_rho_s::Bool = false, R_s = NaN, phot_trend_order = 0, as_names = String[],
                      sharing = nothing, transdim_noise = false,
                      ttv_n_transits = nothing, ttv_backend = nothing,
                      # Collects `clip_transit_outliers` reports, when given.
                      iad_reports = nothing,
                      # The star's Gaia identity, for the default `plx` prior of
                      # an astrometric fit (see `fetch_gaia_parallax`). Collects
                      # the lookup into `gaia_report` when given.
                      gaia_id = nothing, name = nothing, gaia_report = nothing)
    rv    = NamedTuple()
    phot  = NamedTuple()
    iad = hgca = gost = relast = nothing
    hip_spec = nothing
    iad_names = String[]          # per IAD instrument; "" = no jitter
    for ch in channels
        src = String(get(ch, "source", ""))
        if src == "RV"
            rv = _rv_namedtuple(ch)
            trend_order = get(ch, "trend_order", trend_order)
        elseif src == "PM"
            phot = _phot_namedtuple(ch)
        elseif src == "AS"
            # ACCUMULATE across AS channels. Assigning here used to mean the
            # last channel silently won, so `fit_joint` with a Hipparcos
            # channel and a Gaia channel quietly fitted only Gaia — and threw
            # away that channel's parallax and primary mass with it.
            ch_iad, ch_names, ch_reps = _resolve_iad(get(ch, "iad", nothing))
            if ch_iad !== nothing
                iad = iad === nothing ? ch_iad : merge_iad(iad, ch_iad)
                append!(iad_names, ch_names)
                iad_reports === nothing || append!(iad_reports, ch_reps)
            end
            ch_relast = _as_relast(get(ch, "relast", nothing))
            if ch_relast !== nothing
                relast = relast === nothing ? ch_relast :
                         merge_relast(relast, ch_relast)
            end
            ch_hgca = _as_hgca(get(ch, "hgca", nothing))
            if ch_hgca !== nothing
                hgca === nothing || throw(ArgumentError(
                    "two AS channels both supply `hgca`; HGCA is one " *
                    "catalogue row per system, not a per-instrument product"))
                hgca = ch_hgca
            end
            ch_gost = _as_gost(get(ch, "gost", nothing))
            if ch_gost !== nothing
                gost === nothing || throw(ArgumentError(
                    "two AS channels both supply `gost`; GOST is one scan " *
                    "plan per sky position"))
                gost = ch_gost
            end
            # The Hipparcos scans of the HGCA Hipparcos epoch (`_hip_scans_for`).
            ch_hip = get(ch, "hip_scans", nothing)
            if ch_hip !== nothing
                (hip_spec === nothing || isequal(hip_spec, ch_hip)) || throw(ArgumentError(
                    "two AS channels supply different `hip_scans`; it is the one " *
                    "Hipparcos scan geometry of the system"))
                hip_spec = ch_hip
            end
            # No parallax/m_pri here: both are priors and belong in `priors`.
        elseif src in ("RM", "RM_R", "RM_A", "TOMO", "TTV", "TTV_NB", "SB")
            # handled by the technique-specific entry points below
        else
            throw(ArgumentError("unknown channel source $(repr(src))"))
        end
    end
    # Fetched by the HGCA row's hip_id unless the IAD already has the scans or
    # the channel says otherwise.
    hip_scans = _hip_scans_for(hip_spec, iad, hgca, nothing)
    # Only pass stellar mass when we actually have one: build_target types
    # M_s::Real, and an RV-only fit legitimately has no stellar mass (K-driven
    # parametrisation needs none). Passing nothing is a TypeError, not a default.
    # plx and M_pri are priors, so they live in `priors` and nowhere else.
    # They used to also be keywords, which meant a prior could be specified in
    # two places, and M_pri additionally doubled as the numeric M_s -- one name
    # meaning two things depending on whether the fit had astrometry.
    plx = M_pri = nothing
    if priors !== nothing
        pr = Dict{String,Any}(String(k) => v for (k, v) in pairs(priors))
        haskey(pr, "plx")   && (plx   = _as_prior(pop!(pr, "plx")))
        haskey(pr, "M_pri") && (M_pri = _as_prior(pop!(pr, "M_pri")))
        priors = isempty(pr) ? nothing :
                 Dict{String, PriorSpec}(k => _as_prior(v) for (k, v) in pr)
    end

    # Gaia DR3 parallax, when asked for: only for a fit with astrometry, and
    # never over a `plx` prior the caller gave. It ranks ahead of the HGCA
    # catalogue value, which is why it is stated as the prior here and not left
    # to the default. A failed lookup throws.
    if plx === nothing && (gaia_id !== nothing || name !== nothing) &&
       (iad !== nothing || hgca !== nothing || gost !== nothing || relast !== nothing)
        g = _gaia_lookup_from_star((; gaia_id = gaia_id, name = name))
        plx = _gaia_plx_prior(g)
        gaia_report === nothing || push!(gaia_report, g)
    end

    # A photometry channel's `dilution` / `jitter` are priors on
    # `dilution_<inst>` / `jitter_<inst>`; the same parameter in `priors` would
    # be a second answer to the same question.
    for (inst, spec) in pairs(phot), key in keys(spec)
        key in (:data, :exposure_time) && continue
        nm = "$(key)_$(inst)"
        priors !== nothing && haskey(priors, nm) && throw(ArgumentError(
            "`$nm` is set by the transit channel's `$key` and again in `priors`; " *
            "give it once"))
    end

    # Pick the default planet block from the channels actually present.
    # `_planet_spec(n)` with no block yields planets with NO parameters at all,
    # and build_target then refuses with "planet must declare a mass
    # parameter". fit_rv and fit_astrometry each pass their own block, so only
    # the multi-channel path (fit_joint) ever hit this -- which meant fit_joint
    # could not run at all with a default planet spec.
    #
    # Astrometry present => a + M_sec: the whole reason to add astrometry is a
    # TRUE mass, and a K-driven block cannot express one.
    planet_spec = planets isa NamedTuple ? planets :
        (iad === nothing && hgca === nothing && gost === nothing && relast === nothing) ?
            _planet_spec(planets) :
            _planet_spec(planets; block = default_astrom_planet())
    kw = Dict{Symbol,Any}(:planets => planet_spec, :rv => rv,
                          :phot => phot, :iad => iad, :hgca => hgca,
                          :gost => gost, :relAST => relast, :plx => plx,
                          :hip_scans => hip_scans,
                          :trend_order => trend_order,
                          :noise_models => noise_models, :priors => priors,
                          :stability => stability,
                          :external_priors => _as_external(external_priors),
                          :phot_trend_order => phot_trend_order,
                          :as_names => as_names,
                          :iad_names => any(!isempty, iad_names) ? iad_names : String[])
    use_rho_s && (kw[:use_rho_s] = true)
    parametrization === nothing || (kw[:parametrization] = Symbol(parametrization))
    time_anchor     === nothing || (kw[:time_anchor]     = Symbol(time_anchor))
    isnan(R_s) || (kw[:R_s] = R_s)
    sharing === nothing || (kw[:sharing] = Dict{Symbol,Vector{Vector{String}}}(
        Symbol(k) => [String.(g) for g in v] for (k, v) in pairs(sharing)))
    transdim_noise && (kw[:transdim_noise] = true)
    ttv_n_transits === nothing || (kw[:ttv_n_transits] =
        Dict{Int,Int}(Int(k) => Int(v) for (k, v) in pairs(ttv_n_transits)))
    ttv_backend === nothing || (kw[:ttv_backend] = Symbol(ttv_backend))
    M_pri === nothing || (kw[:M_pri] = M_pri)
    isnan(M_s) || (kw[:M_s] = M_s)
    return build_target(; kw...)
end

# ---------------------------------------------------------------------------
# Input-shape validation for the fit_* entry points
# ---------------------------------------------------------------------------
#
# Every fit_* function takes a PER-INSTRUMENT MAPPING, not flat vectors:
# `Dict("HARPS" => (t=..., rv=..., rv_err=...))`. Passing the vectors directly
# is the obvious first guess and it used to die four frames down inside a
# generator with `type Array has no field t`, which says nothing about what the
# caller should have written.
#
# Validated here for the same reason `features.jl` validates its payloads: a
# MethodError from deep in dispatch is unreadable, and this is the first call
# anyone makes. The check is shape-only — it never touches values — so it costs
# nothing and cannot reject a legitimate table.
function _require_instrument_map(payload, kind::String, required::NTuple{N,Symbol}) where {N}
    ex = kind == "RV" ?
        """Dict("HARPS" => (t = bjd, rv = rv, rv_err = err))""" :
        """Dict("TESS" => (t = bjd, flux = f, flux_err = ferr))"""

    if payload isa AbstractVector || payload isa Tuple
        throw(ArgumentError(
            "fit_$(lowercase(kind)): expected a per-instrument mapping, got a " *
            "$(typeof(payload)). Name your instrument:\n    $ex"))
    end

    pairs_ = payload isa AbstractDict ?
        [(String(k), v) for (k, v) in payload] :
        [(String(k), getfield(payload, k)) for k in propertynames(payload)]

    isempty(pairs_) && throw(ArgumentError("fit_$(lowercase(kind)): no data given"))

    for (name, tbl) in pairs_
        # A bare vector here means the caller passed columns where instruments
        # belong — `(t=..., rv=...)` instead of `Dict("X" => (t=..., rv=...))`.
        if tbl isa AbstractVector || tbl isa Number
            throw(ArgumentError(
                "fit_$(lowercase(kind)): \"$name\" maps to a $(typeof(tbl)), not a " *
                "table of columns. This usually means the columns were passed " *
                "directly instead of being nested under an instrument name:" *
                "\n    $ex"))
        end
        have = tbl isa AbstractDict ? Set(Symbol.(keys(tbl))) : Set(propertynames(tbl))
        miss = [f for f in required if !(f in have)]
        isempty(miss) || throw(ArgumentError(
            "fit_$(lowercase(kind)): instrument \"$name\" is missing " *
            join(string.(miss), ", ") * " (has " *
            join(sort(string.(collect(have))), ", ") * ")\n    $ex"))
    end
    return payload
end

# ---------------------------------------------------------------------------
# Entry points — one per technique
# ---------------------------------------------------------------------------

"""
    fit_rv(rv; planets, engine, stopping, output_dir, ...)

Fit radial velocities alone.

`rv` is a PER-INSTRUMENT mapping, not flat columns — the instrument name is
what the jitter and offset terms are keyed on, so there is no unnamed form:

    fit_rv(Dict("HARPS" => (t = bjd, rv = rv, rv_err = err)))
    fit_rv(Dict("HARPS" => (...), "HIRES" => (...)); planets = 2)

`planets` is an integer for a fixed-dimension fit or a range for
trans-dimensional model selection over planet count.

The orbital phase is sampled as the transit phase `phic_k<k>` =
2π (Tc − t_ref_c) / P, on U(−π, π), unless `time_anchor` asks for `"Mo"`,
`"Tp"` or `"Tc"` (see `ParametrizationConfig` for why phic is the default:
it removes the low-e ridge between Mo and ω). Mo, Tp and Tc are reported per
draw in the derived table. `fit_transit`, `fit_astrometry` and `fit_joint`
take `time_anchor` the same way.

Stellar keywords are those of a `run_job` star block. `M_s` and `R_s` (M☉, R☉)
are model keywords. `T_eff` (K), `J_mag`, `K_mag`, `Ab` (Bond albedo, default
0.3) and the 1σ uncertainties `sigma_M_s`, `sigma_R_s` and `sigma_T_eff` feed
only the derived table, which draws M★, R★ and T_eff per sample from them; a σ
left out is assumed (10% of M★, 5% of R★, 150 K) and flagged `assumed` in
`run_info.stellar`. The same keywords apply to `fit_transit`,
`fit_astrometry` and `fit_joint`.
"""
function fit_rv(rv; planets = 1, engine = nothing, transdim = nothing,
                plots = nothing, plot_kwargs = nothing, save_pdf = false,
                science::Bool = true, output = nothing,
                stopping = nothing, stop_file = nothing, output_dir = nothing,
                noise = nothing, trend_order = 0, kwargs...)
    _require_instrument_map(rv, "RV", (:t, :rv, :rv_err))
    ch = _as_channel(rv, "RV")
    if !(planets isa NamedTuple)
        d = collect(values(get(ch, "data", Dict())))
        isempty(d) && throw(ArgumentError("fit_rv: no RV data given"))
        allt  = reduce(vcat, [collect(x.t)  for x in d])
        allrv = reduce(vcat, [collect(x.rv) for x in d])
        planets = _planet_spec(planets; block = default_rv_planet(allt, allrv))
    end
    star, kwargs = _split_star_kwargs(kwargs)
    tgt = _target_from([ch], planets;
                       trend_order, noise_models = noise, kwargs...)
    return _finish(tgt, engine, stopping, output_dir; op = "fit_rv", stop_file, transdim, plots, plot_kwargs, save_pdf, science, output,
                   star)
end

"""
    fit_transit(phot; planets, use_rho_s, limb_darkening, rho_star, gravity_darkening, ...)

Fit transit photometry alone.

`phot` is a PER-INSTRUMENT mapping, as for `fit_rv`:

    fit_transit(Dict("TESS" => (t = bjd, flux = f, flux_err = ferr)))

Limb darkening is quadratic, sampled as Kipping's (q1, q2) per instrument
(`:quadratic` or `:kipping`).

`use_rho_s = true` samples the stellar density ρ★ (`rho_s`, solar units) and
turns it and the period into a/R★, as `model.parametrization.use_rho_s` does in
a `run_job` config; the prior on it is `priors["rho_s"]` (default log-uniform on
[0.001, 100]) or an `external_priors` entry on `rho_s`. The same keyword and
semantics as [`fit_joint`](@ref).

The gravity-darkened transit (planet mode `PM_GD`, GD_SOURCE, which constrains
i* separately from λ) is a `run_job` option; this entry point does not build
it. `limb_darkening`, `rho_star` and `gravity_darkening` are accepted at their
defaults only and throw otherwise: they were never read, so
`gravity_darkening = true` fitted a symmetric transit and said nothing.
`rho_star` is replaced by `use_rho_s` plus `priors["rho_s"]`.
"""
function fit_transit(phot; planets = 1, use_rho_s::Bool = false,
                     limb_darkening = :quadratic, rho_star = nothing, gravity_darkening = false,
                     engine = nothing, transdim = nothing,
                plots = nothing, plot_kwargs = nothing, save_pdf = false,
                science::Bool = true, output = nothing, stopping = nothing,
                     stop_file = nothing, output_dir = nothing, kwargs...)
    # A transit CHANNEL (a dict with "source", as `fit_joint` takes) carries the
    # photometry options; a bare instrument map has none.
    phot_ch = phot isa AbstractDict && haskey(phot, "source") ? phot :
              Dict{String,Any}("source" => "PM", "data" => phot)
    _require_instrument_map(phot_ch["data"], "transit", (:t, :flux, :flux_err))
    Symbol(limb_darkening) in (:quadratic, :kipping) || throw(ArgumentError(
        "fit_transit: limb_darkening = $(repr(limb_darkening)) is not implemented; " *
        "the transit model is quadratic, sampled in Kipping's (q1, q2)"))
    rho_star === nothing || throw(ArgumentError(
        "fit_transit: `rho_star` was never read. Sample the stellar density with " *
        "`use_rho_s = true` and put its prior (solar units) in " *
        "`priors = Dict(\"rho_s\" => ...)`"))
    gravity_darkening == false || throw(ArgumentError(
        "fit_transit: `gravity_darkening` was never read, and this entry point " *
        "does not build a gravity-darkened transit; use run_job with " *
        "planet_modes = [\"PM_GD\"]"))
    star, kwargs = _split_star_kwargs(kwargs)
    tgt = _target_from([phot_ch], planets; use_rho_s, kwargs...)
    return _finish(tgt, engine, stopping, output_dir; op = "fit_transit", stop_file, transdim, plots, plot_kwargs, save_pdf, science, output,
                   star)
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
    Dict("catalogue" => "hipparcos", "hip" => 27321)

A VECTOR of specs resolves to one multi-instrument `IADData`, numbered in
the order given — this is how a joint Hipparcos + Gaia DR4 fit is
expressed over the wire:

    [Dict("catalogue" => "hipparcos", "hip" => 27321),
     Dict("catalogue" => "gaia_dr4",  "source_id" => 1457486023639239296)]
"""
resolve_astrometry(spec) = first(_resolve_iad(spec))

"""
    _resolve_iad(spec) -> (iad, jitter_names, outlier_reports)

`resolve_astrometry`, plus the two per-instrument options a spec may carry:

    Dict("catalogue" => "gaia_dr4", "source_id" => ..., "jitter" => true,
         "clip_outliers" => true)

- `"jitter" => true` gives this instrument a `sigma_iad_<name>` along-scan
  jitter (`InstrumentConfig`'s `iad`). `jitter_names` has one entry per
  instrument of the resolved `iad`, `""` for those without.
- `"clip_outliers" => true` (or a number of σ, default 5) runs
  [`clip_transit_outliers`](@ref) on this instrument before it is merged;
  its report is returned in `outlier_reports`, tagged with the name.

`<name>` is the spec's `"instrument"` if given, else `HIP<hip>` /
`GaiaDR4_<source_id>` -- the names a run_job `data.iad` block gets.
"""
function _resolve_iad(spec)
    spec === nothing && return (nothing, String[], Any[])
    if spec isa AbstractVector && !isempty(spec) && !(spec isa AbstractVector{<:Real})
        parts = [_resolve_iad(s) for s in spec]
        filter!(p -> p[1] !== nothing, parts)
        isempty(parts) && return (nothing, String[], Any[])
        return (merge_iad(collect(IADData, first.(parts))),
                reduce(vcat, [p[2] for p in parts]),
                reduce(vcat, [p[3] for p in parts]))
    end
    if !(spec isa AbstractDict)                    # already an IADData
        return (spec, spec isa IADData ? fill("", n_iad_inst(spec)) : String[], Any[])
    end
    iad = _load_iad_spec(spec)
    name = _iad_block_name(spec, 1)
    reports = Any[]
    nsig = _iad_clip_nsigma(_get(spec, :clip_outliers; default = false))
    if nsig !== nothing
        iad, rep = clip_transit_outliers(iad; nsigma = nsig, label = name)
        push!(reports, merge((instrument = name,), rep))
    end
    jit = _iad_jitter_flag(_get(spec, :jitter; default = false))
    return (iad, fill(jit ? name : "", n_iad_inst(iad)), reports)
end

function _load_iad_spec(spec::AbstractDict)
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
        # The catalogue reference solution comes from the file header, not
        # from the caller — this used to pass three keywords `fetch_hip_iad`
        # has never accepted, so every hipparcos spec died with a MethodError.
        return fetch_hip_iad(Int(hip); verbose = false)
    end
    error("unknown astrometry catalogue $(repr(cat)). " *
          "Supported: gaia_dr4, hipparcos")
end

# `jitter`: a boolean, nothing else. A string "true" or a number is a typo for
# something, and the safe reading of a typo here is an error, not a model.
function _iad_jitter_flag(x)
    x isa Bool && return x
    throw(ArgumentError("iad `jitter` must be true or false, got $(repr(x))"))
end

# `clip_outliers`: false/absent = off, true = the default 5σ, a number = that.
function _iad_clip_nsigma(x)
    x === false && return nothing
    x === true && return 5.0
    (x isa Real && !(x isa Bool) && x > 0) && return Float64(x)
    throw(ArgumentError("iad `clip_outliers` must be true, false or a positive " *
                        "number of σ, got $(repr(x))"))
end

# A clip report as plain JSON-able data for a result summary.
_iad_outlier_summary(reports) =
    [Dict{String,Any}(String(k) => v for (k, v) in pairs(r)) for r in reports]

"""
    _as_relast(x) -> RelAstromData or nothing

A channel's relative astrometry. The Python client sends it as a `values` (or
`csv`) block, the form a run_job config takes; it used to reach `Data` as a raw
dict, which rejects it, so relative astrometry could not be fitted from Python.
"""
_as_relast(x) = x
_as_relast(d::AbstractDict) = _parse_relastrom_block(d)
_as_relast(v::AbstractVector{<:AbstractDict}) = merge_relast([_as_relast(d) for d in v])

"""
    _as_hgca(x) -> HGCAData or nothing
    _as_gost(x) -> GOSTData or nothing

As `_as_relast`: the Python client sends HGCA and GOST as JSON blocks, which
reached `Data` as raw dicts and were rejected there, so neither could be
fitted from Python. A block is anything a run_job config takes, or the field
dict the client's `load_hgca` / `load_gost` returns.
"""
_as_hgca(x) = x
_as_hgca(d::AbstractDict) = _parse_hgca_block(d)
_as_gost(x) = x
_as_gost(d::AbstractDict) = _parse_gost_block(d)

"""
    fit_astrometry(; iad, hgca, gost, relast, planets, priors, ...)

Fit astrometry alone, no RV.

`priors["plx"]` should be INFORMATIVE: the abscissae constrain a0 ∝ M_sec·ϖ,
so mass and parallax are degenerate without it. `priors["M_pri"]` sets the
primary mass. Both are priors and both go in `priors` — there is no separate
keyword for either.

`gaia_id` (a Gaia DR3 source_id: an integer, digits, or `"Gaia DR3 <id>"`) or
`name` (resolved to one through Simbad) makes the fit query the Gaia DR3
parallax at runtime and use it as the default `plx` prior, a Gaussian truncated
at ±10σ (raw catalogue value, no zero-point correction). It takes precedence
over the HGCA parallax; a `priors["plx"]` takes precedence over it, and then
nothing is queried. If the query fails the fit stops. See
[`fetch_gaia_parallax`](@ref). The lookup is recorded in
`summary["star"]["gaia"]`.

`hip_scans` is the Hipparcos scan geometry the HGCA Hipparcos epoch is fitted
through. Left out, it is the IAD's own scans when `iad` holds Hipparcos data,
and otherwise fetched by the HGCA row's `hip_id`; `false` models the Hipparcos
epoch instantaneously instead (with a warning). See `_hip_scans_for` for the
other forms. An astrometry channel of `fit_joint` takes the same `"hip_scans"`
key.
"""
function fit_astrometry(; iad = nothing, hgca = nothing, gost = nothing,
                        relast = nothing, hip_scans = nothing, planets = 1,
                        engine = nothing, transdim = nothing,
                plots = nothing, plot_kwargs = nothing, save_pdf = false,
                science::Bool = true, output = nothing,
                        stopping = nothing, stop_file = nothing,
                        output_dir = nothing, kwargs...)
    # NOT resolved here: `_target_from` resolves the spec, and has to see it
    # unresolved to read the per-instrument `jitter` / `clip_outliers` keys.
    ch = Dict("source" => "AS", "iad" => iad, "hgca" => hgca, "gost" => gost,
              "relast" => relast, "hip_scans" => hip_scans)
    planets isa NamedTuple ||
        (planets = _planet_spec(planets; block = default_astrom_planet()))
    star, kwargs = _split_star_kwargs(kwargs)
    reps = Any[]
    greps = Any[]
    tgt = _target_from([ch], planets; iad_reports = reps, gaia_report = greps, kwargs...)
    return _finish(tgt, engine, stopping, output_dir; op = "fit_astrometry", stop_file, transdim, plots, plot_kwargs, save_pdf, science, output,
                   star, iad_outliers = reps, gaia = greps)
end





"""
    fit_joint(channels...; planets, use_rho_s, engine, stopping, ...)

Fit several techniques simultaneously. One operation for "these together",
rather than a named function per combination.

`use_rho_s = true` samples the stellar density ρ★ (`rho_s`, in solar units) and
turns it and the period into the transit's a/R★, as a `run_job` config does with
`model.parametrization.use_rho_s`. It needs a transit channel (`"source" => "PM"`)
among `channels` and throws otherwise. The prior on ρ★ is `priors["rho_s"]`
(default: log-uniform on [0.001, 100]), or an `external_priors` entry on
`rho_s`:

    fit_joint(rv_channel, phot_channel; use_rho_s = true,
              priors = Dict("rho_s" => NormalPrior(1.79, 0.11, 0.1, 10.0)),
              M_s = 0.767, R_s = 0.754)
"""
function fit_joint(channels...; planets = 1, use_rho_s::Bool = false,
                engine = nothing, transdim = nothing,
                plots = nothing, plot_kwargs = nothing, save_pdf = false,
                science::Bool = true, output = nothing,
                   stopping = nothing, stop_file = nothing,
                   output_dir = nothing, kwargs...)
    length(channels) >= 2 || throw(ArgumentError(
        "fit_joint needs at least two channels; for one technique use its " *
        "dedicated entry point, which has a clearer signature"))
    use_rho_s && !any(c -> String(get(c, "source", "")) == "PM", channels) &&
        throw(ArgumentError(
            "fit_joint: `use_rho_s` samples the stellar density through the " *
            "transit a/R★, so it needs a transit channel (\"source\" => \"PM\") " *
            "among the channels"))
    star, kwargs = _split_star_kwargs(kwargs)
    reps = Any[]
    greps = Any[]
    tgt = _target_from(collect(channels), planets; iad_reports = reps,
                       gaia_report = greps, use_rho_s, kwargs...)
    return _finish(tgt, engine, stopping, output_dir; op = "fit_joint", stop_file, transdim, plots, plot_kwargs, save_pdf, science, output,
                   star, iad_outliers = reps, gaia = greps)
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

"""
    _phot_namedtuple(ch) -> NamedTuple

A transit channel's per-instrument photometry for `build_target`, with the
channel's options applied:

- `exposure_time` -- seconds, one value for every instrument or an
  `instrument => seconds` mapping. It switches on finite-exposure
  supersampling (the sub-sample count follows from it: none up to 3 minutes,
  one per minute above, at most 30). Without it the model is instantaneous.
- `dilution`, `jitter` -- a value (fixed), a prior (`PriorSpec` or the wire's
  `{"type": ...}` dict), or an `instrument => value` mapping, for
  `dilution_<instrument>` and `jitter_<instrument>`. `"default"` / absent
  leaves the default prior. Giving the same parameter here and in `priors`
  is an error (checked in `_target_from`).
- `limb_darkening` (`quadratic` / `kipping`: one model, sampled in Kipping's
  (q1, q2)), `gravity_darkening` (false), `supersample` (absent): the only
  values the model has. Anything else throws; they used to be ignored.
"""
function _phot_namedtuple(ch::AbstractDict)
    ld = get(ch, "limb_darkening", "quadratic")
    Symbol(ld) in (:quadratic, :kipping) || throw(ArgumentError(
        "transit channel: limb_darkening = $(repr(ld)) is not implemented; " *
        "the transit model is quadratic, sampled in Kipping's (q1, q2)"))
    get(ch, "gravity_darkening", false) == false || throw(ArgumentError(
        "transit channel: gravity_darkening was never read, and this entry point " *
        "does not build a gravity-darkened transit; use run_job with " *
        "planet_modes = [\"PM_GD\"]"))
    get(ch, "supersample", nothing) === nothing || throw(ArgumentError(
        "transit channel: there is no explicit supersampling count. Set " *
        "`exposure_time` (seconds) and the sub-sample count follows from it " *
        "(none up to 3 minutes, one per minute above, at most 30)"))
    data = get(ch, "data", Dict())
    names = Tuple(Symbol.(collect(keys(data))))
    per(v, nm) = (v isa AbstractDict && !haskey(v, "type")) ?
                 get(v, String(nm), nothing) : v
    vals = Tuple(begin
        pairs_ = Pair{Symbol,Any}[:data => data[String(n)]]
        ex = per(get(ch, "exposure_time", nothing), n)
        ex === nothing || push!(pairs_, :exposure_time => ex)
        for key in ("dilution", "jitter")
            v = per(get(ch, key, nothing), n)
            (v === nothing || v == "default") && continue
            push!(pairs_, Symbol(key) => _as_prior(v))
        end
        NamedTuple(pairs_)
    end for n in names)
    return NamedTuple{names}(vals)
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

It names no time anchor: the fit's `time_anchor` (default the transit phase
`phic`, see `ParametrizationConfig`) picks the slot, and `default_priors` gives
it its one-period prior, `phic_k` on U(−π, π) or `Mo_k` on U(0, 2π). A block
that named one would leave a prior on a slot the model does not have whenever
another anchor is asked for.
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
            secosw = UniformPrior(-1.0, 1.0))
end

"""
Default a+M_sec block for an astrometric companion. Like `default_rv_planet`
it names no time anchor: the fit's `time_anchor` picks it.
"""
default_astrom_planet(; a_lo = 0.1, a_hi = 100.0, m_lo = 1e-4, m_hi = 1.0) =
    (a      = LogUniformPrior(a_lo, a_hi),
     M_sec  = LogUniformPrior(m_lo, m_hi),
     sesinw = UniformPrior(-1.0, 1.0),
     secosw = UniformPrior(-1.0, 1.0),
     inc    = SinePrior(),
     Omega  = UniformPrior(0.0, 2pi))

"""
    _normalize(raw) -> (chains, log_z, extra)

Samplers return different shapes: `sample_pt`/`sample_nested` a
`(chains, log_Z)` tuple, `sample_pt_emcee` a `PTemceeResult`, `sample_map` a
`MAPResult` with no chains at all. Unifying that here is the whole point of a
facade — callers should not branch on which engine they picked.
"""
function _normalize(raw)
    extra = Dict{String,Any}()
    if raw isa Tuple && length(raw) >= 2
        # Slot 2 is log evidence for pt / pt_hmc / nested / daedalus but an
        # EVALUATION COUNT for moms and rjmcmc, which were therefore reporting
        # a count as their evidence. A count is an Int; evidence is a Float.
        lz = raw[2]
        return raw[1], (lz isa Integer ? NaN : lz), extra
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
    # nuts, ensemble, ess and ofti return a BARE MCMCChains.Chains rather than
    # a wrapper struct. `hasproperty(raw, :chains)` is false for one of those
    # (its fields are value/logevidence/name_map/info), so they were handed
    # back chains = nothing: no saved posterior, no params block, no
    # diagnostics, from a run that had sampled perfectly well.
    raw isa MCMCChains.Chains && return raw, NaN, extra
    ch = hasproperty(raw, :chains) ? raw.chains : nothing
    lz = hasproperty(raw, :log_evidence) ? raw.log_evidence : NaN
    for f in (:acceptance_within, :acceptance_swap, :betas, :n_evals)
        hasproperty(raw, f) && (extra[String(f)] = getproperty(raw, f))
    end
    _stop_to_summary!(extra, raw)
    return ch, lz, extra
end


"""
    _summarise_params(chains, params) -> Dict

Per-parameter posterior summary: median plus asymmetric 1/2/3-sigma bounds.
The six bounds come from one sorted copy of the draws (the same values as six
`quantile` calls, which sorted it six times).
"""
function _summarise_params(chains, params)
    out = Dict{String,Any}()
    chains === nothing && return out
    names_in = Set(MCMCChains.names(chains, :parameters))
    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        sym in names_in || continue
        v = _draws(chains, sym)
        q = quantile(v, (0.16, 0.84, 0.025, 0.975, 0.0015, 0.9985))
        out[name] = Dict{String,Float64}(
            "median" => median(v),
            "lo16"   => q[1], "hi84"   => q[2],
            "lo2sig" => q[3], "hi2sig" => q[4],
            "lo3sig" => q[5], "hi3sig" => q[6])
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
"""Coerce `external_priors=` into `Vector{ExternalPrior}`.

Priors on DERIVED quantities rather than sampled parameters. Only `ecc` and
`rho_s` exist (`_VALID_EXTERNAL_QUANTITIES`). Uses the same prior spelling as
the `priors` dict, so you never switch vocabularies mid-call:

    external_priors = [Dict("quantity" => "ecc",
                            "prior" => Dict("type" => "normal",
                                            "mu" => 0.0, "sigma" => 0.3))]

`per_planet` defaults by quantity -- true for `ecc`, false for `rho_s` -- since
that is the only sensible choice for each, and getting it wrong is silent.
"""
_as_external(::Nothing) = ExternalPrior[]
_as_external(v::Vector{ExternalPrior}) = v
_as_external(e::Union{AbstractDict, NamedTuple}) = _as_external([e])
function _as_external(v::AbstractVector)
    out = ExternalPrior[]
    for e in v
        e isa ExternalPrior && (push!(out, e); continue)
        d = Dict{String,Any}(String(k) => val for (k, val) in pairs(e))
        haskey(d, "quantity") || throw(ArgumentError(
            "external_priors entry needs a `quantity` (ecc or rho_s)"))
        q = Symbol(String(d["quantity"]))
        q in (:ecc, :rho_s) || throw(ArgumentError(
            "external prior quantity $(repr(q)) unknown; use :ecc or :rho_s"))
        haskey(d, "prior") || throw(ArgumentError(
            "external_priors entry for $(q) needs a `prior`"))
        pr = d["prior"]
        spec = pr isa PriorSpec ? pr : _as_prior(pr)
        push!(out, ExternalPrior(q, spec,
                                 Bool(get(d, "per_planet", q === :ecc))))
    end
    return out
end

function _as_prior(d::AbstractDict)
    haskey(d, "type") || throw(ArgumentError(
        "a prior dict needs a `type` (one of $(prior_names())); " *
        "got keys $(collect(keys(d)))"))
    t = canonical_prior(d["type"])   # any spelling in PRIOR_ALIASES
    g(k, dflt=nothing) = get(d, k, dflt)
    # `lo`/`hi`, either or both, truncate a normal: a bounded quantity such as an
    # inclination is refused a prior that reaches past its physical range. A
    # truncnorm without either is refused rather than left untruncated.
    if t === :normal || t === :truncnorm
        bounded = haskey(d, "lo") || haskey(d, "hi")
        t === :truncnorm && !bounded && throw(ArgumentError(
            "a truncnorm prior needs `lo` and/or `hi`; without bounds use \"normal\""))
        bounded || return NormalPrior(float(g("mu")), float(g("sigma")))
        return NormalPrior(float(g("mu")), float(g("sigma")),
                           float(g("lo", -Inf)), float(g("hi", Inf)))
    end
    t === :uniform        && return UniformPrior(float(g("lo")), float(g("hi")))
    t === :loguniform     && return LogUniformPrior(float(g("lo")), float(g("hi")))
    t === :modjeffreys    && return ModJeffreysPrior(float(g("knee")), float(g("upper")))
    t === :fixed          && return FixedPrior(float(g("value")))
    t === :sine           && return SinePrior()
    t === :beta           && return BetaPrior(float(g("a")), float(g("b"));
                                              lo = float(g("lo", 0.0)),
                                              hi = float(g("hi", 1.0)))
    t === :wrappeduniform && return WrappedUniformPrior(float(g("lo")), float(g("hi")))
    error("canonical prior :$t has no dict form in _as_prior")
end

"Stellar mass from the target config, or nothing if the fit never needed one."
function _M_s_of(target)
    ms = try target.params.config.M_s catch; nothing end
    (ms === nothing || (ms isa Real && isnan(ms))) ? nothing : ms
end

# The stellar keywords the derived table needs and the model does not: T_eff,
# the J/K magnitudes, the Bond albedo and the 1σ's of M★, R★ and T_eff -- the
# keys of a job's star block other than M_s and R_s, which are model keywords
# and go to the target. Split off before `_target_from`, which takes none of
# them, and handed to `_finish`.
const _STAR_DERIVED_KEYS = (:T_eff, :J_mag, :K_mag, :Ab,
                            :sigma_M_s, :sigma_R_s, :sigma_T_eff)
function _split_star_kwargs(kwargs)
    star = Dict{Symbol,Any}(k => v for (k, v) in pairs(kwargs)
                            if k in _STAR_DERIVED_KEYS)
    rest = (; (k => v for (k, v) in pairs(kwargs)
               if !(k in _STAR_DERIVED_KEYS))...)
    return star, rest
end

"""Coerce a `transdim=` argument into a `TransDimConfig`.

`max_k` comes from the built target, so `planets` is the single place the
planet count is stated and `transdim=true` cannot disagree with it.

    transdim = true                  # birth/death over 1..planets
    transdim = 4                      # explicit max_kplanet
    transdim = (noise = true, ...)    # full config block
    transdim = TransDimConfig(...)    # built yourself
"""
_as_td(::Nothing, max_k) = nothing
_as_td(td::TransDimConfig, max_k) = td
_as_td(td::Bool, max_k) = td ? TransDimConfig(; max_kplanet = max_k) : nothing
_as_td(td::Integer, max_k) = TransDimConfig(; max_kplanet = Int(td))
function _as_td(td::Union{AbstractDict, NamedTuple}, max_k)
    d = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(td))
    haskey(d, :max_kplanet) || (d[:max_kplanet] = max_k)
    return _build_transdim(d)
end
_as_td(td, max_k) = throw(ArgumentError(
    "transdim must be true, an Int, a config block, or a TransDimConfig; " *
    "got $(typeof(td))"))

"""Run the engine, then assemble the standard result."""
function _finish(target, engine, stopping, output_dir; op::String, stop_file = nothing,
                science::Bool = true, output = nothing,
                 transdim = nothing, plots = nothing, plot_kwargs = nothing,
                 save_pdf::Bool = false, star = Dict{Symbol,Any}(),
                 iad_outliers = nothing, gaia = nothing)
    t0 = time()
    # `Stopping` and `stop_file` are folded into the engine's own options below,
    # once the engine is known; a field the engine cannot honour throws.
    # The star as a job's star block holds it: M★ and R★ from the target, the
    # rest (T_eff, magnitudes, albedo, σ's) from the entry point's keywords.
    # Built before the engine runs, so a bad σ fails here and not after it.
    _cfg = target.params.config
    _rs = hasproperty(_cfg, :R_s) && !isnan(_cfg.R_s) ? Float64(_cfg.R_s) : nothing
    st = _build_star(merge(Dict{Symbol,Any}(:M_s => _M_s_of(target), :R_s => _rs),
                           Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(star))))
    td = _as_td(transdim, target.params.config.max_kplanet)
    # Engine defaults by SHAPE, so the trans-dim and fixed-dim paths are
    # symmetric: you ask for trans-dim by saying `transdim=`, not by also
    # having to know which sampler can do it.
    spec = engine === nothing ?
        Dict{String,Any}("engine" =>
            td === nothing ? "pt_emcee" : "transdim_pt_emcee") :
        (engine isa AbstractDict ?
            Dict{String,Any}(String(k) => v for (k, v) in pairs(engine)) :
            Dict{String,Any}("engine" => String(engine)))
    if td !== nothing
        o = Dict{Symbol,Any}(Symbol(k) => v
                             for (k, v) in pairs(get(spec, "options", Dict())))
        o[:td] = td
        spec["options"] = o
    end
    spec_name = String(get(spec, "engine", "pt_emcee"))
    # `Stopping` and `stop_file` become the engine's own options. A file's
    # default is `<output_dir>/STOP`; with no output_dir only an explicit one.
    let o = _stopping_options(spec_name, stopping, get(spec, "options", Dict()))
        sf = stop_file !== nothing ? String(stop_file) :
             (output_dir !== nothing ? joinpath(String(output_dir), "STOP") : nothing)
        if sf !== nothing
            if haskey(ENGINES, spec_name) && :stop_file in _kwarg_names(ENGINES[spec_name])
                haskey(o, :stop_file) || (o[:stop_file] = sf)
            elseif stop_file !== nothing
                throw(ArgumentError(
                    "`stop_file` is honoured by $(join(_engines_with(:stop_file), ", ")); " *
                    "engine \"$spec_name\" cannot be stopped early."))
            end
        end
        isempty(o) || (spec["options"] = o)
    end
    spec_opts = Dict{String,Any}(String(k) => v
                                 for (k, v) in pairs(get(spec, "options", Dict())))
    raw = run_engine(target, spec)
    chains, log_z, extra = _normalize(raw)
    # The re-chart run_job does (src/circular.jl), before sampler_diagnostics,
    # the params and derived blocks, chains.nc or a plot reads a draw. In place,
    # so `raw.chains` stays the same object and the engine result agrees with
    # the chains returned. MAP has no chains and nothing to re-chart. The
    # window the engine sampled in is kept for fit_health (see run_job).
    sampled_windows = circular_windows(target.params)
    chains isa MCMCChains.Chains &&
        recenter_circular!(chains, target.params; transforms = (target.transform,))
    summary = Dict{String,Any}(
        "op" => op,
        "status" => "ok",
        "log_z" => log_z,
        "elapsed_sec" => time() - t0,
    )
    merge!(summary, extra)
    # What `clip_outliers` took out of the data, so a client that never sees
    # the Julia log still sees how many abscissae went, and which.
    (iad_outliers === nothing || isempty(iad_outliers)) ||
        (summary["iad_outliers"] = _iad_outlier_summary(iad_outliers))
    # Where the parallax prior came from, when it was looked up.
    (gaia === nothing || isempty(gaia)) ||
        (summary["star"] = Dict{String,Any}("gaia" => _gaia_summary(gaia[1])))
    # Run health, per engine. Harvested from whatever the engine actually
    # returned plus, where the chain structure supports it, R-hat and ESS --
    # which pt_emcee has always computed for the progress bar and thrown away.
    try
        summary["diagnostics"] = sampler_diagnostics(spec_name, raw, chains)
    catch err
        summary["diagnostics"] = Dict{String,Any}("error" => sprint(showerror, err))
    end
    summary["params"] = _summarise_params(chains, target.params)
    _n_planets_posterior!(summary, chains)
    try
        # compute_derived(chains, params; M_s, R_s, ...) — stellar quantities
        # are keywords, not positional. Only pass what we actually know; the
        # derived block then contains whatever is computable.
        # summarize_derived(chains, params; M_s, ...) does the compute itself.
        # R_s and T_eff were NOT forwarded, only M_s -- so every quantity that
        # needs them silently never computed: planet_radius, planet_density,
        # equilibrium_temperature, incident_flux, TSM, ESM. The values are
        # sitting on target.params.config; nothing was reading them.
        # T_eff and the magnitudes were read off the config too, which has no
        # such fields, so T_eq, insolation, TSM and ESM still never computed.
        # They arrive as keywords now, as in a job's star block (`star`).
        summary["derived"] = chains === nothing ? Dict() :
            summarize_derived(chains, target.params;
                              M_s   = st.M_s, R_s = st.R_s, T_eff = st.T_eff,
                              J_mag = st.J_mag, K_mag = st.K_mag, Ab = st.Ab)
    catch err
        summary["derived_error"] = sprint(showerror, err)
    end
    # The data measure the spin axis but not its sense: ψ is folded to the axis
    # angle in the derived block, i★ and λ are one representative of two.
    spin_flip(target.params) === nothing || (summary["spin_sense"] = SPIN_SENSE_NOTE)
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

        # Nereus's own figures, the same ones run_job renders. These used to be
        # reachable ONLY by dropping to a run_job JSON config, which meant
        # anyone wanting a phase-fold from the simple API had to rewrite their
        # call in a different vocabulary. `plots = ["auto"]` takes the default
        # set for whatever channels are present.
        # One cfg for the whole post-fit block. `run_job` runs PPC, detection
        # limits, LOO, the health guard, the bridge-evidence recompute and the
        # science tables after every fit; `_finish` ran none of them, so the
        # simple API -- the one the Python client actually drives -- returned
        # `ppc = loo = fit_health = detection_limits = nothing` forever, wrote
        # no tables in any format, and reported the tempered log_z that the
        # code's own comment says not to trust. Those helpers all read this
        # shape, so building it once serves plots and science alike.
        ocfg = Dict{String,Any}(
            "plots"       => plots === nothing ? String[] :
                             String.(plots isa AbstractString ? [plots] : collect(plots)),
            "plot_kwargs" => plot_kwargs === nothing ? Dict{String,Any}() : plot_kwargs,
            "save_pdf"    => save_pdf)
        if output !== nothing                      # caller overrides win
            for (k, v) in pairs(output); ocfg[String(k)] = v; end
        end
        cfg = Dict{String,Any}(
            "output"  => ocfg,
            "seed"    => get(spec_opts, "seed", 1),
            # _ensemble_n_walkers reads this to de-interleave per-walker traces
            "sampler" => Dict{String,Any}("name" => spec_name,
                                          "kwargs" => spec_opts))
        if plots !== nothing
            pcfg = cfg
            proot = joinpath(output_dir, "plots")
            before = _png_mtimes(proot)
            _make_plots(pcfg, chains, target.params, target.data, output_dir;
                        result = raw)
            # Manifest: logical name -> path, relative to plots/ without the
            # extension, exactly as run_job builds it. A bare scan also picks
            # up figures an EARLIER fit into the same output_dir left behind,
            # so `plots = ["corner"]` came back advertising three figures with
            # no way to tell which were this run's. `_png_mtimes` snapshots
            # the tree first and only files this call created or rewrote are
            # reported; the rest stay on disk untouched.
            figs = Dict{String,Any}()
            pdfs = Dict{String,Any}()
            if isdir(proot)
                for (root, _, files) in walkdir(proot), f in files
                    ispng = endswith(f, ".png")
                    # save_pdf writes a .pdf beside every .png, and the manifest
                    # filtered on .png -- so a user who asked for publication
                    # figures was never told where they went.
                    ispdf = endswith(f, ".pdf")
                    (ispng || ispdf) || continue
                    full = joinpath(root, f)
                    _is_fresh_png(full, before) || continue
                    key = splitext(relpath(full, proot))[1]
                    ispng ? (figs[key] = full) : (pdfs[key] = full)
                end
            end
            summary["figures"] = figs
            isempty(pdfs) || (summary["figures_pdf"] = pdfs)
        end

        # --- post-fit science, the same sequence run_job runs -------------
        # Every one of these is fail-soft on its own; `science = false` skips
        # the lot, which is what you want while iterating on short chains.
        # The same `post-fit` progress bar as run_job's (see `_PostFit`).
        show_pf = Bool(_get(ocfg, :show_progress; default = true))
        pf = _PostFit(science ? ("ppc", "detection_limits", "loo", "fit_health",
                                 "evidence", "science", "write") : ("write",);
                      enabled = show_pf, release = _large_chain(chains))
        bridge_t = Dict{String, Any}()
        if science
            try
                _postfit_stage!(pf, "ppc")
                _run_ppc!(cfg, chains, target.params, target.data, output_dir, summary)
                _postfit_stage!(pf, "detection_limits")
                _run_detection_limits!(cfg, chains, target.params, target.data,
                                        output_dir, summary)
                _postfit_stage!(pf, "loo")
                _run_loo!(cfg, chains, target.params, target.data, raw, summary)
                _postfit_stage!(pf, "fit_health")
                _run_fit_health!(cfg, chains, target.params, summary;
                                 sampled_windows)
                _postfit_stage!(pf, "evidence")
                # The evidence spread, before augmenting it. `_augment_evidence!`
                # AUGMENTS -- it returns early unless summary["evidence"] is
                # already a Dict -- and on this route nothing ever built one, so
                # the bridge cross-check was being skipped silently even with
                # the call in place. run_job gets this from _populate_summary!.
                # A bare log_z is not quotable: the tempered stack shares one
                # mean_logL array, so its four variants agreeing is not a
                # cross-check and they can sit >100 nats low together.
                _evidence_block!(summary, raw)
                # The tempered stack sits >100 nats low on a signal-locked
                # posterior while its four variants agree with each other to a
                # decimal, so a bridge cross-check is not optional.
                _augment_evidence!(summary, target.params, target.data, chains;
                                   show_progress = show_pf, timings = bridge_t)
                _postfit_stage!(pf, "science")
                # Conditioned, unit-tagged fitted/derived + model_selection +
                # run_info, and tables/ in json, csv, ecsv, dat and TEX.
                # NOTE: run_job deletes summary["params"] here in favour of the
                # science contract. `_finish` must NOT -- `JobResult.params` is
                # the documented accessor on this path and dropping it would
                # break every existing caller. Both blocks coexist.
                sci = science_summary(output_dir, chains, target.params, target.data;
                                       n_walkers = _ensemble_n_walkers(cfg, target.params),
                                       result = raw, _star_science_kwargs(st)...)
                # NOT `figures`. science_summary returns
                #   "figures" => Dict{String,Any}()   # filled by the plotting layer
                # as a placeholder, and copying every key blindly overwrote the
                # manifest built above with an empty dict -- so `plots=["auto"]`
                # rendered every figure to disk and then reported none, which
                # reads exactly like the plots never ran. run_job dodges this by
                # assigning summary["figures"] AFTER this loop; the order here
                # is the other way round, so the key is skipped instead.
                for (k, v) in sci
                    (k == "status" || k == "figures") && continue
                    summary[k] = v
                end
            catch err
                summary["science_error"] = sprint(showerror, err)
            end
        end

        # Where the chains actually are. run_job reports a RELATIVE
        # "chains_path" and no output_dir; `_finish` reported output_dir and no
        # chains_path. Give both, absolute, on both routes.
        summary["chains_path"] = joinpath(output_dir, "chains.nc")
        summary["timing"] = _postfit_timing(pf, bridge_t)
        # An on-disk record of the run. Without this a fit_* output_dir held
        # chains.nc and nothing else: restart the kernel and the posterior
        # summary, diagnostics, evidence and derived block were gone.
        _postfit_stage!(pf, "write")
        try
            _write_summary(output_dir, summary)
            _postfit_done!(pf)
        catch err
            summary["summary_write_error"] = sprint(showerror, err)
            _postfit_done!(pf; stage = "write_failed")
        end
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

"""Every evidence estimate the sampler produced, with its error, and which one
`log_z` reports. Mirrors the block `_populate_summary!` builds for run_job."""
function _evidence_block!(summary::AbstractDict, result)
    result === nothing && return
    ev = Dict{String,Any}()
    for (key, fld) in ("bridge" => :log_evidence_bridge,
                       "mode_laplace" => :log_evidence_laplace)
        hasproperty(result, fld) || continue
        v = getproperty(result, fld)
        v isa Real && isfinite(v) && (ev[key] = Float64(v))
    end
    for rf in (:evidence, :evidence_report)
        hasproperty(result, rf) || continue
        rep = getproperty(result, rf)
        for (key, fld) in ("ti" => :ti, "ti_plus" => :ti_plus,
                           "ss_plus" => :ss_plus, "hybrid" => :hybrid)
            hasproperty(rep, fld) || continue
            t = getproperty(rep, fld)
            (t isa Tuple && length(t) == 2 && isfinite(t[1])) || continue
            ev[key] = Dict("log_z" => Float64(t[1]), "se" => Float64(t[2]))
        end
        hasproperty(rep, :hybrid_beta_star) &&
            (ev["hybrid_beta_star"] = Float64(getproperty(rep, :hybrid_beta_star)))
        break
    end
    isempty(ev) && return
    lz = get(summary, "log_z", NaN)
    which = "tempered"
    if lz isa Real && isfinite(lz)
        for (k, v) in ev
            val = v isa Dict ? get(v, "log_z", NaN) : v
            (val isa Real && isfinite(val) && val ≈ lz) && (which = k; break)
        end
    end
    ev["reported"] = which
    summary["evidence"] = ev
    return
end

"""Assemble the JOB_CONFIG `output` block from the fit_* plotting keywords."""
function _output_block(plots, plot_kwargs, save_pdf, output)
    o = Dict{String,Any}(
        "plots"       => plots === nothing ? String[] :
                         String.(plots isa AbstractString ? [plots] : collect(plots)),
        "plot_kwargs" => plot_kwargs === nothing ? Dict{String,Any}() : plot_kwargs,
        "save_pdf"    => save_pdf)
    output === nothing || for (k, v) in pairs(output); o[String(k)] = v; end
    return o
end

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
                engine = nothing, transdim = nothing,
                plots = nothing, plot_kwargs = nothing, save_pdf = false,
                science::Bool = true, output = nothing, output_dir = nothing,
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
        # The `output` block. It was MISSING on all three config-route fits,
        # which declared `plots`, `plot_kwargs` and `save_pdf` in their
        # signatures and then never referenced them -- so `_make_plots` got an
        # empty list and every RM, TTV and binary fit from Python came back
        # with zero figures and no warning. Julia accepted the keywords
        # because they were declared, which is what made it silent.
        "output"  => _output_block(plots, plot_kwargs, save_pdf, output),
        "sampler" => _sampler_block(engine))
    # Config path: the transdim BLOCK is what enables trans-dim, and
    # _dispatch_sampler defaults to transdim_pt_emcee when it sees one --
    # so these entry points stay symmetric with fit_rv/fit_joint.
    if transdim !== nothing
        td = _as_td(transdim, Int(get(get(cfg, "model", Dict()), "max_kplanet", 1)))
        cfg["transdim"] = Dict{String,Any}(
            "max_kplanet"       => td.max_kplanet,
            "planets"           => td.planets,
            "noise"             => td.noise,
            "transdim_fraction" => td.transdim_fraction)
    end
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
                 engine = nothing, transdim = nothing,
                plots = nothing, plot_kwargs = nothing, save_pdf = false,
                science::Bool = true, output = nothing, output_dir = nothing,
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
        # The `output` block. It was MISSING on all three config-route fits,
        # which declared `plots`, `plot_kwargs` and `save_pdf` in their
        # signatures and then never referenced them -- so `_make_plots` got an
        # empty list and every RM, TTV and binary fit from Python came back
        # with zero figures and no warning. Julia accepted the keywords
        # because they were declared, which is what made it silent.
        "output"  => _output_block(plots, plot_kwargs, save_pdf, output),
        "sampler" => _sampler_block(engine))
    transit_times === nothing || (cfg["data"]["transit_times"] = transit_times)
    # Config path: the transdim BLOCK is what enables trans-dim, and
    # _dispatch_sampler defaults to transdim_pt_emcee when it sees one --
    # so these entry points stay symmetric with fit_rv/fit_joint.
    if transdim !== nothing
        td = _as_td(transdim, Int(get(get(cfg, "model", Dict()), "max_kplanet", 1)))
        cfg["transdim"] = Dict{String,Any}(
            "max_kplanet"       => td.max_kplanet,
            "planets"           => td.planets,
            "noise"             => td.noise,
            "transdim_fraction" => td.transdim_fraction)
    end
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
                    priors::AbstractDict = Dict(), engine = nothing, transdim = nothing,
                plots = nothing, plot_kwargs = nothing, save_pdf = false,
                science::Bool = true, output = nothing,
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
        # The `output` block. It was MISSING on all three config-route fits,
        # which declared `plots`, `plot_kwargs` and `save_pdf` in their
        # signatures and then never referenced them -- so `_make_plots` got an
        # empty list and every RM, TTV and binary fit from Python came back
        # with zero figures and no warning. Julia accepted the keywords
        # because they were declared, which is what made it silent.
        "output"  => _output_block(plots, plot_kwargs, save_pdf, output),
        "sampler" => _sampler_block(engine))
    # Config path: the transdim BLOCK is what enables trans-dim, and
    # _dispatch_sampler defaults to transdim_pt_emcee when it sees one --
    # so these entry points stay symmetric with fit_rv/fit_joint.
    if transdim !== nothing
        td = _as_td(transdim, Int(get(get(cfg, "model", Dict()), "max_kplanet", 1)))
        cfg["transdim"] = Dict{String,Any}(
            "max_kplanet"       => td.max_kplanet,
            "planets"           => td.planets,
            "noise"             => td.noise,
            "transdim_fraction" => td.transdim_fraction)
    end
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
