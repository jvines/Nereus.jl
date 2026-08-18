# Convenience builder for NereusTarget.
#
# `build_target(...)` collapses the boilerplate of manually wiring
# `Dict{String, PriorSpec}` keys with `_kN` / `_INSTRUMENT` suffixes,
# picking the right `PlanetDataSources` tag, and configuring
# `ParametrizationConfig.mass` by hand. Pass NamedTuples; everything
# else is auto-derived.
#
# Example
# -------
#
#   target = build_target(
#       M_pri = 0.856,                        # fixed; pass a PriorSpec for sampled
#       planets = (
#           b = (
#               a     = LogUniformPrior(1.0, 100.0),
#               M_sec = LogUniformPrior(0.001, 0.5),
#               sesinw = UniformPrior(-1.0, 1.0),
#               secosw = UniformPrior(-1.0, 1.0),
#               Mo     = UniformPrior(0.0, 2π),
#               inc    = SinePrior(),
#               Omega  = UniformPrior(0.0, 2π),
#           ),
#       ),
#       rv = (HIRES = (data = rvdat, sigma = LogUniformPrior(0.1, 30.0)),),
#       relAST = relast,
#       hgca   = hgca,
#   )
#
# Auto-derived from the inputs:
#   * `n_planets`             — `length(planets)`
#   * `planet_modes`          — RVAS / RVPMAS / RV_ONLY etc. from which
#                               observation kinds are present
#   * `parametrization.mass`  — `:a_driven` if `a` declared, else
#                               `:M_sec_driven` if `M_sec` declared without
#                               `K`, else `:K_driven`
#   * `instruments.rv_names`  — keys of `rv` NamedTuple in order
#   * per-key namespacing     — planet params get `_kN`, instrument
#                               params get `_NAME`
#
# Anything you want to override (parametrization, stability check,
# explicit `M_s` for transit a/R*) is a keyword. Defaults are sensible.

"""
    build_target(; M_pri=nothing, planets=NamedTuple(),
                 rv=NamedTuple(), relAST=nothing, hgca=nothing,
                 iad=nothing, gost=nothing, g23h=nothing,
                 plx=nothing, parametrization=nothing,
                 M_s=NaN, stability=:none) -> NereusTarget

Build a `NereusTarget` from a high-level NamedTuple spec. See module
docstring above for the full block grammar and example.

# Keywords

- `M_pri` — primary mass. Pass a `Real` for fixed (becomes
  `FixedPrior`), a `PriorSpec` for sampled (e.g. `NormalPrior(0.856,
  0.014)` for a Crepp+ 2016-style spectroscopic prior). `nothing`
  leaves it unset (uses default).
- `planets` — `NamedTuple` keyed by planet name (`b`, `c`, …); each
  value is a `NamedTuple` of `param_name => PriorSpec` (or `Real` for
  fixed) entries. Order of keys determines the `_kN` suffix.
- `rv` — `NamedTuple` keyed by RV-instrument name; each value is a
  `NamedTuple` with at least `:data` (an object with `.t`, `.rv`,
  `.rv_err` fields, e.g. the output of `load_orvara_rv`). Other
  entries become per-instrument priors with `_NAME` suffix
  (`sigma_HIRES`, `gamma_HIRES`, …). Multi-instrument: just add more
  named entries.
- `relAST`, `hgca`, `iad`, `gost`, `g23h` — astrometry data objects.
- `plx` — parallax prior. `Real` for fixed, `PriorSpec` for sampled.
- `parametrization` — `:K_driven`, `:M_sec_driven`, or `:a_driven`.
  When `nothing` (default), auto-detected from which params are
  declared.
- `M_s` — primary mass for transit `a/R*` calculations. Defaults to
  `M_pri` if numeric, else `NaN`.
- `stability` — `:none`, `:amd`, or `:gladman`. Default `:none`.

# Returns

A `NereusTarget` ready for any sampler — `sample_pt`, `sample_nuts`,
`sample_pt_warm`, `ofti_sample`, etc.

# Notes

`build_target` is a convenience wrapper around the explicit `Params +
Data + NereusTarget` chain. For full control (custom layout
overrides, noise models, sharing groups, transit photometry), use the
underlying constructors directly.
"""
function build_target(;
    M_pri = nothing,
    planets::NamedTuple = NamedTuple(),
    rv::NamedTuple = NamedTuple(),
    phot::NamedTuple = NamedTuple(),
    relAST = nothing,
    hgca = nothing,
    iad = nothing,
    gost = nothing,
    g23h = nothing,
    gaia_dr3 = nothing,
    plx = nothing,
    parametrization::Union{Nothing, Symbol} = nothing,
    time_anchor::Union{Nothing, Symbol} = nothing,
    M_s::Real = NaN,
    R_s::Real = NaN,
    stability::Symbol = :none,
    trend_order::Int = 0,
    phot_trend_order::Int = 0,
    noise_models = nothing,
    priors::Union{Nothing, Dict{String, <:PriorSpec}} = nothing,
)
    isempty(planets) &&
        throw(ArgumentError("build_target: at least one planet required " *
                            "(pass `planets = (b = (...),)`)"))

    has_rv     = !isempty(rv)
    has_phot   = !isempty(phot)
    has_astrom = (relAST !== nothing) || (hgca !== nothing) ||
                 (iad !== nothing) || (gost !== nothing) ||
                 (g23h !== nothing) || (gaia_dr3 !== nothing)

    mode = if has_rv && has_phot && has_astrom; RVPMAS
          elseif has_rv && has_astrom;          RVAS
          elseif has_rv && has_phot;            RVPM
          elseif has_rv;                        RV_ONLY
          elseif has_phot;                      PM_ONLY
          elseif has_astrom;                    RVAS
          else
              throw(ArgumentError(
                  "build_target: at least one observation kind required " *
                  "(rv, phot, relAST, hgca, iad, gost, or g23h)"))
          end

    mass_kind = parametrization === nothing ?
                _detect_mass_parametrization(planets;
                                             require_mass = has_rv) :
                parametrization

    # Auto-detect time anchor from the first planet's declared params.
    time_kind = if time_anchor !== nothing
        time_anchor
    else
        first_spec = first(values(planets))
        ks = first_spec isa NamedTuple ? keys(first_spec) : ()
        if      :Tc in ks; :Tc
        elseif  :Tp in ks; :Tp
        else                :Mo
        end
    end

    # Assemble RV vectors.
    t_rv    = Float64[]
    rv_arr  = Float64[]
    rv_err  = Float64[]
    rv_inst = Int[]
    rv_names = String[]
    for (i, (name, spec)) in enumerate(pairs(rv))
        push!(rv_names, String(name))
        spec isa NamedTuple ||
            throw(ArgumentError("rv.$name must be a NamedTuple " *
                                "(use `($name = (data = ..., sigma = ...),)`)"))
        haskey(spec, :data) ||
            throw(ArgumentError("rv.$name: missing required `:data` field"))
        src = spec.data
        n_i = length(src.t)
        append!(t_rv,    src.t)
        append!(rv_arr,  src.rv)
        append!(rv_err,  src.rv_err)
        append!(rv_inst, fill(i, n_i))
    end

    # Assemble photometry vectors.
    t_phot    = Float64[]
    flux_arr  = Float64[]
    flux_err  = Float64[]
    phot_inst = Int[]
    phot_names = String[]
    for (i, (name, spec)) in enumerate(pairs(phot))
        push!(phot_names, String(name))
        spec isa NamedTuple ||
            throw(ArgumentError("phot.$name must be a NamedTuple " *
                                "(use `($name = (data = ...),)`)"))
        haskey(spec, :data) ||
            throw(ArgumentError("phot.$name: missing required `:data` field"))
        src = spec.data
        n_i = length(src.t)
        append!(t_phot,    src.t)
        append!(flux_arr,  src.flux)
        append!(flux_err,  src.flux_err)
        append!(phot_inst, fill(i, n_i))
    end

    data = Data(;
        t_rv      = t_rv,
        rv        = rv_arr,
        rv_err    = rv_err,
        rv_inst   = isempty(t_rv) ? nothing : rv_inst,
        t_phot    = t_phot,
        flux      = flux_arr,
        flux_err  = flux_err,
        phot_inst = isempty(t_phot) ? nothing : phot_inst,
        relastrom = relAST,
        hgca      = hgca,
        iad       = iad,
        gost      = gost,
        g23h      = g23h,
        gaia_dr3  = gaia_dr3,
    )

    instruments = InstrumentConfig(rv = rv_names, pm = phot_names)

    # Priors dict — system, planets, instruments.
    priors_dict = Dict{String, PriorSpec}()
    priors_dict["n_p"] = FixedPrior(Float64(length(planets)))

    # M_pri and plx are only LAYOUT slots when the planet has astrometry
    # (RVAS / RVPMAS). For RV-only fits, M_pri's numeric value is used
    # for transit a/R* via M_s instead — see M_s_val below.
    if has_astrom
        if M_pri !== nothing
            priors_dict["M_pri"] = _to_prior_spec(M_pri)
        end
        if plx !== nothing
            priors_dict["plx"] = _to_prior_spec(plx)
        end
    elseif plx !== nothing
        @warn "build_target: plx supplied but no astrometry data — ignored"
    end

    for (k, (pname, pspec)) in enumerate(pairs(planets))
        pspec isa NamedTuple ||
            throw(ArgumentError("planets.$pname must be a NamedTuple"))
        for (param_name, val) in pairs(pspec)
            priors_dict["$(param_name)_k$k"] = _to_prior_spec(val)
        end
    end

    for (name, spec) in pairs(rv)
        for (key, val) in pairs(spec)
            key === :data && continue
            priors_dict["$(key)_$(name)"] = _to_prior_spec(val)
        end
    end

    # M_s default for transit a/R*: numeric M_pri if user passed one,
    # else fall back to user-supplied M_s, else NaN.
    M_s_val = if !isnan(M_s)
        Float64(M_s)
    elseif M_pri isa Real
        Float64(M_pri)
    else
        NaN
    end

    R_s_val = isnan(R_s) ? NaN : Float64(R_s)
    nm_vec = noise_models === nothing ? NoiseModel[] :
             collect(NoiseModel, noise_models)

    # Merge user-supplied priors into the auto-generated dict; user
    # entries (e.g., narrow GP rotation period) override defaults.
    if priors !== nothing
        for (k, v) in priors
            priors_dict[k] = v
        end
    end

    params = Params(
        max_kplanet     = length(planets),
        planet_modes    = [mode for _ in 1:length(planets)],
        instruments     = instruments,
        data            = data,
        stability       = stability,
        M_s             = M_s_val,
        R_s             = R_s_val,
        parametrization = ParametrizationConfig(mass = mass_kind, time = time_kind),
        priors          = priors_dict,
        trend_order     = trend_order,
        phot_trend_order = phot_trend_order,
        noise_models    = nm_vec,
    )
    return NereusTarget(params, data)
end

# Coerce a user-supplied prior value to a PriorSpec. `Real` becomes
# `FixedPrior`; existing `PriorSpec` passes through.
function _to_prior_spec(x)
    if x isa PriorSpec
        return x
    elseif x isa Real
        return FixedPrior(Float64(x))
    else
        throw(ArgumentError(
            "build_target: prior value must be a PriorSpec or a Real " *
            "(got $(typeof(x)))"))
    end
end

# Auto-detect mass parametrization from declared planet parameters.
# Looks at the FIRST planet (all planets share the system-level
# parametrization). Rules:
#   * `a` and `M_sec` declared → `:a_driven`
#   * `M_sec` and `P` declared (no `K`) → `:M_sec_driven`
#   * `K` and `P` declared → `:K_driven`
function _detect_mass_parametrization(planets::NamedTuple;
                                       require_mass::Bool = true)
    isempty(planets) && return :K_driven
    first_spec = first(values(planets))
    first_spec isa NamedTuple || return :K_driven
    keys_set = keys(first_spec)
    if :a in keys_set
        :M_sec in keys_set ||
            throw(ArgumentError(
                "planet declares `a` but not `M_sec` — :a_driven needs both"))
        return :a_driven
    elseif :M_sec in keys_set
        :P in keys_set ||
            throw(ArgumentError(
                "planet declares `M_sec` but not `P` — :M_sec_driven needs both"))
        :K in keys_set &&
            throw(ArgumentError(
                "planet declares both `M_sec` and `K` — pick one parametrization"))
        return :M_sec_driven
    elseif :K in keys_set
        :P in keys_set ||
            throw(ArgumentError(
                "planet declares `K` but not `P` — :K_driven needs both"))
        return :K_driven
    elseif !require_mass
        # T-only (PM_ONLY) fit — no mass param needed; pick `:K_driven`
        # as a no-op default so the layout builder doesn't trip.
        return :K_driven
    else
        throw(ArgumentError(
            "planet must declare a mass parameter: " *
            "one of `a + M_sec`, `P + M_sec`, or `P + K`"))
    end
end
