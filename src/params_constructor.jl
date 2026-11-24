# Public Params constructor.
# Separated from model.jl because it depends on Data and default_priors.

# A Tp or Tc prior wider than one period holds several copies of the same orbit:
# the likelihood repeats every P, and a Uniform prior weighs every copy equally.
# An ensemble cannot move between them -- a stretch between walkers in copies n
# and m lands between copies -- so each walker stays in the copy it found, and
# R-hat fails on every parameter however long the run. Measured on an easy
# 12.3 d RV target with the old run_job default (Tp over the 929 d baseline):
# 100 walkers spread over 46 copies, R-hat 1.9-5 with every walker in the mode.
# φ_c (the default) and Mo cover exactly one period and have no copies.
function _warn_repeating_time_window(priors, max_kplanet::Int, time::Symbol)
    time in (:Tp, :Tc) || return nothing
    for k in 1:max_kplanet
        tname, pname = "$(time)_k$k", "P_k$k"
        (haskey(priors, tname) && haskey(priors, pname)) || continue
        lo, hi = bounds(priors[tname])
        P_min = bounds(priors[pname])[1]
        (isfinite(lo) && isfinite(hi) && P_min > 0 && hi - lo > P_min) || continue
        @warn "$tname's prior spans $(round(hi - lo, sigdigits = 4)) d, but " *
              "$pname can be as short as $(round(P_min, sigdigits = 4)) d. The " *
              "likelihood repeats every period, so that window holds up to " *
              "$(floor(Int, (hi - lo) / P_min)) identical copies of the orbit; an " *
              "ensemble sampler cannot move between them and will not converge. " *
              "Use time = \"phic\" (the default) or time = \"Mo\", or give " *
              "$tname a prior no wider than one period."
    end
    return nothing
end

"""
    Params(; max_kplanet, planet_modes, instruments, data, [priors], ...)

Construct a `Params` with auto-generated default priors from the data.
User-provided priors override defaults. If no data is provided, generic
fallback bounds are used.

# Required
- `max_kplanet::Int`
- `planet_modes::Vector{PlanetDataSources}` — e.g. `[RV_ONLY]`, `[RVPM, RV_ONLY]`

# Recommended
- `instruments::InstrumentConfig` — names of RV / photometric instruments
- `data::Data` — observation data; drives prior bounds (rv_max, baseline, etc.)

# Optional
- `priors::Dict{String, PriorSpec}` — override any auto-generated prior
- `parametrization::ParametrizationConfig` — eccentricity, time anchor, geometry.
  With no `time` given (`:auto`) the anchor is the one `priors` name, if they
  name one (`Mo_k1` → `:Mo`, see `resolve_time_anchor`), else the transit phase
  `:phic` (`:Mo` under `obs_prior`), which needs `data` to fix its epoch.
- `trend_order::Int` — 0 (none), 1 (linear), 2 (quadratic)
- `phot_trend_order` — photometric baseline polynomial order: an integer (0, 1, 2)
  for every PM instrument, or a dictionary `instrument => order` (unlisted
  instruments get none). Each trend is measured in the time of its own offset
  group: centred on the group's mean time and scaled to its half span.
- `noise_models::Vector{NoiseModel}` — noise models to include
- `stability::Symbol` — `:none`, `:amd`, `:gladman`; hard prior on orbital stability
- `M_s::Float64` — stellar mass (M_sun); needed for stability and derived quantities
- `R_s::Float64` — stellar radius (R_sun); needed for transit a/R* computation
- `gaia_plx` — `(plx, plx_err)` in mas (e.g. from `fetch_gaia_parallax`); centres the default `plx` prior of an astrometric fit ahead of the HGCA/G23H catalogue values. A `priors["plx"]` overrides it
- `external_priors::Vector{ExternalPrior}` — priors on derived quantities (ecc, rho_s)
- `sharing::Dict{Symbol, Vector{Vector{String}}}` — instrument parameter sharing groups
- `obliquity::ObliquityConfig` — RM / Doppler-tomography model options
  (per-night CCF widths, occulted-flux formula, spectroscopic limb darkening,
  shared shadow amplitude). The default is the pre-option model.
- `tt_jitter` — planets whose measured transit times (`data.transit_times`) get
  a `sigma_tt_k<p>` jitter. `nothing` (the default) takes them from the sets'
  own `jitter` flags.
- `ttv_n_transits::Dict{Int,Int}` — free per-transit time offsets of a `:TTV`
  planet: `N` gives `ttv_k<p>_t1..tN` on transits 0..N−1, counted from the
  planet's first observed transit (src/ttv.jl).
- `ttv_epochs::Dict{Int,<:AbstractVector{<:Integer}}` — the same, on the
  transits listed (strictly increasing, ≥ 0, counted the same way): slot
  `ttv_k<p>_t<I>` holds transit `epochs[I]`, and every unlisted transit has no
  offset. For sparse transit times. `ttv_n_transits = N` is
  `ttv_epochs = 0:N-1`; naming a planet in both is an error.

# Examples

Minimal (all priors auto-generated from data):
```julia
params = Params(max_kplanet=1, planet_modes=[RV_ONLY],
                instruments=InstrumentConfig(rv=["HARPS"]), data=data)
```

With overrides:
```julia
params = Params(max_kplanet=1, planet_modes=[RV_ONLY],
                instruments=InstrumentConfig(rv=["HARPS"]), data=data,
                priors=Dict("P_k1" => LogUniformPrior(3.0, 5.0)))
```
"""
function Params(;
    max_kplanet::Int,
    planet_modes::AbstractVector{PlanetDataSources},
    instruments::InstrumentConfig = InstrumentConfig(),
    parametrization::ParametrizationConfig = ParametrizationConfig(),
    priors::Union{Nothing, Dict{String, <:PriorSpec}} = nothing,
    data::Union{Nothing, Data} = nothing,
    trend_order::Int = 0,
    phot_trend_order::Union{Integer, AbstractDict} = 0,
    noise_models::Vector{<:NoiseModel} = NoiseModel[],
    stability::Symbol = :amd,
    M_s::Float64 = NaN,
    R_s::Float64 = NaN,
    external_priors::Vector{ExternalPrior} = ExternalPrior[],
    sharing::Dict{Symbol, Vector{Vector{String}}} = Dict{Symbol, Vector{Vector{String}}}(),
    transdim_noise::Bool = false,
    ttv_n_transits::Dict{Int, Int} = Dict{Int, Int}(),
    ttv_epochs::AbstractDict = Dict{Int, Vector{Int}}(),
    ttv_backend::Symbol = :ttvfaster,
    obliquity::ObliquityConfig = ObliquityConfig(),
    gaia_plx = nothing,
    tt_jitter::Union{Nothing, AbstractVector{<:Integer}} = nothing,
)
    # `time = :auto` (no anchor given): the anchor the priors name, else the
    # default. Resolved first, so everything below -- and the Params, its
    # checkpoint fingerprint and its equality -- sees the anchor sampled.
    parametrization = resolve_time_anchor(parametrization, priors)
    max_kplanet >= 0 || throw(ArgumentError("max_kplanet must be ≥ 0"))
    trend_order in (0, 1, 2) || throw(ArgumentError(
        "trend_order must be 0, 1, or 2; got $trend_order"))
    phot_trend_order, phot_trend_orders =
        _phot_trend_orders(phot_trend_order, instruments.pm_names)
    stability in (:none, :amd, :gladman) || throw(ArgumentError(
        "stability must be :none, :amd, or :gladman; got :$stability"))
    if stability !== :none && isnan(M_s) && max_kplanet >= 2
        throw(ArgumentError(
            "M_s (stellar mass) is required when stability != :none " *
            "and max_kplanet ≥ 2"))
    end
    for ep in external_priors
        ep.quantity in _VALID_EXTERNAL_QUANTITIES || throw(ArgumentError(
            "unknown external prior quantity :$(ep.quantity); " *
            "valid: $(_VALID_EXTERNAL_QUANTITIES)"))
        if ep.quantity === :rho_s && !parametrization.use_rho_s
            throw(ArgumentError(
                "external prior on :rho_s requires use_rho_s=true in parametrization"))
        end
    end
    for (cat, groups) in sharing
        cat in _VALID_SHARING_CATEGORIES || throw(ArgumentError(
            "unknown sharing category :$cat; " *
            "valid: $(_VALID_SHARING_CATEGORIES)"))
        valid_names = cat in _RV_SHARING_CATEGORIES ? instruments.rv_names :
                      cat in _AS_SHARING_CATEGORIES ? instruments.as_names :
                      instruments.pm_names
        kind = cat in _RV_SHARING_CATEGORIES ? "RV" :
               cat in _AS_SHARING_CATEGORIES ? "astrometric" : "PM"
        for group in groups
            for ins in group
                ins in valid_names || throw(ArgumentError(
                    "instrument '$ins' in :$cat sharing group is not a " *
                    "$kind instrument; available: $(valid_names)"))
            end
        end
    end
    length(planet_modes) == max_kplanet || throw(ArgumentError(
        "planet_modes length ($(length(planet_modes))) must equal " *
        "max_kplanet ($max_kplanet)"))
    ttv_backend in (:ttvfaster, :nbody) || throw(ArgumentError(
        "ttv_backend must be :ttvfaster or :nbody; got :$ttv_backend"))

    modes_vec = Vector{PlanetDataSources}(planet_modes)
    # N-body TTVs come from the planets' masses, and a planet's mass comes from
    # its RV semi-amplitude: a :TTV_NB planet without :RV (PM_TTV_NB) has mass
    # ratio 0 in the integration, pulls on nothing, and with no RV planet
    # beside it the predicted TTVs are zero -- a linear ephemeris, silently.
    for (k, m) in enumerate(modes_vec)
        (has_ttv_nb(m) && !has_rv(m)) || continue
        throw(ArgumentError(
            "planet $k has N-body TTVs without RV ($m): the N-body TTVs need the " *
            "planets' masses, and its mass comes from its RV semi-amplitude, so it " *
            "would be massless and the TTVs it causes zero. Add RV data and use " *
            "RVPM_TTV_NB (or RVPMAS_TTV_NB), or use PM_TTV (free per-transit offsets, " *
            "with model.ttv_n_transits)."))
    end
    ttv_n_transits, ttv_ep = _ttv_slot_config(ttv_n_transits, ttv_epochs)
    tt_jit = tt_jitter !== nothing ? sort!(unique(Int.(collect(tt_jitter)))) :
             data === nothing ? Int[] :
             sort!([s.planet for s in data.transit_times if s.jitter])
    nm_vec = Vector{NoiseModel}(noise_models)
    if !isempty(nm_vec)
        validate_noise_models(nm_vec; transdim=transdim_noise)
        _check_noise_instruments(nm_vec, instruments)
    end

    # The γ-marginalized RV likelihood integrates the per-instrument
    # systemic offset in closed form under a diagonal (white-noise)
    # Gaussian. RV-channel noise models break that: a mean-modifier
    # (ActivityDecorrelation) shifts the conditional-γ residual, and a
    # covariance model (GP / AR / MA / ActivityGP) makes the per-point
    # covariance non-diagonal, so γ no longer has the simple per-group
    # Gaussian conditional. Reject the combination up front rather than
    # silently dropping the marginalization.
    if parametrization.marginalize_gamma
        rv_nm = filter(nm -> noise_channel(nm) === :rv, nm_vec)
        isempty(rv_nm) || throw(ArgumentError(
            "parametrization.marginalize_gamma=true is incompatible with " *
            "RV-channel noise models ($(join(typeof.(rv_nm), ", "))). The " *
            "analytic γ-marginalization assumes a diagonal (white-noise) RV " *
            "covariance at fixed jitter; GP/AR/MA/activity models violate " *
            "that. Sample γ explicitly (marginalize_gamma=false) when using " *
            "RV-channel noise."))
    end

    # φ_c is measured from an epoch of the data being fitted. With data it is
    # fixed here, with the parametrization, so it travels with every Theta: a
    # likelihood evaluated on a subset of the data (LOO, PPC) reads φ_c the same
    # way. Without data there is no epoch yet and it stays NaN, as Mo's epoch
    # (`data.t_ref`) does until a likelihood is evaluated: `NereusTarget` fixes it
    # when the data arrives (`with_phic_epoch`), and a likelihood evaluated
    # before that throws. Without planets there is no φ_c to measure.
    if parametrization.time === :phic && isnan(parametrization.t_ref_c) &&
       max_kplanet > 0 && data !== nothing
        parametrization = _with_t_ref_c(parametrization, phic_reference_epoch(data))
    end

    # Build config with placeholder priors for default generation
    config = ParamsConfig(max_kplanet, parametrization, modes_vec,
                          instruments, Dict{String, PriorSpec}(),
                          trend_order, phot_trend_order, nm_vec, stability, M_s, R_s,
                          external_priors, sharing, ttv_n_transits,
                          ttv_backend, obliquity, tt_jit, ttv_ep, phot_trend_orders)

    # Generate defaults from data (or use generic fallbacks)
    data_for_priors = data !== nothing ? data :
        Data(; t_rv=[0.0, 1000.0], rv=[-100.0, 100.0], rv_err=[1.0, 1.0])
    defaults = default_priors(config, data_for_priors; gaia_plx)

    # User overrides take precedence
    if priors !== nothing
        for (k, v) in priors
            # Under γ-marginalization the offset is integrated out, not sampled,
            # so there is nothing for a prior on it to act on. Accepting one
            # unfroze the slot: the sampler then walked a dimension the
            # likelihood never read, and the "posterior" on the offset was the
            # prior handed in.
            if parametrization.marginalize_gamma && startswith(k, "gamma_") &&
               !is_fixed(v)
                throw(ArgumentError(
                    "a prior was given for `$k`, but " *
                    "parametrization.marginalize_gamma=true integrates the RV " *
                    "offsets analytically and does not sample them, so it " *
                    "cannot be applied. Drop the prior, or set " *
                    "marginalize_gamma=false to sample the offset under it."))
            end
            defaults[k] = v
        end
    end

    _warn_repeating_time_window(defaults, max_kplanet, parametrization.time)

    # Rebuild config with final priors
    config = ParamsConfig(max_kplanet, parametrization, modes_vec,
                          instruments, defaults, trend_order, phot_trend_order, nm_vec,
                          stability, M_s, R_s, external_priors, sharing,
                          ttv_n_transits, ttv_backend, obliquity, tt_jit, ttv_ep,
                          phot_trend_orders)
    layout = _build_layout(config; data=data)
    return Params(config, layout)
end

"""
    _ttv_slot_config(ttv_n_transits, ttv_epochs) -> (counts, epochs)

The free TTV offsets per planet as `ParamsConfig` holds them: `counts[k]` the
number of slots of planet `k` (a listed planet's list length), `epochs[k]` its
transits when listed. Throws for a planet named in both, and for a list that
is not strictly increasing or has a transit before the first observed one.
"""
function _ttv_slot_config(ttv_n_transits::AbstractDict, ttv_epochs::AbstractDict)
    counts = Dict{Int, Int}(Int(k) => Int(v) for (k, v) in ttv_n_transits)
    epochs = Dict{Int, Vector{Int}}()
    for (k0, ep0) in ttv_epochs
        k = Int(k0)
        haskey(counts, k) && throw(ArgumentError(
            "planet $k is in both ttv_n_transits and ttv_epochs. Give one: " *
            "ttv_n_transits = N means ttv_epochs = 0:N-1 (transits counted from " *
            "the planet's first observed one)."))
        ep = Int[(x isa Real && isinteger(x)) ? Int(x) : throw(ArgumentError(
                 "ttv_epochs of planet $k: $x is not an integer transit number."))
                 for x in ep0]
        bad = findfirst(i -> ep[i] <= ep[i - 1], 2:length(ep))
        bad === nothing || throw(ArgumentError(
            "ttv_epochs of planet $k must be strictly increasing (sorted, no " *
            "repeats); it has $(ep[bad]) then $(ep[bad + 1]). Slot ttv_k$(k)_t<I> " *
            "is the I-th transit of the list."))
        (isempty(ep) || ep[1] >= 0) || throw(ArgumentError(
            "ttv_epochs of planet $k starts at $(ep[1]); transits are counted from " *
            "the planet's first observed one, 0, and none before it has data."))
        counts[k] = length(ep)
        isempty(ep) || (epochs[k] = ep)
    end
    return counts, epochs
end

"""
    _phot_trend_orders(spec, pm_names) -> (order::Int, orders::Dict{String, Int})

Split a `phot_trend_order` argument into the default order and the per-instrument
overrides. An integer applies to every PM instrument. A dictionary maps PM
instrument names to orders; instruments it does not list get none. Orders are 0,
1 or 2, and every name must be a PM instrument.
"""
function _phot_trend_orders(spec::Integer, pm_names)
    spec in (0, 1, 2) || throw(ArgumentError(
        "phot_trend_order must be 0, 1, or 2; got $spec"))
    return Int(spec), Dict{String, Int}()
end
function _phot_trend_orders(spec::AbstractDict, pm_names)
    out = Dict{String, Int}()
    for (k, v) in spec
        name = String(k)
        name in pm_names || throw(ArgumentError(
            "phot_trend_order names `$name`, which is not a photometric instrument " *
            "(known: $(join(pm_names, ", ")))"))
        (v isa Integer && v in (0, 1, 2)) || throw(ArgumentError(
            "phot_trend_order[\"$name\"] must be 0, 1, or 2; got $v"))
        out[name] = Int(v)
    end
    return 0, out
end
