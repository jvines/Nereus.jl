# Public Params constructor.
# Separated from model.jl because it depends on Data and default_priors.

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
- `parametrization::ParametrizationConfig` — eccentricity, time anchor, geometry
- `trend_order::Int` — 0 (none), 1 (linear), 2 (quadratic)
- `noise_models::Vector{NoiseModel}` — noise models to include
- `stability::Symbol` — `:none`, `:amd`, `:gladman`; hard prior on orbital stability
- `M_s::Float64` — stellar mass (M_sun); needed for stability and derived quantities
- `R_s::Float64` — stellar radius (R_sun); needed for transit a/R* computation
- `external_priors::Vector{ExternalPrior}` — priors on derived quantities (ecc, rho_s)
- `sharing::Dict{Symbol, Vector{Vector{String}}}` — instrument parameter sharing groups

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
    phot_trend_order::Int = 0,
    noise_models::Vector{<:NoiseModel} = NoiseModel[],
    stability::Symbol = :amd,
    M_s::Float64 = NaN,
    R_s::Float64 = NaN,
    external_priors::Vector{ExternalPrior} = ExternalPrior[],
    sharing::Dict{Symbol, Vector{Vector{String}}} = Dict{Symbol, Vector{Vector{String}}}(),
    transdim_noise::Bool = false,
    ttv_n_transits::Dict{Int, Int} = Dict{Int, Int}(),
    ttv_backend::Symbol = :ttvfaster,
)
    max_kplanet >= 0 || throw(ArgumentError("max_kplanet must be ≥ 0"))
    trend_order in (0, 1, 2) || throw(ArgumentError(
        "trend_order must be 0, 1, or 2; got $trend_order"))
    phot_trend_order in (0, 1, 2) || throw(ArgumentError(
        "phot_trend_order must be 0, 1, or 2; got $phot_trend_order"))
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
        valid_names = cat in _RV_SHARING_CATEGORIES ?
            instruments.rv_names : instruments.pm_names
        kind = cat in _RV_SHARING_CATEGORIES ? "RV" : "PM"
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
    nm_vec = Vector{NoiseModel}(noise_models)
    if !isempty(nm_vec)
        validate_noise_models(nm_vec; transdim=transdim_noise)
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

    # Build config with placeholder priors for default generation
    config = ParamsConfig(max_kplanet, parametrization, modes_vec,
                          instruments, Dict{String, PriorSpec}(),
                          trend_order, phot_trend_order, nm_vec, stability, M_s, R_s,
                          external_priors, sharing, ttv_n_transits,
                          ttv_backend)

    # Generate defaults from data (or use generic fallbacks)
    data_for_priors = data !== nothing ? data :
        Data(; t_rv=[0.0, 1000.0], rv=[-100.0, 100.0], rv_err=[1.0, 1.0])
    defaults = default_priors(config, data_for_priors)

    # User overrides take precedence
    if priors !== nothing
        for (k, v) in priors
            defaults[k] = v
        end
    end

    # Rebuild config with final priors
    config = ParamsConfig(max_kplanet, parametrization, modes_vec,
                          instruments, defaults, trend_order, phot_trend_order, nm_vec,
                          stability, M_s, R_s, external_priors, sharing,
                          ttv_n_transits, ttv_backend)
    layout = _build_layout(config; data=data)
    return Params(config, layout)
end
