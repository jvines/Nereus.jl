# Chain I/O: save/load posterior chains + metadata as NetCDF (.nc).
# Compatible with Python arviz.from_netcdf() for cross-language plotting.

# Narrow import: a bare `using NCDatasets` also pulls in its `bounds`, which
# collides with the `bounds` accessor/kwarg used elsewhere in Nereus. `import`
# keeps the module binding available for qualified `NCDatasets.Dataset(...)`
# calls; the `using` line brings in only the helpers used unqualified.
import NCDatasets
using NCDatasets: defDim, defVar, dimnames
using MCMCChains

"""
    save_chains(path, chains, params; data=nothing, log_evidence=NaN)

Save posterior chains and model metadata to a NetCDF file.
Readable from Python via `arviz.open_datatree(path)` or
`xarray.open_dataset(path)`.

Stores:
- `posterior/` group: all chain parameters as variables
- Attributes: parameter names, instrument names, M_s, R_s, t_ref, etc.
- Optionally: observed data (t_rv, rv, rv_err, rv_inst)
"""
function save_chains(path::String, chains, params::Params;
                      data::Union{Nothing, Data}=nothing,
                      log_evidence::Float64=NaN)
    layout = params.layout
    chain_names = names(chains, :parameters)
    # MCMCChains.Chains dims are (n_iter, n_params, n_chains). Save the CUBE
    # (iter × chain), NOT a flattened draw axis, so every reader gets the full
    # per-chain structure — proper (multi-chain) Rhat / ESS and per-chain plots.
    # Flatten only in-memory where a single draw-pool is genuinely wanted.
    n_iter_per_chain = size(chains, 1)
    n_chain_groups   = size(chains, 3)

    NCDatasets.Dataset(path, "c") do ds
        # Dimensions: the cube axes.
        defDim(ds, "iter", n_iter_per_chain)
        defDim(ds, "chain", n_chain_groups)
        # Keep the chain count as an attribute too (back-compat + clarity).
        ds.attrib["n_chain_groups"] = n_chain_groups

        # Global attributes
        ds.attrib["n_params"] = length(layout.unfrozen_names)
        ds.attrib["param_names"] = join(layout.unfrozen_names, ",")
        ds.attrib["max_kplanet"] = params.config.max_kplanet
        ds.attrib["M_s"] = isnan(params.config.M_s) ? -1.0 : params.config.M_s
        ds.attrib["R_s"] = isnan(params.config.R_s) ? -1.0 : params.config.R_s
        ds.attrib["stability"] = String(params.config.stability)
        ds.attrib["trend_order"] = params.config.trend_order
        ds.attrib["phot_trend_order"] = params.config.phot_trend_order
        ds.attrib["log_evidence"] = isnan(log_evidence) ? -1e300 : log_evidence
        ds.attrib["instrument_rv"] = join(params.config.instruments.rv_names, ",")
        ds.attrib["instrument_pm"] = join(params.config.instruments.pm_names, ",")

        # Posterior samples — one variable per parameter
        for name in layout.unfrozen_names
            sym = Symbol(name)
            sym in Set(chain_names) || continue
            v = defVar(ds, name, Float64, ("iter", "chain"))
            v[:, :] = Array(chains[sym])
        end

        # Log-posterior if available
        if :lp in Set(chain_names)
            v = defVar(ds, "lp", Float64, ("iter", "chain"))
            v[:, :] = Array(chains[:lp])
        end

        # n_planets for trans-dim
        if :n_planets in Set(chain_names)
            v = defVar(ds, "n_planets", Float64, ("iter", "chain"))
            v[:, :] = Array(chains[:n_planets])
        end

        # Activity indicators for trans-dim runs. Both families are added
        # by the samplers outside of `unfrozen_names` and are needed to
        # condition the posterior downstream: noise_active_* to filter by
        # winning noise model, planet_active_* to condition per-slot params
        # on the slot being OCCUPIED (inactive slots hold stale junk, and
        # the n_planets fallback is order-assuming).
        for sym in chain_names
            s = String(sym)
            (startswith(s, "noise_active_") || startswith(s, "planet_active_")) ||
                continue
            v = defVar(ds, s, Float64, ("iter", "chain"))
            v[:, :] = Array(chains[sym])
        end

        # Observed data
        if data !== nothing
            if length(data.t_rv) > 0
                defDim(ds, "rv_obs", length(data.t_rv))
                defVar(ds, "t_rv", Float64, ("rv_obs",))[:] = data.t_rv
                defVar(ds, "rv", Float64, ("rv_obs",))[:] = data.rv
                defVar(ds, "rv_err", Float64, ("rv_obs",))[:] = data.rv_err
                defVar(ds, "rv_inst", Int32, ("rv_obs",))[:] = data.rv_inst
                ds.attrib["t_ref"] = data.t_ref
                # Activity indicators (ActivityDecorrelation / ActivityGP).
                # Stored as ind_<name> / ind_err_<name> / ind_deriv_<name>;
                # the indicator-name list is stored as a comma-separated
                # string attribute so load_chains can rebuild the Dicts.
                if !isempty(data.indicators)
                    ind_names = sort!(collect(keys(data.indicators)))
                    ds.attrib["indicator_names"] = join(ind_names, ",")
                    for name in ind_names
                        defVar(ds, "ind_$name", Float64, ("rv_obs",))[:] =
                            data.indicators[name]
                        if haskey(data.indicator_errs, name)
                            defVar(ds, "ind_err_$name", Float64, ("rv_obs",))[:] =
                                data.indicator_errs[name]
                        end
                        if haskey(data.indicator_derivs, name)
                            defVar(ds, "ind_deriv_$name", Float64, ("rv_obs",))[:] =
                                data.indicator_derivs[name]
                        end
                    end
                end
            end
            if length(data.t_phot) > 0
                defDim(ds, "phot_obs", length(data.t_phot))
                defVar(ds, "t_phot", Float64, ("phot_obs",))[:] = data.t_phot
                defVar(ds, "flux", Float64, ("phot_obs",))[:] = data.flux
                defVar(ds, "flux_err", Float64, ("phot_obs",))[:] = data.flux_err
                defVar(ds, "phot_inst", Int32, ("phot_obs",))[:] = data.phot_inst
            end
        end
    end

    return path
end


"""
    load_chains(path) -> (chains::MCMCChains.Chains, metadata::Dict)

Load posterior chains from a NetCDF file saved by `save_chains`.
Returns the chains and a metadata dict with attributes.
"""
function load_chains(path::String)
    NCDatasets.Dataset(path, "r") do ds
        # Read metadata
        meta = Dict{String, Any}()
        for (k, v) in ds.attrib
            meta[k] = v
        end

        # Read parameter names
        param_names_str = get(meta, "param_names", "")
        param_names = split(param_names_str, ",")

        # Cube axes. New files store (iter × chain); legacy files store a single
        # flat "draw" axis (= iter × chain, column-major) + n_chain_groups.
        local n_iter, n_chain, read_param
        if haskey(ds.dim, "chain")                         # cube format
            n_iter  = ds.dim["iter"]
            n_chain = ds.dim["chain"]
            read_param = nm -> Array(ds[nm])               # (n_iter, n_chain)
        else                                               # legacy flat "draw"
            n_total = ds.dim["draw"]
            n_chain = Int(get(meta, "n_chain_groups", 1))
            n_iter  = n_total ÷ max(n_chain, 1)
            read_param = nm -> reshape(Array(ds[nm]), n_iter, n_chain)
        end

        # Collect each variable as an (n_iter, n_chain) matrix.
        cols = Matrix{Float64}[]
        syms = Symbol[]
        addcol!(nm, sym) = (haskey(ds, nm) &&
                            (push!(cols, read_param(nm)); push!(syms, sym)))
        for name in param_names
            addcol!(String(name), Symbol(name))
        end
        addcol!("lp", :lp)
        addcol!("n_planets", :n_planets)
        for vname in keys(ds)
            vs = String(vname)
            (startswith(vs, "noise_active_") || startswith(vs, "planet_active_")) &&
                addcol!(vs, Symbol(vname))
        end

        # Activity indicators (rebuild the Dicts the same way Data
        # itself stores them).
        ind_names_str = String(get(meta, "indicator_names", ""))
        if !isempty(ind_names_str)
            ind_dict   = Dict{String, Vector{Float64}}()
            ind_errs   = Dict{String, Vector{Float64}}()
            ind_derivs = Dict{String, Vector{Float64}}()
            for name in split(ind_names_str, ",")
                haskey(ds, "ind_$name") &&
                    (ind_dict[name] = Array(ds["ind_$name"]))
                haskey(ds, "ind_err_$name") &&
                    (ind_errs[name] = Array(ds["ind_err_$name"]))
                haskey(ds, "ind_deriv_$name") &&
                    (ind_derivs[name] = Array(ds["ind_deriv_$name"]))
            end
            meta["indicators"]        = ind_dict
            meta["indicator_errs"]    = ind_errs
            meta["indicator_derivs"]  = ind_derivs
        end

        # Stack into the (iter, param, chain) cube.
        cube = Array{Float64}(undef, n_iter, length(cols), n_chain)
        for (j, c) in enumerate(cols)
            cube[:, j, :] = c
        end
        chains = MCMCChains.Chains(cube, syms)
        return chains, meta
    end
end


"""
    save_lightcurve(path, t, flux, flux_err;
                     flux_detrended=nothing,
                     trend=nothing,
                     mask=nothing,
                     sector_id=nothing,
                     attrs=Dict{String, Any}())

Write a light curve (raw, detrended, or both) to a NetCDF file. JSON
is the wrong format for these — even a modest TESS sector is 20-50k
points, and stringifying that for every diagnostics action chews
hundreds of ms per write + disk space + downstream parse time.

# Variables (all share the `obs` dim)
- `t`, `flux`, `flux_err` — always written
- `flux_detrended` — if provided (the model-subtracted LC)
- `trend` — if provided (the detrending model evaluated on `t`)
- `mask` — if provided, `UInt8` (0/1) marking in-transit / outlier
- `sector_id` — if provided, `Int32` per-point sector tag

# Attributes
- All keys in `attrs` are written as global NetCDF attributes. Use this
  for metadata like `star_id`, `detrend_method`, `window_days`,
  `bls_score`, `R_p_earth`, etc. — anything scalar / short-string the
  downstream consumer needs without re-parsing the LC.
"""
function save_lightcurve(path::String,
                          t::AbstractVector{<:Real},
                          flux::AbstractVector{<:Real},
                          flux_err::AbstractVector{<:Real};
                          flux_detrended::Union{Nothing, AbstractVector{<:Real}}=nothing,
                          trend::Union{Nothing, AbstractVector{<:Real}}=nothing,
                          mask::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                          sector_id::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                          extra_arrays::AbstractDict=Dict{String, AbstractVector{<:Real}}(),
                          segments::Union{Nothing, AbstractVector}=nothing,
                          attrs::AbstractDict=Dict{String, Any}())
    n = length(t)
    length(flux) == n && length(flux_err) == n ||
        throw(ArgumentError("t, flux, flux_err length mismatch"))
    for (name, v) in (("flux_detrended", flux_detrended), ("trend", trend),
                      ("mask", mask), ("sector_id", sector_id))
        v === nothing || length(v) == n ||
            throw(ArgumentError("$name length mismatch (got $(length(v)), expected $n)"))
    end
    for (name, v) in extra_arrays
        length(v) == n ||
            throw(ArgumentError("extra_arrays[$name] length $(length(v)) != obs length $n"))
    end

    NCDatasets.Dataset(path, "c") do ds
        defDim(ds, "obs", n)
        defVar(ds, "t",        Float64, ("obs",))[:] = collect(Float64, t)
        defVar(ds, "flux",     Float64, ("obs",))[:] = collect(Float64, flux)
        defVar(ds, "flux_err", Float64, ("obs",))[:] = collect(Float64, flux_err)
        flux_detrended === nothing ||
            (defVar(ds, "flux_detrended", Float64, ("obs",))[:] =
                collect(Float64, flux_detrended))
        trend === nothing ||
            (defVar(ds, "trend", Float64, ("obs",))[:] =
                collect(Float64, trend))
        mask === nothing ||
            (defVar(ds, "mask", UInt8, ("obs",))[:] =
                UInt8.(collect(mask)))
        sector_id === nothing ||
            (defVar(ds, "sector_id", Int32, ("obs",))[:] =
                Int32.(collect(sector_id)))

        # Method-specific per-obs series (e.g. notch `depth`, `delta_bic`,
        # `dur_best`, `outlier`). Stored under their dict keys; column
        # type follows the input array's eltype, coerced to Float64 for
        # consistency with the rest of the file.
        for (name, v) in extra_arrays
            defVar(ds, String(name), Float64, ("obs",))[:] = collect(Float64, v)
        end

        # `segments` is the partitioning vector returned by detrending
        # methods (vector of UnitRange). Stored as paired start/stop
        # vectors under a separate `segment` dim.
        if segments !== nothing && !isempty(segments)
            nseg = length(segments)
            defDim(ds, "segment", nseg)
            defVar(ds, "segment_start", Int32, ("segment",))[:] =
                Int32.(first.(segments))
            defVar(ds, "segment_stop", Int32, ("segment",))[:] =
                Int32.(last.(segments))
        end

        # Persist user-supplied scalar metadata.
        for (k, v) in attrs
            ds.attrib[String(k)] = v
        end
    end
    return path
end

"""
    save_detrend_result(path, t, flux, flux_err, result; method, attrs=Dict())

Bundle the NamedTuple returned by `detrend_savgol` / `detrend_gp` /
`detrend_notch` (or by `find_transits`'s `detrend_result` field) into
a single NetCDF file. Method-specific per-point arrays land as extra
columns, scalar hyperparams as global attributes.

`method ∈ (:savgol, :gp, :notch, :locor)` controls which fields are unpacked.

Always written: `t`, `flux`, `flux_err`, plus `flux_detrended`,
`trend`, `mask` (= `transit_mask`) when present, `segment_*` when
`segments` is present. Method-specific extras:

- `:savgol` — nothing extra (savgol stores only weights, no per-obs
  diagnostics).
- `:gp` — `gp_params` + per-segment `offsets` go into `attrs`
  (`gp_params_*` keyed by position).
- `:notch` — `depth`, `delta_bic`, `dur_best`, `outlier` as per-obs
  columns.
- `:locor` — `phase`, `cycle_id`, `outlier`, `used_notch` as per-obs
  columns; `P_rot_used` as a (finite) global attr.

Caller-supplied `attrs` is merged in last (its keys win).
"""
function save_detrend_result(path::String,
                              t::AbstractVector{<:Real},
                              flux::AbstractVector{<:Real},
                              flux_err::AbstractVector{<:Real},
                              result;
                              method::Symbol,
                              attrs::AbstractDict=Dict{String, Any}())
    method in (:savgol, :gp, :notch, :locor) ||
        throw(ArgumentError("method must be :savgol | :gp | :notch | :locor (got $method)"))

    extras = Dict{String, AbstractVector{<:Real}}()
    final_attrs = Dict{String, Any}("detrend_method" => String(method))

    if method === :notch
        # Notch carries per-obs diagnostics from the sliding-window fit.
        for k in (:depth, :delta_bic, :dur_best, :outlier)
            hasproperty(result, k) || continue
            v = getproperty(result, k)
            v === nothing && continue
            extras[String(k)] = v
        end
    elseif method === :locor
        # LOCoR carries per-obs diagnostics from the per-cycle fit.
        # cycle_id / used_notch are not Float64 but save_lightcurve coerces
        # them — acceptable for a diagnostics file.
        for k in (:phase, :cycle_id, :outlier, :used_notch)
            hasproperty(result, k) || continue
            v = getproperty(result, k)
            v === nothing && continue
            extras[String(k)] = v
        end
        # Rotation period used (skip when NaN — NetCDF attrs dislike NaN).
        if hasproperty(result, :P_rot_used) && isfinite(result.P_rot_used)
            final_attrs["P_rot_used"] = Float64(result.P_rot_used)
        end
    elseif method === :gp
        # GP hyperparams + per-segment offsets are small — stash as
        # NetCDF global attrs (positional keys for the GP vector).
        if hasproperty(result, :gp_params) && result.gp_params !== nothing
            for (i, v) in enumerate(result.gp_params)
                final_attrs["gp_param_$i"] = Float64(v)
            end
        end
        if hasproperty(result, :offsets) && result.offsets !== nothing
            for (i, v) in enumerate(result.offsets)
                final_attrs["segment_offset_$i"] = Float64(v)
            end
        end
    end
    # :savgol has no extra arrays / scalars worth persisting.

    # User-supplied attrs override method-derived ones.
    for (k, v) in attrs
        final_attrs[String(k)] = v
    end

    transit_mask = hasproperty(result, :transit_mask) ?
                    result.transit_mask : nothing
    mask_int = transit_mask === nothing ? nothing :
                Int.(collect(transit_mask))

    # Per-cadence sector tag, echoed by all detrenders (nothing when the
    # caller did not provide one). Persists sector identity alongside the
    # detrended LC so downstream can slice multi-sector concatenations back.
    sid = hasproperty(result, :sector_id) ? result.sector_id : nothing

    return save_lightcurve(path, t, flux, flux_err;
                            flux_detrended = hasproperty(result, :flux_detrended) ?
                                              result.flux_detrended : nothing,
                            trend = hasproperty(result, :trend) ? result.trend : nothing,
                            mask = mask_int,
                            sector_id = sid,
                            extra_arrays = extras,
                            segments = hasproperty(result, :segments) ?
                                        result.segments : nothing,
                            attrs = final_attrs)
end

"""
    load_lightcurve(path) -> NamedTuple

Read a NetCDF light curve written by [`save_lightcurve`](@ref). Returns
a NamedTuple with the variables that were saved (`t`, `flux`,
`flux_err`, plus any of `flux_detrended`, `trend`, `mask`, `sector_id`
that were present), and an `attrs::Dict{String, Any}` field with the
global attributes.
"""
function load_lightcurve(path::String)
    NCDatasets.Dataset(path, "r") do ds
        out = Dict{Symbol, Any}()
        out[:t]        = Array(ds["t"])
        out[:flux]     = Array(ds["flux"])
        out[:flux_err] = Array(ds["flux_err"])
        for name in ("flux_detrended", "trend", "mask", "sector_id")
            haskey(ds, name) && (out[Symbol(name)] = Array(ds[name]))
        end
        # Method-specific per-obs extras (e.g. notch `depth`/`delta_bic`,
        # LOCoR `phase`/`cycle_id`/`outlier`/`used_notch`) live on the `obs`
        # dim under their own variable names. Surface any obs-dimensioned
        # variable we haven't already grabbed so `save_detrend_result`
        # diagnostics round-trip as fields of the returned NamedTuple.
        if haskey(ds.dim, "obs")
            already = Set(keys(out))
            for vname in keys(ds)
                sym = Symbol(vname)
                sym in already && continue
                v = ds[vname]
                dimnames(v) == ("obs",) || continue
                out[sym] = Array(v)
            end
        end
        attrs = Dict{String, Any}()
        for (k, v) in ds.attrib
            attrs[String(k)] = v
        end
        out[:attrs] = attrs
        return NamedTuple(out)
    end
end

"""
    save_periodogram_set(path, pset::PgramSet)

Write a multi-channel `PgramSet` (the return of `find_rv_planets`) to
NetCDF. JSON for these is brutal — even a modest run is `n_freqs`
(typically 10-50k) × 3 columns × (1 RV + N indicators + 1 window),
or `n_obs × n_freqs` for SBGLS — easily MBs as JSON strings.

# Layout
Global attrs: `method`, `n_obs`, `baseline_days`, `channel_names`
(comma-separated, includes `"window"` last).

For each channel (`rv`, each indicator name, `window`):

  `<chan>_freq[freq]`, `<chan>_period[freq]`, `<chan>_power[freq]`

GLS / BGLS channels additionally carry:

  `<chan>_fap_levels[fap]`, `<chan>_fap_thresholds[fap]`

SBGLS channels carry the cumulative matrix:

  `<chan>_matrix[freq, obs]`, `<chan>_obs_times[obs]`

Per-channel peaks (≤ `max_candidates`):

  `<chan>_peak_period[peak]`, `<chan>_peak_frequency[peak]`,
  `<chan>_peak_power[peak]`, `<chan>_peak_fap[peak]`
"""
function save_periodogram_set(path::String, pset)
    # Collect channels in order: rv, indicators..., window
    channels = Tuple{String, Any}[]
    push!(channels, ("rv", pset.rv))
    for pair in pset.indicators
        push!(channels, (pair.first, pair.second))
    end
    push!(channels, ("window", pset.window))
    channel_names = join((c for (c, _) in channels), ",")

    NCDatasets.Dataset(path, "c") do ds
        ds.attrib["method"] = String(pset.method)
        ds.attrib["n_obs"] = Int(pset.n_obs)
        ds.attrib["baseline_days"] = Float64(pset.baseline_days)
        ds.attrib["channel_names"] = channel_names

        for (chan, pg) in channels
            _save_periodogram_into!(ds, chan, pg)
        end
    end
    return path
end

# Helper: write one periodogram's arrays into `ds` under prefix `chan`.
# Each channel gets its own freq dim (`<chan>_freq_dim`) and peak dim
# (`<chan>_peak_dim`) so heterogeneous-length channels don't fight.
function _save_periodogram_into!(ds, chan::AbstractString, pg)
    freq_dim = "$(chan)_freq_dim"
    defDim(ds, freq_dim, length(pg.frequencies))
    defVar(ds, "$(chan)_freq",   Float64, (freq_dim,))[:] = collect(Float64, pg.frequencies)
    defVar(ds, "$(chan)_period", Float64, (freq_dim,))[:] = collect(Float64, pg.periods)
    defVar(ds, "$(chan)_power",  Float64, (freq_dim,))[:] = collect(Float64, pg.power)

    # FAP info (present on GLS / BGLS-like periodograms).
    if hasproperty(pg, :fap_levels) && hasproperty(pg, :fap_thresholds)
        nfap = length(pg.fap_levels)
        if nfap > 0
            fap_dim = "$(chan)_fap_dim"
            defDim(ds, fap_dim, nfap)
            defVar(ds, "$(chan)_fap_levels",     Float64, (fap_dim,))[:] =
                collect(Float64, pg.fap_levels)
            defVar(ds, "$(chan)_fap_thresholds", Float64, (fap_dim,))[:] =
                collect(Float64, pg.fap_thresholds)
        end
    end

    # SBGLS extras: matrix + obs_times.
    if hasproperty(pg, :matrix) && hasproperty(pg, :obs_times)
        nobs = length(pg.obs_times)
        obs_dim = "$(chan)_obs_dim"
        defDim(ds, obs_dim, nobs)
        defVar(ds, "$(chan)_obs_times", Float64, (obs_dim,))[:] =
            collect(Float64, pg.obs_times)
        # Matrix shape in Nereus is (obs, freq); store as (freq, obs)
        # in NetCDF row-major language — NCDatasets follows Julia's
        # column-major convention so the dim order maps cleanly.
        m = collect(Float64, pg.matrix)
        defVar(ds, "$(chan)_matrix", Float64,
                (freq_dim, obs_dim))[:, :] = reshape(m, length(pg.frequencies), nobs)
    end

    # Peaks (always present, possibly empty).
    npeak = length(pg.peaks)
    peak_dim = "$(chan)_peak_dim"
    defDim(ds, peak_dim, npeak)
    if npeak > 0
        defVar(ds, "$(chan)_peak_period",    Float64, (peak_dim,))[:] =
            Float64[Float64(p.period)    for p in pg.peaks]
        defVar(ds, "$(chan)_peak_frequency", Float64, (peak_dim,))[:] =
            Float64[Float64(p.frequency) for p in pg.peaks]
        defVar(ds, "$(chan)_peak_power",     Float64, (peak_dim,))[:] =
            Float64[Float64(p.power)     for p in pg.peaks]
        defVar(ds, "$(chan)_peak_fap",       Float64, (peak_dim,))[:] =
            Float64[Float64(p.fap)       for p in pg.peaks]
    end
    return
end

"""
    save_acf_result(path, res::NamedTuple)

Write the return of `find_rotation_period` to NetCDF. `res` is the
NamedTuple with `lags`, `acf`, `pacf`, `peaks`, plus scalar fields
`P_rot`, `P_rot_err`, `bin_cadence`, `n_bins`, `threshold`.

# Layout
Global attrs: `P_rot`, `P_rot_err`, `bin_cadence`, `n_bins`, `threshold`.

Vars: `lags[lag]`, `acf[lag]`, optional `pacf[lag]`, and peak arrays
`peak_lag[peak]`, `peak_value[peak]`.
"""
function save_acf_result(path::String, res)
    n = length(res.lags)
    npk = length(res.peaks)
    NCDatasets.Dataset(path, "c") do ds
        defDim(ds, "lag", n)
        defVar(ds, "lags", Float64, ("lag",))[:] = collect(Float64, res.lags)
        defVar(ds, "acf",  Float64, ("lag",))[:] = collect(Float64, res.acf)
        # PACF is shorter than ACF — the Durbin-Levinson recursion is
        # truncated at `lag_max_idx` to keep cost O(L²) tractable. Give
        # it its own dim so we don't need to zero-pad.
        if res.pacf !== nothing
            defDim(ds, "pacf_lag", length(res.pacf))
            defVar(ds, "pacf", Float64, ("pacf_lag",))[:] =
                collect(Float64, res.pacf)
        end

        defDim(ds, "peak", npk)
        if npk > 0
            defVar(ds, "peak_lag",   Float64, ("peak",))[:] =
                Float64[Float64(p.lag) for p in res.peaks]
            defVar(ds, "peak_value", Float64, ("peak",))[:] =
                Float64[Float64(p.value) for p in res.peaks]
        end

        ds.attrib["P_rot"]       = isnan(res.P_rot) ? -1.0 : Float64(res.P_rot)
        ds.attrib["P_rot_err"]   = isnan(res.P_rot_err) ? -1.0 : Float64(res.P_rot_err)
        ds.attrib["bin_cadence"] = Float64(res.bin_cadence)
        ds.attrib["n_bins"]      = Int(res.n_bins)
        ds.attrib["threshold"]   = Float64(res.threshold)
    end
    return path
end

"""
    save_detection_limits(path, res::DetectionLimitResult)

Write the per-bin arrays of a detection-limits run to NetCDF.
Replaces the JSON-array-stuffed `summary["detection_limits"]` dict
for the array fields; the scalar metadata (`planet_index`,
`confidence`, `prior_P_lo`, `prior_P_hi`) is still small enough to
keep in `summary.json`.

The reading side gets a NamedTuple matching the original
`DetectionLimitResult` field names so downstream plot code is
unaffected.
"""
function save_detection_limits(path::String, res)
    n = length(res.P_centers)
    NCDatasets.Dataset(path, "c") do ds
        defDim(ds, "bin", n)
        defDim(ds, "edge", length(res.P_edges))
        defVar(ds, "P_centers", Float64, ("bin",))[:] = collect(Float64, res.P_centers)
        defVar(ds, "P_edges",   Float64, ("edge",))[:] = collect(Float64, res.P_edges)
        defVar(ds, "K_limit",   Float64, ("bin",))[:] = collect(Float64, res.K_limit)
        defVar(ds, "n_in_bin",  Int32,   ("bin",))[:] = Int32.(res.n_in_bin)
        ds.attrib["planet_index"] = Int(res.planet_index)
        ds.attrib["confidence"]   = Float64(res.confidence)
        ds.attrib["prior_P_lo"]   = Float64(res.prior_P_lo)
        ds.attrib["prior_P_hi"]   = Float64(res.prior_P_hi)
    end
    return path
end
