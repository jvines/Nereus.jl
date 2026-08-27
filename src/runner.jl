# Nereus job runner.
#
# Single entry-point that consumes a config (JSON file path OR an
# in-memory Dict) and runs an end-to-end fit. Used by Dockerised
# batch workers and by Python (juliacall) callers that want to pass
# arrays directly without round-tripping through disk.
#
# Schema spec: Nereus.jl/docs/JOB_CONFIG.md
#
# Two call forms:
#   run_job("/path/to/config.json")   # reads JSON from disk
#   run_job(config::AbstractDict)     # accepts a pre-built dict, arrays inline
#
# Both write `summary.json`, `chains.nc`, and plots into `output_dir`
# AND return the summary dict (so juliacall callers can skip disk).

using JSON3
using DelimitedFiles: readdlm
using Statistics: median, quantile
using Random: MersenneTwister
import Dates

export run_job

# =====================================================================
# Entry points
# =====================================================================

"""
    run_job(config_path::AbstractString) -> Dict
    run_job(cfg::AbstractDict) -> Dict

Run an end-to-end fit from a single job config and return the summary dict.

The one entry point used by the Dockerised batch workers and by Python
(juliacall) callers -- the latter can pass arrays inline through the `Dict`
form and skip disk entirely. Both forms write `summary.json`, `chains.nc`
and the plot set into the config's `output_dir`, *and* return the summary,
so nothing has to be read back to inspect a result.

The config selects the sampler by name through [`ENGINES`](@ref), the
feature operations through [`FEATURE_ACTIONS`](@ref), and carries the data,
priors, noise models and stopping rules for the run.

The full schema is documented in `docs/JOB_CONFIG.md`.

```julia
summary = run_job("configs/hd18599.json")
summary = run_job(Dict("engine" => "ptemcee", "output_dir" => "out", ...))
```
"""
run_job(config_path::AbstractString) = run_job(JSON3.read(read(config_path,
                                                                String); jsonlines = false))

function run_job(cfg::AbstractDict)
    t0 = time()

    # Tomography is not a sampler job. It has no TOMO source flag and no
    # tomographic likelihood — the estimator is a matched filter over lambda
    # against the predicted shadow track, with significance from a scrambled
    # -frame null. So it cannot go through _build_model/_run_sampler below;
    # it is dispatched here as its own job kind, and _validate_config would
    # reject it for lacking `model`/`priors` it does not need.
    if lowercase(String(_get(cfg, :kind; default = ""))) == "tomography"
        return _run_tomography_job(cfg, t0)
    end

    _validate_config(cfg)
    out_dir = String(_get(cfg, :output_dir; required = true))
    mkpath(out_dir)
    mkpath(joinpath(out_dir, "models"))

    summary = Dict{String, Any}("status" => "ok",
                                  "config_version" => String(_get(cfg, :version; default = "1.0")),
                                  "chains_path" => "chains.nc")

    try
        seed = Int(_get(cfg, :seed; default = 1))

        # Honour requested thread cap if provided; otherwise inherit env.
        n_thr = _get(cfg, :n_threads; default = nothing)
        n_thr === nothing || @info "Nereus.run_job: n_threads requested = $n_thr " *
                                   "(actual threading honours JULIA_NUM_THREADS)"

        # --- Build data + star + params --------------------------------
        data, inst_names_rv, inst_names_pm =
            _build_data(_get(cfg, :data; default = Dict()))
        star       = _build_star(_get(cfg, :star; default = Dict()))
        params, target, menu = _build_model(cfg, data, star, inst_names_rv,
                                              inst_names_pm)

        # --- Run sampler (with optional wall-clock watchdog) ----------
        # `timeout_sec` (top-level config) bounds the sampler call so a
        # pathological fit can't tie up a worker forever. Default = no
        # timeout (sampler runs to completion).
        timeout_sec = _get(cfg, :timeout_sec; default = nothing)
        result = _run_sampler_with_timeout(cfg, target, data, seed, timeout_sec; menu = menu)

        # --- Save outputs ----------------------------------------------
        chains = result.chains
        save_chains(joinpath(out_dir, "chains.nc"), chains, params; data = data)

        plot_paths = _make_plots(cfg, chains, params, data, out_dir)
        summary["plots"] = plot_paths
        # figures manifest: scan the rendered plots/ tree → logical_name → path,
        # where logical_name is the path relative to plots/ without extension
        # (e.g. "models/RV_phasefold_P1", "transdim/occupancy",
        # "posteriors/raw/raw_K_k1"). Nereus stays S3-agnostic; the exoautomata
        # worker uploads these and rewrites the paths.
        figs = Dict{String, Any}()
        plots_root = joinpath(out_dir, "plots")
        if isdir(plots_root)
            for (root, _, files) in walkdir(plots_root), f in files
                endswith(f, ".png") || continue
                full = joinpath(root, f)
                key = splitext(relpath(full, plots_root))[1]
                figs[key] = full
            end
        end

        # --- Posterior predictive checks (default on; fail-soft) -------
        _run_ppc!(cfg, chains, params, data, out_dir, summary)

        # --- Detection limits (PT samplers only; fail-soft) ------------
        _run_detection_limits!(cfg, chains, params, data, out_dir, summary)

        # --- PSIS-LOO / WAIC (sampler-agnostic; fail-soft) -------------
        _run_loo!(cfg, chains, params, data, result, summary)

        # --- Post-fit health guard (sampler-agnostic; fail-soft) -------
        # Flags silently-wrong posteriors (disjoint modes, railed bounds,
        # corrupt log-post) as a logged warning + summary entry. Never
        # alters the chains or the rest of the output.
        _run_fit_health!(cfg, chains, params, summary)

        # --- Build summary ---------------------------------------------
        _populate_summary!(summary, result, params, chains)
        # Authoritative science contract: conditioned, unit-tagged fitted/derived
        # + model_selection + run_info, and the multi-format tables. Supersedes
        # the legacy unconditioned `params` block (which is inactive-slot junk in
        # a trans-dim run), so drop it.
        delete!(summary, "params")
        _teff = (hasproperty(params.config, :T_eff) && !isnan(params.config.T_eff)) ?
                params.config.T_eff : nothing
        sci = science_summary(out_dir, chains, params, data;
                              n_walkers = _ensemble_n_walkers(cfg, params),
                              result = result, T_eff = _teff)
        for (k, v) in sci; k == "status" || (summary[k] = v); end
        summary["figures"] = figs

        summary["elapsed_sec"] = time() - t0
        summary["config_path"] = config_path_of(cfg)

        _write_summary(out_dir, summary)
        return summary
    catch err
        @error "Nereus.run_job failed" exception = (err, catch_backtrace())
        summary["status"] = "failed"
        summary["error"]  = sprint(showerror, err)
        summary["traceback"] = sprint(Base.show_backtrace, catch_backtrace())
        summary["elapsed_sec"] = time() - t0
        _write_summary(out_dir, summary)
        rethrow()
    end
end

config_path_of(cfg::AbstractDict) = get(cfg, "config_path",
                                          get(cfg, :config_path, ""))

# =====================================================================
# Helpers — generic config access
# =====================================================================

# JSON3 returns OrderedDict / NamedTuple-ish containers depending on
# parser path. We support any AbstractDict and accept Symbol or String
# keys interchangeably.
function _has(d, k)
    haskey(d, k) || haskey(d, String(k)) || haskey(d, Symbol(k))
end
function _get(d::AbstractDict, k; default = nothing, required::Bool = false)
    v = get(d, k, get(d, String(k), get(d, Symbol(k), missing)))
    if v === missing
        required && error("Missing required config key: $k")
        return default
    end
    v === nothing && required && error("Required config key $k is null")
    return v === nothing ? default : v
end
_to_dict(x::AbstractDict) = x
_to_dict(x) = x  # JSON3.Object behaves dict-like

# Effective ensemble walker count for the active-conditional convergence gate
# (matches the sampler's n_walkers_eff = max(NW, 2·n_dim+2), even). Returns
# `nothing` for non-ensemble samplers, where walker-reshape R̂ doesn't apply.
function _ensemble_n_walkers(cfg, params)
    s = _get(cfg, :sampler; default = Dict())
    name = String(_get(s, :name; default = ""))
    name in ("ptemcee", "transdim_ptemcee", "pt", "pt_warm", "pt_whitening") ||
        return nothing
    kw = _get(s, :kwargs; default = Dict())
    nw = Int(_get(kw, :n_walkers; default = 100))
    ndim = length(params.layout.unfrozen_names)
    neff = max(nw, 2 * ndim + 2)
    return isodd(neff) ? neff + 1 : neff
end

# =====================================================================
# Schema validation — fail FAST with explicit messages before any heavy
# allocation / data parsing.
# =====================================================================

# Schema-known string sets. Kept inline (no dependency on later constants)
# so the validator can run before the rest of the module is fully built.
const _KNOWN_PLANET_MODES   = ("RV_ONLY", "PM_ONLY", "RVPM", "RVAS",
                                "RVPMAS", "RVPM_RM", "RVPMAS_RM",
                                "RVPM_TTV", "RVPMAS_TTV",
                                "RVPM_RM_TTV", "PM_TTV",
                                "RVPM_TTV_NB", "RVPMAS_TTV_NB",
                                "PM_TTV_NB",
                                "RVPM_RM_R", "RVPMAS_RM_R",
                                "RVPM_RM_A", "RVPMAS_RM_A",
                                "RVPM_GD", "RVPM_RM_GD", "PM_GD",  # gravity darkening
                                "BINARY", "BINARY_RV")   # SB2 double-lined binary
const _KNOWN_PARAMETRIZATIONS = (
    mass = ("K_driven", "M_sec_driven", "a_driven"),
    time = ("Tp", "Tc", "Mo"),
    ew   = ("sesinw", "ew"),
    geom = ("b_rr", "b_r"),
)
const _KNOWN_STABILITY      = ("none", "amd", "gladman")
const _KNOWN_PRIORS         = ("UniformPrior", "LogUniformPrior",
                                "ModJeffreysPrior", "NormalPrior",
                                "FixedPrior", "SinePrior", "BetaPrior")
const _KNOWN_NOISE_KINDS    = ("CeleriteRotation", "CeleriteSHO",
                                "CeleriteRotationFM17",
                                "ActivityDecorrelation", "ARModel",
                                "MAModel", "ActivityJitter", "ActivityGP")
const _KNOWN_SAMPLERS       = ("ptemcee", "transdim_ptemcee", "pt",
                                "pt_warm", "rjmcmc", "moms", "moms_ns",
                                "nested", "nested_ins", "nested_dynamic",
                                "pa", "smc", "pt_whitening",
                                "nuts", "pt_hmc", "ofti")
const _SAMPLERS_REQUIRE_TD  = Set(("transdim_ptemcee", "rjmcmc", "moms",
                                    "moms_ns"))
const _KNOWN_BIRTH_STRATEGIES = ("PriorBirth", "InformedBirth",
                                  "JointInformedBirth", "DonorBirth",
                                  "MoMSBirth")
const _KNOWN_PLOTS          = ("rv_timeseries", "rv_components", "rv_phasefold",
                                "pm_timeseries", "pm_phasefold",
                                "rv_astrom_phasefold", "orbit_skyplane",
                                "iad_residuals", "hgca_pm_residuals",
                                "relastrom_timeseries", "relastrom_residuals",
                                "g23h_residuals", "pm_anomaly",
                                "ttv_oc", "transit_overlay", "rm_anomaly",
                                "activity_gp_latent", "activity_gp_decomposition",
                                "corner", "trace", "histograms", "posteriors",
                                "transdim_occupancy", "posteriors_raw",
                                "posteriors_parameters", "posteriors_histograms",
                                "traces_grouped",
                                "auto")

# Collect all schema errors before throwing; gives the user a single
# message with everything wrong instead of fix-one-rerun-cycle hell.
function _validate_config(cfg::AbstractDict)
    errs = String[]
    _has(cfg, :output_dir) || push!(errs, "missing `output_dir`")
    _has(cfg, :sampler)    || push!(errs, "missing `sampler` block")
    _has(cfg, :model)      || push!(errs, "missing `model` block")
    _has(cfg, :data)       || push!(errs, "missing `data` block")

    # ---- data ----
    if _has(cfg, :data)
        data_cfg = _get(cfg, :data)
        has_any = any(_has(data_cfg, k) for k in
            (:rv, :transit_photometry, :iad, :gost, :hgca, :relastrom, :gaia_dr3))
        has_any || push!(errs,
            "`data` block must contain at least one of " *
            "rv, transit_photometry, iad, gost, hgca, relastrom, gaia_dr3")

        if _has(data_cfg, :rv)
            rv = _get(data_cfg, :rv)
            if !_has(rv, :csv) && !_has(rv, :values)
                push!(errs, "`data.rv` requires either `csv` or `values`")
            end
            if _has(rv, :values)
                vals = _get(rv, :values)
                for req in ("bjd", "rv", "rv_err", "instrument")
                    _has(vals, req) || push!(errs,
                        "`data.rv.values.$req` is required")
                end
            end
        end
        if _has(data_cfg, :iad)
            iad = _get(data_cfg, :iad)
            if !_has(iad, :hip_id) && !_has(iad, :values)
                push!(errs, "`data.iad` requires either `hip_id` or `values`")
            end
        end
        if _has(data_cfg, :gost)
            g = _get(data_cfg, :gost)
            ok = _has(g, :values) || (_has(g, :ra_deg) && _has(g, :dec_deg))
            ok || push!(errs,
                "`data.gost` requires either `values` or both `ra_deg` and `dec_deg`")
        end
    end

    # ---- model ----
    if _has(cfg, :model)
        mdl = _get(cfg, :model)
        _has(mdl, :max_kplanet) || push!(errs, "missing `model.max_kplanet`")
        if _has(mdl, :planet_modes)
            for m in _get(mdl, :planet_modes)
                String(m) in _KNOWN_PLANET_MODES || push!(errs,
                    "unknown planet_mode `$m` (known: $(join(_KNOWN_PLANET_MODES, ", ")))")
            end
        end
        if _has(mdl, :parametrization)
            par = _get(mdl, :parametrization)
            for (k, allowed) in pairs(_KNOWN_PARAMETRIZATIONS)
                _has(par, k) || continue
                v = String(_get(par, k))
                v in allowed || push!(errs,
                    "unknown parametrization.$k=`$v` (known: $(join(allowed, ", ")))")
            end
        end
        if _has(mdl, :stability)
            v = String(_get(mdl, :stability))
            v in _KNOWN_STABILITY || push!(errs,
                "unknown model.stability=`$v` (known: $(join(_KNOWN_STABILITY, ", ")))")
        end
        if _has(mdl, :trend_order)
            t = _get(mdl, :trend_order)
            (t == 0 || t == 1 || t == 2) || push!(errs,
                "model.trend_order must be 0, 1, or 2; got $t")
        end
        if _has(mdl, :phot_trend_order)
            t = _get(mdl, :phot_trend_order)
            (t == 0 || t == 1 || t == 2) || push!(errs,
                "model.phot_trend_order must be 0, 1, or 2; got $t")
        end
    end

    # ---- priors ----
    if _has(cfg, :priors)
        for (name, spec) in _get(cfg, :priors)
            _has(spec, :type) || (push!(errs, "priors[`$name`] missing `type`"); continue)
            ptype = String(_get(spec, :type))
            ptype in _KNOWN_PRIORS || push!(errs,
                "priors[`$name`].type=`$ptype` unknown (known: $(join(_KNOWN_PRIORS, ", ")))")
        end
    end

    # ---- model.external_priors ----
    if _has(cfg, :model) && _has(_get(cfg, :model), :external_priors)
        for (i, e) in enumerate(_get(_get(cfg, :model), :external_priors))
            if !_has(e, :quantity)
                push!(errs, "model.external_priors[$i] missing `quantity`"); continue
            end
            q = String(_get(e, :quantity))
            q in ("ecc", "rho_s") || push!(errs,
                "model.external_priors[$i].quantity=`$q` unknown (known: ecc, rho_s)")
            if !_has(e, :prior)
                push!(errs, "model.external_priors[$i] missing `prior`"); continue
            end
            pr = _get(e, :prior)
            if !_has(pr, :type)
                push!(errs, "model.external_priors[$i].prior missing `type`")
            else
                pt = String(_get(pr, :type))
                pt in _KNOWN_PRIORS || push!(errs,
                    "model.external_priors[$i].prior.type=`$pt` unknown " *
                    "(known: $(join(_KNOWN_PRIORS, ", ")))")
            end
        end
    end

    # ---- noise_models ----
    if _has(cfg, :noise_models)
        for (i, nm) in enumerate(_get(cfg, :noise_models))
            _has(nm, :kind) || (push!(errs,
                "noise_models[$i] missing `kind`"); continue)
            k = String(_get(nm, :kind))
            k in _KNOWN_NOISE_KINDS || push!(errs,
                "noise_models[$i].kind=`$k` unknown (known: $(join(_KNOWN_NOISE_KINDS, ", ")))")
        end
    end

    # ---- noise_menu ----
    if _has(cfg, :noise_menu)
        _has(cfg, :noise_models) && !isempty(_get(cfg, :noise_models)) && push!(errs,
            "`noise_menu` and `noise_models` are mutually exclusive (the menu builds the model list)")
        nm = _get(cfg, :noise_menu)
        if _has(nm, :indicators)
            inds = _get(nm, :indicators)
            inds isa AbstractVector || push!(errs,
                "noise_menu.indicators must be an array of channel names")
        end
        if _has(cfg, :transdim)
            td = _get(cfg, :transdim)
            (isempty(_get(td, :toggleable; default = [])) &&
             isempty(_get(td, :noise_exclusion_groups; default = []))) || push!(errs,
                "transdim.toggleable/noise_exclusion_groups cannot be set with `noise_menu` (the menu defines both)")
        end
    end

    # ---- sampler ----
    if _has(cfg, :sampler)
        smp = _get(cfg, :sampler)
        if !_has(smp, :name)
            push!(errs, "missing `sampler.name`")
        else
            name = String(_get(smp, :name))
            name in _KNOWN_SAMPLERS || push!(errs,
                "unknown sampler.name=`$name` (known: $(join(_KNOWN_SAMPLERS, ", ")))")
            if name in _SAMPLERS_REQUIRE_TD && !_has(cfg, :transdim)
                push!(errs, "sampler `$name` requires a top-level `transdim` block")
            end
        end
    end

    # ---- transdim ----
    if _has(cfg, :transdim)
        td = _get(cfg, :transdim)
        _has(td, :max_kplanet) || push!(errs, "missing `transdim.max_kplanet`")
        if _has(td, :birth_strategies)
            for s in _get(td, :birth_strategies)
                String(s) in _KNOWN_BIRTH_STRATEGIES || push!(errs,
                    "unknown transdim.birth_strategy=`$s` " *
                    "(known: $(join(_KNOWN_BIRTH_STRATEGIES, ", ")))")
            end
        end
        if _has(td, :birth_strategies) && _has(td, :birth_weights)
            ns = length(collect(_get(td, :birth_strategies)))
            nw = length(collect(_get(td, :birth_weights)))
            ns == nw || push!(errs,
                "transdim.birth_strategies length $ns ≠ birth_weights length $nw")
        end
    end

    # ---- output ----
    if _has(cfg, :output)
        out = _get(cfg, :output)
        if _has(out, :plots)
            for p in _get(out, :plots)
                String(p) in _KNOWN_PLOTS || push!(errs,
                    "unknown output.plot=`$p` (known: $(join(_KNOWN_PLOTS, ", ")))")
            end
        end
    end

    if !isempty(errs)
        msg = "Nereus.run_job config validation failed with " *
              "$(length(errs)) error(s):\n  - " * join(errs, "\n  - ")
        throw(ArgumentError(msg))
    end
    return nothing
end

# =====================================================================
# Data builder
# =====================================================================

function _build_data(data_cfg)
    inst_names_rv = String[]
    inst_names_pm = String[]
    isempty(data_cfg) && return Data(), inst_names_rv, inst_names_pm

    kw = Dict{Symbol, Any}()

    if _has(data_cfg, :rv)
        rv_block = _get(data_cfg, :rv)
        t, v, σ, inst_idx, names, indicators, indicator_errs, comp = _parse_rv_block(rv_block)
        kw[:t_rv]     = t
        kw[:rv]       = v
        kw[:rv_err]   = σ
        kw[:rv_inst]  = inst_idx
        comp === nothing || (kw[:rv_comp] = comp)
        indicators === nothing || (kw[:indicators] = indicators)
        indicator_errs === nothing || (kw[:indicator_errs] = indicator_errs)
        inst_names_rv = names
    end

    # Photometry: list of per-instrument LCs.
    if _has(data_cfg, :transit_photometry)
        lcs = _get(data_cfg, :transit_photometry)
        if !isempty(lcs)
            t_all, flux_all, err_all, inst_idx, names, exp_times =
                _parse_photometry_blocks(lcs)
            kw[:t_phot]    = t_all
            kw[:flux]      = flux_all
            kw[:flux_err]  = err_all
            kw[:phot_inst] = inst_idx
            kw[:exposure_times] = exp_times
            inst_names_pm = names
        end
    end

    # IAD: HIP-id auto-fetch (recommended) or inline values
    if _has(data_cfg, :iad)
        iad_block = _get(data_cfg, :iad)
        kw[:iad] = _parse_iad_block(iad_block)
    end

    # GOST: ra/dec auto-fetch or inline values
    if _has(data_cfg, :gost)
        gost_block = _get(data_cfg, :gost)
        kw[:gost] = _parse_gost_block(gost_block)
    end

    if _has(data_cfg, :hgca)
        kw[:hgca] = _parse_hgca_block(_get(data_cfg, :hgca))
    end
    if _has(data_cfg, :relastrom)
        kw[:relastrom] = _parse_relastrom_block(_get(data_cfg, :relastrom))
    end
    if _has(data_cfg, :gaia_dr3)
        kw[:gaia_dr3] = _parse_gaia_dr3_block(_get(data_cfg, :gaia_dr3))
    end

    data = Data(; kw...)
    return data, inst_names_rv, inst_names_pm
end

# ---- RV block ---------------------------------------------------------
function _parse_rv_block(block)
    if _has(block, :values)
        return _parse_rv_values(_get(block, :values))
    elseif _has(block, :csv)
        return _parse_rv_csv(block)
    else
        error("rv block requires either `csv` or `values`")
    end
end

# Map a stellar-component tag to 1 (primary A) or 2 (secondary B) for SB2
# double-lined fits. Accepts numeric 1/2 or the strings 1/A/primary/pri and
# 2/B/secondary/sec (case-insensitive). NOTE: the reserved word is
# "component"/"star", NOT "channel" — "channel" already means rv/phot in
# noise_models.
function _rv_component_code(x)
    if x isa Real
        c = round(Int, x)
        c in (1, 2) && return c
        throw(ArgumentError("RV component must be 1 or 2; got $x"))
    end
    s = lowercase(strip(String(x)))
    s in ("1", "a", "primary", "pri") && return 1
    s in ("2", "b", "secondary", "sec") && return 2
    throw(ArgumentError("unrecognized RV component `$x` " *
                        "(use 1/A/primary or 2/B/secondary)"))
end

function _parse_rv_values(vals)
    t   = Float64.(_get(vals, :bjd; required = true))
    v   = Float64.(_get(vals, :rv; required = true))
    σ   = Float64.(_get(vals, :rv_err; required = true))
    inst_str = String.(_get(vals, :instrument; required = true))
    inst_names = sort!(unique(inst_str))
    inst_map   = Dict(name => i for (i, name) in enumerate(inst_names))
    inst_idx   = Int[inst_map[s] for s in inst_str]

    # Any key beyond the reserved RV columns is an activity indicator. A `<name>_err`
    # key carrying a matching `<name>` value is that indicator's 1σ uncertainty
    # (ActivityGP/indicator GPs require it); a lone `<name>_err` is itself a value.
    # `component`/`star` is ALSO reserved → parsed into rv_comp (SB2), not an indicator.
    extra = Dict{String, Vector{Float64}}()
    comp = nothing
    for k in keys(vals)
        ks = String(k)
        ks in ("bjd", "rv", "rv_err", "instrument") && continue
        if ks in ("component", "star")
            comp = Int[_rv_component_code(x) for x in _get(vals, k)]
            continue
        end
        extra[ks] = Float64.(_get(vals, k))
    end
    indicators, indicator_errs = _split_indicator_errs(extra)
    return t, v, σ, inst_idx, inst_names, indicators, indicator_errs, comp
end

# Split a flat `name => column` dict into (indicators, indicator_errs) by the
# `<name>_err` naming convention. A `<name>_err` column is an error only when its
# base `<name>` is also present; otherwise it is treated as an indicator value.
function _split_indicator_errs(extra::Dict{String, Vector{Float64}})
    isempty(extra) && return nothing, nothing
    indicators = Dict{String, Vector{Float64}}()
    indicator_errs = Dict{String, Vector{Float64}}()
    for (k, arr) in extra
        if endswith(k, "_err") && haskey(extra, k[1:end-4])
            indicator_errs[k[1:end-4]] = arr
        else
            indicators[k] = arr
        end
    end
    return (isempty(indicators) ? nothing : indicators),
           (isempty(indicator_errs) ? nothing : indicator_errs)
end

function _parse_rv_csv(block)
    path = String(_get(block, :csv; required = true))
    time_col = String(_get(block, :time_col; default = "bjd"))
    rv_col   = String(_get(block, :rv_col;   default = "rv"))
    err_col  = String(_get(block, :err_col;  default = "rv_err"))
    inst_col = String(_get(block, :inst_col; default = "instrument"))
    comp_col = String(_get(block, :comp_col; default = "component"))
    ind_cols = String.(_get(block, :indicator_cols; default = String[]))

    raw, headers = readdlm(path, ',', Any, '\n'; header = true)
    header = vec(headers)
    col(name) = findfirst(==(name), header)

    # Blank / "NaN" / "null" cells are normal in real activity-indicator
    # tables (a night with no BIS, a pipeline that did not report log R'HK).
    # `Float64(x)` on those raised a bare MethodError from deep inside the
    # reader. Indicators tolerate NaN -- Nereus masks non-finite entries per
    # channel -- so parse them to NaN; the core columns (time, rv, error) must
    # not, and say which cell is bad.
    _MISSING = ("", "nan", "NaN", "NA", "na", "null", "NULL", "-")
    function _num(x, name, i; strict::Bool)
        x isa Real && return Float64(x)
        s = strip(string(x))
        if s in _MISSING || (v = tryparse(Float64, s)) === nothing
            strict && error("Column `$name` row $i in $path is not a number: " *
                            "`$(s)`. Time, RV and error columns may not be blank.")
            return NaN
        end
        return v
    end
    function fcol(name; strict::Bool = true)
        idx = col(name)
        idx === nothing && error("Column `$name` not in $path; have $header")
        return Float64[_num(raw[i, idx], name, i; strict = strict)
                       for i in axes(raw, 1)]
    end
    function scol(name)
        idx = col(name)
        idx === nothing && error("Column `$name` not in $path; have $header")
        return String[String(x) for x in raw[:, idx]]
    end

    t = fcol(time_col); v = fcol(rv_col); σ = fcol(err_col)
    inst_str = scol(inst_col)
    inst_names = sort!(unique(inst_str))
    inst_map = Dict(n => i for (i, n) in enumerate(inst_names))
    inst_idx = Int[inst_map[s] for s in inst_str]

    # Indicator values from `indicator_cols`; per-indicator 1σ from a `<col>_err`
    # column when the CSV provides one (required by ActivityGP / indicator GPs).
    indicators = isempty(ind_cols) ? nothing :
                  Dict{String, Vector{Float64}}(c => fcol(c; strict = false)
                                                for c in ind_cols)
    indicator_errs = nothing
    for c in ind_cols
        col("$(c)_err") === nothing && continue
        indicator_errs === nothing && (indicator_errs = Dict{String, Vector{Float64}}())
        indicator_errs[c] = fcol("$(c)_err"; strict = false)
    end
    # Optional SB2 component tag: `component` (or `star`) column → rv_comp.
    comp = nothing
    for cn in (comp_col, "star")
        if col(cn) !== nothing
            comp = Int[_rv_component_code(x) for x in raw[:, col(cn)]]
            break
        end
    end
    return t, v, σ, inst_idx, inst_names, indicators, indicator_errs, comp
end

# ---- Photometry blocks ------------------------------------------------
function _parse_photometry_blocks(lcs)
    t_all = Float64[]; flux_all = Float64[]; err_all = Float64[]
    inst_str_all = String[]
    exp_times    = Float64[]
    for lc in lcs
        if _has(lc, :values)
            vals = _get(lc, :values)
            t = Float64.(_get(vals, :bjd; required = true))
            f = Float64.(_get(vals, :flux; required = true))
            e = Float64.(_get(vals, :flux_err; required = true))
        else
            path = String(_get(lc, :csv; required = true))
            tcol = String(_get(lc, :time_col; default = "bjd"))
            fcol = String(_get(lc, :flux_col; default = "flux"))
            ecol = String(_get(lc, :err_col;  default = "flux_err"))
            raw, headers = readdlm(path, ',', Any, '\n'; header = true)
            header = vec(headers)
            geti(name) = findfirst(==(name), header)
            t = Float64[Float64(x) for x in raw[:, geti(tcol)]]
            f = Float64[Float64(x) for x in raw[:, geti(fcol)]]
            e = Float64[Float64(x) for x in raw[:, geti(ecol)]]
        end
        inst = String(_get(lc, :instrument; default = "TESS"))
        # exposure_time is per-LC in SECONDS (default 120 = TESS 2-min); store it
        # per-cadence in DAYS to match t_phot, for finite-exposure supersampling.
        exp_d = Float64(_get(lc, :exposure_time; default = 120.0)) / 86_400.0
        append!(t_all, t); append!(flux_all, f); append!(err_all, e)
        for _ in 1:length(t); push!(inst_str_all, inst); push!(exp_times, exp_d); end
    end
    inst_names = sort!(unique(inst_str_all))
    inst_map = Dict(n => i for (i, n) in enumerate(inst_names))
    inst_idx = Int[inst_map[s] for s in inst_str_all]
    return t_all, flux_all, err_all, inst_idx, inst_names, exp_times
end

# ---- IAD / GOST / HGCA / etc -----------------------------------------
function _parse_iad_block(block)
    if _has(block, :hip_id)
        hip = Int(_get(block, :hip_id))
        return fetch_hip_iad(hip; verbose = false)
    elseif _has(block, :values)
        vals = _get(block, :values)
        return IADData(
            t              = Float64.(_get(vals, :t; required = true)),
            abscissa       = Float64.(_get(vals, :abscissa; required = true)),
            abscissa_err   = Float64.(_get(vals, :abscissa_err; required = true)),
            psi            = Float64.(_get(vals, :psi; required = true)),
            parallax_factor = Float64.(_get(vals, :parallax_factor; required = true)),
            pm_factor      = Float64.(_get(vals, :pm_factor; required = true)),
        )
    else
        error("iad block needs `hip_id` (auto-fetch) or `values`")
    end
end

function _parse_gost_block(block)
    if _has(block, :values)
        vals = _get(block, :values)
        return GOSTData(
            t                = Float64.(_get(vals, :t; required = true)),
            psi              = Float64.(_get(vals, :psi; required = true)),
            parallax_factor  = Float64.(_get(vals, :parallax_factor; required = true)),
        )
    elseif _has(block, :ra_deg) && _has(block, :dec_deg)
        ra  = Float64(_get(block, :ra_deg))
        dec = Float64(_get(block, :dec_deg))
        from = String(_get(block, :from; default = "2014-07-26T00:00:00"))
        to   = String(_get(block, :to;   default = "2025-01-15T00:00:00"))
        return fetch_gost(ra, dec; from = from, to = to, verbose = false)
    else
        error("gost block needs `{ra_deg, dec_deg}` (auto-fetch) or `values`")
    end
end

function _parse_hgca_block(block)
    # HGCA from CSV file (Brandt 2018 catalog format) or inline values.
    if _has(block, :csv)
        return load_hgca(String(_get(block, :csv)); hip_id = _get(block, :hip_id))
    elseif _has(block, :hip_id)
        # Auto-fetch the Brandt eDR3 HGCA row for this Hipparcos source.
        return fetch_hgca(Int(_get(block, :hip_id)))
    elseif _has(block, :values)
        # Inline three-epoch HGCA arrays. HGCAData stores epochs/pmra/pmdec as
        # 3-tuples plus per-epoch 2×2 covariance matrices and the parallax —
        # build it via the positional constructor (the keyword ctor instead
        # takes σ_pmra/σ_pmdec, not cov_ep).
        vals = _get(block, :values)
        ep    = Float64.(_get(vals, :t;      required = true))
        pmra  = Float64.(_get(vals, :pmra;   required = true))
        pmdec = Float64.(_get(vals, :pmdec;  required = true))
        # JSON gives nested arrays (rows), not a Matrix — stack explicitly.
        _json_mat(c) = permutedims(reduce(hcat, [Float64.(collect(r)) for r in c]))
        cov   = [_json_mat(c) for c in _get(vals, :cov_ep; required = true)]
        (length(ep) == 3 && length(pmra) == 3 && length(pmdec) == 3 && length(cov) == 3) ||
            error("hgca inline values need exactly 3 epochs (Hipparcos, HG, Gaia)")
        return HGCAData((ep[1], ep[2], ep[3]),
                        (pmra[1], pmra[2], pmra[3]),
                        (pmdec[1], pmdec[2], pmdec[3]),
                        (cov[1], cov[2], cov[3]),
                        Float64(_get(vals, :plx;     default = NaN)),
                        Float64(_get(vals, :plx_err; default = NaN)),
                        Int(_get(vals, :hip_id;      default = 0)))
    else
        error("hgca block needs `csv` or `values`")
    end
end

function _parse_relastrom_block(block)
    if _has(block, :csv)
        return load_relastrom(String(_get(block, :csv)))
    elseif _has(block, :values)
        vals = _get(block, :values)
        return RelAstromData(;
            t           = Float64.(_get(vals, :t;       required = true)),
            ra_off      = Float64.(_get(vals, :ra_off;  required = true)),
            dec_off     = Float64.(_get(vals, :dec_off; required = true)),
            ra_err      = Float64.(_get(vals, :ra_err;  required = true)),
            dec_err     = Float64.(_get(vals, :dec_err; required = true)),
            planet_idx  = Int.(_get(vals, :planet_idx; default = ones(Int,
                                                                       length(_get(vals, :t))))),
        )
    else
        error("relastrom block needs `csv` or `values`")
    end
end

function _parse_gaia_dr3_block(block)
    return GaiaDR3Data(
        params = Float64.(_get(block, :params; required = true)),
        cov    = Matrix{Float64}(_get(block, :cov; required = true)),
        t_ref  = Float64(_get(block, :t_ref; required = true)),
    )
end

# =====================================================================
# Star + Model + Priors + Noise
# =====================================================================

function _build_star(s)
    return (;
        M_s   = _get(s, :M_s;   default = nothing),
        R_s   = _get(s, :R_s;   default = nothing),
        T_eff = _get(s, :T_eff; default = nothing),
        J_mag = _get(s, :J_mag; default = nothing),
        K_mag = _get(s, :K_mag; default = nothing),
        Ab    = Float64(_get(s, :Ab; default = 0.3)),
    )
end

const _MASS_MODE = Dict(
    "K_driven" => :K_driven, "M_sec_driven" => :M_sec_driven,
    "a_driven" => :a_driven,
)
const _TIME_ANCHOR = Dict("Tp" => :Tp, "Tc" => :Tc, "Mo" => :Mo)
const _EW_PARAM   = Dict("sesinw" => :sesinw, "ew" => :ew)
const _GEOM_PARAM = Dict("b_rr" => :b_rr, "b_r" => :b_r)
const _STAB       = Dict("none" => :none, "amd" => :amd, "gladman" => :gladman)
const _PLANET_MODES = Dict(
    "RV_ONLY" => RV_ONLY, "PM_ONLY" => PM_ONLY,
    "RVPM" => RVPM, "RVAS" => RVAS, "RVPMAS" => RVPMAS,
    "RVPM_RM" => RVPM_RM, "RVPMAS_RM" => RVPMAS_RM,
    "RVPM_TTV" => RVPM_TTV, "RVPMAS_TTV" => RVPMAS_TTV,
    "RVPM_RM_TTV" => RVPM_RM_TTV, "PM_TTV" => PM_TTV,
    "RVPM_TTV_NB" => RVPM_TTV_NB, "RVPMAS_TTV_NB" => RVPMAS_TTV_NB,
    "PM_TTV_NB" => PM_TTV_NB,
    "RVPM_RM_R" => RVPM_RM_R, "RVPMAS_RM_R" => RVPMAS_RM_R,
    "RVPM_RM_A" => RVPM_RM_A, "RVPMAS_RM_A" => RVPMAS_RM_A,
    # Gravity-darkened transits: the only route to i_star, hence to psi, when
    # no rotation period is available. RVPM_RM_GD fits RM/tomography and the
    # darkening asymmetry against the SAME lambda.
    "RVPM_GD" => RVPM_GD, "RVPM_RM_GD" => RVPM_RM_GD, "PM_GD" => PM_GD,
    "BINARY" => BINARY, "BINARY_RV" => BINARY_RV,   # SB2 double-lined binary
)

function _build_model(cfg, data, star, inst_names_rv, inst_names_pm)
    mdl = _get(cfg, :model; required = true)
    max_k = Int(_get(mdl, :max_kplanet; default = 1))
    modes_str = String.(_get(mdl, :planet_modes;
                              default = fill("RV_ONLY", max_k)))
    pm = [_PLANET_MODES[m] for m in modes_str]

    par = _get(mdl, :parametrization; default = Dict())
    parametrization = ParametrizationConfig(
        mass = _MASS_MODE[String(_get(par, :mass; default = "K_driven"))],
        time = _TIME_ANCHOR[String(_get(par, :time; default = "Tp"))],
        ew   = _EW_PARAM[String(_get(par, :ew; default = "sesinw"))],
        geom = _GEOM_PARAM[String(_get(par, :geom; default = "b_rr"))],
        marginalize_gamma = Bool(_get(par, :marginalize_gamma; default = false)),
        # Sample the stellar density ρ⋆ (the quantity transits actually
        # constrain) instead of deriving a/R★ from M_s×R_s. The whole
        # model/likelihood/prior/plotting stack already dispatches on this
        # field; this read is what lets ExoAutomata turn it on
        # (model.parametrization.use_rho_s). Default false = exact
        # backward compatibility. Enabling it auto-adds the default ρ⋆
        # prior (default_priors.jl); an informative override arrives via
        # ExternalPrior(:rho_s, ...).
        use_rho_s = Bool(_get(par, :use_rho_s; default = false)),
    )

    ic = InstrumentConfig(rv = inst_names_rv, pm = inst_names_pm)

    priors        = _build_priors(_get(cfg, :priors; default = Dict()))
    noise_models  = _build_noise_models(_get(cfg, :noise_models; default = []),
                                          ic)

    # ---- noise_menu: server-built default menu with USER-SELECTED indicators.
    # The exoautomata-facing alternative to hand-listing `noise_models`: the
    # backend builds `default_noise_menu(data; indicators=...)` and wires the
    # resulting models + toggleable set + exclusion groups into the trans-dim
    # config (see `_build_transdim(...; menu)`). `indicators` selects WHICH
    # activity channels the menu's AD / ActivityGP / IndicatorFloor use —
    # default: every channel present in the RV data. Not every indicator
    # correlates with the RVs on every target, and a non-correlating channel is
    # pure leak surface (a near-commensurate indicator can absorb planet
    # amplitude through its regression coefficient), so the frontend should
    # expose this as a per-channel multi-select. Menu built HERE (not at
    # sampler dispatch) so the exclusion groups hold the SAME NoiseModel
    # objects as params.config.noise_models — toggleable membership is matched
    # by identity downstream (Vector-field structs fall back to ===).
    menu = nothing
    menu_cfg = _get(cfg, :noise_menu; default = nothing)
    if menu_cfg !== nothing
        isempty(noise_models) || error(
            "`noise_menu` and `noise_models` are mutually exclusive: the menu " *
            "BUILDS the model list. Remove one of the two blocks.")
        provided = sort!(collect(keys(data.indicators)))
        sel = String.(collect(_get(menu_cfg, :indicators; default = provided)))
        bad = setdiff(sel, provided)
        isempty(bad) || error(
            "noise_menu.indicators $(bad) not present in the RV data " *
            "(provided channels: $(provided))")
        mkw = Dict{Symbol, Any}()
        for k in (:include_matern, :include_studentt, :include_harmonic,
                  :include_ad_ffprime, :include_nightly, :include_activity_gp)
            v = _get(menu_cfg, k; default = nothing)
            v === nothing || (mkw[k] = Bool(v))
        end
        menu = default_noise_menu(data; indicators = sel, mkw...)
        noise_models = menu.noise_models
    end

    # Params accepts only the model-defining kwargs; T_eff / mags / Ab
    # are stored in `star` (a NamedTuple) and consumed later by
    # `compute_derived` from the chains, not by Params itself.
    p_kw = Dict{Symbol, Any}(
        :max_kplanet     => max_k,
        :planet_modes    => pm,
        :instruments     => ic,
        :data            => data,
        :parametrization => parametrization,
        :trend_order     => Int(_get(mdl, :trend_order; default = 0)),
        :phot_trend_order => Int(_get(mdl, :phot_trend_order; default = 0)),
        :stability       => _STAB[String(_get(mdl, :stability;
                                                default = "none"))],
        :noise_models    => noise_models,
    )
    isempty(priors) || (p_kw[:priors] = priors)
    star.M_s !== nothing && (p_kw[:M_s] = Float64(star.M_s))
    star.R_s !== nothing && (p_kw[:R_s] = Float64(star.R_s))

    # Instrument parameter sharing, e.g.
    #   "sharing": {"ld": [["TESS", "PESTi"]], "sigma": [["HARPS", "HARPSa"]]}
    # `Params` has always accepted this, but it was unreachable from a job
    # config -- the runner never read it -- so JSON-driven fits could not share
    # a limb-darkening pair between two bands that observe the same star
    # through nearly the same filter.
    share_cfg = _get(mdl, :sharing; default = Dict())
    if !isempty(share_cfg)
        p_kw[:sharing] = Dict{Symbol, Vector{Vector{String}}}(
            Symbol(k) => [Vector{String}(String.(g)) for g in v]
            for (k, v) in share_cfg)
    end

    ttv_cfg = _get(mdl, :ttv_n_transits; default = Dict())
    if !isempty(ttv_cfg)
        p_kw[:ttv_n_transits] = Dict{Int, Int}(
            parse(Int, String(k)) => Int(v) for (k, v) in ttv_cfg)
    end
    ttv_backend = _get(mdl, :ttv_backend; default = nothing)
    if ttv_backend !== nothing
        p_kw[:ttv_backend] = Symbol(ttv_backend)
    end

    # External priors -- priors on DERIVED quantities (:ecc, :rho_s), which are
    # not sampled slots and so cannot be expressed through `priors`. Params has
    # always accepted these; the runner never read them, which meant a job
    # config could not state the eccentricity prior a sparse activity-
    # contaminated RV fit needs (uniform sesinw/secosw gives p(e) ~ e and rails
    # e -> 1), nor an informative rho_s.
    ext_cfg = _get(mdl, :external_priors; default = Any[])
    if !isempty(ext_cfg)
        p_kw[:external_priors] = [_build_external_prior(e) for e in ext_cfg]
    end

    menu === nothing || (p_kw[:transdim_noise] = true)

    params = Params(; p_kw...)
    target = NereusTarget(params, data; unconstrained = false)
    return params, target, menu
end

# ---- Priors -----------------------------------------------------------
const _PRIOR_TYPES = Dict(
    "UniformPrior"     => UniformPrior,
    "LogUniformPrior"  => LogUniformPrior,
    "ModJeffreysPrior" => ModJeffreysPrior,
    "NormalPrior"      => NormalPrior,
    "FixedPrior"       => FixedPrior,
    "SinePrior"        => SinePrior,
    "BetaPrior"        => BetaPrior,
)

function _build_priors(priors_cfg)
    out = Dict{String, PriorSpec}()
    for (name, spec) in priors_cfg
        T = _PRIOR_TYPES[String(_get(spec, :type; required = true))]
        args = collect(_get(spec, :args; default = Any[]))
        out[String(name)] = T(args...)
    end
    return out
end

function _build_external_prior(spec)
    q = Symbol(_get(spec, :quantity; required = true))
    pr = _get(spec, :prior; required = true)
    T = _PRIOR_TYPES[String(_get(pr, :type; required = true))]
    args = collect(_get(pr, :args; default = Any[]))
    # :ecc is per-planet, :rho_s is global; default from the quantity so a
    # config cannot silently pair the wrong scope with the right prior.
    per_planet = Bool(_get(spec, :per_planet; default = (q === :ecc)))
    return ExternalPrior(q, T(args...), per_planet)
end

# ---- Noise models -----------------------------------------------------
const _NOISE_TYPES = Dict(
    "CeleriteRotation"       => CeleriteRotation,
    "CeleriteSHO"            => CeleriteSHO,
    "CeleriteRotationFM17"   => CeleriteRotationFM17,
    "ActivityDecorrelation"  => ActivityDecorrelation,
    "ARModel"                => ARModel,
    "MAModel"                => MAModel,
    "ActivityJitter"         => ActivityJitter,
    "ActivityGP"             => ActivityGP,
    "ErrorScale"             => ErrorScale,
    "NightlyOffset"          => NightlyOffset,
    "HarmonicBlock"          => HarmonicBlock,
    "StudentT"               => StudentT,
    "MaternGP"               => MaternGP,
)

"Does any constructor method of `T` accept keyword `kw` (or slurp `kwargs...`)?"
function _ctor_accepts(@nospecialize(T), kw::Symbol)
    for m in methods(T)
        for d in Base.kwarg_decl(m)
            d === kw && return true
            endswith(String(d), "...") && return true   # `; kwargs...` slurp
        end
    end
    return false
end

function _build_noise_models(nm_cfg, ic)
    out = NoiseModel[]
    for nm in nm_cfg
        kind = String(_get(nm, :kind; required = true))
        T = get(_NOISE_TYPES, kind, nothing)
        T === nothing && error("Unknown noise_model kind: $kind. " *
                                "Known: $(keys(_NOISE_TYPES))")
        channel = Symbol(_get(nm, :channel; default = "rv"))
        instruments = Vector{String}(String.(_get(nm, :instruments;
                                                    default = String[])))
        kwargs = _get(nm, :kwargs; default = Dict())
        # Symbolize keys AND narrow JSON array values: JSON3 parses a JSON
        # array as a `JSON3.Array`, which does not satisfy concretely-typed
        # noise-model fields like `indicators::Vector{String}` (TypeError on
        # construction). `collect` narrows it to a plain `Vector` (eltype
        # preserved) so the constructor's field convert takes over. Covers any
        # vector-valued noise-model kwarg, not just `indicators`.
        kw_sym = Dict{Symbol, Any}(
            Symbol(k) => (v isa AbstractVector ? collect(v) : v)
            for (k, v) in kwargs)
        # ActivityGP `channels` is a Vector{Symbol}; JOB_CONFIG supplies them as
        # strings (e.g. ["bis","fwhm"]) — symbolize so the constructor's typed
        # field accepts them instead of a TypeError.
        haskey(kw_sym, :channels) &&
            (kw_sym[:channels] = Symbol.(kw_sym[:channels]))
        # ActivityGP `latent_kernel`/`backend` are Symbols; JSON supplies them
        # as strings (e.g. "mep", "auto") — symbolize so the typed keyword
        # constructor accepts them instead of a MethodError.
        for k in (:latent_kernel, :backend)
            haskey(kw_sym, k) && (kw_sym[k] = Symbol(kw_sym[k]))
        end
        # Inject the top-level `channel`/`instruments` ONLY into constructors
        # that accept them. The noise models are not uniform: Celerite GPs take
        # both; AR/MAModel take `channel` only; ActivityGP takes `channels`;
        # ActivityDecorrelation/ActivityJitter (MeanModifiers) take neither.
        # Splatting both unconditionally MethodErrors on the latter (only hit
        # now that activity models are driven via JOB_CONFIG).
        _ctor_accepts(T, :channel) && !haskey(kw_sym, :channel) &&
            (kw_sym[:channel] = channel)
        _ctor_accepts(T, :instruments) && !haskey(kw_sym, :instruments) &&
            (kw_sym[:instruments] = instruments)
        push!(out, T(; kw_sym...))
    end
    # Cross-model consistency check (per-instrument exclusivity, channel
    # validity, AR/MA conflict, etc.). Errors here surface as `failed`
    # in summary.json with the explicit complaint instead of a deeper
    # crash inside the likelihood.
    validate_noise_models(out)
    return out
end

# =====================================================================
# Sampler dispatcher
# =====================================================================

const _SAMPLERS_NEED_TD = Set(["transdim_ptemcee", "rjmcmc", "moms",
                                "moms_ns", "pt_warm", "pt"])

# Wall-clock watchdog around `_dispatch_sampler`. The sampler runs on a
# dedicated Julia task; the main task polls every second until the task
# finishes or the budget elapses. On timeout we mark the result as
# `JobTimeoutError` and surface it through run_job's catch block.
#
# Note: Julia has no preemption — the spawned task keeps running until it
# yields. For samplers that respect `show_progress=true` (most do) the
# tight inner loops yield often enough that the task tears down quickly
# after the timer fires. The watchdog returns control to the parent and
# lets the orphan task finish on its own; this is the same compromise
# Pigeons + emcee use.
struct JobTimeoutError <: Exception
    seconds::Float64
end
Base.showerror(io::IO, e::JobTimeoutError) =
    print(io, "Nereus.run_job: sampler exceeded `timeout_sec` = ",
           e.seconds, " s — aborted")

function _run_sampler_with_timeout(cfg, target, data, seed, timeout_sec; menu = nothing)
    if timeout_sec === nothing
        return _dispatch_sampler(cfg, target, data, seed; menu = menu)
    end
    # Under juliacall, the `Threads.@spawn` + `sleep`-poll watchdog
    # deadlocks. Python holds the GIL while Julia executes; the
    # spawned worker task can't be reliably scheduled on a Julia
    # worker thread (the GIL handshake stalls juliacall's task
    # interop), so `istaskdone(task)` stays false forever and the
    # main task loops `sleep(1.0)` indefinitely. NOI-106823 froze
    # for 23 minutes inside `juliacall.Main.__call__` with 17
    # parked-on-futex Julia worker threads and no progress — this
    # was the culprit even after the Pathfinder `ThreadedEx →
    # SequentialEx` and `BLAS.set_num_threads(1)` fixes.
    #
    # Workaround: when juliacall is detected, skip the watchdog and
    # run the sampler synchronously. Python-side timeout enforcement
    # (subprocess.kill / SIGTERM) is the right mechanism in that
    # environment anyway — it cleanly tears down the worker process
    # without relying on cooperative Julia task cancellation.
    if _is_under_juliacall()
        @warn "`timeout_sec=$timeout_sec` ignored: Nereus detected " *
              "juliacall and the `Threads.@spawn`-based watchdog " *
              "deadlocks under the Python GIL. Enforce timeout on the " *
              "Python side (subprocess.kill, signal.alarm, " *
              "concurrent.futures with timeout, etc.). Sampler will " *
              "run to completion."
        return _dispatch_sampler(cfg, target, data, seed; menu = menu)
    end
    budget = Float64(timeout_sec)
    budget > 0 || error("`timeout_sec` must be > 0 (got $timeout_sec)")
    t_start = time()
    task = Threads.@spawn _dispatch_sampler(cfg, target, data, seed; menu = menu)
    while !istaskdone(task)
        if time() - t_start > budget
            throw(JobTimeoutError(budget))
        end
        sleep(1.0)
    end
    return fetch(task)
end

# Fail fast with an actionable message when the config carries kwargs a
# sampler doesn't declare. This is deliberately strict (throws, never
# silently drops) — a dropped kwarg would mask both typos and genuine
# config/sampler mismatches behind a silently-wrong run. If the sampler
# slurps `kwargs...`, anything is allowed.
function _assert_supported_kwargs(f, kw::AbstractDict; name::AbstractString)
    supported = Set{Symbol}()
    for m in methods(f)
        for k in Base.kwarg_decl(m)
            endswith(String(k), "...") && return   # slurps → accepts anything
            push!(supported, k)
        end
    end
    bad = setdiff(keys(kw), supported)
    isempty(bad) && return
    throw(ArgumentError(
        "sampler \"$name\" does not accept kwarg(s) $(sort!(collect(bad))). " *
        "Valid kwargs: $(sort!(collect(supported))). " *
        "Check sampler.kwargs in the job config against this sampler."))
end

# Maps sampler.name → the function it dispatches to, for kwarg
# validation. `td` is passed separately by _dispatch_sampler, so the
# validator (which only inspects sampler.kwargs) tolerates it being a
# declared-but-absent kwarg.
const _SAMPLER_FNS = Dict{String,Function}(
    "ptemcee"          => sample_ptemcee,
    "transdim_ptemcee" => sample_transdim_ptemcee,
    "pt"               => sample_pt,
    "pt_warm"          => sample_pt_warm,
    "rjmcmc"           => sample_rjmcmc,
    "moms"             => sample_moms,
    "moms_ns"          => sample_moms_ns,
    "nested"           => sample_nested,
    "nested_ins"       => sample_nested_ins,
    "nested_dynamic"   => sample_nested_dynamic,
    "pa"               => sample_pa,
    "smc"              => sample_smc,
    "pt_whitening"     => sample_pt_whitening,
    "nuts"             => sample_nuts,
    "ofti"             => ofti_sample,
)

function _dispatch_sampler(cfg, target, data, seed::Int; menu = nothing)
    smp = _get(cfg, :sampler; required = true)
    name = String(_get(smp, :name; required = true))
    kw = Dict{Symbol, Any}(Symbol(k) => v
                            for (k, v) in _get(smp, :kwargs; default = Dict()))
    kw[:seed] = get(kw, :seed, seed)

    # JSON has no Symbol type, so every enum-style sampler kwarg
    # (bounds, proposal, mutation_kernel, within_model, ad_backend,
    # init_strategy, method, backend, …) arrives as a String. No sampler
    # declares a String-typed kwarg (verified across src/samplers), so
    # within sampler.kwargs a String is always meant to be a Julia
    # Symbol. Coerce uniformly here instead of per-branch — kills the
    # whole String→Symbol mismatch class for every sampler at once.
    for k in collect(keys(kw))
        kw[k] isa AbstractString && (kw[k] = Symbol(kw[k]))
    end

    # Reject kwargs the chosen sampler doesn't declare, with an
    # actionable error, for EVERY sampler — not just the nested family.
    # `td` is supplied separately (from the transdim block), so it's
    # excluded from the check. Functions that slurp `kwargs...` accept
    # anything and pass through untouched (see _assert_supported_kwargs).
    if haskey(_SAMPLER_FNS, name)
        _assert_supported_kwargs(_SAMPLER_FNS[name], kw; name = name)
    end

    td_block = _get(cfg, :transdim; default = nothing)
    td = td_block === nothing ? nothing : _build_transdim(td_block; menu = menu)

    if name == "ptemcee"
        return sample_ptemcee(target, data; kw...)
    elseif name == "transdim_ptemcee"
        td === nothing && error("transdim_ptemcee requires `transdim` block")
        return sample_transdim_ptemcee(target, data; td = td, kw...)
    elseif name == "pt" || name == "pt_warm"
        # Returns (chains, log_ev) tuple — wrap into a synthetic result.
        chains, log_z = name == "pt" ?
            sample_pt(target; kw...) :
            (sample_pt_warm(target; kw...))[1:2]
        return (chains = chains, log_evidence = log_z,
                 acceptance_within = Float64[], acceptance_swap = Float64[],
                 n_evals = 0)
    elseif name == "pt_hmc"
        # Hamiltonian parallel tempering (fixed-dim, differentiable). Returns
        # (chains, log_ev, report).
        chains, log_z, _ = sample_pt_hmc(target; kw...)
        return (chains = chains, log_evidence = log_z,
                 acceptance_within = Float64[], acceptance_swap = Float64[],
                 n_evals = 0)
    elseif name == "rjmcmc"
        td === nothing && error("rjmcmc requires `transdim` block")
        chains, n_evals = sample_rjmcmc(target, data; td = td, kw...)
        return (chains = chains, log_evidence = NaN,
                 acceptance_within = Float64[], acceptance_swap = Float64[],
                 n_evals = n_evals)
    elseif name == "moms"
        td === nothing && error("moms requires `transdim` block")
        chains, n_evals, _ = sample_moms(target, data; td = td, kw...)
        return (chains = chains, log_evidence = NaN,
                 acceptance_within = Float64[], acceptance_swap = Float64[],
                 n_evals = n_evals)
    elseif name == "nested"
        chains, log_z = sample_nested(target, data; kw...)
        return (chains = chains, log_evidence = log_z,
                 acceptance_within = Float64[], acceptance_swap = Float64[],
                 n_evals = 0)
    elseif name == "pa"
        res = sample_pa(target, data; kw...)
        return (chains = res.chains, log_evidence = res.log_evidence,
                 acceptance_within = Float64[res.acceptance],
                 acceptance_swap = Float64[], n_evals = res.n_evals)
    elseif name == "smc"
        # SMC delegates to PA (Hukushima & Iba 2003 population annealing).
        res = sample_smc(target, data; kw...)
        return (chains = res.chains, log_evidence = res.log_evidence,
                 acceptance_within = Float64[res.acceptance],
                 acceptance_swap = Float64[], n_evals = res.n_evals)
    elseif name == "pt_whitening"
        res = sample_pt_whitening(target, data; kw...)
        return (chains = res.chains, log_evidence = res.log_evidence,
                 acceptance_within = res.acceptance_within,
                 acceptance_swap   = res.acceptance_swap,
                 n_evals = res.n_evals)
    elseif name == "nested_ins"
        res = sample_nested_ins(target, data; kw...)
        return (chains = res.chains, log_evidence = res.log_z_ins,
                 acceptance_within = Float64[], acceptance_swap = Float64[],
                 n_evals = res.n_evals)
    elseif name == "nested_dynamic"
        res = sample_nested_dynamic(target, data; kw...)
        return (chains = res.chains, log_evidence = res.log_z,
                 acceptance_within = Float64[], acceptance_swap = Float64[],
                 n_evals = res.n_evals)
    elseif name == "moms_ns"
        td === nothing && error("moms_ns requires `transdim` block")
        # sample_moms_ns returns (chains, log_Z, strategy). It's a nested
        # sampler — surface its real evidence, don't discard it as NaN.
        chains, log_z, _ = sample_moms_ns(target, data; td = td, kw...)
        return (chains = chains, log_evidence = log_z,
                 acceptance_within = Float64[], acceptance_swap = Float64[],
                 n_evals = 0)
    elseif name == "nuts"
        chains = sample_nuts(target; kw...)
        return (chains = chains, log_evidence = NaN,
                 acceptance_within = Float64[], acceptance_swap = Float64[],
                 n_evals = 0)
    elseif name == "ofti"
        # OFTI rejection sampling — relative-astrometry-only orbits,
        # no MCMC chain quality metrics to report.
        chains = ofti_sample(target; kw...)
        return (chains = chains, log_evidence = NaN,
                 acceptance_within = Float64[], acceptance_swap = Float64[],
                 n_evals = 0)
    else
        error("Unknown sampler: $name. Supported: " *
              join(_KNOWN_SAMPLERS, ", "))
    end
end

const _BIRTH_TYPES = Dict(
    "PriorBirth"          => PriorBirth,
    "InformedBirth"       => InformedBirth,
    "JointInformedBirth"  => JointInformedBirth,
    "DonorBirth"          => DonorBirth,
    "MoMSBirth"           => MoMSBirth,
)

# Normalize a JSON-supplied toggleable list to Vector{NoiseModel}.
# TransDimConfig's constructor declares `toggleable::Vector{<:NoiseModel}`,
# which neither the `Any[]`/`NoiseModel[]` default NOR a JSON empty array
# (parsed as `JSON3.Array{Union{}}` / `Vector{Any}`) satisfies until it's
# concretely typed. Empty → typed empty (the planets-only common case);
# non-empty must already hold NoiseModels (JSON can't encode them, so a
# non-empty value here is a config error worth naming).
function _coerce_noise_models(v)
    isempty(v) && return NoiseModel[]
    all(x -> x isa NoiseModel, v) ||
        throw(ArgumentError("transdim.toggleable must be empty or hold " *
            "NoiseModel objects; JSON-specified toggleable models aren't " *
            "supported via run_job."))
    return Vector{NoiseModel}(collect(v))
end

function _coerce_exclusion_groups(v)
    isempty(v) && return Vector{NoiseModel}[]
    return [_coerce_noise_models(g) for g in v]
end

function _build_transdim(td_cfg; menu = nothing)
    bstr = String.(_get(td_cfg, :birth_strategies; default = ["PriorBirth"]))
    strategies = [_BIRTH_TYPES[s]() for s in bstr]
    weights    = Float64.(_get(td_cfg, :birth_weights; default = ones(length(strategies))))
    # With a `noise_menu` block, the toggleable set + exclusion groups come from
    # the SERVER-BUILT menu (same NoiseModel objects as params.config.noise_models
    # — identity matters downstream). JSON cannot express NoiseModel objects, so
    # a config that sets both is contradictory: hard error, not silent override.
    if menu !== nothing
        (isempty(_get(td_cfg, :toggleable; default = [])) &&
         isempty(_get(td_cfg, :noise_exclusion_groups; default = []))) || error(
            "`transdim.toggleable`/`transdim.noise_exclusion_groups` cannot be " *
            "set together with a `noise_menu` block — the menu defines both.")
    end
    return TransDimConfig(
        max_kplanet       = Int(_get(td_cfg, :max_kplanet; required = true)),
        birth_strategies  = strategies,
        birth_weights     = weights,
        transdim_fraction = Float64(_get(td_cfg, :transdim_fraction; default = 0.3)),
        planets           = Bool(_get(td_cfg, :planets; default = true)),
        # menu ⇒ noise selection is the point; default flips to true.
        noise             = Bool(_get(td_cfg, :noise;   default = menu !== nothing)),
        # Coerce the value (not just the default): a config that passes
        # `toggleable: []` delivers an empty JSON3.Array, which still
        # isn't a Vector{<:NoiseModel}.
        toggleable = menu !== nothing ? menu.toggleable : _coerce_noise_models(
            _get(td_cfg, :toggleable; default = NoiseModel[])),
        noise_exclusion_groups = menu !== nothing ? menu.exclusion_groups :
            _coerce_exclusion_groups(
                _get(td_cfg, :noise_exclusion_groups; default = Vector{NoiseModel}[])),
    )
end

# =====================================================================
# Plot dispatch
# =====================================================================

function _make_plots(cfg, chains, params, data, out_dir)
    out_cfg = _get(cfg, :output; default = Dict())
    plot_list = _get(out_cfg, :plots; default = String[])
    plot_kwargs_raw = _get(out_cfg, :plot_kwargs; default = Dict())
    plot_kwargs = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in plot_kwargs_raw)

    # Top-level `output.save_pdf` toggle: when true, every plot saves a
    # `.pdf` companion next to its `.png`. Default is PNG only — PDF is
    # opt-in via this single flag so users don't have to thread the
    # toggle through `output.plot_kwargs` per-plot. An explicit
    # `plot_kwargs.save_pdf` still wins (per-plot override).
    save_pdf_global = Bool(_get(out_cfg, :save_pdf; default = false))
    haskey(plot_kwargs, :save_pdf) || (plot_kwargs[:save_pdf] = save_pdf_global)

    # All figures live under `out_dir/plots/` (models, transdim, posteriors,
    # traces). The figures manifest is built by scanning this tree afterwards.
    plots_dir = joinpath(out_dir, "plots"); mkpath(plots_dir)
    nw_eff = _ensemble_n_walkers(cfg, params)   # per-walker trace de-interleave
    generated = String[]
    for plot_name in plot_list
        try
            fname = _dispatch_plot(String(plot_name), chains, params, data,
                                     plots_dir, plot_kwargs; n_walkers = nw_eff)
            fname !== nothing && push!(generated, fname)
        catch err
            @warn "Plot `$plot_name` failed" exception = (err, catch_backtrace())
        end
    end
    return generated
end

# Keep only the kwargs the SB2 plot functions accept (they take fewer than the
# generic RV plots — no show_keplerian/show_gp/decompose).
_sb2_kw(kw) = (; (k => v for (k, v) in pairs(kw)
                  if k in (:fmt, :save_pdf, :figsize, :bf_cutoff))...)

function _dispatch_plot(name, chains, params, data, out_dir, kw; n_walkers=nothing)
    if name == "rv_timeseries"
        n_rv(data) > 0 || return nothing
        # SB2 double-lined fit → component-colored timeseries with both the
        # primary (K_A + planet) and secondary (−K_B) model curves.
        if _is_sb2_data(data)
            plot_rv_sb2_timeseries(chains, params, data; output = out_dir,
                                    _sb2_kw(kw)...)
            return "models/rv_sb2_timeseries.png"
        end
        plot_rv_timeseries(chains, params, data; output = out_dir, kw...)
        return "models/rv_timeseries.png"
    elseif name == "rv_components"
        n_rv(data) > 0 || return nothing
        # For SB2 the component-split IS the timeseries view; reuse it.
        if _is_sb2_data(data)
            plot_rv_sb2_timeseries(chains, params, data; output = out_dir,
                                    _sb2_kw(kw)...)
            return "models/rv_sb2_timeseries.png"
        end
        plot_rv_components(chains, params, data; output = out_dir, kw...)
        return "models/rv_components.png"
    elseif name == "rv_phasefold"
        n_rv(data) > 0 || return nothing
        for k in 1:params.config.max_kplanet
            try
                plot_rv_phasefold(chains, params, data;
                                    planet = k, output = out_dir, kw...)
            catch err
                @warn "rv_phasefold K$k failed" exception = err
            end
        end
        return "models/rv_phased_K*.png"
    elseif name == "pm_timeseries"
        n_phot(data) > 0 || return nothing
        plot_pm_timeseries(chains, params, data; output = out_dir, kw...)
        return "models/pm_timeseries_*.png"
    elseif name == "pm_phasefold"
        n_phot(data) > 0 || return nothing
        for k in 1:params.config.max_kplanet
            try
                plot_pm_phasefold(chains, params, data;
                                    planet = k, output = out_dir, kw...)
            catch err
                @warn "pm_phasefold K$k failed" exception = err
            end
        end
        return "models/pm_phased_*_K*.png"
    elseif name == "rv_astrom_phasefold"
        (n_rv(data) > 0 && has_astrometry(data)) || return nothing
        for k in 1:params.config.max_kplanet
            try
                plot_rv_astrom_phasefold(chains, params, data;
                                            planet_idx = k, output = out_dir, kw...)
            catch
            end
        end
        return "models/rv_astrom_phasefold_K*.png"
    elseif name == "orbit_skyplane"
        data.relastrom === nothing && return nothing
        for k in 1:params.config.max_kplanet
            try
                plot_orbit_skyplane(chains, params, data;
                                      planet_idx = k, output = out_dir, kw...)
            catch
            end
        end
        return "models/orbit_skyplane_K*.png"
    elseif name == "relastrom_timeseries"
        data.relastrom === nothing && return nothing
        for k in 1:params.config.max_kplanet
            try
                plot_relastrom_timeseries(chains, params, data;
                                            planet_idx = k, output = out_dir, kw...)
            catch
            end
        end
        return "models/relastrom_timeseries_K*.png"
    elseif name == "relastrom_residuals"
        data.relastrom === nothing && return nothing
        for k in 1:params.config.max_kplanet
            try
                plot_relastrom_residuals(chains, params, data;
                                           planet_idx = k, output = out_dir, kw...)
            catch
            end
        end
        return "models/relastrom_residuals_K*.png"
    elseif name == "iad_residuals"
        data.iad === nothing && return nothing
        plot_iad_residuals(chains, params, data; output = out_dir, kw...)
        return "models/iad_residuals.png"
    elseif name == "hgca_pm_residuals"
        data.hgca === nothing && return nothing
        for k in 1:params.config.max_kplanet
            try
                plot_pm_residuals(chains, params, data;
                                    planet_idx = k, output = out_dir, kw...)
            catch
            end
        end
        return "models/hgca_pm_residuals_K*.png"
    elseif name == "g23h_residuals"
        data.g23h === nothing && return nothing
        plot_g23h_residuals(chains, params, data; output = out_dir, kw...)
        return "models/g23h_residuals.png"
    elseif name == "pm_anomaly"
        data.gost === nothing && return nothing
        for k in 1:params.config.max_kplanet
            try
                plot_pm_anomaly(chains, params, data;
                                  planet_idx = k, output = out_dir, kw...)
            catch
            end
        end
        return "models/pm_anomaly_K*.png"
    elseif name == "ttv_oc"
        n_phot(data) > 0 || return nothing
        # With ≥2 planets the perturber drives the N-body O−C envelope; with a
        # single transiting planet there is no perturber, so show the data-only
        # O−C (measured per-transit Tc vs the linear ephemeris) by passing
        # planet_b_k=0. An explicit plot_kwargs.planet_b_k still wins.
        ttv_kw = copy(kw)
        haskey(ttv_kw, :planet_b_k) ||
            (ttv_kw[:planet_b_k] = params.config.max_kplanet >= 2 ? 2 : 0)
        try
            plot_ttv_oc(chains, data, params;
                          filename = joinpath(out_dir, "models", "ttv_oc.png"),
                          ttv_kw...)
        catch err
            @warn "plot_ttv_oc failed" exception = err
            return nothing
        end
        return "models/ttv_oc.png"
    elseif name == "transit_overlay"
        n_phot(data) > 0 || return nothing
        try
            plot_transit_overlay_fit(chains, params, data; output = out_dir, kw...)
        catch err
            @warn "plot_transit_overlay_fit failed" exception = err
            return nothing
        end
        return "models/transit_overlay_K*.png"
    elseif name == "rm_anomaly"
        # Only for RM-enabled fits (a planet with an *_RM mode).
        any(has_any_rm, params.config.planet_modes) || return nothing
        try
            plot_rm(chains, params, data; output = out_dir, kw...)
        catch err
            @warn "plot_rm failed" exception = err
            return nothing
        end
        return "models/rm_anomaly_K*.png"
    elseif name == "corner"
        plot_corner(chains, params; output = out_dir, kw...)
        return "corner.png"
    elseif name == "trace"
        plot_trace(chains, params; output = out_dir, kw...)
        return "traces/*.png"
    elseif name == "histograms"
        plot_histograms(chains, params; output = out_dir, kw...)
        return "histograms/*.png"
    elseif name == "posteriors"
        plot_posteriors(chains, params; output = out_dir, kw...)
        return "posteriors/*.png"
    elseif name == "transdim_occupancy"
        :n_planets in Set(names(chains, :parameters)) || return nothing
        plot_transdim_occupancy(chains, params;
            output = joinpath(out_dir, "transdim", "occupancy.png"),
            save_pdf = get(kw, :save_pdf, false))
        return "transdim/occupancy.png"
    elseif name == "posteriors_raw"
        :lp in Set(names(chains, :parameters)) || return nothing
        plot_posteriors_raw(chains, params; output = out_dir,
                            save_pdf = get(kw, :save_pdf, false))
        return "posteriors/raw/*.png"
    elseif name == "posteriors_parameters"
        :lp in Set(names(chains, :parameters)) || return nothing
        plot_posteriors_parameters(chains, params; output = out_dir,
                            save_pdf = get(kw, :save_pdf, false))
        return "posteriors/parameters/*.png"
    elseif name == "posteriors_histograms"
        plot_posteriors_histograms(chains, params; output = out_dir,
                            save_pdf = get(kw, :save_pdf, false))
        return "posteriors/histograms/*.png"
    elseif name == "traces_grouped"
        plot_traces_grouped(chains, params; output = out_dir,
                            save_pdf = get(kw, :save_pdf, false), n_walkers = n_walkers)
        return "traces/*.png"
    elseif name == "activity_gp_latent"
        # No-op unless an ActivityGP is configured.
        any(nm -> nm isa ActivityGP, params.config.noise_models) || return nothing
        try
            plot_activity_gp_latent(chains, params, data;
                                       filename = joinpath(out_dir, "activity_gp_latent.png"),
                                       kw...)
        catch err
            @warn "plot_activity_gp_latent failed" exception = err
            return nothing
        end
        return "activity_gp_latent.png"
    elseif name == "activity_gp_decomposition"
        any(nm -> nm isa ActivityGP, params.config.noise_models) || return nothing
        try
            plot_activity_gp_decomposition(chains, params, data;
                                              filename = joinpath(out_dir, "activity_gp_decomposition.png"),
                                              kw...)
        catch err
            @warn "plot_activity_gp_decomposition failed" exception = err
            return nothing
        end
        return "activity_gp_decomposition.png"
    elseif name == "auto"
        # Pick the plot set automatically from what's in `data`. Each
        # individual plot still no-ops when its required data is absent.
        produced = String[]
        kinds = String["corner", "posteriors_raw", "posteriors_parameters",
                       "posteriors_histograms", "traces_grouped"]
        :n_planets in Set(names(chains, :parameters)) && push!(kinds, "transdim_occupancy")
        n_rv(data) > 0           && append!(kinds, ["rv_timeseries", "rv_phasefold"])
        # component decomposition adds over the timeseries only with multiple
        # model pieces — ≥2 planets or a smooth (GP/AGP) activity curve.
        (n_rv(data) > 0 && (params.config.max_kplanet ≥ 2 ||
            any(nm -> nm isa ActivityGP || nm isa CovarianceNoise,
                params.config.noise_models))) && push!(kinds, "rv_components")
        n_phot(data) > 0         && append!(kinds, ["pm_timeseries", "pm_phasefold"])
        data.relastrom !== nothing && append!(kinds, ["orbit_skyplane",
                                                       "relastrom_timeseries",
                                                       "relastrom_residuals"])
        data.iad      !== nothing && push!(kinds, "iad_residuals")
        data.hgca     !== nothing && push!(kinds, "hgca_pm_residuals")
        data.g23h     !== nothing && push!(kinds, "g23h_residuals")
        data.gost     !== nothing && push!(kinds, "pm_anomaly")
        (n_rv(data) > 0 && has_astrometry(data)) && push!(kinds, "rv_astrom_phasefold")
        n_phot(data) > 0 && append!(kinds, ["ttv_oc", "transit_overlay"])
        any(has_any_rm, params.config.planet_modes) && push!(kinds, "rm_anomaly")
        if any(nm -> nm isa ActivityGP, params.config.noise_models)
            push!(kinds, "activity_gp_latent")
            push!(kinds, "activity_gp_decomposition")
        end
        for k in unique(kinds)
            try
                f = _dispatch_plot(k, chains, params, data, out_dir, kw; n_walkers = n_walkers)
                f === nothing || push!(produced, f)
            catch err
                @warn "auto-dispatch plot `$k` failed" exception = err
            end
        end
        return produced
    else
        @warn "Unknown plot kind: $name"
        return nothing
    end
end

# =====================================================================
# Posterior predictive checks
# =====================================================================

function _run_ppc!(cfg, chains, params::Params, data::Data,
                    out_dir::AbstractString, summary::Dict)
    out_cfg = _to_dict(_get(cfg, :output; default = Dict()))
    ppc_enabled = Bool(_get(out_cfg, :ppc; default = true))
    ppc_enabled || return
    (n_rv(data) > 0 || n_phot(data) > 0) || return

    n_draws = Int(_get(out_cfg, :ppc_n_draws; default = 500))
    seed = Int(_get(cfg, :seed; default = 1))
    save_pdf = Bool(_get(out_cfg, :save_pdf; default = false))

    try
        ppc = posterior_predictive_check(chains, params, data;
                                          n_draws = n_draws,
                                          rng = MersenneTwister(seed))
        # Figure
        fig_path = joinpath(out_dir, "ppc.png")
        plot_ppc(ppc; filename = fig_path, save_pdf = save_pdf)
        plots = get(summary, "plots", String[])
        push!(plots, fig_path)
        summary["plots"] = plots
        # Scalar summary
        json_path = joinpath(out_dir, "ppc.json")
        open(json_path, "w") do io
            JSON3.pretty(io, ppc.summary; allow_inf = true)
        end
        summary["ppc"] = ppc.summary
    catch err
        @warn "PPC failed (continuing)" exception = (err, catch_backtrace())
        summary["ppc"] = Dict("status" => "failed",
                                "error" => sprint(showerror, err))
    end
end

# =====================================================================
# Detection limits — PT samplers only (need broad prior-seeded P
# coverage; NUTS/NS/NF-PT chains concentrate around modes).
# =====================================================================

# PT samplers preserving EMPEROR-style prior-seeded init across chains
# × temperatures. Detection limits computed from these are honest;
# others would give mode-biased curves.
const _PT_PRIOR_SEEDED = Set(("ptemcee", "transdim_ptemcee", "pt", "pt_warm"))

function _run_detection_limits!(cfg, chains, params::Params, data::Data,
                                 out_dir::AbstractString, summary::Dict)
    out_cfg = _to_dict(_get(cfg, :output; default = Dict()))
    sampler_name = String(_get(_to_dict(_get(cfg, :sampler;
                                              default = Dict())),
                                 :name; default = ""))

    # User can force on/off explicitly; default ON for PT samplers, OFF otherwise.
    explicit = _get(out_cfg, :detection_limits; default = nothing)
    if explicit === nothing
        sampler_name in _PT_PRIOR_SEEDED || return
    else
        Bool(explicit) || return
    end

    n_rv(data) > 0 || return                          # only RV makes sense here
    "P_k1" in params.layout.unfrozen_names || return  # need an RV-planet column

    n_bins = Int(_get(out_cfg, :detection_limits_n_bins; default = 30))
    confidence = Float64(_get(out_cfg, :detection_limits_confidence;
                                default = 0.95))
    planet = Int(_get(out_cfg, :detection_limits_planet; default = 1))
    save_pdf = Bool(_get(out_cfg, :save_pdf; default = false))

    try
        res = detection_limits(chains, params;
                                planet = planet, n_bins = n_bins,
                                confidence = confidence)
        fig_path = joinpath(out_dir, "detection_limits.png")
        plot_detection_limits(res; filename = fig_path, save_pdf = save_pdf)
        plots = get(summary, "plots", String[])
        push!(plots, fig_path)
        summary["plots"] = plots

        # Write the per-bin arrays to a NetCDF sidecar — `summary.json`
        # only carries the scalar metadata + a path to the arrays.
        # JSON-serialising thousands of floats per detection-limits run
        # was the original sin; this keeps summary.json human-skimmable
        # and lets downstream consumers stream the arrays without
        # parsing JSON.
        nc_path = joinpath(out_dir, "detection_limits.nc")
        save_detection_limits(nc_path, res)
        summary["detection_limits"] = Dict{String, Any}(
            "planet_index"  => res.planet_index,
            "confidence"    => res.confidence,
            "prior_P_lo"    => res.prior_P_lo,
            "prior_P_hi"    => res.prior_P_hi,
            "n_bins"        => length(res.P_centers),
            "arrays_path"   => "detection_limits.nc",
        )
    catch err
        @warn "Detection limits failed (continuing)" exception = (err, catch_backtrace())
        summary["detection_limits"] = Dict("status" => "failed",
                                             "error" => sprint(showerror, err))
    end
end

# =====================================================================
# PSIS-LOO / WAIC — sampler-agnostic. Skipped when any active noise
# model is a CovarianceNoise (GP); compute_loo throws and we report it
# in summary.json without failing the job.
# =====================================================================

function _run_loo!(cfg, chains, params::Params, data::Data, result,
                    summary::Dict)
    out_cfg = _to_dict(_get(cfg, :output; default = Dict()))
    Bool(_get(out_cfg, :loo; default = true)) || return
    (n_rv(data) > 0 || n_phot(data) > 0) || return

    n_draws = Int(_get(out_cfg, :loo_n_draws; default = 500))
    seed    = Int(_get(cfg, :seed; default = 1))

    # If the result carries a log_z, pass it so loo_compare_log_z lands
    # in the summary.
    log_z = hasfield(typeof(result), :log_evidence) ?
            getfield(result, :log_evidence) : nothing
    log_z_arg = log_z === nothing || !isfinite(Float64(log_z)) ?
                 nothing : Float64(log_z)

    try
        res = compute_loo(chains, params, data;
                           n_draws = n_draws,
                           rng = MersenneTwister(seed),
                           log_z = log_z_arg)
        summary["loo"] = Dict{String, Any}(
            "n_draws"            => res.n_draws,
            "n_obs"              => res.n_obs,
            "elpd_loo"           => res.elpd_loo,
            "se_elpd_loo"        => res.se_elpd_loo,
            "p_loo"              => res.p_loo,
            "elpd_waic"          => res.elpd_waic,
            "se_elpd_waic"       => res.se_elpd_waic,
            "p_waic"             => res.p_waic,
            "pareto_k_max"       => res.pareto_k_max,
            "pareto_k_warn_count" => res.pareto_k_warn_count,
            "loo_compare_log_z"  => res.loo_compare_log_z,
        )
    catch err
        @warn "LOO/WAIC failed (continuing)" exception = (err, catch_backtrace())
        summary["loo"] = Dict("status" => "skipped",
                                "error" => sprint(showerror, err))
    end
end

# Ensemble samplers store correlated walkers along the chain axis, so
# `assess_fit` must split each walker's own trace rather than treat
# walkers as independent chains.
const _ENSEMBLE_SAMPLERS = Set(("ptemcee", "transdim_ptemcee",
                                 "pt_whitening", "pa", "smc",
                                 "ensemble", "ess"))

# Post-fit "silent-wrong" guard. Fail-soft: never breaks the run, only
# logs a verdict and records it in the summary. Builds the prior bounds
# from the model layout so the prior-edge rail check is exercised. See
# `src/diagnostics/fit_health.jl`.
function _run_fit_health!(cfg, chains, params::Params, summary::Dict)
    out_cfg = _to_dict(_get(cfg, :output; default = Dict()))
    Bool(_get(out_cfg, :fit_health; default = true)) || return

    try
        # Hard prior bounds for every fitted parameter, keyed by name.
        layout = params.layout
        prior_bounds = Dict{Symbol, Tuple{Float64, Float64}}()
        for (nm, ps) in zip(layout.unfrozen_names, layout.unfrozen_priors)
            lo, hi = bounds(ps)
            prior_bounds[Symbol(nm)] = (Float64(lo), Float64(hi))
        end

        # `sampler` config is a block ({name, kwargs}), not a bare string — pull
        # the name out (tolerate a bare string too). The old String(::Dict) threw
        # for every job, so fit_health was silently dropped from the contract.
        sampler_cfg = _get(cfg, :sampler; default = "")
        sampler = sampler_cfg isa AbstractDict ?
                  String(_get(_to_dict(sampler_cfg), :name; default = "")) :
                  String(sampler_cfg)
        is_ensemble = sampler in _ENSEMBLE_SAMPLERS

        report = assess_fit(chains; prior_bounds = prior_bounds,
                             ensemble = is_ensemble)

        summary["fit_health"] = Dict{String, Any}(
            "overall"  => String(report.overall),
            "checks"   => Dict(String(c.name) => Dict(
                              "status"  => String(c.status),
                              "message" => c.message) for c in report.checks),
        )

        if report.overall === :fail
            @warn "Nereus.run_job: FIT HEALTH = FAIL — posterior may be " *
                  "silently wrong (disjoint modes / railed bound / corrupt " *
                  "log-post). Inspect before trusting results.\n" *
                  sprint(show, MIME"text/plain"(), report)
        elseif report.overall === :warn
            @warn "Nereus.run_job: fit health = WARN — review flagged checks.\n" *
                  sprint(show, MIME"text/plain"(), report)
        else
            @info "Nereus.run_job: fit health = OK (no red flags)."
        end
    catch err
        @warn "fit-health assessment failed (continuing)" exception = (err, catch_backtrace())
        summary["fit_health"] = Dict("status" => "skipped",
                                      "error" => sprint(showerror, err))
    end
end

# =====================================================================
# Summary writer
# =====================================================================

function _populate_summary!(summary, result, params, chains)
    summary["log_z"] = Float64(getfield(result, :log_evidence))
    summary["n_evals"] = Int(getfield(result, :n_evals))
    summary["sampler"] = Dict{String, Any}(
        "acceptance_within" => collect(Float64.(getfield(result,
                                                            :acceptance_within))),
        "acceptance_swap"   => collect(Float64.(getfield(result,
                                                            :acceptance_swap))),
    )
    if hasfield(typeof(result), :acceptance_transdim)
        summary["sampler"]["acceptance_transdim"] =
            collect(Float64.(getfield(result, :acceptance_transdim)))
    end

    # n_planets posterior (only when the chain has :n_planets)
    chain_names = Set(names(chains, :parameters))
    if :n_planets in chain_names
        np = Int.(vec(Array(chains[:n_planets])))
        counts = Dict{String, Float64}()
        for v in np
            k = string(v)
            counts[k] = get(counts, k, 0.0) + 1
        end
        n_total = length(np)
        for k in keys(counts)
            counts[k] /= n_total
        end
        summary["n_planets_posterior"] = counts
    end

    # Per-parameter median + 1/2/3-σ
    params_summary = Dict{String, Any}()
    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        sym in chain_names || continue
        v = vec(Array(chains[sym]))
        params_summary[name] = Dict{String, Float64}(
            "median" => median(v),
            "lo16"   => quantile(v, 0.16),
            "hi84"   => quantile(v, 0.84),
            "lo2sig" => quantile(v, 0.025),
            "hi2sig" => quantile(v, 0.975),
            "lo3sig" => quantile(v, 0.0015),
            "hi3sig" => quantile(v, 0.9985),
        )
    end
    summary["params"] = params_summary

    # --- Derived parameters: physical quantities (M_p, a, T_eq, ρ_p,
    # TSM, ESM, …) computed from the posterior. One block per planet
    # under summary["derived"]["k$k"] with the same median/lo16/hi84/2σ/3σ
    # schema as `params`. Skipped if stellar params not set in config.
    M_s = params.config.M_s
    R_s = params.config.R_s
    T_eff = hasproperty(params.config, :T_eff) ? params.config.T_eff : NaN
    if !isnan(M_s) && M_s > 0
        try
            derived_per_planet = compute_derived(chains, params;
                                                  M_s = M_s,
                                                  R_s = isnan(R_s) ? nothing : R_s,
                                                  T_eff = (isnan(T_eff) || T_eff <= 0) ?
                                                            nothing : T_eff)
            derived_summary = Dict{String, Any}()
            for (k, dp) in enumerate(derived_per_planet)
                per_planet = Dict{String, Any}()
                for (nm, vs) in dp.values
                    finite = filter(isfinite, vs)
                    isempty(finite) && continue
                    per_planet[nm] = Dict{String, Float64}(
                        "median" => median(finite),
                        "lo16"   => quantile(finite, 0.16),
                        "hi84"   => quantile(finite, 0.84),
                        "lo2sig" => quantile(finite, 0.025),
                        "hi2sig" => quantile(finite, 0.975),
                        "lo3sig" => quantile(finite, 0.0015),
                        "hi3sig" => quantile(finite, 0.9985),
                    )
                end
                isempty(per_planet) || (derived_summary["k$k"] = per_planet)
            end
            isempty(derived_summary) || (summary["derived"] = derived_summary)
        catch err
            @warn "compute_derived failed; skipping derived block in summary" exception = err
        end
    end
end

function _write_summary(out_dir, summary)
    # `allow_inf=true` is needed because samplers without evidence
    # estimators (NUTS, OFTI, rjmcmc, moms, moms_ns) write
    # `log_z = NaN` into the summary. The JSON spec forbids NaN, but
    # most parsers (Python json.loads with `parse_constant`,
    # JavaScript) handle it fine.
    open(joinpath(out_dir, "summary.json"), "w") do io
        JSON3.pretty(io, summary; allow_inf = true)
    end
end


"""
    _run_tomography_job(cfg, t0) -> summary

JOB_CONFIG route for Doppler tomography. Config shape:

    {"kind": "tomography",
     "output_dir": "...",
     "orbit": {"P":…, "a_Rs":…, "inc":…, "vsini":…, "T14":…, "vsys":…},
     "nights": [{"profiles": [[...]], "vgrid": [...], "t": [...],
                 "in_transit": [...], "bervs": [...]}, ...],
     "options": {"n_null": 300, "n_lambda": 721}}

Writes `summary.json` into `output_dir` like every other job, so a client
cannot tell a tomography run from a fit by how it reads the result.
"""
function _run_tomography_job(cfg::AbstractDict, t0::Float64)
    out_dir = String(_get(cfg, :output_dir; required = true))
    mkpath(out_dir)
    summary = Dict{String, Any}("status" => "ok", "kind" => "tomography")
    try
        orbit  = _get(cfg, :orbit; required = true)
        nights = _get(cfg, :nights; required = true)
        opts   = _get(cfg, :options; default = Dict())
        need(k) = begin
            v = _get(orbit, Symbol(k); default = nothing)
            v === nothing && error("tomography job: orbit.$k is required")
            Float64(v)
        end
        nλ = Int(_get(opts, :n_lambda; default = 721))
        r = fit_tomography(nights;
                P = need("P"), a_Rs = need("a_Rs"), inc = need("inc"),
                vsini = need("vsini"), T14 = need("T14"),
                vsys = Float64(_get(orbit, :vsys; default = 0.0)),
                λs = range(-π, π; length = nλ),
                n_null = Int(_get(opts, :n_null; default = 300)))
        merge!(summary, r.summary)
        summary["status"] = "ok"
    catch err
        summary["status"] = "failed"
        summary["error"] = sprint(showerror, err)
        summary["traceback"] = sprint(Base.show_backtrace, catch_backtrace())
    end
    summary["elapsed_sec"] = time() - t0
    open(joinpath(out_dir, "summary.json"), "w") do io
        JSON3.write(io, summary)
    end
    return summary
end
