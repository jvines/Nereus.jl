# run_job: obliquity fits -- RM velocities, Doppler shadow, or both.
#
# An obliquity fit is a job config like any other. The data block gains two
# kinds of entry, `rm_nights` (one spectroscopic transit's velocities each) and
# `tomography` (one transit's stack of line profiles each), and the model
# block an `obliquity` section that states the fit kind and the model options.
# `_build_obliquity_model` turns that into `obliquity_params`, the same
# builder a Julia caller uses, so the two cannot drift. Everything after the
# model -- sampler dispatch, checkpoint / resume, chains.nc, summary.json,
# plots -- is the ordinary run_job path. See docs/src/obliquity.md for the
# schema and a complete example.

const _OBL_FITS        = ("velocities", "shadow", "joint")
const _OBL_RV_NOISE    = ("sho", "matern", "white")
const _OBL_TOMO_NOISE  = ("matern", "sho", "white")
const _OBL_OCCULTATION = ("disc", "point")
const _OBL_LAMBDA      = ("wrapped", "bounded")
const _OBL_ARS         = ("a_Rs", "rho_s", "kepler")

_obliquity_cfg(cfg) = _has(cfg, :model) ? _get(_get(cfg, :model), :obliquity; default = nothing) :
                      nothing

function _obl_fit(ocfg)
    f = String(_get(ocfg, :fit; default = "joint"))
    return Symbol(f)
end

# A geometry value: a number (fixed) or [mean, sd] (Gaussian).
function _obl_value(x, name)
    x isa Real && return Float64(x)
    if x isa AbstractVector && length(x) == 2 && all(v -> v isa Real, x)
        return (Float64(x[1]), Float64(x[2]))
    end
    error("model.obliquity.$name must be a number (fixed) or [mean, sd]; got $(repr(x))")
end
_obl_center_value(x) = x isa Tuple ? x[1] : x

function _obl_validate_value!(errs, ocfg, key; required::Bool = true)
    if !_has(ocfg, key)
        required && push!(errs, "missing `model.obliquity.$key` (a number to fix it, " *
                               "or [mean, sd] for a Gaussian prior)")
        return
    end
    x = _get(ocfg, key)
    ok = x isa Real || (x isa AbstractVector && length(x) == 2 &&
                        all(v -> v isa Real, x) && x[2] > 0)
    ok || push!(errs, "`model.obliquity.$key` must be a number or [mean, sd > 0]; got $(repr(x))")
end

function _obl_enum!(errs, ocfg, key, allowed)
    _has(ocfg, key) || return
    v = String(_get(ocfg, key))
    v in allowed || push!(errs,
        "`model.obliquity.$key` = `$v` unknown (known: $(join(allowed, ", ")))")
end

"""
Schema checks for an obliquity job, collected into `errs` before anything is
read from disk.
"""
function _validate_obliquity!(errs::Vector{String}, cfg)
    ocfg = _obliquity_cfg(cfg)
    ocfg === nothing && return
    data_cfg = _get(cfg, :data; default = Dict())
    if !_has(ocfg, :fit)
        push!(errs, "missing `model.obliquity.fit` (one of $(join(_OBL_FITS, ", ")))")
        fit = "joint"
    else
        fit = String(_get(ocfg, :fit))
        fit in _OBL_FITS || push!(errs,
            "`model.obliquity.fit` = `$fit` unknown (known: $(join(_OBL_FITS, ", ")))")
    end
    for k in (:P, :Tc, :b, :a_Rs, :rr, :vsini)
        _obl_validate_value!(errs, ocfg, k)
    end
    _obl_validate_value!(errs, ocfg, :K; required = false)
    _obl_enum!(errs, ocfg, :rv_noise, _OBL_RV_NOISE)
    _obl_enum!(errs, ocfg, :tomo_noise, _OBL_TOMO_NOISE)
    _obl_enum!(errs, ocfg, :occultation, _OBL_OCCULTATION)
    _obl_enum!(errs, ocfg, :lambda, _OBL_LAMBDA)
    _obl_enum!(errs, ocfg, :a_Rs_param, _OBL_ARS)
    if _has(ocfg, :ecc)
        e = _get(ocfg, :ecc)
        (e == 0 || (e isa AbstractString && e == "free")) || push!(errs,
            "`model.obliquity.ecc` must be 0 (circular, fixed) or \"free\"; got $(repr(e))")
    end
    if _has(ocfg, :limb_darkening)
        ld = _get(ocfg, :limb_darkening)
        (ld isa AbstractVector && length(ld) == 2 && all(v -> v isa Real, ld)) ||
            push!(errs, "`model.obliquity.limb_darkening` must be [u1, u2]")
    end
    if _has(ocfg, :beta_p_floor)
        f = _get(ocfg, :beta_p_floor)
        (f isa Real && f >= 0) || push!(errs,
            "`model.obliquity.beta_p_floor` must be a number >= 0 (m/s)")
    end

    rm = _get(data_cfg, :rm_nights; default = Any[])
    tm = _get(data_cfg, :tomography; default = Any[])
    if fit in ("velocities", "joint") && isempty(rm) && !_has(data_cfg, :rv)
        push!(errs, "`model.obliquity.fit` = \"$fit\" needs RM velocities: " *
                    "add `data.rm_nights`")
    end
    if fit in ("shadow", "joint") && isempty(tm)
        push!(errs, "`model.obliquity.fit` = \"$fit\" needs line profiles: " *
                    "add `data.tomography`")
    end
    tags = String[]
    for (i, n) in enumerate(rm)
        where = "data.rm_nights[$i]"
        _has(n, :tag) ? push!(tags, String(_get(n, :tag))) :
                        push!(errs, "`$where` missing `tag` (it names the night's instrument)")
        (_has(n, :file) || _has(n, :values)) ||
            push!(errs, "`$where` needs `file` or `values`")
        if _has(n, :file) && !isfile(String(_get(n, :file)))
            push!(errs, "`$where.file` does not exist: $(_get(n, :file))")
        end
        if !_has(n, :sigma0)
            push!(errs, "`$where` missing `sigma0`: the measured dispersion (m/s) of a " *
                        "Gaussian fitted to that night's out-of-transit CCF. The ARoME " *
                        "anomaly needs it, and it must be measured, not derived from v sin i")
        else
            s0 = _get(n, :sigma0)
            (s0 isa Real && s0 > 0) || push!(errs, "`$where.sigma0` must be a positive number (m/s)")
        end
    end
    length(unique(tags)) == length(tags) ||
        push!(errs, "`data.rm_nights` tags must be unique; got $(tags)")
    ttags = String[]
    for (i, s) in enumerate(tm)
        where = "data.tomography[$i]"
        _has(s, :tag) ? push!(ttags, String(_get(s, :tag))) :
                        push!(errs, "`$where` missing `tag`")
        for k in (:profiles, :vgrid, :times)
            if !_has(s, k)
                push!(errs, "`$where` missing `$k` (a file path)")
            elseif !isfile(String(_get(s, k)))
                push!(errs, "`$where.$k` does not exist: $(_get(s, k))")
            end
        end
        if _has(s, :berv) && !isfile(String(_get(s, :berv)))
            push!(errs, "`$where.berv` does not exist: $(_get(s, :berv))")
        end
        length(tm) > 1 && !_has(s, :berv) && push!(errs,
            "`$where` missing `berv`: with several nights pooled, the barycentric " *
            "velocities are not optional -- without them the shadow tracks of " *
            "different nights sit at different offsets")
        (_has(s, :t14_hours) || _has(ocfg, :t14_hours)) || push!(errs,
            "`$where` needs `t14_hours` (or a shared `model.obliquity.t14_hours`): " *
            "the in-transit exposures are excluded from the out-of-transit mean")
        _has(s, :vsys) || push!(errs, "`$where` missing `vsys` (km/s)")
    end
    length(unique(ttags)) == length(ttags) ||
        push!(errs, "`data.tomography` tags must be unique; got $(ttags)")
    if Bool(_get(ocfg, :noise_menu; default = false))
        _has(cfg, :transdim) || push!(errs,
            "`model.obliquity.noise_menu` = true selects the noise per night trans-" *
            "dimensionally: add a `transdim` block (with `noise: true`) and a trans-dim sampler")
        _has(cfg, :noise_models) && push!(errs,
            "`model.obliquity.noise_menu` and `noise_models` are mutually exclusive")
    end
    _has(cfg, :noise_menu) && push!(errs,
        "the top-level `noise_menu` is the RV planet-search menu; an obliquity fit " *
        "selects its per-night noise with `model.obliquity.noise_menu = true`")
    return nothing
end

# ---- data -------------------------------------------------------------

function _read_cols(path::AbstractString)
    return readdlm(path, Float64; comments = true, comment_char = '#')
end

"""
Parse `data.rm_nights` into `(t, rv, err, tag)` per night.

Each entry: `tag` (instrument name for this night), `file` (whitespace
columns, `#` comments) or `values` (`bjd`, `rv`, `rv_err`), optional
`columns` ([time, rv, err], 1-based; default [1, 2, 3]) and `rv_unit`
(`"m/s"` default or `"km/s"`). `sigma0` is read by the model builder;
`instrument` and `pipeline` are recorded in the summary.
"""
function _parse_rm_nights(blocks)
    out = NamedTuple{(:tag, :t, :rv, :err), Tuple{String, Vector{Float64},
                                                    Vector{Float64}, Vector{Float64}}}[]
    for b in blocks
        tag = String(_get(b, :tag; required = true))
        if _has(b, :values)
            v = _get(b, :values)
            t  = Float64.(_get(v, :bjd; required = true))
            rv = Float64.(_get(v, :rv; required = true))
            er = Float64.(_get(v, :rv_err; required = true))
        else
            M = _read_cols(String(_get(b, :file)))
            c = Int.(collect(_get(b, :columns; default = [1, 2, 3])))
            t, rv, er = M[:, c[1]], M[:, c[2]], M[:, c[3]]
        end
        unit = String(_get(b, :rv_unit; default = "m/s"))
        unit in ("m/s", "km/s") || error("data.rm_nights `$tag`: rv_unit must be m/s or km/s")
        if unit == "km/s"
            rv = rv .* 1000; er = er .* 1000
        end
        push!(out, (; tag, t = Vector{Float64}(t), rv = Vector{Float64}(rv),
                      err = Vector{Float64}(er)))
    end
    return out
end

"""
Build one `TomoNight` per `data.tomography` entry: the profiles (n_exposure ×
n_v), their velocity grid (km/s), the exposure times (BJD) and barycentric
velocities (km/s), reduced to a residual map by `tomogram_residuals` --
linear continuum from the wings, shift to the stellar rest frame at `vsys`,
the mean OUT-OF-TRANSIT profile subtracted, scaled to unit line depth. The
night's transit is the one of the model ephemeris (`P`, `Tc`) nearest its
exposures; exposures within `t14_hours / 2` of it are in transit.

Optional per stack: `grid` ([min, max, n] in km/s; default [-60, 60, 241]),
`norm_depth` (default true).
"""
function _parse_tomo_stacks(blocks, P::Real, Tc::Real, ocfg)
    out = TomoNight[]
    for b in blocks
        tag  = String(_get(b, :tag; required = true))
        prof = readdlm(String(_get(b, :profiles; required = true)))
        vg   = vec(readdlm(String(_get(b, :vgrid; required = true))))
        t    = vec(readdlm(String(_get(b, :times; required = true))))
        berv = _has(b, :berv) ? vec(readdlm(String(_get(b, :berv)))) : nothing
        size(prof, 1) == length(t) || error("data.tomography `$tag`: $(size(prof, 1)) " *
            "profiles but $(length(t)) times")
        size(prof, 2) == length(vg) || error("data.tomography `$tag`: profiles have " *
            "$(size(prof, 2)) velocity columns but the grid has $(length(vg))")
        berv === nothing || length(berv) == length(t) || error(
            "data.tomography `$tag`: $(length(berv)) BERVs for $(length(t)) exposures")
        g = _get(b, :grid; default = [-60.0, 60.0, 241])
        grid = range(Float64(g[1]), Float64(g[2]); length = Int(g[3]))
        t14 = Float64(_get(b, :t14_hours; default = _get(ocfg, :t14_hours; default = NaN)))
        # The transit of the model ephemeris nearest this night, written as
        # Tc + E·P so that it is the very number the likelihood places it at.
        E   = round((sum(t) / length(t) - Tc) / P)
        Tcn = Tc + E * P
        intr = abs.((t .- Tcn) .* 24) .<= t14 / 2
        any(intr) || error("data.tomography `$tag`: no exposure within t14/2 of the " *
                           "transit at $Tcn -- wrong ephemeris or times?")
        all(intr) && error("data.tomography `$tag`: every exposure is in transit, so " *
                           "there is no out-of-transit profile to subtract")
        gg, R = tomogram_residuals(prof, vg, intr; vsys = Float64(_get(b, :vsys)),
                                   bervs = berv, grid = grid,
                                   norm_depth = Bool(_get(b, :norm_depth; default = true)))
        push!(out, TomoNight(tag, Vector{Float64}(t), R, Vector{Float64}(gg), Tcn))
    end
    return out
end

"""
Add the obliquity data blocks to the `Data` keyword set `kw` (as built by
`_build_data` from the other blocks) and return the extended RV instrument
names. RM nights are appended as RV instruments after any `rv` block's.
Which blocks are read follows `model.obliquity.fit`: the velocities fit reads
no maps, the shadow fit no RM nights.
"""
function _add_obliquity_data!(kw::Dict{Symbol,Any}, inst_names_rv::Vector{String},
                              data_cfg, ocfg)
    fit = _obl_fit(ocfg)
    names = copy(inst_names_rv)
    rm_blocks = _get(data_cfg, :rm_nights; default = Any[])
    if fit in (:velocities, :joint) && !isempty(rm_blocks)
        nights = _parse_rm_nights(rm_blocks)
        t  = get(kw, :t_rv, Float64[]);  rv = get(kw, :rv, Float64[])
        er = get(kw, :rv_err, Float64[]); ii = get(kw, :rv_inst, Int[])
        t, rv, er, ii = copy(t), copy(rv), copy(er), copy(ii)
        for n in nights
            n.tag in names && error("RM night tag `$(n.tag)` collides with an RV " *
                                    "instrument of the same name")
            push!(names, n.tag)
            append!(t, n.t); append!(rv, n.rv); append!(er, n.err)
            append!(ii, fill(length(names), length(n.t)))
        end
        kw[:t_rv] = t; kw[:rv] = rv; kw[:rv_err] = er; kw[:rv_inst] = ii
        haskey(kw, :rv_comp) && (kw[:rv_comp] = vcat(kw[:rv_comp],
                                                     ones(Int, length(t) - length(kw[:rv_comp]))))
        haskey(kw, :t_ref) || (kw[:t_ref] = sum(t) / length(t))
    elseif !isempty(rm_blocks)
        @info "run_job: model.obliquity.fit = $fit -- data.rm_nights are not used"
    end
    tomo_blocks = _get(data_cfg, :tomography; default = Any[])
    if fit in (:shadow, :joint) && !isempty(tomo_blocks)
        P  = _obl_center_value(_obl_value(_get(ocfg, :P), "P"))
        Tc = _obl_center_value(_obl_value(_get(ocfg, :Tc), "Tc"))
        kw[:tomo] = _parse_tomo_stacks(tomo_blocks, P, Tc, ocfg)
    elseif !isempty(tomo_blocks)
        @info "run_job: model.obliquity.fit = $fit -- data.tomography is not used"
    end
    return names
end

# ---- model ------------------------------------------------------------

"""
Build `(params, target, menu)` for an obliquity job through
`obliquity_params`. The top-level `priors` override any prior by name;
`noise_models`, when given, replace the standard per-night noise.
"""
function _build_obliquity_model(cfg, data::Data, inst_names_rv, inst_names_pm)
    ocfg = _obliquity_cfg(cfg)
    fit = _obl_fit(ocfg)
    rm_blocks = fit in (:velocities, :joint) ?
                _get(_get(cfg, :data), :rm_nights; default = Any[]) : Any[]
    sigma0 = Dict{String,Float64}(String(_get(b, :tag)) => Float64(_get(b, :sigma0))
                                  for b in rm_blocks)
    for (k, v) in _get(ocfg, :sigma0; default = Dict())
        sigma0[String(k)] = Float64(v)
    end
    rv_names = Vector{String}(inst_names_rv)
    tomo_tags = [nt.tag for nt in data.tomo]

    menu = nothing
    noise_models = nothing
    if Bool(_get(ocfg, :noise_menu; default = false))
        menu = obliquity_noise_menu(rv_names, tomo_tags)
        noise_models = menu.noise_models
    elseif _has(cfg, :noise_models)
        noise_models = _build_noise_models(_get(cfg, :noise_models),
                                           InstrumentConfig(rv = rv_names, pm = inst_names_pm))
    end

    ld = _get(ocfg, :limb_darkening; default = nothing)
    ecc = _get(ocfg, :ecc; default = 0)
    K = _has(ocfg, :K) ? _obl_value(_get(ocfg, :K), "K") : nothing
    priors = _build_priors(_get(cfg, :priors; default = Dict()))
    star = _get(cfg, :star; default = Dict())
    kw = Dict{Symbol,Any}(
        :P => _obl_value(_get(ocfg, :P), "P"),
        :Tc => _obl_value(_get(ocfg, :Tc), "Tc"),
        :b => _obl_value(_get(ocfg, :b), "b"),
        :a_Rs => _obl_value(_get(ocfg, :a_Rs), "a_Rs"),
        :rr => _obl_value(_get(ocfg, :rr), "rr"),
        :vsini => _obl_value(_get(ocfg, :vsini), "vsini"),
        :K => (isempty(rv_names) ? nothing : K),
        :ecc => (ecc isa AbstractString && ecc == "free") ? :free : 0.0,
        :lambda_prior => Symbol(_get(ocfg, :lambda; default = "wrapped")),
        :sigma0 => (isempty(sigma0) ? nothing : sigma0),
        :beta_p_floor => Float64(_get(ocfg, :beta_p_floor; default = 0.0)),
        :occultation => Symbol(_get(ocfg, :occultation; default = "disc")),
        :ld => ld === nothing ? nothing : (Float64(ld[1]), Float64(ld[2])),
        :shared_alpha => Bool(_get(ocfg, :shared_alpha; default = false)),
        :rv_noise => Symbol(_get(ocfg, :rv_noise; default = "sho")),
        :tomo_noise => Symbol(_get(ocfg, :tomo_noise; default = "matern")),
        :noise_models => noise_models,
        :transdim_noise => menu !== nothing,
        :pm_names => Vector{String}(inst_names_pm),
        :a_Rs_param => Symbol(_get(ocfg, :a_Rs_param; default = "a_Rs")),
        :priors => priors,
    )
    _has(star, :M_s) && (kw[:M_s] = Float64(_get(star, :M_s)))
    _has(star, :R_s) && (kw[:R_s] = Float64(_get(star, :R_s)))
    params = obliquity_params(data, rv_names; kw...)
    target = NereusTarget(params, data; unconstrained = false)
    return params, target, menu
end

"""
What an obliquity job ran on, for summary.json: the fit kind, every RM night
(tag, instrument, pipeline, σ0, number of points) and every residual map (tag,
exposures, in-transit nights' Tc), and the model options.
"""
function _obliquity_summary(cfg, data::Data, params::Params)
    ocfg = _obliquity_cfg(cfg)
    fit = _obl_fit(ocfg)
    data_cfg = _get(cfg, :data)
    rm = Any[]
    if fit in (:velocities, :joint)
        rn = params.config.instruments.rv_names
        for b in _get(data_cfg, :rm_nights; default = Any[])
            tag = String(_get(b, :tag))
            i = findfirst(==(tag), rn)
            push!(rm, Dict{String,Any}("tag" => tag,
                "instrument" => String(_get(b, :instrument; default = tag)),
                "pipeline" => String(_get(b, :pipeline; default = "")),
                "sigma0" => Float64(_get(b, :sigma0)),
                "n_points" => i === nothing ? 0 : count(==(i), data.rv_inst)))
        end
    end
    tomo = Any[Dict{String,Any}("tag" => nt.tag, "n_exposures" => length(nt.t),
                                "n_velocity" => length(nt.grid), "Tc" => nt.Tc)
               for nt in data.tomo]
    o = params.config.obliquity
    return Dict{String,Any}("fit" => String(fit), "rm_nights" => rm,
        "tomography" => tomo,
        "options" => Dict{String,Any}("occultation" => String(o.occultation),
            "beta_p_floor" => o.beta_p_floor, "shared_alpha" => o.shared_alpha,
            "spectroscopic_ld" => o.spec_ld,
            "sigma0" => Dict{String,Any}(k => v for (k, v) in o.sigma0)))
end

"""
An obliquity job's `output` defaults: the posterior predictive check, PSIS-LOO
and the detection-limit curve are RV-planet diagnostics -- a periodogram of
the residuals, a pointwise RV likelihood that leaves the maps out, and a
K_lim(P) curve -- and are OFF unless the job asks for them.
"""
function _obliquity_output_defaults(cfg)
    c = Dict{String,Any}(String(k) => v for (k, v) in pairs(cfg))
    out = Dict{String,Any}(String(k) => v for (k, v) in pairs(_get(cfg, :output; default = Dict())))
    for k in ("ppc", "loo", "detection_limits")
        haskey(out, k) || (out[k] = false)
    end
    c["output"] = out
    return c
end
