# Nereus non-inference operations — detection, detrending, diagnostics.
#
# These are separate surfaces, not modes of a fit. Each is a thin adapter over
# the package function that already does the work; the adapter's whole job is
# to accept plain JSON-able arguments and return a plain JSON-able Dict, so the
# Python client stays transport-only.
#
# Argument validation happens HERE rather than being left to Julia's method
# dispatch, because a MethodError arriving over a socket is unreadable.

using Statistics: median, std, quantile
using Random

_vec(x) = Float64.(collect(x))

# Accept either key form. `_pget` already reads Symbol- and String-keyed
# payloads, and the JSON transport delivers Strings — so checking only Symbols
# made every op reject the very payload shape it was written for.
_haskey2(p::AbstractDict, k) = haskey(p, k) || haskey(p, String(k))
_haskey2(p, k) = haskey(p, k)

function _need(p, keys...)
    for k in keys
        _haskey2(p, k) || error("missing required argument $(k). Got: " *
                                join(sort!(String.(collect(Base.keys(p)))), ", "))
    end
end

_kw(p, drop) = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(p)
                                if !(String(k) in drop))

# --- detection ---------------------------------------------------------------

"""
Savitzky–Golay window in CADENCES, odd, spanning `days` of time.

`find_transits` defaults to detrend=:savgol, and `detrend_savgol` has no
default `window_length` — so a caller who just wants transits found would hit
`UndefKeywordError` for a parameter they have no way to guess. Derive it from
the actual cadence instead: ~0.5 d is long enough to leave a few-hour transit
untouched while removing stellar variability.
"""
function _savgol_window(t::AbstractVector; days::Real = 0.5, minimum_len::Int = 11)
    length(t) < 3 && return minimum_len
    cad = median(diff(sort(collect(t))))
    cad <= 0 && return minimum_len
    w = max(minimum_len, round(Int, days / cad))
    return isodd(w) ? w : w + 1
end

_nt(d::AbstractDict) = NamedTuple(Symbol(k) => v for (k, v) in d)
_nt(x) = x

function act_detect_transits(p)
    _need(p, :t, :flux, :flux_err)
    t = _vec(_pget(p, :t))
    kw = _kw(p, ["t", "flux", "flux_err"])
    dk = Dict{Symbol,Any}(pairs(_nt(get(kw, :detrend_kwargs, Dict()))))
    if get(kw, :detrend, :savgol) === :savgol && !haskey(dk, :window_length)
        dk[:window_length] = _savgol_window(t)
    end
    kw[:detrend_kwargs] = NamedTuple(dk)
    r = find_transits(t, _vec(_pget(p, :flux)), _vec(_pget(p, :flux_err)); kw...)
    return _as_dict(r)
end

function act_detect_rv_planets(p)
    _need(p, :t, :rv, :rv_err)
    r = find_rv_planets(_vec(_pget(p, :t)), _vec(_pget(p, :rv)), _vec(_pget(p, :rv_err));
                        _kw(p, ["t", "rv", "rv_err"])...)
    return _as_dict(r)
end

function act_detect_rotation(p)
    _need(p, :t, :flux, :flux_err)
    r = find_rotation_period(_vec(_pget(p, :t)), _vec(_pget(p, :flux)), _vec(_pget(p, :flux_err));
                             _kw(p, ["t", "flux", "flux_err"])...)
    return _as_dict(r)
end

function act_detect_segments(p)
    _need(p, :t)
    return Dict("segments" => find_segments(_vec(_pget(p, :t)); _kw(p, ["t"])...))
end

# --- detrending --------------------------------------------------------------

for (name, fn) in (("savgol", :detrend_savgol), ("notch", :detrend_notch),
                   ("locor", :detrend_locor))
    @eval function $(Symbol("act_detrend_", name))(p)
        _need(p, :t, :flux, :flux_err)
        t = _vec(_pget(p, :t))
        kw = _kw(p, ["t", "flux", "flux_err"])
        if $(QuoteNode(fn)) in (:detrend_notch, :detrend_locor) && get(kw, :transit_mask, nothing) !== nothing
            # JSON delivers the mask as a Vector{Any}; the kernel wants Bool.
            kw[:transit_mask] = Bool[Bool(x) for x in kw[:transit_mask]]
        end
        $(QuoteNode(fn)) === :detrend_savgol && !haskey(kw, :window_length) &&
            (kw[:window_length] = _savgol_window(t))
        r = $fn(t, _vec(_pget(p, :flux)), _vec(_pget(p, :flux_err)); kw...)
        return _as_dict(r)
    end
end

# `detrend_gp` dispatches on the kernel TYPE (via `_kernel_coefs`) and never
# reads the kernel's channel/instruments fields, so default construction is
# right here. Kinds resolve through the same `_NOISE_TYPES` registry `run_job`
# uses, so the two routes cannot drift apart on kernel names.
const _DETREND_GP_KERNELS = ("CeleriteSHO", "CeleriteRotation",
                             "CeleriteRotationFM17")

"""
Build the `CovarianceNoise` kernel for `detrend.gp` from a JSON-able spec:
either a bare kind string (`"CeleriteSHO"`) or `{"kind": ..., "kwargs": {...}}`.
"""
function _detrend_gp_kernel(spec)
    if spec isa AbstractString
        kind, kwargs = String(spec), Dict{Symbol,Any}()
    elseif spec isa AbstractDict
        kd = Dict(String(k) => v for (k, v) in pairs(spec))
        haskey(kd, "kind") || error("detrend.gp: `kernel` object needs a " *
                                    "\"kind\" key. Got: " *
                                    join(sort!(collect(keys(kd))), ", "))
        kind   = String(kd["kind"])
        kwargs = Dict{Symbol,Any}(Symbol(k) => v
                                  for (k, v) in pairs(get(kd, "kwargs", Dict())))
    else
        error("detrend.gp: `kernel` must be a kind string or an object with " *
              "a \"kind\" key; got $(typeof(spec))")
    end
    kind in _DETREND_GP_KERNELS || error(
        "detrend.gp: kernel must be one of " *
        join(_DETREND_GP_KERNELS, ", ") * "; got $(repr(kind)). " *
        "(ActivityGP is a multivariate RV+indicator model, not a light-curve " *
        "detrending kernel.)")
    return _NOISE_TYPES[kind](; kwargs...)
end

function act_detrend_gp(p)
    _need(p, :t, :flux, :flux_err, :kernel)
    # `_pget` (not `p.field`) so the op works with the Dict payloads the JSON
    # transport and the tomogram ops already use, as well as a NamedTuple.
    kernel = _detrend_gp_kernel(_pget(p, :kernel))
    kw = _kw(p, ["t", "flux", "flux_err", "kernel"])
    r = detrend_gp(_vec(_pget(p, :t)), _vec(_pget(p, :flux)),
                   _vec(_pget(p, :flux_err)), kernel; kw...)
    return _as_dict(r)
end

# --- diagnostics -------------------------------------------------------------
# These need a fitted posterior, so they take a chains file rather than arrays.

function _load(p)
    # `chains_path`, or the `output_dir` of a fit (its chains.nc).
    path = if _haskey2(p, :chains_path) && _pget(p, :chains_path) !== nothing
        String(_pget(p, :chains_path))
    elseif _haskey2(p, :output_dir) && _pget(p, :output_dir) !== nothing
        joinpath(String(_pget(p, :output_dir)), "chains.nc")
    else
        _need(p, :chains_path)
    end
    isfile(path) || error("no chains at $path; run the fit first")
    chains, meta = load_chains(path)
    return chains, meta
end

# NOTE: there is deliberately no `diagnostics.loo` feature op. `compute_loo`
# needs the Params AND Data of the original fit to re-evaluate the pointwise
# likelihood, and `load_chains` recovers neither — a chains file carries the
# draws plus NetCDF attributes, not the model layout, priors, noise models or
# the observations. So LOO cannot cross the JSON feature boundary; call
# `compute_loo(chains, params, data)` in the session that produced the fit.

function act_diag_ess_rhat(p)
    chains, _ = _load(p)
    out = Dict{String,Any}()
    for s in MCMCChains.names(chains, :parameters)
        v = Array(chains[s])
        out[String(s)] = Dict("ess" => _ess(vec(v)), "rhat" => _rhat(v))
    end
    return out
end

"Effective sample size via the initial-positive-sequence autocorrelation sum."
function _ess(x::AbstractVector)
    n = length(x); n < 4 && return float(n)
    x̄ = mean(x); v = var(x); v == 0 && return float(n)
    ρsum = 0.0
    for lag in 1:min(n - 2, 1000)
        ρ = sum((x[1:end-lag] .- x̄) .* (x[1+lag:end] .- x̄)) / ((n - lag) * v)
        ρ < 0 && break
        ρsum += ρ
    end
    return n / (1 + 2ρsum)
end

"Gelman-Rubin Rhat across chains; 1.0 when there is only one chain."
function _rhat(a::AbstractArray)
    ndims(a) < 3 || size(a, 3) < 2 ? 1.0 : begin
        m = size(a, 3); n = size(a, 1)
        cm = [mean(view(a, :, 1, c)) for c in 1:m]
        cv = [var(view(a, :, 1, c)) for c in 1:m]
        B = n * var(cm); W = mean(cv)
        W == 0 ? 1.0 : sqrt(((n - 1) / n * W + B / n) / W)
    end
end

_as_dict(x::AbstractDict) = Dict{String,Any}(String(k) => v for (k, v) in x)
_as_dict(x::NamedTuple)   = Dict{String,Any}(String(k) => getfield(x, k) for k in keys(x))
function _as_dict(x)
    isstructtype(typeof(x)) || return Dict("value" => x)
    return Dict{String,Any}(String(f) => getfield(x, f) for f in fieldnames(typeof(x)))
end



# --- Doppler tomography --------------------------------------------------------
# Six ops over src/tomography.jl. These are NOT samplers: the estimator is a
# matched filter over lambda against the predicted shadow track, with
# significance from a scrambled-frame null. They are exposed individually
# because a tomography analysis is genuinely stepwise — you build CCF profiles,
# look at the residual tomogram, then filter — and each step is worth seeing.

"Matrix from a JSON list-of-rows (or pass a Matrix straight through)."
_fmat(x::AbstractMatrix) = Float64.(x)
function _fmat(rows)
    rs = collect(rows)
    isempty(rs) && error("empty profile matrix")
    n = length(first(rs))
    all(r -> length(r) == n, rs) || error("ragged profile matrix: rows must " *
        "all have $(n) velocity bins")
    M = Matrix{Float64}(undef, length(rs), n)
    for (i, r) in enumerate(rs); M[i, :] .= Float64.(collect(r)); end
    return M
end

_fbool(x) = Bool[Bool(v) for v in x]

# The payload is a JSON3.Object over the socket but a plain Dict when called
# from Julia; `p.x` only works for the first. Go through the key either way.
_pget(p::AbstractDict, k) = p[haskey(p, k) ? k : String(k)]
_pget(p, k) = getproperty(p, k)

"Orbit/geometry keywords every filter op needs, pulled off the payload."
function _tomo_geom(p)
    _need(p, :t, :Tc, :P, :a_Rs, :inc, :vsini)
    return (_vec(_pget(p, :t)), Float64(_pget(p, :Tc)), Float64(_pget(p, :P)), Float64(_pget(p, :a_Rs)),
            Float64(_pget(p, :inc)), Float64(_pget(p, :vsini)))
end

function act_tomo_ccf_profile(p)
    _need(p, :lambda_obs, :flux, :mask_lambda, :mask_weight, :vgrid)
    prof = ccf_profile(_vec(_pget(p, :lambda_obs)), _vec(_pget(p, :flux)), _vec(_pget(p, :mask_lambda)),
                       _vec(_pget(p, :mask_weight)), _vec(_pget(p, :vgrid));
                       berv = Float64(get(p, :berv, 0.0)))
    return Dict{String,Any}("vgrid" => _vec(_pget(p, :vgrid)), "profile" => prof)
end

function act_tomo_residuals(p)
    _need(p, :profiles, :vgrid, :in_transit)
    gopt = get(p, :grid, nothing)
    g, R = tomogram_residuals(_fmat(_pget(p, :profiles)), _vec(_pget(p, :vgrid)),
                              _fbool(_pget(p, :in_transit));
                              vsys  = Float64(get(p, :vsys, 0.0)),
                              bervs = (haskey(p, :bervs) && _pget(p, :bervs) !== nothing) ?
                                      _vec(_pget(p, :bervs)) : nothing,
                              grid  = gopt === nothing ?
                                      range(-60, 60; length = 241) : _vec(gopt))
    return Dict{String,Any}("grid" => collect(g),
                            "residuals" => [collect(r) for r in eachrow(R)],
                            "n_frames" => size(R, 1))
end

function act_tomo_shadow_track(p)
    t, Tc, P, a_Rs, inc, vsini = _tomo_geom(p)
    _need(p, :lambda)
    v = shadow_track(t, Tc, P, a_Rs, inc, Float64(_pget(p, :lambda)), vsini)
    return Dict{String,Any}("t" => t, "v_shadow" => v,
                            "in_transit" => .!isnan.(v),
                            "n_in_transit" => count(!isnan, v))
end

"Common keyword tail for the filter ops (lambda scan, line width, weight)."
function _tomo_filter_kw(p)
    kw = Dict{Symbol,Any}()
    if haskey(p, :lambdas) && _pget(p, :lambdas) !== nothing
        kw[:λs] = _vec(_pget(p, :lambdas))
    elseif haskey(p, :n_lambda)
        kw[:λs] = range(-π, π; length = Int(_pget(p, :n_lambda)))
    end
    haskey(p, :weight)  && (kw[:weight]  = Float64(_pget(p, :weight)))
    haskey(p, :sigma_line) && (kw[:σ_line] = Float64(_pget(p, :sigma_line)))
    return kw
end

function act_tomo_matched_filter(p)
    _need(p, :residuals, :grid)
    t, Tc, P, a_Rs, inc, vsini = _tomo_geom(p)
    λv, s = tomogram_matched_filter(_fmat(_pget(p, :residuals)), _vec(_pget(p, :grid)), t, Tc, P,
                                    a_Rs, inc, vsini; _tomo_filter_kw(p)...)
    # abs, not max: the template is a deficit, so a shadow scores NEGATIVE
    # against a mask-CCF (which peaks) and positive against a DRS dip. Taking
    # the signed max silently returns the opposite lobe for one of the two
    # conventions. `tomogram_pooled` picks on |score| for the same reason.
    i = argmax(abs.(s))
    return Dict{String,Any}("lambda_rad" => λv[i], "lambda_deg" => rad2deg(λv[i]),
                            "score" => s[i], "lambda_scan_rad" => collect(λv),
                            "lambda_scan_deg" => rad2deg.(collect(λv)),
                            "lambda_scan_score" => collect(s))
end

function act_tomo_null_distribution(p)
    _need(p, :residuals, :grid)
    t, Tc, P, a_Rs, inc, vsini = _tomo_geom(p)
    R    = _fmat(_pget(p, :residuals))
    grid = _vec(_pget(p, :grid))
    n    = Int(get(p, :n, 400))
    seed = get(p, :seed, nothing)
    rng  = seed === nothing ? Random.default_rng() : Random.MersenneTwister(Int(seed))
    kw   = _tomo_filter_kw(p)
    null = tomogram_null_distribution(R, grid, t, Tc, P, a_Rs, inc, vsini;
                                      n, rng, kw...)
    out = Dict{String,Any}("null" => null, "n" => n,
                           "null_median" => median(null),
                           "null_p99" => quantile(null, 0.99))
    # A null distribution on its own answers nothing — report the observed peak
    # against it, which is the only number anyone actually wants.
    _, s = tomogram_matched_filter(R, grid, t, Tc, P, a_Rs, inc, vsini; kw...)
    # The null is built from the SIGNED maximum, so the observed statistic has
    # to be the signed maximum too or the p-value compares two different
    # quantities. On a convention where the shadow scores negative this is
    # underpowered rather than wrong; `peak_abs` reports the real peak.
    obs = maximum(s)
    out["observed"] = obs
    out["peak_abs"] = maximum(abs.(s))
    out["p_value"]  = (count(>=(obs), null) + 1) / (n + 1)
    return out
end

function act_tomo_injection_test(p)
    _need(p, :profiles, :vgrid, :in_transit, :rr)
    t, Tc, P, a_Rs, inc, vsini = _tomo_geom(p)
    λ_true = Float64(get(p, :lambda_true, 0.0))
    λ̂, score = tomogram_injection_test(_fmat(_pget(p, :profiles)), _vec(_pget(p, :vgrid)), t,
        _fbool(_pget(p, :in_transit)), Tc, P, a_Rs, inc, vsini, Float64(_pget(p, :rr));
        λ_true, u1 = Float64(get(p, :u1, 0.3)), u2 = Float64(get(p, :u2, 0.3)),
        vsys = Float64(get(p, :vsys, 0.0)),
        bervs = (haskey(p, :bervs) && _pget(p, :bervs) !== nothing) ? _vec(_pget(p, :bervs)) : nothing,
        _tomo_filter_kw(p)...)
    return Dict{String,Any}("lambda_rad" => λ̂, "lambda_deg" => rad2deg(λ̂),
                            "lambda_true_deg" => rad2deg(λ_true),
                            "error_deg" => rad2deg(λ̂ - λ_true), "score" => score)
end


# --- operations on a finished fit ----------------------------------------------
# `diagnostics.loo`, `diagnostics.ppc`, `diagnostics.fit_health`, `evidence`,
# `detection_limits` and `diagnostics.label_switching` read a posterior. A chains
# file holds the draws and not the model (layout, priors, noise models, data), so
# the ones that re-evaluate the likelihood take the fit's own request again, as
# `replot` does: the channels, `planets`, `priors` and the model keywords the fit
# was called with, and `output_dir` (whose `chains.nc` is read) or `chains_path`.
# The target is rebuilt with `_target_from`, the route fit_rv / fit_transit /
# fit_astrometry / fit_joint build theirs by, and each block comes from the same
# helper the post-fit stage of those fits runs, so it is the block the fit wrote.

"JSON3 objects/arrays -> plain Dict/Vector, recursively; anything else as is."
_plain(x::AbstractDict) = Dict{String,Any}(String(k) => _plain(v) for (k, v) in pairs(x))
_plain(x::AbstractVector) = Any[_plain(v) for v in x]
_plain(x) = x

# Per-instrument tables of a channel -> the NamedTuples of Float64 columns the
# fit_* functions take (the daemon does the same on the way in).
function _replay_channels(p)
    out = Any[]
    for c in _pget(p, :channels)
        d = _plain(c)
        if haskey(d, "data") && d["data"] isa AbstractDict
            d["data"] = Dict{String,Any}(String(inst) => NamedTuple(
                Symbol(k) => (v isa AbstractVector && !isempty(v) && all(x -> x isa Real, v) ?
                              Float64.(v) : v) for (k, v) in blob)
                for (inst, blob) in d["data"])
        end
        push!(out, d)
    end
    return out
end

# What a replay does not hand to `_target_from`: transport, sampling and output.
const _REPLAY_DROP = ("op", "channels", "planets", "priors", "engine", "stopping",
                      "output_dir", "chains_path", "stop_file", "plots",
                      "plot_kwargs", "save_pdf", "science", "output", "transdim",
                      "config_overrides", "gaia_id", "name", "result_summary",
                      "n_draws", "seed", "planet", "confidence", "n_bins",
                      "show_progress", "n_proposal")
const _REPLAY_SYMBOLS = (:parametrization, :time_anchor, :stability, :ttv_backend,
                         :density)

"""
    _replay(p) -> (; chains, target, cfg, outdir, meta, planets)

Chains and rebuilt target of a finished fit; see the block comment above.
"""
function _replay(p)
    haskey2(k) = _haskey2(p, k)
    outdir = haskey2(:output_dir) && _pget(p, :output_dir) !== nothing ?
             String(_pget(p, :output_dir)) : nothing
    chpath = haskey2(:chains_path) && _pget(p, :chains_path) !== nothing ?
             String(_pget(p, :chains_path)) :
             (outdir === nothing ? error("give `output_dir` (a fit's, holding " *
                                         "chains.nc) or `chains_path`") :
              joinpath(outdir, "chains.nc"))
    isfile(chpath) || error("no chains at $chpath; run the fit first")
    outdir === nothing && (outdir = dirname(abspath(chpath)))
    haskey2(:channels) || error("this operation re-evaluates the model: give the " *
        "fit's `channels` (and its `planets`, `priors` and model keywords), as " *
        "`replot` takes them; `result=` does it from a fit this client ran")
    chains, meta = load_chains(chpath)
    channels = _replay_channels(p)
    planets = haskey2(:planets) ? _pget(p, :planets) : 1
    planets isa JSON3.Array && (planets = collect(planets))
    priors = if haskey2(:priors) && _pget(p, :priors) !== nothing
        raw = _plain(_pget(p, :priors))
        isempty(raw) ? nothing :
            Dict{String, PriorSpec}(String(k) => _as_prior(v) for (k, v) in pairs(raw))
    else
        nothing
    end
    if !(planets isa NamedTuple)
        rvi = findfirst(c -> String(get(c, "source", "")) == "RV", channels)
        has_as = any(c -> String(get(c, "source", "")) == "AS", channels)
        if rvi !== nothing && !has_as
            dv = collect(values(get(channels[rvi], "data", Dict())))
            if !isempty(dv)
                allt  = reduce(vcat, [collect(x.t)  for x in dv])
                allrv = reduce(vcat, [collect(x.rv) for x in dv])
                blk = default_rv_planet(allt, allrv)
                planets = length(channels) == 1 ?
                    _planet_spec(planets; block = blk) :
                    _planet_spec(planets; block = (P = blk.P, K = blk.K))
            end
        end
    end
    kw = Dict{Symbol,Any}()
    # Only what builds the target: `_target_from`'s keywords and the star's. The
    # rest of a recorded request (sampling, output, this operation's own
    # arguments) is not the model and is left out.
    model_kw = Set(String.(Base.kwarg_decl(first(methods(_target_from)))))
    union!(model_kw, String.(_STAR_DERIVED_KEYS))
    for (k, v) in pairs(p)
        String(k) in _REPLAY_DROP && continue
        String(k) in model_kw || continue
        v === nothing && continue
        sk = Symbol(k)
        jv = _plain(v)
        kw[sk] = (sk in _REPLAY_SYMBOLS && jv isa AbstractString) ? Symbol(jv) : jv
    end
    if haskey2(:config_overrides) && _pget(p, :config_overrides) !== nothing
        kw = Dict{Symbol,Any}(pairs(_apply_target_overrides(
            _plain(_pget(p, :config_overrides)), (; kw...), "replay")))
    end
    _, kw2 = _split_star_kwargs((; kw...))
    for k in (:gaia_id, :name)
        haskey2(k) && _pget(p, k) !== nothing && (kw2 = merge(kw2, NamedTuple{(k,)}((_pget(p, k),))))
    end
    menu = Any[]
    target = _target_from(channels, planets; priors, menu_out = menu, kw2...)
    eng = haskey2(:engine) && _pget(p, :engine) !== nothing ? _plain(_pget(p, :engine)) : nothing
    ename = eng isa AbstractDict ? String(get(eng, "engine", "pt_emcee")) :
            eng === nothing ? "pt_emcee" : String(eng)
    eopts = eng isa AbstractDict ? get(eng, "options", Dict{String,Any}()) : Dict{String,Any}()
    outc = Dict{String,Any}()
    for k in ("ppc_n_draws", "loo_n_draws", "loo_max_dense_obs", "detection_limits_n_bins",
              "detection_limits_confidence", "detection_limits_planet", "save_pdf")
        haskey2(Symbol(k)) && (outc[k] = _pget(p, Symbol(k)))
    end
    cfg = Dict{String,Any}(
        "output" => outc,
        "seed" => haskey2(:seed) ? Int(_pget(p, :seed)) : 1,
        "sampler" => Dict{String,Any}("name" => ename, "kwargs" => eopts))
    return (; chains, target, cfg, outdir, meta)
end

function act_diag_loo(p)
    r = _replay(p)
    r.cfg["output"]["loo"] = true
    summary = Dict{String,Any}()
    lz = get(r.meta, "log_evidence", NaN)
    res = lz isa Real && lz > -1e299 ? (; log_evidence = Float64(lz)) : (;)
    _run_loo!(r.cfg, r.chains, r.target.params, r.target.data, res, summary)
    return get(summary, "loo", Dict{String,Any}(
        "status" => "skipped", "reason" => "the fit has no RV or photometry"))
end

function act_diag_ppc(p)
    r = _replay(p)
    r.cfg["output"]["ppc"] = true
    summary = Dict{String,Any}()
    mkpath(r.outdir)
    _run_ppc!(r.cfg, r.chains, r.target.params, r.target.data, r.outdir, summary)
    return get(summary, "ppc", Dict{String,Any}(
        "status" => "skipped", "reason" => "the fit has no RV or photometry"))
end

function act_diag_fit_health(p)
    r = _replay(p)
    r.cfg["output"]["fit_health"] = true
    summary = Dict{String,Any}()
    _run_fit_health!(r.cfg, r.chains, r.target.params, summary)
    return get(summary, "fit_health", Dict{String,Any}("status" => "skipped"))
end

function act_detection_limits(p)
    r = _replay(p)
    r.cfg["output"]["detection_limits"] = true
    summary = Dict{String,Any}()
    mkpath(r.outdir)
    _run_detection_limits!(r.cfg, r.chains, r.target.params, r.target.data,
                           r.outdir, summary)
    return get(summary, "detection_limits", Dict{String,Any}(
        "status" => "skipped", "reason" => "the fit has no RV planet column"))
end

"""
`evidence`: the post-fit evidence of a fit, recomputed from its chains: mode-
Laplace and bridge sampling, with the same selection rule as the fit's own
post-fit stage. Returns `{"log_z", "evidence": {...}}`.
"""
function act_evidence(p)
    r = _replay(p)
    lz = get(r.meta, "log_evidence", NaN)
    summary = Dict{String,Any}(
        "log_z" => lz isa Real && lz > -1e299 ? Float64(lz) : NaN,
        "evidence" => Dict{String,Any}("reported" => "none"))
    _augment_evidence!(summary, r.target.params, r.target.data, r.chains;
                       show_progress = false)
    return Dict{String,Any}("log_z" => summary["log_z"], "evidence" => summary["evidence"])
end

# A struct (or a vector of them) as plain JSON values.
_jsonable(x::Symbol) = String(x)
_jsonable(x::Pair) = [_jsonable(x.first), _jsonable(x.second)]
_jsonable(x::AbstractVector) = Any[_jsonable(v) for v in x]
_jsonable(x::AbstractDict) = Dict{String,Any}(string(k) => _jsonable(v) for (k, v) in x)
_jsonable(x::Union{Real,AbstractString,Nothing,Bool}) = x
function _jsonable(x)
    isstructtype(typeof(x)) || return string(x)
    return Dict{String,Any}(String(f) => _jsonable(getfield(x, f))
                            for f in fieldnames(typeof(x)))
end

"""
`diagnostics.label_switching`: slot-permutation entropy of a trans-dim or
multi-planet chain (`label_switching`); needs only the chains. Optional
`reference_periods`, `labels`, `tol`, `min_occupancy`, `period_prefix`.
"""
function act_diag_label_switching(p)
    chains, _ = _load(p)
    kw = Dict{Symbol,Any}()
    for k in (:reference_periods, :labels, :tol, :min_occupancy, :period_prefix)
        (_haskey2(p, k) && _pget(p, k) !== nothing) || continue
        v = _pget(p, k)
        kw[k] = k === :reference_periods ? _vec(v) :
                k === :labels ? String.(collect(v)) :
                k === :period_prefix ? String(v) : Float64(v)
    end
    return _jsonable(label_switching(chains; kw...))
end

"""
`select_planets` of a TRANS-DIM fit's chains: P(N_p = k | data) and the Bayes
factor of each k against the best-supported one (`model_probabilities`,
`bayes_factors`). Fitting each fixed count and comparing log Z is the client's
loop (`Session.select_planets`).
"""
function act_select_planets(p)
    chains, _ = _load(p)
    "n_planets" in String.(names(chains, :parameters)) || error(
        "select_planets reads a trans-dim fit's chains (an `n_planets` column); " *
        "fit with planets=range(...) or run the candidates one by one")
    probs = model_probabilities(chains)
    ks = sort(collect(keys(probs)))
    best = ks[argmax([probs[k] for k in ks])]
    bf = probs[best] > 0 ? bayes_factors(chains; reference = best) : Dict{Int,Float64}()
    rows = [Dict{String,Any}("n_planets" => k, "probability" => probs[k],
                             "bayes_factor_vs_best" => k == best ? 1.0 :
                                 (haskey(bf, k) ? bf[k] : NaN)) for k in ks]
    return Dict{String,Any}("table" => rows, "best" => best)
end

"""
`select_noise` of a trans-dim-noise fit's chains: the fraction of draws each
noise model is active (`noise_active_<model>` columns), most occupied first.
"""
function act_select_noise(p)
    chains, _ = _load(p)
    cols = [String(s) for s in names(chains, :parameters)
            if startswith(String(s), "noise_active_")]
    isempty(cols) && error("select_noise reads a fit with trans-dim noise " *
        "(`noise_active_*` columns); fit with a noise_menu and transdim=...")
    rows = [Dict{String,Any}("model" => c[length("noise_active_")+1:end],
                             "occupancy" => mean(Array(chains[Symbol(c)])))
            for c in cols]
    sort!(rows; by = r -> -r["occupancy"])
    return Dict{String,Any}("table" => rows, "best" => rows[1]["model"])
end

# --- light curves and prewhitening ---------------------------------------------

"""
`detect.lc_periodogram`: a Lomb-Scargle (GLS) periodogram of a light curve
(`gls_periodogram`). Takes `t`, `flux`, `flux_err` and its keywords
(`period_min`, `period_max`, `samples_per_peak`, `max_peaks`, `fap_levels`,
`fap_method`, `n_bootstrap`). The false-alarm probability is analytic unless
asked otherwise: bootstrapping a TESS sector is minutes.
"""
function act_detect_lc_periodogram(p)
    _need(p, :t, :flux, :flux_err)
    kw = _kw(p, ["t", "flux", "flux_err"])
    haskey(kw, :fap_method) || (kw[:fap_method] = :analytic)
    for k in (:fap_method,)
        kw[k] isa AbstractString && (kw[k] = Symbol(kw[k]))
    end
    haskey(kw, :fap_levels) && (kw[:fap_levels] = _vec(kw[:fap_levels]))
    r = gls_periodogram(_vec(_pget(p, :t)), _vec(_pget(p, :flux)),
                        _vec(_pget(p, :flux_err)); kw...)
    return _jsonable(r)
end

"""
`pre_white`: iterative prewhitening of a time series. Each pass takes the
highest GLS peak, fits a sinusoid at that period by weighted least squares
(with a constant) and subtracts it; it stops at `max_signals` or when the best
peak's false-alarm probability exceeds `fap` (default 0.01). Takes `t`, `y`,
`y_err` (or `rv`, `rv_err`), `max_signals` (5), `fap`, `period_min`,
`period_max`. Returns the signals found (period, amplitude, phase, FAP) and the
residual series.
"""
function act_pre_white(p)
    t = _vec(_pget(p, _haskey2(p, :t) ? :t : error("missing required argument t")))
    ykey = _haskey2(p, :y) ? :y : (_haskey2(p, :rv) ? :rv : (_haskey2(p, :flux) ? :flux :
           error("pre_white needs `y` (or `rv`, or `flux`)")))
    ekey = _haskey2(p, :y_err) ? :y_err : (_haskey2(p, :rv_err) ? :rv_err :
           (_haskey2(p, :flux_err) ? :flux_err : error("pre_white needs `y_err`")))
    y = copy(_vec(_pget(p, ykey))); e = _vec(_pget(p, ekey))
    length(t) == length(y) == length(e) || error("t, y and y_err differ in length")
    max_signals = _haskey2(p, :max_signals) ? Int(_pget(p, :max_signals)) : 5
    fap_max = _haskey2(p, :fap) ? Float64(_pget(p, :fap)) : 0.01
    gkw = Dict{Symbol,Any}(:fap_method => :analytic, :fap_levels => [fap_max],
                           :max_peaks => 1)
    for k in (:period_min, :period_max, :samples_per_peak)
        _haskey2(p, k) && _pget(p, k) !== nothing && (gkw[k] = _pget(p, k))
    end
    w = 1 ./ e .^ 2
    signals = Dict{String,Any}[]
    for it in 1:max_signals
        pg = gls_periodogram(t, y, e; gkw...)
        isempty(pg.peaks) && break
        pk = pg.peaks[1]
        fap = Float64(pk.fap)
        isfinite(fap) && fap > fap_max && break
        P = Float64(pk.period)
        X = hcat(ones(length(t)), sin.(2π .* t ./ P), cos.(2π .* t ./ P))
        sw = sqrt.(w)
        β = (X .* sw) \ (y .* sw)
        y .-= X * β
        push!(signals, Dict{String,Any}("iteration" => it, "period" => P,
              "amplitude" => hypot(β[2], β[3]), "phase" => atan(β[3], β[2]),
              "fap" => fap))
    end
    return Dict{String,Any}("signals" => signals, "n_signals" => length(signals),
                            "residuals" => y,
                            "rms" => sqrt(sum(w .* y .^ 2) / sum(w)))
end


# --- registration -------------------------------------------------------------
# Only operations with a working implementation are registered. An action that
# is declared but not implemented is worse than an absent one: the client
# advertises it, the user calls it, and it fails deep in Julia.

"""
    FEATURE_ACTIONS :: Dict{String, Function}

Name-to-implementation table for the standalone feature operations —
detection, detrending, and diagnostics that run outside a fit. Keys are the
dotted action names a job config may request (`"detect.transits"`,
`"detrend.savgol"`, ...); values are the functions that carry them out.

`keys(FEATURE_ACTIONS)` is the authoritative list of available actions.
"""
const FEATURE_ACTIONS = Dict{String, Function}(
    "detect.transits"       => act_detect_transits,
    "detect.rv_planets"     => act_detect_rv_planets,
    "detect.rv_periodogram" => act_detect_rv_planets,
    "detect.rotation"       => act_detect_rotation,
    "detect.segments"       => act_detect_segments,
    "detrend.savgol"        => act_detrend_savgol,
    "detrend.notch"         => act_detrend_notch,
    "detrend.locor"         => act_detrend_locor,
    "detrend.gp"            => act_detrend_gp,
    "diagnostics.ess_rhat"  => act_diag_ess_rhat,
    "diagnostics.loo"       => act_diag_loo,
    "diagnostics.ppc"       => act_diag_ppc,
    "diagnostics.fit_health" => act_diag_fit_health,
    "diagnostics.label_switching" => act_diag_label_switching,
    "detect.lc_periodogram" => act_detect_lc_periodogram,
    "pre_white"             => act_pre_white,
    "select_planets"        => act_select_planets,
    "select_noise"          => act_select_noise,
    "evidence"              => act_evidence,
    "detection_limits"      => act_detection_limits,
    "tomogram.ccf_profile"       => act_tomo_ccf_profile,
    "tomogram.residuals"         => act_tomo_residuals,
    "tomogram.shadow_track"      => act_tomo_shadow_track,
    "tomogram.matched_filter"    => act_tomo_matched_filter,
    "tomogram.null_distribution" => act_tomo_null_distribution,
    "tomogram.injection_test"    => act_tomo_injection_test,
)

"Declared in the client but not implemented here: fail with WHY, not 'unknown'."
const FEATURE_NOT_IMPLEMENTED = Dict{String,String}()
