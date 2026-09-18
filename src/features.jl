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
    _need(p, :chains_path)
    chains, meta = load_chains(String(_pget(p, :chains_path)))
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
    "tomogram.ccf_profile"       => act_tomo_ccf_profile,
    "tomogram.residuals"         => act_tomo_residuals,
    "tomogram.shadow_track"      => act_tomo_shadow_track,
    "tomogram.matched_filter"    => act_tomo_matched_filter,
    "tomogram.null_distribution" => act_tomo_null_distribution,
    "tomogram.injection_test"    => act_tomo_injection_test,
)

"Declared in the client but not implemented here — fail with WHY, not 'unknown'."
const FEATURE_NOT_IMPLEMENTED = Dict{String,String}(
    "detrend.gp"                 => "needs a CovarianceNoise model; use the noise menu",
    "detect.lc_periodogram"      => "not yet wired",
    "diagnostics.loo"            => "needs the Params/Data of the originating fit",
    "diagnostics.ppc"            => "needs the originating fit",
    "diagnostics.label_switching"=> "not yet wired",
    "diagnostics.fit_health"     => "needs the originating fit",
    "pre_white"                  => "not yet wired",
    "select_planets"             => "use a trans-dim engine (moms, rjmcmc, transdim_ptemcee)",
    "select_noise"               => "use the noise menu with a trans-dim engine",
    "evidence"                   => "returned as log_z by every fit_* call",
    "detection_limits"           => "needs the originating fit's Params",
)
