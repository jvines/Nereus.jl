# PSIS-LOO and WAIC cross-validation, computed by replaying the
# posterior chain through the per-datapoint log-likelihood. Sits next
# to the PT-based evidence stack (`TI+/SS+/H+` from
# [Peña & Jenkins 2026](https://ui.adsabs.harvard.edu/abs/2025arXiv250924870P/abstract))
# as the honest counterweight under model misspecification —
# [Vehtari, Gelman & Gabry 2017](https://ui.adsabs.harvard.edu/abs/2017S&C....27.1413V/abstract).
#
# Restriction (first cut): the log-likelihood per data point is well
# defined for white-noise + ActivityDecorrelation + AR/MA models
# (independence after the sequential noise stage). For CovarianceNoise
# (GP) models the joint covariance breaks pointwise factorisation; we
# emit a warning and skip rather than report a misleading number.

using ParetoSmooth: psis_loo
using Statistics: mean, std, var
using LogExpFunctions: logsumexp

# =====================================================================
# Result struct
# =====================================================================

"""
    LooResult

Output of [`compute_loo`](@ref). Carries Bayesian leave-one-out
(LOO) and widely-applicable information criterion (WAIC) estimates
of the expected log pointwise predictive density (elpd), plus
diagnostics.

Fields:
- `n_draws::Int` — posterior draws used.
- `n_obs::Int` — total data points (RV + photometry).
- `elpd_loo::Float64` — PSIS-LOO elpd estimate (higher = better).
- `se_elpd_loo::Float64` — standard error of `elpd_loo`.
- `p_loo::Float64` — effective number of parameters from LOO.
- `elpd_waic::Float64` — WAIC elpd estimate.
- `se_elpd_waic::Float64` — standard error of `elpd_waic`.
- `p_waic::Float64` — effective number of parameters from WAIC.
- `pareto_k::Vector{Float64}` — per-point Pareto k̂ diagnostic.
- `pareto_k_max::Float64` — `maximum(pareto_k)`.
- `pareto_k_warn_count::Int` — points with k̂ > 0.7 (PSIS-LOO unreliable).
- `loo_compare_log_z::Union{Nothing, Float64}` — `elpd_loo` minus the
  PT log-evidence, when `log_z` is passed in. Positive means LOO
  prefers the model more than PT log Z does.
"""
struct LooResult
    n_draws::Int
    n_obs::Int
    elpd_loo::Float64
    se_elpd_loo::Float64
    p_loo::Float64
    elpd_waic::Float64
    se_elpd_waic::Float64
    p_waic::Float64
    pareto_k::Vector{Float64}
    pareto_k_max::Float64
    pareto_k_warn_count::Int
    loo_compare_log_z::Union{Nothing, Float64}
end

# =====================================================================
# Top-level API
# =====================================================================

"""
    compute_loo(chains, params, data;
                n_draws=500, rng=default_rng(),
                log_z=nothing) -> LooResult

Compute PSIS-LOO and WAIC by replaying `n_draws` posterior samples
through the per-datapoint log-likelihood and feeding the resulting
matrix to `ParetoSmooth.psis_loo`. If `log_z` (e.g., the PT
log-evidence) is passed, `loo_compare_log_z = elpd_loo - log_z` is
also reported.

**Restriction.** Pointwise log-likelihood is only well-defined under
independence after the sequential noise stage. Nereus GP / celerite
kernels (`CeleriteSHO`, `CeleriteRotation`, `CeleriteRotationFM17`)
introduce a non-diagonal covariance that breaks pointwise
factorisation. If any active noise model is a `CovarianceNoise`,
this function throws an `ArgumentError`. AR/MA / activity-jitter /
activity-decorrelation are supported.
"""
function compute_loo(chains, params::Params, data::Data;
                      n_draws::Int = 500,
                      rng::AbstractRNG = default_rng(),
                      log_z::Union{Nothing, Real} = nothing)
    n_rv_obs = n_rv(data)
    n_pm_obs = n_phot(data)
    n_obs_total = n_rv_obs + n_pm_obs
    n_obs_total > 0 ||
        throw(ArgumentError("data has no RV or photometry"))

    # Decide between the diagonal-noise pointwise path and the
    # ActivityGP-conditional path. Other CovarianceNoise models
    # (CeleriteRotation / SHO / FM17) need their own per-point LOO
    # formula (closed-form available, but not implemented) — refused
    # for now.
    agp_marg = nothing
    for nm in params.config.noise_models
        if nm isa ActivityGP && nm.marginalize_indicators
            agp_marg = nm
            continue
        end
        if nm isa CovarianceNoise
            throw(ArgumentError(
                "compute_loo: model contains a CovarianceNoise " *
                "(`$(typeof(nm).name.name)`). PSIS-LOO supports the " *
                "ActivityGP path only when " *
                "`marginalize_indicators = true` (RV-conditional " *
                "Gaussian with closed-form Sundararajan-Keerthi LOO). " *
                "For other GP kernels, use PT log Z (TI+/SS+/H+) or " *
                "strip the GP and refit."))
        end
    end

    chains_flat, n_total = _flatten_chains(chains)
    n_draws_eff = min(n_draws, n_total)
    n_draws_eff > 0 || throw(ArgumentError("chain is empty"))
    idx_pool = randperm(rng, n_total)[1:n_draws_eff]

    # Per-point log-likelihood matrix (n_obs × n_draws).
    log_L = Matrix{Float64}(undef, n_obs_total, n_draws_eff)

    for (d, idx) in enumerate(idx_pool)
        theta = _theta_from_row(chains_flat, idx, params)
        offset = 0
        if n_rv_obs > 0
            rv_ll = agp_marg === nothing ?
                _pointwise_loglik_rv(theta, data) :
                _pointwise_loglik_rv_agp_marg(theta, data, agp_marg)
            @inbounds for i in 1:n_rv_obs
                log_L[i, d] = rv_ll[i]
            end
            offset = n_rv_obs
        end
        if n_pm_obs > 0
            pm_ll = _pointwise_loglik_phot(theta, data)
            @inbounds for i in 1:n_pm_obs
                log_L[offset + i, d] = pm_ll[i]
            end
        end
    end

    # PSIS-LOO via ParetoSmooth. The 3D form takes (n_obs, n_samples,
    # n_chains); we treat the flattened chain as 1 chain.
    log_L3 = reshape(log_L, n_obs_total, n_draws_eff, 1)
    # Note: ParetoSmooth defaults to source=:mcmc which uses autocorrelation-
    # aware r_eff. For our randomly-permuted subsample of a (possibly)
    # autocorrelated chain, this is a reasonable conservative default.
    loo = psis_loo(log_L3)

    # ParetoSmooth's `estimates` keyed array has rows :cv_elpd, :naive_lpd,
    # :p_eff and columns :total, :se_total, ... — pull what we need.
    elpd_loo     = Float64(loo.estimates(:cv_elpd, :total))
    se_elpd_loo  = Float64(loo.estimates(:cv_elpd, :se_total))
    p_loo        = Float64(loo.estimates(:p_eff, :total))
    pareto_k     = Float64.(loo.psis_object.pareto_k)
    pareto_k_max = maximum(pareto_k)
    pareto_k_warn = count(>(0.7), pareto_k)

    # WAIC from the same matrix.
    elpd_waic, se_elpd_waic, p_waic = _waic(log_L)

    loo_minus_logz = log_z === nothing ? nothing :
                      elpd_loo - Float64(log_z)

    return LooResult(
        n_draws_eff, n_obs_total,
        elpd_loo, se_elpd_loo, p_loo,
        elpd_waic, se_elpd_waic, p_waic,
        pareto_k, pareto_k_max, pareto_k_warn,
        loo_minus_logz,
    )
end

# =====================================================================
# Per-point log-likelihood helpers
# =====================================================================

# RV channel: stage-1 mean (with optional AR on predictions), then
# stage-2 MA on residuals, then per-point white-noise log L using the
# variances returned by rv_predictions (which already include σ_inst²
# and any ActivityJitter contributions).
function _pointwise_loglik_rv(theta::Theta{Float64}, data::Data)
    preds, vars = rv_predictions(theta, data)
    n = length(data.rv)
    # Apply AR on a writable copy of predictions; result still feeds
    # into `residuals = data - preds_after_AR`.
    preds_mut = collect(preds)
    noise_models = theta.params.config.noise_models
    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa ARModel && nm.channel === :rv
            apply_ar!(preds_mut, data.t_rv, data.rv_inst, theta, nm)
        end
    end
    residuals = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        residuals[i] = data.rv[i] - preds_mut[i]
    end
    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa MAModel && nm.channel === :rv
            apply_ma!(residuals, data.t_rv, data.rv_inst, theta, nm)
        end
    end
    return _gaussian_logpdf_per_point(residuals, vars)
end

# Photometry channel: same skeleton, channel === :phot for AR/MA.
function _pointwise_loglik_phot(theta::Theta{Float64}, data::Data)
    preds, vars = phot_predictions(theta, data)
    n = length(data.flux)
    preds_mut = collect(preds)
    noise_models = theta.params.config.noise_models
    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa ARModel && nm.channel === :phot
            apply_ar!(preds_mut, data.t_phot, data.phot_inst, theta, nm)
        end
    end
    residuals = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        residuals[i] = data.flux[i] - preds_mut[i]
    end
    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa MAModel && nm.channel === :phot
            apply_ma!(residuals, data.t_phot, data.phot_inst, theta, nm)
        end
    end
    return _gaussian_logpdf_per_point(residuals, vars)
end

# Per-RV-point log p(y_i | y_{-i}, θ) for the ActivityGP-conditional
# Gaussian (marginalize_indicators = true) via the closed-form
# Sundararajan & Keerthi (2001) leave-one-out formula. For a
# multivariate Gaussian y ~ N(μ, Σ), the leave-one-out predictive
# at point i is N(y_i; μ_i − r_i/d_i, 1/d_i) with
#   r = Σ⁻¹ (y − μ),   d_i = (Σ⁻¹)_{ii}
# so log p(y_i | y_{-i}, θ) = 0.5 log(d_i) − 0.5 log(2π) − 0.5 r_i² / d_i.
# Both `r` and `diag(Σ⁻¹)` come straight out of the Cholesky of Σ.
function _pointwise_loglik_rv_agp_marg(theta::Theta{Float64}, data::Data,
                                         agp::ActivityGP)
    layout = theta.params.layout
    s = _gp_suffix(agp)
    # Unit-variance G(t) (Rajpaul+ 2015); guarded lookup for legacy chains.
    amp_idx = get(layout.name_to_idx, "gp_act_amp$s", 0)
    amp = amp_idx == 0 ? 1.0 : theta.values[amp_idx]
    P   = theta.values[layout.name_to_idx["gp_act_period$s"]]
    λe  = theta.values[layout.name_to_idx["gp_act_lambda_e$s"]]
    λp  = theta.values[layout.name_to_idx["gp_act_lambda_p$s"]]
    n_rv_obs = length(data.t_rv)
    if !(amp > 0 && P > 0 && λe > 0 && λp > 0)
        return fill(-Inf, n_rv_obs)
    end

    # Derivative couplings sampled as amplitudes — physical Ġ
    # coefficient = amplitude / std(Ġ) (mirrors _activity_gp_joint_ll).
    inv_sdG = 1 / sqrt(1 / (λe * λe) + π * π / (P * P * λp * λp))
    Vc = theta.values[layout.name_to_idx["Vc$s"]]
    Vr = agp.use_derivative ?
         theta.values[layout.name_to_idx["Vr$s"]] * inv_sdG : 0.0

    # Build joint Σ at observations (same path as _activity_gp_joint_ll).
    ind_meta = Tuple{Symbol, Vector{Float64}, Vector{Float64}, Float64, Float64, Float64}[]
    n_obs_total = n_rv_obs
    for ch in agp.channels
        ch === :rv && continue
        name = String(ch)
        haskey(data.indicators, name) || return fill(-Inf, n_rv_obs)
        haskey(data.indicator_errs, name) || return fill(-Inf, n_rv_obs)
        vals = data.indicators[name]; errs = data.indicator_errs[name]
        cg, cd = _ACTIVITY_GP_COEFFS[ch]
        a_coef = theta.values[layout.name_to_idx[string(cg, s)]]
        b_coef = (cd === nothing || !agp.use_derivative) ? 0.0 :
                  theta.values[layout.name_to_idx[string(cd, s)]] * inv_sdG
        jit_idx = get(layout.name_to_idx, "gp_act_jit_$(ch)$s", 0)
        jit² = jit_idx == 0 ? 0.0 : Float64(theta.values[jit_idx])^2
        push!(ind_meta, (ch, vals, errs, a_coef, b_coef, jit²))
        n_obs_total += length(vals)
    end

    preds, vars = rv_predictions(theta, data)
    t_obs = Vector{Float64}(undef, n_obs_total)
    y_obs = Vector{Float64}(undef, n_obs_total)
    σ²_obs = Vector{Float64}(undef, n_obs_total)
    a_obs = Vector{Float64}(undef, n_obs_total)
    b_obs = Vector{Float64}(undef, n_obs_total)
    @inbounds for i in 1:n_rv_obs
        t_obs[i]  = data.t_rv[i]; y_obs[i] = data.rv[i] - preds[i]
        σ²_obs[i] = vars[i]; a_obs[i] = Vc; b_obs[i] = Vr
    end
    offset = n_rv_obs
    for (_, vals, errs, a_coef, b_coef, jit²) in ind_meta
        n_ch = length(vals)
        @inbounds for i in 1:n_ch
            t_obs[offset + i]  = data.t_rv[i]; y_obs[offset + i] = vals[i]
            σ²_obs[offset + i] = errs[i]^2 + jit²
            a_obs[offset + i]  = a_coef;  b_obs[offset + i] = b_coef
        end
        offset += n_ch
    end
    channel_obs = vcat(fill(:rv, n_rv_obs),
                        vcat(fill.(getindex.(ind_meta, 1),
                                     length.(getindex.(ind_meta, 2)))...))
    Σ = activity_gp_covariance(t_obs, channel_obs, a_obs, b_obs, amp, P, λe, λp)
    @inbounds for i in 1:n_obs_total
        Σ[i, i] += σ²_obs[i]
    end

    # Condition: Σ_R|I = Σ_RR − Σ_RI · Σ_II⁻¹ · Σ_IR; μ_R|I from y_I.
    Σ_RR = Matrix(view(Σ, 1:n_rv_obs, 1:n_rv_obs))
    Σ_RI = Matrix(view(Σ, 1:n_rv_obs, (n_rv_obs + 1):n_obs_total))
    Σ_II = Matrix(view(Σ, (n_rv_obs + 1):n_obs_total, (n_rv_obs + 1):n_obs_total))
    y_R  = view(y_obs, 1:n_rv_obs)
    y_I  = view(y_obs, (n_rv_obs + 1):n_obs_total)

    F_II = cholesky(Symmetric(Σ_II); check = false)
    issuccess(F_II) || return fill(-Inf, n_rv_obs)
    μ_cond = Σ_RI * (F_II \ y_I)
    Σ_cond = Symmetric(Σ_RR .- Σ_RI * (F_II \ transpose(Σ_RI)))

    F_R = cholesky(Σ_cond; check = false)
    issuccess(F_R) || return fill(-Inf, n_rv_obs)

    # Sundararajan-Keerthi: need r = Σ_cond⁻¹ (y_R − μ_cond) and
    # diag(Σ_cond⁻¹). Both come from the Cholesky factor.
    r_vec = F_R \ (y_R .- μ_cond)
    # diag of inverse via Σ⁻¹ = L⁻ᵀ L⁻¹, so diag(Σ⁻¹)_i = ||L⁻ᵀ e_i||² = ||L⁻¹ e_i_perm||².
    # Simpler: invert in-place (n_rv ≲ 1000 makes this fine).
    Σ_inv = inv(F_R)
    d_diag = Vector{Float64}(undef, n_rv_obs)
    @inbounds for i in 1:n_rv_obs
        d_diag[i] = Σ_inv[i, i]
    end

    ll = Vector{Float64}(undef, n_rv_obs)
    half_log_2π = 0.5 * log(2π)
    @inbounds for i in 1:n_rv_obs
        d_i = d_diag[i]
        d_i > 0 || (ll[i] = -Inf; continue)
        ll[i] = 0.5 * log(d_i) - half_log_2π -
                 0.5 * r_vec[i]^2 / d_i
    end
    return ll
end

@inline function _gaussian_logpdf_per_point(r::AbstractVector{<:Real},
                                              σ²::AbstractVector{<:Real})
    n = length(r)
    out = Vector{Float64}(undef, n)
    two_pi = 2π
    @inbounds for i in 1:n
        out[i] = -0.5 * (log(two_pi * σ²[i]) + r[i]^2 / σ²[i])
    end
    return out
end

# =====================================================================
# WAIC (Vehtari+ 2017 eq. 11 / 12)
# =====================================================================

function _waic(log_L::AbstractMatrix{<:Real})
    n_obs, n_draws = size(log_L)
    # lppd_i = log mean(exp(log_L[i, :]))
    lppd = Vector{Float64}(undef, n_obs)
    p_waic_per_i = Vector{Float64}(undef, n_obs)
    @inbounds for i in 1:n_obs
        row = view(log_L, i, :)
        lppd[i] = logsumexp(row) - log(n_draws)
        p_waic_per_i[i] = var(row; corrected = true)
    end
    elpd_per_i = lppd .- p_waic_per_i
    elpd = sum(elpd_per_i)
    p   = sum(p_waic_per_i)
    se  = sqrt(n_obs * var(elpd_per_i; corrected = true))
    return (elpd, se, p)
end
