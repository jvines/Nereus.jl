# PSIS-LOO and WAIC cross-validation, computed by replaying the
# posterior chain through the per-datapoint log-likelihood. Sits next
# to the PT-based evidence stack (`TI+/SS+/H+` from
# [Peña & Jenkins 2026](https://ui.adsabs.harvard.edu/abs/2025arXiv250924870P/abstract))
# as the honest counterweight under model misspecification —
# [Vehtari, Gelman & Gabry 2017](https://ui.adsabs.harvard.edu/abs/2017S&C....27.1413V/abstract).
#
# The pointwise term is the leave-one-out predictive density at fixed θ,
# log p(y_i | y_{-i}, θ), which PSIS then reweights over the posterior draws.
# With independent points (white noise, Student-t, mean modifiers, AR on the
# model) that is the density of point i alone. A noise model that couples the
# points -- a GP, an additive covariance (HarmonicBlock, NightlyOffset), MA on
# the residuals, the analytic γ marginalization, the ActivityGP conditional --
# makes the residuals a correlated Gaussian r ~ N(0, C), and the density of
# point i alone is then NOT the leave-one-out predictive (it ignores what the
# other points say about it). For a Gaussian the predictive is closed form
# (Sundararajan & Keerthi 2001): with Q = C⁻¹, g = Q r and d = diag(Q),
#
#     y_i | y_{-i}, θ  ~  N(r_i − g_i/d_i, 1/d_i)        (in residual units)
#     log p(y_i | y_{-i}, θ) = ½ log d_i − ½ log 2π − ½ g_i²/d_i.
#
# C is rebuilt here from the noise models active in each draw, and whether a
# channel is diagonal is read off that covariance, not off a list of noise
# types. The rebuilt covariance must also reproduce the likelihood the fit
# scored: its Gaussian log-density is checked against the likelihood itself on
# every draw. A noise model this file does not know therefore makes the check
# fail and LOO refuses, rather than scoring correlated points as independent.
#
# The joint (non-marginalized) ActivityGP, which scores the RV together with
# the activity indicators, is refused: use `marginalize_indicators = true`
# (the conditional log p(RV | indicators)) or PT log Z.

using ParetoSmooth: psis_loo
using Statistics: mean, std, var
using LogExpFunctions: logsumexp
using LinearAlgebra: cholesky, cholesky!, Symmetric, issuccess, logdet, dot,
                     Diagonal

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
                log_z=nothing, max_dense_obs=2000) -> LooResult

Compute PSIS-LOO and WAIC by replaying `n_draws` posterior samples
through the per-datapoint leave-one-out log predictive density and
feeding the resulting matrix to `ParetoSmooth.psis_loo`. If `log_z`
(e.g., the PT log-evidence) is passed, `loo_compare_log_z = elpd_loo -
log_z` is also reported.

**Correlated noise.** The pointwise term is log p(y_i | y_{-i}, θ). For
independent points (white noise, `StudentT`, `ActivityJitter`,
`ErrorScale`, `ActivityDecorrelation`, `ARModel`, which acts on the
model) it is the density of point i alone. Under any noise model that
couples points -- a celerite or Matérn GP (global or per-instrument),
`HarmonicBlock`, `NightlyOffset`, `MAModel` (the residual transform
makes the residuals a correlated Gaussian), the analytic γ
marginalization, or an `ActivityGP` with `marginalize_indicators = true`
-- it is the exact Gaussian leave-one-out predictive from the full
covariance C of that draw (Sundararajan & Keerthi 2001): mean
`r_i − (C⁻¹r)_i / (C⁻¹)_ii` and variance `1 / (C⁻¹)_ii` in residual
units. Whether a channel is diagonal is decided from the covariance
itself, and each draw's covariance is checked against the likelihood
the fit scored; if it does not reproduce it (a noise model LOO does not
know), this throws an `ArgumentError` rather than reporting a wrong
elpd.

A white base with additive low-rank terms is solved by Woodbury in
O(n·k²). Anything with a GP or MA takes a dense n×n solve per draw
(instrument-restricted GPs one block per GP), O(n³); a block larger
than `max_dense_obs` points throws an `ArgumentError` instead of running
for hours -- raise it if that cost is acceptable.

**Refused.** An `ActivityGP` scored jointly with its indicators
(`marginalize_indicators = false`), an instrument-restricted or more
than one active `ActivityGP`, `StudentT` combined with `MAModel`, a
γ-marginalized instrument group with a single point (its leave-one-out
predictive is improper), and a draw whose likelihood is not finite
throw an `ArgumentError`.
"""
function compute_loo(chains, params::Params, data::Data;
                      n_draws::Int = 500,
                      rng::AbstractRNG = default_rng(),
                      log_z::Union{Nothing, Real} = nothing,
                      max_dense_obs::Int = 2000)
    n_rv_obs = n_rv(data)
    n_pm_obs = n_phot(data)
    n_obs_total = n_rv_obs + n_pm_obs
    n_obs_total > 0 ||
        throw(ArgumentError("data has no RV or photometry"))

    chains_flat, n_total = _flatten_chains(chains)
    n_draws_eff = min(n_draws, n_total)
    n_draws_eff > 0 || throw(ArgumentError("chain is empty"))
    idx_pool = randperm(rng, n_total)[1:n_draws_eff]

    # Per-point log-likelihood matrix (n_obs × n_draws).
    log_L = Matrix{Float64}(undef, n_obs_total, n_draws_eff)

    tdc = _td_cols(chains_flat, params)      # each draw with its own live slots
    for (d, idx) in enumerate(idx_pool)
        theta = _theta_from_row(chains_flat, idx, params; tdc = tdc)
        offset = 0
        if n_rv_obs > 0
            rv_ll = _pointwise_loo_rv(theta, data, max_dense_obs)
            @inbounds for i in 1:n_rv_obs
                log_L[i, d] = rv_ll[i]
            end
            offset = n_rv_obs
        end
        if n_pm_obs > 0
            pm_ll = _pointwise_loo_phot(theta, data, max_dense_obs)
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
# Per-point leave-one-out log predictive density
# =====================================================================

# RV channel. The mean is the stage-1 model with AR applied to it, as in the
# likelihood; `ref` is what the likelihood scores for the RV data, against
# which every rebuilt covariance is checked.
function _pointwise_loo_rv(theta::Theta{Float64}, data::Data, max_dense::Int)
    n = length(data.rv)
    noise_models = theta.params.config.noise_models
    preds, vars = rv_predictions(theta, data)
    preds = collect(preds)
    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa ARModel && nm.channel === :rv
            apply_ar!(preds, data.t_rv, data.rv_inst, theta, nm)
        end
    end
    # ActivityGP. An indicators-only one adds log p(y_I | θ), a separate
    # factor, and leaves the RV to the channel below; any other one scores
    # the RV itself, jointly with the indicators (see `_apply_noise_and_eval`).
    agp = nothing
    for (nm_idx, nm) in enumerate(noise_models)
        (nm isa ActivityGP && is_noise_model_active(theta, nm_idx)) || continue
        nm.indicators_only && continue
        agp === nothing || throw(ArgumentError(
            "compute_loo: more than one active ActivityGP; LOO supports a " *
            "single global ActivityGP with `marginalize_indicators = true`."))
        agp = nm
    end
    if agp !== nothing
        agp.marginalize_indicators || throw(ArgumentError(
            "compute_loo: the active ActivityGP scores the RV jointly with the " *
            "activity indicators (`marginalize_indicators = false`), so the " *
            "data points of that model are not the RV points LOO compares. Use " *
            "`marginalize_indicators = true` (log p(RV | indicators), scored " *
            "here by its exact Gaussian leave-one-out) or PT log Z (TI+/SS+/H+)."))
        isempty(agp.instruments) || throw(ArgumentError(
            "compute_loo: an instrument-restricted ActivityGP is not supported; " *
            "use a global one (`instruments = String[]`)."))
    end

    ref = _rv_log_likelihood_core(theta, data)
    _finite_draw(ref, :rv)
    for (nm_idx, nm) in enumerate(noise_models)
        (nm isa ActivityGP && nm.indicators_only &&
         is_noise_model_active(theta, nm_idx)) || continue
        ref -= _activity_gp_joint_ll(theta, data, preds, vars, nm)
    end

    if agp !== nothing
        g = _agp_conditional_gaussian(theta, data, preds, vars, agp)
        g === nothing && _finite_draw(-Inf, :rv)
        r_c, C_c = g
        _check_dense_size(n, max_dense, :rv)
        ll, joint = _dense_gaussian_loo(r_c, C_c, nothing)
        _loo_check(joint, ref, :rv)
        return ll
    end

    if theta.params.config.parametrization.marginalize_gamma
        # γ is integrated out of the likelihood, not added to the mean.
        @inbounds for i in 1:n
            preds[i] -= rv_gamma(theta, data.rv_inst[i])
        end
        return _gamma_marginal_loo(theta, data, data.rv .- preds, vars, ref)
    end

    return _channel_loo(theta, data.rv .- preds, vars, data.t_rv, data.rv_inst,
                        :rv, ref, max_dense)
end

# Photometry channel: the same, against the transit likelihood.
function _pointwise_loo_phot(theta::Theta{Float64}, data::Data, max_dense::Int)
    preds, vars = phot_predictions(theta, data)
    preds = collect(preds)
    for (nm_idx, nm) in enumerate(theta.params.config.noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa ARModel && nm.channel === :phot
            apply_ar!(preds, data.t_phot, data.phot_inst, theta, nm)
        end
    end
    ref = transit_log_likelihood(theta, data)
    _finite_draw(ref, :phot)
    return _channel_loo(theta, data.flux .- preds, vars, data.t_phot, data.phot_inst,
                        :phot, ref, max_dense)
end

# Leave-one-out log predictive density of each residual `r` on `channel`,
# under the covariance the channel's active noise models give it -- the one
# `_eval_channel_likelihood` scores, after the MA transform of
# `_apply_noise_and_eval` / the transit likelihood.
function _channel_loo(theta::Theta{Float64}, r::Vector{Float64}, vars::Vector{Float64},
                      t::Vector{Float64}, inst::Vector{Int}, channel::Symbol,
                      ref::Float64, max_dense::Int)
    n = length(r)
    config = theta.params.config
    inst_names = channel === :rv ? config.instruments.rv_names :
                                   config.instruments.pm_names
    ma  = MAModel[]
    gps = NoiseModel[]
    add = AdditiveCovariance[]
    for (nm_idx, nm) in enumerate(config.noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        noise_channel(nm) === channel || continue
        if nm isa MAModel
            push!(ma, nm)
        elseif nm isa CovarianceNoise && !(nm isa ActivityGP)
            push!(gps, nm)
        elseif nm isa AdditiveCovariance
            push!(add, nm)
        end
    end

    # Student-t: independent points, so the pointwise density is exact. With
    # a covariance the likelihood is -Inf (caught above); with MA the points
    # are coupled and not Gaussian.
    st = _active_studentt(theta, config.noise_models, channel)
    if st !== nothing
        isempty(ma) || throw(ArgumentError(
            "compute_loo: StudentT with MAModel on :$channel couples the points " *
            "under a non-Gaussian likelihood; its leave-one-out predictive has no " *
            "closed form."))
        ν = theta.values[theta.params.layout.name_to_idx["studentt_nu"]]
        ll = _studentt_logpdf_per_point(r, vars, ν)
        _loo_check(sum(ll), ref, channel)
        return ll
    end

    F = isempty(add) ? zeros(n, 0) :
        Matrix(_additive_factor(add, theta, t, inst, inst_names))

    # White base, no MA: C = diag(vars) + F Fᵀ.
    if isempty(gps) && isempty(ma)
        if size(F, 2) == 0                       # diagonal: independent points
            ll = _gaussian_logpdf_per_point(r, vars)
            _loo_check(sum(ll), ref, channel)
            return ll
        end
        ll, joint = _lowrank_gaussian_loo(r, vars, F)
        _loo_check(joint, ref, channel)
        return ll
    end

    sels = [_gp_points(nm, inst, inst_names) for nm in gps]

    # Instrument-restricted GPs only: C is block diagonal, one block per GP
    # and the uncovered points alone.
    if isempty(ma) && size(F, 2) == 0 && all(s -> s !== nothing, sels)
        ll = _gaussian_logpdf_per_point(r, vars)
        joint = 0.0
        covered = falses(n)
        for (nm, sel) in zip(gps, sels)
            isempty(sel) && continue
            any(view(covered, sel)) && throw(ArgumentError(
                "compute_loo: two active GPs on :$channel cover the same points."))
            _check_dense_size(length(sel), max_dense, channel)
            C = Matrix(Diagonal(vars[sel]))
            _add_gp_cov!(C, theta, nm, t[sel])
            llb, jb = _dense_gaussian_loo(r[sel], C, nothing)
            ll[sel] = llb
            joint += jb
            covered[sel] .= true
        end
        @inbounds for i in 1:n
            covered[i] || (joint += ll[i])
        end
        _loo_check(joint, ref, channel)
        return ll
    end

    # General case: one dense covariance over the channel.
    _check_dense_size(n, max_dense, channel)
    C = Matrix(Diagonal(vars))
    for (nm, sel) in zip(gps, sels)
        if sel === nothing
            _add_gp_cov!(C, theta, nm, t)
        else
            Cs = zeros(length(sel), length(sel))
            _add_gp_cov!(Cs, theta, nm, t[sel])
            C[sel, sel] .+= Cs
        end
    end
    size(F, 2) > 0 && (C .+= F * F')
    B = isempty(ma) ? nothing : _ma_matrix(theta, ma, t, inst)
    ll, joint = _dense_gaussian_loo(r, C, B)
    _loo_check(joint, ref, channel)
    return ll
end

# Points of the channel a GP covers: `nothing` for a global GP (all of them),
# else those of its instruments (in data order).
function _gp_points(nm::NoiseModel, inst::Vector{Int}, inst_names::Vector{String})
    insts = noise_instruments(nm)
    isempty(insts) && return nothing
    idxs = Int[]
    for name in insts
        k = findfirst(==(name), inst_names)
        k === nothing || push!(idxs, k)
    end
    return [i for i in eachindex(inst) if inst[i] in idxs]
end

# Add the GP kernel matrix over times `t` to `C`.
function _add_gp_cov!(C::Matrix{Float64}, theta::Theta{Float64},
                      nm::Union{CeleriteSHO, CeleriteRotation, CeleriteRotationFM17},
                      t::AbstractVector{Float64})
    ar, cr, ac, bc, cc, dc = _celerite_coeffs(theta, nm)
    n = length(t)
    @inbounds for j in 1:n, i in 1:n
        τ = abs(t[i] - t[j])
        k = 0.0
        for q in eachindex(ar)
            k += ar[q] * exp(-cr[q] * τ)
        end
        for q in eachindex(ac)
            k += exp(-cc[q] * τ) * (ac[q] * cos(dc[q] * τ) + bc[q] * sin(dc[q] * τ))
        end
        C[i, j] += k
    end
    return C
end

function _add_gp_cov!(C::Matrix{Float64}, theta::Theta{Float64}, nm::MaternGP,
                      t::AbstractVector{Float64})
    σ, ρ = _matern_params(theta, nm)
    kern = SSMatern32(σ, ρ)
    n = length(t)
    @inbounds for j in 1:n, i in 1:n
        C[i, j] += _kfuncs(kern, abs(t[i] - t[j]))[1]
    end
    return C
end

_add_gp_cov!(C, theta, nm, t) = throw(ArgumentError(
    "compute_loo: no covariance is known for the noise model $(typeof(nm)), so " *
    "its exact leave-one-out cannot be computed; refusing rather than scoring " *
    "its points as independent."))

# The MA transform ε = B r of the active MA models on a channel, as a dense
# matrix: it is linear in r and unit lower triangular in time order (each
# point minus damped earlier ones), so det B = 1 and r ~ N(0, B⁻¹ C B⁻ᵀ).
function _ma_matrix(theta::Theta{Float64}, ma::Vector{MAModel}, t::Vector{Float64},
                    inst::Vector{Int})
    n = length(t)
    B = zeros(n, n)
    e = zeros(n)
    for k in 1:n
        fill!(e, 0.0); e[k] = 1.0
        for nm in ma
            apply_ma!(e, t, inst, theta, nm)
        end
        B[:, k] .= e
    end
    return B
end

# A posterior draw the likelihood gives zero probability has no leave-one-out
# predictive; the chain is not a posterior of this model and data.
_finite_draw(ref::Float64, channel::Symbol) =
    isfinite(ref) || throw(ArgumentError(
        "compute_loo: a draw has a non-finite :$channel log-likelihood ($ref); " *
        "the chain is not a posterior sample of this model and data."))

_check_dense_size(n::Int, max_dense::Int, channel::Symbol) =
    n <= max_dense || throw(ArgumentError(
        "compute_loo: the exact leave-one-out under correlated noise on :$channel " *
        "needs a dense $n × $n solve per draw, more than `max_dense_obs = " *
        "$max_dense`. Pass a larger `max_dense_obs` if that cost is acceptable."))

# The rebuilt covariance must reproduce what the likelihood scored. A
# mismatch means a noise model contributes something this file does not
# rebuild; the pointwise values would then be wrong, so refuse.
function _loo_check(joint::Float64, ref::Float64, channel::Symbol)
    abs(joint - ref) <= 1e-6 * max(1.0, abs(ref)) && return nothing
    throw(ArgumentError(
        "compute_loo: the covariance rebuilt for the :$channel data has " *
        "log-density $joint, but the likelihood scores $ref. Some active noise " *
        "model is scored in a way LOO does not reproduce; refusing rather than " *
        "reporting a wrong elpd."))
end

# Exact Gaussian leave-one-out for r ~ N(0, B⁻¹ C B⁻ᵀ) -- the residuals after
# a unit-triangular transform B (MA; `nothing` = identity) are N(0, C). With
# C = L Lᵀ and Z = L⁻¹B, the precision is Q = Zᵀ Z, so diag(Q) is the column
# sums of Z.^2 and Q r = Zᵀ (L⁻¹ B r). Returns the per-point log predictive
# densities and the joint log-density (for the check).
function _dense_gaussian_loo(r::AbstractVector{Float64}, C::Matrix{Float64},
                             B::Union{Nothing, Matrix{Float64}})
    n = length(r)
    F = cholesky!(Symmetric(C); check = false)
    issuccess(F) || throw(ArgumentError(
        "compute_loo: the rebuilt covariance is not positive definite in double " *
        "precision, though the likelihood was finite; cannot compute its " *
        "leave-one-out."))
    L = F.L
    w = L \ (B === nothing ? r : B * r)
    joint = -0.5 * (dot(w, w) + logdet(F) + n * log(2π))
    Z = B === nothing ? Matrix(inv(L)) : Matrix(L \ B)
    g = Z' * w
    ll = Vector{Float64}(undef, n)
    half_log_2π = 0.5 * log(2π)
    @inbounds for i in 1:n
        d_i = 0.0
        for k in 1:n
            d_i += Z[k, i]^2
        end
        ll[i] = 0.5 * log(d_i) - half_log_2π - 0.5 * g[i]^2 / d_i
    end
    return ll, joint
end

# The same for C = diag(v) + F Fᵀ, by Woodbury in O(n k²): with D = diag(v),
# W = D⁻¹F and M = I + Fᵀ D⁻¹ F = L Lᵀ,
#   C⁻¹ = D⁻¹ − W M⁻¹ Wᵀ,  diag(C⁻¹)_i = 1/v_i − ‖L⁻¹ W[i, :]‖²,
#   log det C = Σ log v_i + log det M.
function _lowrank_gaussian_loo(r::Vector{Float64}, v::Vector{Float64},
                               F::Matrix{Float64})
    n, k = size(F)
    W = F ./ v
    M = F' * W
    @inbounds for j in 1:k
        M[j, j] += 1.0
    end
    cM = cholesky!(Symmetric(M); check = false)
    issuccess(cM) || throw(ArgumentError(
        "compute_loo: I + FᵀD⁻¹F is not positive definite; cannot compute the " *
        "leave-one-out of the additive covariance."))
    Dr = r ./ v
    g = Dr .- W * (cM \ (F' * Dr))
    X = cM.L \ Matrix(W')                       # k × n
    joint = -0.5 * (dot(r, g) + sum(log, v) + logdet(cM) + n * log(2π))
    ll = Vector{Float64}(undef, n)
    half_log_2π = 0.5 * log(2π)
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:k
            s += X[j, i]^2
        end
        d_i = 1 / v[i] - s
        ll[i] = 0.5 * log(d_i) - half_log_2π - 0.5 * g[i]^2 / d_i
    end
    return ll, joint
end

# γ marginalized analytically (flat prior, one γ per sharing group): the
# points of a group share an unknown offset, so they are coupled. With the
# group's other points, γ | y_{-i} ~ N(B₋ᵢ/A₋ᵢ, 1/A₋ᵢ) (A = Σ 1/v, B = Σ d/v
# over the group less point i), and the predictive of d_i = y_i − μ_i (μ
# without γ) is N(B₋ᵢ/A₋ᵢ, v_i + 1/A₋ᵢ) -- the flat-prior limit of the
# Gaussian formula above, in O(n).
function _gamma_marginal_loo(theta::Theta{Float64}, data::Data, d::Vector{Float64},
                             vars::Vector{Float64}, ref::Float64)
    n = length(d)
    slot = theta.params.layout.systemic.rv_gamma
    joint = _rv_ll_gamma_marginalized(d, vars, data.rv_inst, slot, n, 2π)
    _loo_check(joint, ref, :rv)
    A = Dict{Int, Float64}(); B = Dict{Int, Float64}()
    @inbounds for i in 1:n
        g = slot[data.rv_inst[i]]; w = 1 / vars[i]
        A[g] = get(A, g, 0.0) + w
        B[g] = get(B, g, 0.0) + w * d[i]
    end
    ll = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        g = slot[data.rv_inst[i]]; w = 1 / vars[i]
        A_i = A[g] - w
        A_i > 0 || throw(ArgumentError(
            "compute_loo: a γ-marginalized instrument group has a single point; " *
            "with γ integrated out under a flat prior, its leave-one-out " *
            "predictive is improper."))
        μ = (B[g] - w * d[i]) / A_i
        s2 = vars[i] + 1 / A_i
        ll[i] = -0.5 * (log(2π * s2) + (d[i] - μ)^2 / s2)
    end
    return ll
end

# Conditional Gaussian of the RV residuals given the activity indicators
# under a global ActivityGP (`marginalize_indicators = true`): returns
# (y_R − μ_R|I, Σ_R|I), the residual and covariance whose log-density is the
# likelihood's log p(RV | indicators, θ) (see the marginalize branch of
# `_activity_gp_joint_ll`, which this mirrors), or `nothing` where that
# likelihood is -Inf.
function _agp_conditional_gaussian(theta::Theta{Float64}, data::Data,
                                   preds::Vector{Float64}, vars::Vector{Float64},
                                   agp::ActivityGP)
    ix = _agp_index(theta.params.layout.name_to_idx, data, agp)
    v = theta.values
    amp = ix.amp == 0 ? 1.0 : v[ix.amp]
    P = v[ix.P]; λe = v[ix.λe]; λp = v[ix.λp]
    (amp > 0 && P > 0 && λe > 0 && λp > 0) || return nothing
    inv_sdG = 1 / sqrt(1 / (λe * λe) + π * π / (P * P * λp * λp))
    n_rv = length(data.rv)
    n_ind = length(ix.a)
    C = n_ind + 1
    chan_a = Vector{Float64}(undef, C); chan_b = Vector{Float64}(undef, C)
    chan_a[1] = ix.Vc == 0 ? 0.0 : v[ix.Vc]
    chan_b[1] = ix.Vr == 0 ? 0.0 : v[ix.Vr] * inv_sdG
    n_total = n_rv
    for k in 1:n_ind
        chan_a[k + 1] = v[ix.a[k]]
        chan_b[k + 1] = ix.b[k] == 0 ? 0.0 : v[ix.b[k]] * inv_sdG
        n_total += length(ix.vals[k])
    end
    y = Vector{Float64}(undef, n_total); σ² = Vector{Float64}(undef, n_total)
    @inbounds for i in 1:n_rv
        y[i] = data.rv[i] - preds[i]; σ²[i] = vars[i]
    end
    off = n_rv
    for k in 1:n_ind
        jit² = ix.jit[k] == 0 ? 0.0 : v[ix.jit[k]]^2
        vals = ix.vals[k]; errs = ix.errs[k]
        @inbounds for i in eachindex(vals)
            y[off + i] = vals[i]; σ²[off + i] = errs[i]^2 + jit²
        end
        off += length(vals)
    end
    Σ = activity_gp_covariance_blocked(view(data.t_rv, 1:n_rv), chan_a, chan_b,
                                       amp, P, λe, λp)
    @inbounds for i in 1:n_total
        Σ[i, i] += σ²[i]
    end
    R = 1:n_rv
    n_total == n_rv && return (y, Matrix(Σ))
    Iind = (n_rv + 1):n_total
    F_II = cholesky(Symmetric(Σ[Iind, Iind]); check = false)
    issuccess(F_II) || return nothing
    Σ_RI = Σ[R, Iind]
    r = y[R] .- Σ_RI * (F_II \ y[Iind])
    Σ_cond = Σ[R, R] .- Σ_RI * (F_II \ Matrix(Σ_RI'))
    Σ_cond = (Σ_cond .+ Σ_cond') ./ 2
    return (r, Σ_cond)
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

# Independent Student-t densities, term by term as `_studentt_diag_ll` sums
# them.
function _studentt_logpdf_per_point(r::AbstractVector{<:Real},
                                    σ²::AbstractVector{<:Real}, ν::Real)
    half = (ν + 1) / 2
    c = loggamma(half) - loggamma(ν / 2) - 0.5 * log(ν * π)
    out = Vector{Float64}(undef, length(r))
    @inbounds for i in eachindex(r)
        out[i] = c - 0.5 * log(σ²[i]) - half * log1p(r[i]^2 / (ν * σ²[i]))
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
