# Posterior predictive checks.
#
# Subsamples a posterior chain, re-evaluates the forward model per
# draw, and aggregates residual diagnostics: per-instrument χ²/dof,
# point-wise residual mean ± std, GLS of residuals (signal you missed)
# and ACF of residuals (correlated noise you didn't model). The intent
# is a single object passed to `plot_ppc` for the standard diagnostic
# figure, plus a small summary Dict written to `summary.json`.
#
# Works on any MCMCChains-style chain object — trans-dim chains are
# handled by reading `n_planets` per draw.

using Statistics: mean, median, std, quantile
using Random: AbstractRNG, default_rng, randperm, shuffle
using FFTW: rfft, irfft

# =====================================================================
# Result struct
# =====================================================================

"""
    PPCResult

Output of [`posterior_predictive_check`](@ref). Per-channel residual
mean/std arrays, per-instrument χ²/dof, residual GLS, and residual
ACF. Pass to [`plot_ppc`](@ref) for the figure or read `result.summary`
for the scalar summary written into `summary.json`.

Three χ²/dof statistics are computed and stored, addressing different
failure modes:

- `rv_red_chi2_total` — `min(chi2_per_draw) / dof`: the **best fit
  found anywhere in the posterior**. Robust to degenerate / circular
  axes (e.g., uniform ω at e≈0, where the median-parameter point sits
  at a coordinate singularity); stays good whenever a good fit *exists*
  in the posterior.
- `rv_red_chi2_median_model` — χ²/dof at the per-parameter chain
  median model. Equals the headline for well-constrained posteriors;
  can be inflated when any structural axis is degenerate at the
  median (e.g., atan2(median(sesinw), median(secosw)) when both
  marginals straddle zero).
- `rv_red_chi2_per_draw_p16 / p50 / p84` — distribution across draws.
  A wide spread means parameter uncertainty contributes meaningfully.

The per-draw χ² vector itself is exposed (`rv_chi2_per_draw`) for
custom diagnostics.

Fields:
- `n_draws::Int` — number of posterior draws used.
- RV block (`has_rv` true when `n_rv(data) > 0`):
  - `rv_t`, `rv_inst`, `rv_data` — observed time / instrument index / values
  - `rv_resid_med` — `data - preds` at the median-parameter model
  - `rv_resid_std` — pointwise std of `(data - preds)` across draws (use as plot envelope)
  - `rv_err_med` — sqrt of per-point variance at the median-parameter model (incl. jitter)
  - `rv_chi2_per_inst`, `rv_dof_per_inst`, `rv_red_chi2_per_inst` — at median model
  - `rv_chi2_per_draw::Vector{Float64}` — total RV χ² for each posterior draw
  - `rv_pgram` — `GLSPgram` of median-model residuals (or `nothing` if too few points)
  - `rv_acf_lags`, `rv_acf` — autocorrelation of median-model residuals (binned uniformly)
- PM block — analogous fields for photometry.
- `summary::Dict{String, Any}` — flat dictionary of summary stats for `summary.json`.
"""
struct PPCResult
    n_draws::Int

    has_rv::Bool
    rv_t::Vector{Float64}
    rv_inst::Vector{Int}
    rv_data::Vector{Float64}
    rv_resid_med::Vector{Float64}
    rv_resid_std::Vector{Float64}
    rv_err_med::Vector{Float64}
    rv_chi2_per_inst::Dict{Int, Float64}
    rv_dof_per_inst::Dict{Int, Int}
    rv_red_chi2_per_inst::Dict{Int, Float64}
    rv_chi2_per_draw::Vector{Float64}
    rv_pgram::Union{Nothing, GLSPgram}
    rv_acf_lags::Vector{Float64}
    rv_acf::Vector{Float64}

    has_pm::Bool
    phot_t::Vector{Float64}
    phot_inst::Vector{Int}
    phot_data::Vector{Float64}
    phot_resid_med::Vector{Float64}
    phot_resid_std::Vector{Float64}
    phot_err_med::Vector{Float64}
    phot_chi2_per_inst::Dict{Int, Float64}
    phot_dof_per_inst::Dict{Int, Int}
    phot_red_chi2_per_inst::Dict{Int, Float64}
    phot_chi2_per_draw::Vector{Float64}
    phot_acf_lags::Vector{Float64}
    phot_acf::Vector{Float64}

    summary::Dict{String, Any}
end

# =====================================================================
# Top-level API
# =====================================================================

# GP-detrend an RV residual vector before χ²: when a global covariance
# GP (celerite or Rajpaul ActivityGP) is active, the activity lives in
# the COVARIANCE, so a mean-model χ² double-counts it as white scatter
# (HD 18599 AGP read χ²/dof ≈ 4.7 because the GP component was scored as
# noise). Subtracting the GP predictive mean makes the reported χ² the
# honest white-residual goodness-of-fit. No-op (returns the input) for
# white / decorrelation-only models — `channel_gp_mean_at` returns
# `nothing` after a cheap noise-model scan when no covariance GP is on.
function _rv_gp_detrend(theta, data, resid::Vector{Float64},
                         vars::Vector{Float64})
    isempty(resid) && return resid
    gp = try
        channel_gp_mean_at(theta, resid, vars, data.t_rv, data.t_rv,
                            data.rv_inst, :rv; data = data)
    catch
        nothing
    end
    gp === nothing ? resid : resid .- gp
end

"""
    posterior_predictive_check(chains, params, data;
                                n_draws=500, rng=default_rng(),
                                pgram=true, acf=true,
                                pgram_period_min=1.0,
                                pgram_period_max=nothing) -> PPCResult

Sample `n_draws` rows uniformly from the posterior chain, evaluate the
forward model at each draw, and aggregate residual diagnostics. The
output object is the diagnostic; `plot_ppc(result)` produces the
figure.

Trans-dim chains are handled transparently — `n_planets` is read per
draw so the residuals are computed under the actual active-K of each
sample.
"""
function posterior_predictive_check(chains, params::Params, data::Data;
                                     n_draws::Int = 500,
                                     rng::AbstractRNG = default_rng(),
                                     pgram::Bool = true,
                                     acf::Bool = true,
                                     pgram_period_min::Real = 1.0,
                                     pgram_period_max::Union{Nothing, Real} = nothing)
    n_rv_obs = n_rv(data)
    n_pm_obs = n_phot(data)
    (n_rv_obs > 0 || n_pm_obs > 0) ||
        throw(ArgumentError("data has no RV or photometry"))

    # Build a flat per-sample dictionary {Symbol => Vector{Float64}}
    chains_flat, n_total = _flatten_chains(chains)
    n_draws_eff = min(n_draws, n_total)
    n_draws_eff > 0 || throw(ArgumentError("chain is empty"))

    idx_pool = randperm(rng, n_total)[1:n_draws_eff]

    # ---- Median-parameter Theta: the headline model ---------------------
    theta_med = _theta_from_chain_summary(chains_flat, params)
    preds_med_rv  = n_rv_obs > 0 ? rv_predictions(theta_med, data)[1] : Float64[]
    vars_med_rv   = n_rv_obs > 0 ? rv_predictions(theta_med, data)[2] : Float64[]
    preds_med_phot = n_pm_obs > 0 ? phot_predictions(theta_med, data)[1] : Float64[]
    vars_med_phot  = n_pm_obs > 0 ? phot_predictions(theta_med, data)[2] : Float64[]

    rv_resid_med = n_rv_obs > 0 ? data.rv .- preds_med_rv : Float64[]
    rv_resid_med = _rv_gp_detrend(theta_med, data, rv_resid_med, vars_med_rv)
    rv_err_med   = n_rv_obs > 0 ? sqrt.(vars_med_rv) : Float64[]

    phot_resid_med = n_pm_obs > 0 ? data.flux .- preds_med_phot : Float64[]
    phot_err_med   = n_pm_obs > 0 ? sqrt.(vars_med_phot) : Float64[]

    # ---- Per-draw accumulators ------------------------------------------
    # rv_resid_std comes from the std of `(data - preds)` across draws —
    # parameter uncertainty propagated to the data space. Useful as a
    # plot envelope around the median-model residual.
    rv_resid_sum = zeros(Float64, n_rv_obs)
    rv_resid_sq  = zeros(Float64, n_rv_obs)
    rv_chi2_per_draw = zeros(Float64, n_draws_eff)

    phot_resid_sum = zeros(Float64, n_pm_obs)
    phot_resid_sq  = zeros(Float64, n_pm_obs)
    phot_chi2_per_draw = zeros(Float64, n_draws_eff)

    for (draw_idx, idx) in enumerate(idx_pool)
        theta = _theta_from_row(chains_flat, idx, params)

        if n_rv_obs > 0
            preds, vars = rv_predictions(theta, data)
            resid = data.rv .- preds
            resid = _rv_gp_detrend(theta, data, resid, vars)
            chi2 = 0.0
            @inbounds for i in 1:n_rv_obs
                r = resid[i]
                rv_resid_sum[i] += r
                rv_resid_sq[i]  += r * r
                chi2 += r * r / vars[i]
            end
            rv_chi2_per_draw[draw_idx] = chi2
        end

        if n_pm_obs > 0
            preds, vars = phot_predictions(theta, data)
            chi2 = 0.0
            @inbounds for i in 1:n_pm_obs
                r = data.flux[i] - preds[i]
                phot_resid_sum[i] += r
                phot_resid_sq[i]  += r * r
                chi2 += r * r / vars[i]
            end
            phot_chi2_per_draw[draw_idx] = chi2
        end
    end

    # ---- Std envelope (per-draw scatter of residual at each point) ------
    rv_resid_std = if n_rv_obs > 0
        μ = rv_resid_sum ./ n_draws_eff
        sqrt.(max.((rv_resid_sq ./ n_draws_eff) .- μ .^ 2, 0.0))
    else
        Float64[]
    end
    phot_resid_std = if n_pm_obs > 0
        μ = phot_resid_sum ./ n_draws_eff
        sqrt.(max.((phot_resid_sq ./ n_draws_eff) .- μ .^ 2, 0.0))
    else
        Float64[]
    end

    # ---- Per-instrument χ²/dof at the median model ----------------------
    rv_chi2_inst   = Dict{Int, Float64}()
    rv_n_inst      = Dict{Int, Int}()
    rv_dof_inst    = Dict{Int, Int}()
    rv_red_chi2    = Dict{Int, Float64}()
    if n_rv_obs > 0
        for i in 1:n_rv_obs
            inst = data.rv_inst[i]
            rv_chi2_inst[inst] = get(rv_chi2_inst, inst, 0.0) +
                                  rv_resid_med[i]^2 / vars_med_rv[i]
            rv_n_inst[inst]    = get(rv_n_inst, inst, 0) + 1
        end
        for (inst, n_obs) in rv_n_inst
            # Conservative: 2 instrument-specific dof (γ, σ).
            dof = max(1, n_obs - 2)
            rv_dof_inst[inst] = dof
            rv_red_chi2[inst] = rv_chi2_inst[inst] / dof
        end
    end

    phot_chi2_inst = Dict{Int, Float64}()
    phot_n_inst    = Dict{Int, Int}()
    phot_dof_inst  = Dict{Int, Int}()
    phot_red_chi2  = Dict{Int, Float64}()
    if n_pm_obs > 0
        for i in 1:n_pm_obs
            inst = data.phot_inst[i]
            phot_chi2_inst[inst] = get(phot_chi2_inst, inst, 0.0) +
                                    phot_resid_med[i]^2 / vars_med_phot[i]
            phot_n_inst[inst]    = get(phot_n_inst, inst, 0) + 1
        end
        for (inst, n_obs) in phot_n_inst
            dof = max(1, n_obs - 3)
            phot_dof_inst[inst] = dof
            phot_red_chi2[inst] = phot_chi2_inst[inst] / dof
        end
    end

    # ---- Residual GLS / ACF — driven by the median-model residual -------
    rv_pgram = nothing
    if pgram && n_rv_obs >= 16
        T = maximum(data.t_rv) - minimum(data.t_rv)
        Pmax = pgram_period_max === nothing ? T : Float64(pgram_period_max)
        # This is a "is there leftover periodicity?" DIAGNOSTIC, not a publication
        # periodogram. The default bootstrap FAP (1000 resamples) over a
        # baseline-sized grid is intractable on real long-baseline RV — e.g. a
        # 19-yr ε Eri set drives ~3.5e5 frequencies × 1000 bootstraps ≈ hours and
        # blows run_job's wall-clock. Cap the grid and use the analytic FAP
        # (≈3× over-confident, fine for a flag); explicit periodogram plots still
        # use the full bootstrap path.
        f_min = 1.0 / Pmax
        f_max = 1.0 / Float64(pgram_period_min)
        nf = clamp(ceil(Int, (f_max - f_min) * T * 50), 64, 20_000)
        try
            rv_pgram = gls_periodogram(data.t_rv, rv_resid_med, rv_err_med;
                                        period_min = Float64(pgram_period_min),
                                        period_max = Pmax,
                                        n_freqs = nf,
                                        fap_method = :analytic,
                                        max_peaks = 5)
        catch err
            @warn "PPC: residual GLS failed" exception = err
        end
    end

    rv_acf_lags, rv_acf_vals = if acf && n_rv_obs >= 16
        _residual_acf(Float64.(data.t_rv), rv_resid_med)
    else
        (Float64[], Float64[])
    end
    phot_acf_lags, phot_acf_vals = if acf && n_pm_obs >= 64
        _residual_acf(Float64.(data.t_phot), phot_resid_med)
    else
        (Float64[], Float64[])
    end

    # ---- Summary --------------------------------------------------------
    # Headline χ²/dof is the median-model value. Per-draw distribution
    # exposed as p16/p50/p84 — a wide distribution at fixed median-χ²
    # means parameter uncertainty is the dominant contribution.
    summary = Dict{String, Any}()
    summary["n_draws"] = n_draws_eff

    if n_rv_obs > 0
        dof_total = max(1, sum(values(rv_dof_inst)))
        red_per_draw = rv_chi2_per_draw ./ dof_total
        # Headline: best fit found anywhere in the posterior. Robust
        # to coordinate singularities at the per-parameter median.
        summary["rv_red_chi2_total"] = minimum(red_per_draw)
        summary["rv_red_chi2_median_model"] =
            sum(values(rv_chi2_inst)) / dof_total
        summary["rv_red_chi2_per_inst"] =
            Dict(string(k) => v for (k, v) in rv_red_chi2)
        summary["rv_outliers_3sigma"] = count(
            i -> abs(rv_resid_med[i]) > 3 * rv_err_med[i], 1:n_rv_obs)
        summary["rv_red_chi2_per_draw_p16"] = quantile(red_per_draw, 0.16)
        summary["rv_red_chi2_per_draw_p50"] = quantile(red_per_draw, 0.50)
        summary["rv_red_chi2_per_draw_p84"] = quantile(red_per_draw, 0.84)

        if rv_pgram !== nothing && !isempty(rv_pgram.peaks)
            top = rv_pgram.peaks[1]
            summary["rv_residual_top_peak_period"] = top.period
            summary["rv_residual_top_peak_power"]  = top.power
            summary["rv_residual_top_peak_fap"]    = top.fap
        end
        if !isempty(rv_acf_vals) && length(rv_acf_vals) > 1
            summary["rv_residual_acf_lag1"] = rv_acf_vals[2]
        end
    end
    if n_pm_obs > 0
        dof_total = max(1, sum(values(phot_dof_inst)))
        red_per_draw = phot_chi2_per_draw ./ dof_total
        summary["phot_red_chi2_total"] = minimum(red_per_draw)
        summary["phot_red_chi2_median_model"] =
            sum(values(phot_chi2_inst)) / dof_total
        summary["phot_red_chi2_per_inst"] =
            Dict(string(k) => v for (k, v) in phot_red_chi2)
        summary["phot_outliers_3sigma"] = count(
            i -> abs(phot_resid_med[i]) > 3 * phot_err_med[i], 1:n_pm_obs)
        summary["phot_red_chi2_per_draw_p16"] = quantile(red_per_draw, 0.16)
        summary["phot_red_chi2_per_draw_p50"] = quantile(red_per_draw, 0.50)
        summary["phot_red_chi2_per_draw_p84"] = quantile(red_per_draw, 0.84)

        if !isempty(phot_acf_vals) && length(phot_acf_vals) > 1
            summary["phot_residual_acf_lag1"] = phot_acf_vals[2]
        end
    end

    return PPCResult(
        n_draws_eff,
        n_rv_obs > 0,
        Float64.(data.t_rv), Int.(data.rv_inst), Float64.(data.rv),
        rv_resid_med, rv_resid_std, rv_err_med,
        rv_chi2_inst, rv_dof_inst, rv_red_chi2,
        rv_chi2_per_draw,
        rv_pgram, rv_acf_lags, rv_acf_vals,
        n_pm_obs > 0,
        Float64.(data.t_phot), Int.(data.phot_inst), Float64.(data.flux),
        phot_resid_med, phot_resid_std, phot_err_med,
        phot_chi2_inst, phot_dof_inst, phot_red_chi2,
        phot_chi2_per_draw,
        phot_acf_lags, phot_acf_vals,
        summary,
    )
end

# =====================================================================
# Helpers
# =====================================================================

function _flatten_chains(chains)
    chain_names = collect(names(chains, :parameters))
    flat = Dict{Symbol, Vector{Float64}}()
    n_total = 0
    for nm in chain_names
        v = vec(Array(chains[nm]))
        flat[Symbol(nm)] = Float64.(v)
        n_total = length(v)
    end
    return flat, n_total
end

# The "typical" parameter point used as the median model for the
# headline χ² / outlier / residual GLS. For n_planets (trans-dim), the
# posterior mode is used since the chain is integer-valued.
function _theta_from_chain_summary(chains_flat::Dict{Symbol, Vector{Float64}},
                                    params::Params)
    theta = Theta{Float64}(params)
    if haskey(chains_flat, :n_planets)
        np_samp = chains_flat[:n_planets]
        n_p_set = _mode_int(np_samp)
        set_n_p!(theta, n_p_set)
    end
    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        haskey(chains_flat, sym) || continue
        set_param!(theta, name, median(chains_flat[sym]))
    end
    return theta
end

# `_mode_int` lives in reporting.jl (included first); this file used to carry
# a byte-identical copy specialised to AbstractVector{<:Real}.

function _theta_from_row(chains_flat::Dict{Symbol, Vector{Float64}},
                          idx::Int, params::Params)
    theta = Theta{Float64}(params)
    if haskey(chains_flat, :n_planets)
        np_val = chains_flat[:n_planets][idx]
        set_n_p!(theta, Int(round(np_val)))
    end
    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        haskey(chains_flat, sym) || continue
        set_param!(theta, name, chains_flat[sym][idx])
    end
    return theta
end

# Bin residuals onto a uniform grid and compute the normalised
# autocorrelation via FFT (Wiener-Khinchin), same pattern as
# `find_rotation_period`. Returns up to 25% of the binned length.
function _residual_acf(t::Vector{Float64}, r::Vector{Float64};
                        cadence::Union{Nothing, Float64} = nothing,
                        max_lag_fraction::Float64 = 0.25)
    length(t) == length(r) || throw(ArgumentError("t / r length mismatch"))
    length(t) >= 8 || return (Float64[], Float64[])
    dt = cadence === nothing ? median(diff(sort(t))) : cadence
    (dt > 0 && isfinite(dt)) || return (Float64[], Float64[])

    t0 = minimum(t)
    n_bins = max(8, Int(round((maximum(t) - t0) / dt)) + 1)
    sums = zeros(Float64, n_bins)
    counts = zeros(Int, n_bins)
    @inbounds for i in eachindex(t)
        b = clamp(Int(round((t[i] - t0) / dt)) + 1, 1, n_bins)
        sums[b] += r[i]
        counts[b] += 1
    end
    binned = [counts[i] > 0 ? sums[i] / counts[i] : NaN for i in 1:n_bins]

    # Linear-interpolate empty bins.
    _fill_nans_linear!(binned)
    μ = mean(binned)
    y = binned .- μ

    n_pad = nextpow(2, 2 * length(y))
    pad = zeros(Float64, n_pad)
    pad[1:length(y)] .= y
    F = rfft(pad)
    @inbounds for k in eachindex(F)
        F[k] = F[k] * conj(F[k])
    end
    ac = irfft(F, n_pad)[1:length(y)]
    ac[1] != 0 || return (Float64[], Float64[])
    ac ./= ac[1]
    max_lag = max(1, Int(floor(length(ac) * max_lag_fraction)))
    lags = collect(0:max_lag) .* dt
    return (lags, ac[1:length(lags)])
end

# `_fill_nans_linear!` is reused from `preprocessing/rotation_period.jl`.
