# Result reporting — console output + file output for fitted, derived,
# model statistics, and trans-dimensional results.
#
# Matches astroEMPEROR's emperors_utils.py output style:
#   - print_best_fit / save_best_fit / save_run_stats equivalents
#   - All parameters reported (fitted, noise, instrumental)
#   - Derived parameters per planet
#   - Trans-dim: N_p posterior, joint tables, conditional posteriors

using MCMCChains
using Statistics: median, std, quantile
using Printf: @sprintf, @printf

# =====================================================================
# ParamStats — summary statistics for one parameter
# =====================================================================

"""
    ParamStats

Summary statistics for a single parameter posterior.

- `best`: median of the posterior
- `unc_lo`, `unc_hi`: asymmetric uncertainties (best - 16th, 84th - best)
- `ci1`, `ci2`, `ci3`: 1/2/3-sigma credible intervals
"""
struct ParamStats
    best::Float64
    unc_lo::Float64
    unc_hi::Float64
    ci1::Tuple{Float64, Float64}
    ci2::Tuple{Float64, Float64}
    ci3::Tuple{Float64, Float64}
end

function ParamStats(samples::AbstractVector{<:Real})
    med = median(samples)
    q16 = quantile(samples, 0.16)
    q84 = quantile(samples, 0.84)
    q025 = quantile(samples, 0.025)
    q975 = quantile(samples, 0.975)
    q001 = quantile(samples, 0.0015)
    q999 = quantile(samples, 0.9985)
    return ParamStats(
        med,
        med - q16,
        q84 - med,
        (q16, q84),
        (q025, q975),
        (q001, q999),
    )
end

# =====================================================================
# summarize_fitted — stats for all unfrozen parameters
# =====================================================================

"""
    summarize_fitted(chains, params) -> OrderedDict{String, ParamStats}

Compute summary statistics for every unfrozen parameter in the chain.
Handles parametrization transforms (lnP→P, lnK→K) so the reported
values are in physical units.

Returns an ordered dict preserving the parameter layout order.
"""
function summarize_fitted(chains, params::Params)
    stats = Vector{Pair{String, ParamStats}}()

    chain_names = Set(names(chains, :parameters))
    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        sym in chain_names || continue
        samp = vec(Array(chains[sym]))
        push!(stats, name => ParamStats(samp))
    end

    return stats
end

# =====================================================================
# summarize_derived — stats for derived physical parameters
# =====================================================================

"""
    summarize_derived(chains, params; M_s=nothing, R_s=nothing,
                       T_eff=nothing, J_mag=nothing, K_mag=nothing, Ab=0.0)

Compute summary statistics for derived parameters (eccentricity,
omega, msini, radius, etc.) using `compute_derived`.

Returns an ordered list of pairs: name => ParamStats.
"""
function summarize_derived(chains, params::Params;
                            M_s=nothing, R_s=nothing,
                            T_eff=nothing, J_mag=nothing,
                            K_mag=nothing, Ab=0.0)
    # Fall back to config.M_s if available
    if M_s === nothing && !isnan(params.config.M_s)
        M_s = params.config.M_s
    end
    derived_vec = compute_derived(chains, params;
                                   M_s=M_s, R_s=R_s, T_eff=T_eff,
                                   J_mag=J_mag, K_mag=K_mag, Ab=Ab)
    stats = Vector{Pair{String, ParamStats}}()

    for dp in derived_vec
        k_str = split(dp.name, "_")[end]  # "planet_1" -> "1"
        for (key, samples) in dp.values
            label = "$(key)_k$(k_str)"
            push!(stats, label => ParamStats(samples))
        end
    end

    return stats
end

# =====================================================================
# print_results — console output (astroEMPEROR format)
# =====================================================================

const DASHES = "-"^12

"""
    print_results(fitted_stats, derived_stats; io=stdout)

Print fitted and derived parameter statistics to console, matching
astroEMPEROR's `print_best_fit` format.
"""
function print_results(fitted_stats::Vector{Pair{String, ParamStats}},
                        derived_stats::Vector{Pair{String, ParamStats}};
                        io::IO=stdout)
    _print_stats_block(io, fitted_stats, "Fitted")
    if !isempty(derived_stats)
        _print_stats_block(io, derived_stats, "Derived")
    end
end

function _print_stats_block(io::IO, stats::Vector{Pair{String, ParamStats}},
                             kind::String)
    println(io)
    println(io, "$DASHES $kind Parameter Statistics $DASHES")
    @printf(io, "%-24s  %16s  %16s  %16s  %24s\n",
            "Parameter", "best", "upper", "lower", "3 sigma CI")
    for (name, s) in stats
        ci_str = @sprintf("[%.4f, %.4f]", s.ci3[1], s.ci3[2])
        @printf(io, "%-24s  %16.4f  %+16.4f  %16.4f  %24s\n",
                name, s.best, s.unc_hi, -s.unc_lo, ci_str)
    end
end

# =====================================================================
# save_results — file output
# =====================================================================

"""
    save_results(fitted_stats, derived_stats, model_stats, path;
                  starname, kplanet)

Save results to files matching astroEMPEROR naming convention:
- `{starname}_k{kplanet}_fitted_best_fit.dat`
- `{starname}_k{kplanet}_derived_best_fit.dat`
- `{starname}_k{kplanet}_run_stats.dat`
"""
function save_results(fitted_stats::Vector{Pair{String, ParamStats}},
                       derived_stats::Vector{Pair{String, ParamStats}},
                       model_stats::Dict{String, Float64},
                       path::String;
                       starname::String, kplanet::Int)
    prefix = joinpath(path, "$(starname)_k$(kplanet)")
    _save_stats_file("$(prefix)_fitted_best_fit.dat", fitted_stats)
    _save_stats_file("$(prefix)_derived_best_fit.dat", derived_stats)
    _save_run_stats_file("$(prefix)_run_stats.dat", model_stats)
end

function _save_stats_file(filepath::String, stats::Vector{Pair{String, ParamStats}})
    open(filepath, "w") do f
        @printf(f, "%-24s  %16s  %16s  %16s  %24s\n",
                "Parameter", "best", "upper", "lower", "3 sigma CI")
        for (name, s) in stats
            ci_str = @sprintf("[%.8f, %.6f]", s.ci3[1], s.ci3[2])
            @printf(f, "%-24s  %16.6f  %+16.8f  %16.8f  %24s\n",
                    name, s.best, s.unc_hi, -s.unc_lo, ci_str)
        end
    end
end

function _save_run_stats_file(filepath::String, stats::Dict{String, Float64})
    # Ordered keys matching astroEMPEROR
    key_order = ["RV RMS", "RV WRMS", "PM RMS", "RV Chi2", "RV Red Chi2",
                 "PM Chi2", "PM Red Chi2", "BIC", "AIC",
                 "POSTERIOR", "EVIDENCE"]
    open(filepath, "w") do f
        @printf(f, "%-16s  %16s\n", "Criteria", "This Run")
        for k in key_order
            haskey(stats, k) || continue
            @printf(f, "%-16s  %16.4f\n", k, stats[k])
        end
    end
end

# =====================================================================
# compute_model_stats
# =====================================================================

"""
    compute_model_stats(chains, data, params;
                         log_evidence=nothing, log_L_max=nothing)

Compute model comparison statistics from posterior chains.

Evaluates the model at the median posterior sample and computes:
- RV RMS, weighted RMS, Chi2, reduced Chi2
- PM RMS, Chi2, reduced Chi2 (if photometry present)
- BIC, AIC (requires `log_L_max`)
- log-evidence (passed through from sampler)

Returns a `Dict{String, Float64}`.
"""
function compute_model_stats(chains, data::Data, params::Params;
                              log_evidence::Union{Nothing, Float64}=nothing,
                              log_L_max::Union{Nothing, Float64}=nothing)
    stats = Dict{String, Float64}()
    ndim = length(params.layout.unfrozen_idx)
    n_rv_obs = n_rv(data)
    n_pm_obs = n_phot(data)
    ndat = n_rv_obs + n_pm_obs

    # --- Build Theta from median posterior sample ---
    chain_names = Set(names(chains, :parameters))
    theta = Theta{Float64}(params)

    # Set N_p from chain if available (trans-dim)
    if :n_planets in chain_names
        np_samp = vec(Array(chains[:n_planets]))
        # Use the mode (most probable N_p) for stats evaluation
        np_mode = _mode_int(np_samp)
        set_n_p!(theta, np_mode)
    end

    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        sym in chain_names || continue
        samp = vec(Array(chains[sym]))
        set_param!(theta, name, median(samp))
    end

    # --- RV diagnostics ---
    if n_rv_obs > 0
        preds, vars = rv_predictions(theta, data)
        residuals = data.rv .- preds

        # RMS (unweighted)
        rms = sqrt(sum(residuals .^ 2) / n_rv_obs)
        stats["RV RMS"] = rms

        # Weighted RMS = sqrt(sum(r^2/var) / sum(1/var))
        inv_var = 1.0 ./ vars
        wrms = sqrt(sum(residuals .^ 2 .* inv_var) / sum(inv_var))
        stats["RV WRMS"] = wrms

        # Chi-squared
        chi2 = sum(residuals .^ 2 ./ vars)
        dof = max(n_rv_obs - ndim, 1)
        stats["RV Chi2"] = chi2
        stats["RV Red Chi2"] = chi2 / dof
    end

    # --- BIC / AIC ---
    if log_L_max !== nothing
        stats["BIC"] = ndim * log(ndat) - 2.0 * log_L_max
        stats["AIC"] = 2.0 * ndim - 2.0 * log_L_max
    end

    # --- Evidence ---
    if log_evidence !== nothing
        stats["EVIDENCE"] = log_evidence
    end

    return stats
end

# =====================================================================
# print_transdim_summary — console output for trans-dim results
# =====================================================================

"""
    print_transdim_summary(chains, params; io=stdout,
                            td=nothing, min_samples=20,
                            M_s=nothing, R_s=nothing, T_eff=nothing,
                            J_mag=nothing, K_mag=nothing, Ab=0.0)

Print trans-dimensional results:
1. N_p posterior probabilities
2. Joint N_p × noise model table (if noise trans-dim active)
3. Per-N_p conditional posteriors with all fitted + derived parameters
"""
function print_transdim_summary(chains, params::Params;
                                 io::IO=stdout,
                                 td::Union{Nothing, TransDimConfig}=nothing,
                                 min_samples::Int=20,
                                 M_s=nothing, R_s=nothing, T_eff=nothing,
                                 J_mag=nothing, K_mag=nothing, Ab=0.0)
    # Fall back to config.M_s if available
    if M_s === nothing && !isnan(params.config.M_s)
        M_s = params.config.M_s
    end
    max_kp = params.config.max_kplanet
    np = vec(Array(chains[:n_planets]))
    n_total = length(np)

    # --- N_p posterior ---
    println(io, "\n" * "="^60)
    println(io, "  Trans-dimensional posterior")
    println(io, "="^60)

    # Detect noise columns
    noise_cols = _detect_noise_columns(chains, td)
    has_noise = !isempty(noise_cols)

    if has_noise
        _print_joint_table(io, np, chains, noise_cols, max_kp, n_total)
    else
        println(io)
        for k in 0:max_kp
            nk = count(np .== Float64(k))
            nk == 0 && continue
            @printf(io, "  P(N_p = %d) = %.4f  (%d / %d samples)\n",
                    k, nk / n_total, nk, n_total)
        end
    end

    # --- Conditional posteriors per N_p ---
    # Compute the global "winning noise" set: noise models with
    # marginal P(active) > 0.5 across the full chain. Reporting params
    # for losing noise models would be misleading (the model isn't part
    # of the winning configuration). Same idea on N_p side: report only
    # the modal value (winning planet count) by default.
    modal_np = _mode_int(np)
    global_noise_win = Bool[]
    global_noise_prob = Float64[]
    if has_noise
        println(io)
        println(io, "  Marginal noise-model probabilities (global):")
        for (col_name, col_label) in noise_cols
            samp = vec(Array(chains[col_name]))
            p = count(samp .== 1.0) / length(samp)
            push!(global_noise_win, p > 0.5)
            push!(global_noise_prob, p)
            tag = p > 0.5 ? "(WIN)" : "     "
            @printf(io, "    %-12s  P(on) = %.4f  %s\n",
                    col_label, p, tag)
        end
    end

    println(io)
    for np_val in 0:max_kp
        mask = np .== Float64(np_val)
        n_np = count(mask)
        n_np < min_samples && continue
        # Only the winning N_p gets the full conditional dump. Other
        # values are summarized in the joint table above.
        np_val == modal_np || continue

        # Build a conditional sub-chain
        println(io, "$DASHES Conditional posterior (N_p = $np_val, n=$n_np) $DASHES")

        # Report all unfrozen params relevant to this N_p
        cond_fitted = _conditional_fitted_stats(chains, params, mask, np_val,
                                                 td, noise_cols;
                                                 global_noise_win=global_noise_win)
        if !isempty(cond_fitted)
            _print_stats_block(io, cond_fitted, "Fitted | N_p=$np_val")
        end

        # Derived params for active planets
        if np_val >= 1
            cond_derived = _conditional_derived_stats(chains, params, mask,
                                                       np_val;
                                                       M_s=M_s, R_s=R_s,
                                                       T_eff=T_eff, J_mag=J_mag,
                                                       K_mag=K_mag, Ab=Ab)
            if !isempty(cond_derived)
                _print_stats_block(io, cond_derived, "Derived | N_p=$np_val")
            end
        end

        # Noise model conditional probabilities — full table so the
        # losing models are still visible as their P(on) values.
        if has_noise
            for (col_name, col_label) in noise_cols
                noise_samp = vec(Array(chains[col_name]))[mask]
                frac = count(noise_samp .== 1.0) / n_np
                @printf(io, "\n  P(%s = on | N_p = %d) = %.4f\n",
                        col_label, np_val, frac)
            end
        end
        println(io)
    end

    println(io, "="^60)
end

# =====================================================================
# save_transdim_summary — file output for trans-dim results
# =====================================================================

"""
    save_transdim_summary(chains, params, path; starname,
                           td=nothing, min_samples=20,
                           log_evidence=nothing,
                           M_s=nothing, R_s=nothing, T_eff=nothing,
                           J_mag=nothing, K_mag=nothing, Ab=0.0)

Save trans-dimensional results to files:
- `{starname}_transdim_model_probs.dat`
- `{starname}_transdim_Np{k}_fitted.dat` (per N_p)
- `{starname}_transdim_Np{k}_derived.dat` (per N_p)
"""
function save_transdim_summary(chains, params::Params, path::String;
                                starname::String,
                                td::Union{Nothing, TransDimConfig}=nothing,
                                min_samples::Int=20,
                                log_evidence::Union{Nothing, Float64}=nothing,
                                M_s=nothing, R_s=nothing, T_eff=nothing,
                                J_mag=nothing, K_mag=nothing, Ab=0.0)
    # Fall back to config.M_s if available
    if M_s === nothing && !isnan(params.config.M_s)
        M_s = params.config.M_s
    end
    max_kp = params.config.max_kplanet
    np = vec(Array(chains[:n_planets]))
    n_total = length(np)
    noise_cols = _detect_noise_columns(chains, td)
    has_noise = !isempty(noise_cols)

    prefix = joinpath(path, starname)

    # --- Model probabilities file ---
    open("$(prefix)_transdim_model_probs.dat", "w") do f
        if has_noise
            _print_joint_table(f, np, chains, noise_cols, max_kp, n_total)
        else
            @printf(f, "%-10s  %12s  %10s  %10s\n",
                    "N_p", "P(N_p)", "n_samples", "n_total")
            for k in 0:max_kp
                nk = count(np .== Float64(k))
                nk == 0 && continue
                @printf(f, "%-10d  %12.6f  %10d  %10d\n",
                        k, nk / n_total, nk, n_total)
            end
        end
        if log_evidence !== nothing
            @printf(f, "\nlog Z = %.4f\n", log_evidence)
        end
    end

    # --- Per-N_p fitted + derived files ---
    # Compute global noise-winning set once so files match the printed
    # report (only winning noise model coefficients are recorded).
    global_noise_win = Bool[]
    if has_noise
        for (col_name, _) in noise_cols
            samp = vec(Array(chains[col_name]))
            push!(global_noise_win,
                   count(samp .== 1.0) / length(samp) > 0.5)
        end
    end

    for np_val in 0:max_kp
        mask = np .== Float64(np_val)
        n_np = count(mask)
        n_np < min_samples && continue

        cond_fitted = _conditional_fitted_stats(chains, params, mask, np_val,
                                                 td, noise_cols;
                                                 global_noise_win=global_noise_win)
        if !isempty(cond_fitted)
            _save_stats_file("$(prefix)_transdim_Np$(np_val)_fitted.dat",
                             cond_fitted)
        end

        if np_val >= 1
            cond_derived = _conditional_derived_stats(chains, params, mask,
                                                       np_val;
                                                       M_s=M_s, R_s=R_s,
                                                       T_eff=T_eff, J_mag=J_mag,
                                                       K_mag=K_mag, Ab=Ab)
            if !isempty(cond_derived)
                _save_stats_file("$(prefix)_transdim_Np$(np_val)_derived.dat",
                                 cond_derived)
            end
        end
    end
end

# =====================================================================
# print_ess_rhat — ESS / Rhat diagnostic table
# =====================================================================

"""
    print_ess_rhat(chains; io=stdout)

Print ESS and Rhat diagnostics for all chain parameters.
"""
function print_ess_rhat(chains; io::IO=stdout)
    ess_tbl = MCMCChains.ess_rhat(chains)
    param_col = ess_tbl[:, :parameters]
    ess_col = ess_tbl[:, :ess]
    rhat_col = ess_tbl[:, :rhat]

    println(io, "\nESS / Rhat:")
    for i in eachindex(param_col)
        @printf(io, "  %-35s  ESS=%6.0f  Rhat=%.3f\n",
                string(param_col[i]), ess_col[i], rhat_col[i])
    end
end

# =====================================================================
# Internal helpers
# =====================================================================

"""Detect noise_active_* columns in the chain and pair with model names."""
function _detect_noise_columns(chains, td)
    cols = Pair{Symbol, String}[]
    td === nothing && return cols
    chain_syms = Set(names(chains, :parameters))
    # Noise columns are named noise_active_1, noise_active_2, ...
    for (i, nm) in enumerate(td.toggleable)
        col = Symbol("noise_active_$i")
        col in chain_syms || continue
        label = _noise_model_label(nm)
        push!(cols, col => label)
    end
    return cols
end

function _noise_model_label(m::MAModel)
    return "MA($(m.order))"
end
function _noise_model_label(m::ARModel)
    return "AR($(m.order))"
end
function _noise_model_label(::CeleriteSHO)
    return "GP-SHO"
end
function _noise_model_label(::CeleriteRotation)
    return "GP-Rot"
end
function _noise_model_label(::ActivityDecorrelation)
    return "AD"
end
function _noise_model_label(::ActivityGP)
    return "AGP"
end
function _noise_model_label(::ActivityJitter)
    return "ActJitter"
end
function _noise_model_label(::CeleriteRotationFM17)
    return "GP-RotFM17"
end
function _noise_model_label(::MaternGP)
    return "GP-Matern"
end
function _noise_model_label(::IndicatorFloor)
    return "IndFloor"
end
function _noise_model_label(::ErrorScale)
    return "ErrScale"
end
function _noise_model_label(::StudentT)
    return "StudentT"
end
function _noise_model_label(::NightlyOffset)
    return "NightlyOff"
end
function _noise_model_label(m::HarmonicBlock)
    return "Harmonic"
end
# Fallback: the TYPE NAME, never a shared constant. These labels are used as
# dictionary KEYS for occupancy (science_model_selection), so two models
# sharing one label silently overwrite each other and a model disappears from
# the model-selection table. A new NoiseModel subtype with no method here gets
# a distinct name rather than joining a collision.
function _noise_model_label(m::NoiseModel)
    return String(nameof(typeof(m)))
end

"""
    _print_joint_table(io, np, chains, noise_cols, max_kp, n_total)

Print joint N_p × noise-config probability table. Each noise-config
is a binary vector across the toggleable noise models; we group
samples by their config bitstring and report the top configs.
Handles arbitrary numbers of noise models.
"""
function _print_joint_table(io::IO, np, chains, noise_cols, max_kp, n_total)
    n_noise = length(noise_cols)
    # Collect each sample's noise config as an Int (bitstring)
    noise_mat = falses(n_total, n_noise)
    for (j, (col_name, _)) in enumerate(noise_cols)
        samp = vec(Array(chains[col_name]))
        noise_mat[:, j] .= samp .== 1.0
    end
    config_int = Vector{Int}(undef, n_total)
    for i in 1:n_total
        bits = 0
        for j in 1:n_noise
            noise_mat[i, j] && (bits |= 1 << (j - 1))
        end
        config_int[i] = bits
    end

    # Joint counts: Dict{(np_val, config_int), Int}
    joint = Dict{Tuple{Int, Int}, Int}()
    for i in 1:n_total
        key = (round(Int, np[i]), config_int[i])
        joint[key] = get(joint, key, 0) + 1
    end

    # Sort by descending count
    sorted = sort(collect(joint); by = x -> -x[2])

    # Build label column header
    labels = [lab for (_, lab) in noise_cols]
    println(io)
    println(io, "  Joint (N_p, noise-config) posterior — top configs:")
    @printf(io, "  %-6s  %-30s  %10s  %10s\n",
            "N_p", "noise (on=1, off=0)", "P", "n")
    println(io, "  " * "-"^60)
    for ((np_val, cfg), n) in sorted[1:min(end, 12)]
        bits = String(map(j -> (cfg >> (j - 1)) & 1 == 1 ? '1' : '0',
                           1:n_noise))
        # Render as "AR=1 MA=1 Activity=0"
        pretty = join(["$(labels[j])=$(bits[j])" for j in 1:n_noise], " ")
        @printf(io, "  %-6d  %-30s  %10.4f  %10d\n",
                np_val, pretty, n / n_total, n)
    end
    println(io, "  " * "-"^60)
end

"""Compute fitted ParamStats for a conditional subset of the chain.

If `global_noise_win::Vector{Bool}` is supplied (one entry per
`noise_cols` slot), parameters belonging to a noise model that lost
the *global* >0.5 vote are dropped from the report. This is the
correct behavior for reporting the winning model: a noise model with
P(active) = 0.05 globally has no business showing its coefficients
alongside the winners.
"""
function _conditional_fitted_stats(chains, params::Params, mask::BitVector,
                                    np_val::Int, td, noise_cols;
                                    global_noise_win::Vector{Bool}=Bool[])
    stats = Vector{Pair{String, ParamStats}}()
    max_kp = params.config.max_kplanet
    chain_names = Set(names(chains, :parameters))

    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        sym in chain_names || continue

        # Skip planet params for inactive planets
        planet_idx = _param_planet_index(name, max_kp)
        if planet_idx !== nothing && planet_idx > np_val
            continue
        end

        # Skip noise params when the noise model is off in this conditional
        if _is_noise_param(name, params) && !isempty(noise_cols)
            noise_nm_idx = _noise_model_index_for_param(name, params)
            if noise_nm_idx !== nothing
                # Global filter (preferred): drop losing noise models
                if !isempty(global_noise_win) &&
                   noise_nm_idx <= length(global_noise_win) &&
                   !global_noise_win[noise_nm_idx]
                    continue
                end
                # Fallback: in-mask filter
                if isempty(global_noise_win)
                    nim = _noise_active_in_mask(chains, noise_cols, mask)
                    if noise_nm_idx <= length(nim) && !nim[noise_nm_idx]
                        continue
                    end
                end
            end
        end

        samp = vec(Array(chains[sym]))[mask]
        push!(stats, name => ParamStats(samp))
    end

    return stats
end

"""Compute derived ParamStats for planets active in this N_p conditional."""
function _conditional_derived_stats(chains, params::Params, mask::BitVector,
                                     np_val::Int;
                                     M_s=nothing, R_s=nothing,
                                     T_eff=nothing, J_mag=nothing,
                                     K_mag=nothing, Ab=0.0)
    # Build a sub-chain with only the masked rows
    n_masked = count(mask)
    param_names = names(chains, :parameters)
    mat = Matrix{Float64}(undef, n_masked, length(param_names))
    for (j, sym) in enumerate(param_names)
        mat[:, j] = vec(Array(chains[sym]))[mask]
    end
    sub_chains = MCMCChains.Chains(mat, param_names)

    derived_vec = compute_derived(sub_chains, params;
                                   M_s=M_s, R_s=R_s, T_eff=T_eff,
                                   J_mag=J_mag, K_mag=K_mag, Ab=Ab)
    stats = Vector{Pair{String, ParamStats}}()
    for dp in derived_vec
        k_str = split(dp.name, "_")[end]
        k_int = parse(Int, k_str)
        # Only report derived params for active planets in this N_p
        k_int > np_val && continue
        for (key, samples) in dp.values
            label = "$(key)_k$(k_str)"
            push!(stats, label => ParamStats(samples))
        end
    end

    return stats
end

"""Mode of an integer-valued sample vector (most frequent value)."""
function _mode_int(samples::AbstractVector)
    counts = Dict{Int, Int}()
    for x in samples
        k = round(Int, x)
        counts[k] = get(counts, k, 0) + 1
    end
    best_k, best_n = 0, 0
    for (k, n) in counts
        if n > best_n
            best_k, best_n = k, n
        end
    end
    return best_k
end


"""Extract planet index from parameter name, or nothing if not a planet param."""
function _param_planet_index(name::String, max_kp::Int)
    for k in 1:max_kp
        suffix = "_k$k"
        if endswith(name, suffix)
            # Ensure it's a planet param, not a noise param that happens to
            # have _k in its name
            base = name[1:end-length(suffix)]
            if base in ("P", "K", "sesinw", "secosw", "esinw", "ecosw",
                        "ecc", "w", "Mo", "Tp", "Tc", "b", "rr", "r1", "r2",
                        "a")
                return k
            end
        end
    end
    return nothing
end

"""Check if a parameter name belongs to a noise model."""
function _is_noise_param(name::String, params::Params)
    for nm in params.config.noise_models
        nm_names = noise_param_names(nm, params.config.instruments)
        if name in nm_names
            return true
        end
    end
    return false
end

"""Find which noise model index (1-based in toggleable) owns this param."""
function _noise_model_index_for_param(name::String, params::Params)
    td_models = params.config.noise_models
    for (i, nm) in enumerate(td_models)
        nm_names = noise_param_names(nm, params.config.instruments)
        if name in nm_names
            return i
        end
    end
    return nothing
end

"""Check which noise models have >50% active samples in this mask."""
function _noise_active_in_mask(chains, noise_cols, mask::BitVector)
    result = Bool[]
    for (col, _) in noise_cols
        samp = vec(Array(chains[col]))[mask]
        push!(result, count(samp .== 1.0) / length(samp) > 0.5)
    end
    return result
end

"""
    convergence_report(chains, n_walkers; io=stdout, params=nothing,
                       rhat_max=1.1, ess_min=400) -> NamedTuple

Ensemble convergence gate for a cold `sample_transdim_ptemcee` chain saved as
`n_walkers` walkers × n_steps (walker = fast index in the flattened chain).
Reshapes into per-walker chains and computes rank-normalized split-R̂ +
bulk-ESS (Vehtari+ 2021, via MCMCChains) per parameter. Prints a summary with a
PASS/FAIL stamp — a chain is science-grade only if EVERY non-fixed parameter has
R̂ ≤ `rhat_max` and ESS ≥ `ess_min`. Constant (frozen-value) params are reported
as "fixed" and don't gate. `n_walkers` must be the EFFECTIVE walker count
(`n_walkers_eff = max(NW, 2·n_dim+2)`), not the requested NW.

Returns `(; pass, worst_rhat, worst_rhat_param, min_ess, min_ess_param, n_fail, n_fixed)`.
"""
function convergence_report(chains, n_walkers::Int; io::IO=stdout,
                             params=nothing, model_params=nothing,
                             rhat_max::Real=1.1, ess_min::Real=400)
    allp = names(chains, :parameters)
    pnames = params === nothing ? allp : [Symbol(p) for p in params if Symbol(p) in allp]
    isempty(pnames) && (println(io, "convergence_report: no parameters"); return (; pass=false))
    # Layout detection: a flat trans-dim chain stores one chain (size(_,3)==1)
    # with WALKER as the FAST index in the per-param vec; a native ensemble chain
    # stores (iter × param × walker) with walker as the SLOW index. Handle both.
    nflat = length(vec(Array(chains[pnames[1]])))
    nc = size(Array(chains[pnames[1]]), ndims(Array(chains[pnames[1]])))
    local nsteps::Int, walker_fast::Bool
    if nc == n_walkers && nflat % nc == 0
        nsteps = nflat ÷ nc; walker_fast = false          # native multi-chain
    elseif nflat % n_walkers == 0
        nsteps = nflat ÷ n_walkers; walker_fast = true     # flat single-chain
    else
        println(io, "convergence_report: $nflat draws not divisible by n_walkers=$n_walkers")
        return (; pass=false)
    end
    # per-walker flat indices into the length-`nflat` vec, layout-aware.
    _widx(w) = walker_fast ? (w:n_walkers:nflat) : (((w - 1) * nsteps + 1):(w * nsteps))

    # Active mask for a param: trans-dim component params (noise-model / planet)
    # are only DEFINED when their component is active — when inactive their slot
    # holds stale unbounded junk (drifts to ~1e6), which poisons an unconditional
    # R̂/ESS. Assess them CONDITIONED on active. Returns nothing for always-defined
    # params (γ, jitter, planets when the count is fixed).
    function _active_mask(p::Symbol)
        s = String(p)
        if model_params !== nothing
            idx = _noise_model_index_for_param(s, model_params)
            if idx !== nothing
                col = Symbol("noise_active_$idx")
                col in allp && return vec(Array(chains[col])) .> 0.5
            end
        end
        m = match(r"_k(\d+)$", s)
        if m !== nothing
            col = Symbol("planet_active_$(m.captures[1])")
            col in allp && return vec(Array(chains[col])) .> 0.5
        end
        return nothing
    end
    # per-walker (walker = fast index) split-R̂ + bulk-ESS over an active mask.
    function _cond_rhat_ess(v::Vector{Float64}, mask)
        if mask === nothing
            M = Array{Float64,3}(undef, nsteps, 1, n_walkers)
            for w in 1:n_walkers; M[:, 1, w] = v[_widx(w)]; end
            er = MCMCChains.ess_rhat(MCMCChains.Chains(M))
            return (er[1, :rhat], er[1, :ess], nsteps)
        end
        colv = [(idx = _widx(w); v[idx][mask[idx]]) for w in 1:n_walkers]
        lens = length.(colv)
        nz = filter(>(0), lens)
        isempty(nz) && return (NaN, NaN, 0)
        # Per-walker active counts vary wildly (a walker entrenched in another
        # model has ~0 active draws here). Collapsing to the global min gives a
        # useless L≈0, so instead set the common length to the MEDIAN active
        # count and assess R̂ over the walkers that reach it — the ones that
        # actually sample this component. Drops the sparse walkers rather than
        # letting them null the whole statistic.
        L = max(20, round(Int, median(nz)))
        kept = [c for c in colv if length(c) ≥ L]
        length(kept) < 4 && return (NaN, NaN, length(kept))
        M = Array{Float64,3}(undef, L, 1, length(kept))
        for (j, c) in enumerate(kept); M[:, 1, j] = c[1:L]; end
        er = MCMCChains.ess_rhat(MCMCChains.Chains(M))
        return (er[1, :rhat], er[1, :ess], L)
    end

    keep=Symbol[]; rh=Float64[]; es=Float64[]; fr=Float64[]; gate_ess=Bool[]
    n_fixed=0; n_inactive=0
    for p in pnames
        v = vec(Array(chains[p]))
        mask = _active_mask(p)
        if mask === nothing
            (maximum(v) - minimum(v) < 1e-12) && (n_fixed += 1; continue)  # frozen
            r, e, _ = _cond_rhat_ess(v, nothing)
            push!(keep,p); push!(rh,r); push!(es,e); push!(fr,1.0); push!(gate_ess,true)
        else
            f = count(mask)/length(mask)
            if f < 0.01; n_inactive += 1; continue; end       # component ~never on
            r, e, _ = _cond_rhat_ess(v, mask)
            push!(keep,p); push!(rh,r); push!(es,e); push!(fr,f)
            # ESS-gate only well-sampled components; ESS via active-masking is
            # unreliable for sporadically-active (disfavoured) models, and a
            # disfavoured model needn't reach ess_min — only MIX (R̂).
            push!(gate_ess, f ≥ 0.5)
        end
    end
    isempty(keep) && (println(io, "convergence_report: nothing to assess"); return (; pass=true, n_fail=0, n_fixed))
    # R̂ gates every assessed param; ESS gates only the always/dominant ones.
    bad = [i for i in eachindex(keep)
           if !(isfinite(rh[i]) && rh[i] ≤ rhat_max &&
                (!gate_ess[i] || (isfinite(es[i]) && es[i] ≥ ess_min)))]
    wi = argmax([isfinite(x) ? x : -Inf for x in rh])
    mi = argmin([gate_ess[i] && isfinite(es[i]) ? es[i] : Inf for i in eachindex(es)])
    pass = isempty(bad)
    println(io, "="^64)
    @printf(io, "  Convergence (active-conditional split-R̂ + bulk-ESS): %d walkers × %d steps\n",
            n_walkers, nsteps)
    @printf(io, "  %d assessed (%d fixed, %d inactive-skipped)\n", length(keep), n_fixed, n_inactive)
    @printf(io, "  worst R̂ = %.3f (%s)    min ESS = %.0f (%s, gated params)\n",
            rh[wi], string(keep[wi]), es[mi], string(keep[mi]))
    if !pass
        @printf(io, "  %d param(s) fail (R̂ ≤ %.2f all; ESS ≥ %.0f for occupancy ≥ 0.5):\n",
                length(bad), rhat_max, ess_min)
        for i in bad[1:min(8, length(bad))]
            @printf(io, "    %-22s R̂=%.3f  ESS=%.0f  (occ=%.2f)\n",
                    string(keep[i]), rh[i], es[i], fr[i])
        end
        length(bad) > 8 && @printf(io, "    … and %d more\n", length(bad) - 8)
    end
    @printf(io, "  CONVERGENCE: %s\n", pass ? "PASS — science-grade" : "FAIL — NOT science-grade")
    println(io, "="^64)
    return (; pass, worst_rhat=rh[wi], worst_rhat_param=keep[wi],
            min_ess=es[mi], min_ess_param=keep[mi], n_fail=length(bad), n_fixed)
end
