# Exponentially-damped MA/AR models (Tuomi+ 2013).
# Sequential O(n) evaluation — no matrix inversion.
#
# `apply_ma!`/`apply_ar!` are called by both the RV likelihood (channel
# `:rv`) and the transit likelihood (channel `:phot`). The model's
# `channel` field selects the parameter-name suffix; `per_instrument`
# selects whether one set of coefficients applies to all cadences
# (global) or each instrument has its own and the sequential
# correlation only runs within an instrument's cadences.

"""
    apply_ma!(residuals, times, inst, theta, model)

Apply MA(p) correction to residuals in-place. Sequential:
`residuals[i] -= sum_j omega_j * exp(-dt/beta_j) * residuals[i-j]`

`inst` is the per-cadence 1-based instrument index (`data.rv_inst`
for `:rv` or `data.phot_inst` for `:phot`). The per-instrument path
restricts the lag chain to within-instrument cadences.
"""
function apply_ma!(residuals::AbstractVector{T}, times::Vector{Float64},
                    inst::AbstractVector{Int},
                    theta::Theta{T}, model::MAModel) where {T}
    layout = theta.params.layout
    p = model.order
    s = _channel_suffix(model.channel)
    chan_inst_names = model.channel === :rv ?
                      theta.params.config.instruments.rv_names :
                      theta.params.config.instruments.pm_names

    # Tuomi+ 2013 MA acts on the Stage-1 residuals (the deterministic-
    # model residual), not on the recursively-corrected residuals.
    # astroEMPEROR's moav00.model / moav01.model match this by taking a
    # numpy copy via `res_ = residuals[mask]` before the loop. The
    # Nereus port previously used `residuals[i-j]` directly, which (in
    # the natural Julia loop order) reads the *post-correction* residual
    # and gives an infinite-step recursive smoothing instead of the
    # one-step lag the paper specifies. Snapshot here to match.
    res_orig = copy(residuals)

    if !model.per_instrument
        omegas = Vector{T}(undef, p)
        betas  = Vector{T}(undef, p)
        for j in 1:p
            omegas[j] = theta.values[layout.name_to_idx["ma_omega_$j$s"]]
            betas[j]  = theta.values[layout.name_to_idx["ma_beta_$j$s"]]
        end
        @inbounds for i in 2:length(residuals)
            correction = zero(T)
            for j in 1:min(p, i - 1)
                dt = times[i] - times[i - j]
                correction += omegas[j] * exp(-dt / betas[j]) * res_orig[i - j]
            end
            residuals[i] -= correction
        end
        return residuals
    end

    # Per-instrument path: iterate within each instrument's cadences,
    # using its own (omega, beta) set. `inst` is assumed 1-based and
    # to align with `chan_inst_names` ordering.
    n_inst = length(chan_inst_names)
    omegas = Matrix{T}(undef, p, n_inst)
    betas  = Matrix{T}(undef, p, n_inst)
    for k in 1:n_inst
        ins_name = chan_inst_names[k]
        for j in 1:p
            omegas[j, k] = theta.values[layout.name_to_idx["ma_omega_$(j)_$(ins_name)$s"]]
            betas[j, k]  = theta.values[layout.name_to_idx["ma_beta_$(j)_$(ins_name)$s"]]
        end
    end
    n_obs = length(residuals)
    last_idx = zeros(Int, n_inst, p)  # rolling buffer of last p indices per instrument
    @inbounds for i in 1:n_obs
        k = inst[i]
        correction = zero(T)
        for j in 1:p
            prev = last_idx[k, j]
            prev == 0 && continue
            dt = times[i] - times[prev]
            correction += omegas[j, k] * exp(-dt / betas[j, k]) * res_orig[prev]
        end
        residuals[i] -= correction
        # Shift rolling buffer
        for j in p:-1:2
            last_idx[k, j] = last_idx[k, j - 1]
        end
        last_idx[k, 1] = i
    end
    return residuals
end

"""
    apply_ar!(predictions, times, inst, theta, model)

Apply AR(q) to the model predictions in-place (Tuomi convention:
AR on model, not residuals).
`predictions[i] += sum_j phi_j * exp(-dt/alpha_j) * predictions[i-j]`

`inst` is the per-cadence 1-based instrument index. Per-instrument
mode restricts the lag chain to within-instrument cadences.
"""
function apply_ar!(predictions::AbstractVector{T}, times::Vector{Float64},
                    inst::AbstractVector{Int},
                    theta::Theta{T}, model::ARModel) where {T}
    layout = theta.params.layout
    q = model.order
    s = _channel_suffix(model.channel)
    chan_inst_names = model.channel === :rv ?
                      theta.params.config.instruments.rv_names :
                      theta.params.config.instruments.pm_names

    # Like MA: snapshot predictions before the loop so the AR correction
    # uses the Stage-1 (deterministic-model) prediction, not the
    # recursively-corrected one. Matches the EMPEROR convention.
    pred_orig = copy(predictions)

    if !model.per_instrument
        phis   = Vector{T}(undef, q)
        alphas = Vector{T}(undef, q)
        for j in 1:q
            phis[j]   = theta.values[layout.name_to_idx["ar_phi_$j$s"]]
            alphas[j] = theta.values[layout.name_to_idx["ar_alpha_$j$s"]]
        end
        @inbounds for i in 2:length(predictions)
            ar = zero(T)
            for j in 1:min(q, i - 1)
                dt = times[i] - times[i - j]
                ar += phis[j] * exp(-dt / alphas[j]) * pred_orig[i - j]
            end
            predictions[i] += ar
        end
        return predictions
    end

    n_inst = length(chan_inst_names)
    phis   = Matrix{T}(undef, q, n_inst)
    alphas = Matrix{T}(undef, q, n_inst)
    for k in 1:n_inst
        ins_name = chan_inst_names[k]
        for j in 1:q
            phis[j, k]   = theta.values[layout.name_to_idx["ar_phi_$(j)_$(ins_name)$s"]]
            alphas[j, k] = theta.values[layout.name_to_idx["ar_alpha_$(j)_$(ins_name)$s"]]
        end
    end
    n_obs = length(predictions)
    last_idx = zeros(Int, n_inst, q)
    @inbounds for i in 1:n_obs
        k = inst[i]
        ar = zero(T)
        for j in 1:q
            prev = last_idx[k, j]
            prev == 0 && continue
            dt = times[i] - times[prev]
            ar += phis[j, k] * exp(-dt / alphas[j, k]) * pred_orig[prev]
        end
        predictions[i] += ar
        for j in q:-1:2
            last_idx[k, j] = last_idx[k, j - 1]
        end
        last_idx[k, 1] = i
    end
    return predictions
end
