# Activity decorrelation: rv_pred += sum_j C_j * indicator_j(t)
#
# Reads correlation coefficients from theta, indicator values from
# data.indicators. Raw indicator values — no preprocessing. The
# coefficients absorb the scale.

# Parameter names of the per-point RV terms (see noise/param_names.jl).
_aj_base_name(m::ActivityJitter, ins) = "jit_base_$(m.indicator)_$(ins)"
_aj_act_name(m::ActivityJitter, ins)  = "jit_act_$(m.indicator)_$(ins)"
_ad_coef_name(m::ActivityDecorrelation, ind, ins) =
    m.per_instrument ? "C_$(ind)_$(ins)$(_ad_suffix(m))" : "C_$(ind)$(_ad_suffix(m))"
_ad_cdot_name(m::ActivityDecorrelation, ind, ins) =
    m.per_instrument ? "Cdot_$(ind)_$(ins)$(_ad_suffix(m))" : "Cdot_$(ind)$(_ad_suffix(m))"
_errscale_name(ins) = "errscale_$(ins)"

"""
    apply_activity_jitter(variance, theta, data, model, ins_idx, obs_idx) -> variance

Replace the constant jitter with activity-dependent jitter:
`σ²_eff = σ²_obs + (jit_base + jit_act * indicator(t))²`

Note: when ActivityJitter is active, the constant per-instrument sigma
is NOT added (the base jitter replaces it). The user should set the
constant sigma prior to a small fixed value or remove it.

The likelihood loops use `_aj_variance`, the same arithmetic with the
layout slots resolved once (`RVModifierSlots`).
"""
@inline function apply_activity_jitter(
    obs_err_sq, theta::Theta{T}, data::Data,
    model::ActivityJitter, ins_idx::Int, obs_idx::Int,
) where {T}
    layout = theta.params.layout
    ins_name = theta.params.config.instruments.rv_names[ins_idx]
    jit_base = theta.values[layout.name_to_idx[_aj_base_name(model, ins_name)]]
    jit_act  = theta.values[layout.name_to_idx[_aj_act_name(model, ins_name)]]
    ind_val  = T(data.indicators[model.indicator][obs_idx])
    sigma_j  = jit_base + jit_act * ind_val
    return obs_err_sq + sigma_j * sigma_j
end

"""
    apply_activity_decorrelation(pred, theta, data, model, ins_idx, obs_idx) -> pred

Add activity indicator correlations to the prediction for observation
`obs_idx` with instrument `ins_idx`. Returns the updated prediction.

The likelihood loops use `_ad_term`, the same arithmetic with the layout
slots resolved once (`RVModifierSlots`).
"""
@inline function apply_activity_decorrelation(
    pred, theta::Theta{T}, data::Data,
    model::ActivityDecorrelation, ins_idx::Int, obs_idx::Int,
) where {T}
    layout = theta.params.layout
    instruments = theta.params.config.instruments
    for ind_name in model.indicators
        ind_val = data.indicators[ind_name][obs_idx]
        isfinite(ind_val) || continue  # skip NaN/missing indicators
        pname = _ad_coef_name(model, ind_name, instruments.rv_names[ins_idx])
        idx = layout.name_to_idx[pname]
        C = theta.values[idx]
        pred += C * T(ind_val)

        # FF'-style derivative term: pred += Cdot · d(indicator)/dt.
        # The per-instrument central-difference derivative is precomputed
        # at Data construction; NaN at endpoints/gaps is skipped.
        if model.derivative
            d_val = data.indicator_derivs[ind_name][obs_idx]
            isfinite(d_val) || continue
            pdname = _ad_cdot_name(model, ind_name, instruments.rv_names[ins_idx])
            Cdot = theta.values[layout.name_to_idx[pdname]]
            pred += Cdot * T(d_val)
        end
    end
    return pred
end

# =====================================================================
# Layout slots of the per-point RV terms, resolved once
# =====================================================================
#
# ActivityDecorrelation, ActivityJitter and ErrorScale used to build each
# coefficient's name and look it up for every (point, indicator), and to
# look the indicator vector up by name for every point. `RVModifierSlots`
# holds those slots per noise model and per RV instrument, resolved once
# per (layout, noise-model list, instrument list), plus the indicator
# vectors, looked up once per call. The terms themselves (`_ad_term`,
# `_aj_variance`, `_es_variance`) do the same arithmetic as the reference
# functions above, operation for operation.
#
# A slot that does not exist is stored as 0 and raises, at the point where
# the reference function would have raised, the same KeyError; a missing
# indicator vector likewise.

const _NO_INDICATOR = Float64[]   # sentinel: indicator absent from the data

mutable struct RVModifierSlots
    key_idx::Any                    # layout.name_to_idx the slots come from
    key_models::Any                 # the noise_models vector
    key_inst::Any                   # instruments.rv_names
    # per noise model (empty for models of another kind)
    ad_c::Vector{Matrix{Int}}       # n_ind × n_inst slot of C_<ind>[_<ins>]
    ad_d::Vector{Matrix{Int}}       # same for Cdot (only when derivative)
    aj_base::Vector{Vector{Int}}    # n_inst slots of jit_base_<ind>_<ins>
    aj_act::Vector{Vector{Int}}     # n_inst slots of jit_act_<ind>_<ins>
    es_cov::Vector{Vector{Bool}}    # does the ErrorScale cover instrument q
    es_f::Vector{Vector{Int}}       # n_inst slots of errscale_<ins>
    # indicator vectors, refreshed on every call
    ad_vals::Vector{Vector{Vector{Float64}}}
    ad_dvals::Vector{Vector{Vector{Float64}}}
    aj_vals::Vector{Vector{Float64}}
end

RVModifierSlots() = RVModifierSlots(nothing, nothing, nothing,
    Matrix{Int}[], Matrix{Int}[], Vector{Int}[], Vector{Int}[], Vector{Bool}[],
    Vector{Int}[], Vector{Vector{Float64}}[], Vector{Vector{Float64}}[],
    Vector{Float64}[])

"""
    _modifier_slots!(sl, theta, data, models = theta.params.config.noise_models) -> sl

Bring `sl` up to date for `theta`'s layout and the noise-model list
`models` (re-resolving the slots only when one of them changed) and look
up `data`'s indicator vectors.
"""
function _modifier_slots!(sl::RVModifierSlots, theta::Theta, data::Data,
                          models = theta.params.config.noise_models)
    idx    = theta.params.layout.name_to_idx
    inst   = theta.params.config.instruments.rv_names
    if !(sl.key_idx === idx && sl.key_models === models && sl.key_inst === inst)
        _resolve_modifier_slots!(sl, idx, models, inst)
    end
    @inbounds for m in eachindex(models)
        nm = models[m]
        if nm isa ActivityDecorrelation
            v = sl.ad_vals[m]
            dv = sl.ad_dvals[m]
            for k in eachindex(nm.indicators)
                v[k] = get(data.indicators, nm.indicators[k], _NO_INDICATOR)
                nm.derivative &&
                    (dv[k] = get(data.indicator_derivs, nm.indicators[k], _NO_INDICATOR))
            end
        elseif nm isa ActivityJitter
            sl.aj_vals[m] = get(data.indicators, nm.indicator, _NO_INDICATOR)
        end
    end
    return sl
end

function _resolve_modifier_slots!(sl::RVModifierSlots, idx, models, inst)
    nm_ = length(models)
    nq = length(inst)
    sl.ad_c     = [zeros(Int, 0, 0) for _ in 1:nm_]
    sl.ad_d     = [zeros(Int, 0, 0) for _ in 1:nm_]
    sl.aj_base  = [Int[] for _ in 1:nm_]
    sl.aj_act   = [Int[] for _ in 1:nm_]
    sl.es_cov   = [Bool[] for _ in 1:nm_]
    sl.es_f     = [Int[] for _ in 1:nm_]
    sl.ad_vals  = [Vector{Float64}[] for _ in 1:nm_]
    sl.ad_dvals = [Vector{Float64}[] for _ in 1:nm_]
    sl.aj_vals  = [_NO_INDICATOR for _ in 1:nm_]
    for (m, nm) in enumerate(models)
        if nm isa ActivityDecorrelation
            ni = length(nm.indicators)
            c = zeros(Int, ni, nq)
            d = zeros(Int, nm.derivative ? ni : 0, nm.derivative ? nq : 0)
            for k in 1:ni, q in 1:nq
                c[k, q] = get(idx, _ad_coef_name(nm, nm.indicators[k], inst[q]), 0)
                nm.derivative &&
                    (d[k, q] = get(idx, _ad_cdot_name(nm, nm.indicators[k], inst[q]), 0))
            end
            sl.ad_c[m] = c
            sl.ad_d[m] = d
            sl.ad_vals[m]  = fill(_NO_INDICATOR, ni)
            sl.ad_dvals[m] = fill(_NO_INDICATOR, nm.derivative ? ni : 0)
        elseif nm isa ActivityJitter
            sl.aj_base[m] = [get(idx, _aj_base_name(nm, inst[q]), 0) for q in 1:nq]
            sl.aj_act[m]  = [get(idx, _aj_act_name(nm, inst[q]), 0) for q in 1:nq]
        elseif nm isa ErrorScale
            sl.es_cov[m] = [isempty(nm.instruments) || inst[q] in nm.instruments for q in 1:nq]
            sl.es_f[m]   = [get(idx, _errscale_name(inst[q]), 0) for q in 1:nq]
        end
    end
    sl.key_idx = idx
    sl.key_models = models
    sl.key_inst = inst
    return sl
end

@noinline _missing_slot(name) = throw(KeyError(name))

# `apply_activity_decorrelation` with resolved slots; `m` is the model's
# index in the noise-model list.
@inline function _ad_term(pred, theta::Theta{T}, data::Data,
                          model::ActivityDecorrelation, sl::RVModifierSlots,
                          m::Int, ins_idx::Int, obs_idx::Int) where {T}
    vals = sl.ad_vals[m]
    cidx = sl.ad_c[m]
    for k in eachindex(vals)
        vk = vals[k]
        vk === _NO_INDICATOR && _missing_slot(model.indicators[k])
        ind_val = vk[obs_idx]
        isfinite(ind_val) || continue  # skip NaN/missing indicators
        ci = cidx[k, ins_idx]
        ci == 0 && _missing_slot(_ad_coef_name(model, model.indicators[k],
                                 theta.params.config.instruments.rv_names[ins_idx]))
        C = theta.values[ci]
        pred += C * T(ind_val)
        if model.derivative
            dk = sl.ad_dvals[m][k]
            dk === _NO_INDICATOR && _missing_slot(model.indicators[k])
            d_val = dk[obs_idx]
            isfinite(d_val) || continue
            di = sl.ad_d[m][k, ins_idx]
            di == 0 && _missing_slot(_ad_cdot_name(model, model.indicators[k],
                                     theta.params.config.instruments.rv_names[ins_idx]))
            Cdot = theta.values[di]
            pred += Cdot * T(d_val)
        end
    end
    return pred
end

# `apply_activity_jitter` with resolved slots.
@inline function _aj_variance(obs_err_sq, theta::Theta{T}, data::Data,
                              model::ActivityJitter, sl::RVModifierSlots,
                              m::Int, ins_idx::Int, obs_idx::Int) where {T}
    bi = sl.aj_base[m][ins_idx]
    bi == 0 && _missing_slot(_aj_base_name(model,
                             theta.params.config.instruments.rv_names[ins_idx]))
    jit_base = theta.values[bi]
    ai = sl.aj_act[m][ins_idx]
    ai == 0 && _missing_slot(_aj_act_name(model,
                             theta.params.config.instruments.rv_names[ins_idx]))
    jit_act = theta.values[ai]
    vals = sl.aj_vals[m]
    vals === _NO_INDICATOR && _missing_slot(model.indicator)
    ind_val = T(vals[obs_idx])
    sigma_j = jit_base + jit_act * ind_val
    return obs_err_sq + sigma_j * sigma_j
end

# ErrorScale's f²·σ_formal² for a covered instrument (`sl.es_cov[m][ins_idx]`),
# as `error_scale_factor(theta, model, ins_idx) * obs_err * obs_err`.
@inline function _es_variance(obs_err, theta::Theta{T}, model::ErrorScale,
                              sl::RVModifierSlots, m::Int, ins_idx::Int) where {T}
    fi = sl.es_f[m][ins_idx]
    fi == 0 && _missing_slot(_errscale_name(
                             theta.params.config.instruments.rv_names[ins_idx]))
    f = theta.values[fi]
    f2 = f * f
    return f2 * obs_err * obs_err
end
