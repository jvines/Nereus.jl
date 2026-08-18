# Activity decorrelation: rv_pred += sum_j C_j * indicator_j(t)
#
# Reads correlation coefficients from theta, indicator values from
# data.indicators. Raw indicator values — no preprocessing. The
# coefficients absorb the scale.

"""
    apply_activity_jitter(variance, theta, data, model, ins_idx, obs_idx) -> variance

Replace the constant jitter with activity-dependent jitter:
`σ²_eff = σ²_obs + (jit_base + jit_act * indicator(t))²`

Note: when ActivityJitter is active, the constant per-instrument sigma
is NOT added (the base jitter replaces it). The user should set the
constant sigma prior to a small fixed value or remove it.
"""
@inline function apply_activity_jitter(
    obs_err_sq, theta::Theta{T}, data::Data,
    model::ActivityJitter, ins_idx::Int, obs_idx::Int,
) where {T}
    layout = theta.params.layout
    ins_name = theta.params.config.instruments.rv_names[ins_idx]
    jit_base = theta.values[layout.name_to_idx["jit_base_$(model.indicator)_$(ins_name)"]]
    jit_act  = theta.values[layout.name_to_idx["jit_act_$(model.indicator)_$(ins_name)"]]
    ind_val  = T(data.indicators[model.indicator][obs_idx])
    sigma_j  = jit_base + jit_act * ind_val
    return obs_err_sq + sigma_j * sigma_j
end

"""
    apply_activity_decorrelation(pred, theta, data, model, ins_idx, obs_idx) -> pred

Add activity indicator correlations to the prediction for observation
`obs_idx` with instrument `ins_idx`. Returns the updated prediction.
"""
@inline function apply_activity_decorrelation(
    pred, theta::Theta{T}, data::Data,
    model::ActivityDecorrelation, ins_idx::Int, obs_idx::Int,
) where {T}
    layout = theta.params.layout
    instruments = theta.params.config.instruments
    suf = _ad_suffix(model)
    for ind_name in model.indicators
        ind_val = data.indicators[ind_name][obs_idx]
        isfinite(ind_val) || continue  # skip NaN/missing indicators
        pname = model.per_instrument ?
            "C_$(ind_name)_$(instruments.rv_names[ins_idx])$suf" :
            "C_$(ind_name)$suf"
        idx = layout.name_to_idx[pname]
        C = theta.values[idx]
        pred += C * T(ind_val)

        # FF'-style derivative term: pred += Cdot · d(indicator)/dt.
        # The per-instrument central-difference derivative is precomputed
        # at Data construction; NaN at endpoints/gaps is skipped.
        if model.derivative
            d_val = data.indicator_derivs[ind_name][obs_idx]
            isfinite(d_val) || continue
            pdname = model.per_instrument ?
                "Cdot_$(ind_name)_$(instruments.rv_names[ins_idx])$suf" :
                "Cdot_$(ind_name)$suf"
            Cdot = theta.values[layout.name_to_idx[pdname]]
            pred += Cdot * T(d_val)
        end
    end
    return pred
end
