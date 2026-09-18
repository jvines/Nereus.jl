# OLS-informed birth for ActivityDecorrelation (linear-Gaussian).
#
# The one informed proposal in this codebase that demonstrably works, and the
# template the Thiele-Innes coupling proposal should follow.
#
# Split out of the former single proposals.jl (2035 lines). Pure move: no logic
# changed, and the split was verified to reconstruct the original byte for byte.

# =====================================================================
# OLS-informed birth for ActivityDecorrelation (linear-Gaussian)
# =====================================================================
"""
    ad_ols_fit(theta, data, model) -> (cnames, C_ols, prec_chol)

Weighted-OLS fit of an `ActivityDecorrelation` model's linear
coefficients on the CURRENT residual (RV minus the mean model WITHOUT
AD — at a birth `theta` has AD inactive, so `rv_predictions` already
excludes it). AD adds `Σ_k C_k I_k(t)` to the RV mean, a linear-Gaussian
regression whose coefficients have an exact Gaussian conditional, so
blind prior draws make AD uncompetitive against simpler models in a
trans-dim selection (the climb can't reach the joint optimum for
correlated regressors). `prec_chol` is `cholesky(XᵀWX)` — the proposal
precision (before inflation). Returns empty on degenerate fit.
"""
function ad_ols_fit(theta::Theta{T}, data::Data,
                    model::ActivityDecorrelation) where {T}
    layout = theta.params.layout
    instruments = theta.params.config.instruments
    cnames = String[]; inds = String[]; insts = Int[]
    suf = _ad_suffix(model)
    for ind in model.indicators
        if model.per_instrument
            for (k, ins) in enumerate(instruments.rv_names)
                nm = "C_$(ind)_$(ins)$suf"
                haskey(layout.name_to_idx, nm) &&
                    (push!(cnames, nm); push!(inds, ind); push!(insts, k))
            end
        else
            nm = "C_$(ind)$suf"
            haskey(layout.name_to_idx, nm) &&
                (push!(cnames, nm); push!(inds, ind); push!(insts, 0))
        end
    end
    K = length(cnames)
    K == 0 && return (cnames, Float64[], nothing)
    preds, vars = rv_predictions(theta, data)
    n = length(data.rv)
    X = zeros(Float64, n, K)
    @inbounds for k in 1:K
        iv = data.indicators[inds[k]]
        for i in 1:n
            (insts[k] == 0 || data.rv_inst[i] == insts[k]) || continue
            v = iv[i]
            X[i, k] = isfinite(v) ? Float64(v) : 0.0
        end
    end
    w = Float64[1.0 / vars[i] for i in 1:n]
    r = Float64[data.rv[i] - preds[i] for i in 1:n]
    Xw = X .* w
    XtWX = X' * Xw
    @inbounds for k in 1:K; XtWX[k, k] += 1e-8; end
    F = cholesky(Symmetric(XtWX); check = false)
    issuccess(F) || return (cnames, Float64[], nothing)
    C_ols = F \ (X' * (w .* r))
    return (cnames, C_ols, F)
end

