# GP noise models (Stage 3: covariance).
#
# O(n) celerite solver — generic over element type T for ForwardDiff.
# Based on Foreman-Mackey et al. 2017, AJ, 154, 220.
#
# The celerite kernel is a sum of complex exponentials:
#   k(τ) = Σ_j [a_j cos(d_j τ) + b_j sin(d_j τ)] exp(-c_j τ)   (complex)
#        + Σ_j a_j exp(-c_j τ)                                     (real)
#
# This produces a semi-separable covariance matrix factorable in O(n).

using LinearAlgebra: dot

# =====================================================================
# Kernel coefficient extraction (real + complex terms)
# =====================================================================

"""
    sho_coefficients(S0, Q, ω0) -> (ar, cr, ac, bc, cc, dc)

Return celerite coefficients for a SHO kernel.
Overdamped (Q < 0.5): two real terms. Underdamped (Q ≥ 0.5): one complex term.
"""
function sho_coefficients(S0, Q, ω0)
    T = promote_type(typeof(S0), typeof(Q), typeof(ω0))
    eps = T(1e-5)
    half = T(0.5)
    if Q < half
        f = sqrt(max(1 - 4Q^2, eps))
        a = half * S0 * ω0 * Q
        c = half * ω0 / Q
        return (T[a * (1 + 1/f), a * (1 - 1/f)], T[c * (1 - f), c * (1 + f)],
                T[], T[], T[], T[])
    else
        f = sqrt(max(4Q^2 - 1, eps))
        a = S0 * ω0 * Q
        c = half * ω0 / Q
        return (T[], T[], T[a], T[a/f], T[c], T[c * f])
    end
end

"""
    rotation_fm17_coefficients(amp, τ_decay, period, factor) -> (ar, cr, ac, bc, cc, dc)

Return celerite coefficients for the original Foreman-Mackey+ 2017
rotation kernel (one real + one complex term sharing decay timescale):

    k_real(τ)    = a (1+f)/(2+f) · exp(−τ/τ_decay)
    k_complex(τ) = a /  (2+f)    · exp(−τ/τ_decay) · cos(2π τ / P)

Used by `CeleriteRotationFM17`. Matches astroEMPEROR's `RotationTerm`.
"""
function rotation_fm17_coefficients(amp, τ_decay, period, factor)
    T = promote_type(typeof(amp), typeof(τ_decay), typeof(period), typeof(factor))
    inv_τ = 1 / T(τ_decay)
    two_pi_inv_p = T(2π) / T(period)
    denom = T(2) + T(factor)
    a_real = T(amp) * (1 + T(factor)) / denom
    a_cmpx = T(amp) / denom
    # Real term: amplitude a_real, decay rate inv_τ
    # Complex term: a + i·b = a_cmpx + 0i, decay c = inv_τ, frequency d = 2π/P
    return (T[a_real], T[inv_τ],
            T[a_cmpx], T[zero(T)], T[inv_τ], T[two_pi_inv_p])
end

"""
    rotation_coefficients(σ, period, Q0, dQ, frac) -> (ar, cr, ac, bc, cc, dc)

Return celerite coefficients for the rotation kernel (two SHO terms).
"""
function rotation_coefficients(σ, period, Q0, dQ, frac)
    T = promote_type(typeof(σ), typeof(period), typeof(Q0), typeof(dQ), typeof(frac))
    amp = σ^2 / (1 + frac)
    four_pi = T(4π)
    eight_pi = T(8π)
    half = T(0.5)

    Q1 = half + Q0 + dQ
    ω1 = four_pi * Q1 / (period * sqrt(4Q1^2 - 1))
    S1 = amp / (ω1 * Q1)

    Q2 = half + Q0
    ω2 = eight_pi * Q2 / (period * sqrt(4Q2^2 - 1))
    S2 = frac * amp / (ω2 * Q2)

    ar1, cr1, ac1, bc1, cc1, dc1 = sho_coefficients(S1, Q1, ω1)
    ar2, cr2, ac2, bc2, cc2, dc2 = sho_coefficients(S2, Q2, ω2)
    return (vcat(ar1, ar2), vcat(cr1, cr2),
            vcat(ac1, ac2), vcat(bc1, bc2),
            vcat(cc1, cc2), vcat(dc1, dc2))
end

# =====================================================================
# O(n) celerite log-likelihood
# =====================================================================

"""
    celerite_loglike(times, residuals, variances, ar, cr, ac, bc, cc, dc) -> logL

O(n) celerite log-likelihood. Generic over element type.
Algorithm from Foreman-Mackey et al. 2017 (Appendix B) and
the Celerite2.jl reference implementation.
"""
function celerite_loglike(times::Vector{Float64},
                           residuals::AbstractVector{T},
                           variances::AbstractVector{T},
                           ar::Vector{T}, cr::Vector{T},
                           ac::Vector{T}, bc::Vector{T},
                           cc::Vector{T}, dc::Vector{T}) where {T}
    N = length(times)
    Jr = length(ar)
    Jc = length(ac)
    J = Jr + 2Jc

    # --- Factorize (O(n) Cholesky of semi-separable matrix) ----------
    A = variances .+ sum(ar; init=zero(T)) .+ sum(ac; init=zero(T))

    D = Vector{T}(undef, N)
    U = Matrix{T}(undef, J, N)
    W = Matrix{T}(undef, J, N)
    phi = Matrix{T}(undef, J, N - 1)
    S = zeros(T, J, J)

    # First element
    D[1] = A[1]
    inv_d1 = 1 / D[1]

    # Initialize cos/sin accumulators for complex terms
    cosdt = Vector{T}(undef, Jc)
    sindt = Vector{T}(undef, Jc)

    for j in 1:Jr
        U[j, 1] = ar[j]
        W[j, 1] = inv_d1
    end
    for j in 1:Jc
        cosdt[j] = cos(dc[j] * T(times[1]))
        sindt[j] = sin(dc[j] * T(times[1]))
        U[Jr + 2j - 1, 1] = ac[j] * cosdt[j] + bc[j] * sindt[j]
        U[Jr + 2j,     1] = ac[j] * sindt[j] - bc[j] * cosdt[j]
        W[Jr + 2j - 1, 1] = cosdt[j] * inv_d1
        W[Jr + 2j,     1] = sindt[j] * inv_d1
    end

    @inbounds for n in 2:N
        dx = T(times[n] - times[n - 1])

        # Update phi, U, W for real terms
        for j in 1:Jr
            phi[j, n - 1] = exp(-cr[j] * dx)
            U[j, n] = ar[j]
            W[j, n] = one(T)
        end

        # Update phi, U, W for complex terms (recursive cos/sin)
        for j in 1:Jc
            expdx = exp(-cc[j] * dx)
            phi[Jr + 2j - 1, n - 1] = expdx
            phi[Jr + 2j,     n - 1] = expdx
            cdj = cosdt[j]
            dcd = cos(dc[j] * dx)
            dsd = sin(dc[j] * dx)
            cosdt[j] = cdj * dcd - sindt[j] * dsd
            sindt[j] = sindt[j] * dcd + cdj * dsd
            U[Jr + 2j - 1, n] = ac[j] * cosdt[j] + bc[j] * sindt[j]
            U[Jr + 2j,     n] = ac[j] * sindt[j] - bc[j] * cosdt[j]
            W[Jr + 2j - 1, n] = cosdt[j]
            W[Jr + 2j,     n] = sindt[j]
        end

        # Recursive S update
        Dn_prev = D[n - 1]
        for j in 1:J
            phij = phi[j, n - 1]
            Wj = W[j, n - 1]
            for k in 1:j
                S[k, j] = phij * phi[k, n - 1] * (S[k, j] + Dn_prev * Wj * W[k, n - 1])
            end
        end

        # Update D and W
        Dn = zero(T)
        for j in 1:J
            Uj = U[j, n]
            Wj = W[j, n]
            for k in 1:(j - 1)
                Sk = S[k, j]
                tmp = Uj * Sk
                Dn += U[k, n] * tmp
                Wj -= U[k, n] * Sk
                W[k, n] -= tmp
            end
            tmp = Uj * S[j, j]
            Dn += Uj * tmp / 2
            W[j, n] = Wj - tmp
        end

        D[n] = A[n] - 2Dn
        if D[n] <= zero(T)
            return T(-Inf)  # not positive definite
        end
        inv_dn = 1 / D[n]
        for j in 1:J
            W[j, n] *= inv_dn
        end
    end

    # --- Solve (O(n) forward-backward substitution) ------------------
    z = Vector{T}(undef, N)
    f = zeros(T, J)
    z[1] = residuals[1]

    @inbounds for n in 2:N
        for j in 1:J
            f[j] = phi[j, n - 1] * (f[j] + W[j, n - 1] * z[n - 1])
        end
        z[n] = residuals[n] - dot(view(U, :, n), f)
    end

    z ./= D
    fill!(f, zero(T))

    @inbounds for n in (N - 1):-1:1
        for j in 1:J
            f[j] = phi[j, n] * (f[j] + U[j, n + 1] * z[n + 1])
        end
        z[n] -= dot(view(W, :, n), f)
    end

    # --- Log-likelihood ----------------------------------------------
    logdet_K = sum(log, D)
    chi2 = dot(residuals, z)
    return -(logdet_K + N * T(log(2π)) + chi2) / 2
end

"""
    celerite_solve(times, residuals, variances, ar, cr, ac, bc, cc, dc) -> α

Return α = K_full⁻¹ · residuals using the same O(n) factorization as
`celerite_loglike`. Useful for prediction: the GP signal mean at the
data points is `μ = residuals − variances .* α`, since
K_full · α = y and K_signal = K_full − diag(variances).

(Detrending convention: pass `flux − offset` as residuals, then the
GP-detrended flux is `flux − μ = offset + variances .* α`.)
"""
function celerite_solve(times::Vector{Float64},
                         residuals::AbstractVector{T},
                         variances::AbstractVector{T},
                         ar::Vector{T}, cr::Vector{T},
                         ac::Vector{T}, bc::Vector{T},
                         cc::Vector{T}, dc::Vector{T}) where {T}
    N = length(times)
    Jr = length(ar)
    Jc = length(ac)
    J = Jr + 2Jc

    A = variances .+ sum(ar; init=zero(T)) .+ sum(ac; init=zero(T))
    D = Vector{T}(undef, N)
    U = Matrix{T}(undef, J, N)
    W = Matrix{T}(undef, J, N)
    phi = Matrix{T}(undef, J, N - 1)
    S = zeros(T, J, J)

    D[1] = A[1]
    inv_d1 = 1 / D[1]
    cosdt = Vector{T}(undef, Jc)
    sindt = Vector{T}(undef, Jc)

    for j in 1:Jr
        U[j, 1] = ar[j]; W[j, 1] = inv_d1
    end
    for j in 1:Jc
        cosdt[j] = cos(dc[j] * T(times[1]))
        sindt[j] = sin(dc[j] * T(times[1]))
        U[Jr + 2j - 1, 1] = ac[j] * cosdt[j] + bc[j] * sindt[j]
        U[Jr + 2j,     1] = ac[j] * sindt[j] - bc[j] * cosdt[j]
        W[Jr + 2j - 1, 1] = cosdt[j] * inv_d1
        W[Jr + 2j,     1] = sindt[j] * inv_d1
    end

    @inbounds for n in 2:N
        dx = T(times[n] - times[n - 1])
        for j in 1:Jr
            phi[j, n - 1] = exp(-cr[j] * dx)
            U[j, n] = ar[j]; W[j, n] = one(T)
        end
        for j in 1:Jc
            expdx = exp(-cc[j] * dx)
            phi[Jr + 2j - 1, n - 1] = expdx
            phi[Jr + 2j,     n - 1] = expdx
            cdj = cosdt[j]
            dcd = cos(dc[j] * dx); dsd = sin(dc[j] * dx)
            cosdt[j] = cdj * dcd - sindt[j] * dsd
            sindt[j] = sindt[j] * dcd + cdj * dsd
            U[Jr + 2j - 1, n] = ac[j] * cosdt[j] + bc[j] * sindt[j]
            U[Jr + 2j,     n] = ac[j] * sindt[j] - bc[j] * cosdt[j]
            W[Jr + 2j - 1, n] = cosdt[j]
            W[Jr + 2j,     n] = sindt[j]
        end

        Dn_prev = D[n - 1]
        for j in 1:J
            phij = phi[j, n - 1]; Wj = W[j, n - 1]
            for k in 1:j
                S[k, j] = phij * phi[k, n - 1] * (S[k, j] + Dn_prev * Wj * W[k, n - 1])
            end
        end

        Dn = zero(T)
        for j in 1:J
            Uj = U[j, n]; Wj = W[j, n]
            for k in 1:(j - 1)
                Sk = S[k, j]; tmp = Uj * Sk
                Dn += U[k, n] * tmp
                Wj -= U[k, n] * Sk
                W[k, n] -= tmp
            end
            tmp = Uj * S[j, j]
            Dn += Uj * tmp / 2
            W[j, n] = Wj - tmp
        end

        D[n] = A[n] - 2Dn
        D[n] > zero(T) || error("celerite_solve: factorization failed (D[$n] ≤ 0)")
        inv_dn = 1 / D[n]
        for j in 1:J
            W[j, n] *= inv_dn
        end
    end

    # --- Solve for α = K_full⁻¹ y --------------------------------------
    z = Vector{T}(undef, N)
    f = zeros(T, J)
    z[1] = residuals[1]
    @inbounds for n in 2:N
        for j in 1:J
            f[j] = phi[j, n - 1] * (f[j] + W[j, n - 1] * z[n - 1])
        end
        z[n] = residuals[n] - dot(view(U, :, n), f)
    end
    z ./= D
    fill!(f, zero(T))
    @inbounds for n in (N - 1):-1:1
        for j in 1:J
            f[j] = phi[j, n] * (f[j] + U[j, n + 1] * z[n + 1])
        end
        z[n] -= dot(view(W, :, n), f)
    end
    return z
end

"""
    celerite_predict_mean(times, residuals, variances, ar, cr, ac, bc, cc, dc) -> μ

GP signal mean prediction at the observation times. Returns
`μ = residuals - variances .* α`, which is the contribution from the
covariance structure (excluding white noise).
"""
function celerite_predict_mean(times::Vector{Float64},
                                residuals::AbstractVector{T},
                                variances::AbstractVector{T},
                                ar::Vector{T}, cr::Vector{T},
                                ac::Vector{T}, bc::Vector{T},
                                cc::Vector{T}, dc::Vector{T}) where {T}
    α = celerite_solve(times, residuals, variances, ar, cr, ac, bc, cc, dc)
    return residuals .- variances .* α
end

"""
    celerite_predict_at(t_pred, t_data, alpha, ar, cr, ac, bc, cc, dc) -> μ

Out-of-sample GP mean prediction at arbitrary new times `t_pred`,
given `alpha = K_full⁻¹ y` from `celerite_solve` and the same kernel
coefficient blocks. For each prediction time `t*`:

    μ(t*) = Σᵢ K_signal(|t* - tᵢ|) · αᵢ

where `K_signal(τ) = Σⱼ arⱼ e^(-crⱼ τ) + Σⱼ e^(-ccⱼ τ) (acⱼ cos dcⱼ τ
+ bcⱼ sin dcⱼ τ)` is the kernel without the diagonal-variance term.

O(N_pred × N_data); use only for plotting/diagnostics. For larger
problems use the O(N + M) celerite predict algorithm.
"""
function celerite_predict_at(t_pred::AbstractVector{<:Real},
                              t_data::AbstractVector{<:Real},
                              alpha::AbstractVector{T},
                              ar::Vector{T}, cr::Vector{T},
                              ac::Vector{T}, bc::Vector{T},
                              cc::Vector{T}, dc::Vector{T}) where {T}
    N_p = length(t_pred)
    N_d = length(t_data)
    Jr = length(ar)
    Jc = length(ac)
    μ = zeros(T, N_p)
    @inbounds for p in 1:N_p
        tp = T(t_pred[p])
        s = zero(T)
        for i in 1:N_d
            τ = abs(tp - T(t_data[i]))
            kτ = zero(T)
            for j in 1:Jr
                kτ += ar[j] * exp(-cr[j] * τ)
            end
            for j in 1:Jc
                ej = exp(-cc[j] * τ)
                kτ += ej * (ac[j] * cos(dc[j] * τ) + bc[j] * sin(dc[j] * τ))
            end
            s += kτ * alpha[i]
        end
        μ[p] = s
    end
    return μ
end

# =====================================================================
# Interface for Nereus noise models
# =====================================================================

function gp_log_likelihood(
    residuals::AbstractVector{T}, variances::AbstractVector{T},
    times::AbstractVector{Float64}, theta::Theta{T},
    nm::CeleriteSHO,
) where {T}
    layout = theta.params.layout
    s = _gp_suffix(nm)
    log_S0 = theta.values[layout.name_to_idx["gp_log_S0$s"]]
    log_Q  = theta.values[layout.name_to_idx["gp_log_Q$s"]]
    log_w0 = theta.values[layout.name_to_idx["gp_log_omega0$s"]]
    S0 = exp(log_S0); Q = exp(log_Q); ω0 = exp(log_w0)

    ar, cr, ac, bc, cc, dc = sho_coefficients(S0, Q, ω0)
    return celerite_loglike(_as_t_vec(times), residuals, variances,
                             ar, cr, ac, bc, cc, dc)
end

function gp_log_likelihood(
    residuals::AbstractVector{T}, variances::AbstractVector{T},
    times::AbstractVector{Float64}, theta::Theta{T},
    nm::CeleriteRotation,
) where {T}
    layout = theta.params.layout
    s = _gp_suffix(nm)
    σ      = theta.values[layout.name_to_idx["gp_sigma$s"]]
    period = theta.values[layout.name_to_idx["gp_period$s"]]
    Q0     = theta.values[layout.name_to_idx["gp_Q0$s"]]
    dQ     = theta.values[layout.name_to_idx["gp_dQ$s"]]
    frac   = theta.values[layout.name_to_idx["gp_f$s"]]

    ar, cr, ac, bc, cc, dc = rotation_coefficients(σ, period, Q0, dQ, frac)
    return celerite_loglike(_as_t_vec(times), residuals, variances,
                             ar, cr, ac, bc, cc, dc)
end

function gp_log_likelihood(
    residuals::AbstractVector{T}, variances::AbstractVector{T},
    times::AbstractVector{Float64}, theta::Theta{T},
    nm::CeleriteRotationFM17,
) where {T}
    layout = theta.params.layout
    s = _gp_suffix(nm)
    log_amp  = theta.values[layout.name_to_idx["gp_log_amp$s"]]
    log_tau  = theta.values[layout.name_to_idx["gp_log_timescale$s"]]
    log_P    = theta.values[layout.name_to_idx["gp_log_period$s"]]
    log_f    = theta.values[layout.name_to_idx["gp_log_factor$s"]]
    amp     = exp(log_amp)
    τ_decay = exp(log_tau)
    period  = exp(log_P)
    factor  = exp(log_f)

    ar, cr, ac, bc, cc, dc = rotation_fm17_coefficients(amp, τ_decay, period, factor)
    return celerite_loglike(_as_t_vec(times), residuals, variances,
                             ar, cr, ac, bc, cc, dc)
end

# `celerite_loglike` is signed `times::Vector{Float64}`; per-instrument
# slicing paths produce views (`SubArray`) so collect to a concrete
# vector before dispatch. Residuals/variances are `AbstractVector{T}`
# in the signature, so they pass through unchanged.
_as_t_vec(t::Vector{Float64}) = t
_as_t_vec(t::AbstractVector{<:Real}) = Vector{Float64}(t)

# =====================================================================
# Out-of-sample GP mean — kernel-aware dispatch (parallels gp_log_likelihood)
# =====================================================================
#
# `gp_mean_at(residuals, variances, t_data, t_pred, theta, nm)`
# returns the GP signal-mean at `t_pred`, conditioned on data
# `residuals` at `t_data`. Used by `plot_rv_timeseries` to overlay
# the GP component as a continuous curve. Same kernel-coefficient
# extraction as `gp_log_likelihood` — keep them in sync.

function gp_mean_at(residuals::AbstractVector{T}, variances::AbstractVector{T},
                     t_data::AbstractVector{Float64},
                     t_pred::AbstractVector{Float64}, theta::Theta{T},
                     nm::CeleriteSHO) where {T}
    layout = theta.params.layout
    s = _gp_suffix(nm)
    log_S0 = theta.values[layout.name_to_idx["gp_log_S0$s"]]
    log_Q  = theta.values[layout.name_to_idx["gp_log_Q$s"]]
    log_w0 = theta.values[layout.name_to_idx["gp_log_omega0$s"]]
    S0 = exp(log_S0); Q = exp(log_Q); ω0 = exp(log_w0)
    ar, cr, ac, bc, cc, dc = sho_coefficients(S0, Q, ω0)
    alpha = celerite_solve(_as_t_vec(t_data), residuals, variances,
                            ar, cr, ac, bc, cc, dc)
    return celerite_predict_at(t_pred, t_data, alpha, ar, cr, ac, bc, cc, dc)
end

function gp_mean_at(residuals::AbstractVector{T}, variances::AbstractVector{T},
                     t_data::AbstractVector{Float64},
                     t_pred::AbstractVector{Float64}, theta::Theta{T},
                     nm::CeleriteRotation) where {T}
    layout = theta.params.layout
    s = _gp_suffix(nm)
    σ      = theta.values[layout.name_to_idx["gp_sigma$s"]]
    period = theta.values[layout.name_to_idx["gp_period$s"]]
    Q0     = theta.values[layout.name_to_idx["gp_Q0$s"]]
    dQ     = theta.values[layout.name_to_idx["gp_dQ$s"]]
    frac   = theta.values[layout.name_to_idx["gp_f$s"]]
    ar, cr, ac, bc, cc, dc = rotation_coefficients(σ, period, Q0, dQ, frac)
    alpha = celerite_solve(_as_t_vec(t_data), residuals, variances,
                            ar, cr, ac, bc, cc, dc)
    return celerite_predict_at(t_pred, t_data, alpha, ar, cr, ac, bc, cc, dc)
end

function gp_mean_at(residuals::AbstractVector{T}, variances::AbstractVector{T},
                     t_data::AbstractVector{Float64},
                     t_pred::AbstractVector{Float64}, theta::Theta{T},
                     nm::CeleriteRotationFM17) where {T}
    layout = theta.params.layout
    s = _gp_suffix(nm)
    log_amp  = theta.values[layout.name_to_idx["gp_log_amp$s"]]
    log_tau  = theta.values[layout.name_to_idx["gp_log_timescale$s"]]
    log_P    = theta.values[layout.name_to_idx["gp_log_period$s"]]
    log_f    = theta.values[layout.name_to_idx["gp_log_factor$s"]]
    amp     = exp(log_amp); τ_decay = exp(log_tau)
    period  = exp(log_P);   factor  = exp(log_f)
    ar, cr, ac, bc, cc, dc = rotation_fm17_coefficients(amp, τ_decay, period, factor)
    alpha = celerite_solve(_as_t_vec(t_data), residuals, variances,
                            ar, cr, ac, bc, cc, dc)
    return celerite_predict_at(t_pred, t_data, alpha, ar, cr, ac, bc, cc, dc)
end

"""
    channel_gp_mean_at(theta, residuals, variances, t_data, t_pred,
                        inst, channel) -> Vector{Float64} or nothing

Channel-level GP mean prediction at arbitrary times. Returns the
out-of-sample GP curve at `t_pred` if a single *global* `CovarianceNoise`
GP is active on `channel`; returns `nothing` for the no-GP or
restricted-GP cases (per-instrument GPs don't map onto a global
out-of-sample curve cleanly — plot them per-instrument instead).
"""
function channel_gp_mean_at(theta::Theta{T},
                             residuals::AbstractVector{T},
                             variances::AbstractVector{T},
                             t_data::AbstractVector{Float64},
                             t_pred::AbstractVector{Float64},
                             inst::AbstractVector{Int},
                             channel::Symbol;
                             data::Union{Nothing, Data} = nothing) where {T}
    config = theta.params.config
    for (i, nm) in enumerate(config.noise_models)
        is_noise_model_active(theta, i) || continue
        # ActivityGP — multivariate-GP "GP curve" is the inferred RV
        # activity contribution Vc·G + Vr·dG/dt conditioned on the
        # full joint of RV + indicator data. Needs the indicator data,
        # which we route via the optional `data` kwarg.
        if nm isa ActivityGP && channel === :rv
            data === nothing && continue
            return _activity_gp_rv_mean_at_theta(theta, data, t_pred, nm)
        end
        nm isa CovarianceNoise || continue
        noise_channel(nm) === channel || continue
        if isempty(noise_instruments(nm))
            return gp_mean_at(residuals, variances, t_data, t_pred, theta, nm)
        end
    end
    return nothing
end

# Single-Theta variant of `activity_gp_predict` — returns the inferred
# `Vc·G(t_pred) + Vr·dG/dt(t_pred)` curve at one parameter point.
# Cheaper than the chain version (one Cholesky); right tool for an
# rv_timeseries overlay at the per-parameter median.
function _activity_gp_rv_mean_at_theta(theta::Theta{T}, data::Data,
                                         t_pred::AbstractVector{Float64},
                                         agp::ActivityGP) where {T}
    layout = theta.params.layout
    s = _gp_suffix(agp)
    # Unit-variance G(t) (Rajpaul+ 2015); guarded lookup for legacy chains.
    amp_idx = get(layout.name_to_idx, "gp_act_amp$s", 0)
    amp = amp_idx == 0 ? 1.0 : Float64(theta.values[amp_idx])
    P   = Float64(theta.values[layout.name_to_idx["gp_act_period$s"]])
    λe  = Float64(theta.values[layout.name_to_idx["gp_act_lambda_e$s"]])
    λp  = Float64(theta.values[layout.name_to_idx["gp_act_lambda_p$s"]])
    (amp > 0 && P > 0 && λe > 0 && λp > 0) || return nothing
    # Derivative couplings sampled as amplitudes — physical Ġ
    # coefficient = amplitude / std(Ġ) (mirrors _activity_gp_joint_ll).
    inv_sdG = 1 / sqrt(1 / (λe * λe) + π * π / (P * P * λp * λp))
    Vc = Float64(theta.values[layout.name_to_idx["Vc$s"]])
    Vr = agp.use_derivative ?
         Float64(theta.values[layout.name_to_idx["Vr$s"]]) * inv_sdG : 0.0

    n_rv_obs = length(data.t_rv)

    # Build the indicator metadata blocks.
    ind_meta = Tuple{Symbol, Vector{Float64}, Vector{Float64}, Float64, Float64, Float64}[]
    n_obs_total = n_rv_obs
    for ch in agp.channels
        ch === :rv && continue
        name = String(ch)
        haskey(data.indicators, name) || return nothing
        haskey(data.indicator_errs, name) || return nothing
        vals = data.indicators[name]; errs = data.indicator_errs[name]
        cg, cd = _ACTIVITY_GP_COEFFS[ch]
        a_coef = Float64(theta.values[layout.name_to_idx[string(cg, s)]])
        b_coef = (cd === nothing || !agp.use_derivative) ? 0.0 :
                  Float64(theta.values[layout.name_to_idx[string(cd, s)]]) * inv_sdG
        jit_idx = get(layout.name_to_idx, "gp_act_jit_$(ch)$s", 0)
        jit² = jit_idx == 0 ? 0.0 : Float64(theta.values[jit_idx])^2
        push!(ind_meta, (ch, vals, errs, a_coef, b_coef, jit²))
        n_obs_total += length(vals)
    end

    preds, vars = rv_predictions(theta, data)
    t_obs  = Vector{Float64}(undef, n_obs_total)
    y_obs  = Vector{Float64}(undef, n_obs_total)
    σ²_obs = Vector{Float64}(undef, n_obs_total)
    a_obs  = Vector{Float64}(undef, n_obs_total)
    b_obs  = Vector{Float64}(undef, n_obs_total)
    @inbounds for i in 1:n_rv_obs
        t_obs[i]  = data.t_rv[i]
        y_obs[i]  = Float64(data.rv[i] - preds[i])
        σ²_obs[i] = Float64(vars[i])
        a_obs[i]  = Vc
        b_obs[i]  = Vr
    end
    offset = n_rv_obs
    for (_, vals, errs, a_coef, b_coef, jit²) in ind_meta
        n_ch = length(vals)
        @inbounds for i in 1:n_ch
            t_obs[offset + i]  = data.t_rv[i]
            y_obs[offset + i]  = Float64(vals[i])
            σ²_obs[offset + i] = Float64(errs[i]^2) + jit²
            a_obs[offset + i]  = a_coef
            b_obs[offset + i]  = b_coef
        end
        offset += n_ch
    end
    channel_obs = vcat(fill(:rv, n_rv_obs),
                        vcat(fill.(getindex.(ind_meta, 1),
                                     length.(getindex.(ind_meta, 2)))...))

    Σ_oo = activity_gp_covariance(t_obs, channel_obs, a_obs, b_obs,
                                    amp, P, λe, λp)
    @inbounds for i in 1:n_obs_total
        Σ_oo[i, i] += σ²_obs[i]
    end
    F = cholesky(Symmetric(Σ_oo); check = false)
    issuccess(F) || return nothing
    α = F \ y_obs

    # Cross-covariance for RV activity prediction: each pred point
    # contributes `Vc·G(t*) + Vr·dG/dt(t*)`, so the cross-kernel block
    # is `Vc·k_GG + Vr·k_dotGG` summed with `Vc·k_GdotG + Vr·k_dotGdotG`
    # multipliers from the obs side coefficients. Final result is the
    # inferred RV activity contribution.
    n_pred = length(t_pred)
    rv_act = Vector{Float64}(undef, n_pred)
    @inbounds for j in 1:n_pred
        # K_op[i, j] (obs i to G(t*_j))     = a_i·k_GG + b_i·k_dotGG
        # K_op_dG[i, j] (obs i to dG/dt(t*_j)) = a_i·k_GdotG + b_i·k_dotGdotG
        # G_pred[j]  = K_op[:, j]ᵀ α
        # dG_pred[j] = K_op_dG[:, j]ᵀ α
        # rv_act[j]  = Vc·G_pred[j] + Vr·dG_pred[j]
        g_acc  = 0.0
        dg_acc = 0.0
        for i in 1:n_obs_total
            τ = t_obs[i] - t_pred[j]
            k_GG, k_GdotG, k_dotGG, k_dotGdotG =
                activity_kernel_blocks(τ, amp, P, λe, λp)
            g_acc  += (a_obs[i] * k_GG    + b_obs[i] * k_dotGG)    * α[i]
            dg_acc += (a_obs[i] * k_GdotG + b_obs[i] * k_dotGdotG) * α[i]
        end
        rv_act[j] = Vc * g_acc + Vr * dg_acc
    end
    return rv_act
end

# =====================================================================
# Channel dispatch — global GP, per-instrument GPs, or white noise
# =====================================================================

"""
    _eval_channel_likelihood(theta, residuals, variances, times,
                              inst, channel, two_pi) -> T

Evaluate the channel-level log-likelihood given pre-computed
residuals/variances/times and a per-observation instrument index.
Walks `theta.params.config.noise_models`, dispatches based on which
`CovarianceNoise` models are active on `channel`:

- No GP active            → diagonal Gaussian sum.
- One global GP (`instruments == []`) → existing single-GP path.
- N restricted GPs        → per-GP slice + sum, plus white-noise
                            contribution for any uncovered observations.

`validate_noise_models` already guarantees that within a channel we
have either zero/one global GP OR several restricted GPs with disjoint
instrument sets — never both, never overlapping — so the slicing here
doesn't double-count.
"""
function _eval_channel_likelihood(theta::Theta{T},
                                   residuals::AbstractVector{T},
                                   variances::AbstractVector{T},
                                   times::AbstractVector{Float64},
                                   inst::AbstractVector{Int},
                                   channel::Symbol,
                                   two_pi::T) where {T}
    config = theta.params.config
    noise_models = config.noise_models
    inst_names = channel === :rv ? config.instruments.rv_names :
                                    config.instruments.pm_names
    n_obs = length(residuals)

    global_nm = nothing
    restricted_idx = Int[]   # indices into noise_models
    for (i, nm) in enumerate(noise_models)
        nm isa CovarianceNoise || continue
        # ActivityGP is fully handled by the Rajpaul routing in
        # rv_log_likelihood — in indicators_only mode the RV channel
        # deliberately falls through to THIS white/celerite path, and
        # the multi-channel model must not be treated as an RV GP here.
        nm isa ActivityGP && continue
        noise_channel(nm) === channel || continue
        is_noise_model_active(theta, i) || continue
        if isempty(noise_instruments(nm))
            global_nm = nm
        else
            push!(restricted_idx, i)
        end
    end

    # Additive low-rank covariance corrections (NightlyOffset / HarmonicBlock)
    # compose on top of the base (white default or one celerite GP) via
    # Woodbury — Σ = B + FFᵀ. Collected here so the base dispatch below is
    # reused for B.
    add_models = AdditiveCovariance[]
    for (i, nm) in enumerate(noise_models)
        nm isa AdditiveCovariance || continue
        noise_channel(nm) === channel || continue
        is_noise_model_active(theta, i) || continue
        push!(add_models, nm)
    end
    # Student-t heavy-tailed white likelihood. Valid ONLY on a diagonal base:
    # it destroys the Gaussian marginalization, so combining it with ANY
    # covariance / additive structure is inadmissible (fail-loud: −Inf, so the
    # state carries zero posterior mass rather than a silently wrong value).
    st = _active_studentt(theta, noise_models, channel)
    if st !== nothing
        (global_nm === nothing && isempty(restricted_idx) && isempty(add_models)) ||
            return T(-Inf)
        ν = theta.values[theta.params.layout.name_to_idx["studentt_nu"]]
        return _studentt_diag_ll(residuals, variances, ν)
    end

    if !isempty(add_models)
        F = _additive_factor(add_models, theta, times, inst, inst_names)
        if global_nm !== nothing
            return global_nm isa MaternGP ?
                _woodbury_matern_ll(residuals, variances, times,
                                     theta, global_nm, F, two_pi) :
                _woodbury_celerite_ll(residuals, variances, times,
                                       theta, global_nm, F, two_pi)
        elseif isempty(restricted_idx)
            return _woodbury_white_ll(residuals, variances, F, two_pi)
        else
            error("AdditiveCovariance $(typeof.(add_models)) with per-instrument " *
                  "restricted GPs on channel :$channel is unsupported; use a " *
                  "single global GP or a white base.")
        end
    end

    if global_nm !== nothing
        return gp_log_likelihood(residuals, variances, times, theta, global_nm)
    end

    if isempty(restricted_idx)
        return _diag_gaussian_ll(residuals, variances, two_pi)
    end

    total = zero(T)
    covered = falses(n_obs)
    for nm_idx in restricted_idx
        nm = noise_models[nm_idx]
        inst_idxs = Int[]
        for name in noise_instruments(nm)
            k = findfirst(==(name), inst_names)
            k === nothing && error(
                "noise_models[$nm_idx] references instrument `$name` " *
                "not present in $(channel) instrument list $(inst_names)")
            push!(inst_idxs, k)
        end
        sel = Int[]
        @inbounds for i in 1:n_obs
            if inst[i] in inst_idxs
                push!(sel, i)
                covered[i] = true
            end
        end
        isempty(sel) && continue
        total += gp_log_likelihood(view(residuals, sel),
                                    view(variances, sel),
                                    view(times, sel),
                                    theta, nm)
    end

    @inbounds for i in 1:n_obs
        covered[i] && continue
        total += -(log(two_pi * variances[i]) + residuals[i]^2 / variances[i]) / 2
    end
    return total
end

@inline function _diag_gaussian_ll(residuals::AbstractVector{T},
                                    variances::AbstractVector{T},
                                    two_pi::T) where {T}
    total = zero(T)
    @inbounds for i in eachindex(residuals)
        total += -(log(two_pi * variances[i]) + residuals[i]^2 / variances[i]) / 2
    end
    return total
end
