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
    over, p1, p2, p3, p4 = _sho_terms(S0, Q, ω0)
    return over ? (T[p1, p2], T[p3, p4], T[], T[], T[], T[]) :
                  (T[], T[], T[p1], T[p2], T[p3], T[p4])
end

# The four numbers of `sho_coefficients`: (true, ar₁, ar₂, cr₁, cr₂) for an
# overdamped oscillator (two real terms), (false, ac, bc, cc, dc) for an
# underdamped one (one complex term). The single source of these expressions
# for `sho_coefficients`, the allocation-free `_sho_push!` (CeleriteRotation
# on the workspace path) and `_sho_loglike!` (instrument-restricted SHOs).
@inline function _sho_terms(S0, Q, ω0)
    T = promote_type(typeof(S0), typeof(Q), typeof(ω0))
    eps = T(1e-5)
    half = T(0.5)
    if Q < half
        f = sqrt(max(1 - 4Q^2, eps))
        a = half * S0 * ω0 * Q
        c = half * ω0 / Q
        return (true, T(a * (1 + 1/f)), T(a * (1 - 1/f)), T(c * (1 - f)), T(c * (1 + f)))
    else
        f = sqrt(max(4Q^2 - 1, eps))
        a = S0 * ω0 * Q
        c = half * ω0 / Q
        return (false, T(a), T(a/f), T(c), T(c * f))
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
    S1, Q1, ω1, S2, Q2, ω2 = _rotation_sho_params(σ, period, Q0, dQ, frac)
    ar1, cr1, ac1, bc1, cc1, dc1 = sho_coefficients(S1, Q1, ω1)
    ar2, cr2, ac2, bc2, cc2, dc2 = sho_coefficients(S2, Q2, ω2)
    return (vcat(ar1, ar2), vcat(cr1, cr2),
            vcat(ac1, ac2), vcat(bc1, bc2),
            vcat(cc1, cc2), vcat(dc1, dc2))
end

# (S, Q, ω0) of the rotation kernel's two SHO terms, the fundamental at P
# and the harmonic at P/2. Shared by `rotation_coefficients` and the
# allocation-free `_rotation_coefficients!`.
@inline function _rotation_sho_params(σ, period, Q0, dQ, frac)
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
    return S1, Q1, ω1, S2, Q2, ω2
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
    Jc = length(ac)
    J = length(ar) + 2Jc
    return _celerite_loglike!(CeleriteBuffers{T}(J, N, Jc), times, residuals,
                              variances, ar, cr, ac, bc, cc, dc)
end

"""
    CeleriteBuffers{T}(J, N, Jc)

Work arrays of `celerite_loglike` for `J` semiseparable terms (`Jc` of them
complex pairs) and `N` points, sized exactly — the solve's `dot`s and
`sum(log, D)` see the same arrays they always did. Reused across calls
through `CeleriteWork` on the PTWorkspace path.
"""
struct CeleriteBuffers{T}
    A::Vector{T}
    D::Vector{T}
    U::Matrix{T}
    W::Matrix{T}
    phi::Matrix{T}
    S::Matrix{T}
    cosdt::Vector{T}
    sindt::Vector{T}
    z::Vector{T}
    f::Vector{T}
end

CeleriteBuffers{T}(J::Int, N::Int, Jc::Int) where {T} = CeleriteBuffers{T}(
    Vector{T}(undef, N), Vector{T}(undef, N), Matrix{T}(undef, J, N),
    Matrix{T}(undef, J, N), Matrix{T}(undef, J, N - 1), Matrix{T}(undef, J, J),
    Vector{T}(undef, Jc), Vector{T}(undef, Jc), Vector{T}(undef, N),
    Vector{T}(undef, J))

# `celerite_loglike` on caller-supplied buffers (sized by `CeleriteBuffers`).
_celerite_loglike!(B::CeleriteBuffers{T}, times::Vector{Float64},
                   residuals::AbstractVector{T}, variances::AbstractVector{T},
                   ar::Vector{T}, cr::Vector{T}, ac::Vector{T}, bc::Vector{T},
                   cc::Vector{T}, dc::Vector{T}) where {T} =
    _celerite_loglike!(B.A, B.D, B.U, B.W, B.phi, B.S, B.cosdt, B.sindt, B.z, B.f,
                       times, residuals, variances, ar, cr, ac, bc, cc, dc)

"""
    _celerite_loglike!(A, D, U, W, phi, S, cosdt, sindt, z, f, times, residuals,
                       variances, ar, cr, ac, bc, cc, dc) -> logL

`celerite_loglike` in caller-supplied scratch: `A`, `D`, `z` of length N,
`U`, `W` of J x N, `phi` J x (N-1), `S` J x J, `f` of length J and `cosdt`,
`sindt` of length (at least) the number of complex terms. Their contents on
entry do not matter. `celerite_loglike` is this with fresh arrays.
"""
function _celerite_loglike!(A, D, U, W, phi, S, cosdt, sindt, z, f,
                            times::Vector{Float64},
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
    A .= variances .+ sum(ar; init=zero(T)) .+ sum(ac; init=zero(T))
    fill!(S, zero(T))

    # First element
    D[1] = A[1]
    inv_d1 = 1 / D[1]

    # cos/sin accumulators for complex terms: cosdt, sindt

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
    # The J-term `dot`s stay BLAS calls: a plain loop sums in another order
    # and moves log L by an ulp.
    fill!(f, zero(T))
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

# ---------------------------------------------------------------------
# CeleriteRotation on the PTWorkspace path
# ---------------------------------------------------------------------
"""
    CeleriteWork

Scratch for the RV-channel celerite GP on the PTWorkspace path (held by
`ws.rv_noise`): the kernel coefficient lists, the solver's work arrays and
the kernel's layout slots, resolved once per (layout, noise model). Nothing
in it is chain state.
"""
mutable struct CeleriteWork
    ar::Vector{Float64}
    cr::Vector{Float64}
    ac::Vector{Float64}
    bc::Vector{Float64}
    cc::Vector{Float64}
    dc::Vector{Float64}
    buf::CeleriteBuffers{Float64}
    key_idx::Any                    # layout.name_to_idx the slots come from
    key_nm::Any                     # the CeleriteRotation they belong to
    slots::NTuple{5, Int}           # gp_sigma, gp_period, gp_Q0, gp_dQ, gp_f
end

CeleriteWork() = CeleriteWork(Float64[], Float64[], Float64[], Float64[], Float64[],
                              Float64[], CeleriteBuffers{Float64}(0, 1, 0),
                              nothing, nothing, (0, 0, 0, 0, 0))

# `sho_coefficients`, appended to the coefficient lists in `cw`.
@inline function _sho_push!(cw::CeleriteWork, S0, Q, ω0)
    over, x1, x2, x3, x4 = _sho_terms(S0, Q, ω0)
    if over
        push!(cw.ar, x1, x2); push!(cw.cr, x3, x4)
    else
        push!(cw.ac, x1); push!(cw.bc, x2); push!(cw.cc, x3); push!(cw.dc, x4)
    end
    return cw
end

# `rotation_coefficients` into `cw`'s lists: the same values in the same
# order as its vcat of the two SHO terms, without allocating.
function _rotation_coefficients!(cw::CeleriteWork, σ, period, Q0, dQ, frac)
    empty!(cw.ar); empty!(cw.cr); empty!(cw.ac)
    empty!(cw.bc); empty!(cw.cc); empty!(cw.dc)
    S1, Q1, ω1, S2, Q2, ω2 = _rotation_sho_params(σ, period, Q0, dQ, frac)
    _sho_push!(cw, S1, Q1, ω1)
    _sho_push!(cw, S2, Q2, ω2)
    return cw
end

function _rotation_slots!(cw::CeleriteWork, layout, nm::CeleriteRotation)
    idx = layout.name_to_idx
    (cw.key_idx === idx && cw.key_nm === nm) && return cw.slots
    s = _gp_suffix(nm)
    cw.slots = (get(idx, "gp_sigma$s", 0), get(idx, "gp_period$s", 0),
                get(idx, "gp_Q0$s", 0), get(idx, "gp_dQ$s", 0), get(idx, "gp_f$s", 0))
    cw.key_idx = idx
    cw.key_nm = nm
    return cw.slots
end

"""
    gp_log_likelihood(residuals, variances, times, theta, nm, ws) -> T

`gp_log_likelihood(residuals, variances, times, theta, nm)` with scratch
from the PTWorkspace `ws`. For a `Float64` theta and `CeleriteRotation`
it is the same computation without a per-call allocation; every other
case takes the allocating method.
"""
function gp_log_likelihood(residuals::AbstractVector{Float64},
                           variances::AbstractVector{Float64},
                           times::AbstractVector{Float64}, theta::Theta{Float64},
                           nm::CeleriteRotation, ws)
    cw = ws.rv_noise.cel
    i1, i2, i3, i4, i5 = _rotation_slots!(cw, theta.params.layout, nm)
    # A missing slot raises its KeyError in the generic method.
    (i1 == 0 || i2 == 0 || i3 == 0 || i4 == 0 || i5 == 0) &&
        return gp_log_likelihood(residuals, variances, times, theta, nm)
    v = theta.values
    _rotation_coefficients!(cw, v[i1], v[i2], v[i3], v[i4], v[i5])
    t = _as_t_vec(times)
    N = length(t)
    Jc = length(cw.ac)
    J = length(cw.ar) + 2Jc
    B = cw.buf
    if size(B.U, 1) != J || length(B.D) != N || length(B.cosdt) != Jc
        B = cw.buf = CeleriteBuffers{Float64}(J, N, Jc)
    end
    return _celerite_loglike!(B, t, residuals, variances,
                              cw.ar, cw.cr, cw.ac, cw.bc, cw.cc, cw.dc)
end

gp_log_likelihood(residuals, variances, times, theta::Theta, nm, ws) =
    gp_log_likelihood(residuals, variances, times, theta, nm)

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

A sampler passes its workspace as an eighth argument (see the method on
`ws` below). This general path then scores a global GP with the workspace's
RV-noise scratch (`gp_log_likelihood(..., ws)`); the result is the same.
"""
_eval_channel_likelihood(theta::Theta{T}, residuals::AbstractVector{T},
                         variances::AbstractVector{T}, times::AbstractVector{Float64},
                         inst::AbstractVector{Int}, channel::Symbol,
                         two_pi::T) where {T} =
    _eval_channel_likelihood_general(theta, residuals, variances, times, inst,
                                     channel, two_pi, nothing)

# `ws`: a PTWorkspace, or nothing.
function _eval_channel_likelihood_general(theta::Theta{T},
                                          residuals::AbstractVector{T},
                                          variances::AbstractVector{T},
                                          times::AbstractVector{Float64},
                                          inst::AbstractVector{Int},
                                          channel::Symbol,
                                          two_pi::T,
                                          ws) where {T}
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
        return ws === nothing ?
            gp_log_likelihood(residuals, variances, times, theta, global_nm) :
            gp_log_likelihood(residuals, variances, times, theta, global_nm, ws)
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

# ---------------------------------------------------------------------
# The per-instrument GP channel in a sampler's scratch
# ---------------------------------------------------------------------
#
# An obliquity fit scores each RM night with its own oscillator: one
# instrument-restricted CeleriteSHO per night on :rv. For every call the
# path above rebuilds each night's selection (`inst_idxs`, `sel`,
# `covered`), copies the night's times, builds the hyperparameter names to
# look them up, allocates the coefficient vectors and every celerite array:
# ~16 KB per call on NGTS-33. Here the selections and times are built once
# per (Params, times, instruments) and the arrays live in the workspace; the
# arithmetic, and so every bit, is that path's. Anything else on the channel
# -- a global GP, an additive covariance, Student-t, an instrument name that
# does not resolve -- takes that path unchanged.

"""Scratch for one instrument-restricted GP: its points, times and celerite arrays."""
struct ChannelGPWork
    sel::Vector{Int}            # the observations it covers, in order
    t::Vector{Float64}          # their times, as `_as_t_vec(view(times, sel))` builds them
    sho::NTuple{3,Int}          # (log S0, log Q, log omega0) slots; zeros unless CeleriteSHO
    A::Vector{Float64}
    D::Vector{Float64}
    z::Vector{Float64}
    U::Matrix{Float64}
    W::Matrix{Float64}
    phi::Matrix{Float64}
    S::Matrix{Float64}
    f::Vector{Float64}
    cosdt::Vector{Float64}
    sindt::Vector{Float64}
end

"""Per-channel scratch held by a sampler workspace; see `_eval_channel_likelihood(..., ws)`."""
mutable struct ChannelWork
    params::Any
    times::Vector{Float64}
    inst::Vector{Int}
    channel::Symbol
    gps::Vector{Union{Nothing,ChannelGPWork}}   # per noise model; nothing = not resolvable here
    covered::BitVector
    active::Vector{Int}
    ar2::Vector{Float64}        # oscillator coefficients, overdamped (two real terms)
    cr2::Vector{Float64}
    ac1::Vector{Float64}        # underdamped (one complex term)
    bc1::Vector{Float64}
    cc1::Vector{Float64}
    dc1::Vector{Float64}
    none::Vector{Float64}
end

function ChannelWork(params, times::Vector{Float64}, inst::Vector{Int}, channel::Symbol)
    config = params.config
    names = channel === :rv ? config.instruments.rv_names : config.instruments.pm_names
    L = params.layout
    nms = config.noise_models
    gps = Vector{Union{Nothing,ChannelGPWork}}(nothing, length(nms))
    for (i, nm) in enumerate(nms)
        (nm isa CovarianceNoise && !(nm isa ActivityGP) && noise_channel(nm) === channel) ||
            continue
        insts = noise_instruments(nm)
        isempty(insts) && continue
        idxs = Int[]
        for name in insts
            k = findfirst(==(name), names)
            k === nothing && (idxs = nothing; break)
            push!(idxs, k)
        end
        idxs === nothing && continue          # the general path reports it
        sel = Int[q for q in eachindex(inst) if inst[q] in idxs]
        N = length(sel)
        sho = (0, 0, 0)
        if nm isa CeleriteSHO
            sfx = _gp_suffix(nm)
            sho = (get(L.name_to_idx, "gp_log_S0$sfx", 0), get(L.name_to_idx, "gp_log_Q$sfx", 0),
                   get(L.name_to_idx, "gp_log_omega0$sfx", 0))
            any(==(0), sho) && (sho = (0, 0, 0))
        end
        J = 2                                  # an oscillator: two real or one complex term
        gps[i] = ChannelGPWork(sel, Vector{Float64}(view(times, sel)), sho,
                               zeros(N), zeros(N), zeros(N), zeros(J, N), zeros(J, N),
                               zeros(J, max(N - 1, 0)), zeros(J, J), zeros(J), zeros(1),
                               zeros(1))
    end
    return ChannelWork(params, times, inst, channel, gps, falses(length(inst)),
                       zeros(Int, length(nms)), zeros(2), zeros(2), zeros(1), zeros(1),
                       zeros(1), zeros(1), Float64[])
end

"""
    _eval_channel_likelihood(theta, residuals, variances, times, inst, channel,
                             two_pi, ws) -> Float64

`_eval_channel_likelihood` scored in the scratch of a sampler workspace `ws`
(a `PTWorkspace`, or a `ChannelWork`): the same terms in the same order, bit
for bit, without rebuilding the per-instrument selections or allocating the
celerite arrays of the instrument-restricted oscillators. Anything else on
the channel takes the general path, with a PTWorkspace's RV-noise scratch for
a global GP.
"""
function _eval_channel_likelihood(theta::Theta{Float64}, residuals::Vector{Float64},
                                  variances::Vector{Float64}, times::Vector{Float64},
                                  inst::Vector{Int}, channel::Symbol, two_pi::Float64, ws)
    cw = _channel_work!(ws, theta.params, times, inst, channel)
    _channel_fast_ok(cw, theta, channel) ||
        return _eval_channel_likelihood_general(theta, residuals, variances, times, inst,
                                                channel, two_pi, _noise_ws(ws))
    noise_models = theta.params.config.noise_models
    n_active = 0
    for (i, nm) in enumerate(noise_models)
        nm isa CovarianceNoise || continue
        nm isa ActivityGP && continue
        noise_channel(nm) === channel || continue
        is_noise_model_active(theta, i) || continue
        n_active += 1
        cw.active[n_active] = i
    end
    n_active == 0 && return _diag_gaussian_ll(residuals, variances, two_pi)

    n_obs = length(residuals)
    covered = cw.covered
    fill!(covered, false)
    total = 0.0
    for a in 1:n_active
        i = cw.active[a]
        g = cw.gps[i]::ChannelGPWork
        @inbounds for k in g.sel
            covered[k] = true
        end
        isempty(g.sel) && continue
        total += g.sho[1] > 0 ?
            _sho_loglike!(g, cw, view(residuals, g.sel), view(variances, g.sel), theta) :
            _restricted_gp_ll(residuals, variances, times, g.sel, theta, noise_models[i])::Float64
    end
    @inbounds for i in 1:n_obs
        covered[i] && continue
        total += -(log(two_pi * variances[i]) + residuals[i]^2 / variances[i]) / 2
    end
    return total
end

@noinline _restricted_gp_ll(residuals, variances, times, sel, theta, nm) =
    gp_log_likelihood(view(residuals, sel), view(variances, sel), view(times, sel), theta, nm)

# Whether the active models on `channel` are all instruments-restricted
# covariance models this workspace resolved -- nothing additive, no global
# GP, no Student-t. Otherwise the general path decides.
function _channel_fast_ok(cw::ChannelWork, theta::Theta, channel::Symbol)
    noise_models = theta.params.config.noise_models
    for (i, nm) in enumerate(noise_models)
        if nm isa AdditiveCovariance
            (noise_channel(nm) === channel && is_noise_model_active(theta, i)) && return false
        end
        nm isa CovarianceNoise || continue
        nm isa ActivityGP && continue
        noise_channel(nm) === channel || continue
        is_noise_model_active(theta, i) || continue
        (isempty(noise_instruments(nm)) || cw.gps[i] === nothing) && return false
    end
    return _active_studentt(theta, noise_models, channel) === nothing
end

# Any other element type (ForwardDiff): the general path.
_eval_channel_likelihood(theta::Theta, residuals, variances, times, inst, channel::Symbol,
                         two_pi, ws) =
    _eval_channel_likelihood_general(theta, residuals, variances, times, inst, channel,
                                     two_pi, _noise_ws(ws))

# The workspace the general path may hand to `gp_log_likelihood(..., ws)`: a
# PTWorkspace (its `rv_noise` scratch), never a bare ChannelWork.
_noise_ws(ws) = ws
_noise_ws(::ChannelWork) = nothing

function _channel_work!(cw::ChannelWork, params, times, inst, channel)
    (cw.params === params && cw.times === times && cw.inst === inst &&
     cw.channel === channel) && return cw
    return ChannelWork(params, times, inst, channel)
end

# `gp_log_likelihood(r, v, view(times, sel), theta, ::CeleriteSHO)` in the
# GP's scratch.
function _sho_loglike!(g::ChannelGPWork, cw::ChannelWork, r, v, theta::Theta{Float64})
    S0 = exp(theta.values[g.sho[1]])
    Q  = exp(theta.values[g.sho[2]])
    ω0 = exp(theta.values[g.sho[3]])
    over, p1, p2, p3, p4 = _sho_terms(S0, Q, ω0)
    if over
        cw.ar2[1] = p1; cw.ar2[2] = p2; cw.cr2[1] = p3; cw.cr2[2] = p4
        ar, cr, ac, bc, cc, dc = cw.ar2, cw.cr2, cw.none, cw.none, cw.none, cw.none
    else
        cw.ac1[1] = p1; cw.bc1[1] = p2; cw.cc1[1] = p3; cw.dc1[1] = p4
        ar, cr, ac, bc, cc, dc = cw.none, cw.none, cw.ac1, cw.bc1, cw.cc1, cw.dc1
    end
    return _celerite_loglike!(g.A, g.D, g.U, g.W, g.phi, g.S, g.cosdt, g.sindt, g.z, g.f,
                              g.t, r, v, ar, cr, ac, bc, cc, dc)
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
