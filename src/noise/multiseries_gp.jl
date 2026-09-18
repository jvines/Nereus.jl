# O(N·r²) semiseparable multi-series activity GP ("S+LEAF backend").
#
# Shared-latent Rajpaul model: every series s is y_s(t) = α_s·G(t) + β_s·Ġ(t)
# for one latent GP G with a *once-differentiable* covariance k(τ). This gives
# the joint covariance an exact semiseparable (celerite/S+LEAF) representation,
# so the likelihood factorizes in O(N·r²) instead of the dense (C·N)³ or the
# Woodbury (2N)³ of the existing `activity_gp_joint_logpdf_lowrank` path.
#
# Why this works where FM17-block assembly fails (see activity_gp.jl:78): a
# kernel with a kink at τ=0 puts a Dirac in k″, so Var(Ġ) diverges and the joint
# matrix is indefinite. The fix is NOT to avoid semiseparable — it is to (a)
# restrict the latent kernel to differentiable ones (assembled kink
# Σ(λa−νb)=0) and (b) let Ġ enter through closed-form generator derivatives
# (dU,dV), never through block assembly. See NEREUS_MULTISERIES_GP_SPEC.md.
#
# Two identities gate correctness (both tested against the dense oracle):
#   (1) −k″(0) = Σ_s dU_s·dV_s  (per-term closed form: a(ν²−λ²)+2λνb for QP).
#   (2) The Cov(Ġ_i,G_j)=+k′ vs Cov(G_i,Ġ_j)=−k′ antisymmetry emerges from dU
#       differentiating the later time and dV the earlier — no explicit sign flip.
#
# Reference: Delisle et al. 2020 (A&A 638 A95) + 2022 (multi-series/derivative);
# Rajpaul et al. 2015 (the shared-latent structure). Implemented clean-room from
# the equations, not by translating libspleaf.c.

using LinearAlgebra: dot
using SpecialFunctions: besselix        # exponentially-scaled I_n(x)=I_n(x)·e^{−|x|} (ESP weights)
import ForwardDiff

# ForwardDiff-differentiable scaled modified Bessel (ESP η-gradient). besselix has
# no Dual rule; d/dx[I_n(x)e^{−x}] = (ive_{n−1}+ive_{n+1})/2 − ive_n (recurrence).
_besselix(n::Integer, x::Real) = besselix(n, x)
function _besselix(n::Integer, x::ForwardDiff.Dual{Tg}) where {Tg}
    v  = ForwardDiff.value(x)
    b  = besselix(n, v)
    db = (besselix(abs(n - 1), v) + besselix(n + 1, v)) / 2 - b
    return ForwardDiff.Dual{Tg}(b, db * ForwardDiff.partials(x))
end

# ---------------------------------------------------------------------------
# Kernel representation: every kernel expands to a list of semiseparable terms.
#   :qp   (a,b,λ,ν)  — QP/celerite term e^{−λτ}(a cos ντ + b sin ντ);
#                      rank 2 if ν>0, rank 1 ("real"/Exp) if ν==0 (b must be 0).
#   :m32  (a,λ)      — Matérn-3/2, rank 2, NON-stationary generators.
# ---------------------------------------------------------------------------

abstract type SemiseparableKernel end

# A term as a NamedTuple. `_terms(k)` returns the ordered list for kernel k.
# Types are preserved (promote) so ForwardDiff.Dual hyperparameters flow through.
function _qp(a, b, λ, ν)
    aa, bb, ll, nn = promote(a, b, λ, ν)
    (kind = :qp, a = aa, b = bb, λ = ll, ν = nn)
end
function _m32(a, λ)
    aa, ll = promote(a, λ); z = zero(aa)
    (kind = :m32, a = aa, b = z, λ = ll, ν = z)
end

_term_rank(t) = t.kind === :m32 ? 2 : (t.ν > 0 ? 2 : 1)
_kernel_rank(terms) = sum(_term_rank, terms; init = 0)
_terms_eltype(terms) = mapreduce(t -> typeof(t.a), promote_type, terms; init = Float64)

# --- closed-form k(τ), k′(τ), k″(τ) per term (for the dense oracle) ---------
# Even extension in τ; u = |τ|. k′ is odd (carry sign(τ)).
function _term_f(t, u)          # f(u) = term value at |τ|=u
    if t.kind === :m32
        return t.a * (1 + t.λ * u) * exp(-t.λ * u)
    else
        return exp(-t.λ * u) * (t.a * cos(t.ν * u) + t.b * sin(t.ν * u))
    end
end
function _term_df(t, u)         # f′(u)
    if t.kind === :m32
        return -t.a * t.λ^2 * u * exp(-t.λ * u)
    else
        da = -t.λ * t.a + t.ν * t.b
        db = -t.λ * t.b - t.ν * t.a
        return exp(-t.λ * u) * (da * cos(t.ν * u) + db * sin(t.ν * u))
    end
end
function _term_ddf(t, u)        # f″(u)
    if t.kind === :m32
        return -t.a * t.λ^2 * (1 - t.λ * u) * exp(-t.λ * u)
    else
        da = -t.λ * t.a + t.ν * t.b
        db = -t.λ * t.b - t.ν * t.a
        return exp(-t.λ * u) * ((-t.λ * da + t.ν * db) * cos(t.ν * u) +
                                (-t.λ * db - t.ν * da) * sin(t.ν * u))
    end
end

# Generic hooks: sum kernels use their `_terms`; product kernels (SSESP) override.
_generators(k::SemiseparableKernel, t, dt) = _base_generators(_terms(k), t, dt)
_kfuncs(k::SemiseparableKernel, τ) = _k_kp_kpp(_terms(k), τ)

k0(k::SemiseparableKernel)    = _kfuncs(k, 0.0)[1]                # k(0)
mk2_0(k::SemiseparableKernel) = -_kfuncs(k, 0.0)[3]              # −k″(0)
# assembled kink Σ(λa − νb); must be ≈0 to admit a β≠0 (derivative) coupling.
derivative_kink(k::SemiseparableKernel) =
    sum(t -> t.kind === :m32 ? 0.0 : (t.λ * t.a - t.ν * t.b), _terms(k); init = 0.0)

# k(τ), k′(τ), k″(τ) from the term list (τ signed).
function _k_kp_kpp(terms, τ)
    u = abs(τ); s = sign(τ)
    kk  = sum(t -> _term_f(t, u),   terms; init = 0.0)
    kp  = s * sum(t -> _term_df(t, u), terms; init = 0.0)
    kpp = sum(t -> _term_ddf(t, u),  terms; init = 0.0)
    return kk, kp, kpp
end

# ---------------------------------------------------------------------------
# Base generators: fill (U0,V0,dU0,dV0) [r×N] and φ [r×(N−1)] for a term list,
# given sorted times `t` and gaps `dt`. Stationary terms use absolute-time
# cos/sin (the φ-telescope reconstructs cos(ν(t_i−t_j))); Matérn-3/2 uses the
# midpoint-centred non-stationary generators (x = λ(t−t0)) or a·x overflows.
# ---------------------------------------------------------------------------
function _base_generators(terms, t::AbstractVector{<:Real}, dt::AbstractVector{<:Real})
    N = length(t); r = _kernel_rank(terms)
    T = _terms_eltype(terms)
    U0 = zeros(T, N, r); V0 = zeros(T, N, r); dU0 = zeros(T, N, r); dV0 = zeros(T, N, r)
    φ  = zeros(T, max(N - 1, 0), r)
    t0 = N == 0 ? 0.0 : (t[1] + t[end]) / 2       # Matérn midpoint centre (concrete Float64)
    one_T = one(T)
    c = 1
    for term in terms
        λ = term.λ
        if term.kind === :m32
            a = term.a
            @inbounds for i in 1:N
                x = λ * (t[i] - t0)
                U0[i, c]     = a * x;      V0[i, c]     = one_T
                U0[i, c + 1] = a;          V0[i, c + 1] = one_T - x
                dU0[i, c]     = λ * a * (one_T - x); dV0[i, c]     = λ
                dU0[i, c + 1] = -λ * a;              dV0[i, c + 1] = -λ * x
            end
            @inbounds for i in 1:(N - 1)
                e = exp(-λ * dt[i]); φ[i, c] = e; φ[i, c + 1] = e
            end
            c += 2
        elseif term.ν > 0                          # QP complex term (rank 2)
            a = term.a; b = term.b; ν = term.ν
            da = -λ * a + ν * b; db = -λ * b - ν * a
            @inbounds for i in 1:N
                cs = cos(ν * t[i]); sn = sin(ν * t[i])
                U0[i, c]     = a * cs + b * sn;  V0[i, c]     = cs
                U0[i, c + 1] = a * sn - b * cs;  V0[i, c + 1] = sn
                dU0[i, c]     = da * cs + db * sn; dV0[i, c]     = λ * cs - ν * sn
                dU0[i, c + 1] = da * sn - db * cs; dV0[i, c + 1] = λ * sn + ν * cs
            end
            @inbounds for i in 1:(N - 1)
                e = exp(-λ * dt[i]); φ[i, c] = e; φ[i, c + 1] = e
            end
            c += 2
        else                                       # real/Exp term (rank 1)
            a = term.a
            @inbounds for i in 1:N
                U0[i, c] = a;      V0[i, c] = one_T
                dU0[i, c] = -λ * a; dV0[i, c] = λ
            end
            @inbounds for i in 1:(N - 1)
                φ[i, c] = exp(-λ * dt[i])
            end
            c += 1
        end
    end
    return U0, V0, dU0, dV0, φ
end

# ---------------------------------------------------------------------------
# Semiseparable Cholesky + solve (celerite LDLᵀ; adapted from gp.jl:104 to take
# generic (A,U,V,φ)). U,V are r×N, φ is r×(N−1). Returns T(−Inf) on D≤0.
# ---------------------------------------------------------------------------
function semiseparable_loglike(t::Vector{Float64}, y::AbstractVector{T},
                               A::AbstractVector{T}, U::AbstractMatrix{T},
                               V::AbstractMatrix{T}, φ::AbstractMatrix{T}) where {T}
    N = length(t); r = size(U, 1)
    N == 0 && return zero(T)
    D = Vector{T}(undef, N)
    W = Matrix{T}(undef, r, N)
    S = zeros(T, r, r)
    D[1] = A[1]
    D[1] <= zero(T) && return T(-Inf)
    invd = 1 / D[1]
    @inbounds for s in 1:r
        W[s, 1] = V[s, 1] * invd
    end
    @inbounds for n in 2:N
        Dn_prev = D[n - 1]
        for s in 1:r
            φs = φ[s, n - 1]; Ws = W[s, n - 1]
            for k in 1:s
                S[k, s] = φs * φ[k, n - 1] * (S[k, s] + Dn_prev * Ws * W[k, n - 1])
            end
        end
        for s in 1:r
            W[s, n] = V[s, n]
        end
        Dn = zero(T)
        for s in 1:r
            Us = U[s, n]; Wsn = W[s, n]
            for k in 1:(s - 1)
                Sk = S[k, s]; tmp = Us * Sk
                Dn += U[k, n] * tmp
                Wsn -= U[k, n] * Sk
                W[k, n] -= tmp
            end
            tmp = Us * S[s, s]
            Dn += Us * tmp / 2
            W[s, n] = Wsn - tmp
        end
        D[n] = A[n] - 2Dn
        D[n] <= zero(T) && return T(-Inf)
        invdn = 1 / D[n]
        for s in 1:r
            W[s, n] *= invdn
        end
    end

    # forward-backward solve  z = K⁻¹ y
    z = Vector{T}(undef, N)
    f = zeros(T, r)
    z[1] = y[1]
    @inbounds for n in 2:N
        for s in 1:r
            f[s] = φ[s, n - 1] * (f[s] + W[s, n - 1] * z[n - 1])
        end
        z[n] = y[n] - dot(view(U, :, n), f)
    end
    z ./= D
    fill!(f, zero(T))
    @inbounds for n in (N - 1):-1:1
        for s in 1:r
            f[s] = φ[s, n] * (f[s] + U[s, n + 1] * z[n + 1])
        end
        z[n] -= dot(view(W, :, n), f)
    end

    logdet_K = sum(log, D)
    chi2 = dot(y, z)
    return -(logdet_K + N * T(log(2π)) + chi2) / 2
end

# Same O(N·r²) LDLᵀ factorization as `semiseparable_loglike`, but returns the
# SOLVE Z = K⁻¹·RHS (for a matrix RHS, one column each) together with logdet(K).
# Lets a semiseparable covariance serve as a Woodbury base (K + FFᵀ): pass
# RHS = [y F] in one factorization. Returns (Z, logdet) or (nothing, Inf) if the
# factorization is not positive-definite.
function semiseparable_solve_logdet(t::Vector{Float64}, RHS::AbstractMatrix{T},
                                    A::AbstractVector{T}, U::AbstractMatrix{T},
                                    V::AbstractMatrix{T}, φ::AbstractMatrix{T}) where {T}
    N = length(t); r = size(U, 1); m = size(RHS, 2)
    D = Vector{T}(undef, N)
    W = Matrix{T}(undef, r, N)
    S = zeros(T, r, r)
    D[1] = A[1]
    D[1] <= zero(T) && return (nothing, T(Inf))
    @inbounds for s in 1:r
        W[s, 1] = V[s, 1] / D[1]
    end
    @inbounds for n in 2:N
        Dn_prev = D[n - 1]
        for s in 1:r
            φs = φ[s, n - 1]; Ws = W[s, n - 1]
            for k in 1:s
                S[k, s] = φs * φ[k, n - 1] * (S[k, s] + Dn_prev * Ws * W[k, n - 1])
            end
        end
        for s in 1:r
            W[s, n] = V[s, n]
        end
        Dn = zero(T)
        for s in 1:r
            Us = U[s, n]; Wsn = W[s, n]
            for k in 1:(s - 1)
                Sk = S[k, s]; tmp = Us * Sk
                Dn += U[k, n] * tmp
                Wsn -= U[k, n] * Sk
                W[k, n] -= tmp
            end
            tmp = Us * S[s, s]
            Dn += Us * tmp / 2
            W[s, n] = Wsn - tmp
        end
        D[n] = A[n] - 2Dn
        D[n] <= zero(T) && return (nothing, T(Inf))
        invdn = 1 / D[n]
        for s in 1:r
            W[s, n] *= invdn
        end
    end
    # forward-backward solve, applied to every RHS column.
    Z = Matrix{T}(undef, N, m)
    f = Vector{T}(undef, r)
    @inbounds for c in 1:m
        Z[1, c] = RHS[1, c]
        fill!(f, zero(T))
        for n in 2:N
            for s in 1:r
                f[s] = φ[s, n - 1] * (f[s] + W[s, n - 1] * Z[n - 1, c])
            end
            Z[n, c] = RHS[n, c] - dot(view(U, :, n), f)
        end
        for n in 1:N
            Z[n, c] /= D[n]
        end
        fill!(f, zero(T))
        for n in (N - 1):-1:1
            for s in 1:r
                f[s] = φ[s, n] * (f[s] + U[s, n + 1] * Z[n + 1, c])
            end
            Z[n, c] -= dot(view(W, :, n), f)
        end
    end
    return (Z, sum(log, D))
end

# ---------------------------------------------------------------------------
# Multi-series assembly. Global time-sort across all channels (stable ties),
# per-row (α,β) by series. Arbitrary non-simultaneous per-channel grids — the
# semiseparable form needs only sorted t (a capability edge over kima).
#   y_all, var_all, series_id, t_all : length-N vectors (any interleaving)
#   α, β : per-series coefficients (length = n_series); β_s=0 ⇒ series has no Ġ.
#   jitter : optional per-series extra white noise (added in quadrature).
# ---------------------------------------------------------------------------
function multiseries_loglike(t_all::AbstractVector{<:Real}, y_all::AbstractVector{<:Real},
                             var_all::AbstractVector{<:Real}, series_id::AbstractVector{<:Integer},
                             α::AbstractVector, β::AbstractVector,
                             kernel::SemiseparableKernel;
                             jitter::AbstractVector = zeros(length(α)))
    N = length(t_all)
    # common working type (data Float64 + Dual couplings/hyperparams → Dual)
    T = promote_type(eltype(y_all), eltype(α), eltype(β), eltype(jitter),
                     _kernel_eltype(kernel))
    N == 0 && return zero(T)
    # β≠0 requires a differentiable latent kernel.
    if any(!iszero, β) && abs(derivative_kink(kernel)) > 1e-8 * max(1.0, abs(k0(kernel)))
        throw(ArgumentError("multiseries_loglike: latent kernel is not once-" *
            "differentiable (derivative_kink = $(derivative_kink(kernel)) ≠ 0); a " *
            "derivative coupling β≠0 gives a divergent Var(Ġ). Use β=0 or an " *
            "admissible kernel (SSMatern32/SSSHO/SSMEP/SSES/SSESP)."))
    end

    ord = sortperm(t_all)                          # stable → simultaneous ties keep order
    t  = Float64.(t_all[ord])
    y  = Vector{T}(y_all[ord])
    v  = var_all[ord]
    sid = series_id[ord]
    dt = N > 1 ? diff(t) : Float64[]

    U0, V0, dU0, dV0, φ0 = _generators(kernel, t, dt)
    r = size(U0, 2)
    kk0  = T(k0(kernel))
    mk20 = T(mk2_0(kernel))

    U = Matrix{T}(undef, r, N)
    V = Matrix{T}(undef, r, N)
    A = Vector{T}(undef, N)
    @inbounds for i in 1:N
        s = sid[i]
        αi = T(α[s]); βi = T(β[s])
        for c in 1:r
            U[c, i] = αi * U0[i, c] + βi * dU0[i, c]
            V[c, i] = αi * V0[i, c] + βi * dV0[i, c]
        end
        A[i] = T(v[i]) + T(jitter[s])^2 + αi^2 * kk0 + βi^2 * mk20
    end
    φ = Matrix{T}(undef, r, max(N - 1, 0))
    @inbounds for i in 1:(N - 1), c in 1:r
        φ[c, i] = T(φ0[i, c])
    end
    return semiseparable_loglike(t, y, A, U, V, φ)
end

# Element type of a kernel's hyperparameters (for AD promotion).
_kernel_eltype(k::SemiseparableKernel) = typeof(k0(k))

# ---------------------------------------------------------------------------
# Dense oracle (parity backstop, gate 1). Builds the joint covariance directly
# from the closed-form k,k′,k″ of the same term list — the correct oracle (the
# true-QP AGP is a *different* kernel that MEP/ESP only approximate).
# ---------------------------------------------------------------------------
function dense_multiseries_loglike(t_all::AbstractVector{<:Real}, y_all::AbstractVector{T},
                                   var_all::AbstractVector{<:Real}, series_id::AbstractVector{<:Integer},
                                   α::AbstractVector{T}, β::AbstractVector{T},
                                   kernel::SemiseparableKernel;
                                   jitter::AbstractVector{T} = zeros(T, length(α))) where {T}
    N = length(t_all)
    N == 0 && return zero(T)
    K = Matrix{T}(undef, N, N)
    @inbounds for j in 1:N, i in 1:N
        τ = t_all[i] - t_all[j]
        kk, kp, kpp = _kfuncs(kernel, τ)
        si = series_id[i]; sj = series_id[j]
        αi = α[si]; βi = β[si]; αj = α[sj]; βj = β[sj]
        # Cov(y_i,y_j) = αiαj·k − αiβj·k′ + βiαj·k′ − βiβj·k″
        K[i, j] = αi * αj * kk - αi * βj * kp + βi * αj * kp - βi * βj * kpp
    end
    @inbounds for i in 1:N
        K[i, i] += T(var_all[i]) + jitter[series_id[i]]^2
    end
    C = cholesky(Symmetric(K); check = false)
    issuccess(C) || return T(-Inf)
    z = C \ y_all
    logdet_K = 2 * sum(log, diag(C.U))
    return -(logdet_K + N * T(log(2π)) + dot(y_all, z)) / 2
end

# ---------------------------------------------------------------------------
# Kernels (fully specified by the spec; composites MEP/ES/ESP added separately).
# ---------------------------------------------------------------------------

"""QP building block: e^{−λτ}(a cos ντ + b sin ντ). Differentiable iff λa=νb."""
struct SSQP{T<:Real} <: SemiseparableKernel
    a::T; b::T; λ::T; ν::T
end
SSQP(a, b, λ, ν) = SSQP(promote(a, b, λ, ν)...)
_terms(k::SSQP) = (_qp(k.a, k.b, k.λ, k.ν),)

"""Matérn-3/2, k(τ)=σ²(1+√3|τ|/ρ)e^{−√3|τ|/ρ}. Differentiable (k′(0)=0)."""
struct SSMatern32{T<:Real} <: SemiseparableKernel
    σ::T; ρ::T
end
SSMatern32(σ, ρ) = SSMatern32(promote(σ, ρ)...)
_terms(k::SSMatern32) = (_m32(k.σ^2, sqrt(3.0) / k.ρ),)

"""Stochastic harmonic oscillator, variance σ², period P₀, quality Q.
Underdamped (Q>½): one QP term. Overdamped (Q<½): two real terms (`OSHO`), still
differentiable — the kink cancels between the two exponentials."""
struct SSSHO{T<:Real} <: SemiseparableKernel
    σ::T; P₀::T; Q::T
end
SSSHO(σ, P₀, Q) = SSSHO(promote(σ, P₀, Q)...)
function _terms(k::SSSHO)
    ω0 = 2π / k.P₀
    S0 = k.σ^2 / (ω0 * k.Q)                        # S0 chosen so k(0)=σ²
    ar, cr, ac, bc, cc, dc = sho_coefficients(S0, k.Q, ω0)
    ts = Any[]
    for i in eachindex(ar); push!(ts, _qp(ar[i], 0.0, cr[i], 0.0)); end
    for i in eachindex(ac); push!(ts, _qp(ac[i], bc[i], cc[i], dc[i])); end
    return Tuple(ts)
end

"""Matérn-3/2 exponential-periodic (MEP), rank 6 — approximates the
squared-exponential-periodic kernel σ²·exp(−Δt²/2ρ² − sin²(πΔt/P)/2η²) as
Matérn-3/2 ⊕ QP(ν) ⊕ QP(2ν) (Delisle et al. 2022). Differentiable by
construction (b_i = a_iλ/(iν))."""
struct SSMEP{T<:Real} <: SemiseparableKernel
    σ::T; ρ::T; P::T; η::T
end
SSMEP(σ, ρ, P, η) = SSMEP(promote(σ, ρ, P, η)...)
function _terms(k::SSMEP)
    λ = 1 / k.ρ; var = k.σ^2; η2 = k.η^2
    f = 1 / (4η2); f2_4 = f^2 / 4
    deno = 1 + f + f2_4
    a0 = var / deno; a1 = f * a0; a2 = f2_4 * a0
    ν = 2π / k.P
    b1 = a1 * λ / ν; b2 = a2 * λ / (2ν)
    return (_m32(a0, sqrt(3.0) / k.ρ), _qp(a1, b1, λ, ν), _qp(a2, b2, λ, 2ν))
end

"""Exponential-sine (ES), rank 3 — approximates the squared-exponential
kernel σ²·exp(−Δt²/2ρ²) as Exp(a₀,λ) ⊕ QP (Delisle et al. 2022). The bare Exp
term's kink cancels against the QP's, so the sum is differentiable."""
struct SSES{T<:Real} <: SemiseparableKernel
    σ::T; ρ::T
end
SSES(σ, ρ) = SSES(promote(σ, ρ)...)
const _ES_COEF_LA = 1.0907260149419182
const _ES_MU      = 1.326644517327145
function _terms(k::SSES)
    coef_b  = 1 / _ES_MU
    coef_a0 = (2 / 3) * (1 + coef_b^2)
    coef_a  = 1 - coef_a0
    λ = _ES_COEF_LA / k.ρ; ν = _ES_MU * λ; var = k.σ^2
    return (_qp(coef_a0 * var, 0.0, λ, 0.0),        # Exp (real term)
            _qp(coef_a * var, coef_b * var, λ, ν))  # QP
end

# --- ESP: a PRODUCT kernel (ES radial × cosine-harmonic periodic) ----------
# Normalized (k(0)=1) periodic part: Σ_n a_n cos(nντ) with a_n ∝ I_n(f)e^{−f},
# f = 1/(4η²), a_0 halved (Delisle et al. 2022). λ=0 (no decay) on every term.
function _esp_periodic_terms(P, η, nharm::Int)
    f = 1 / (4 * η^2)
    a = [_besselix(n, f) for n in 0:nharm]          # I_n(f)·e^{−f}, n=0..nharm
    a[1] /= 2                                        # halve the n=0 (constant) weight
    a ./= sum(a)                                     # normalize so k_periodic(0)=1
    ν = 2π / P
    ts = Any[_qp(a[1], 0.0, 0.0, 0.0)]              # n=0 constant (λ=0, ν=0)
    for n in 1:nharm
        push!(ts, _qp(a[n + 1], 0.0, 0.0, n * ν))  # cos(nντ) harmonic (λ=0)
    end
    return Tuple(ts)
end

"""Exponential-sine periodic (ESP), rank 3·(1+2·nharm) — twice-differentiable
approximation to the squared-exponential-periodic kernel, as the elementwise
PRODUCT of an ES radial kernel and a normalized cosine-harmonic periodic kernel
(Delisle et al. 2022). Generators are the Kronecker product of the two factors'
generators; derivatives follow the product rule."""
struct SSESP{T<:Real} <: SemiseparableKernel
    σ::T; ρ::T; P::T; η::T; nharm::Int
end
function SSESP(σ, ρ, P, η, nharm::Int)
    v = promote(σ, ρ, P, η); SSESP{typeof(v[1])}(v..., nharm)
end
SSESP(σ, ρ, P, η; nharm::Int = 3) = SSESP(σ, ρ, P, η, nharm)
_esp_factors(k::SSESP) = (_terms(SSES(k.σ, k.ρ)), _esp_periodic_terms(k.P, k.η, k.nharm))

function _generators(k::SSESP, t, dt)
    t1, t2 = _esp_factors(k)
    U1, V1, dU1, dV1, φ1 = _base_generators(t1, t, dt)
    U2, V2, dU2, dV2, φ2 = _base_generators(t2, t, dt)
    N = length(t); r1 = size(U1, 2); r2 = size(U2, 2); r = r1 * r2
    T = promote_type(eltype(U1), eltype(U2))
    U = zeros(T, N, r); V = zeros(T, N, r); dU = zeros(T, N, r); dV = zeros(T, N, r)
    φ = zeros(T, max(N - 1, 0), r)
    c = 1
    @inbounds for s in 1:r1, u in 1:r2
        for i in 1:N
            U[i, c]  = U1[i, s] * U2[i, u]
            V[i, c]  = V1[i, s] * V2[i, u]
            dU[i, c] = dU1[i, s] * U2[i, u] + U1[i, s] * dU2[i, u]   # product rule
            dV[i, c] = dV1[i, s] * V2[i, u] + V1[i, s] * dV2[i, u]
        end
        for i in 1:(N - 1)
            φ[i, c] = φ1[i, s] * φ2[i, u]
        end
        c += 1
    end
    return U, V, dU, dV, φ
end

function _kfuncs(k::SSESP, τ)
    t1, t2 = _esp_factors(k)
    k1, kp1, kpp1 = _k_kp_kpp(t1, τ)
    k2, kp2, kpp2 = _k_kp_kpp(t2, τ)
    return k1 * k2, kp1 * k2 + k1 * kp2, kpp1 * k2 + 2 * kp1 * kp2 + k1 * kpp2
end
# product of two once-differentiable factors is once-differentiable (k′(0)=0).
derivative_kink(k::SSESP) = begin
    t1, t2 = _esp_factors(k)
    kink1 = sum(t -> t.kind === :m32 ? 0.0 : (t.λ * t.a - t.ν * t.b), t1; init = 0.0)
    kink2 = sum(t -> t.kind === :m32 ? 0.0 : (t.λ * t.a - t.ν * t.b), t2; init = 0.0)
    kink1 * k0(SSES(k.σ, k.ρ)) + k0(SSES(k.σ, k.ρ)) * kink2  # both ≈0 ⇒ 0
end

# ── ActivityGP → semiseparable kernel mapping ────────────────────────────────
# Map Nereus's quasi-periodic activity hyperparameters onto a semiseparable
# latent kernel when `ActivityGP.latent_kernel` opts into the O(N·r²) backend.
# Convention (spec §3.4): amp→σ (amplitude), λe→ρ (decay length), P→P (period),
# λp→η (coherence). η is HALVED for :mep/:esp — those kernels' periodic factor
# uses the sin²(πτ/P) half-angle argument, so their η is twice the QP λp (the
# ÷2 convention trap). `amp` is a VARIANCE (k(0)=amp in the QP builder), so the
# std maps as σ=√amp.
function _ss_latent_kernel(sym::Symbol, amp, P, λe, λp)
    σ = sqrt(amp)
    ρ = λe
    sym === :matern32 && return SSMatern32(σ, ρ)
    sym === :sho      && return SSSHO(σ, P, λp)          # λp ≙ quality factor Q
    sym === :es       && return SSES(σ, ρ)
    sym === :mep      && return SSMEP(σ, ρ, P, λp / 2)
    sym === :esp      && return SSESP(σ, ρ, P, λp / 2)
    throw(ArgumentError("unknown ActivityGP latent_kernel :$sym " *
                        "(expected :qp_dense, :matern32, :sho, :es, :mep, :esp)"))
end
