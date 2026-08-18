# Parametric (non-GP) correlated-noise models — the ladder between white
# noise and a full GP. Three menu members (design memo NEREUS_NOISE_MENU.md):
#
#   ErrorScale     — per-instrument multiplicative error-scale f²σ² (white)
#   NightlyOffset  — per-night rank-1 calibration blocks   (AdditiveCovariance)
#   HarmonicBlock  — marginalized coherent rotation harmonics (AdditiveCovariance)
#
# The two AdditiveCovariance members add a low-rank term FFᵀ on top of the
# channel's base covariance B (white diagonal, or one celerite GP) and are
# scored by Woodbury — Σ = B + FFᵀ factored through B's own solve + logdet,
# so they compose freely with the base and with each other without tripping
# the "at most one CovarianceNoise" rule. All routines are T-generic
# (ForwardDiff-safe): the base celerite solve is generic and the Woodbury
# core is plain linear algebra.
#
# References: Delisle, Hara & Ségransan 2020 (S+LEAF, arXiv:2004.10678) for
# the LEAF calibration block NightlyOffset approximates; Boisse+ 2011 for the
# coherent rotation-harmonic activity model HarmonicBlock realizes.

using LinearAlgebra: dot, cholesky, Symmetric, logdet, issuccess
using SpecialFunctions: loggamma

# ── StudentT (heavy-tailed white likelihood) ─────────────────────────────────
# First active StudentT on `channel`, or nothing.
function _active_studentt(theta::Theta, noise_models, channel::Symbol)
    for (i, nm) in enumerate(noise_models)
        nm isa StudentT || continue
        noise_channel(nm) === channel || continue
        is_noise_model_active(theta, i) || continue
        return nm
    end
    return nothing
end

# Independent Student-t log-likelihood: r_i ~ t_ν(0, σ_i²).
#   log t = logΓ((ν+1)/2) − logΓ(ν/2) − ½log(νπσ_i²) − (ν+1)/2·log(1 + r_i²/(νσ_i²))
# (the Student-t normalizer carries π, NOT 2π). ForwardDiff-safe via
# SpecialFunctions.loggamma. ν → ∞ recovers the Gaussian.
function _studentt_diag_ll(residuals, variances, ν::T) where {T}
    n = length(residuals)
    half = (ν + one(T)) / 2
    c = loggamma(half) - loggamma(ν / 2) - T(0.5) * log(ν * T(π))
    total = n * c
    @inbounds for i in 1:n
        s2 = variances[i]
        total += -T(0.5) * log(s2) - half * log1p(residuals[i]^2 / (ν * s2))
    end
    return total
end

# ── instrument coverage ──────────────────────────────────────────────────────
# Set of RV-instrument indices a model covers; empty model list ⇒ all present.
function _covered_inst_idx(model_insts::Vector{String}, inst_names::Vector{String})
    isempty(model_insts) && return Set{Int}()          # sentinel: means "all"
    out = Set{Int}()
    for name in model_insts
        k = findfirst(==(name), inst_names)
        k === nothing && error("noise model references instrument `$name` " *
                               "not present in instrument list $(inst_names)")
        push!(out, k)
    end
    return out
end
@inline _is_covered(cov::Set{Int}, ci::Int) = isempty(cov) || (ci in cov)

# ── ErrorScale ───────────────────────────────────────────────────────────────
# Does ErrorScale `m` cover RV instrument `ins_idx`? (empty list ⇒ all).
@inline function _errorscale_covers(theta::Theta, m::ErrorScale, ins_idx::Int)
    inst_names = theta.params.config.instruments.rv_names
    return isempty(m.instruments) || inst_names[ins_idx] in m.instruments
end

# f² for RV instrument `ins_idx` under ErrorScale `m`, or 1 if not covered.
# NOTE: a returned 1.0 is ambiguous with "covered, f=1" — gate application on
# `_errorscale_covers`, since ErrorScale REPLACES jitter whenever it covers,
# regardless of the drawn f.
@inline function error_scale_factor(theta::Theta{T}, m::ErrorScale, ins_idx::Int) where {T}
    _errorscale_covers(theta, m, ins_idx) || return one(T)
    f = theta.values[theta.params.layout.name_to_idx["errscale_$(theta.params.config.instruments.rv_names[ins_idx])"]]
    return f * f
end

# ── night grouping (gap-based) ───────────────────────────────────────────────
# Assign a global night-group id to each observation (0 ⇒ not covered by `m`).
# A night = a maximal run of one instrument's epochs with consecutive gaps
# < `m.gap` days. Also returns, per group, the instrument index it belongs to.
function _night_group_ids(m::NightlyOffset, times::AbstractVector{Float64},
                          inst::AbstractVector{Int}, inst_names::Vector{String})
    N = length(times)
    ids = zeros(Int, N)
    grp_inst = Int[]
    cov = _covered_inst_idx(m.instruments, inst_names)
    insts = isempty(cov) ? sort(unique(inst)) : sort(collect(cov))
    gid = 0
    for ci in insts
        idxs = [i for i in 1:N if inst[i] == ci]
        isempty(idxs) && continue
        idxs = idxs[sortperm([times[i] for i in idxs])]
        gid += 1; push!(grp_inst, ci); ids[idxs[1]] = gid
        for k in 2:length(idxs)
            if times[idxs[k]] - times[idxs[k - 1]] >= m.gap
                gid += 1; push!(grp_inst, ci)
            end
            ids[idxs[k]] = gid
        end
    end
    return ids, grp_inst
end

# Low-rank factor of a NightlyOffset: one column per night-group, entries
# σ_night(instrument) on that night's points. FFᵀ = Σ_g σ² 1_g 1_gᵀ.
function _nightly_factor(m::NightlyOffset, theta::Theta{T},
                         times, inst, inst_names) where {T}
    ids, grp_inst = _night_group_ids(m, times, inst, inst_names)
    ng = length(grp_inst)
    N = length(times)
    s = _channel_suffix(m.channel)
    σ = T[theta.values[theta.params.layout.name_to_idx["night_sigma_$(inst_names[ci])$s"]]
          for ci in grp_inst]
    F = zeros(T, N, ng)
    @inbounds for i in 1:N
        g = ids[i]
        g == 0 || (F[i, g] = σ[g])
    end
    return F
end

# ── harmonic block ───────────────────────────────────────────────────────────
# Low-rank factor of a HarmonicBlock: 2·nharm columns [cos,sin] per harmonic,
# each row scaled by the point's per-instrument amplitude A_ins. The shared
# (a_k,b_k)~N(0,I) latent → shared phase; A_ins → per-instrument amplitude.
# FFᵀ[i,j] = Σ_k A_i A_j cos(ωk(t_i − t_j)).
function _harmonic_factor(m::HarmonicBlock, theta::Theta{T},
                          times, inst, inst_names) where {T}
    N = length(times)
    K = m.nharm
    s = _channel_suffix(m.channel)
    layout = theta.params.layout
    # Two modes: harmonics of a fitted rotation period, or a fixed external
    # frequency comb (see the HarmonicBlock docstring).
    external = !isempty(m.freqs)
    ω = external ? T(0) : T(2π) / theta.values[layout.name_to_idx["harm_period$s"]]
    cov = _covered_inst_idx(m.instruments, inst_names)
    F = zeros(T, N, 2K)
    @inbounds for i in 1:N
        _is_covered(cov, inst[i]) || continue
        A = theta.values[layout.name_to_idx["harm_amp_$(inst_names[inst[i]])$s"]]
        for k in 1:K
            arg = external ? T(2π) * T(m.freqs[k]) * T(times[i]) : ω * k * T(times[i])
            F[i, 2k - 1] = A * cos(arg)
            F[i, 2k]     = A * sin(arg)
        end
    end
    return F
end

# Combined low-rank factor of all active additive-covariance models.
function _additive_factor(add_models, theta::Theta{T},
                          times, inst, inst_names) where {T}
    blocks = Matrix{T}[]
    for m in add_models
        if m isa NightlyOffset
            push!(blocks, _nightly_factor(m, theta, times, inst, inst_names))
        elseif m isa HarmonicBlock
            push!(blocks, _harmonic_factor(m, theta, times, inst, inst_names))
        end
    end
    isempty(blocks) && return zeros(T, length(times), 0)
    return reduce(hcat, blocks)
end

# ── Woodbury core ────────────────────────────────────────────────────────────
# log N(y | 0, B + FFᵀ) given B⁻¹y, B⁻¹F and logdet(B):
#   logdet(Σ)   = logdet(B) + logdet(I + FᵀB⁻¹F)
#   yᵀΣ⁻¹y      = yᵀB⁻¹y − (FᵀB⁻¹y)ᵀ(I + FᵀB⁻¹F)⁻¹(FᵀB⁻¹y)
function _woodbury_core(y, F, Biy, BiF, logdetB, N::Int, two_pi::T) where {T}
    k = size(F, 2)
    qB = dot(y, Biy)
    k == 0 && return -T(0.5) * (qB + logdetB + N * log(two_pi))
    FtBiy = F' * Biy
    M = Matrix{T}(F' * BiF)
    @inbounds for j in 1:k
        M[j, j] += one(T)
    end
    cM = cholesky(Symmetric(M); check = false)
    issuccess(cM) || return T(-Inf)
    z = cM \ FtBiy
    quad = qB - dot(FtBiy, z)
    logdetΣ = logdetB + logdet(cM)
    return -T(0.5) * (quad + logdetΣ + N * log(two_pi))
end

# Base = white diagonal.
function _woodbury_white_ll(y, variances, F, two_pi::T) where {T}
    Biy = y ./ variances
    BiF = F ./ variances                         # per-row broadcast
    logdetB = sum(log, variances)
    return _woodbury_core(y, F, Biy, BiF, logdetB, length(y), two_pi)
end

# Base = one celerite GP. celerite_solve/loglike are T-generic, so we recover
# B⁻¹y, B⁻¹F column-by-column and logdet(B) from the base log-likelihood.
function _woodbury_celerite_ll(y, variances, times, theta::Theta{T}, nm, F,
                                two_pi::T) where {T}
    ar, cr, ac, bc, cc, dc = _celerite_coeffs(theta, nm)
    tv = _as_t_vec(times)
    Biy = celerite_solve(tv, y, variances, ar, cr, ac, bc, cc, dc)
    N = length(y); k = size(F, 2)
    BiF = Matrix{T}(undef, N, k)
    @inbounds for j in 1:k
        BiF[:, j] = celerite_solve(tv, view(F, :, j), variances,
                                    ar, cr, ac, bc, cc, dc)
    end
    llB = celerite_loglike(tv, y, variances, ar, cr, ac, bc, cc, dc)
    isfinite(llB) || return T(-Inf)
    qB = dot(y, Biy)
    logdetB = -2 * llB - qB - N * log(two_pi)     # invert the base loglike
    return _woodbury_core(y, F, Biy, BiF, logdetB, N, two_pi)
end

# celerite coefficients for a base GP (mirror of the gp_log_likelihood methods;
# kept separate so the validated scoring path is untouched).
function _celerite_coeffs(theta::Theta, nm::CeleriteSHO)
    s = _gp_suffix(nm); L = theta.params.layout
    S0 = exp(theta.values[L.name_to_idx["gp_log_S0$s"]])
    Q  = exp(theta.values[L.name_to_idx["gp_log_Q$s"]])
    ω0 = exp(theta.values[L.name_to_idx["gp_log_omega0$s"]])
    return sho_coefficients(S0, Q, ω0)
end
function _celerite_coeffs(theta::Theta, nm::CeleriteRotation)
    s = _gp_suffix(nm); L = theta.params.layout
    return rotation_coefficients(theta.values[L.name_to_idx["gp_sigma$s"]],
                                 theta.values[L.name_to_idx["gp_period$s"]],
                                 theta.values[L.name_to_idx["gp_Q0$s"]],
                                 theta.values[L.name_to_idx["gp_dQ$s"]],
                                 theta.values[L.name_to_idx["gp_f$s"]])
end
function _celerite_coeffs(theta::Theta, nm::CeleriteRotationFM17)
    s = _gp_suffix(nm); L = theta.params.layout
    return rotation_fm17_coefficients(exp(theta.values[L.name_to_idx["gp_log_amp$s"]]),
                                      exp(theta.values[L.name_to_idx["gp_log_timescale$s"]]),
                                      exp(theta.values[L.name_to_idx["gp_log_period$s"]]),
                                      exp(theta.values[L.name_to_idx["gp_log_factor$s"]]))
end

# ── MaternGP (standalone semiseparable short-memory RV GP) ───────────────────
function _matern_params(theta::Theta{T}, nm::MaternGP) where {T}
    s = _gp_suffix(nm); L = theta.params.layout
    return theta.values[L.name_to_idx["matern_sigma$s"]],
           theta.values[L.name_to_idx["matern_rho$s"]]
end

# Standalone Matérn-3/2 GP: a single-series (α=1, β=0) case of the multiseries
# semiseparable machinery, so RV_resid ~ GP(0, K_Matern + diag(var)).
function gp_log_likelihood(residuals::AbstractVector{T}, variances::AbstractVector{T},
                           times::AbstractVector{Float64}, theta::Theta{T},
                           nm::MaternGP) where {T}
    σ, ρ = _matern_params(theta, nm)
    n = length(residuals)
    return multiseries_loglike(times, residuals, variances, ones(Int, n),
                                T[one(T)], T[zero(T)], SSMatern32(σ, ρ))
end

# Woodbury with a Matérn (semiseparable) base: B = K_Matern + diag(var).
function _woodbury_matern_ll(y, variances, times, theta::Theta{T}, nm::MaternGP, F,
                             two_pi::T) where {T}
    σ, ρ = _matern_params(theta, nm)
    kern = SSMatern32(σ, ρ)
    perm = sortperm(times)
    ts = Float64.(times[perm]); ys = y[perm]; Fs = F[perm, :]
    # _base_generators returns N×r; the solver (like multiseries_loglike) wants
    # r×N. α=1, β=0 ⇒ no derivative generators, A = var + k(0).
    U0, V0, _, _, φ0 = _base_generators(_terms(kern), ts, diff(ts))
    U = permutedims(U0); V = permutedims(V0)
    φ = length(ts) > 1 ? permutedims(φ0) : similar(U, size(U, 1), 0)
    A = variances[perm] .+ k0(kern)
    Z, logdetB = semiseparable_solve_logdet(ts, hcat(ys, Fs), A, U, V, φ)
    Z === nothing && return T(-Inf)
    Biy = Z[:, 1]
    BiF = size(Fs, 2) == 0 ? zeros(T, length(ys), 0) : Z[:, 2:end]
    return _woodbury_core(ys, Fs, Biy, BiF, logdetB, length(ys), two_pi)
end

# ── dense oracle (testing only) ──────────────────────────────────────────────
# Explicit Σ = base + FFᵀ Cholesky log-likelihood, for parity gates.
function dense_additive_ll(y, variances, times, inst, inst_names,
                            add_models, theta::Theta{T};
                            base_nm = nothing, two_pi::T = T(2π)) where {T}
    N = length(y)
    Σ = zeros(T, N, N)
    if base_nm === nothing
        @inbounds for i in 1:N
            Σ[i, i] = variances[i]
        end
    else
        ar, cr, ac, bc, cc, dc = _celerite_coeffs(theta, base_nm)
        @inbounds for i in 1:N, j in 1:N
            τ = abs(T(times[i] - times[j]))
            kτ = zero(T)
            for r in eachindex(ar); kτ += ar[r] * exp(-cr[r] * τ); end
            for c in eachindex(ac)
                kτ += exp(-cc[c] * τ) * (ac[c] * cos(dc[c] * τ) + bc[c] * sin(dc[c] * τ))
            end
            Σ[i, j] = kτ
        end
        @inbounds for i in 1:N
            Σ[i, i] += variances[i]
        end
    end
    F = _additive_factor(add_models, theta, times, inst, inst_names)
    Σ .+= F * F'
    C = cholesky(Symmetric(Σ); check = false)
    issuccess(C) || return T(-Inf)
    α = C \ y
    return -T(0.5) * (dot(y, α) + logdet(C) + N * log(two_pi))
end
