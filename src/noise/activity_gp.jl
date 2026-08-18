# Multivariate activity GP (Rajpaul+ 2015).
#
# A single latent process `G(t)` drives RV and a list of activity
# indicators jointly via linear combinations of `G(t)` and its time
# derivative `dG/dt`. The data-vector covariance is built from four
# analytic kernel blocks evaluated on every pair of observation times,
# weighted by the channel coefficients of each observation. Captures
# rotation-phase-shifted activity-RV coupling that single-channel
# decorrelation (`ActivityDecorrelation`, FF′) cannot reach.
#
# This module exposes the **kernel math + a standalone multivariate
# log-likelihood**, so the algebra can be unit-tested directly. Wiring
# `ActivityGP` into the production `rv_log_likelihood` and Params
# auto-priors is a follow-up commit.
#
# Reference:
#   Rajpaul, V., Aigrain, S., Osborne, M.A., Reece, S., Roberts, S.,
#   2015, MNRAS, 452, 2269 — "A Gaussian process framework for
#   modelling stellar activity signals in radial velocity data."

using LinearAlgebra: Cholesky, cholesky, cholesky!, ldiv!, Symmetric, Diagonal,
                     dot, logdet, diagind, I

# =====================================================================
# Quasi-periodic kernel + analytic derivatives
# =====================================================================
#
# k(τ) = σ² · exp(f(τ))
# f(τ) = -τ²/(2λ_e²) - sin²(πτ/P) / (2λ_p²)
#
# With τ = t_i - t_j and ∂τ/∂t_j = -1:
#   k_GG(t_i, t_j)     = k(τ)
#   k_GdotG(t_i, t_j)  = ∂_{t_j} k(τ) = -f'(τ) · k(τ)
#   k_dotGG(t_i, t_j)  = ∂_{t_i} k(τ) =  f'(τ) · k(τ)
#   k_dotGdotG(t_i, t_j) = ∂²_{t_i, t_j} k(τ) = -(f''(τ) + f'(τ)²) · k(τ)
#
# Closed form:
#   f'(τ)  = -τ/λ_e² - π / (P λ_p²) · sin(π τ / P) · cos(π τ / P)
#          = -τ/λ_e² - π / (2 P λ_p²) · sin(2π τ / P)
#   f''(τ) = -1/λ_e² - π² / (P² λ_p²) · cos(2π τ / P)

"""
    activity_kernel_blocks(τ, amp, period, λ_e, λ_p)
        -> (k_GG, k_GdotG, k_dotGG, k_dotGdotG)

Evaluate the four kernel blocks at lag `τ = t_i - t_j` for the
Rajpaul quasi-periodic kernel with overall amplitude `amp`, rotation
`period`, exponential length `λ_e`, and periodic length `λ_p`.
"""
@inline function activity_kernel_blocks(τ::Real, amp::Real,
                                         period::Real,
                                         λ_e::Real, λ_p::Real)
    # f(τ) and derivatives
    s = sin(π * τ / period)
    c = cos(π * τ / period)
    s2 = sin(2π * τ / period)
    c2 = cos(2π * τ / period)

    f_val   = -τ^2 / (2 * λ_e^2) - s^2 / (2 * λ_p^2)
    f_p     = -τ / λ_e^2 - π / (2 * period * λ_p^2) * s2
    f_pp    = -1.0 / λ_e^2 - π^2 / (period^2 * λ_p^2) * c2

    k       = amp^2 * exp(f_val)
    k_GG       = k
    k_GdotG    = -f_p * k
    k_dotGG    =  f_p * k
    k_dotGdotG = -(f_pp + f_p^2) * k
    return (k_GG, k_GdotG, k_dotGG, k_dotGdotG)
end

# =====================================================================
# FM17 rotation kernel + analytic derivatives (celerite-form)
# =====================================================================
#
# *** WARNING — do NOT use these blocks in a Rajpaul joint covariance ***
#
# FM17 has `exp(-c|τ|)` factors, so `k(τ)` is only C⁰ at `τ = 0`
# (kink). As a distribution, `k''(τ)` then picks up a
# `-2c·(ar+ac)·δ(τ)` Dirac term from the |τ| kink. The formal
# `Var(dG/dt) = -k''(0)` is therefore **infinite**, not the finite
# right-limit `-c²(ar+ac) + d²·ac` that a naive evaluation gives.
#
# In practice this means assembling a Rajpaul joint Σ from the FM17
# blocks below produces a matrix that's NOT positive-definite for
# generic parameter / data configurations — the diagonal is finite
# while the true variance is infinite, so the conditioning fails.
# Empirically confirmed: synthetic 30-point mixed-channel dataset at
# `amp=0.8, τ_decay=35d, P=9.6d, factor=0.4` gives min eigenvalue
# ≈ −0.005 on the assembled covariance.
#
# This is structural to every celerite-form kernel — they all carry the
# |τ| kink. QP avoids the issue because its exponent
# `-τ²/(2λ_e²) - sin²(πτ/P)/(2λ_p²)` is C^∞ at 0. The O(n) celerite-
# block route for joint Rajpaul is therefore a dead-end; an exact
# O(n) AGP needs the state-space / Kalman-filter formulation (state
# vector carries `G` and `dG/dt` jointly, so the kink-induced delta is
# absorbed into the transition dynamics rather than the covariance).
#
# The functions below remain useful as building blocks for any
# *single-channel* celerite kernel work using FM17 + its derivatives
# (e.g., a future Kalman filter implementation, or a Gaussian process
# regression with FM17 plus a separate observation operator). They
# are NOT used to assemble the Rajpaul joint Σ in Nereus.
#
# The quasi-periodic kernel above is NOT a finite sum of damped
# exponentials, so its joint covariance is O(n³) Cholesky-bound.
# Foreman-Mackey+ 2017 (FM17) introduced a rotation kernel that IS
# celerite-form:
#
#   k(τ) = ar · exp(−cr |τ|) + ac · exp(−cc |τ|) · cos(dc τ)
#
# with `ar = a (1+f)/(2+f)`, `cr = cc = 1/τ_decay`, `ac = a/(2+f)`,
# `dc = 2π/P`. Single-channel celerite for `k` already lives in
# `noise/gp.jl::rotation_fm17_coefficients`.
#
# `k'` and `k''` are also in celerite sum-of-exponentials form —
# derivatives of `aₖ e^(−c|τ|) cos(d τ)` stay in that closed class
# with a permuted (a, b) cos/sin amplitude pair. The closed forms are
# derived analytically:
#
#   k'(τ)  = −cr·ar e^(−cr|τ|) · sign(τ)
#          + e^(−cc|τ|) · [−cc·ac · sign(τ) · cos(dc τ)
#                          − dc·ac          · sin(dc τ)]
#   k''(τ) = cr²·ar e^(−cr|τ|)
#          + e^(−cc|τ|) · [(cc² − dc²)·ac        · cos(dc τ)
#                          + 2·cc·dc·ac · sign(τ) · sin(dc τ)]
#
# The `sign(τ)` factors make `k'` antisymmetric (as it must be, since
# `∂_{t_j} k(t_i − t_j) = −∂_{t_i} k(t_i − t_j)`); `k` and `k''` are
# symmetric. These signs vanish in the final Σ because each pair lag
# enters once with τ = t_i − t_j of a definite sign.
#
# Validated against finite-difference of the FM17 closed form to 5+
# digits at τ ∈ {0.5, 1.7, 5.0, 12.3} d for amp=2, τ_decay=50 d,
# P=10 d, f=0.3.

"""
    activity_kernel_blocks_fm17(τ, amp, τ_decay, period, factor)
        -> (k_GG, k_GdotG, k_dotGG, k_dotGdotG)

Evaluate the four Rajpaul kernel blocks at lag `τ = t_i - t_j` for the
Foreman-Mackey+ 2017 rotation kernel. Shape-compatible with
[`activity_kernel_blocks`](@ref) (which uses the quasi-periodic kernel)
so it can be dropped in via a kernel-form switch.

This evaluates the closed-form `k, k', k''` directly. The corresponding
celerite sum-of-exponentials coefficients (needed by the O(n) joint
solver in Phase 2) are returned by
[`activity_kernel_blocks_fm17_celerite`](@ref).
"""
@inline function activity_kernel_blocks_fm17(τ::Real, amp::Real,
                                              τ_decay::Real,
                                              period::Real, factor::Real)
    ar = amp * (1 + factor) / (2 + factor)
    cr = 1 / τ_decay
    ac = amp / (2 + factor)
    cc = 1 / τ_decay
    dc = 2π / period

    abs_τ = abs(τ)
    sτ    = sign(τ)
    er    = exp(-cr * abs_τ)
    ec    = exp(-cc * abs_τ)
    cos_d = cos(dc * τ)
    sin_d = sin(dc * τ)

    # k(τ) — symmetric
    k = ar * er + ac * ec * cos_d

    # k'(τ) — antisymmetric
    kp = -cr * sτ * ar * er +
          ec * (-cc * sτ * ac * cos_d - dc * ac * sin_d)

    # k''(τ) — symmetric
    kpp = cr * cr * ar * er +
           ec * ((cc * cc - dc * dc) * ac * cos_d +
                  2 * cc * sτ * dc * ac * sin_d)

    k_GG       = k
    k_GdotG    = -kp     # = ∂_{t_j} k(τ) = -k'(τ)
    k_dotGG    =  kp     # = ∂_{t_i} k(τ) =  k'(τ)
    k_dotGdotG = -kpp    # = ∂²_{t_i,t_j} k(τ) = -k''(τ)
    return (k_GG, k_GdotG, k_dotGG, k_dotGdotG)
end

"""
    activity_kernel_blocks_fm17_celerite(amp, τ_decay, period, factor)
        -> NamedTuple{(:k, :kp, :kpp)}

Return celerite (real + complex) sum-of-exponentials coefficients for
the FM17 rotation kernel `k(τ)` and its first two derivatives `k'(τ)`,
`k''(τ)`, packaged for the multi-channel O(n) celerite-block solver.

Each entry is a NamedTuple `(ar, cr, ac, bc, cc, dc, sign_real,
sign_cos, sign_sin)`:

  - `ar, cr` — coefficient + decay rate of the (single) real term:
    `ar · e^(−cr|τ|) · [sign(τ) if sign_real == 1 else 1]`
  - `ac, bc, cc, dc` — celerite complex term:
    `e^(−cc|τ|) · (ac · [sign(τ)^sign_cos] · cos(dc τ)
                    + bc · [sign(τ)^sign_sin] · sin(dc τ))`
  - `sign_real, sign_cos, sign_sin` ∈ {0, 1}: extra `sign(τ)`
    factors enforcing the symmetry (k, k'' symmetric in τ; k'
    antisymmetric). `cos(dc τ)` is naturally even, `sin(dc τ)`
    naturally odd — so to keep `k'` antisymmetric we toggle the cos
    factor (and the real-term factor), while to keep `k''` symmetric
    we toggle the sin factor.

Symmetry table (all flags ∈ {0, 1}):

| Term | sign_real | sign_cos | sign_sin |
|------|-----------|----------|----------|
| k    |     0     |     0    |     0    |
| k'   |     1     |     1    |     0    |
| k''  |     0     |     0    |     1    |
"""
function activity_kernel_blocks_fm17_celerite(amp::Real, τ_decay::Real,
                                                period::Real, factor::Real)
    ar = amp * (1 + factor) / (2 + factor)
    cr = 1 / τ_decay
    ac = amp / (2 + factor)
    cc = 1 / τ_decay
    dc = 2π / period

    k_terms   = (ar = ar,       cr = cr,
                  ac = ac,       bc = zero(ac),
                  cc = cc,       dc = dc,
                  sign_real = 0, sign_cos = 0, sign_sin = 0)

    kp_terms  = (ar = -cr * ar, cr = cr,
                  ac = -cc * ac, bc = -dc * ac,
                  cc = cc,        dc = dc,
                  sign_real = 1,  sign_cos = 1, sign_sin = 0)

    kpp_terms = (ar = cr * cr * ar,   cr = cr,
                  ac = (cc * cc - dc * dc) * ac,
                  bc = 2 * cc * dc * ac,
                  cc = cc,             dc = dc,
                  sign_real = 0,       sign_cos = 0, sign_sin = 1)

    return (k = k_terms, kp = kp_terms, kpp = kpp_terms)
end

# =====================================================================
# Channel coefficient lookup
# =====================================================================

"""
    ActivityGPCoeffs(coeffs::Dict{Symbol, Tuple{Float64, Float64}})

Maps `:rv => (Vc, Vr)`, `:bis => (Bc, Br)`, … for the active
channels. Pass to [`activity_gp_log_likelihood`](@ref). Missing
channels mean the channel doesn't contribute (its observations are
ignored, or in a flat layout, its rows would zero out).
"""
const ActivityGPCoeffs = Dict{Symbol, NTuple{2, Float64}}

# =====================================================================
# Multivariate log-likelihood
# =====================================================================

"""
    activity_gp_log_likelihood(t, y, σ, channel,
                                amp, period, λ_e, λ_p, coeffs;
                                jitter_per_channel=nothing) -> Float64

Joint Gaussian log-likelihood under the Rajpaul multivariate-GP
covariance. All observations from every channel are passed in flat
arrays:

- `t::Vector{Float64}` — observation times (any order)
- `y::Vector{Float64}` — observed value at each time
  (mean-subtracted by the caller — i.e., RV residuals after
  Keplerian + offsets, indicators ideally zero-mean)
- `σ::Vector{Float64}` — measurement uncertainty per point
- `channel::Vector{Symbol}` — `:rv` / `:bis` / `:fwhm` / `:logrhk`
  / `:halpha` per point
- `amp`, `period`, `λ_e`, `λ_p` — quasi-periodic kernel hyperparams
- `coeffs::ActivityGPCoeffs` — `(coef_on_G, coef_on_dGdt)` per channel
- `jitter_per_channel::Union{Nothing, Dict{Symbol, Float64}}` —
  additional per-channel jitter added in quadrature to `σ`

Returns the log-likelihood `log p(y | hyperparams, coeffs)`.
"""
function activity_gp_log_likelihood(t::AbstractVector{<:Real},
                                     y::AbstractVector{<:Real},
                                     σ::AbstractVector{<:Real},
                                     channel::AbstractVector{Symbol},
                                     amp::Real, period::Real,
                                     λ_e::Real, λ_p::Real,
                                     coeffs::ActivityGPCoeffs;
                                     jitter_per_channel::Union{Nothing,
                                         Dict{Symbol, <:Real}} = nothing)
    n = length(t)
    length(y) == n && length(σ) == n && length(channel) == n ||
        throw(ArgumentError("t, y, σ, channel length mismatch"))
    amp > 0 && period > 0 && λ_e > 0 && λ_p > 0 ||
        throw(ArgumentError("kernel hyperparameters must be positive"))
    isempty(coeffs) &&
        throw(ArgumentError("empty coeffs Dict"))

    # Per-point (a, b) coefficients on (G, dG/dt) and per-point jitter.
    a = Vector{Float64}(undef, n)
    b = Vector{Float64}(undef, n)
    σ_total = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ch = channel[i]
        haskey(coeffs, ch) ||
            throw(ArgumentError("no coefficients for channel $ch"))
        a[i], b[i] = coeffs[ch]
        j = jitter_per_channel === nothing ? 0.0 :
            get(jitter_per_channel, ch, 0.0)
        σ_total[i] = sqrt(σ[i]^2 + j^2)
    end

    Σ = activity_gp_covariance(t, channel, a, b, amp, period, λ_e, λ_p)
    @inbounds for i in 1:n
        Σ[i, i] += σ_total[i]^2
    end

    # Cholesky-based log-likelihood. Numerically stable for moderate n.
    F = cholesky(Symmetric(Σ); check = false)
    if !issuccess(F)
        return -Inf
    end
    α = F \ y
    return -0.5 * (dot(y, α) + logdet(F) + n * log(2π))
end

"""
    activity_gp_covariance(t, channel, a, b, amp, period, λ_e, λ_p) -> Matrix

Build the `n × n` Rajpaul-GP covariance (no measurement noise added).
Each off-diagonal `Σ[i, j]` is

    a_i a_j k_GG(τ) + a_i b_j k_GdotG(τ) + b_i a_j k_dotGG(τ) + b_i b_j k_dotGdotG(τ)

with `τ = t_i - t_j`. Sign of the derivative blocks comes from
`∂_{t_j} k(τ)` having opposite sign to `∂_{t_i} k(τ)`.
"""
function activity_gp_covariance(t::AbstractVector{<:Real},
                                  channel::AbstractVector{Symbol},
                                  a::AbstractVector{<:Real},
                                  b::AbstractVector{<:Real},
                                  amp::Real, period::Real,
                                  λ_e::Real, λ_p::Real)
    n = length(t)
    T = promote_type(eltype(a), eltype(b),
                      typeof(amp), typeof(period),
                      typeof(λ_e), typeof(λ_p))
    Σ = Matrix{T}(undef, n, n)

    # Σ is symmetric (cov(y_i, y_j) = cov(y_j, y_i)), so we only compute
    # the upper triangle (i ≤ j) and mirror. Kernel math is inlined to
    # avoid the per-pair tuple allocation from activity_kernel_blocks
    # — about 2× faster overall.
    amp2     = amp * amp
    inv_λe2  = 1 / (λ_e * λ_e)
    inv_2λp2 = 1 / (2 * λ_p * λ_p)
    inv_P    = 1 / period
    π_P      = π * inv_P
    two_π_P  = 2 * π_P
    π2_P2λp2 = π * π * inv_P * inv_P * (2 * inv_2λp2)
    π_P_λp2  = π_P * (2 * inv_2λp2)

    @inbounds for j in 1:n
        a_j = a[j]; b_j = b[j]
        t_j = t[j]
        for i in 1:j
            τ = t[i] - t_j
            # f(τ) = -τ²/(2λe²) - sin²(πτ/P)/(2λp²)
            # f'(τ) = -τ/λe² - π/(P λp²) sin(πτ/P) cos(πτ/P)
            #       = -τ/λe² - π/(2P λp²) sin(2πτ/P)
            # f''(τ) = -1/λe² - π²/(P² λp²) cos(2πτ/P)
            s     = sin(π_P * τ)
            c     = cos(π_P * τ)
            s2    = sin(two_π_P * τ)
            c2    = cos(two_π_P * τ)
            f_val = -τ * τ * (0.5 * inv_λe2) - s * s * inv_2λp2
            f_p   = -τ * inv_λe2 - 0.5 * π_P_λp2 * s2
            f_pp  = -inv_λe2 - π2_P2λp2 * c2
            k     = amp2 * exp(f_val)
            k_GG       = k
            k_GdotG    = -f_p * k    # = ∂_{t_j} k
            k_dotGG    =  f_p * k    # = ∂_{t_i} k
            k_dotGdotG = -(f_pp + f_p * f_p) * k

            a_i = a[i]; b_i = b[i]
            val = a_i * a_j * k_GG +
                   a_i * b_j * k_GdotG +
                   b_i * a_j * k_dotGG +
                   b_i * b_j * k_dotGdotG
            Σ[i, j] = val
            i == j || (Σ[j, i] = val)
        end
    end
    return Σ
end

"""
    activity_gp_covariance_blocked(epochs, chan_a, chan_b, amp, period, λ_e, λ_p)
        -> Matrix

Block-factored assembly of the Rajpaul-GP covariance for the common case where
**every channel shares the same `epochs`** (Rajpaul-standard: indicators come
from the same spectra as the RV, so Nereus places every channel at `data.t_rv`)
and the `(a, b)` coefficients are **constant within each channel** (`chan_a[c]`,
`chan_b[c]`).

The 4 kernel blocks `k_GG, k_GdotG, k_dotGG, k_dotGdotG` depend only on the
epoch-pair lag `τ`, so they are computed ONCE on the `N×N` epoch grid (the only
transcendentals — `sin/cos/exp`), then the full `(N·C)×(N·C)` covariance is a
`C×C` grid of `N×N` blocks, each a scalar combination of those 4 base matrices:

    Σ_block(ci,cj)[p,q] = a_ci·a_cj·K_GG + a_ci·b_cj·K_GdotG
                        + b_ci·a_cj·K_dotGG + b_ci·b_cj·K_dotGdotG

This is **numerically identical** (to machine precision) to building the dense
`(N·C)×(N·C)` matrix with `activity_gp_covariance` on the channel-major flat
arrays, but evaluates the transcendentals `C²×` fewer times (e.g. 25× for the
5-channel HD 18599 layout). Channel-major layout: block `c` occupies indices
`(c-1)·N+1 : c·N`, matching the flat construction in the likelihood (RV first,
then indicators) so the marginalize-indicators partitioning still lines up.
"""
function activity_gp_covariance_blocked(epochs::AbstractVector{<:Real},
                                         chan_a::AbstractVector,
                                         chan_b::AbstractVector,
                                         amp::Real, period::Real,
                                         λ_e::Real, λ_p::Real)
    N = length(epochs); C = length(chan_a)
    C == length(chan_b) ||
        throw(ArgumentError("chan_a/chan_b length mismatch"))
    T = promote_type(eltype(chan_a), eltype(chan_b), typeof(amp),
                      typeof(period), typeof(λ_e), typeof(λ_p))

    # 4 base kernel blocks on the shared epoch grid (only transcendentals).
    KGG = Matrix{T}(undef, N, N)   # k_GG
    KGd = Matrix{T}(undef, N, N)   # k_GdotG  = ∂_{t_j} k
    KdG = Matrix{T}(undef, N, N)   # k_dotGG  = ∂_{t_i} k
    Kdd = Matrix{T}(undef, N, N)   # k_dotGdotG
    amp2     = amp * amp
    inv_λe2  = 1 / (λ_e * λ_e)
    inv_2λp2 = 1 / (2 * λ_p * λ_p)
    inv_P    = 1 / period
    π_P      = π * inv_P
    two_π_P  = 2 * π_P
    π2_P2λp2 = π * π * inv_P * inv_P * (2 * inv_2λp2)
    π_P_λp2  = π_P * (2 * inv_2λp2)
    @inbounds for q in 1:N
        t_q = epochs[q]
        for p in 1:q
            τ    = epochs[p] - t_q
            s    = sin(π_P * τ)
            s2   = sin(two_π_P * τ)
            c2   = cos(two_π_P * τ)
            f_val = -τ * τ * (0.5 * inv_λe2) - s * s * inv_2λp2
            f_p   = -τ * inv_λe2 - 0.5 * π_P_λp2 * s2
            f_pp  = -inv_λe2 - π2_P2λp2 * c2
            k     = amp2 * exp(f_val)
            gd    = -f_p * k          # ∂_{t_q}
            dg    =  f_p * k          # ∂_{t_p}
            dd    = -(f_pp + f_p * f_p) * k
            KGG[p, q] = k;   KGG[q, p] = k       # even in τ
            Kdd[p, q] = dd;  Kdd[q, p] = dd      # even in τ
            KGd[p, q] = gd;  KGd[q, p] = dg      # f_p odd → swap on mirror
            KdG[p, q] = dg;  KdG[q, p] = gd
        end
    end

    # Assemble Σ as a C×C grid of N×N blocks via per-channel scalar weights.
    n = N * C
    Σ = Matrix{T}(undef, n, n)
    @inbounds for cj in 1:C
        aj = chan_a[cj]; bj = chan_b[cj]; colo = (cj - 1) * N
        for ci in 1:C
            ai = chan_a[ci]; bi = chan_b[ci]; rowo = (ci - 1) * N
            w_gg = ai * aj; w_gd = ai * bj; w_dg = bi * aj; w_dd = bi * bj
            for q in 1:N, p in 1:N
                Σ[rowo + p, colo + q] =
                    w_gg * KGG[p, q] + w_gd * KGd[p, q] +
                    w_dg * KdG[p, q] + w_dd * Kdd[p, q]
            end
        end
    end
    return Σ
end

# =====================================================================
# Posterior over the latent G(t) at a prediction grid
# =====================================================================

"""
    activity_gp_predict(chains, params, data;
                         t_pred=nothing, n_draws=100,
                         rng=default_rng()) -> NamedTuple

Posterior over the latent activity process `G(t)` from an
`ActivityGP`-fitted chain. For each posterior draw, builds the joint
RV+indicator covariance at observation times, conditions on the
data, and computes the **posterior mean of G** at every point in
`t_pred`. Stacking across draws gives the parameter-uncertainty
band — the paper-figure-grade Rajpaul plot.

Skips the inner GP-sampling step (which would also add intrinsic GP
uncertainty per draw) — for the activity-recovery diagnostic the
parameter band is what matters; intrinsic GP uncertainty at the
prediction grid is small once the data has constrained the kernel.

Returns a `NamedTuple` with:
- `t_pred::Vector{Float64}` — the prediction grid
- `G_samples::Matrix{Float64}` of size `(n_pred, n_draws_eff)`
- `G_mean`, `G_lo`, `G_hi::Vector{Float64}` — across-draw mean and
  16/84 percentiles per prediction time

Throws `ArgumentError` if no `ActivityGP` is configured.
"""
function activity_gp_predict(chains, params::Params, data::Data;
                              t_pred::Union{Nothing, AbstractVector{<:Real}} = nothing,
                              n_draws::Int = 100,
                              rng::AbstractRNG = default_rng())
    agp = nothing
    for nm in params.config.noise_models
        if nm isa ActivityGP
            agp = nm
            break
        end
    end
    agp === nothing &&
        throw(ArgumentError("activity_gp_predict requires an ActivityGP in params.config.noise_models"))

    n_rv_obs = length(data.t_rv)
    n_rv_obs > 0 ||
        throw(ArgumentError("activity_gp_predict requires RV observations"))

    if t_pred === nothing
        t_min, t_max = extrema(data.t_rv)
        t_pred = collect(range(t_min, t_max; length = 200))
    else
        t_pred = collect(Float64, t_pred)
    end
    n_pred = length(t_pred)

    chains_flat, n_total = _flatten_chains(chains)
    # Eligibility filter: under transdim_noise=true the chain stores
    # AGP-kernel parameter values regardless of whether AGP was the
    # active noise model on that draw. Without a likelihood gradient
    # pulling those params toward physical values they drift across
    # the temperature ladder. Two-stage filter:
    #   (a) keep only draws where noise_active_<agp_idx> == 1
    #   (b) keep only draws whose kernel hyperparams (gp_act_amp,
    #       gp_act_period, gp_act_lambda_e, gp_act_lambda_p) fall
    #       inside the user prior bounds — guards against post-swap
    #       hot-chain pollution that survives in the cold chain.
    eligible = collect(1:n_total)
    nm_idx_agp = findfirst(nm -> nm === agp, params.config.noise_models)
    active_col = nm_idx_agp === nothing ? nothing :
                  Symbol("noise_active_", nm_idx_agp)
    if active_col !== nothing && haskey(chains_flat, active_col)
        eligible = findall(chains_flat[active_col] .> 0.5)
    end
    s = _gp_suffix(agp)
    function _in_prior(name)
        idx = findfirst(==(name), params.layout.unfrozen_names)
        idx === nothing && return nothing
        return bounds(params.layout.unfrozen_priors[idx])
    end
    bounds_check = Tuple{Symbol, Float64, Float64}[]
    for nm in ("gp_act_period$s",
                "gp_act_lambda_e$s", "gp_act_lambda_p$s")
        b = _in_prior(nm)
        b === nothing && continue
        push!(bounds_check, (Symbol(nm), b[1], b[2]))
    end
    if !isempty(bounds_check)
        keep = trues(length(eligible))
        @inbounds for (sym, lo, hi) in bounds_check
            haskey(chains_flat, sym) || continue
            v = chains_flat[sym]
            for (k, i) in enumerate(eligible)
                keep[k] && (keep[k] = lo <= v[i] <= hi)
            end
        end
        eligible = eligible[keep]
    end
    isempty(eligible) && throw(ArgumentError(
        "no posterior draws survive the ActivityGP filter " *
        "(noise_active=1 AND kernel hyperparams in prior bounds). " *
        "Likely cause: sampler explored the AGP branch poorly — " *
        "increase n_steps, use a prior-seeded init, or tighten the " *
        "gp_act_* priors."))
    n_draws_eff = min(n_draws, length(eligible))
    idx_pool = shuffle(rng, eligible)[1:n_draws_eff]

    s = _gp_suffix(agp)
    layout = params.layout

    G_samples  = Matrix{Float64}(undef, n_pred, n_draws_eff)
    dG_samples = Matrix{Float64}(undef, n_pred, n_draws_eff)
    Vc_samples = Vector{Float64}(undef, n_draws_eff)
    Vr_samples = Vector{Float64}(undef, n_draws_eff)

    for (d, idx) in enumerate(idx_pool)
        theta = _theta_from_row(chains_flat, idx, params)

        # Unit-variance G(t) (Rajpaul+ 2015); guarded for legacy chains.
        amp_idx = get(layout.name_to_idx, "gp_act_amp$s", 0)
        amp = amp_idx == 0 ? 1.0 : theta.values[amp_idx]
        P   = theta.values[layout.name_to_idx["gp_act_period$s"]]
        λe  = theta.values[layout.name_to_idx["gp_act_lambda_e$s"]]
        λp  = theta.values[layout.name_to_idx["gp_act_lambda_p$s"]]
        if !(amp > 0 && P > 0 && λe > 0 && λp > 0)
            G_samples[:, d]  .= NaN
            dG_samples[:, d] .= NaN
            Vc_samples[d] = NaN
            Vr_samples[d] = NaN
            continue
        end

        # Derivative couplings sampled as amplitudes — physical Ġ
        # coefficient = amplitude / std(Ġ) (mirrors the likelihood).
        inv_sdG = 1 / sqrt(1 / (λe * λe) + π * π / (P * P * λp * λp))
        Vc = theta.values[layout.name_to_idx["Vc$s"]]
        Vr = agp.use_derivative ?
             theta.values[layout.name_to_idx["Vr$s"]] * inv_sdG : 0.0
        Vc_samples[d] = Vc
        Vr_samples[d] = Vr

        # Build flat observation arrays — same shape as the likelihood
        # path.
        ind_meta = Tuple{Symbol, Vector{Float64}, Vector{Float64}, Float64, Float64, Float64}[]
        n_obs_total = n_rv_obs
        for ch in agp.channels
            ch === :rv && continue
            name = String(ch)
            haskey(data.indicators, name) || throw(ArgumentError(
                "activity_gp_predict requires data.indicators[\"$name\"]"))
            haskey(data.indicator_errs, name) || throw(ArgumentError(
                "activity_gp_predict requires data.indicator_errs[\"$name\"]"))
            vals = data.indicators[name]
            errs = data.indicator_errs[name]
            cg, cd = _ACTIVITY_GP_COEFFS[ch]
            a_coef = theta.values[layout.name_to_idx[string(cg, s)]]
            b_coef = (cd === nothing || !agp.use_derivative) ? 0.0 :
                      theta.values[layout.name_to_idx[string(cd, s)]] * inv_sdG
            jit_idx = get(layout.name_to_idx, "gp_act_jit_$(ch)$s", 0)
            jit² = jit_idx == 0 ? 0.0 : Float64(theta.values[jit_idx])^2
            push!(ind_meta, (ch, vals, errs, a_coef, b_coef, jit²))
            n_obs_total += length(vals)
        end

        # Re-evaluate Stage-1 predictions to get the RV mean to subtract
        # (Keplerian + offsets + activity-decorrelation mean modifier).
        preds, vars = rv_predictions(theta, data)

        t_obs  = Vector{Float64}(undef, n_obs_total)
        y_obs  = Vector{Float64}(undef, n_obs_total)
        σ²_obs = Vector{Float64}(undef, n_obs_total)
        a_obs  = Vector{Float64}(undef, n_obs_total)
        b_obs  = Vector{Float64}(undef, n_obs_total)

        @inbounds for i in 1:n_rv_obs
            t_obs[i]  = data.t_rv[i]
            y_obs[i]  = data.rv[i] - preds[i]
            σ²_obs[i] = vars[i]
            a_obs[i]  = Vc
            b_obs[i]  = Vr
        end
        offset = n_rv_obs
        for (_, vals, errs, a_coef, b_coef, jit²) in ind_meta
            n_ch = length(vals)
            @inbounds for i in 1:n_ch
                t_obs[offset + i]  = data.t_rv[i]
                y_obs[offset + i]  = vals[i]
                σ²_obs[offset + i] = errs[i]^2 + jit²
                a_obs[offset + i]  = a_coef
                b_obs[offset + i]  = b_coef
            end
            offset += n_ch
        end

        # Σ_oo: joint covariance at observation times.
        ch_obs = vcat(fill(:rv, n_rv_obs),
                       vcat(fill.(getindex.(ind_meta, 1),
                                    length.(getindex.(ind_meta, 2)))...))
        Σ_oo = activity_gp_covariance(t_obs, ch_obs, a_obs, b_obs,
                                       amp, P, λe, λp)
        @inbounds for i in 1:n_obs_total
            Σ_oo[i, i] += σ²_obs[i]
        end

        # Cross-covariance between obs (linear combo a·G + b·dG/dt) and
        # the two prediction quantities:
        #   K_op  [i,j]  = a·k_GG(τ)    + b·k_dotGG(τ)        → for G(t*_j)
        #   K_op_dG[i,j] = a·k_GdotG(τ) + b·k_dotGdotG(τ)     → for dG/dt(t*_j)
        # τ = t_obs[i] - t_pred[j].
        K_op    = Matrix{Float64}(undef, n_obs_total, n_pred)
        K_op_dG = Matrix{Float64}(undef, n_obs_total, n_pred)
        @inbounds for j in 1:n_pred
            for i in 1:n_obs_total
                τ = t_obs[i] - t_pred[j]
                k_GG, k_GdotG, k_dotGG, k_dotGdotG =
                    activity_kernel_blocks(τ, amp, P, λe, λp)
                K_op[i, j]    = a_obs[i] * k_GG    + b_obs[i] * k_dotGG
                K_op_dG[i, j] = a_obs[i] * k_GdotG + b_obs[i] * k_dotGdotG
            end
        end

        F = cholesky(Symmetric(Σ_oo); check = false)
        if !issuccess(F)
            G_samples[:, d]  .= NaN
            dG_samples[:, d] .= NaN
            continue
        end
        α = F \ y_obs
        @inbounds for j in 1:n_pred
            G_samples[j, d]  = dot(view(K_op, :, j), α)
            dG_samples[j, d] = dot(view(K_op_dG, :, j), α)
        end
    end

    G_mean,  G_lo,  G_hi  = _quantile_band(G_samples)
    dG_mean, dG_lo, dG_hi = _quantile_band(dG_samples)

    return (; t_pred,
              G_samples,  G_mean,  G_lo,  G_hi,
              dG_dt_samples = dG_samples,
              dG_dt_mean = dG_mean, dG_dt_lo = dG_lo, dG_dt_hi = dG_hi,
              Vc_samples, Vr_samples)
end

# Helper: per-row mean and 16/84 percentiles of a (n_pred × n_draws)
# matrix, dropping any non-finite columns row-by-row.
function _quantile_band(M::AbstractMatrix{<:Real})
    n_pred = size(M, 1)
    μ  = Vector{Float64}(undef, n_pred)
    lo = Vector{Float64}(undef, n_pred)
    hi = Vector{Float64}(undef, n_pred)
    @inbounds for j in 1:n_pred
        row = view(M, j, :)
        finite = filter(isfinite, row)
        if isempty(finite)
            μ[j]  = NaN; lo[j] = NaN; hi[j] = NaN
        else
            μ[j]  = mean(finite)
            lo[j] = quantile(finite, 0.16)
            hi[j] = quantile(finite, 0.84)
        end
    end
    return μ, lo, hi
end

"""
    activity_gp_decompose_rv(chains, params, data; n_draws=100,
                               rng=default_rng()) -> NamedTuple

Inferred RV activity contribution `Vc·G(t) + Vr·dG/dt(t)` at the RV
observation times, plus the activity-corrected RV residual
`rv - <rv_activity>`. Uses the same draw indices as
[`activity_gp_predict`](@ref) so each draw's coefficient
multiplications line up consistently with the latent posterior.

Returns:
- `t_rv` — observation times.
- `rv_data` — original RV values.
- `rv_activity_samples::Matrix{Float64}` of size `(n_rv, n_draws)`.
- `rv_activity_mean, rv_activity_lo, rv_activity_hi` — across-draw mean and 16/84 band.
- `rv_corrected::Vector{Float64}` — `rv - rv_activity_mean`.
- `rv_corrected_err::Vector{Float64}` — quadrature sum of measurement
  uncertainty and the per-point activity uncertainty (half-width of
  the 16/84 band).
"""
function activity_gp_decompose_rv(chains, params::Params, data::Data;
                                    n_draws::Int = 100,
                                    rng::AbstractRNG = default_rng())
    pred = activity_gp_predict(chains, params, data;
                                 t_pred = data.t_rv,
                                 n_draws = n_draws, rng = rng)
    n_rv_obs = length(data.t_rv)
    n_draws_eff = size(pred.G_samples, 2)
    rv_activity = Matrix{Float64}(undef, n_rv_obs, n_draws_eff)
    @inbounds for d in 1:n_draws_eff
        Vc_d = pred.Vc_samples[d]
        Vr_d = pred.Vr_samples[d]
        for i in 1:n_rv_obs
            rv_activity[i, d] = Vc_d * pred.G_samples[i, d] +
                                  Vr_d * pred.dG_dt_samples[i, d]
        end
    end
    μ, lo, hi = _quantile_band(rv_activity)
    rv_corrected = Vector{Float64}(undef, n_rv_obs)
    rv_corrected_err = Vector{Float64}(undef, n_rv_obs)
    @inbounds for i in 1:n_rv_obs
        rv_corrected[i] = data.rv[i] - μ[i]
        # Symmetric half-width of the activity band.
        act_sigma = max(hi[i] - μ[i], μ[i] - lo[i])
        rv_corrected_err[i] = sqrt(data.rv_err[i]^2 + act_sigma^2)
    end
    return (; t_rv = collect(Float64, data.t_rv),
              rv_inst = collect(Int, data.rv_inst),
              rv_data = collect(Float64, data.rv),
              rv_activity_samples = rv_activity,
              rv_activity_mean = μ, rv_activity_lo = lo, rv_activity_hi = hi,
              rv_corrected, rv_corrected_err)
end

"""
    activity_gp_joint_logpdf_lowrank(epochs, chan_a, chan_b, amp, P, λe, λp,
                                      y_flat, σ²_flat) -> logpdf

Marginal log-likelihood of the JOINT Rajpaul model exploiting the
latent-variable LOW-RANK structure. The C channels all observe linear
combinations `a_c·G(t) + b_c·Ġ(t)` of the SAME 2N-dimensional latent
`g = [G(t₁..N); Ġ(t₁..N)]`, so

    Σ = M·K_g·Mᵀ + D ,   M ∈ ℝ^{CN×2N},  K_g ∈ ℝ^{2N×2N},  D diagonal.

Woodbury + the matrix-determinant lemma reduce the dominant factorization
from the dense (C·N)² to (2N)² — for the 5-channel HD 18599 AGP that is
228² vs 570², ~15× fewer flops. `Mᵀ D⁻¹ M` and `Mᵀ D⁻¹ y` are 2×2-block
diagonals (built in O(CN)) because M is `a_c·δ`/`b_c·δ` structured.

`y_flat`, `σ²_flat` are the channel-stacked residuals and per-point
variances (RV block first, then each indicator), length C·N. Numerically
identical to the dense builder to machine precision (validated). Returns
`-Inf` if either Cholesky fails.
"""
function activity_gp_joint_logpdf_lowrank(epochs::AbstractVector{<:Real},
        chan_a::AbstractVector{T}, chan_b::AbstractVector{T},
        amp::Real, P::Real, λe::Real, λp::Real,
        y_flat::AbstractVector{T}, σ²_flat::AbstractVector{T}) where {T<:Real}
    N = length(epochs)
    C = length(chan_a)
    twoN = 2N

    # --- K_g: 2N×2N joint [G; Ġ] covariance from the 4 kernel blocks ---
    Kg = Matrix{T}(undef, twoN, twoN)
    @inbounds for j in 1:N, i in 1:j
        τ = epochs[i] - epochs[j]
        kGG, kGdG, kdGG, kdGdG = activity_kernel_blocks(τ, amp, P, λe, λp)
        Kg[i, j]         = kGG;   Kg[j, i]         = kGG
        Kg[i, N+j]       = kGdG;  Kg[N+j, i]       = kGdG    # cov(G_i, Ġ_j)
        Kg[N+i, j]       = kdGG;  Kg[j, N+i]       = kdGG    # cov(Ġ_i, G_j)
        Kg[N+i, N+j]     = kdGdG; Kg[N+j, N+i]     = kdGdG
    end
    # symmetric jitter for conditioning of the latent covariance
    jit = T(1e-10) * (one(T) + maximum(abs, @view Kg[1:N, 1:N]))
    @inbounds for d in 1:twoN; Kg[d, d] += jit; end

    L_g = cholesky!(Symmetric(Kg); check = false)
    issuccess(L_g) || return convert(T, -Inf)
    Lg = L_g.L                                          # lower factor (2N×2N)

    # --- B = Mᵀ D⁻¹ M is 2×2-block diagonal: per epoch j only the three
    # coefficients (bGG, bGĠ, bĠĠ). v = Mᵀ D⁻¹ y. ---------------------
    # WHITENED Woodbury avoids forming K_g⁻¹:
    #   logdetΣ = logdetD + logdet(I + Lgᵀ B Lg)
    #   quad    = yᵀD⁻¹y − uᵀ(I + Lgᵀ B Lg)⁻¹u,   u = Lgᵀ v
    bGG = zeros(T, N); bGdG = zeros(T, N); bdGdG = zeros(T, N)
    v = zeros(T, twoN)
    yDy = zero(T); logdetD = zero(T)
    @inbounds for c in 1:C
        ac = chan_a[c]; bc = chan_b[c]; off = (c - 1) * N
        for j in 1:N
            d = σ²_flat[off + j]; inv_d = inv(d)
            yj = y_flat[off + j]; wj = yj * inv_d
            yDy += yj * wj; logdetD += log(d)
            v[j] += ac * wj; v[N+j] += bc * wj
            bGG[j]   += ac * ac * inv_d
            bGdG[j]  += ac * bc * inv_d
            bdGdG[j] += bc * bc * inv_d
        end
    end

    # BLg = B·Lg directly from the block structure (4N² vs a 2N³ gemm on a
    # 99%-zero B): rows j and N+j are linear combos of Lg rows j, N+j.
    Lgf = Lg                      # LowerTriangular wrapper over the factor
    BLg = Matrix{T}(undef, twoN, twoN)
    @inbounds for k in 1:twoN
        for j in 1:N
            gjk = Lgf[j, k]; djk = Lgf[N+j, k]
            BLg[j,   k] = bGG[j]  * gjk + bGdG[j]  * djk
            BLg[N+j, k] = bGdG[j] * gjk + bdGdG[j] * djk
        end
    end
    # W = Lgᵀ·BLg — triangular×dense (trmm), not a full gemm.
    W = transpose(Lgf) * BLg
    @inbounds for d in 1:twoN; W[d, d] += one(T); end
    F_C = cholesky!(Symmetric(W); check = false)
    issuccess(F_C) || return convert(T, -Inf)
    logdet_C = 2 * sum(log, @view F_C.factors[diagind(F_C.factors)])

    u = transpose(Lgf) * v
    quad = yDy - dot(u, F_C \ u)
    logdetΣ = logdetD + logdet_C
    return convert(T, -0.5 * (quad + logdetΣ + (C * N) * log(2π)))
end
