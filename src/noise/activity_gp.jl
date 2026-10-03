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
                     dot, logdet, diagind, I, UpperTriangular
import ForwardDiff

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

    tdc = _td_cols(chains_flat, params)       # extracted once, not per draw
    for (d, idx) in enumerate(idx_pool)
        theta = _theta_from_row(chains_flat, idx, params; tdc = tdc)

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

# Where an ActivityGP's inputs live: parameter indices in the layout, 0 where
# there is no parameter (no amplitude in legacy layouts, no RV coupling in
# indicators_only mode, no derivative coupling, no per-channel jitter), and the
# data of each non-RV channel in `agp.channels` order. Resolving the names
# builds strings, so the sampler path caches this in its AGPWorkspace.
struct _AGPIndex
    amp::Int
    P::Int
    λe::Int
    λp::Int
    Vc::Int
    Vr::Int
    vals::Vector{Vector{Float64}}   # indicator values, parallel to the RVs
    errs::Vector{Vector{Float64}}   # their 1σ uncertainties
    a::Vector{Int}                  # coupling to G
    b::Vector{Int}                  # coupling to Ġ (0: none)
    jit::Vector{Int}                # per-channel jitter (0: none)
end

function _agp_index(name_to_idx::Dict{String, Int}, data::Data, agp::ActivityGP)
    s = _gp_suffix(agp)
    amp = get(name_to_idx, "gp_act_amp$s", 0)
    P   = name_to_idx["gp_act_period$s"]
    λe  = name_to_idx["gp_act_lambda_e$s"]
    λp  = name_to_idx["gp_act_lambda_p$s"]
    Vc  = agp.indicators_only ? 0 : name_to_idx["Vc$s"]
    Vr  = (agp.use_derivative && !agp.indicators_only) ? name_to_idx["Vr$s"] : 0
    n_rv_obs = length(data.rv)
    vals_c = Vector{Float64}[]; errs_c = Vector{Float64}[]
    a = Int[]; b = Int[]; jit = Int[]
    for ch in agp.channels
        ch === :rv && continue
        name = String(ch)
        haskey(data.indicators, name) || throw(ArgumentError(
            "ActivityGP requires data.indicators[\"$name\"] for channel :$ch"))
        haskey(data.indicator_errs, name) || throw(ArgumentError(
            "ActivityGP requires data.indicator_errs[\"$name\"] for channel :$ch"))
        vals = data.indicators[name]
        errs = data.indicator_errs[name]
        # Data-model invariant: indicators are parallel to the RV data (one
        # value per RV epoch). The joint path places indicator k at epoch
        # data.t_rv[k]; only valid when lengths match. Fail clearly rather
        # than OOB (n_ind > n_rv) or silently-wrong epochs (n_ind < n_rv).
        length(vals) == n_rv_obs && length(errs) == n_rv_obs ||
            throw(ArgumentError(
                "ActivityGP indicator :$ch has $(length(vals)) values (errs " *
                "$(length(errs))) but there are $n_rv_obs RV observations — " *
                "indicators must be parallel to the RV data (one value per " *
                "RV epoch)."))
        cg, cd = _ACTIVITY_GP_COEFFS[ch]
        push!(a, name_to_idx[string(cg, s)])
        push!(b, (cd === nothing || !agp.use_derivative) ? 0 :
                 name_to_idx[string(cd, s)])
        push!(jit, get(name_to_idx, "gp_act_jit_$(ch)$s", 0))
        push!(vals_c, vals); push!(errs_c, errs)
    end
    return _AGPIndex(amp, P, λe, λp, Vc, Vr, vals_c, errs_c, a, b, jit)
end

# What an `_AGPIndex` was resolved from, compared by identity.
const _AGPIndexKey = Tuple{ActivityGP, Dict{String, Int}, Vector{Float64},
                           Dict{String, Vector{Float64}},
                           Dict{String, Vector{Float64}}}

"""
    AGPWorkspace()

Reusable buffers for the ActivityGP joint likelihood, held by each
`PTWorkspace` (one per sampler slot) so the per-call matrices of
[`activity_gp_joint_logpdf_lowrank!`](@ref) are allocated once. Empty until an
ActivityGP is evaluated, and resized when the number of epochs `N` changes.
Also caches the resolved parameter indices of each ActivityGP it has seen.
"""
mutable struct AGPWorkspace
    # Parameter indices per (ActivityGP, layout, data), see `_agp_index!`.
    index_keys::Vector{_AGPIndexKey}
    index::Vector{_AGPIndex}
    # Channel coefficients and jitter² (length C), channel-stacked
    # residuals and variances (length C·N).
    chan_a::Vector{Float64}
    chan_b::Vector{Float64}
    jit2::Vector{Float64}
    y_flat::Vector{Float64}
    σ²_flat::Vector{Float64}
    # Low-rank solver buffers for N epochs.
    N::Int
    A::Matrix{Float64}       # 2N×2N I + R·K_g·Rᵀ, factored in place
    bGG::Vector{Float64}     # N: the 2×2 blocks of B = Mᵀ D⁻¹ M
    bGd::Vector{Float64}
    bdd::Vector{Float64}
    r11::Vector{Float64}     # N: their factors B_j = R_jᵀ R_j
    r12::Vector{Float64}
    r22::Vector{Float64}
    v::Vector{Float64}       # 2N: Mᵀ D⁻¹ y
    w::Vector{Float64}       # 2N: R⁻ᵀ v
    t::Vector{Float64}       # 2N: U⁻ᵀ w, with A = UᵀU
end

function AGPWorkspace()
    m() = Matrix{Float64}(undef, 0, 0)
    z() = Float64[]
    return AGPWorkspace(_AGPIndexKey[], _AGPIndex[], z(), z(), z(), z(), z(),
                        0, m(), z(), z(), z(), z(), z(), z(), z(), z(), z())
end

# The `_AGPIndex` of `agp` under this layout and data: resolved by name on
# first use, then looked up by identity.
function _agp_index!(w::AGPWorkspace, name_to_idx::Dict{String, Int}, data::Data,
                     agp::ActivityGP)
    @inbounds for k in eachindex(w.index_keys)
        key = w.index_keys[k]
        (key[1] === agp && key[2] === name_to_idx && key[3] === data.rv &&
         key[4] === data.indicators && key[5] === data.indicator_errs) &&
            return w.index[k]
    end
    ix = _agp_index(name_to_idx, data, agp)
    # A workspace serves one run, so a handful of entries is the most it
    # needs; start over rather than grow if it is handed many datasets.
    if length(w.index_keys) >= 8
        empty!(w.index_keys); empty!(w.index)
    end
    push!(w.index_keys, (agp, name_to_idx, data.rv, data.indicators,
                         data.indicator_errs))
    push!(w.index, ix)
    return ix
end

# Coefficient buffers (length C) and channel-stacked buffers (length n).
function _agp_coef_buffers!(w::AGPWorkspace, C::Int)
    resize!(w.chan_a, C); resize!(w.chan_b, C); resize!(w.jit2, C)
    return w.chan_a, w.chan_b, w.jit2
end
function _agp_flat_buffers!(w::AGPWorkspace, n::Int)
    resize!(w.y_flat, n); resize!(w.σ²_flat, n)
    return w.y_flat, w.σ²_flat
end

function _agp_resize!(w::AGPWorkspace, N::Int)
    w.N == N && return w
    n2 = 2N
    w.A = Matrix{Float64}(undef, n2, n2)
    for f in (:bGG, :bGd, :bdd, :r11, :r12, :r22)
        setfield!(w, f, Vector{Float64}(undef, N))
    end
    for f in (:v, :w, :t)
        setfield!(w, f, Vector{Float64}(undef, n2))
    end
    w.N = N
    return w
end

"""
    activity_gp_joint_logpdf_lowrank(epochs, chan_a, chan_b, amp, P, λe, λp,
                                      y_flat, σ²_flat) -> logpdf

Marginal log-likelihood of the JOINT Rajpaul model through its latent
low-rank structure. The C channels all observe linear combinations
`a_c·G(t) + b_c·Ġ(t)` of the SAME 2N-dimensional latent
`g = [G(t₁..N); Ġ(t₁..N)]`, so

    Σ = M·K_g·Mᵀ + D ,   M ∈ ℝ^{CN×2N},  K_g ∈ ℝ^{2N×2N},  D diagonal.

`B = Mᵀ D⁻¹ M` is block diagonal, one 2×2 block per epoch, and each block is
factored exactly as `B_j = R_jᵀ R_j` (R_j upper triangular). With
`A = I + R·K_g·Rᵀ` and `w = R⁻ᵀ·Mᵀ D⁻¹ y`, Woodbury and the
matrix-determinant lemma give

    log det Σ = log det D + log det A
    yᵀ Σ⁻¹ y  = yᵀ D⁻¹ y − wᵀw + wᵀ A⁻¹ w

A is built directly from the kernel and is the only matrix factored: one
(2N)² Cholesky instead of the dense (C·N)² (228² against 570² for the
five-channel HD 18599 fit). K_g itself is never factored, so it needs no
jitter and may be singular. A rank-deficient block (no Ġ coupling, as with
`use_derivative = false`, or no G coupling at an epoch) gets a zero row in R,
and the identities above still hold exactly.

The result is therefore the exact Gaussian log-density of Σ up to
floating-point rounding. That rounding error grows with the conditioning of A
(the signal-to-noise ratio and coherence length of the GP) as it does for a
Float64 Cholesky of the dense Σ. Against the dense likelihood evaluated in
BigFloat, on the five-channel HD 18599 fit: at most 4e-12 nats near the
posterior; on prior draws a median of 2e-9 and at most 2e-6 nats, the same
as the Float64 dense Cholesky. test/test_activity_gp_lowrank.jl checks it.

Dual numbers (ForwardDiff) take the same route, with two exceptions. R_j is
a square root of B_j, so it is not differentiable where B_j is singular
(every G coupling 0, every Ġ coupling 0, or all channels' (a_c, b_c)
parallel), and its derivative loses accuracy as B_j approaches that. And
w = R⁻ᵀ·Mᵀ D⁻¹ y divides by R, so when every coupling is tiny against the
noise its derivatives cancel to a relative error of about eps/SNR. When a
block's smaller eigenvalue is below 1e-10 of its larger, or the GP's
signal-to-noise is below 1e-6 at every epoch, dual inputs are evaluated with
the dense (C·N)² Cholesky instead, which is smooth there. A coupling that is
a constant 0 (every Ġ coupling under `use_derivative = false`) is not a
direction of the block and does not count. On the HD 18599 job this happens
at couplings that are exactly 0 (the prior box centre), at none of ~250
prior draws and 150 near-posterior points with five channels, and at 2 of
236 prior draws with RV and logR'HK alone. `Float64` values are exact at
singular blocks and always take the low-rank path.

`y_flat`, `σ²_flat` are the channel-stacked residuals and per-point
variances (RV block first, then each indicator), length C·N. A NaN among
the couplings, variances or residuals gives `NaN`. `-Inf` is returned if the
Cholesky of A fails, which A ⪰ I rules out for finite inputs.
"""
function activity_gp_joint_logpdf_lowrank(epochs::AbstractVector{<:Real},
        chan_a::AbstractVector{<:Real}, chan_b::AbstractVector{<:Real},
        amp::Real, P::Real, λe::Real, λp::Real,
        y_flat::AbstractVector{<:Real}, σ²_flat::AbstractVector{<:Real})
    N = length(epochs); twoN = 2N
    T  = promote_type(eltype(chan_a), eltype(chan_b), eltype(y_flat), eltype(σ²_flat))
    TA = promote_type(T, typeof(amp), typeof(P), typeof(λe), typeof(λp))
    vN() = Vector{T}(undef, N)
    return _agp_whitened_core!(Matrix{TA}(undef, twoN, twoN),
                               vN(), vN(), vN(), vN(), vN(), vN(),
                               Vector{T}(undef, twoN), Vector{T}(undef, twoN),
                               Vector{TA}(undef, twoN),
                               epochs, chan_a, chan_b, amp, P, λe, λp, y_flat, σ²_flat)
end

"""
    activity_gp_joint_logpdf_lowrank!(w::AGPWorkspace, epochs, chan_a, chan_b,
                                       amp, P, λe, λp, y_flat, σ²_flat) -> logpdf

[`activity_gp_joint_logpdf_lowrank`](@ref) with its matrix and vectors taken
from `w` instead of allocated, for the per-slot workspace of the samplers. Same
operations, same result bit for bit. Inputs that are not `Float64` (dual
numbers) take the allocating method.
"""
function activity_gp_joint_logpdf_lowrank!(w::AGPWorkspace, epochs::AbstractVector{<:Real},
        chan_a::AbstractVector{Float64}, chan_b::AbstractVector{Float64},
        amp::Float64, P::Float64, λe::Float64, λp::Float64,
        y_flat::AbstractVector{Float64}, σ²_flat::AbstractVector{Float64})
    _agp_resize!(w, length(epochs))
    return _agp_whitened_core!(w.A, w.bGG, w.bGd, w.bdd, w.r11, w.r12, w.r22,
                               w.v, w.w, w.t,
                               epochs, chan_a, chan_b, amp, P, λe, λp, y_flat, σ²_flat)
end
activity_gp_joint_logpdf_lowrank!(::AGPWorkspace, epochs, chan_a, chan_b,
                                   amp, P, λe, λp, y_flat, σ²_flat) =
    activity_gp_joint_logpdf_lowrank(epochs, chan_a, chan_b, amp, P, λe, λp,
                                     y_flat, σ²_flat)

# Constants of the quasi-periodic kernel blocks, hoisted out of the pair loop.
@inline function _qp_consts(amp, P, λe, λp)
    amp2     = amp * amp
    inv_λe2  = 1 / (λe * λe)
    inv_2λp2 = 1 / (2 * λp * λp)
    π_P      = π / P
    c_fp     = π_P * inv_2λp2              # π/(2Pλp²)
    c_fpp    = π_P * π_P * 2 * inv_2λp2    # π²/(P²λp²)
    return (amp2, 0.5 * inv_λe2, inv_λe2, inv_2λp2, π_P, c_fp, c_fpp)
end

# The four blocks of `activity_kernel_blocks` from one sincos(πτ/P), using
# sin(2x) = 2 sin x cos x and cos(2x) = 1 − 2 sin²x. Same closed forms,
# different rounding.
@inline function _qp_blocks_sincos(τ, c)
    amp2, half_inv_λe2, inv_λe2, inv_2λp2, π_P, c_fp, c_fpp = c
    s, co = sincos(π_P * τ)
    s2  = 2 * s * co
    c2  = 1 - 2 * s * s
    f   = -τ * τ * half_inv_λe2 - s * s * inv_2λp2
    fp  = -τ * inv_λe2 - c_fp * s2
    fpp = -inv_λe2 - c_fpp * c2
    k   = amp2 * exp(f)
    return (k, -fp * k, fp * k, -(fpp + fp * fp) * k)
end

# Zero in its value and in every partial derivative, at any nesting of
# ForwardDiff duals (ForwardDiff 0.10's `iszero` looks at the value only).
_agp_strictly_zero(x::Real) = iszero(x)
_agp_strictly_zero(x::ForwardDiff.Dual) =
    _agp_strictly_zero(ForwardDiff.value(x)) &&
    all(_agp_strictly_zero, ForwardDiff.partials(x))

# Whether derivatives taken through the factors R_j of B are accurate here.
# Two things break them. (1) R_j is a square root of B_j: not differentiable
# where the block is singular, inaccurate near it. Each block's smaller
# eigenvalue must be at least 1e-10 of its larger (det B_j ≥ 1e-10·(tr B_j)²,
# as λ_min/λ_max ≈ det/tr² when small). A direction with no coupling at all
# (every a_c, or every b_c, a constant 0) is not part of the blocks, which are
# then 1×1 and need only be nonzero. (2) w = R⁻ᵀv divides by R, so its
# derivatives grow as 1/R while the GP's share of log L shrinks as R²; their
# cancellation leaves a relative error of about eps/SNR. The largest
# per-epoch signal-to-noise tr(R_j K_jj R_jᵀ) = k_GG(0)·bGG + k_ĠĠ(0)·bĠĠ
# must be at least 1e-6. False for NaN.
function _agp_blocks_smooth(bGG::AbstractVector, bGd::AbstractVector,
                            bdd::AbstractVector, chan_a::AbstractVector,
                            chan_b::AbstractVector, k0GG::Real, k0dd::Real)
    noG = all(_agp_strictly_zero, chan_a)
    noD = all(_agp_strictly_zero, chan_b)
    noG && noD && return true
    snr = zero(promote_type(eltype(bGG), eltype(bdd), typeof(k0GG), typeof(k0dd)))
    @inbounds for j in eachindex(bGG)
        g = bGG[j]; e = bdd[j]
        if noD
            g > 0 || return false
        elseif noG
            e > 0 || return false
        else
            (g > 0 && e > 0) || return false
            h = bGd[j]
            # g·q, with q = e − h²/g the Schur complement the solver factors.
            g * (e - h * h / g) >= 1e-10 * (g + e)^2 || return false
        end
        snr = max(snr, k0GG * g + k0dd * e)
    end
    return snr >= 1e-6
end

# The dense (C·N)² Gaussian log-density of the same Σ = M·K_g·Mᵀ + D,
# generic in the element type: the dual-number route at near-singular blocks.
function _agp_dense_logpdf(epochs, chan_a, chan_b, amp, P, λe, λp, y_flat, σ²_flat)
    Σ0 = activity_gp_covariance_blocked(epochs, chan_a, chan_b, amp, P, λe, λp)
    TS = promote_type(eltype(Σ0), eltype(y_flat), eltype(σ²_flat))
    Σ  = eltype(Σ0) === TS ? Σ0 : convert(Matrix{TS}, Σ0)
    @inbounds for i in eachindex(σ²_flat); Σ[i, i] += σ²_flat[i]; end
    F = cholesky!(Symmetric(Σ, :U); check = false)
    issuccess(F) || return convert(TS, -Inf)
    z = F.U' \ y_flat
    return -(dot(z, z) + logdet(F) + length(y_flat) * log(2π)) / 2
end

# The whitened low-rank solve on caller-supplied buffers: `A` is 2N×2N,
# `bGG`…`r22` length N, `v`, `w`, `t` length 2N. Every element read is
# written first (only the upper triangle of A is used), so nothing needs
# clearing.
function _agp_whitened_core!(A::AbstractMatrix,
        bGG::AbstractVector{T}, bGd::AbstractVector{T}, bdd::AbstractVector{T},
        r11::AbstractVector{T}, r12::AbstractVector{T}, r22::AbstractVector{T},
        v::AbstractVector{T}, w::AbstractVector{T}, t::AbstractVector,
        epochs::AbstractVector{<:Real},
        chan_a::AbstractVector{<:Real}, chan_b::AbstractVector{<:Real},
        amp::Real, P::Real, λe::Real, λp::Real,
        y_flat::AbstractVector{<:Real}, σ²_flat::AbstractVector{<:Real}) where {T<:Real}
    N = length(epochs)
    C = length(chan_a)
    twoN = 2N
    TA = eltype(A)

    # B = Mᵀ D⁻¹ M per epoch (bGG, bGĠ, bĠĠ) and v = Mᵀ D⁻¹ y.
    fill!(bGG, zero(T)); fill!(bGd, zero(T)); fill!(bdd, zero(T))
    fill!(v, zero(T))
    yDy = zero(T); logdetD = zero(T)
    @inbounds for c in 1:C
        ac = chan_a[c]; bc = chan_b[c]; off = (c - 1) * N
        for j in 1:N
            d = σ²_flat[off + j]; inv_d = inv(d)
            yj = y_flat[off + j]; wj = yj * inv_d
            yDy += yj * wj; logdetD += log(d)
            v[j] += ac * wj; v[N+j] += bc * wj
            bGG[j] += ac * ac * inv_d
            bGd[j] += ac * bc * inv_d
            bdd[j] += bc * bc * inv_d
        end
    end

    # B_j = R_jᵀ R_j with R_j = [r11 r12; 0 r22], and w = R⁻ᵀ v. A block with
    # no G information (bGG = 0, hence bGĠ = 0) factors as diag(0, √bĠĠ); a
    # rank-1 block (no Ġ information independent of G) gets r22 = 0. A zero
    # row of R leaves that row and column of A at the identity and w at 0
    # there, so both cases are exact rather than approximated. Only a value
    # that is zero (or rounded below it) means "no information": a NaN takes
    # the square root and reaches w, so it is not mistaken for one.
    ww = zero(T)
    @inbounds for j in 1:N
        g = bGG[j]
        if g <= 0
            x11 = zero(T); x12 = zero(T); q = bdd[j]; wG = zero(T)
        else
            x11 = sqrt(g); x12 = bGd[j] / x11
            q   = bdd[j] - x12 * x12
            wG  = v[j] / x11
        end
        if q <= 0
            x22 = zero(T); wd = zero(T)
        else
            x22 = sqrt(q); wd = (v[N+j] - x12 * wG) / x22
        end
        r11[j] = x11; r12[j] = x12; r22[j] = x22
        w[j] = wG; w[N+j] = wd
        ww += wG * wG + wd * wd
    end
    # A NaN coupling, variance or residual gives NaN, as the dense likelihood
    # does, rather than whatever the Cholesky of a NaN matrix returns.
    isnan(ww) && return convert(TA, NaN)

    kc = _qp_consts(amp, P, λe, λp)

    # Dual numbers: where derivatives through R lose their accuracy (a
    # singular or nearly singular block of B, or a GP too faint against the
    # noise; see _agp_blocks_smooth) the dense likelihood is smooth and is
    # used instead. Compiled out for Float64.
    if T <: ForwardDiff.Dual
        k0GG, _, _, k0dd = _qp_blocks_sincos(zero(eltype(epochs)), kc)
        _agp_blocks_smooth(bGG, bGd, bdd, chan_a, chan_b, k0GG, k0dd) ||
            return convert(TA, _agp_dense_logpdf(epochs, chan_a, chan_b,
                                                 amp, P, λe, λp, y_flat, σ²_flat))
    end

    # Upper triangle of A = I + R·K_g·Rᵀ, straight from the kernel blocks at
    # τ = t_j − t_k (j ≤ k). Rows j and N+j of R·g are r11·G_j + r12·Ġ_j and
    # r22·Ġ_j; cov(G_j, Ġ_k) = cGd and cov(G_k, Ġ_j) = cov(Ġ_j, G_k) = cdG.
    @inbounds for k in 1:N
        r11k = r11[k]; r12k = r12[k]; r22k = r22[k]; tk = epochs[k]
        for j in 1:k
            cGG, cGd, cdG, cdd = _qp_blocks_sincos(epochs[j] - tk, kc)
            r11j = r11[j]; r12j = r12[j]; r22j = r22[j]
            A[j, k]         = r11j * (r11k * cGG + r12k * cGd) +
                              r12j * (r11k * cdG + r12k * cdd)
            A[j, N + k]     = r22k * (r11j * cGd + r12j * cdd)
            A[N + j, N + k] = r22j * r22k * cdd
            j == k || (A[k, N + j] = r22j * (r11k * cdG + r12k * cdd))
        end
    end
    @inbounds for d in 1:twoN; A[d, d] += one(TA); end

    # A ⪰ I, so the factorization succeeds for any finite input.
    F = cholesky!(Symmetric(A, :U); check = false)
    issuccess(F) || return convert(TA, -Inf)
    logdetA = zero(TA)
    @inbounds for d in 1:twoN; logdetA += log(A[d, d]); end
    logdetA *= 2
    copyto!(t, w)
    ldiv!(transpose(UpperTriangular(A)), t)       # t = U⁻ᵀ w, so wᵀA⁻¹w = tᵀt
    quad = yDy - ww + dot(t, t)
    return -(quad + logdetD + logdetA + (C * N) * log(2π)) / 2
end
