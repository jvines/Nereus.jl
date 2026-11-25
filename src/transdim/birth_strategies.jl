# Informed birth strategy for trans-dim proposals.
#
# InformedBirth: Lomb-Scargle periodogram of residuals → propose period
# from high-power peaks. Mixture proposal (α informed + (1-α) uniform)
# ensures ergodicity. Hastings ratio correction for non-uniform proposal.

using Random
using Distributions
using Statistics: median

# =====================================================================
# Lomb-Scargle periodogram (minimal, no external dependency)
# =====================================================================

"""
    _lomb_scargle(t, y, freqs; σ=nothing) -> power::Vector{Float64}

Generalized Lomb-Scargle periodogram (Zechmeister & Kürster 2009 §2.1).
With per-point uncertainties `σ`, computes the proper inverse-variance-
weighted GLS — weighted mean, weighted sums everywhere. Without σ
(default), reduces to the unit-weight LS. The weighted form is
essential for multi-instrument data: e.g. ESPRESSO at σ≈0.7 m/s gets
~50× more leverage than HIRES at σ≈5 m/s, which makes low-K planets
much easier to detect in the residual periodogram.
"""
function _lomb_scargle(t::Vector{Float64}, y::Vector{Float64},
                        freqs::Vector{Float64};
                        σ::Union{Vector{Float64}, Nothing} = nothing)
    n = length(t)
    # Inverse-variance weights, normalized to sum to 1 (Z&K eq 6).
    w = if σ === nothing
        fill(1.0 / n, n)
    else
        wraw = 1.0 ./ (σ .^ 2)
        wraw ./ sum(wraw)
    end
    y_mean = sum(w .* y)
    r = y .- y_mean
    var_y = sum(w .* r .^ 2)         # weighted variance (Z&K eq 10)
    var_y ≤ 0 && return zeros(length(freqs))

    power = Vector{Float64}(undef, length(freqs))

    @inbounds for (fi, f) in enumerate(freqs)
        omega = 2π * f
        # Weighted τ: balances cos and sin sums under weights (Z&K eq 19).
        s2 = 0.0; c2 = 0.0
        for i in 1:n
            wt = 2 * omega * t[i]
            s2 += w[i] * sin(wt)
            c2 += w[i] * cos(wt)
        end
        tau = atan(s2, c2) / (2 * omega)

        # Weighted dot products for the periodogram statistic.
        cc = 0.0; ss = 0.0; c2s = 0.0; s2s = 0.0
        for i in 1:n
            wt = omega * (t[i] - tau)
            c = cos(wt); s = sin(wt)
            cc += w[i] * r[i] * c
            ss += w[i] * r[i] * s
            c2s += w[i] * c * c
            s2s += w[i] * s * s
        end
        power[fi] = 0.5 * (cc^2 / max(c2s, 1e-30) + ss^2 / max(s2s, 1e-30)) / var_y
    end
    return power
end

# Tuning constants
const INFORMED_ALPHA = 0.7
const INFORMED_N_PEAKS = 8
const INFORMED_SIGMA_P = 0.05    # width in log-period space (RV peaks)
const INFORMED_SIGMA_P_FLOOR = 1e-5

# Fractional tolerance for harmonic/family dedup of proposal peaks. True BLS
# harmonics of an active/listed planet land at EXACT small-integer ratios
# (within the period-grid resolution, ≲0.1% — WASP-47 e's 2× family member
# measured 1.9988×). Resonant-chain NEIGHBOURS sit systematically off the
# exact ratio (K2-138: 0.9–3.0% from 3:2; g at 1.6% from 5:1 of e). The old
# 3% tolerance could not tell them apart and silently removed REAL chain
# members from the proposal list (K2-138 blind run: b, d, g all deduped once
# c+e locked → slots filled with junk). 0.5% keeps true harmonics dead and
# chain members alive.
const INFORMED_DEDUP_TOL = 0.005  # tightest transit-peak σ (~ refine res)
const INFORMED_SIGMA_K = 0.3     # width in log-K space for amplitude proposal
const INFORMED_SIGMA_E = 0.15    # width for sesinw/secosw birth (bias toward e≈0)
const INFORMED_SIGMA_MO = 0.3    # width for Mo proposal around LS phase estimate
const INFORMED_OVERSAMPLE = 5    # Lomb-Scargle oversampling factor
# Recompute the informed-birth periodogram/BLS peaks every N birth attempts
# (per thread). The peaks are data-driven and barely shift run-to-run, so a
# stale cache only costs the occasional re-proposal of an already-found period
# (rejected), never a missed planet — so we cache for many attempts and pay the
# (decimated) BLS fold rarely. 500 vs the old 100 ≈ 5× fewer recomputes.
const INFORMED_CACHE_INTERVAL = 500
# Cap on cadences fed to the birth-proposal BLS. The BLS fold is O(n_period ×
# n_obs) and dominates at high cadence (155k+ on K2+TESS). Since this BLS only
# proposes a candidate PERIOD (the MCMC refines amplitude/epoch), a strided
# subsample preserves period coverage — the transit signal stacks in phase — at
# a fraction of the cost. find_transits' BLS is unaffected (uses the full LC).
const MAX_BLS_OBS = 20_000

# =====================================================================
# Periodogram cache — avoid recomputing on every birth attempt.
# Residuals change slowly between MCMC steps; stale peaks are fine
# since the mixture proposal includes a uniform component for ergodicity.
# =====================================================================

mutable struct PeakCache
    periods::Vector{Float64}
    weights::Vector{Float64}
    K_est::Vector{Float64}
    Mo_est::Vector{Float64}
    log_periods::Vector{Float64}
    calls_since_update::Int
end
PeakCache() = PeakCache(Float64[], Float64[], Float64[], Float64[], Float64[], INFORMED_CACHE_INTERVAL)

# Window function cache — computed once per dataset (depends only on t_rv)
mutable struct WindowCache
    freqs::Vector{Float64}
    power::Vector{Float64}
    initialized::Bool
end
WindowCache() = WindowCache(Float64[], Float64[], false)

const _PEAK_CACHES = Dict{UInt64, PeakCache}()
const _WINDOW_CACHES = Dict{UInt64, WindowCache}()
const _PEAK_CACHE_LOCK = ReentrantLock()

function _get_cache(chain_id::UInt64)
    lock(_PEAK_CACHE_LOCK) do
        # Signature-keyed since the state-consistency fix; junk-state churn can
        # mint many keys, so reset when oversized (entries rebuild on demand).
        length(_PEAK_CACHES) > 512 && empty!(_PEAK_CACHES)
        get!(() -> PeakCache(), _PEAK_CACHES, chain_id)
    end
end

function _get_window_cache(data_id::UInt64)
    lock(_PEAK_CACHE_LOCK) do
        # Same guard as _PEAK_CACHES/_BLS_CACHES: one entry per distinct Data
        # id, strong-ref, so a long-lived daemon grows it monotonically —
        # reset when oversized (entries rebuild on demand).
        length(_WINDOW_CACHES) > 512 && empty!(_WINDOW_CACHES)
        get!(() -> WindowCache(), _WINDOW_CACHES, data_id)
    end
end

"""
    _window_function(t, freqs) -> power

Spectral window function: squared modulus of the DFT of the sampling
pattern (Dawson & Fabrycky 2010). Peaks show where the observation
cadence creates aliases (1d, 0.5d, sidereal day, etc.).

W(f) = (1/N²) × [( Σ cos(2πf tⱼ) )² + ( Σ sin(2πf tⱼ) )²]
"""
function _window_function(t::Vector{Float64}, freqs::Vector{Float64})
    n = length(t)
    inv_n2 = 1.0 / (n * n)
    power = Vector{Float64}(undef, length(freqs))
    @inbounds for (fi, f) in enumerate(freqs)
        sc = 0.0; cc = 0.0
        omega = 2π * f
        for i in 1:n
            phase = omega * t[i]
            sc += sin(phase)
            cc += cos(phase)
        end
        power[fi] = (sc * sc + cc * cc) * inv_n2
    end
    return power
end

# =====================================================================
# Residual computation
# =====================================================================

# Per-instrument median cache for γ-substitute. Keyed by `objectid(data)`
# so it's computed once per Data object and reused across many birth
# attempts. Avoids depending on the chain's current γ values, which are
# noisy at NS init and contaminate the LS residual peak-finding.
const _GAMMA_MEDIAN_CACHE = Dict{UInt64, Vector{Float64}}()
const _GAMMA_MEDIAN_LOCK = ReentrantLock()

function _per_instrument_medians(data::Data)
    key = objectid(data)
    lock(_GAMMA_MEDIAN_LOCK) do
        # Bounded like the sibling caches: objectid-keyed strong-ref dict in a
        # warm daemon = one entry per Data forever. Reset when oversized.
        length(_GAMMA_MEDIAN_CACHE) > 512 && empty!(_GAMMA_MEDIAN_CACHE)
        get!(_GAMMA_MEDIAN_CACHE, key) do
            n_inst = isempty(data.rv_inst) ? 1 : maximum(data.rv_inst)
            meds = zeros(Float64, n_inst)
            for i in 1:n_inst
                mask = data.rv_inst .== i
                if any(mask)
                    meds[i] = median(view(data.rv, mask))
                end
            end
            meds
        end
    end
end

"""
    _compute_rv_residuals(theta, data) -> Vector{Float64}

RV residuals (data − model) for currently active planets, using the
**per-instrument median** in place of the chain's γ. The LS peak-finder
that consumes these residuals must be chain-state-independent, otherwise
γ noise at chain init dominates the residuals and Mo_est gets pinned to
spurious peaks. Per-instrument median is the data-conditional best
zero-point estimate that doesn't depend on chain state.
"""
function _compute_rv_residuals(theta::Theta, data::Data)
    n_obs = length(data.t_rv)
    pred = zeros(Float64, n_obs)
    parametrization = theta.params.config.parametrization
    t_ref = data.t_ref

    medians = _per_instrument_medians(data)
    for i in 1:n_obs
        pred[i] = medians[data.rv_inst[i]]
    end

    for k in planet_indices(theta)
        block = theta.params.layout.planet_blocks[k]
        has_K(block) || continue
        P = Float64(planet_P(theta, k))
        # SB2 binary: subtract BOTH channels (K_A on comp 1, −K_B on comp 2)
        # via _comp_rv so the informed-birth residual is whitened of the
        # binary too — else the periodogram is dominated by the binary and
        # births propose at its period. planet_K throws on SB2, so branch.
        KA, cB = if block isa SB2Block
            Float64(planet_K_A(theta, k)), -Float64(planet_K_B(theta, k))
        else
            Float64(planet_K(theta, k)), 0.0
        end
        e, w = planet_e_w(theta, k)
        e = Float64(e); w = Float64(w)
        ta = Float64(planet_time_anchor(theta, k))

        Tp = Float64(anchor_to_tp(parametrization.time, ta, P, e, w, t_ref,
                                  parametrization.t_ref_c))

        for i in 1:n_obs
            M = 2π * (data.t_rv[i] - Tp) / P
            E = kepler_solve(M, e)
            f = true_anomaly(E, e)
            geom = cos(f + w) + e * cos(w)
            pred[i] += _comp_rv(geom, data.rv_comp[i], KA, cB)
        end
    end

    resid = data.rv .- pred
    # Rotation pre-whitening for planet-finding: the informed-birth GLS runs on
    # this residual, but with only planets+zero-points removed the periodogram is
    # DOMINATED by stellar rotation on active stars — so births propose at the
    # rotation period (the GP then rejects them) and the real planets, buried
    # under the activity, are never proposed → blind Np=0. When the model carries
    # a GP rotation-period prior we therefore subtract a least-squares rotation
    # model (fundamental + harmonics) at the best period inside that prior's
    # range BEFORE the periodogram, so sub-rotation planet peaks surface. This is
    # STATE-INDEPENDENT (uses the prior's period range + a data LS fit, not the
    # chain's GP hyperparameters), preserving the peak-finder's design intent. No
    # rotation prior (toy / AD / ARMA-only configs) ⇒ no-op, behaviour unchanged.
    rp = _rotation_period_prior(theta)
    if rp !== nothing
        lo, hi = bounds(rp)
        _prewhiten_rotation!(resid, data.t_rv, Float64(lo), Float64(hi))
    end
    return resid
end

# Prior on the GP rotation period (`gp_period` for CeleriteRotation, `gp_act_period`
# for ActivityGP), or `nothing` if the model has no GP rotation component.
function _rotation_period_prior(theta::Theta)
    layout = theta.params.layout
    for (i, nm) in enumerate(layout.unfrozen_names)
        ln = lowercase(nm)
        (occursin("gp", ln) && occursin("period", ln)) && return layout.unfrozen_priors[i]
    end
    return nothing
end

# Subtract a least-squares rotation model (const + fundamental + `n_harm`
# harmonics) at the best single period in [P_lo, P_hi] from `resid`, in place.
# The period is picked by a coarse Lomb-Scargle-style scan over the prior range,
# so the rotation is removed data-adaptively without using chain state.
function _prewhiten_rotation!(resid::Vector{Float64}, t::Vector{Float64},
                               P_lo::Float64, P_hi::Float64; n_harm::Int = 2)
    n = length(t)
    (n < 8 && return resid)
    (P_hi > P_lo > 0) || return resid
    μ = sum(resid) / n
    r0 = resid .- μ
    bestP = NaN; best = -Inf
    @inbounds for P in range(P_lo, P_hi; length = 80)
        ω = 2π / P
        Sss = 0.0; Scc = 0.0; Ssc = 0.0; Ssy = 0.0; Scy = 0.0
        for i in 1:n
            s = sin(ω * t[i]); c = cos(ω * t[i])
            Sss += s*s; Scc += c*c; Ssc += s*c; Ssy += s*r0[i]; Scy += c*r0[i]
        end
        det = Sss*Scc - Ssc*Ssc
        abs(det) < 1e-12 && continue
        a = (Scc*Ssy - Ssc*Scy) / det
        b = (Sss*Scy - Ssc*Ssy) / det
        pow = a*Ssy + b*Scy           # explained sum of squares at P
        pow > best && (best = pow; bestP = P)
    end
    isfinite(bestP) || return resid
    m = 1 + 2*n_harm
    X = ones(Float64, n, m)
    @inbounds for h in 1:n_harm
        for i in 1:n
            X[i, 2h]   = sin(2π*h * t[i] / bestP)
            X[i, 2h+1] = cos(2π*h * t[i] / bestP)
        end
    end
    β = X \ resid
    resid .-= X * β
    return resid
end

# =====================================================================
# Peak finding
# =====================================================================

"""
    _estimate_K_phase(t, residuals, freq, t_ref) -> (K, Mo_est)

Estimate semi-amplitude K and mean anomaly Mo at a given frequency
by least-squares fit of A*sin(2πft) + B*cos(2πft) to the residuals.
Phase converted to Mo at t_ref assuming circular orbit.
"""
function _estimate_K_phase(t::Vector{Float64}, residuals::Vector{Float64},
                           freq::Float64, t_ref::Float64;
                           σ::Union{Vector{Float64}, Nothing} = nothing)
    n = length(t)
    # Per-point inverse-variance weights (1.0 if no σ supplied).
    sc = 0.0; cc = 0.0; ss = 0.0; sy = 0.0; cy = 0.0
    @inbounds for i in 1:n
        wi = σ === nothing ? 1.0 : 1.0 / (σ[i] * σ[i])
        phase = 2π * freq * t[i]
        s = sin(phase); c = cos(phase)
        sc += wi * s * c
        cc += wi * c * c
        ss += wi * s * s
        sy += wi * s * residuals[i]
        cy += wi * c * residuals[i]
    end
    det = ss * cc - sc * sc
    abs(det) < 1e-30 && return (0.0, 0.0)
    A = (cc * sy - sc * cy) / det
    B = (ss * cy - sc * sy) / det
    K = sqrt(A^2 + B^2)
    # Phase of the sinusoid: y = K*cos(2πft - φ), φ = atan(A, B)
    # Mo at t_ref: Mo = 2πf*t_ref - φ, wrapped to [-π, π]
    φ = atan(A, B)
    Mo = mod(2π * freq * t_ref - φ + π, 2π) - π
    return (K, Mo)
end

function _find_peaks(t::Vector{Float64}, residuals::Vector{Float64},
                     P_min::Float64, P_max::Float64;
                     t_ref::Float64 = isempty(t) ? 0.0 :
                                      (minimum(t) + maximum(t)) / 2,
                     σ::Union{Vector{Float64}, Nothing} = nothing,
                     max_nfreq::Int = typemax(Int))
    # No RVs, no periodogram. An astrometry-only fit has none at all, and
    # `maximum(t)` over the empty vector threw "reducing over an empty
    # collection" from inside a birth proposal — killing the whole run rather
    # than reporting no peaks, which every caller already handles (it is what
    # they do when the periodogram is flat, and what the BLS side does below
    # `n_phot > 50`). Four points is the least a Lomb-Scargle can fit.
    length(t) > 3 || return (Float64[], Float64[], Float64[], Float64[])
    f_min = 1.0 / P_max
    f_max = 1.0 / P_min
    # Adaptive frequency grid: oversample factor × baseline. `max_nfreq` caps it:
    # over a long baseline + wide period range the full grid is tens of thousands
    # of frequencies, and a periodogram run at every call (e.g. the planet↔AD
    # swap, which fires per trans-dim move) then dominates wall-time. For a
    # PROPOSAL the peak only needs to be located, not resolved — the swap caps
    # this; informed births leave it uncapped for needle resolution.
    baseline = maximum(t) - minimum(t)
    df = 1.0 / (INFORMED_OVERSAMPLE * max(baseline, 1.0))
    n_freq = min(max_nfreq, max(500, ceil(Int, (f_max - f_min) / df)))
    freqs = collect(range(f_min, f_max, length=n_freq))
    power = _lomb_scargle(t, residuals, freqs; σ = σ)

    # Suppress window function aliases (Dawson & Fabrycky 2010).
    # Compute window function once per dataset, cache it.
    # Suppress window function aliases (Dawson & Fabrycky 2010).
    # Compute window function on the same frequency grid.
    # Cache keyed by (data, n_freq, f_min, f_max) to handle different
    # planet slots with different period priors.
    cache_key = hash((objectid(t), n_freq, f_min, f_max))
    wcache = _get_window_cache(cache_key)
    if !wcache.initialized
        wcache.freqs = freqs
        wcache.power = _window_function(t, freqs)
        wcache.initialized = true
    end
    W = wcache.power
    W_max = maximum(W)
    if W_max > 0
        # Linear (rather than squared) suppression: zeros out exact
        # window peaks but leaves frequencies even modestly far from a
        # peak essentially untouched. The squared form aggressively
        # dampened genuine signals at integer-day-aliased periods —
        # e.g. a real P≈1d planet would be hit with 0.391 suppression
        # under (1-W/Wmax)². Linear gives 0.625 at the same frequency
        # and falls off cleanly to 1.0 within a few window-peak widths.
        suppression = max.(0.0, 1.0 .- W ./ W_max)
        power = power .* suppression
    end

    periods = 1.0 ./ freqs

    # Extract top peaks with guard-band
    peak_idx = Int[]
    min_spacing = max(3, div(n_freq, 20))
    sorted = sortperm(power, rev=true)

    for idx in sorted
        too_close = any(abs(idx - pi) < min_spacing for pi in peak_idx)
        too_close && continue
        push!(peak_idx, idx)
        length(peak_idx) >= INFORMED_N_PEAKS && break
    end

    isempty(peak_idx) && return (Float64[], Float64[], Float64[], Float64[])

    peak_periods = periods[peak_idx]
    peak_power = power[peak_idx]
    peak_K = Float64[]
    peak_Mo = Float64[]
    for i in peak_idx
        K_est, Mo_est = _estimate_K_phase(t, residuals, freqs[i], t_ref; σ = σ)
        push!(peak_K, K_est)
        push!(peak_Mo, Mo_est)
    end
    total = sum(peak_power)
    weights = total > 0 ? peak_power ./ total : fill(1.0 / length(peak_power), length(peak_power))

    return peak_periods, weights, peak_K, peak_Mo
end

# =====================================================================
# Informed proposal density
# =====================================================================

"""
    _informed_log_q(log_P, peak_log_P, weights, log_P_min, log_P_max) -> Float64

Log-density of the informed period mixture proposal at log(P).
"""
function _informed_log_q(log_P::Float64, peak_log_P::Vector{Float64},
                          weights::Vector{Float64},
                          log_P_min::Float64, log_P_max::Float64;
                          sigmas::Union{Nothing,Vector{Float64}}=nothing)
    α = INFORMED_ALPHA
    uniform = 1.0 / (log_P_max - log_P_min)
    gauss = 0.0
    for (i, mu) in enumerate(peak_log_P)
        # Per-peak width: transit (BLS) peaks use a tight, basin-matched σ
        # (so the precise refined period isn't blurred back out of the
        # transit basin); RV peaks keep the wide default. MUST match the
        # σ used to DRAW the proposal, or the Hastings ratio is wrong.
        σ = sigmas === nothing ? INFORMED_SIGMA_P : sigmas[i]
        gauss += weights[i] * exp(-0.5 * ((log_P - mu) / σ)^2) / (σ * sqrt(2π))
    end
    return log(max(α * gauss + (1 - α) * uniform, 1e-300))
end

# =====================================================================
# The period slot in its own units
# =====================================================================
# The informed proposals find peaks in PERIOD (days); the slot they fill holds
# that period, except under the `:a_driven` mass parametrization, where it holds
# the semi-major axis a (AU) and P follows from Kepler's third law at the total
# mass, P = KEPLER_YEAR_DAYS·√(a³/M). Reading that slot's bounds as days ran the
# periodogram over 0.3–3 d for an a ∈ [0.3, 3] AU prior: ~10⁵ frequencies of
# meaningless short periods, which dominated a trans-dim burn-in, and every
# "informed" a was a sub-3-day period peak written into an AU slot.
#
# At fixed M, log a = (2/3)·log P + (log M − 2 log KEPLER_YEAR_DAYS)/3 is linear
# in log P, so a period peak maps to an a peak and its log-normal width scales by
# 2/3: the proposal stays a mixture in the slot's own variable, drawn, clamped
# and evaluated there by the same `_informed_log_q`. Every other
# parametrization gets the identity, which leaves its draws bit for bit as they
# were.
struct _PeriodSlotMap
    identity::Bool
    slope::Float64      # d log(slot) / d log P
    offset::Float64     # log(slot) at log P = 0
end

# `values` carries the newborn's M_sec in `block.K` under `:a_driven`.
function _period_slot_map(theta::Theta, block, values)
    theta.params.config.parametrization.mass === :a_driven ||
        return _PeriodSlotMap(true, 1.0, 0.0)
    M = Float64(astrom_M_pri(theta) + values[block.K])
    return _PeriodSlotMap(false, 2 / 3, (log(M) - 2 * log(KEPLER_YEAR_DAYS)) / 3)
end

_slot_log(m::_PeriodSlotMap, log_P) = m.identity ? log_P : m.slope * log_P + m.offset
_slot_to_period(m::_PeriodSlotMap, x) =
    m.identity ? x : exp((log(x) - m.offset) / m.slope)
_slot_sigmas(m::_PeriodSlotMap, sigmas) =
    m.identity ? sigmas : m.slope .* sigmas

# The slot's log-range for the uniform part of the mixture (periods below 0.1 d
# excluded), and the periods the periodogram searches. The identity passes the
# prior's own bounds to the periodogram, as it always did.
function _slot_period_range(m::_PeriodSlotMap, x_lo, x_hi)
    log_x_min = max(log(max(x_lo, 0.0)), _slot_log(m, log(0.1)))
    log_x_max = log(x_hi)
    P_range = m.identity ? (x_lo, x_hi) :
        (_slot_to_period(m, exp(log_x_min)), _slot_to_period(m, x_hi))
    return log_x_min, log_x_max, P_range
end

# Under `:a_driven` the map needs the newborn's M_sec, which `block.K` holds, so
# that slot is drawn from its prior BEFORE the period. Returns whether it was,
# and its log-density for log q_fwd.
function _predraw_M_sec!(new_values, theta::Theta, block, rng)
    theta.params.config.parametrization.mass === :a_driven || return (false, 0.0)
    layout = theta.params.layout
    uf_pos = findfirst(==(block.K), layout.unfrozen_idx)
    uf_pos === nothing && return (false, 0.0)
    prior = layout.unfrozen_priors[uf_pos]
    lo, hi = bounds(prior)
    v = clamp(rand(rng, prior.dist), lo, hi)
    new_values[block.K] = v
    return (true, eval_packed_logpdf(v, layout.packed_priors, uf_pos))
end

# =====================================================================
# The informed planet proposal as a density that can be EVALUATED
# =====================================================================
#
# An informed birth draws a mixture component c (the uniform-in-log-period
# floor, c = 0, with probability 1 − α, or periodogram peak i with
# probability α·wᵢ), then the period from that component, then every other
# slot from a kernel that depends on c (K around the peak's LS amplitude, the
# time anchor around its phase, ...). Its density at a set of values is the
# MIXTURE over c, not the period mixture times the kernels of the component
# that happened to be drawn: the latter is not a function of the values alone.
#
# The same density is what the paired death needs: the reverse of killing a
# planet is an informed birth from the post-death state, so the death's
# Hastings ratio carries that birth's density at the killed values. The death
# used to carry none (the strategy-blind combinatorial death), which made the
# informed pair irreversible by log q_informed(β_killed) in every round trip —
# measured on a two-slot toy with an exact evidence: P(N = 2) = 0.43 against
# the true 0.63.
#
# `_InformedSetup` holds what a proposal from one state into one slot depends
# on; `_informed_draw!` and `_informed_slot_log_q` share `_informed_slot_kernel`,
# so the kernel a value is drawn from and the one it is scored under cannot
# drift apart.

struct _InformedSetup
    m::_PeriodSlotMap
    log_x_min::Float64
    log_x_max::Float64
    log_x::Vector{Float64}     # peak centres, in the slot's own log variable
    sigma_x::Vector{Float64}   # peak widths, same variable
    w::Vector{Float64}         # normalised peak weights
    K_ls::Vector{Float64}      # LS amplitude at each peak (0: none)
    Mo_ls::Vector{Float64}     # LS mean anomaly at data.t_ref
    source::Vector{Symbol}     # :ls, :bls or :both
    depth::Vector{Float64}     # BLS depth (0: none)
    t0::Vector{Float64}        # BLS absolute epoch (0: none)
    dur::Vector{Float64}       # BLS duration [d] (0: none)
end

const _QK_PRIOR = 0
const _QK_LOGNORMAL = 1
const _QK_NORMAL = 2
const _QK_WRAPPED = 3
const _QK_HALFNORMAL = 4

_bls_backed(s::Symbol) = s === :bls || s === :both

# K estimate of component c at period P_now (0: no informed K).
function _informed_K_est(theta::Theta, S::_InformedSetup, c::Int, P_now::Float64)
    c > 0 || return 0.0
    cfg = theta.params.config
    M_s = cfg.M_s; R_s = cfg.R_s
    have_stellar = !isnan(M_s) && !isnan(R_s) && M_s > 0 && R_s > 0
    if _bls_backed(S.source[c]) && S.depth[c] > 0 && have_stellar
        # depth → R_p (R_⊕) → M_p (R_⊕) → K (m/s); the larger of that and LS.
        M_p = _mass_radius_chenkipping(_radius_from_depth(S.depth[c], R_s))
        return max(_K_from_mass(M_p, P_now, M_s), S.K_ls[c])
    end
    return S.K_ls[c]
end

# The kernel slot `slot` of `block` is drawn from under component c:
# (kind, μ, σ, period). Period P_now is the newborn's period in days.
function _informed_slot_kernel(theta::Theta, block, slot::Int, S::_InformedSetup,
                                c::Int, P_now::Float64, data::Data)
    cfg = theta.params.config
    if has_K(block) && slot == block.K && cfg.parametrization.mass === :K_driven
        K_est = _informed_K_est(theta, S, c, P_now)
        K_est > 0 && return (_QK_LOGNORMAL, K_est, INFORMED_SIGMA_K, 0.0)
        return (_QK_PRIOR, 0.0, 0.0, 0.0)
    end
    if slot == block.e1 || slot == block.e2
        # sesinw/secosw from N(0, σ_e): biased toward circular.
        return (_QK_NORMAL, 0.0, INFORMED_SIGMA_E, 0.0)
    end
    if slot == block.t && c > 0
        # Map the peak's phase to whatever the parametrization stores
        # (:Mo / :phic / :Tp / :Tc): Tp = t_ref − Mo·P/(2π), Tc = Tp + P/4
        # (circular). A BLS epoch, when the peak has one, beats the LS phase.
        time_param = cfg.parametrization.time
        t_ref_c = cfg.parametrization.t_ref_c
        src = S.source[c]
        if _bls_backed(src) && S.t0[c] > 0 && time_param === :phic
            return (_QK_WRAPPED, tc_to_phic(S.t0[c], P_now, t_ref_c),
                    clamp(0.3 * S.dur[c], 0.005, 0.1) * 2π / P_now, 2π)
        elseif _bls_backed(src) && S.t0[c] > 0 && time_param === :Tc
            # chosen_t0 is an ABSOLUTE transit epoch near the data start; its
            # width is tied to the BLS duration (the Tc basin is ~a duration).
            return (_QK_WRAPPED, S.t0[c], clamp(0.3 * S.dur[c], 0.005, 0.1), P_now)
        elseif time_param === :Mo
            return (_QK_WRAPPED, S.Mo_ls[c], INFORMED_SIGMA_MO, 2π)
        elseif time_param === :phic
            Tc_est = data.t_ref - S.Mo_ls[c] * P_now / (2π) + P_now / 4
            return (_QK_WRAPPED, tc_to_phic(Tc_est, P_now, t_ref_c), INFORMED_SIGMA_MO, 2π)
        else
            Tp_est = data.t_ref - S.Mo_ls[c] * P_now / (2π)
            center = time_param === :Tp ? Tp_est : Tp_est + P_now / 4
            return (_QK_WRAPPED, center, INFORMED_SIGMA_MO * P_now / (2π), P_now)
        end
    end
    if has_geometry(block) && c > 0 && _bls_backed(S.source[c])
        M_s = cfg.M_s; R_s = cfg.R_s
        have_stellar = !isnan(M_s) && !isnan(R_s) && M_s > 0 && R_s > 0
        if slot == block.r && S.depth[c] > 0 && have_stellar
            # rr from sqrt(BLS depth), log-normal.
            return (_QK_LOGNORMAL, max(sqrt(S.depth[c]), 1e-3), 0.3, 0.0)
        elseif slot == block.b
            # Central-weighted impact parameter |N(0, σ_b)| (not inverted
            # from the BLS duration, which is biased low → grazing b).
            return (_QK_HALFNORMAL, 0.0, 0.35, 0.0)
        end
    end
    return (_QK_PRIOR, 0.0, 0.0, 0.0)
end

function _qk_draw(kind::Int, μ, σ, period, prior, lo, hi, rng)
    if kind == _QK_LOGNORMAL
        return clamp(exp(log(μ) + σ * randn(rng)), lo, hi)
    elseif kind == _QK_NORMAL
        return clamp(μ + σ * randn(rng), lo, hi)
    elseif kind == _QK_WRAPPED
        raw = μ + σ * randn(rng)
        return lo + (raw - lo) - period * floor((raw - lo) / period)
    elseif kind == _QK_HALFNORMAL
        return clamp(abs(σ * randn(rng)), lo, hi)
    else
        return clamp(rand(rng, prior.dist), lo, hi)
    end
end

function _qk_logpdf(kind::Int, μ, σ, period, v::Float64, layout, uf_pos::Int)
    if kind == _QK_LOGNORMAL
        v > 0 || return -Inf
        return -log(v * σ * sqrt(2π)) - 0.5 * ((log(v) - log(μ)) / σ)^2
    elseif kind == _QK_NORMAL
        return -log(σ * sqrt(2π)) - 0.5 * ((v - μ) / σ)^2
    elseif kind == _QK_WRAPPED
        δ = v - μ
        δ -= period * round(δ / period)
        return -log(σ * sqrt(2π)) - 0.5 * (δ / σ)^2
    elseif kind == _QK_HALFNORMAL
        return log(2.0) - log(σ * sqrt(2π)) - 0.5 * (v / σ)^2
    else
        return Float64(eval_packed_logpdf(v, layout.packed_priors, uf_pos))
    end
end

"""
    _informed_draw!(new_values, theta, block, S, data, rng, M_sec_drawn)

Draw the newborn's slot values from the informed mixture `S` into
`new_values`: component, then period, then the component's kernels.
"""
function _informed_draw!(new_values::AbstractVector{T}, theta::Theta, block,
                          S::_InformedSetup, data::Data, rng::AbstractRNG,
                          M_sec_drawn::Bool) where {T}
    layout = theta.params.layout
    c = 0
    if !isempty(S.w) && rand(rng) < INFORMED_ALPHA
        r = rand(rng); cum = 0.0; c = 1
        for (i, wi) in enumerate(S.w)
            cum += wi
            if r < cum
                c = i
                break
            end
        end
        log_x = S.log_x[c] + S.sigma_x[c] * randn(rng)
    else
        log_x = S.log_x_min + (S.log_x_max - S.log_x_min) * rand(rng)
    end
    x = exp(clamp(log_x, S.log_x_min, S.log_x_max))
    new_values[block.P] = convert(T, x)
    P_now = _slot_to_period(S.m, x)
    for slot in planet_slot_indices(block)
        slot == block.P && continue
        M_sec_drawn && slot == block.K && continue
        uf_pos = findfirst(==(slot), layout.unfrozen_idx)
        uf_pos === nothing && continue
        prior = layout.unfrozen_priors[uf_pos]
        lo, hi = bounds(prior)
        kind, μ, σ, period = _informed_slot_kernel(theta, block, slot, S, c, P_now, data)
        new_values[slot] = convert(T, _qk_draw(kind, μ, σ, period, prior, lo, hi, rng))
    end
    return new_values
end

"""
    _informed_slot_log_q(theta, block, vals, S, data, M_sec_drawn) -> Float64

Log-density of the informed mixture `S` at the slot values in `vals`
(excluding a pre-drawn M_sec): log Σ_c π_c q(x | c) Π_slots q(v | c), with the
period density on the slot variable x (the Jacobian from log x is included,
matching the prior's convention in Δlog π).
"""
function _informed_slot_log_q(theta::Theta, block, vals::AbstractVector,
                               S::_InformedSetup, data::Data, M_sec_drawn::Bool)
    layout = theta.params.layout
    x = Float64(vals[block.P])
    x > 0 || return -Inf
    log_x = log(x)
    # The draw clamps into [log_x_min, log_x_max]: nothing outside is reachable.
    tol = 1e-9 * max(1.0, abs(S.log_x_max))
    (S.log_x_min - tol <= log_x <= S.log_x_max + tol) || return -Inf
    P_now = _slot_to_period(S.m, x)
    α = INFORMED_ALPHA
    n_c = length(S.w)
    terms = Vector{Float64}(undef, n_c + 1)
    for c in 0:n_c
        if c == 0
            lw = (n_c == 0 ? 0.0 : log1p(-α)) - log(S.log_x_max - S.log_x_min)
        else
            if !(S.w[c] > 0)
                terms[c + 1] = -Inf
                continue
            end
            σc = S.sigma_x[c]
            lw = log(α) + log(S.w[c]) - log(σc * sqrt(2π)) -
                 0.5 * ((log_x - S.log_x[c]) / σc)^2
        end
        lr = 0.0
        for slot in planet_slot_indices(block)
            slot == block.P && continue
            M_sec_drawn && slot == block.K && continue
            uf_pos = findfirst(==(slot), layout.unfrozen_idx)
            uf_pos === nothing && continue
            kind, μ, σ, period = _informed_slot_kernel(theta, block, slot, S, c, P_now, data)
            lr += _qk_logpdf(kind, μ, σ, period, Float64(vals[slot]), layout, uf_pos)
        end
        terms[c + 1] = lw + lr
    end
    mx = maximum(terms)
    isfinite(mx) || return -Inf
    return mx + log(sum(t -> exp(t - mx), terms)) - log_x
end

# LS-only setup (InformedBirth). `bump`: count this call towards the cache's
# refresh interval (births do; a death's reverse-density evaluation only reads
# the cache, refreshing it when empty).
function _ls_informed_setup(theta::Theta, m::_PeriodSlotMap, log_x_min, log_x_max,
                             P_lo, P_hi, data::Data, rng::AbstractRNG; bump::Bool)
    # Cached periodogram, keyed by (rng, active-set signature): the peaks are
    # refined against the active-subtracted residuals, so the cache must be
    # state-consistent.
    cache = _get_cache(_active_set_signature(theta, objectid(rng)))
    bump && (cache.calls_since_update += 1)
    if (bump && cache.calls_since_update >= INFORMED_CACHE_INTERVAL) ||
            isempty(cache.periods)
        residuals = _compute_rv_residuals(theta, data)
        cache.periods, cache.weights, cache.K_est, cache.Mo_est =
            _find_peaks(data.t_rv, residuals, P_lo, P_hi;
                         t_ref = data.t_ref, σ = data.rv_err)
        cache.log_periods = isempty(cache.periods) ? Float64[] : log.(cache.periods)
        cache.calls_since_update = 0
    end
    isempty(cache.periods) && return nothing
    n = length(cache.periods)
    return _InformedSetup(m, Float64(log_x_min), Float64(log_x_max),
                          _slot_log.(Ref(m), cache.log_periods),
                          fill(m.slope * INFORMED_SIGMA_P, n), copy(cache.weights),
                          copy(cache.K_est), copy(cache.Mo_est), fill(:ls, n),
                          zeros(n), zeros(n), zeros(n))
end

# The prior density of a slot's unfrozen values: the PriorBirth proposal,
# which every informed birth falls back to when it has no peaks.
function _prior_slot_log_q(theta::Theta, block, vals::AbstractVector)
    layout = theta.params.layout
    lq = 0.0
    for slot in planet_slot_indices(block)
        uf_pos = findfirst(==(slot), layout.unfrozen_idx)
        uf_pos === nothing && continue
        lq += eval_packed_logpdf(Float64(vals[slot]), layout.packed_priors, uf_pos)
    end
    return lq
end

# The informed birth density at `vals` into slot k's block, from the state
# `theta_from` (planet k inactive there): the reverse of a death.
function _informed_reverse_log_q(theta_from::Theta, k::Int, vals::AbstractVector,
                                  data::Data, rng::AbstractRNG, joint::Bool)
    layout = theta_from.params.layout
    block = layout.planet_blocks[k]
    P_pos = findfirst(==(block.P), layout.unfrozen_idx)
    P_pos === nothing && return _prior_slot_log_q(theta_from, block, vals)
    x_lo, x_hi = bounds(layout.unfrozen_priors[P_pos])
    M_sec_drawn = false; lq_M = 0.0
    if theta_from.params.config.parametrization.mass === :a_driven
        ufK = findfirst(==(block.K), layout.unfrozen_idx)
        if ufK !== nothing
            M_sec_drawn = true
            lq_M = Float64(eval_packed_logpdf(Float64(vals[block.K]),
                                               layout.packed_priors, ufK))
        end
    end
    m = _period_slot_map(theta_from, block, vals)
    log_x_min, log_x_max, (P_lo, P_hi) = _slot_period_range(m, x_lo, x_hi)
    S = joint ?
        _joint_informed_setup(theta_from, m, log_x_min, log_x_max, P_lo, P_hi,
                              data, rng; bump = false) :
        _ls_informed_setup(theta_from, m, log_x_min, log_x_max, P_lo, P_hi,
                           data, rng; bump = false)
    S === nothing && return _prior_slot_log_q(theta_from, block, vals)
    return lq_M + _informed_slot_log_q(theta_from, block, vals, S, data, M_sec_drawn)
end

# The death paired with an informed birth: uniform choice of the planet, and
# the reverse informed-birth density at its values (see the section header).
function _informed_death(theta::Theta{T}, rng::AbstractRNG, joint::Bool, data::Data,
                          scratch) where {T}
    td = theta.td
    max_k = length(td.planet_active)
    n_active = td.n_planets_active
    n_active > 0 || return (theta, convert(T, -Inf))
    active = active_planets(td)
    k = active[rand(rng, 1:length(active))]
    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        deactivate_planet!(scratch.td, k)
        new_theta = scratch
    else
        new_td = copy(td)
        deactivate_planet!(new_td, k)
        new_theta = Theta{T}(theta.params, copy(theta.values); td = new_td)
    end
    log_q_reverse = _informed_reverse_log_q(new_theta, k, theta.values, data, rng, joint)
    _, p_death = _birth_death_probs(n_active, max_k)
    p_birth_new, _ = _birth_death_probs(n_active - 1, max_k)
    G_after = length(group_first_inactive(new_theta.td, birth_group_keys(theta.params)))
    log_q_ratio = log(p_birth_new) + log(n_active) - log(p_death) +
                  log_q_reverse - log(max(G_after, 1))
    return (new_theta, convert(T, log_q_ratio))
end

# =====================================================================
# InformedBirth proposal
# =====================================================================

"""
    propose_planet_birth(theta, rng, ::InformedBirth; data) -> (new_theta, log_q_ratio)

Period proposed from Lomb-Scargle periodogram peaks (a mixture of Gaussians in
log P over the peaks plus a uniform-in-log floor); for a peak component the RV
semi-amplitude (under `:K_driven`) and the time anchor are centred on the
peak's least-squares amplitude and phase; sesinw/secosw are drawn near zero;
the rest from their priors. The Hastings ratio carries the full mixture
density (`_informed_slot_log_q`), and the paired death,
`propose_planet_death(theta, rng, ::InformedBirth; data)`, carries the same
density at the killed values. Falls back to PriorBirth if data is unavailable
or no peaks are found.

For astrometric block subtypes (`RVASBlock`, `RVPMASBlock`) the
inclination, longitude-of-node, and (under `:M_sec_driven`/`:a_driven`)
secondary-mass slots are drawn from their priors. Under `:a_driven` the period
slot holds a: the periodogram runs over the periods that slot's bounds span,
and the drawn period is placed as a at the newborn's total mass
(`_PeriodSlotMap`).
"""
function propose_planet_birth(theta::Theta{T}, rng::AbstractRNG,
                               ::InformedBirth;
                               data::Union{Data, Nothing}=nothing,
                               population::Union{AbstractVector{<:Theta}, Nothing}=nothing,
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    # Need data for residuals, and RVs for the periodogram that informs the
    # period. An astrometry-only fit has neither, so a birth there is a prior
    # draw — `informed_birth_fraction` cannot inform anything.
    (data === nothing || isempty(data.t_rv)) &&
        return propose_planet_birth(theta, rng, PriorBirth(); scratch=scratch)
    return _informed_birth(theta, rng, data, scratch, false)
end

function _informed_birth(theta::Theta{T}, rng::AbstractRNG, data::Data, scratch,
                          joint::Bool) where {T}
    td = theta.td
    max_k = length(td.planet_active)
    n_active = td.n_planets_active

    cand_slots = group_first_inactive(td, birth_group_keys(theta.params))
    isempty(cand_slots) && return (theta, convert(T, -Inf))
    k = cand_slots[rand(rng, 1:length(cand_slots))]
    G_slot = length(cand_slots)

    layout = theta.params.layout
    block = layout.planet_blocks[k]
    P_pos = findfirst(==(block.P), layout.unfrozen_idx)
    P_pos === nothing && return propose_planet_birth(theta, rng, PriorBirth(); scratch=scratch)
    x_lo, x_hi = bounds(layout.unfrozen_priors[P_pos])

    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        new_values = scratch.values
        new_td = scratch.td
    else
        new_values = copy(theta.values)
        new_td = copy(td)
    end

    # Under :a_driven, M_sec first (_PeriodSlotMap).
    M_sec_drawn, lq_M = _predraw_M_sec!(new_values, theta, block, rng)
    m = _period_slot_map(theta, block, new_values)
    log_x_min, log_x_max, (P_lo, P_hi) = _slot_period_range(m, x_lo, x_hi)
    S = joint ?
        _joint_informed_setup(theta, m, log_x_min, log_x_max, P_lo, P_hi,
                              data, rng; bump = true) :
        _ls_informed_setup(theta, m, log_x_min, log_x_max, P_lo, P_hi,
                           data, rng; bump = true)
    S === nothing && return propose_planet_birth(theta, rng, PriorBirth(); scratch=scratch)

    _informed_draw!(new_values, theta, block, S, data, rng, M_sec_drawn)
    log_q_fwd = lq_M + _informed_slot_log_q(theta, block, new_values, S, data, M_sec_drawn)

    activate_planet!(new_td, k)
    new_theta = scratch !== nothing ? scratch :
                Theta{T}(theta.params, new_values; td = new_td)

    p_birth, _ = _birth_death_probs(n_active, max_k)
    _, p_death_new = _birth_death_probs(n_active + 1, max_k)
    # +log(G_slot): forward slot choice uniform over the birth groups.
    log_q_comb = log(p_death_new) - log(n_active + 1) - log(p_birth) + log(G_slot)

    _sort_group_periods!(new_theta)
    return (new_theta, convert(T, log_q_comb - log_q_fwd))
end

"""
    propose_planet_death(theta, rng, ::InformedBirth; data, scratch) -> (new_theta, log_q_ratio)

Death paired with `InformedBirth`: kill a uniformly chosen active planet and
carry the informed-birth density at its values, evaluated from the post-death
state (the state the reverse birth would start from). Without `data` (or
without RVs) the reverse birth is a prior draw, so this is the PriorBirth death.
"""
function propose_planet_death(theta::Theta{T}, rng::AbstractRNG, ::InformedBirth;
                               data::Union{Data, Nothing}=nothing,
                               population=nothing,
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    (data === nothing || isempty(data.t_rv)) &&
        return propose_planet_death(theta, rng, PriorBirth(); scratch=scratch)
    return _informed_death(theta, rng, false, data, scratch)
end

# =====================================================================
# JointInformedBirth — combine RV-LS and photometry-BLS into one
# proposal, with active-planet harmonic deduplication on the BLS side.
# Falls back to InformedBirth when no photometry is available.
# =====================================================================

# Per-RNG cache for BLS peaks (analogous to PeakCache for LS).
mutable struct BLSCache
    periods::Vector{Float64}
    weights::Vector{Float64}    # SNR-normalized within the BLS top-N
    depths::Vector{Float64}
    t0s::Vector{Float64}
    durs::Vector{Float64}       # best-fit transit duration [days] per peak
    log_periods::Vector{Float64}
    calls_since_update::Int
end
BLSCache() = BLSCache(Float64[], Float64[], Float64[], Float64[], Float64[],
                       Float64[], INFORMED_CACHE_INTERVAL)

const _BLS_CACHES = Dict{UInt64, BLSCache}()
const _BLS_CACHE_LOCK = ReentrantLock()

"""
    reset_transdim_caches!()

Drop every process-global informed-birth cache (periodogram peaks, BLS peaks,
window functions, per-instrument γ medians).

**Call this between runs when benchmarking more than one configuration in a
single Julia process.** These caches are keyed on `objectid(...)` of the RNG
and of the data arrays. `objectid` is address-derived and the GC may reuse an
address once an object is collected, so a cache entry populated by one run can
in principle be served to a later run in the same process. On top of that the
`length > 512 && empty!` guard wipes *everything* at once, so which entries
survive depends on how many distinct keys earlier runs happened to create.
Between them, a fixed seed does not pin the result: the same script can recover
a different period depending on what ran before it in the process.

Resetting makes each configuration see a cold cache, which is what a fair
benchmark needs. It is not a substitute for content-based keys — that is the
real fix, and it is still open.
"""
function reset_transdim_caches!()
    lock(_PEAK_CACHE_LOCK) do
        empty!(_PEAK_CACHES)
        empty!(_WINDOW_CACHES)
        empty!(_GAMMA_MEDIAN_CACHE)
    end
    lock(_BLS_CACHE_LOCK) do
        empty!(_BLS_CACHES)
    end
    # Defined in proposals.jl (included after this file); resolved at call time.
    lock(_GP_HINT_LOCK) do
        empty!(_GP_HINT_CACHE)
    end
    return nothing
end

function _get_bls_cache(chain_id::UInt64)
    lock(_BLS_CACHE_LOCK) do
        # Junk-state churn (hot chains) can mint many signatures; rebuilds are
        # cheap relative to unbounded growth, so just reset when oversized.
        length(_BLS_CACHES) > 512 && empty!(_BLS_CACHES)
        get!(() -> BLSCache(), _BLS_CACHES, chain_id)
    end
end

# Active-set signature for proposal caches. The cached BLS/LS peaks are
# REFINED AGAINST the active-planet-subtracted residuals, so a cache built at
# one walker state is silently wrong at another: a cache built at a hot
# walker's junk state proposes e with a P/t0 refined on d-contaminated flux,
# and every resulting birth misses e's 4e-4-wide phase needle (measured:
# 9,961 cold multi-try selections of e → 22 acceptances with a thread-keyed
# cache). Keying by (rng, active periods) makes proposals state-consistent;
# walkers sharing the locked state share one entry, so cold-chain reuse
# stays high. Rounded to 2 log-P decimals so basin-level param scatter
# across walkers doesn't fragment the key.
#
# The signature is built in two parts, the walker's (`_active_set_key`) and the
# RNG's, so a checkpoint can carry a walker's entries over to a new RNG object
# (`_informed_cache_snapshot` below).
_active_set_signature(theta::Theta, rng_id::UInt64) =
    _active_set_signature(_active_set_key(theta), rng_id)

function _active_set_signature(key::AbstractVector, rng_id::UInt64)
    h = rng_id
    for r in key
        h = hash(r, h)
    end
    return h
end

# 1 decimal in log-P (~10% bins): coarse enough that junk-planet churn
# collapses onto few signatures (each new signature pays a ~0.5 s BLS
# rebuild — at 6 slots fine 2-decimal rounding made burn-in rebuild-bound,
# ~2.4 s/step), while distinct real planets (even 3:2 chain neighbours,
# Δlog P ≈ 0.4) still separate.
_active_set_key(theta::Theta) =
    [round(log(planet_P(theta, k)), digits = 1) for k in planet_indices(theta)]

# --- Checkpoints (src/checkpoint.jl) ---------------------------------------
# A sampler whose walkers make informed births carries their entries of
# _PEAK_CACHES and _BLS_CACHES in its checkpoint: an entry holds the peaks found
# at an earlier walker state and the count towards its next refresh, so a
# continuation that started from cold caches would propose different births.
# Entries are keyed by `objectid` of the walker's RNG, which a restored RNG does
# not keep, so the sampler records each walker's `_active_set_key`s as it births
# and the entries are re-keyed onto the new RNG objects on restore.

"""
    _informed_cache_snapshot(rngs, keys) -> Vector

The cache entries of walker `i` (RNG `rngs[i]`) under each active-set key in
`keys[i]`, as `(i, key, peak cache, BLS cache)`. Keys whose entries are gone (the
oversize reset wiped them, or the birth never reached the cache) are dropped from
`keys[i]`; a later birth there records them again.
"""
function _informed_cache_snapshot(rngs::AbstractVector,
                                  keys::AbstractVector{<:AbstractSet})
    snap = Tuple{Int, Vector, Union{Nothing, PeakCache},
                 Union{Nothing, BLSCache}}[]
    for (i, ks) in enumerate(keys)
        id = objectid(rngs[i])
        for key in collect(ks)
            sig = _active_set_signature(key, id)
            pc = lock(() -> get(_PEAK_CACHES, sig, nothing), _PEAK_CACHE_LOCK)
            bc = lock(() -> get(_BLS_CACHES, sig, nothing), _BLS_CACHE_LOCK)
            if pc === nothing && bc === nothing
                delete!(ks, key)
            else
                push!(snap, (i, key, pc, bc))
            end
        end
    end
    return snap
end

"""
    _informed_cache_restore!(snap, rngs, keys)

Put the entries of a `_informed_cache_snapshot` back, keyed by the RNG objects
`rngs` now in use, and record their keys in `keys`.
"""
function _informed_cache_restore!(snap, rngs::AbstractVector,
                                  keys::AbstractVector{<:AbstractSet})
    for (i, key, pc, bc) in snap
        sig = _active_set_signature(key, objectid(rngs[i]))
        pc === nothing || lock(() -> (_PEAK_CACHES[sig] = pc), _PEAK_CACHE_LOCK)
        bc === nothing || lock(() -> (_BLS_CACHES[sig] = bc), _BLS_CACHE_LOCK)
        push!(keys[i], key)
    end
    return nothing
end

"""
    _bls_peaks_filtered(theta, data, P_lo, P_hi, cache; rng, force_refresh=false)
        -> (periods, weights, depths, t0s)

BLS peak-finder that subtracts active-planet box transits from the
photometric residuals before searching, then deduplicates the surviving
peaks against active planet periods (m·P/n harmonics) so the chain does
not propose aliases of already-found planets. Caches like the LS peaks
so we only pay the BLS cost every INFORMED_CACHE_INTERVAL birth calls.
"""
function _bls_peaks_filtered(theta::Theta, data::Data,
                              P_lo::Float64, P_hi::Float64,
                              cache::BLSCache;
                              n_peaks::Int = INFORMED_N_PEAKS)
    # Subtract active-planet transits using the FULL Mandel-Agol model
    # via `phot_predictions`. Earlier this function used a box-approximation
    # (depth = rr², 4% phase window) which left ~30% shape residuals at the
    # active planet's period — those residuals showed up as BLS peaks at the
    # active planet's period and harmonics, drowning out genuine new-planet
    # signals (the WASP-47 d/e at 9.03d/0.79d were missed because b's
    # imperfect-shape residuals dominated). Forward-eval cost is ~0.13ms
    # per call, called once per BLS proposal — negligible vs the run.
    n_phot = length(data.t_phot)
    n_phot > 50 || return (Float64[], Float64[], Float64[], Float64[], Float64[])
    layout = theta.params.layout
    parametrization = theta.params.config.parametrization

    # Active planet periods (used for harmonic dedup later)
    active_periods = Float64[]
    for k in planet_indices(theta)
        block = layout.planet_blocks[k]
        has_geometry(block) || continue
        push!(active_periods, Float64(planet_P(theta, k)))
    end

    # `phot_predictions` returns the noise-free model:
    # `(1 + offset_inst) × Π_k transit_flux_k(t)`.
    # For BLS we want flux with active transits removed but per-instrument
    # offset preserved → flux_sub = data.flux + (baseline_inst − model).
    flux_sub = if isempty(active_periods)
        copy(data.flux)
    else
        preds, _vars = phot_predictions(theta, data)
        out = Vector{Float64}(undef, n_phot)
        @inbounds for i in 1:n_phot
            ins = data.phot_inst[i]
            offset = pm_offset(theta, ins)
            baseline = 1.0 + Float64(offset)
            out[i] = data.flux[i] + (baseline - Float64(preds[i]))
        end
        out
    end

    # Additional safety mask: zero out (set to baseline) any cadence still
    # within ±4× transit duration of a known active transit. With
    # phot_predictions correctly subtracting the model this is mostly a
    # belt-and-braces step against minor period/shape errors compounding
    # over multi-year baselines.
    if !isempty(active_periods)
        for k in planet_indices(theta)
            block = layout.planet_blocks[k]
            has_geometry(block) || continue
            P_a = Float64(planet_P(theta, k))
            e, w = planet_e_w(theta, k)
            b_raw, rr_raw = planet_b_rr(theta, k)
            bb, rr = parametrization.geom === :r1r2 ?
                     (b_raw * (1 - rr_raw) / 2, b_raw * (1 + rr_raw) / 2) :
                     (b_raw, rr_raw)
            bb = Float64(bb); rr = Float64(rr)
            bb < 1 + rr || continue
            ta = Float64(planet_time_anchor(theta, k))
            Tc_k = Float64(anchor_to_tc(parametrization.time, ta, P_a, Float64(e),
                                        Float64(w), data.t_ref,
                                        parametrization.t_ref_c))
            T_dur_frac = 0.08   # 2× the 4% used previously — covers slight
                                # period drift over multi-year baselines
            @inbounds for i in 1:n_phot
                ph = mod(data.t_phot[i] - Tc_k, P_a) / P_a
                if ph < T_dur_frac/2 || ph > 1.0 - T_dur_frac/2
                    ins = data.phot_inst[i]
                    flux_sub[i] = 1.0 + Float64(pm_offset(theta, ins))
                end
            end
        end
    end

    # BLS over the slot's period range. Period grid sized for proposal
    # use, NOT for spectroscopic-grade peak resolution: σ_P=0.05 in
    # log-space already smears proposals by 5%, so a 2000-period grid
    # spanning [P_lo, P_hi] log-uniformly resolves the relevant peak
    # structure. The previous formula (oversample × baseline) gave
    # 30k+ frequencies for short-period slots on the multi-year combined
    # K2+TESS span, which made each BLS call ~3 sec and brought the chain
    # to a crawl.
    # Grid resolution must keep a transit phase-coherent over the baseline:
    # a peak is lost when the grid misses its period by more than
    # ~duration/(2·baseline) in log-P (the transit smears by > half its width
    # when folded). The binding case is the SHORT-period end — a USP makes
    # 10-100× more cycles over the baseline, so a fixed 2000-point grid that
    # resolves P~9 d (and did) puts a P~0.8 d USP permanently below the coarse
    # noise floor (it never ranks top-n, never gets refined, can never be
    # born — WASP-47 e sat at exactly 0% because of this). Size the grid from
    # that physical requirement, capped for cost; the 500-call cache amortizes.
    _Ms = theta.params.config.M_s; _Rs = theta.params.config.R_s
    _rho = (!isnan(_Ms) && !isnan(_Rs) && _Rs > 0) ? _Ms / _Rs^3 : nothing
    baseline_b = max(data.t_phot[end] - data.t_phot[1], 1.0)
    dur_lo = (_rho === nothing ? 0.04 :
              first(_duty_grid(_rho, P_lo, [0.7]))) * P_lo
    dlogP_req = max(dur_lo / (2 * baseline_b), 1e-5)
    n_period = clamp(ceil(Int, log(P_hi / P_lo) / dlogP_req), 2000, 12_000)
    periods = exp.(range(log(P_lo), log(P_hi); length = n_period))
    # Strided-decimate the LC to ≤ MAX_BLS_OBS cadences for the fold (see the
    # MAX_BLS_OBS comment). Stride is in cadence over the gap-concatenated LC,
    # so the kept points still span all phases over the baseline.
    t_b, f_b, e_b = if length(data.t_phot) > MAX_BLS_OBS
        s = cld(length(data.t_phot), MAX_BLS_OBS)
        (data.t_phot[1:s:end], flux_sub[1:s:end], data.flux_err[1:s:end])
    else
        (data.t_phot, flux_sub, data.flux_err)
    end
    # a/Rs-scaled duration grid (rho_star): USP planets have ~10% duty cycles,
    # which the default fixed fractional grid only matches at the compressed
    # harmonic (so births alias). _rho is computed above for the grid sizing.
    P_bls, depth_bls, t0_bls, snr_bls, dur_bls =
        _box_least_squares(t_b, f_b, e_b, periods;
                            n_phase_bins = 100, n_peaks = 2 * n_peaks,
                            rho_star = _rho)

    # Drop any peak that is a small-integer harmonic of an active planet
    # period. This is THE fix for the v7 lock-in to b's 2P_b/3P_b aliases.
    P_kept, snr_kept = _harmonic_dedup(P_bls, snr_bls, active_periods;
                                         tol = INFORMED_DEDUP_TOL, max_order = 5)
    # Intra-list family dedup: collapse each integer-ratio family to its
    # fundamental (depth-gated — see _family_dedup_keep), so one strong
    # unmodelled planet doesn't fill the list with its own harmonics and
    # dilute its proposal mass.
    dep_kept = Float64[(ci = findfirst(==(P), P_bls);
                        ci === nothing ? 0.0 : depth_bls[ci]) for P in P_kept]
    fam = _family_dedup_keep(P_kept, dep_kept; tol = INFORMED_DEDUP_TOL, max_order = 5)
    P_kept = P_kept[fam]; snr_kept = snr_kept[fam]
    if length(P_kept) > n_peaks
        ord = sortperm(snr_kept; rev = true)[1:n_peaks]
        P_kept = P_kept[ord]
        snr_kept = snr_kept[ord]
    end
    # Refine the top peaks from the coarse grid to transit precision, and
    # carry their (refined) depth/t0/duration. The coarse grid resolves
    # only ~0.2% in period; over a multi-year baseline a transit needs
    # ~1e-5 to stay phase-aligned, so without this an informed birth
    # proposes a period that smears the transits and is rejected.
    # Refinement folds the FULL-resolution active-subtracted LC (NOT the
    # decimated detection LC) via a non-binned anchored box. Weights are
    # hoisted (same flux for all peaks); only the top few peaks by SNR are
    # refined (the dominant cost) — the rest keep their coarse period and
    # get dur=0 so the proposal gives them the WIDE σ (an unrefined period
    # under a tight σ would just be rejected).
    w_full  = 1.0 ./ (data.flux_err .^ 2)
    W_full  = sum(w_full)
    fm_full = sum(w_full .* flux_sub) / W_full
    wf_full = w_full .* (flux_sub .- fm_full)
    refine_set = Set(sortperm(snr_kept; rev = true)[1:min(3, length(P_kept))])
    P_ref = similar(P_kept)
    depth_kept = Float64[]; t0_kept = Float64[]; dur_kept = Float64[]
    for (i, P) in enumerate(P_kept)
        ci    = findfirst(==(P), P_bls)
        t0_c  = ci === nothing ? 0.0 : t0_bls[ci]
        dur_c = (ci === nothing || dur_bls[ci] <= 0) ? 0.04 * P : dur_bls[ci]
        dep_c = ci === nothing ? 0.0 : depth_bls[ci]
        if i in refine_set
            Pr, dep, t0, dur, _ = _refine_bls_period(data.t_phot, w_full, wf_full,
                                                      W_full, P, t0_c, dur_c)
            P_ref[i] = Pr
            push!(depth_kept, dep); push!(t0_kept, t0); push!(dur_kept, dur)
        else
            P_ref[i] = P
            # Convert the BJD-0-anchored fold phase to an ABSOLUTE epoch at the
            # SAME coarse P (where the mod identity is exact) — see
            # _refine_bls_period for the lever-arm failure this avoids.
            t0_abs = data.t_phot[1] + mod(t0_c - data.t_phot[1], P)
            push!(depth_kept, dep_c); push!(t0_kept, t0_abs); push!(dur_kept, 0.0)
        end
    end
    total_snr = sum(snr_kept)
    weights = total_snr > 0 ? snr_kept ./ total_snr :
              fill(1.0 / max(length(P_kept), 1), length(P_kept))
    return (P_ref, weights, depth_kept, t0_kept, dur_kept)
end

# The joint LS+BLS proposal setup (JointInformedBirth). `nothing` when neither
# side has a peak. `bump` as in `_ls_informed_setup`.
function _joint_informed_setup(theta::Theta, m::_PeriodSlotMap, log_x_min, log_x_max,
                                P_lo, P_hi, data::Data, rng::AbstractRNG; bump::Bool)
    # ---- LS peaks (same caching pathway as InformedBirth) ----
    state_sig = _active_set_signature(theta, objectid(rng))
    ls_cache = _get_cache(state_sig)
    bump && (ls_cache.calls_since_update += 1)
    if (bump && ls_cache.calls_since_update >= INFORMED_CACHE_INTERVAL) ||
                isempty(ls_cache.periods)
        if data.t_rv === nothing || isempty(data.t_rv)
            # PM-only target: no RV → no Lomb-Scargle contribution. Birth
            # uses the BLS photometry peaks alone (_find_peaks would reduce
            # over an empty residual vector and throw).
            ls_cache.periods = Float64[]; ls_cache.weights = Float64[]
            ls_cache.K_est   = Float64[]; ls_cache.Mo_est  = Float64[]
        else
            residuals = _compute_rv_residuals(theta, data)
            ls_cache.periods, ls_cache.weights, ls_cache.K_est, ls_cache.Mo_est =
                _find_peaks(data.t_rv, residuals, P_lo, P_hi;
                             t_ref = data.t_ref, σ = data.rv_err)
        end
        ls_cache.log_periods = isempty(ls_cache.periods) ? Float64[] :
                                log.(ls_cache.periods)
        ls_cache.calls_since_update = 0
    end
    # Filter LS peaks against active periods too (harmonic structure
    # leaks into RV LS for high-K planets).
    active_periods = Float64[planet_P(theta, kk) for kk in planet_indices(theta)]
    ls_P_kept, ls_w_kept = _harmonic_dedup(ls_cache.periods, ls_cache.weights,
                                              active_periods; tol = INFORMED_DEDUP_TOL,
                                              max_order = 5)
    # Carry K_est/Mo_est for the kept peaks
    ls_K = Float64[]; ls_Mo = Float64[]
    for P in ls_P_kept
        idx = findfirst(==(P), ls_cache.periods)
        push!(ls_K,  idx === nothing ? 0.0 : ls_cache.K_est[idx])
        push!(ls_Mo, idx === nothing ? 0.0 : ls_cache.Mo_est[idx])
    end

    # ---- BLS peaks (cached, harmonic-deduped) ----
    bls_cache = _get_bls_cache(state_sig)
    bump && (bls_cache.calls_since_update += 1)
    if (bump && bls_cache.calls_since_update >= INFORMED_CACHE_INTERVAL) ||
                isempty(bls_cache.periods)
        bls_cache.periods, bls_cache.weights, bls_cache.depths,
            bls_cache.t0s, bls_cache.durs =
            _bls_peaks_filtered(theta, data, P_lo, P_hi, bls_cache)
        bls_cache.log_periods = isempty(bls_cache.periods) ? Float64[] :
                                  log.(bls_cache.periods)
        bls_cache.calls_since_update = 0
    end

    # Combine into joint peak list. Weighting rule: for each peak we keep
    # the larger of (LS_norm_weight, BLS_norm_weight), so RV-only and
    # transit-only peaks both count, and matched signals get the larger
    # source. Periods within tol are merged (LS K/Mo + BLS depth/t0).
    all_P = Float64[]
    all_w = Float64[]
    all_K = Float64[]
    all_Mo = Float64[]
    all_depth = Float64[]
    all_t0 = Float64[]
    all_dur = Float64[]
    all_source = Symbol[]
    used_bls = falses(length(bls_cache.periods))
    tol_match = 0.03
    for (i, P) in enumerate(ls_P_kept)
        # Look for a BLS peak that matches
        bls_match = 0
        for (j, P_b) in enumerate(bls_cache.periods)
            used_bls[j] && continue
            if abs(P - P_b) / P < tol_match
                bls_match = j
                break
            end
        end
        if bls_match > 0
            # Matched RV+transit: use the REFINED BLS period (transit-
            # precise) as the proposal centre, not the coarse LS period —
            # the transit period is orders of magnitude better localized,
            # and this peak gets the tight transit σ below.
            push!(all_P, bls_cache.periods[bls_match])
            push!(all_w, max(ls_w_kept[i], bls_cache.weights[bls_match]))
            push!(all_K, ls_K[i])
            push!(all_Mo, ls_Mo[i])
            push!(all_depth, bls_cache.depths[bls_match])
            push!(all_t0,    bls_cache.t0s[bls_match])
            push!(all_dur,   bls_cache.durs[bls_match])
            push!(all_source, :both)
            used_bls[bls_match] = true
        else
            push!(all_P, P)
            push!(all_w, ls_w_kept[i])
            push!(all_K, ls_K[i])
            push!(all_Mo, ls_Mo[i])
            push!(all_depth, 0.0)
            push!(all_t0,    0.0)
            push!(all_dur,   0.0)
            push!(all_source, :ls)
        end
    end
    for (j, P_b) in enumerate(bls_cache.periods)
        used_bls[j] && continue
        push!(all_P, P_b)
        push!(all_w, bls_cache.weights[j])
        push!(all_K, 0.0)            # no LS K_est (peak not in LS)
        push!(all_Mo, 0.0)
        push!(all_depth, bls_cache.depths[j])
        push!(all_t0,    bls_cache.t0s[j])
        push!(all_dur,   bls_cache.durs[j])
        push!(all_source, :bls)
    end

    # No peaks at all → fall back to PriorBirth
    isempty(all_P) && return nothing

    # Renormalize weights and proceed with mixture sampling
    total_w = sum(all_w)
    total_w > 0 || (all_w = fill(1.0 / length(all_w), length(all_w));
                    total_w = 1.0)
    all_w ./= total_w
    # Transit-capable share floor: an unmodelled long-period RV planet (e.g.
    # WASP-47 c, K=31.6, period outside the slot window) floods the LS side
    # of the union with alias peaks and starves the transit peaks of proposal
    # mass — measured: e won ~20 cold multi-try selections per run vs ~30k
    # for d, while converting 45% of them. When BLS-backed peaks exist,
    # guarantee them at least half the informed mass. The SAME weights feed
    # `_informed_log_q`, so the draw and the Hastings density stay consistent.
    w_bls = 0.0
    for i in eachindex(all_w)
        (all_source[i] === :bls || all_source[i] === :both) && (w_bls += all_w[i])
    end
    if 0.0 < w_bls < 0.5
        s_b = 0.5 / w_bls
        s_l = 0.5 / (1.0 - w_bls)
        for i in eachindex(all_w)
            all_w[i] *= (all_source[i] === :bls || all_source[i] === :both) ?
                        s_b : s_l
        end
    end
    all_log_P = log.(all_P)

    # Per-peak proposal width. Transit peaks (:bls/:both) get a tight σ
    # matched to the (P,Tc) basin ~ transit_dur/baseline, so the refined
    # transit-precise period isn't blurred back out of the basin; RV peaks
    # (:ls) keep the wide default. The SAME vector is passed to
    # `_informed_log_q` so the Hastings ratio is consistent.
    t_lo, t_hi = extrema(data.t_phot); baseline = t_hi - t_lo
    all_sigma = Vector{Float64}(undef, length(all_P))
    for i in eachindex(all_P)
        # Basin HALF-width dur/(2T), not dur/T: the fold loses the transit when
        # the period is off by > dur/(2·baseline) in log-P, so σ = dur/T put
        # ~2/3 of draws OUTSIDE the basin — measured on WASP-47 e: median
        # ΔlogL of e-period proposals was −156 (worse than no planet at all).
        all_sigma[i] = (all_source[i] === :ls || all_dur[i] <= 0 || baseline <= 0) ?
            INFORMED_SIGMA_P :
            clamp(all_dur[i] / (2 * baseline), INFORMED_SIGMA_P_FLOOR, INFORMED_SIGMA_P)
    end

    # The peaks and their widths in the period slot's own variable.
    all_log_x = _slot_log.(Ref(m), all_log_P)
    all_sigma_x = _slot_sigmas(m, all_sigma)
    n = length(all_P)
    return _InformedSetup(m, Float64(log_x_min), Float64(log_x_max), all_log_x,
                          all_sigma_x, all_w, all_K, all_Mo, all_source,
                          all_depth, all_t0, all_dur)
end

"""
    propose_planet_birth(theta, rng, ::JointInformedBirth; data, scratch)
        -> (new_theta, log_q_ratio)

Joint LS+BLS informed birth. Builds a combined peak mixture by taking
the union of (a) RV-Lomb-Scargle peaks at the slot's period range and
(b) photometry-BLS peaks (with active-planet harmonic dedup), weighted
by `max(LS_norm, BLS_norm)`. Period sampling is α·informed + (1-α)·uniform
in log P. K is proposed from the LS K_est when LS dominates; from the
Chen & Kipping 2017 mass-radius relation applied to the BLS depth when
BLS dominates. Time anchor proposal uses the BLS t0 when available
(more accurate than LS phase for shallow transits) else falls through
to the LS Mo-derived path. The Hastings ratio carries the full mixture
density, and the paired death (`propose_planet_death(theta, rng,
::JointInformedBirth; data)`) the same density at the killed values.

Falls back to `InformedBirth` if no photometry is in `data`. Falls
back to `PriorBirth` if neither LS nor BLS produced any peaks.
"""
function propose_planet_birth(theta::Theta{T}, rng::AbstractRNG,
                               ::JointInformedBirth;
                               data::Union{Data, Nothing}=nothing,
                               population::Union{AbstractVector{<:Theta}, Nothing}=nothing,
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    # No photometry → no BLS contribution; degrade to LS-only InformedBirth.
    (data === nothing || isempty(data.t_phot)) &&
        return propose_planet_birth(theta, rng, InformedBirth();
                                     data=data, scratch=scratch)
    return _informed_birth(theta, rng, data, scratch, true)
end

"""
    propose_planet_death(theta, rng, ::JointInformedBirth; data, scratch)

Death paired with `JointInformedBirth` (see the InformedBirth death); without
photometry it is the InformedBirth death, as the birth is the InformedBirth one.
"""
function propose_planet_death(theta::Theta{T}, rng::AbstractRNG, ::JointInformedBirth;
                               data::Union{Data, Nothing}=nothing,
                               population=nothing,
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    (data === nothing || isempty(data.t_phot)) &&
        return propose_planet_death(theta, rng, InformedBirth(); data=data, scratch=scratch)
    return _informed_death(theta, rng, true, data, scratch)
end
