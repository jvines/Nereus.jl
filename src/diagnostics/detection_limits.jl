# Bayesian detection limits from a posterior chain.
#
# Slice the (P_k<i>, K_k<i>) marginal in log-spaced period bins; take
# the K upper credible limit per bin → K_lim(P) curve. The natural
# Bayesian counterpart to injection-recovery completeness: "given my
# data and priors, what K could I have detected at this period?"
#
# Only meaningful for chains that explore the full prior range in P —
# EMPEROR-style prior-seeded ptemcee is the canonical input. Other
# samplers (NUTS / NS / NF-PT) concentrate around modes and would
# produce biased limits. We don't gatekeep but warn when the chain
# coverage is narrow.

using Statistics: quantile

# =====================================================================
# Result struct
# =====================================================================

"""
    DetectionLimitResult

Output of [`detection_limits`](@ref). Carries the K upper credible
limit curve in log-spaced P bins, plus per-bin sample counts so the
caller can spot bins that were undersampled.

Fields:
- `P_centers::Vector{Float64}` — geometric midpoint of each bin (days)
- `P_edges::Vector{Float64}` — bin edges (`length = n_bins + 1`)
- `K_limit::Vector{Float64}` — K upper limit at `confidence` per bin (m/s); `NaN` for bins with too few draws
- `n_in_bin::Vector{Int}` — number of posterior draws in each bin
- `confidence::Float64` — quantile level used (e.g., `0.95`)
- `planet_index::Int` — planet slot the (P, K) marginal was taken from
- `prior_P_lo`, `prior_P_hi` — period prior bounds (for coverage diagnostics)
"""
struct DetectionLimitResult
    P_centers::Vector{Float64}
    P_edges::Vector{Float64}
    K_limit::Vector{Float64}
    n_in_bin::Vector{Int}
    confidence::Float64
    planet_index::Int
    prior_P_lo::Float64
    prior_P_hi::Float64
end

# =====================================================================
# Top-level API
# =====================================================================

"""
    detection_limits(chains, params;
                     planet=1, P_min=nothing, P_max=nothing,
                     n_bins=30, confidence=0.95,
                     min_samples_per_bin=10) -> DetectionLimitResult

Compute the Bayesian K upper-limit curve `K_lim(P)` from a posterior
chain. For each log-spaced P-bin, takes the `confidence`-quantile of
the K marginal as the detection limit. The interpretation is "given
my data and priors, anything with K above this curve at period P
would have been detected at the stated credible level."

The input chain must span the prior on P meaningfully for this to be
honest — EMPEROR-style prior-seeded ptemcee
([Vousden+ 2016](https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.1919V/abstract))
is the canonical input. NUTS / NS / NF-PT chains concentrate around
modes and would give biased limits; the function emits a warning when
the chain's P range covers less than 10% of the prior range.

Trans-dim chains are handled: draws where `n_planets < planet` are
masked out so only samples where this planet is active contribute.
"""
function detection_limits(chains, params::Params;
                           planet::Int = 1,
                           P_min::Union{Nothing, Real} = nothing,
                           P_max::Union{Nothing, Real} = nothing,
                           n_bins::Int = 30,
                           confidence::Real = 0.95,
                           min_samples_per_bin::Int = 10)
    planet >= 1 || throw(ArgumentError("planet must be ≥ 1"))
    0.5 < confidence < 1.0 ||
        throw(ArgumentError("confidence must be in (0.5, 1.0)"))
    n_bins >= 4 || throw(ArgumentError("n_bins must be ≥ 4"))

    P_name = "P_k$planet"
    K_name = "K_k$planet"

    chain_names = Set(string.(names(chains, :parameters)))
    P_name in chain_names ||
        throw(ArgumentError("chain has no column $P_name"))
    K_name in chain_names ||
        throw(ArgumentError("chain has no column $K_name"))

    P_vals = Float64.(vec(Array(chains[Symbol(P_name)])))
    K_vals = Float64.(vec(Array(chains[Symbol(K_name)])))

    # Trans-dim: mask inactive draws.
    if "n_planets" in chain_names
        np = Int.(round.(Float64.(vec(Array(chains[:n_planets])))))
        active = np .>= planet
        P_vals = P_vals[active]
        K_vals = K_vals[active]
    end

    isempty(P_vals) &&
        throw(ArgumentError("no posterior draws with planet $planet active"))

    # Period prior bounds — used as default bin range and for the
    # coverage warning.
    prior_P = _prior_for(params, P_name)
    lo_prior, hi_prior = prior_P === nothing ?
        (minimum(P_vals), maximum(P_vals)) :
        bounds(prior_P)

    P_min_eff = Float64(P_min === nothing ? lo_prior : P_min)
    P_max_eff = Float64(P_max === nothing ? hi_prior : P_max)
    (P_min_eff > 0 && P_max_eff > P_min_eff) ||
        throw(ArgumentError("require 0 < P_min < P_max"))

    # Coverage check.
    span_log_chain = log(maximum(P_vals) / minimum(P_vals))
    span_log_prior = log(hi_prior / lo_prior)
    if span_log_prior > 0 && span_log_chain / span_log_prior < 0.1
        @warn ("detection_limits: chain P range covers " *
                "$(round(100 * span_log_chain / span_log_prior; digits = 1))% " *
                "of the prior range — limits will only be honest inside that. " *
                "EMPEROR-style prior-seeded ptemcee is recommended.")
    end

    # Log-spaced bins.
    edges = exp.(range(log(P_min_eff), log(P_max_eff); length = n_bins + 1))
    centers = sqrt.(edges[1:end-1] .* edges[2:end])

    K_lim   = fill(NaN, n_bins)
    counts  = zeros(Int, n_bins)
    @inbounds for b in 1:n_bins
        lo, hi = edges[b], edges[b+1]
        # Right-open, except for the very last bin (closed on both ends).
        if b == n_bins
            mask = (P_vals .>= lo) .& (P_vals .<= hi)
        else
            mask = (P_vals .>= lo) .& (P_vals .< hi)
        end
        n = count(mask)
        counts[b] = n
        if n >= min_samples_per_bin
            K_lim[b] = quantile(K_vals[mask], Float64(confidence))
        end
    end

    return DetectionLimitResult(
        collect(Float64, centers), collect(Float64, edges),
        K_lim, counts, Float64(confidence), planet,
        Float64(lo_prior), Float64(hi_prior),
    )
end

# =====================================================================
# Helper
# =====================================================================

function _prior_for(params::Params, name::AbstractString)
    idx = findfirst(==(name), params.layout.unfrozen_names)
    idx === nothing && return nothing
    return params.layout.unfrozen_priors[idx]
end

# =====================================================================
# Detectability: what the data could actually have found
# =====================================================================
#
# `detection_limits` above reports posterior quantiles — a statement about
# what the data DO say. Detectability is the complementary question: at a
# given period and companion mass, could this data set have told a companion
# apart from pure noise at all? Each posterior draw already answers it: it is
# a trial companion with a period, a mass, an orientation and a phase, and its
# likelihood can be compared with the same model carrying NO companion. The
# fraction of draws in a (P, M) cell that clear the threshold is the detection
# probability there, with phase and inclination marginalised by the sampling.
#
# The mass axis is the TRUE mass when the planet block carries an inclination
# (astrometry present), and M sin i otherwise. That distinction is the whole
# point of adding astrometry: RV alone can never detect a face-on companion,
# however massive, so its true-mass detectability saturates while a joint fit's
# does not.

"""
    DetectabilityResult

Output of [`detectability`](@ref).

Fields:
- `P_centers`, `P_edges` — period bins (days)
- `M_centers`, `M_edges` — companion-mass bins (M_sun)
- `fraction::Matrix{Float64}` — detection probability per `(P, M)` cell,
  `NaN` where the chain put fewer than `min_samples` draws
- `M50::Vector{Float64}` — mass at which the probability crosses 0.5 in each
  period bin (M_sun); `NaN` if the crossing is outside the sampled masses
- `threshold::Float64` — Δχ² a draw must beat the no-companion model by
- `quantity::Symbol` — `:mass` (true mass) or `:msini`
- `loglike_null::Float64` — best no-companion log-likelihood found
- `planet_index::Int`
"""
struct DetectabilityResult
    P_centers::Vector{Float64}
    P_edges::Vector{Float64}
    M_centers::Vector{Float64}
    M_edges::Vector{Float64}
    fraction::Matrix{Float64}
    M50::Vector{Float64}
    threshold::Float64
    quantity::Symbol
    loglike_null::Float64
    planet_index::Int
end

_total_loglike(theta, data) =
    _rv_log_likelihood_core(theta, data) +
    (has_astrometry(data) ? astrom_log_likelihood(theta, data) : zero(eltype(theta.values)))

"""
    detectability(chains, params, data; planet=1, threshold=25.0,
                  n_P_bins=18, n_M_bins=28, min_samples=8,
                  P_min=nothing, P_max=nothing,
                  M_min=nothing, M_max=nothing) -> DetectabilityResult

Detection probability over the `(period, companion mass)` plane, computed from
the posterior draws and the likelihood of a no-companion ("pure noise") model.

A draw counts as detectable when `2 (lnL_draw − lnL_null) > threshold`; the
default `threshold = 25` is 5σ. `lnL_null` is the best likelihood obtained by
zeroing planet `planet`'s mass across the draws, which keeps every nuisance at
a value the chain actually visited.

The mass axis is the true mass where the block has an inclination and `M sin i`
otherwise; `res.quantity` records which. Mass-parametrised chains
(`M_sec_k…`) and `K`-parametrised chains are both accepted — `K` is converted
with [`msec_from_K`](@ref).

!!! note "What the map is, and is not"
    The reference is the best no-companion likelihood **among the draws'
    own nuisance values** — no re-optimisation. Two consequences worth
    keeping in mind:

    * On data that contain a real signal, the null is a poor fit by
      construction, so even small companions at the signal's period beat it.
      The map then shows that signal's footprint, which is a statement about
      THIS data set, not survey completeness. For completeness in the
      injection-recovery sense (Cumming et al. 2008) inject into residuals.
    * The null is only as good as the jitter and offsets the chain visited.
      A chain that never samples inflated jitter cannot absorb the signal
      into the noise model, which pushes the whole map up.
"""
function detectability(chains, params::Params, data;
                        planet::Int = 1,
                        threshold::Real = 25.0,
                        n_P_bins::Int = 18,
                        n_M_bins::Int = 28,
                        min_samples::Int = 8,
                        P_min::Union{Nothing, Real} = nothing,
                        P_max::Union{Nothing, Real} = nothing,
                        M_min::Union{Nothing, Real} = nothing,
                        M_max::Union{Nothing, Real} = nothing)
    planet >= 1 || throw(ArgumentError("planet must be ≥ 1"))
    threshold > 0 || throw(ArgumentError("threshold must be > 0"))
    n_P_bins >= 3 && n_M_bins >= 3 ||
        throw(ArgumentError("need at least 3 bins per axis"))

    names_ = params.layout.unfrozen_names
    # Pull every column BY NAME: MCMCChains keeps its own parameter ordering,
    # which need not match `layout.unfrozen_names`. Indexing positionally here
    # silently pairs each value with the wrong parameter.
    chain_names = Set(string.(names(chains, :parameters)))
    cols = Dict{String, Vector{Float64}}()
    for nm in names_
        s = string(nm)
        s in chain_names || continue
        cols[s] = Float64.(vec(Array(chains[Symbol(s)])))
    end
    isempty(cols) && throw(ArgumentError("no fitted parameter of `params` is in the chain"))
    n_draw = length(first(values(cols)))
    n_draw > 0 || throw(ArgumentError("chain has no draws"))

    P_name = "P_k$planet"
    M_name = "M_sec_k$planet"
    K_name = "K_k$planet"
    haskey(cols, P_name) || throw(ArgumentError("chain has no column $P_name"))
    (haskey(cols, M_name) || haskey(cols, K_name)) ||
        throw(ArgumentError("chain has neither $M_name nor $K_name"))
    amp_name = haskey(cols, M_name) ? M_name : K_name

    block = params.layout.planet_blocks[planet]
    has_inc = try
        _planet_inc(block, Theta(params)); true
    catch
        false
    end

    theta = Theta(params)
    apply! = (i, amp) -> begin
        for (s, v) in cols
            set_param!(theta, s, s == amp_name && amp !== nothing ? amp : v[i])
        end
    end

    # Per-draw likelihood, and the same draw with the companion removed.
    ll = Vector{Float64}(undef, n_draw)
    ll0 = Vector{Float64}(undef, n_draw)
    zero_amp = amp_name == K_name ? 1e-6 : 1e-12        # m/s, or M_sun
    mass = Vector{Float64}(undef, n_draw)
    for i in 1:n_draw
        apply!(i, nothing)
        ll[i] = _total_loglike(theta, data)
        e1, _ = planet_e_w(theta, planet)
        sin_i = has_inc ? sin(planet_inc(theta, planet)) : 1.0
        mass[i] = amp_name == K_name ?
            msec_from_K(cols[K_name][i], cols[P_name][i], e1, sin_i, params.config.M_s) :
            cols[M_name][i]
        apply!(i, zero_amp)                              # same nuisances, no companion
        ll0[i] = _total_loglike(theta, data)
    end
    loglike_null = maximum(ll0)

    # Mass axis: true mass, or M sin i when there is no inclination to use.
    quantity = has_inc ? :mass : :msini

    P = cols[P_name]
    P_lo = Float64(P_min === nothing ? minimum(P) : P_min)
    P_hi = Float64(P_max === nothing ? maximum(P) : P_max)
    M_lo = Float64(M_min === nothing ? max(minimum(mass), 1e-9) : M_min)
    M_hi = Float64(M_max === nothing ? maximum(mass) : M_max)
    # A chain can legitimately hold one period (period pinned by a transit
    # ephemeris) or one mass; widen by 1% rather than refusing to bin.
    P_hi > P_lo || ((P_lo, P_hi) = (0.99 * P_lo, 1.01 * P_hi))
    M_hi > M_lo || ((M_lo, M_hi) = (0.99 * M_lo, 1.01 * M_hi))
    (P_lo > 0 && M_lo > 0 && P_hi > P_lo && M_hi > M_lo) ||
        throw(ArgumentError("degenerate binning range: P ∈ [$P_lo, $P_hi], " *
                            "M ∈ [$M_lo, $M_hi] — need positive, non-empty ranges"))

    P_edges = exp.(range(log(P_lo), log(P_hi); length = n_P_bins + 1))
    M_edges = exp.(range(log(M_lo), log(M_hi); length = n_M_bins + 1))
    P_centers = sqrt.(P_edges[1:end-1] .* P_edges[2:end])
    M_centers = sqrt.(M_edges[1:end-1] .* M_edges[2:end])

    detected = @. 2 * (ll - loglike_null) > threshold
    fraction = fill(NaN, n_P_bins, n_M_bins)
    for a in 1:n_P_bins, b in 1:n_M_bins
        hiP = a == n_P_bins ? P .<= P_edges[a+1] : P .< P_edges[a+1]
        hiM = b == n_M_bins ? mass .<= M_edges[b+1] : mass .< M_edges[b+1]
        sel = (P .>= P_edges[a]) .& hiP .& (mass .>= M_edges[b]) .& hiM
        n = count(sel)
        n >= min_samples && (fraction[a, b] = count(detected .& sel) / n)
    end

    M50 = fill(NaN, n_P_bins)
    for a in 1:n_P_bins
        for b in 2:n_M_bins
            f1, f2 = fraction[a, b-1], fraction[a, b]
            (isfinite(f1) && isfinite(f2)) || continue
            if f1 < 0.5 <= f2
                w = (0.5 - f1) / (f2 - f1)
                M50[a] = exp((1 - w) * log(M_centers[b-1]) + w * log(M_centers[b]))
                break
            end
        end
    end

    return DetectabilityResult(collect(P_centers), collect(P_edges),
                               collect(M_centers), collect(M_edges),
                               fraction, M50, Float64(threshold), quantity,
                               loglike_null, planet)
end
