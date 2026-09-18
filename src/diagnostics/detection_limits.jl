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
