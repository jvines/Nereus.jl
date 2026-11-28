# Reporting on the planet count k of a trans-dimensional run: the prior the
# sampler puts on k, ln B between adjacent k, the mixture evidence, an occupancy
# health check, the accepted k transitions, and the k_max truncation warning.
#
# Reporting only. Nothing here touches a chain, a proposal or an acceptance
# rule; every function reads the draws of `n_planets` (and the move counters)
# that a run already produced.

"""A k with at least this posterior mass is held to the occupancy health check
(`transdim_occupancy_health`); below it the k is reported but not gated."""
const TRANSDIM_K_MIN_MASS = 0.01
"""Fewest effective samples of a k's occupancy indicator for ln B to be quoted."""
const TRANSDIM_K_MIN_NEFF = 25.0
"""Fewest walkers that must have visited a k with `TRANSDIM_K_MIN_MASS`."""
const TRANSDIM_K_MIN_WALKERS = 3
"""Fewest separate visits (entries into k, summed over walkers) for a k that
does not hold the whole posterior."""
const TRANSDIM_K_MIN_VISITS = 10
"""P(k = k_max) above this says the prior's upper bound on k is truncating."""
const TRANSDIM_KMAX_WARN_P = 0.5

"""
    transdim_k_prior(max_k; inclusion_prior = 0.5, group_sizes = [max_k],
                      slab_fraction = 1.0) -> Vector{Float64}

The prior on the planet count k = 0, 1, …, `max_k` that a trans-dimensional run
puts on the k-planet models, indexed `k + 1`: P(k) ∝ (p / (1 − p))^k / k!, with
`p = inclusion_prior`, truncated at `max_k` and normalised.

Where it comes from. Every planet slot carries an independent Bernoulli(p)
activation bit and an iid slab prior on its orbit. Active planets of one planet
mode are kept in period order (hard gate in the likelihood, births insert into
the order), so which slot holds a planet carries no information and the chain
lives on the period-ordered region of the k-planet parameter space. That region
holds 1/k! of the slab prior's mass and the sampler does not renormalise it, so
the k-planet model enters the posterior with the weight (p/(1−p))^k / k!
relative to its evidence Z_k, the evidence under the normalised (unordered)
slab prior: P(k | D) ∝ P(k) Z_k. It is NOT uniform at the default p = 0.5, and
not binomial. (Measured with a likelihood that carries no information, max_k = 3:
the occupancy ratios k+1 : k are in the ratio 1 : 1/2 : 1/3 at p = 0.5 and each is
multiplied by p/(1−p) at p = 0.3, as this formula says.) Equivalently, P(k | D)
is uniform-prior at p = 0.5 over the evidences Z_k / k! of the period-ordered
model, which is what a fixed-k fit with the ordering gate integrates.

`slab_fraction` c is the share of one planet's slab prior that the model
allows: the default (√e sin ω, √e cos ω) parametrisation draws both uniformly
in [-1, 1] and the likelihood rejects e ≥ 1, so c = π/4 there. The k-planet model's
evidence under the slab prior normalised over its allowed region is the box-prior
evidence divided by c^k, so each planet carries an extra c in the prior:
P(k) ∝ (c p/(1−p))^k / k!. (This is the factor of about 0.75 per planet the
prior-only run showed.) `transdim_slab_fraction(params)` computes it by Monte Carlo.

`group_sizes` are the numbers of slots of each planet mode (ordering is per
mode, so each group carries its own 1/k_g!); the default is one group. `rjmcmc`
has no activation prior, which is the same as `inclusion_prior = 0.5`.
"""
function transdim_k_prior(max_k::Integer; inclusion_prior::Real = 0.5,
                          group_sizes::AbstractVector{<:Integer} = [Int(max_k)],
                          slab_fraction::Real = 1.0)
    0 < slab_fraction <= 1 || throw(ArgumentError("slab_fraction must be in (0, 1]"))
    max_k >= 0 || throw(ArgumentError("max_k must be ≥ 0"))
    0 < inclusion_prior < 1 || throw(ArgumentError("inclusion_prior must be in (0, 1)"))
    sum(group_sizes) == max_k || throw(ArgumentError(
        "group_sizes $(collect(group_sizes)) must sum to max_k = $max_k"))
    r = slab_fraction * inclusion_prior / (1 - inclusion_prior)
    poly = [1.0]                                        # Π_g Σ_j r^j / j!
    for n in group_sizes
        c = [r^j / factorial(j) for j in 0:Int(n)]
        out = zeros(length(poly) + length(c) - 1)
        for (i, a) in enumerate(poly), (j, b) in enumerate(c)
            out[i + j - 1] += a * b
        end
        poly = out
    end
    return poly ./ sum(poly)
end

# (iteration × walker) matrix of the planet count, from the flat draws of a
# chain. A native multi-chain layout has chain as the SLOW index of `_draws`; the
# flat single-chain layout of the walker ensembles has WALKER as the FAST index
# (see `_convergence_stats`).
function _k_by_walker(np::AbstractVector{<:Integer}, nchains::Int,
                      n_walkers::Union{Nothing, Integer})
    n = length(np)
    if nchains > 1
        return reshape(np, n ÷ nchains, nchains)
    elseif n_walkers !== nothing && n_walkers > 1 && n % n_walkers == 0
        return permutedims(reshape(np, Int(n_walkers), n ÷ Int(n_walkers)))
    end
    return reshape(np, n, 1)
end

# Effective sample size of a 0/1 occupancy indicator, (iteration × walker).
function _indicator_ess(ind::AbstractMatrix{<:Real})
    (all(==(0), ind) || all(==(1), ind)) && return Float64(length(ind))
    size(ind, 1) < 4 && return Float64(length(ind))
    try
        e = _MCDT.ess(reshape(Float64.(ind), size(ind, 1), size(ind, 2), 1); kind = :basic)
        v = e isa Real ? Float64(e) : Float64(first(e))
        return isfinite(v) && v > 0 ? min(v, Float64(length(ind))) : NaN
    catch
        return NaN
    end
end

# Number of separate visits to k: entries into k from another count, plus one
# for every walker that starts there.
function _k_visits(M::AbstractMatrix{<:Integer}, k::Int)
    v = 0
    for w in 1:size(M, 2)
        M[1, w] == k && (v += 1)
        for i in 2:size(M, 1)
            (M[i, w] == k && M[i - 1, w] != k) && (v += 1)
        end
    end
    return v
end

"""
    transdim_k_report(np, max_k; nchains = 1, n_walkers = nothing,
                      inclusion_prior = 0.5, planets = true,
                      group_sizes = [max_k], slab_fraction = 1.0, log_evidence = NaN, evidence_reliable = true,
                      evidence_method = "") -> Dict{String,Any}

The planet-count block of a trans-dim run, from the flat draws `np` of
`n_planets`:

- `prior`: the prior on k (`transdim_k_prior`), its form and `inclusion_prior`;
- `occupancy` and `per_k`: P(k | D) and, per k, the draws, effective samples
  of the occupancy indicator, walkers that visited it and separate visits;
- `ln_bayes_factor_adjacent`: ln B(k+1 vs k) = ln [P(k+1|D) / P(k|D)] − ln
  [P(k+1) / P(k)], with its standard error (delta method on the occupancy
  indicators, effective samples as above). A k with too few draws to estimate
  it (fewer than `TRANSDIM_K_MIN_NEFF` effective samples, or none) gives
  `estimable = false`, a `reason`, and no number where there is no draw;
- `mixture_evidence`: ln Z of the whole trans-dim model, marginalised over k
  with the prior above. It is the engine's own log evidence (a tempered or
  nested estimator integrates against the normalised prior, here the
  k-prior times the slab priors) and is only available when the engine reports
  one; `ln_z_k` then gives ln Z_k = ln Z + ln P(k|D) − ln P(k) per k;
- `k_max_truncation`: P(k = k_max) and whether it exceeds
  `TRANSDIM_KMAX_WARN_P`;
- `occupancy_health` (`transdim_occupancy_health`).
"""
function transdim_k_report(np::AbstractVector{<:Integer}, max_k::Integer;
                           nchains::Integer = 1,
                           n_walkers::Union{Nothing, Integer} = nothing,
                           inclusion_prior::Real = 0.5, planets::Bool = true,
                           group_sizes::AbstractVector{<:Integer} = [Int(max_k)],
                           slab_fraction::Real = 1.0,
                           log_evidence::Real = NaN, evidence_reliable::Bool = true,
                           evidence_method::AbstractString = "")
    K = Int(max_k)
    M = _k_by_walker(np, Int(nchains), n_walkers)
    n = length(M)
    prior = transdim_k_prior(K; inclusion_prior, group_sizes, slab_fraction)
    p = [count(==(k), M) / n for k in 0:K]
    out = Dict{String, Any}("max_k" => K, "n_draws" => n, "n_walkers" => size(M, 2))
    out["prior"] = Dict{String, Any}(
        "form" => "P(k) ∝ (c p/(1-p))^k / k! on k = 0..max_k (per planet mode: product of the groups)",
        "group_sizes" => collect(Int, group_sizes),
        "slab_fraction" => Float64(slab_fraction),
        "meaning" => "the weight of the k-planet model against its evidence Z_k under the " *
                     "normalised slab prior: P(k|D) ∝ P(k) Z_k. The 1/k! is the mass of the " *
                     "period-ordered region the sampler lives on; at p = 0.5 P(k|D) is uniform-prior " *
                     "over Z_k/k!, the evidences of the period-ordered model",
        "inclusion_prior" => Float64(inclusion_prior),
        "p_k" => prior,
        "ordering" => "planets period-ordered; births insert into period order")
    out["occupancy"] = Dict(string(k) => p[k + 1] for k in 0:K)

    per_k = Dict{String, Any}()
    ess = fill(NaN, K + 1)         # effective size of each occupancy indicator
    neff = fill(NaN, K + 1)        # effective DRAWS in k: p_k × that size
    for k in 0:K
        ind = M .== k
        cnt = count(ind)
        ess[k + 1] = cnt == 0 ? 0.0 : _indicator_ess(ind)
        neff[k + 1] = p[k + 1] * ess[k + 1]
        per_k[string(k)] = Dict{String, Any}(
            "p" => p[k + 1], "n_draws" => cnt, "n_eff" => neff[k + 1],
            "ess_indicator" => ess[k + 1],
            "walkers_visiting" => count(w -> any(==(k), view(M, :, w)), 1:size(M, 2)),
            "n_visits" => cnt == 0 ? 0 : _k_visits(M, k))
    end
    out["per_k"] = per_k

    # ln B between adjacent k
    rows = Dict{String, Any}[]
    for k in 0:(K - 1)
        a, b = p[k + 2], p[k + 1]                     # k+1, k
        row = Dict{String, Any}("k_hi" => k + 1, "k_lo" => k,
                                "ln_prior_ratio" => log(prior[k + 2] / prior[k + 1]))
        if a == 0 || b == 0
            row["ln_bayes_factor"] = NaN
            row["se"] = NaN
            row["estimable"] = false
            row["reason"] = "no draws at k = $(a == 0 ? k + 1 : k): the posterior " *
                            "occupancy of that count is below 1/$(n); run longer " *
                            "or read ln B as a bound from the other side"
        else
            lnB = log(a / b) - row["ln_prior_ratio"]
            ne = min(neff[k + 2], neff[k + 1])
            es = min(ess[k + 2], ess[k + 1])
            # delta method on the two occupancies: var ln p̂ = (1 - p) / (p ESS),
            # and the multinomial covariance of the two adds 2 / ESS.
            se = isfinite(ne) && ne > 0 && isfinite(es) && es > 0 ?
                 sqrt((1 - a) / neff[k + 2] + (1 - b) / neff[k + 1] + 2 / es) : NaN
            row["ln_bayes_factor"] = lnB
            row["se"] = se
            ok = isfinite(ne) && ne >= TRANSDIM_K_MIN_NEFF
            row["estimable"] = ok
            ok || (row["reason"] = "k = $(neff[k + 2] <= neff[k + 1] ? k + 1 : k) " *
                   "has $(round(ne, digits = 1)) effective draws " *
                   "(< $(Int(TRANSDIM_K_MIN_NEFF))): too few to estimate ln B")
        end
        push!(rows, row)
    end
    out["ln_bayes_factor_adjacent"] = rows

    # mixture evidence
    me = Dict{String, Any}("what" => "ln Z of the whole trans-dim model, " *
                           "marginalised over k with the prior above")
    if isfinite(log_evidence)
        me["available"] = true
        me["log_z"] = Float64(log_evidence)
        me["method"] = isempty(evidence_method) ? "the engine's own evidence estimator" :
                       String(evidence_method)
        me["how"] = "the engine's log Z integrates the likelihood against the " *
                    "normalised prior (P(k) times the slab priors), so it is the " *
                    "mixture evidence; ln Z_k = ln Z + ln P(k|D) − ln P(k)"
        me["evidence_reliable"] = evidence_reliable
        lz = Dict{String, Any}()
        for k in 0:K
            p[k + 1] > 0 || continue
            lz[string(k)] = Float64(log_evidence) + log(p[k + 1]) - log(prior[k + 1])
        end
        me["ln_z_k"] = lz
    else
        me["available"] = false
        me["log_z"] = NaN
        me["reason"] = "the engine reports no quotable evidence (a tempered " *
                       "estimator needs a ladder that reaches β = 0 and swaps " *
                       "≥ 5%; rjmcmc and moms have no estimator); ln B between " *
                       "adjacent k above still holds"
    end
    out["mixture_evidence"] = me

    # k_max truncation
    pk = p[K + 1]
    trunc = Dict{String, Any}("k_max" => K, "p_k_max" => pk,
                              "warn" => planets && K > 0 && pk > TRANSDIM_KMAX_WARN_P)
    if trunc["warn"]
        trunc["advice"] = "raise k_max (transdim.max_kplanet / " *
                          "TransDimConfig(max_kplanet = …) and the model's " *
                          "max_kplanet) and fit again"
    end
    out["k_max_truncation"] = trunc

    # k changes seen between consecutive recorded draws of a walker: all an engine
    # with no move counters (rjmcmc, moms, daedalus) leaves to read, and a lower
    # bound on the accepted births and deaths of the others (a thinned chain and
    # several moves per step hide some).
    dM = diff(M; dims = 1)
    out["transitions_observed"] = Dict{String, Any}(
        "up" => count(>(0), dM), "down" => count(<(0), dM),
        "net_moves_per_draw" => n > size(M, 2) ? (count(!=(0), dM) / (n - size(M, 2))) : NaN,
        "note" => "k changes between consecutive recorded draws of a walker (births and " *
                  "deaths seen in the chain, a lower bound on the accepted moves)")
    out["occupancy_health"] = transdim_occupancy_health(per_k, size(M, 2))
    return out
end

"""
    transdim_occupancy_health(per_k, n_walkers) -> Dict

Every k with posterior mass of at least `TRANSDIM_K_MIN_MASS` must have been
visited by at least `min(TRANSDIM_K_MIN_WALKERS, n_walkers)` walkers, and unless
it holds the bulk of the posterior (mass ≥ 0.9) by at least
`TRANSDIM_K_MIN_VISITS` separate visits: otherwise its mass rests on one or two
excursions and is not an estimate. `ok` is false with the offending ks in
`problems` when a k fails.
"""
function transdim_occupancy_health(per_k::AbstractDict, n_walkers::Integer)
    need_w = min(TRANSDIM_K_MIN_WALKERS, Int(n_walkers))
    problems = String[]
    for k in sort(parse.(Int, collect(keys(per_k))))
        d = per_k[string(k)]
        d["p"] >= TRANSDIM_K_MIN_MASS || continue
        if d["walkers_visiting"] < need_w
            push!(problems, "k = $k (P = $(round(d["p"], digits = 3))) was visited " *
                  "by $(d["walkers_visiting"]) walker(s), fewer than $need_w")
        elseif d["p"] < 0.9 && d["n_visits"] < TRANSDIM_K_MIN_VISITS
            push!(problems, "k = $k (P = $(round(d["p"], digits = 3))) has " *
                  "$(d["n_visits"]) separate visit(s), fewer than " *
                  "$TRANSDIM_K_MIN_VISITS")
        end
    end
    return Dict{String, Any}("ok" => isempty(problems),
        "status" => isempty(problems) ? "ok" : "fail", "problems" => problems,
        "rule" => "each k with P ≥ $(TRANSDIM_K_MIN_MASS): ≥ $need_w walkers visited " *
                  "and, below P = 0.9, ≥ $TRANSDIM_K_MIN_VISITS separate visits")
end

"""
    transdim_move_counts(result) -> Union{Nothing, Dict}

Accepted k transitions of a trans-dim run from the move counters its result
carries, per move type: planet births, planet deaths (summed over the ladder
and at the β = 1 rung), noise-model toggles and swaps, and, where recorded,
the planet↔noise swap. `nothing` for a result with no such counters.
"""
function transdim_move_counts(result)
    result === nothing && return nothing
    g(f) = hasproperty(result, f) ? getproperty(result, f) : nothing
    mk(prop, acc) = begin
        (prop isa AbstractVector && acc isa AbstractVector && !isempty(prop)) || return nothing
        P, A = Int.(prop), Int.(acc)
        return Dict{String, Any}("proposed" => sum(P), "accepted" => sum(A),
            "rate" => sum(P) > 0 ? sum(A) / sum(P) : NaN,
            "cold_proposed" => P[1], "cold_accepted" => A[1],
            "cold_rate" => P[1] > 0 ? A[1] / P[1] : NaN)
    end
    out = Dict{String, Any}()
    b = mk(g(:planet_birth_proposed), g(:planet_birth_accepted))
    d = mk(g(:planet_death_proposed), g(:planet_death_accepted))
    n = mk(g(:noise_td_proposed), g(:noise_td_accepted))
    b === nothing || (out["birth"] = b)
    d === nothing || (out["death"] = d)
    n === nothing || (out["noise_toggle_swap"] = n)
    sw = g(:planet_noise_swap)
    sw isa AbstractDict && (out["planet_noise_swap"] = Dict{String, Any}(string(k) => v for (k, v) in sw))
    isempty(out) && return nothing
    out["note"] = "per-temperature counters summed over the ladder; cold_* is the β = 1 rung " *
                  "(index 1 of the ladder)"
    return out
end

# The user-facing warnings, one place, so every route words them the same way.
function _warn_transdim_k(rep::AbstractDict; engine::AbstractString = "")
    tr = rep["k_max_truncation"]
    if tr["warn"]
        @warn "Nereus: P(k = k_max = $(tr["k_max"])) = $(round(tr["p_k_max"], digits = 3)) " *
              "> $TRANSDIM_KMAX_WARN_P: the prior's upper bound on the planet count is " *
              "likely truncating the posterior. Raise k_max (transdim `max_kplanet` and the " *
              "model's `max_kplanet`) and fit again." p_k_max = tr["p_k_max"]
    end
    h = rep["occupancy_health"]
    h["ok"] || @warn "Nereus: trans-dim occupancy health = FAIL: " * join(h["problems"], "; ") *
                     ". The occupancy of those planet counts is not an estimate; run more " *
                     "steps or more walkers."
    for r in rep["ln_bayes_factor_adjacent"]
        (r["estimable"] || !isfinite(r["ln_bayes_factor"])) && continue
        @warn "Nereus: ln B(k = $(r["k_hi"]) vs $(r["k_lo"])) is not quotable: $(r["reason"])."
    end
    return nothing
end

"""
    transdim_slab_fraction(params; n = 20_000, seed = 1) -> Float64

The share c of one planet's slab prior that the model allows (the `slab_fraction`
of [`transdim_k_prior`](@ref)), by Monte Carlo over the first planet slot's
(√e sin ω, √e cos ω) priors: the likelihood rejects e ≥ 1, so uniform draws on
[-1, 1]² keep π/4. `1.0` when the slot is not parametrised that way.
"""
function transdim_slab_fraction(params; n::Int = 20_000, seed::Integer = 1)
    try
        lay = params.layout
        (haskey(lay.name_to_idx, "sesinw_k1") && haskey(lay.name_to_idx, "secosw_k1")) || return 1.0
        pos(nm) = findfirst(==(lay.name_to_idx[nm]), lay.unfrozen_idx)
        a, b = pos("sesinw_k1"), pos("secosw_k1")
        (a === nothing || b === nothing) && return 1.0
        rng = MersenneTwister(seed)
        da, db = lay.unfrozen_priors[a].dist, lay.unfrozen_priors[b].dist
        ok = count(_ -> rand(rng, da)^2 + rand(rng, db)^2 < 1, 1:n)
        return ok / n
    catch
        return 1.0
    end
end
