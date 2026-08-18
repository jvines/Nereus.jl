# Post-fit "silent-wrong" guard — assess_fit.
#
# The operational safety net for Nereus. The point is blunt: even a
# sampler we have NOT certified must not be able to hand a user a
# confident-but-wrong posterior with no flag attached. The validation
# matrix caught three real failure modes, all of which produced
# innocent-looking output:
#
#   (i)  nuts — two chains frozen in distinct modes, merged into a
#        credible interval that happens to bracket the truth. A "false
#        recovery": the marginal CI looks fine but it is not a
#        posterior, it is the union of two disjoint basins.
#   (ii) pt_whitening — tight CIs sitting at a wrong mode. Converged
#        by every per-chain metric, just wrong.
#   (iii) map — a parameter railed against a prior bound with
#        converged=true. The optimizer walked into the wall and
#        reported success.
#
# `assess_fit` runs four cheap checks against an MCMCChains.Chains and
# returns a FitHealthReport whose `overall` is the worst verdict. None
# of these checks can *prove* a fit is right — they can only catch the
# specific ways a fit is loudly, structurally broken while looking
# calm. Treat :ok as "no red flags", never as certification.
#
# Checks:
#   1. CONVERGENCE      — rank-normalized split-R-hat + min ESS
#                         (ensemble-aware: split each walker's own
#                          trace instead of treating walkers as chains).
#   2. MULTIMODALITY    — between-chain spread of per-chain medians vs.
#                         mean within-chain std. Large ratio ⇒ chains in
#                         disjoint modes ⇒ merged CI is not a posterior.
#                         The nuts / pt_whitening catcher.
#   3. PRIOR-EDGE RAIL  — posterior mass piled against a prior bound.
#                         The map / PA catcher.
#   4. LOGPOST SANITY   — absurd :lp / :log_density values. The
#                         corrupt-logZ catcher.

using MCMCChains
using Statistics: median, std, mean, quantile
using Printf: @printf

export FitHealthCheck, FitHealthReport, assess_fit

# =====================================================================
# Result types
# =====================================================================

"""
    FitHealthCheck

One diagnostic check result.

- `name`     : short check identifier (e.g. `:convergence`)
- `status`   : `:ok`, `:warn`, or `:fail`
- `message`  : human-readable explanation
- `details`  : optional per-parameter offenders (param => value)
"""
struct FitHealthCheck
    name::Symbol
    status::Symbol
    message::String
    details::Vector{Pair{String, Float64}}
end

FitHealthCheck(name, status, message) =
    FitHealthCheck(name, status, message, Pair{String, Float64}[])

"""
    FitHealthReport

Aggregate of the individual [`FitHealthCheck`](@ref)s produced by
[`assess_fit`](@ref).

- `overall`  : worst status across all checks (`:ok` < `:warn` < `:fail`)
- `checks`   : the individual checks, in evaluation order
- `messages` : flat list of each check's message (for logging)
"""
struct FitHealthReport
    overall::Symbol
    checks::Vector{FitHealthCheck}
    messages::Vector{String}
end

# =====================================================================
# Status helpers
# =====================================================================

# Severity ordering so we can take the worst.
const _STATUS_RANK = Dict(:ok => 0, :warn => 1, :fail => 2)

_worst(a::Symbol, b::Symbol) = _STATUS_RANK[a] >= _STATUS_RANK[b] ? a : b

function _worst(statuses)
    w = :ok
    for s in statuses
        w = _worst(w, s)
    end
    return w
end

_status_icon(s::Symbol) = s === :ok ? "✅" : s === :warn ? "⚠️ " : "❌"

# ChainDataFrame column indexing yields a scalar for single-parameter
# chains and a Vector otherwise; normalize to a Vector either way.
_asvec(x::AbstractVector) = x
_asvec(x) = [x]

# =====================================================================
# assess_fit
# =====================================================================

"""
    assess_fit(chains::MCMCChains.Chains; prior_bounds=nothing,
               ess_min::Int=200, rhat_max::Float64=1.05,
               ensemble::Bool=false, param_names=nothing,
               mode_ratio_max::Float64=3.0, edge_eps::Float64=0.01,
               edge_mass_frac::Float64=0.05, lp_floor::Float64=-1e6)
        -> FitHealthReport

Post-fit health screen. Runs four cheap structural checks against
`chains` and returns a [`FitHealthReport`](@ref) whose `overall` is the
worst verdict found. This is a *guard*, not a certificate: passing all
checks means no red flag was raised, not that the posterior is correct.

# Checks
1. **Convergence** — rank-normalized split-R-hat and minimum ESS over
   the fitted parameters. `:fail` if any R-hat > `rhat_max` or any
   ESS < `ess_min`. When `ensemble=true`, each walker's own trace is
   split in half (so a single ensemble of correlated walkers is still
   diagnosable) rather than treating walkers as independent chains.
2. **Multimodality / non-mixing** — for every parameter, compares the
   spread (std) of the per-chain medians against the mean within-chain
   std. A ratio above `mode_ratio_max` means the chains sit in disjoint
   basins; the merged marginal is then a union of modes, not a
   posterior. `:fail`. Needs ≥ 2 chains.
3. **Prior-edge rail** — with `prior_bounds` (a `Dict` mapping param
   name to `(lo, hi)`), flags any parameter whose posterior piles up
   against a bound: median within `edge_eps` (fraction of the bound
   span) of a bound, or more than `edge_mass_frac` of draws within that
   margin. `:fail`.
4. **Log-posterior sanity** — if an `:lp` / `:log_density` column
   exists, flags values below `lp_floor` (default `-1e6`) as corrupt.

# Arguments
- `prior_bounds` : `Dict{Any,Tuple}` param ⇒ `(lo, hi)`. Keys may be
  `Symbol` or `String`; matched against chain parameter names. If
  `nothing`, the rail check is skipped (`:ok`, "no prior_bounds given").
- `param_names` : restrict checks to this subset of parameters (names
  as `Symbol` or `String`). Defaults to all `:parameters`-section
  columns, excluding obvious bookkeeping columns (`:lp`, `:n_p`,
  `noise_active_*`, …).
- `ensemble` : ensemble-aware convergence (see check 1).

See also [`FitHealthReport`](@ref), [`FitHealthCheck`](@ref).
"""
function assess_fit(
    chains::MCMCChains.Chains;
    prior_bounds = nothing,
    ess_min::Int = 200,
    rhat_max::Float64 = 1.05,
    ensemble::Bool = false,
    param_names = nothing,
    mode_ratio_max::Float64 = 3.0,
    edge_eps::Float64 = 0.01,
    edge_mass_frac::Float64 = 0.05,
    lp_floor::Float64 = -1e6,
)
    # ----- which parameters to assess -------------------------------
    names_to_check = _resolve_param_names(chains, param_names)

    checks = FitHealthCheck[]
    push!(checks, _check_convergence(chains, names_to_check;
                                     ess_min = ess_min, rhat_max = rhat_max,
                                     ensemble = ensemble))
    push!(checks, _check_multimodality(chains, names_to_check;
                                       mode_ratio_max = mode_ratio_max))
    push!(checks, _check_prior_rail(chains, names_to_check, prior_bounds;
                                    edge_eps = edge_eps,
                                    edge_mass_frac = edge_mass_frac))
    push!(checks, _check_logpost(chains; lp_floor = lp_floor))

    overall = _worst(c.status for c in checks)
    messages = [c.message for c in checks]
    return FitHealthReport(overall, checks, messages)
end

"""
    assess_fit(map_point, chains::MCMCChains.Chains; nsigma=5.0, kwargs...)
        -> FitHealthReport

Convenience method: in addition to the standard chain checks, verify
that a MAP / point estimate is consistent with the posterior bulk. A
MAP that lands far outside the posterior body (> `nsigma` robust σ from
the per-parameter median on *any* checked parameter) is a sign the
optimizer found a spurious mode or railed against a bound while MCMC
explored a different basin.

`map_point` may be a `Vector{<:Real}` aligned with the checked
parameters, a `Dict` (param ⇒ value), or a [`MAPResult`](@ref) (its
`x_map` + `param_names` are used). Extra `kwargs` are forwarded to the
chain-only [`assess_fit`](@ref).
"""
function assess_fit(map_point, chains::MCMCChains.Chains;
                    nsigma::Float64 = 5.0,
                    param_names = nothing, kwargs...)
    base = assess_fit(chains; param_names = param_names, kwargs...)
    names_to_check = _resolve_param_names(chains, param_names)
    mapcheck = _check_map_consistency(map_point, chains, names_to_check;
                                      nsigma = nsigma)

    checks = vcat(base.checks, mapcheck)
    overall = _worst(c.status for c in checks)
    messages = [c.message for c in checks]
    return FitHealthReport(overall, checks, messages)
end

# =====================================================================
# Parameter-name resolution
# =====================================================================

# Bookkeeping columns we never want to convergence/rail-test directly.
const _SKIP_NAME_PREFIXES = ("noise_active", "noise_model")
const _SKIP_NAME_EXACT = (:lp, :log_density, :logp, :n_p, :Np, :weights, :weight)

function _is_bookkeeping(sym::Symbol)
    sym in _SKIP_NAME_EXACT && return true
    s = string(sym)
    for p in _SKIP_NAME_PREFIXES
        startswith(s, p) && return true
    end
    return false
end

_to_sym(x::Symbol) = x
_to_sym(x::AbstractString) = Symbol(x)

function _resolve_param_names(chains::MCMCChains.Chains, param_names)
    allnames = names(chains, :parameters)
    if param_names === nothing
        return [n for n in allnames if !_is_bookkeeping(n)]
    end
    want = Set(_to_sym(p) for p in param_names)
    present = Set(allnames)
    return [n for n in allnames if n in want && n in present]
end

# =====================================================================
# Check 1 — convergence (R-hat + ESS)
# =====================================================================

function _check_convergence(chains, names_to_check;
                            ess_min::Int, rhat_max::Float64,
                            ensemble::Bool)
    if isempty(names_to_check)
        return FitHealthCheck(:convergence, :warn,
            "no fitted parameters to assess for convergence")
    end

    # Subset to the parameters of interest.
    sub = chains[:, names_to_check, :]

    # Ensemble-aware reshaping: a single ensemble of correlated walkers
    # is not N independent chains, so treating walkers as chains gives a
    # misleadingly low R-hat. Instead split each walker's own trace in
    # half and stack the halves as the "chains" axis (rank-split-R-hat
    # applied within-walker). With multiple genuine chains this is still
    # valid and only sharpens the diagnostic.
    diag_chains = ensemble ? _ensemble_split(sub) : sub

    local ess_col, rhat_col, pnames
    try
        tbl = MCMCChains.ess_rhat(diag_chains; kind = :rank)
        # ChainDataFrame indexing returns a scalar when there is a single
        # parameter; coerce every column to a Vector so the loop below is
        # uniform.
        pnames = _asvec(tbl[:, :parameters])
        ess_col = _asvec(tbl[:, :ess])
        rhat_col = _asvec(tbl[:, :rhat])
    catch err
        return FitHealthCheck(:convergence, :warn,
            "could not compute ESS / R-hat ($(typeof(err))); " *
            "convergence not assessed")
    end

    bad = Pair{String, Float64}[]
    worst_rhat = -Inf
    min_ess = Inf
    n_fail = 0
    for i in eachindex(pnames)
        rh = rhat_col[i]
        es = ess_col[i]
        (rh isa Real && isfinite(rh)) && (worst_rhat = max(worst_rhat, rh))
        (es isa Real && isfinite(es)) && (min_ess = min(min_ess, es))
        rhat_bad = !(rh isa Real) || !isfinite(rh) || rh > rhat_max
        ess_bad = !(es isa Real) || !isfinite(es) || es < ess_min
        if rhat_bad || ess_bad
            n_fail += 1
            push!(bad, string(pnames[i]) =>
                  (rhat_bad ? Float64(rh isa Real ? rh : NaN) :
                              Float64(es isa Real ? es : NaN)))
        end
    end

    if n_fail == 0
        msg = "converged: all R-hat ≤ $(rhat_max) " *
              "(worst $(round(worst_rhat, digits=3))), " *
              "all ESS ≥ $(ess_min) (min $(round(min_ess, digits=0)))"
        return FitHealthCheck(:convergence, :ok, msg)
    else
        msg = "NOT converged: $(n_fail)/$(length(pnames)) params fail " *
              "R-hat > $(rhat_max) or ESS < $(ess_min) " *
              "(worst R-hat $(round(worst_rhat, digits=3)), " *
              "min ESS $(round(min_ess, digits=0)))"
        return FitHealthCheck(:convergence, :fail, msg, bad)
    end
end

# Split each walker's trace into two contiguous halves and lay the
# halves out along the chain axis. Returns a fresh Chains with the same
# parameter names and 2× the chains.
function _ensemble_split(chains::MCMCChains.Chains)
    arr = _chain_array_3d(chains)          # (n_iter, n_param, n_walker)
    n_iter, n_param, n_walker = size(arr)
    if n_iter < 4
        return chains   # too short to split meaningfully
    end
    h = div(n_iter, 2)
    out = Array{Float64}(undef, h, n_param, 2 * n_walker)
    for w in 1:n_walker
        @views out[:, :, 2w - 1] .= arr[1:h, :, w]
        @views out[:, :, 2w]     .= arr[(h + 1):(2h), :, w]
    end
    return MCMCChains.Chains(out, names(chains, :parameters))
end

# (n_iter, n_param, n_chain) dense array for a Chains object.
function _chain_array_3d(chains::MCMCChains.Chains)
    pnames = names(chains, :parameters)
    nchains = length(MCMCChains.chains(chains))
    # Array(chains[:param]) -> (n_iter, n_chain)
    first_mat = Array(chains[pnames[1]])
    n_iter = size(first_mat, 1)
    out = Array{Float64}(undef, n_iter, length(pnames), nchains)
    out[:, 1, :] .= first_mat
    for (j, p) in enumerate(pnames)
        j == 1 && continue
        out[:, j, :] .= Array(chains[p])
    end
    return out
end

# =====================================================================
# Check 2 — multimodality / non-mixing
# =====================================================================
#
# For each parameter: gather the median of every chain (or walker). The
# std of those per-chain medians measures how far apart the chains sit;
# the mean of the per-chain stds measures the typical within-chain
# spread. If the chains are exploring the same unimodal posterior the
# first should be small relative to the second. If they are frozen in
# distinct basins (the nuts / pt_whitening failure) the per-chain
# medians scatter far more than any single chain's width, and the ratio
# blows up. We flag the largest such ratio.

function _check_multimodality(chains, names_to_check; mode_ratio_max::Float64)
    nchains = length(MCMCChains.chains(chains))
    if nchains < 2
        return FitHealthCheck(:multimodality, :ok,
            "single chain — between-chain mode separation not assessable")
    end
    if isempty(names_to_check)
        return FitHealthCheck(:multimodality, :warn,
            "no fitted parameters to assess for multimodality")
    end

    offenders = Pair{String, Float64}[]
    worst_ratio = 0.0
    worst_name = ""
    for p in names_to_check
        mat = Array(chains[p])          # (n_iter, n_chain)
        size(mat, 2) < 2 && continue
        chain_medians = [median(@view mat[:, k]) for k in 1:size(mat, 2)]
        within_stds = [std(@view mat[:, k]) for k in 1:size(mat, 2)]
        between = std(chain_medians)             # spread of mode centres
        within = mean(within_stds)               # typical chain width

        # Degenerate / fixed parameter: no spread anywhere.
        (between == 0.0 && within == 0.0) && continue

        # within ~ 0 but chains differ → unambiguously disjoint.
        ratio = within <= 0.0 ? (between > 0.0 ? Inf : 0.0) : between / within

        if ratio > worst_ratio
            worst_ratio = ratio
            worst_name = string(p)
        end
        if ratio > mode_ratio_max
            push!(offenders, string(p) => ratio)
        end
    end

    if isempty(offenders)
        msg = "well mixed: between-chain / within-chain median spread " *
              "≤ $(mode_ratio_max) for all params " *
              "(worst $(round(worst_ratio, digits=2)) on $(worst_name))"
        return FitHealthCheck(:multimodality, :ok, msg)
    else
        msg = "chains in disjoint modes — merged CI is not a posterior: " *
              "$(length(offenders)) param(s) with between/within median " *
              "spread > $(mode_ratio_max) " *
              "(worst $(round(worst_ratio, digits=2)) on $(worst_name))"
        return FitHealthCheck(:multimodality, :fail, msg,
                              sort(offenders; by = x -> -x[2]))
    end
end

# =====================================================================
# Check 3 — prior-edge rail
# =====================================================================
#
# A converged-looking fit can still be wrong if a parameter is pinned
# against a prior bound — the optimizer (map) or a collapsed sampler
# (PA) walked into the wall and stopped. We flag a parameter if its
# posterior median sits within `edge_eps` (fraction of the bound span)
# of a finite bound, OR if more than `edge_mass_frac` of its draws are
# within that margin of a bound.

function _check_prior_rail(chains, names_to_check, prior_bounds;
                           edge_eps::Float64, edge_mass_frac::Float64)
    if prior_bounds === nothing
        return FitHealthCheck(:prior_rail, :ok,
            "no prior_bounds given — rail check skipped")
    end

    # Normalize keys to Symbol.
    bounds_sym = Dict{Symbol, Tuple{Float64, Float64}}()
    for (k, v) in prior_bounds
        lo, hi = float(v[1]), float(v[2])
        bounds_sym[_to_sym(k)] = (lo, hi)
    end

    offenders = Pair{String, Float64}[]   # param => fraction at edge (or NaN for median rail)
    rail_msgs = String[]
    present = Set(names(chains, :parameters))

    for p in names_to_check
        haskey(bounds_sym, p) || continue
        p in present || continue
        lo, hi = bounds_sym[p]
        (isfinite(lo) && isfinite(hi) && hi > lo) || continue
        span = hi - lo
        margin = edge_eps * span

        samp = vec(Array(chains[p]))
        isempty(samp) && continue
        med = median(samp)
        n = length(samp)
        n_lo = count(x -> (x - lo) <= margin, samp)
        n_hi = count(x -> (hi - x) <= margin, samp)
        frac_lo = n_lo / n
        frac_hi = n_hi / n

        med_rail_lo = (med - lo) <= margin
        med_rail_hi = (hi - med) <= margin

        if med_rail_lo || med_rail_hi
            which = med_rail_lo ? "lower" : "upper"
            push!(rail_msgs,
                  "$(p): median railed at $(which) bound " *
                  "(median $(round(med, sigdigits=4)), bound " *
                  "$(round(med_rail_lo ? lo : hi, sigdigits=4)))")
            push!(offenders, string(p) => (med_rail_lo ? frac_lo : frac_hi))
        elseif frac_lo > edge_mass_frac || frac_hi > edge_mass_frac
            f = max(frac_lo, frac_hi)
            which = frac_lo >= frac_hi ? "lower" : "upper"
            push!(rail_msgs,
                  "$(p): $(round(100f, digits=1))% of draws within " *
                  "$(round(100edge_eps, digits=1))% of $(which) bound")
            push!(offenders, string(p) => f)
        end
    end

    if isempty(offenders)
        return FitHealthCheck(:prior_rail, :ok,
            "no parameter railed against a prior bound")
    else
        msg = "prior-edge rail: " * join(rail_msgs, "; ")
        return FitHealthCheck(:prior_rail, :fail, msg,
                              sort(offenders; by = x -> -x[2]))
    end
end

# =====================================================================
# Check 4 — log-posterior sanity
# =====================================================================

function _check_logpost(chains; lp_floor::Float64)
    allnames = Set(names(chains))
    lp_col = nothing
    for cand in (:lp, :log_density, :logp)
        if cand in allnames
            lp_col = cand
            break
        end
    end
    if lp_col === nothing
        return FitHealthCheck(:logpost, :ok,
            "no :lp / :log_density column — log-posterior sanity not assessed")
    end

    lp = vec(Array(chains[lp_col]))
    isempty(lp) && return FitHealthCheck(:logpost, :ok,
        "log-posterior column empty — not assessed")

    n_nonfinite = count(!isfinite, lp)
    finite_lp = filter(isfinite, lp)
    n_absurd = count(x -> x < lp_floor, finite_lp)

    if n_nonfinite == 0 && n_absurd == 0
        msg = "log-posterior sane (range " *
              "$(round(minimum(finite_lp), sigdigits=4)) … " *
              "$(round(maximum(finite_lp), sigdigits=4)))"
        return FitHealthCheck(:logpost, :ok, msg)
    else
        msg = "corrupt log-posterior: $(n_nonfinite) non-finite, " *
              "$(n_absurd) below floor $(lp_floor)"
        return FitHealthCheck(:logpost, :fail, msg)
    end
end

# =====================================================================
# Optional MAP-in-bulk consistency
# =====================================================================

# Extract (names, values) from the supported map_point representations.
_map_pairs(m::Dict, _names) =
    [(_to_sym(k), float(v)) for (k, v) in m]

function _map_pairs(m::AbstractVector{<:Real}, names_to_check)
    length(m) == length(names_to_check) || error(
        "assess_fit: MAP vector length $(length(m)) ≠ number of checked " *
        "parameters $(length(names_to_check)); pass a Dict or matching vector")
    return [(names_to_check[i], float(m[i])) for i in eachindex(m)]
end

function _check_map_consistency(map_point, chains, names_to_check; nsigma::Float64)
    # MAPResult support without a hard type dependency in this file.
    pairs = if map_point isa Dict || map_point isa AbstractVector{<:Real}
        _map_pairs(map_point, names_to_check)
    elseif hasproperty(map_point, :x_map) && hasproperty(map_point, :param_names)
        [(_to_sym(n), float(v))
         for (n, v) in zip(getproperty(map_point, :param_names),
                           getproperty(map_point, :x_map))]
    else
        return FitHealthCheck(:map_consistency, :warn,
            "unrecognized MAP representation $(typeof(map_point)); skipped")
    end

    present = Set(names(chains, :parameters))
    offenders = Pair{String, Float64}[]
    worst_z = 0.0
    worst_name = ""
    n_checked = 0
    for (sym, val) in pairs
        sym in present || continue
        # Only assess parameters in the requested set (if restricted).
        isempty(names_to_check) || sym in names_to_check || continue
        samp = vec(Array(chains[sym]))
        isempty(samp) && continue
        n_checked += 1
        med = median(samp)
        # Robust σ from the 16/84 quantiles (half the central 68% width).
        q16 = quantile(samp, 0.16)
        q84 = quantile(samp, 0.84)
        rsig = (q84 - q16) / 2
        z = rsig > 0 ? abs(val - med) / rsig : (val == med ? 0.0 : Inf)
        if z > worst_z
            worst_z = z
            worst_name = string(sym)
        end
        z > nsigma && push!(offenders, string(sym) => z)
    end

    if n_checked == 0
        return FitHealthCheck(:map_consistency, :warn,
            "no MAP parameters matched chain columns; skipped")
    elseif isempty(offenders)
        msg = "MAP consistent with posterior bulk: all params within " *
              "$(nsigma)σ (worst $(round(worst_z, digits=2))σ on $(worst_name))"
        return FitHealthCheck(:map_consistency, :ok, msg)
    else
        msg = "MAP outside posterior bulk: $(length(offenders)) param(s) " *
              "> $(nsigma)σ from posterior median " *
              "(worst $(round(worst_z, digits=2))σ on $(worst_name)) — " *
              "MAP may be a spurious mode or railed bound"
        return FitHealthCheck(:map_consistency, :fail, msg,
                              sort(offenders; by = x -> -x[2]))
    end
end

# =====================================================================
# Pretty printing
# =====================================================================

function Base.show(io::IO, ::MIME"text/plain", r::FitHealthReport)
    println(io, "FitHealthReport")
    println(io, "  overall: $(_status_icon(r.overall)) $(uppercase(string(r.overall)))")
    println(io, "  checks:")
    for c in r.checks
        println(io, "    $(_status_icon(c.status)) [$(c.name)] $(c.message)")
        if !isempty(c.details)
            ndetail = min(length(c.details), 8)
            for k in 1:ndetail
                p = c.details[k]
                @printf(io, "        - %-30s %.4g\n", p.first, p.second)
            end
            if length(c.details) > ndetail
                println(io, "        … and $(length(c.details) - ndetail) more")
            end
        end
    end
    if r.overall === :fail
        println(io)
        println(io, "  " * "="^66)
        println(io, "  ❌  FIT HEALTH: FAIL — DO NOT TRUST THIS POSTERIOR AS-IS.")
        println(io, "      One or more structural checks failed. The reported")
        println(io, "      credible intervals may not be a valid posterior")
        println(io, "      (disjoint modes, railed bound, or corrupt log-post).")
        println(io, "      Inspect the failing check(s) above before using results.")
        println(io, "  " * "="^66)
    elseif r.overall === :warn
        println(io, "  ⚠️  FIT HEALTH: WARN — review the flagged check(s) above.")
    end
end

Base.show(io::IO, r::FitHealthReport) =
    print(io, "FitHealthReport($(_status_icon(r.overall)) ",
          uppercase(string(r.overall)), ", ", length(r.checks), " checks)")
