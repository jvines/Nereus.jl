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

export FitHealthCheck, FitHealthReport, assess_fit, assess_run_summary, RUN_HEALTH_THRESHOLDS

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
               edge_mass_frac::Float64=0.05, lp_floor::Float64=-1e6,
               circular=nothing, log_scale=nothing, active=nothing,
               jitter=nothing)
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
   margin. `:fail`. Parameters in `log_scale` are measured on
   `log(x + s)` — bounds, margin and draws alike — so the 1% is 1% of a
   log-uniform prior's decades, not of its linear span. Parameters named
   in `circular` are full-circle
   angles: their draws are first relabelled into the `prior_bounds`
   window, which should be the chart the ENGINE sampled in. A pile-up
   there means the engine could not carry the angle across that seam and
   the draws beyond it are missing; an engine that moved its seam off the
   posterior shows none. Parameters named in `jitter` are never flagged at
   their LOWER bound: a jitter can be zero, so a posterior there is a
   value, not a rail. The check stays `:ok` and its message names them.
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
- `circular` : names (`Symbol` or `String`) of full-circle angles —
  Uniform over exactly one period, e.g. `circular_names(params)`.
  Measured in their `prior_bounds` window by the rail check (see
  check 3). `nothing` treats every parameter as linear.
- `log_scale` : `Dict` param ⇒ shift `s`, for parameters whose prior is
  flat in `log(x + s)` — `0` for LogUniform, the knee for ModJeffreys; see
  [`log_scale_shift`](@ref). Unlisted parameters are measured linearly.
- `active` : `Dict` param ⇒ Bool vector over the flattened draws, for
  trans-dim parameters that exist only where their planet or noise model is
  active. Every check then uses only those draws: in the other draws the
  column holds a parked value, not a posterior draw, and assessing it failed
  sound trans-dim runs on convergence and rails. Masked convergence is per
  parameter, the active draws in order split in half.
- `jitter` : names of white-noise jitter terms, e.g. `jitter_names(params)`
  (see [`jitter_names`](@ref)). Their lower bound is exempt from the rail
  check (see check 3); their upper bound is not.
- `angle_range` : `Dict` param ⇒ the full range of an angle whose prior
  covers only part of it (`"[0, 2π)"` for a full-circle angle, `"[0, π]"` for
  an inclination), e.g. `restricted_angles(params)`. A rail of one of these is
  still a rail, and the message adds that the bound is the prior's own
  restriction.

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
    circular = nothing,
    log_scale = nothing,
    active = nothing,
    jitter = nothing,
    angle_range = nothing,
)
    # ----- which parameters to assess -------------------------------
    names_to_check = _resolve_param_names(chains, param_names)
    masks = _active_masks(chains, active)

    checks = FitHealthCheck[]
    push!(checks, _check_convergence(chains, names_to_check;
                                     ess_min = ess_min, rhat_max = rhat_max,
                                     ensemble = ensemble, active = masks))
    push!(checks, _check_multimodality(chains, names_to_check;
                                       mode_ratio_max = mode_ratio_max,
                                       active = masks))
    push!(checks, _check_prior_rail(chains, names_to_check, prior_bounds;
                                    edge_eps = edge_eps,
                                    edge_mass_frac = edge_mass_frac,
                                    circular = circular,
                                    log_scale = log_scale, active = masks,
                                    angle_range = angle_range,
                                    jitter = jitter))
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
chain-only [`assess_fit`](@ref). `circular` names full-circle angles
(period 2π): they are exempt from the rail check, and the MAP distance
for them is the shortest arc, since the optimizer may return a different
representative of the same angle than the chart the chains are in.
"""
function assess_fit(map_point, chains::MCMCChains.Chains;
                    nsigma::Float64 = 5.0,
                    param_names = nothing, circular = nothing, kwargs...)
    base = assess_fit(chains; param_names = param_names, circular = circular,
                      kwargs...)
    names_to_check = _resolve_param_names(chains, param_names)
    mapcheck = _check_map_consistency(map_point, chains, names_to_check;
                                      nsigma = nsigma, circular = circular)

    checks = vcat(base.checks, mapcheck)
    overall = _worst(c.status for c in checks)
    messages = [c.message for c in checks]
    return FitHealthReport(overall, checks, messages)
end

# =====================================================================
# Parameter-name resolution
# =====================================================================

# Bookkeeping columns we never want to convergence/rail-test directly. The
# trans-dim indicators are among them: a component present in every draw leaves
# its indicator constant, and a constant column has no R-hat.
const _SKIP_NAME_PREFIXES = ("noise_active", "noise_model", "planet_active")
const _SKIP_NAME_EXACT = (:lp, :log_density, :logp, :n_p, :Np, :n_planets, :weights, :weight)

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

# `active` (param ⇒ Bool mask over the flattened draws, in `_draws(chains, p)`
# order) as Dict{Symbol,BitVector}. A mask that does not match the chain length
# is a caller bug: silently ignoring it would assess junk as if it were draws.
function _active_masks(chains::MCMCChains.Chains, active)
    out = Dict{Symbol, BitVector}()
    active === nothing && return out
    n = size(chains, 1) * size(chains, 3)
    for (k, v) in active
        length(v) == n || throw(ArgumentError(
            "assess_fit: active mask for $(k) has $(length(v)) entries, " *
            "the chain has $n draws"))
        out[_to_sym(k)] = BitVector(v)
    end
    return out
end

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
                            ensemble::Bool,
                            active::Dict{Symbol, BitVector} = Dict{Symbol, BitVector}())
    if isempty(names_to_check)
        return FitHealthCheck(:convergence, :warn,
            "no fitted parameters to assess for convergence")
    end

    # Masked (trans-dim) parameters are assessed one at a time on their active
    # draws, in draw order, split in half; the rest together as before.
    plain  = [p for p in names_to_check if !haskey(active, p)]
    masked = [p for p in names_to_check if haskey(active, p)]

    local ess_col, rhat_col, pnames
    try
        pnames, ess_col, rhat_col = Any[], Any[], Any[]
        if !isempty(plain)
            # Ensemble-aware reshaping: a single ensemble of correlated walkers
            # is not N independent chains, so treating walkers as chains gives a
            # misleadingly low R-hat. Instead split each walker's own trace in
            # half and stack the halves as the "chains" axis (rank-split-R-hat
            # applied within-walker). With multiple genuine chains this is still
            # valid and only sharpens the diagnostic. The `:ensemble` layout of
            # `_rank_diag_batch` is that split, read in place from the chain.
            inpar = haskey(chains.name_map, :parameters) &&
                    all(p -> p in chains.name_map[:parameters], plain)
            r = inpar ? _rank_diag_chain(chains, plain, ensemble ? :ensemble : :split2) :
                        nothing
            if r === nothing
                sub = chains[:, plain, :]
                diag_chains = ensemble ? _ensemble_split(sub) : sub
                tbl = MCMCChains.ess_rhat(diag_chains; kind = :rank)
                # ChainDataFrame indexing returns a scalar when there is a single
                # parameter; coerce every column to a Vector so the loop below is
                # uniform.
                append!(pnames, _asvec(tbl[:, :parameters]))
                append!(ess_col, _asvec(tbl[:, :ess]))
                append!(rhat_col, _asvec(tbl[:, :rhat]))
            else
                append!(pnames, plain)
                append!(ess_col, [x.ess_bulk for x in r])
                append!(rhat_col, [x.rhat_rank for x in r])
            end
        end
        # Masked parameters: each one's active draws, built when its turn comes.
        mp = [p for p in masked if count(active[p]) >= 4]
        if !isempty(mp)
            ntot = size(chains, 1) * size(chains, 3)
            rm = _rank_diag_batch(
                [() -> reshape(Vector{Float64}(_draws(chains, p)[active[p]]), :, 1)
                 for p in mp], :ensemble; n_max = ntot, src_bytes = 16)
            for (p, x) in zip(mp, rm)
                push!(pnames, p)
                push!(ess_col, x.ess_bulk)
                push!(rhat_col, x.rhat_rank)
            end
        end
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

function _check_multimodality(chains, names_to_check; mode_ratio_max::Float64,
                              active::Dict{Symbol, BitVector} = Dict{Symbol, BitVector}())
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
    cube = chains.value.data
    pos = Dict(n => i for (i, n) in enumerate(names(chains)))
    for p in names_to_check
        # (n_iter, n_chain), read in place: copying it cost three copies of the
        # column per parameter at millions of draws.
        mat = eltype(cube) === Float64 && haskey(pos, p) ?
              view(cube, :, pos[p], :) : Array(chains[p])
        size(mat, 2) < 2 && continue
        n_it = size(mat, 1)
        msk = get(active, p, nothing)
        # Each chain's own active draws (all of them for an unmasked param),
        # one chain at a time through one buffer. The std first, in draw order,
        # then the median in place: the numbers `std` and `median` give on a
        # copy of the column.
        buf = eltype(mat)[]
        chain_medians = Float64[]
        within_stds = Float64[]
        for k in 1:size(mat, 2)
            empty!(buf)
            off = (k - 1) * n_it
            for i in 1:n_it
                (msk === nothing || msk[off + i]) && push!(buf, mat[i, k])
            end
            length(buf) >= 2 || continue
            push!(within_stds, std(buf))
            push!(chain_medians, median!(buf))
        end
        length(chain_medians) < 2 && continue
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
#
# "Span" is on the scale the prior is flat on (`log_scale`). Measured linearly,
# 1% of the default LogUniform(0.1, 3000) period prior is 30 d, and a clean
# posterior at P = 12.3 d was reported as "railed at lower bound 0.1".
#
# Full-circle angles (`circular`) are NOT exempt, they are measured in the chart
# the engine sampled in: `prior_bounds` carries that window (taken before
# recenter_circular! moved it for reporting) and the draws are relabelled into
# it first. A seam is not an edge of the support, but an engine that cannot
# cross one truncates the posterior there just as surely. Measured on a
# posterior centred on Mo = 0: NUTS (logit chart) returned only the sliver
# below 2π and nested sampling (unit cube) only the part above 0 -- both
# honestly flagged by this check before the seam work, and both would have
# passed silently had circular params simply been exempted. An engine that moved
# its seam to the emptiest arc shows no pile-up there and passes.
#
# A jitter (`jitter`, see `jitter_names`) piled at its LOWER bound is exempt: a
# jitter can be zero -- the reported errors may carry all the scatter -- so that
# is a value, never a pathology. It is named in the `:ok` message instead. Its
# upper bound is still checked: a jitter railing high is a real noise excess
# the prior truncated.

_sym_set(::Nothing) = Set{Symbol}()
_sym_set(xs) = Set{Symbol}(_to_sym(x) for x in xs)

function _check_prior_rail(chains, names_to_check, prior_bounds;
                           edge_eps::Float64, edge_mass_frac::Float64,
                           circular = nothing, log_scale = nothing,
                           active::Dict{Symbol, BitVector} = Dict{Symbol, BitVector}(),
                           jitter = nothing, angle_range = nothing)
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
    shifts = Dict{Symbol, Float64}()
    log_scale === nothing || for (k, v) in log_scale
        v === nothing || (shifts[_to_sym(k)] = float(v))
    end

    offenders = Pair{String, Float64}[]   # param => fraction at edge (or NaN for median rail)
    rail_msgs = String[]
    present = Set(names(chains, :parameters))
    circ = _sym_set(circular)
    jit = _sym_set(jitter)
    seam_msgs = String[]
    at_zero = String[]                    # jitters consistent with zero
    # Angles whose prior covers only part of their range (`angle_range`:
    # name => the full range, "[0, 2π)" or "[0, π]"). A rail there is the
    # user's own restriction, and the message says so.
    arange = Dict{Symbol, String}()
    angle_range === nothing || for (k, v) in angle_range
        arange[_to_sym(k)] = String(v)
    end
    restricted = String[]

    for p in names_to_check
        haskey(bounds_sym, p) || continue
        p in present || continue
        lo0, hi0 = bounds_sym[p]
        (isfinite(lo0) && isfinite(hi0) && hi0 > lo0) || continue

        samp = _draws(chains, p)
        haskey(active, p) && (samp = samp[active[p]])
        isempty(samp) && continue
        is_circ = p in circ
        is_circ && (samp = [circular_relabel(x, lo0, hi0) for x in samp])
        med0 = median(samp)
        # Measure on the scale the prior is flat on (see `log_scale_shift`).
        s = is_circ ? nothing : get(shifts, p, nothing)
        (s === nothing || lo0 + s > 0) || (s = nothing)
        lo, hi, med = s === nothing ? (lo0, hi0, med0) :
                      (log(lo0 + s), log(hi0 + s), log(med0 + s))
        s === nothing || (samp = [log(max(x + s, lo0 + s)) for x in samp])
        margin = edge_eps * (hi - lo)
        n = length(samp)
        n_lo = count(x -> (x - lo) <= margin, samp)
        n_hi = count(x -> (hi - x) <= margin, samp)
        frac_lo = n_lo / n
        frac_hi = n_hi / n

        med_rail_lo = (med - lo) <= margin
        med_rail_hi = (hi - med) <= margin

        if p in jit && !is_circ && (med_rail_lo || frac_lo > edge_mass_frac)
            push!(at_zero, string(p))
            med_rail_lo = false
            frac_lo = 0.0
        end

        if is_circ && (med_rail_lo || med_rail_hi ||
                       frac_lo > edge_mass_frac || frac_hi > edge_mass_frac)
            f = max(frac_lo, frac_hi)
            push!(seam_msgs,
                  "$(p): $(round(100f, digits=1))% of draws within " *
                  "$(round(100edge_eps, digits=1))% of the seam of the window " *
                  "the sampler used [$(round(lo0, sigdigits=4)), " *
                  "$(round(hi0, sigdigits=4))]")
            push!(offenders, string(p) => f)
        elseif med_rail_lo || med_rail_hi
            which = med_rail_lo ? "lower" : "upper"
            push!(rail_msgs,
                  "$(p): median railed at $(which) bound " *
                  "(median $(round(med0, sigdigits=4)), bound " *
                  "$(round(med_rail_lo ? lo0 : hi0, sigdigits=4)))")
            push!(offenders, string(p) => (med_rail_lo ? frac_lo : frac_hi))
            haskey(arange, p) && push!(restricted, "$(p) (full range $(arange[p]))")
        elseif frac_lo > edge_mass_frac || frac_hi > edge_mass_frac
            f = max(frac_lo, frac_hi)
            which = frac_lo >= frac_hi ? "lower" : "upper"
            push!(rail_msgs,
                  "$(p): $(round(100f, digits=1))% of draws within " *
                  "$(round(100edge_eps, digits=1))% of $(which) bound" *
                  (s === nothing ? "" : " (in log space)"))
            push!(offenders, string(p) => f)
            haskey(arange, p) && push!(restricted, "$(p) (full range $(arange[p]))")
        end
    end

    if isempty(offenders)
        return FitHealthCheck(:prior_rail, :ok,
            "no parameter railed against a prior bound" *
            (isempty(at_zero) ? "" : "; jitter at zero (a jitter can be zero; " *
                                     "not a rail): " * join(at_zero, ", ")))
    else
        parts = String[]
        isempty(rail_msgs) || push!(parts, "prior-edge rail: " * join(rail_msgs, "; "))
        isempty(restricted) || push!(parts,
            "angle held by its own prior: " * join(restricted, ", ") *
            " — the prior covers only part of this angle's range and the " *
            "posterior presses against that restriction, so the bound is the " *
            "prior's, not the data's. Widen the prior to the full range unless the " *
            "restriction is deliberate")
        isempty(seam_msgs) || push!(parts,
            "full-circle angle cut by its seam: " * join(seam_msgs, "; ") *
            " — 0 ≡ 2π is not an edge, but this engine could not carry the " *
            "angle across it, so the draws on the far side are missing. Use an " *
            "engine that moves the seam (pt_emcee, pt_whitening, " *
            "transdim_pt_emcee, pt) or write the prior window so the posterior " *
            "is interior (e.g. U(-π, π) for an angle near 0)")
        msg = join(parts, "; ")
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

    lp = _draws(chains, lp_col)
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

function _check_map_consistency(map_point, chains, names_to_check; nsigma::Float64,
                                circular = nothing)
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
    circ = _sym_set(circular)
    offenders = Pair{String, Float64}[]
    worst_z = 0.0
    worst_name = ""
    n_checked = 0
    for (sym, val) in pairs
        sym in present || continue
        # Only assess parameters in the requested set (if restricted).
        isempty(names_to_check) || sym in names_to_check || continue
        samp = _draws(chains, sym)
        isempty(samp) && continue
        n_checked += 1
        med = median(samp)
        # Robust σ from the 16/84 quantiles (half the central 68% width).
        q16, q84 = quantile(samp, (0.16, 0.84))     # one sort for both
        rsig = (q84 - q16) / 2
        # A full-circle angle: the MAP may sit in another representative of the
        # same angle (6.26 against a chain charted around 0.0), and the linear
        # distance would call that a 100σ miss. Shortest arc instead.
        d = sym in circ ? rem2pi(val - med, RoundNearest) : val - med
        z = rsig > 0 ? abs(d) / rsig : (d == 0 ? 0.0 : Inf)
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
# Run-summary checks — a cold chain trapped in a wrong mode
# =====================================================================
#
# The chain checks above cannot see a population that collapsed into ONE wrong
# basin: it is converged, well mixed between walkers and off every bound.
# HIP 63762 (nereus I, 2026-11-24; pt_emcee, RV + Hipparcos/Gaia astrometry, two
# Keplerians): k2_ast_ms passed every chain check with overall "ok" while its
# HIRES jitter sat at 109 m/s on 1.1 m/s errors (1.4 m/s in the right mode), the
# inner K at half its value, and ln Z ~50-70 nats below the nested one-planet
# fit. What gave it away was in the rest of summary.json: the sampler's
# acceptance, the evidence error and the per-instrument noise. These checks read
# those blocks. They only ever WARN -- each sign has innocent causes too -- and
# a block that is missing is reported as not assessed.

"""
    RUN_HEALTH_THRESHOLDS

Defaults of [`assess_run_summary`](@ref), each with the HIP 63762 numbers it was
set against (trapped runs k2_ast_ms/v4/v5; healthy controls k1_v3, k1lin_v3,
k1quad_v3, k2_v3, k1_ast_v5 of the same star):

- `cold_acceptance_min = 0.01` — within-move acceptance of the β = 1 rung.
  An affine-invariant / MH cold chain under 1% barely moves: its autocorrelation
  ESS can still look large while the population stays where it started.
  Healthy controls: 0.017-0.24.
- `swap_min = 0.01` — the lowest adjacent-rung swap acceptance. Below it the hot
  rungs cannot hand states down to β = 1, and the ladder no longer rescues a
  trapped cold chain. Trapped: 6e-6 (ms), 7e-3 (v4); healthy ≥ 0.077.
- `frozen_rung_max = 1e-3` — a rung whose within-move acceptance is below this
  is frozen. Trapped (ms): 21 of 32 rungs at ~1e-5; healthy min 0.002.
- `evidence_se_max = 1.0` nat — the largest error of a quotable evidence
  estimate (bridge, nested, the tempered stack). Trapped: bridge ±4.3, ±4.7,
  ±23; healthy ≤ ±0.7.
- `evidence_disagree_min = 3.0` nats and `evidence_disagree_nsigma = 3.0` — two
  independent estimators (bridge, nested, any tempered one; the four tempered
  variants share one ⟨ln L⟩(β) and are not compared with each other) disagree
  when they differ by more than the larger of the two.
- `laplace_bridge_max = 20.0` nats — |mode-Laplace − bridge|. Mode-Laplace is a
  Gaussian at the mode, so some offset is normal (healthy 3-11.5 nats);
  trapped: 30, 72, 160.
- `jitter_err_ratio_max = 30.0` — an instrument's median added jitter over its
  median formal error. Trapped: HIRES 109/1.1 ≈ 100, 279/1.1 ≈ 250; right
  mode 1.4/1.1. Above it either the residuals are many times the errors and the
  jitter absorbs them (a misfit hidden in the noise model), or the jitter also
  exceeds `jitter_rms_ratio_max = 2.0` × the residual RMS and that
  instrument's data are effectively ignored. A genuinely active star can carry
  such a jitter on precise data; the warning says to check.
"""
const RUN_HEALTH_THRESHOLDS = (
    cold_acceptance_min      = 0.01,
    swap_min                 = 0.01,
    frozen_rung_max          = 1e-3,
    evidence_se_max          = 1.0,
    evidence_disagree_min    = 3.0,
    evidence_disagree_nsigma = 3.0,
    laplace_bridge_max       = 20.0,
    jitter_err_ratio_max     = 30.0,
    jitter_rms_ratio_max     = 2.0,
)

const _TRAPPED_HINT = "the signature of a cold chain trapped in a wrong mode: " *
    "compare ln Z with a nested simpler model and rerun from another initialisation"

# summary.json blocks may carry String or Symbol keys (JSON3 objects, Dicts).
function _sget(d, k)
    d isa AbstractDict || return nothing
    v = get(d, k, nothing)
    v === nothing && (v = get(d, Symbol(k), nothing))
    return v
end

_finite_vec(v) = v isa AbstractVector ?
    [Float64(x) for x in v if x isa Real && isfinite(x)] : Float64[]

_r3(x) = round(x; sigdigits = 3)

"""
    assess_run_summary(summary; thresholds...) -> Vector{FitHealthCheck}

Checks on a run's `summary.json` blocks (`sampler`, `evidence`, `ppc`) for the
signs of a cold chain trapped in a wrong mode, which the chain checks of
[`assess_fit`](@ref) cannot see. Returns four checks, each `:ok` or `:warn`,
never `:fail`:

- `:cold_acceptance` — within-move acceptance of the β = 1 rung below
  `cold_acceptance_min`.
- `:tempering_ladder` — a swap acceptance below `swap_min`, or rungs frozen at
  within-move acceptance below `frozen_rung_max`.
- `:evidence_precision` — a quotable evidence estimate with error above
  `evidence_se_max`, two independent estimators disagreeing, or mode-Laplace
  more than `laplace_bridge_max` from bridge.
- `:rv_jitter` — an instrument whose median jitter exceeds
  `jitter_err_ratio_max` × its median formal error (details: instrument ⇒ ratio).

Keyword arguments override [`RUN_HEALTH_THRESHOLDS`](@ref), where each default
and the runs it was set against are documented. run_job merges the result into
`summary["fit_health"]` after the evidence stage.
"""
function assess_run_summary(summary::AbstractDict; kw...)
    th = merge(RUN_HEALTH_THRESHOLDS, values(kw))
    sampler = _sget(summary, "sampler")
    sampler isa AbstractDict || (sampler = _sget(summary, "diagnostics"))
    within = _finite_vec(_sget(sampler, "acceptance_within"))
    swap = _finite_vec(_sget(sampler, "acceptance_swap"))
    return FitHealthCheck[
        _check_cold_acceptance(within; th.cold_acceptance_min),
        _check_tempering_ladder(within, swap; th.swap_min, th.frozen_rung_max),
        _check_evidence_precision(_sget(summary, "evidence"); th.evidence_se_max,
                                  th.evidence_disagree_min, th.evidence_disagree_nsigma,
                                  th.laplace_bridge_max),
        _check_rv_jitter(_sget(summary, "ppc"); th.jitter_err_ratio_max,
                         th.jitter_rms_ratio_max),
    ]
end

function _check_cold_acceptance(within; cold_acceptance_min)
    isempty(within) && return FitHealthCheck(:cold_acceptance, :ok,
        "no within-move acceptance recorded — not assessed")
    a = within[1]
    a >= cold_acceptance_min && return FitHealthCheck(:cold_acceptance, :ok,
        "cold-chain within-move acceptance $(_r3(a)) ≥ $(cold_acceptance_min)")
    return FitHealthCheck(:cold_acceptance, :warn,
        "cold-chain within-move acceptance $(_r3(a)) < $(cold_acceptance_min): " *
        "the β = 1 population barely moves, so its ESS and R-hat describe where " *
        "it started — $(_TRAPPED_HINT)", ["acceptance_within[β=1]" => a])
end

function _check_tempering_ladder(within, swap; swap_min, frozen_rung_max)
    (isempty(swap) && length(within) < 2) && return FitHealthCheck(:tempering_ladder,
        :ok, "single-temperature run — ladder not assessed")
    problems = String[]
    details = Pair{String, Float64}[]
    if !isempty(swap)
        i = argmin(swap)
        if swap[i] < swap_min
            push!(problems, "swap acceptance falls to $(_r3(swap[i])) between rungs " *
                  "$i and $(i + 1) (< $(swap_min)): states from the hot rungs " *
                  "cannot reach β = 1")
            push!(details, "min_swap[$i↔$(i + 1)]" => swap[i])
        end
    end
    if length(within) >= 2
        frozen = findall(<(frozen_rung_max), within)
        if !isempty(frozen)
            push!(problems, "$(length(frozen)) of $(length(within)) rungs frozen " *
                  "(within-move acceptance < $(frozen_rung_max), lowest " *
                  "$(_r3(minimum(within[frozen]))))")
            push!(details, "n_frozen_rungs" => Float64(length(frozen)))
        end
    end
    isempty(problems) && return FitHealthCheck(:tempering_ladder, :ok,
        "ladder connected: min swap " *
        (isempty(swap) ? "n/a" : string(_r3(minimum(swap)))) *
        ", no frozen rung")
    return FitHealthCheck(:tempering_ladder, :warn,
        "tempering ladder broken — " * join(problems, "; ") *
        ". The ladder cannot rescue a trapped cold chain; $(_TRAPPED_HINT)", details)
end

# (log_z, se) of an evidence entry: a Dict with "log_z" and "se"/"sigma", or a
# bare number (bridge before its bootstrap), whose error is unknown.
function _ev_entry(v)
    v isa Real && return isfinite(v) ? (Float64(v), NaN) : nothing
    v isa AbstractDict || return nothing
    lz = _sget(v, "log_z")
    (lz isa Real && isfinite(lz)) || return nothing
    se = _sget(v, "se"); se === nothing && (se = _sget(v, "sigma"))
    return (Float64(lz), se isa Real && isfinite(se) ? Float64(se) : NaN)
end

const _TEMPERED_KEYS = ("ti", "ti_plus", "ss_plus", "hybrid")

function _check_evidence_precision(ev; evidence_se_max, evidence_disagree_min,
                                   evidence_disagree_nsigma, laplace_bridge_max)
    ev isa AbstractDict || return FitHealthCheck(:evidence_precision, :ok,
        "no evidence block — not assessed")
    est = Pair{String, Tuple{Float64, Float64}}[]
    for k in ("bridge", "nested", _TEMPERED_KEYS...)
        e = _ev_entry(_sget(ev, k))
        e === nothing || push!(est, k => e)
    end
    problems = String[]
    details = Pair{String, Float64}[]
    for (k, (_, se)) in est
        if isfinite(se) && se > evidence_se_max
            push!(problems, "$k ln Z error ±$(_r3(se)) nats (> $(evidence_se_max))")
            push!(details, "$(k)_se" => se)
        end
    end
    for a in 1:length(est), b in (a + 1):length(est)
        ka, (za, sa) = est[a]; kb, (zb, sb) = est[b]
        (ka in _TEMPERED_KEYS && kb in _TEMPERED_KEYS) && continue
        s = sqrt((isfinite(sa) ? sa^2 : 0.0) + (isfinite(sb) ? sb^2 : 0.0))
        tol = max(evidence_disagree_min, evidence_disagree_nsigma * s)
        d = za - zb
        if abs(d) > tol
            push!(problems, "$ka and $kb disagree by $(_r3(abs(d))) nats " *
                  "(tolerance $(_r3(tol)))")
            push!(details, "$(ka)_minus_$(kb)" => d)
        end
    end
    ml = _sget(ev, "mode_laplace")
    br = _ev_entry(_sget(ev, "bridge"))
    if ml isa Real && isfinite(ml) && br !== nothing
        d = Float64(ml) - br[1]
        if abs(d) > laplace_bridge_max
            push!(problems, "mode-Laplace sits $(_r3(d)) nats from bridge " *
                  "(|Δ| > $(laplace_bridge_max))")
            push!(details, "laplace_minus_bridge" => d)
        end
    end
    if isempty(problems)
        isempty(est) && return FitHealthCheck(:evidence_precision, :ok,
            "no quotable evidence estimate — not assessed")
        return FitHealthCheck(:evidence_precision, :ok,
            "evidence consistent: " * join(("$k $(_r3(z))" *
                (isfinite(s) ? " ± $(_r3(s))" : "") for (k, (z, s)) in est), ", "))
    end
    return FitHealthCheck(:evidence_precision, :warn,
        "evidence poorly determined — " * join(problems, "; ") *
        ". The posterior the estimators see is not one well-sampled mode; " *
        _TRAPPED_HINT, details)
end

function _check_rv_jitter(ppc; jitter_err_ratio_max, jitter_rms_ratio_max)
    jit = _sget(ppc, "rv_jitter_median_per_inst")
    err = _sget(ppc, "rv_err_formal_median_per_inst")
    (jit isa AbstractDict && err isa AbstractDict) ||
        return FitHealthCheck(:rv_jitter, :ok,
            "no per-instrument RV noise in the PPC block — not assessed")
    rms = _sget(ppc, "rv_resid_rms_per_inst")
    chi2f = _sget(ppc, "rv_red_chi2_formal_per_inst")
    problems = String[]
    details = Pair{String, Float64}[]
    worst = (0.0, "")
    for (k, j) in jit
        inst = String(k)
        e = _sget(err, inst)
        (j isa Real && e isa Real && e > 0 && isfinite(j)) || continue
        ratio = j / e
        ratio > worst[1] && (worst = (ratio, inst))
        ratio > jitter_err_ratio_max || continue
        r = _sget(rms, inst)
        c = _sget(chi2f, inst)
        why = if r isa Real && r > 0 && j > jitter_rms_ratio_max * r
            "and $(_r3(j / r))× its residual RMS $(_r3(r)): its data are " *
            "effectively ignored"
        else
            "absorbing residuals" * (r isa Real ? " of RMS $(_r3(r))" : "") *
            (c isa Real ? " (χ²/dof $(_r3(c)) on the formal errors)" : "")
        end
        push!(problems, "$inst jitter $(_r3(j)) is $(_r3(ratio))× its median " *
              "formal error $(_r3(e)), $why")
        push!(details, inst => ratio)
    end
    isempty(problems) && return FitHealthCheck(:rv_jitter, :ok,
        isempty(worst[2]) ? "no instrument with jitter and formal errors — not assessed" :
        "jitter ≤ $(jitter_err_ratio_max)× the formal errors on every instrument " *
        "(worst $(_r3(worst[1]))× on $(worst[2]))")
    return FitHealthCheck(:rv_jitter, :warn,
        "jitter dwarfs the formal RV errors — " * join(problems, "; ") *
        ". A very active star can do this; otherwise it is " * _TRAPPED_HINT,
        sort(details; by = x -> -x.second))
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
