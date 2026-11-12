# The error of a nested-sampling log Z (Higson et al. 2018, 2019; Speagle 2020).
#
# Between seeds a run's log Z scatters for two reasons. The prior volumes its
# dead points stand for are known only in distribution: each shrinkage
# t = X_k / X_{k-1}, at a point removed with n live points, is Beta(n, 1). And
# the points are one draw of the run's path, which an imperfect proposal (a
# chain of finite length from a live point) correlates. sqrt(H / n_live)
# counts the first alone, under perfect sampling; on the four test fits (two RM
# obliquity fits, an eccentric RV planet, a gravity-darkened transit) the
# scatter between seeds was 1.2-1.5 times it, for every engine.
#
# The error quoted here is dynesty's (`simulate_run`): over many realisations,
# resample the run's threads with replacement, recount the live points at
# every point of the resampled run, draw every shrinkage afresh, and take the
# spread of log Z. A thread is the sequence of points one live-point slot
# held; each point is live from the likelihood bound it was drawn above (its
# birth: -Inf for a draw from the prior) to its own likelihood.

using Random: AbstractRNG, MersenneTwister, randexp
using LogExpFunctions: logaddexp
using Statistics: std

# Realisations behind an error: its relative error is 1/sqrt(2·200) = 5 %.
const _NS_ERR_SIMS = 200

"""
    _ns_floor(L) -> Float64

A log-likelihood as the trace keeps it: `-Inf` (an invalid point) as
`-floatmax`, so that `-Inf` stays the birth of a draw from the prior alone,
below every point, and a run's points at an invalid level and the
replacements drawn above them are counted as any other tie
(`_ns_live_counts`). Such points weigh nothing either way.
"""
_ns_floor(L::Real) = max(Float64(L), -floatmax(Float64))

"""
    _NSTrace

Every point a nested run removed, the live points it ended with included:
its log-likelihood, its birth (the bound it was drawn above: `-Inf` from the
prior, else a log-likelihood through `_ns_floor`) and its thread (the
live-point slot it filled).
"""
struct _NSTrace
    logL::Vector{Float64}
    birth::Vector{Float64}
    thread::Vector{Int}
    _NSTrace(logL, birth, thread) = new(_ns_floor.(logL), Vector{Float64}(birth),
                                        Vector{Int}(thread))
end

_NSTrace() = _NSTrace(Float64[], Float64[], Int[])

function _ns_trace_push!(tr::_NSTrace, logL::Real, birth::Real, thread::Integer)
    push!(tr.logL, _ns_floor(logL))
    push!(tr.birth, birth)
    push!(tr.thread, thread)
    return tr
end

"""
    _ns_threads(tr, group) -> (members, by_group)

`members[k]`: the points of the trace's `k`-th thread; `by_group`: the
threads of each group, `group(thread_label)` naming it (all one group by
default).
"""
function _ns_threads(tr::_NSTrace, group)
    labels = unique(tr.thread)
    pos = Dict(t => k for (k, t) in enumerate(labels))
    members = [Int[] for _ in labels]
    for i in eachindex(tr.thread)
        push!(members[pos[tr.thread[i]]], i)
    end
    g = [group(t) for t in labels]
    by_group = [findall(==(v), g) for v in unique(g)]
    return members, by_group
end

"""
    _ns_live_counts(L, birth) -> Vector{Int}

The live points at each point of a run, `L` its log-likelihoods in increasing
order and `birth` its points' births, sorted: those born below its likelihood,
less the points removed so far. Of the births at its own likelihood, as many
count as the points at that likelihood removed before it: a replacement is
drawn above the point it replaces, and ties are a walk that never moved (a
copy of its start), points at an invalid level, a batch's replacements
(drawn above its highest point) and a resampled run's repeated threads.
"""
function _ns_live_counts(L::AbstractVector{<:Real}, birth::AbstractVector{<:Real})
    n = Vector{Int}(undef, length(L))
    nb = length(birth)
    j = 0      # births below the current level
    c = 0      # births at it
    k0 = 1     # its first point
    for k in eachindex(L)
        if k == firstindex(L) || L[k] != L[k - 1]
            k0 = k
            while j < nb && birth[j + 1] < L[k]
                j += 1
            end
            c = 0
            while j + c < nb && birth[j + c + 1] == L[k]
                c += 1
            end
        end
        n[k] = max(j + min(c, k - k0) - (k - 1), 1)
    end
    return n
end

"""
    _ns_logz_sim(rng, tr, members, by_group; resample = true, jitter = true) -> Float64

One realisation of a run's log Z. `resample`: draw each group's threads with
replacement, as many as it has, and recount the live points
(`_ns_live_counts`). `jitter`: each shrinkage drawn from Beta(n, 1); else
`n / (n + 1)`, its mean, which with `resample = false` gives
`sample_nested_dynamic`'s own estimate.
"""
function _ns_logz_sim(rng::AbstractRNG, tr::_NSTrace, members, by_group;
                      resample::Bool = true, jitter::Bool = true)
    idx = if resample
        v = Int[]
        for ths in by_group, _ in eachindex(ths)
            append!(v, members[ths[rand(rng, eachindex(ths))]])
        end
        v
    else
        collect(eachindex(tr.logL))
    end
    L = sort!(tr.logL[idx])
    n = _ns_live_counts(L, sort!(tr.birth[idx]))
    log_X = 0.0
    log_z = -Inf
    for k in eachindex(L)
        log_t = jitter ? -randexp(rng) / n[k] : -log1p(1 / n[k])
        log_z = logaddexp(log_z, L[k] + log_X + log(-expm1(log_t)))
        log_X += log_t
    end
    return log_z
end

"""
    _ns_logz_err(tr; group = t -> 1, n_sims = $_NS_ERR_SIMS, seed = 0,
                 resample = true, jitter = true) -> Float64

The standard deviation of log Z over `n_sims` realisations of the run in `tr`,
its threads resampled and its prior volumes drawn afresh in each (see the top
of this file). `group(thread)` names a thread's pass; each pass's threads are
resampled apart (a dynamic run's baseline and batch, as dynesty resamples
them). Its own random numbers, from `seed`, so the run's are left alone. `NaN`
for a run of fewer than two points. `resample` and `jitter` switch the two
sources off one at a time (`_ns_logz_sim`).
"""
function _ns_logz_err(tr::_NSTrace; group = t -> 1, n_sims::Int = _NS_ERR_SIMS,
                      seed::Integer = 0, resample::Bool = true, jitter::Bool = true)
    length(tr.logL) < 2 && return NaN
    members, by_group = _ns_threads(tr, group)
    rng = MersenneTwister(seed)
    z = Float64[]
    for _ in 1:n_sims
        lz = _ns_logz_sim(rng, tr, members, by_group; resample, jitter)
        isfinite(lz) && push!(z, lz)
    end
    return length(z) < 2 ? NaN : std(z)
end
