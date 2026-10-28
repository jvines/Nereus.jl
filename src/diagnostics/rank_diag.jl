# Rank-normalised ESS and R-hat (Vehtari et al. 2021) for long chains: the
# numbers MCMCDiagnosticTools' `ess(kind = :bulk)`, `ess(kind = :tail)`,
# `rhat(kind = :rank)` and `ess_rhat(kind = :rank)` give, bit for bit, at a
# fraction of the memory and once per parameter per run.
#
# MCMCDiagnosticTools, called through MCMCChains, copies the whole chain into
# its own layout, then for every parameter allocates a rank-normalised copy, a
# sort permutation, a rank vector, a folded copy and a split copy -- about ten
# arrays the size of the chain per parameter and call. The post-fit stages call
# it six times over the same draws (fit_health, the sampler diagnostics, the
# bridge guard, the science table's convergence report, ...): 10 to 20 GB of
# allocation per stage at 5.8 million draws, and up to 4 GiB of extra RSS.
#
# Here a worker holds three buffers the length of one parameter's draws (two
# Float64, one Int32: 20 bytes a draw), reused across parameters; the number of
# workers is capped by `_RANK_DIAG_MEMORY`. Every arithmetic step is the
# library's own (`StatsBase._tiedrank!`, `_normal_quantiles_from_ranks!`,
# `norminvcdf`, `copyto_split!`'s row selection, `Statistics.mean!` / `var`,
# `mean_autocov`, the Geyer truncation), applied to the same values in the same
# order, so the results are identical. Results are remembered by the content
# of the draws (`_RANK_DIAG_MEMO`), so the later stages pay a checksum, not a
# sort.

using MCMCChains
using Statistics

const _MCDT = MCMCChains.MCMCDiagnosticTools

"""
    _RankDiag(ess_bulk, rhat_rank, ess_tail)

One parameter's bulk ESS, rank R-hat (the larger of the bulk and tail R-hat)
and tail ESS, as `ess(kind = :bulk)`, `rhat(kind = :rank)` and
`ess(kind = :tail)` report them. `ess_tail` is NaN in the `:ensemble` layout,
which does not compute it, and when the library's tail ESS would throw
(`tail_ok` false).
"""
struct _RankDiag
    ess_bulk::Float64
    rhat_rank::Float64
    ess_tail::Float64
    tail_ok::Bool          # false: the tail ESS threw (as the library would)
end

"""Bytes of rank-diagnostic workspace in use at once (all workers together)."""
const _RANK_DIAG_MEMORY = 512 * 2^20

const _RANK_DIAG_MEMO = Dict{Tuple{UInt64, Int, Int, Symbol}, _RankDiag}()
const _RANK_DIAG_LOCK = ReentrantLock()
const _RANK_DIAG_MEMO_MAX = 4096

# The library's constants: maxlag, the tail probability, the Blom offset.
const _RANK_MAXLAG = 250
const _RANK_TAIL_PROB = 1 // 10

"""
    _rank_checksum(src) -> UInt64

A 64-bit digest of the draws in `src`, in column-major order: what the memo
is keyed on, with the shape. Each value's bits are mixed in by a multiply and
a xor-shift, so a change to any draw changes the digest.
"""
function _rank_checksum(src::AbstractMatrix{Float64})
    h = 0x9e3779b97f4a7c15 ⊻ UInt64(length(src))
    @inbounds for x in src
        h = (h ⊻ reinterpret(UInt64, x)) * 0xff51afd7ed558ccd
        h ⊻= h >> 32
    end
    return h
end

"""
    _RankWS(n)

A worker's buffers for parameters of up to `n` draws: `x` the draws (later
the folded draws), `z` the ranks and normal scores, `p` the sort permutation.
"""
mutable struct _RankWS
    x::Vector{Float64}
    z::Vector{Float64}
    p::Vector{Int32}
end
_RankWS(n::Int) = _RankWS(Vector{Float64}(undef, n), Vector{Float64}(undef, n),
                          Vector{Int32}(undef, n))
function _rank_ws_fit!(ws::_RankWS, n::Int)
    length(ws.x) == n && return ws
    resize!(ws.x, n); resize!(ws.z, n); resize!(ws.p, n)
    return ws
end
_rank_ws_bytes(n::Int) = 24 * n          # x, z, p, and sortperm!'s Int32 scratch

"""
    _rank_starts(m, nchains, layout) -> (starts, len)

Where each "chain" of the library's computation begins in the column-major
draws of an `m × nchains` matrix, and how long each is.

`:split2` is `copyto_split!` with two splits, as `ess` / `rhat` apply by
default: each chain's first and second halves, the middle row dropped when `m`
is odd. `:ensemble` is fit_health's `_ensemble_split` followed by the same
split: each chain's halves become chains of their own (needs `m` even here),
and each of those is split again.
"""
function _rank_starts(m::Int, nchains::Int, layout::Symbol)
    starts = Int[]
    if layout === :split2
        a, d = divrem(m, 2)
        for c in 1:nchains
            s0 = (c - 1) * m
            push!(starts, s0 + 1, s0 + a + d + 1)
        end
        return starts, a
    elseif layout === :ensemble
        iseven(m) || throw(ArgumentError("the :ensemble layout needs an even length"))
        h = m ÷ 2
        a, d = divrem(h, 2)
        for c in 1:nchains
            s0 = (c - 1) * m
            push!(starts, s0 + 1, s0 + a + d + 1, s0 + h + 1, s0 + h + a + d + 1)
        end
        return starts, a
    end
    throw(ArgumentError("unknown layout $layout"))
end

"""
    _rank_compact!(dst, src, starts, len) -> Matrix

Copies the runs `src[s:s+len-1]` for `s` in `starts` one after another to the
front of `dst` and returns them as a `len × length(starts)` Matrix sharing
`dst`'s memory: the `samples` matrix `copyto_split!` builds. `dst` may be
`src`: every run moves toward the front, and `copyto!` handles the overlap.
"""
function _rank_compact!(dst::Vector{Float64}, src::Vector{Float64},
                        starts::Vector{Int}, len::Int)
    for (j, s) in enumerate(starts)
        o = (j - 1) * len + 1
        (dst === src && o == s) || copyto!(dst, o, src, s, len)
    end
    return unsafe_wrap(Matrix{Float64}, pointer(dst), (len, length(starts)))
end

"""
    _rank_ess_rhat!(S) -> (ess, rhat)

`_ess_rhat_basic!` of MCMCDiagnosticTools for one parameter whose split
`samples` matrix is `S` (centred in place, as there), with its `maxlag`
clamp and its `niter > 4` guard.
"""
function _rank_ess_rhat!(S::Matrix{Float64})
    niter, nchains = size(S)
    ntotal = niter * nchains
    if !(niter > 4)
        return NaN, _rank_rhat!(S)
    end
    maxlag = min(_RANK_MAXLAG, niter - 4)
    chain_mean = Array{Float64}(undef, 1, nchains)
    chain_var = Array{Float64}(undef, nchains)
    correctionfactor = (niter - 1) // niter
    esscache = _MCDT.build_cache(_MCDT.AutocovMethod(), S, chain_var)
    rel_ess_max = log10(oftype(one(Float64), ntotal))

    Statistics.mean!(chain_mean, S)
    @inbounds for j in 1:nchains
        chain_var[j] = Statistics.var(view(S, :, j); mean = chain_mean[j], corrected = true)
    end
    W = Statistics.mean(chain_var)
    var₊ = correctionfactor * W + Statistics.var(chain_mean; corrected = (nchains > 1))
    inv_var₊ = inv(var₊)
    rhat = sqrt(var₊ / W)

    S .-= chain_mean
    _MCDT.update!(esscache)
    ρ_odd = 1 - inv_var₊ * (W - _MCDT.mean_autocov(1, esscache))
    ρ_even = one(ρ_odd)
    pₜ = ρ_even + ρ_odd
    sum_pₜ = pₜ
    k = 2
    while k < (maxlag - 1)
        ρ_even = 1 - inv_var₊ * (W - _MCDT.mean_autocov(k, esscache))
        ρ_odd = 1 - inv_var₊ * (W - _MCDT.mean_autocov(k + 1, esscache))
        Δ = ρ_even + ρ_odd
        Δ > zero(Δ) || break
        pₜ = min(Δ, pₜ)
        sum_pₜ += pₜ
        k += 2
    end
    ρ_even = maxlag > 1 ? 1 - inv_var₊ * (W - _MCDT.mean_autocov(k, esscache)) : zero(ρ_even)
    τ = max(0, 2 * sum_pₜ + max(0, ρ_even) - 1)
    ess = min(inv(τ), rel_ess_max) * ntotal
    return ess, rhat
end

"""`_rhat_basic!` of MCMCDiagnosticTools for one split `samples` matrix `S`."""
function _rank_rhat!(S::Matrix{Float64})
    niter, nchains = size(S)
    chain_mean = Array{Float64}(undef, 1, nchains)
    chain_var = Array{Float64}(undef, nchains)
    correctionfactor = (niter - 1) // niter
    Statistics.mean!(chain_mean, S)
    @inbounds for j in 1:nchains
        chain_var[j] = Statistics.var(view(S, :, j); mean = chain_mean[j], corrected = true)
    end
    W = Statistics.mean(chain_var)
    var₊ = correctionfactor * W + Statistics.var(chain_mean; corrected = (nchains > 1))
    return sqrt(var₊ / W)
end

"""
    _rank_scores!(z, x, p)

`_rank_normalize!`: `z` gets the tied ranks of `x` (from its sort permutation
`p`) mapped to standard-normal quantiles with Blom's offset.
"""
function _rank_scores!(z::Vector{Float64}, x::Vector{Float64}, p::Vector{Int32})
    _MCDT.StatsBase._tiedrank!(z, x, p)
    _MCDT._normal_quantiles_from_ranks!(z, z)
    map!(_MCDT.StatsFuns.norminvcdf, z, z)
    return z
end

"""
    _rank_diag!(ws, src, layouts) -> Dict{Symbol, _RankDiag}

All the rank diagnostics of one parameter, whose draws are the `m × nchains`
matrix `src` (iterations down, chains across), for each layout in `layouts`
(`:split2`, and `:ensemble` when `m` is even). One sort of the draws serves
every layout's bulk numbers and one sort of the folded draws every layout's
tail R-hat. `:split2` also gets the tail ESS.
"""
function _rank_diag!(ws::_RankWS, src::AbstractMatrix{Float64}, layouts)
    m, nchains = size(src)
    n = m * nchains
    _rank_ws_fit!(ws, n)
    x, z, p = ws.x, ws.z, ws.p
    copyto!(x, src)
    lay = Dict(l => _rank_starts(m, nchains, l) for l in layouts)
    out = Dict{Symbol, _RankDiag}()
    ess_bulk = Dict{Symbol, Float64}(); rhat_bulk = Dict{Symbol, Float64}()
    rhat_tail = Dict{Symbol, Float64}(); ess_tail = NaN

    GC.@preserve x z begin
        # Tail ESS: the ESS of the indicators x ≤ q at the 5% and 95% quantiles.
        if :split2 in layouts
            starts, len = lay[:split2]
            T = Float64
            pl = convert(T, _RANK_TAIL_PROB / 2)
            pu = convert(T, 1 - _RANK_TAIL_PROB / 2)
            copyto!(z, x)
            ql = Statistics.quantile!(z, pl)
            qu = Statistics.quantile!(z, pu)        # z is still a permutation of x
            el = eu = NaN
            for (q, which) in ((ql, :lo), (qu, :hi))
                @inbounds for i in 1:n
                    z[i] = x[i] ≤ q
                end
                e, _ = _rank_ess_rhat!(_rank_compact!(z, z, starts, len))
                which === :lo ? (el = e) : (eu = e)
            end
            ess_tail = min(el, eu)
        end

        # Bulk: rank-normalised draws.
        sortperm!(p, x)
        _rank_scores!(z, x, p)
        for (i, l) in enumerate(layouts)
            starts, len = lay[l]
            # Every layout but the last works on a copy in x (no longer needed
            # for the bulk numbers); the last compacts z in place.
            S = i < length(layouts) ? _rank_compact!(x, z, starts, len) :
                                      _rank_compact!(z, z, starts, len)
            ess_bulk[l], rhat_bulk[l] = _rank_ess_rhat!(S)
        end

        # Tail R-hat: rank-normalised distance from the median.
        copyto!(x, src)
        copyto!(z, x)
        med = Statistics.median!(z)
        @inbounds for i in 1:n
            x[i] = abs(x[i] - med)
        end
        sortperm!(p, x)
        _rank_scores!(z, x, p)
        for (i, l) in enumerate(layouts)
            starts, len = lay[l]
            S = i < length(layouts) ? _rank_compact!(x, z, starts, len) :
                                      _rank_compact!(z, z, starts, len)
            rhat_tail[l] = _rank_rhat!(S)
        end
    end
    for l in layouts
        out[l] = _RankDiag(ess_bulk[l], max(rhat_tail[l], rhat_bulk[l]),
                           l === :split2 ? ess_tail : NaN, true)
    end
    return out
end

"""
    _rank_diag_lib(src, layout) -> _RankDiag

The same numbers from MCMCDiagnosticTools itself: for draws the fast path
does not take (non-finite values, too few iterations to split).
"""
function _rank_diag_lib(src::AbstractMatrix, layout::Symbol)
    m, nchains = size(src)
    arr = if layout === :ensemble && m >= 4
        h = m ÷ 2
        e = Array{Float64}(undef, h, 2nchains, 1)
        for c in 1:nchains
            e[:, 2c - 1, 1] .= view(src, 1:h, c)
            e[:, 2c, 1] .= view(src, (h + 1):(2h), c)
        end
        e
    else
        reshape(Array{Float64}(src), m, nchains, 1)
    end
    er = _MCDT.ess_rhat(arr; kind = :rank)
    et, ok = NaN, true
    if layout !== :ensemble
        try
            et = only(_MCDT.ess(arr; kind = :tail))
        catch
            ok = false           # e.g. NaN draws: quantiles are undefined
        end
    end
    return _RankDiag(only(er.ess), only(er.rhat), et, ok)
end

"""
    _rank_diag_batch(srcs, layout; n_max, src_bytes = 0) -> Vector{_RankDiag}

The rank diagnostics of each parameter `srcs[k]()` returns (an `m × nchains`
Float64 matrix of its draws), in `layout` (`:split2` or `:ensemble`).

Each source is looked up by the content of its draws first: a parameter seen
before (by this stage or an earlier one) costs a checksum, not a sort. The
rest are computed by up to `nthreads` workers, as many as `_RANK_DIAG_MEMORY`
holds at `n_max` draws a parameter (plus `src_bytes` a draw when a source is
built rather than viewed), each with one workspace reused across parameters
and allocated only on its first miss. With an even number of iterations a
parameter's `:split2` and `:ensemble` numbers come from the same two sorts and
are remembered together.
"""
function _rank_diag_batch(srcs::AbstractVector, layout::Symbol;
                          n_max::Int, src_bytes::Int = 0)
    K = length(srcs)
    out = Vector{_RankDiag}(undef, K)
    K == 0 && return out
    budget = min(_RANK_DIAG_MEMORY, Int(Sys.total_memory() ÷ 8))
    per = _rank_ws_bytes(n_max) + src_bytes * n_max
    nw = clamp(budget ÷ max(per, 1), 1, min(Threads.nthreads(), K))
    next = Threads.Atomic{Int}(0)
    function work()
        ws = nothing
        while true
            k = Threads.atomic_add!(next, 1) + 1
            k > K && break
            s = srcs[k]()
            m = size(s, 1)
            l = layout === :ensemble && m < 4 ? :split2 : layout   # as `_ensemble_split`
            key = (_rank_checksum(s), m, size(s, 2), l)
            r = lock(() -> get(_RANK_DIAG_MEMO, key, nothing), _RANK_DIAG_LOCK)
            if r === nothing
                ws === nothing && (ws = _RankWS(0))
                r = _rank_diag_one!(ws, s, l, key)
            end
            out[k] = r
        end
        return nothing
    end
    if nw == 1
        work()
    else
        @sync for _ in 1:nw
            Threads.@spawn work()
        end
    end
    return out
end

function _rank_diag_one!(ws::_RankWS, src::AbstractMatrix{Float64}, layout::Symbol, key)
    m, nchains = size(src)
    fast = m >= 20 && all(isfinite, src) && m * nchains <= typemax(Int32)
    if !fast
        r = _rank_diag_lib(src, layout)
        _rank_memo!(key, r)
        return r
    end
    if layout === :split2 || iseven(m)
        layouts = iseven(m) ? (:split2, :ensemble) : (:split2,)
        res = _rank_diag!(ws, src, layouts)
        for (l, r) in res
            _rank_memo!((key[1], key[2], key[3], l), r)
        end
        return res[layout]
    end
    # :ensemble with an odd m: `_ensemble_split` drops the last iteration, so
    # the ranks are over a different set of draws.
    res = _rank_diag!(ws, view(src, 1:(m - 1), :), (:ensemble,))
    r = res[:ensemble]
    _rank_memo!(key, r)
    return r
end

function _rank_memo!(key, r::_RankDiag)
    lock(_RANK_DIAG_LOCK) do
        length(_RANK_DIAG_MEMO) >= _RANK_DIAG_MEMO_MAX && empty!(_RANK_DIAG_MEMO)
        _RANK_DIAG_MEMO[key] = r
    end
    return r
end

"""
    _chain_param_cols(chains) -> (names, columns) or nothing

The `:parameters` section of `chains` (what MCMCChains' `ess` / `rhat` /
`ess_rhat` report by default, in their order) and each one's index in the
chain's value array; `nothing` when there is no such section or the values
are not plain Float64.
"""
function _chain_param_cols(chains::MCMCChains.Chains)
    haskey(chains.name_map, :parameters) || return nothing
    eltype(chains.value.data) === Float64 || return nothing
    allnames = names(chains)
    pos = Dict(n => i for (i, n) in enumerate(allnames))
    ps = collect(chains.name_map[:parameters])
    return ps, [pos[p] for p in ps]
end

"""
    _rank_diag_chain(chains, pnames, layout) -> Vector{_RankDiag} or nothing

The rank diagnostics of the parameters `pnames` of `chains`, read in place
from the chain's values; `nothing` when they are not plain Float64.
"""
function _rank_diag_chain(chains::MCMCChains.Chains, pnames, layout::Symbol)
    eltype(chains.value.data) === Float64 || return nothing
    data = chains.value.data
    pos = Dict(n => i for (i, n) in enumerate(names(chains)))
    cols = [pos[Symbol(p)] for p in pnames]
    return _rank_diag_batch([() -> view(data, :, j, :) for j in cols], layout;
                            n_max = size(data, 1) * size(data, 3))
end
