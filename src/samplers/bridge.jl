# Bridge sampling: evidence from POSTERIOR SAMPLES ONLY.
#
# WHY THIS EXISTS. Every β-path estimator in evidence.jl (TI, TI+, SS+, H+)
# integrates from the prior to the posterior, so all of them depend on ⟨log L⟩
# at the hot end. On a joint RV+transit fit that expectation is
# E_prior[log L], and it is tail-dominated: prior draws on HD 18599 give a
# median log L of -4.9e4 with a minimum of -1.5e7. No ladder resolves a
# quantity whose variance is that large, and measurement bears it out — TI+
# sat 118 nats from the truth with swap acceptance 0.23 and a self-reported
# error of 0.48 nats, and adding rungs, cooling beta_min or enabling the
# adaptive ladder all moved it FURTHER away. That is a bias that does not
# shrink with computation, and it cannot be tuned out.
#
# The four estimators also all read one shared `mean_logL[k]` array, so a bias
# in ⟨log L⟩ per rung appears identically in every one of them. Their mutual
# agreement is therefore not evidence of correctness — on the run above they
# agreed with each other to 2.6 nats while all being 147 nats wrong.
#
# Bridge sampling never touches the prior end. It needs only samples already
# drawn from the posterior plus a proposal we can both sample and evaluate,
# and it costs `n_proposal` likelihood evaluations rather than the ~1.3e7 a
# nested-sampling run spends on the same model, plus log p* at each of the N1
# posterior draws handed in. That comes from the chain's own `:lp` where it is
# this target's log posterior (pt_emcee's is), checked against a full
# evaluation at a few draws (`_bridge_stored_lp`); otherwise the posterior is
# evaluated at every draw, N1 + `n_proposal` evaluations, which grows with the
# chain (5.8 million for a production transit fit). Both evaluation loops are
# threaded (see below). The bootstrap error then solves the bridge equation
# 200 more times on resamples of all of those draws; those solves run in
# parallel (`_bridge_bootstrap`), each in a few Newton passes started at the
# primary root (`_bridge_iterate`). Nothing here holds a copy of the posterior:
# the draws are read off the chain and transformed as they are needed
# (`_BridgeDraws`), and a bootstrap resample is a count per draw.
#
# METHOD. With p*(θ) the unnormalised posterior, q(θ) a normalised proposal,
# {θ_i} ~ p (N1 of them) and {φ_j} ~ q (N2), the optimal bridge (Meng & Wong
# 1996) solves the fixed point
#
#            (1/N2) Σ_j  l2_j / (s1·l2_j + s2·r)
#     r  =   ───────────────────────────────────  ,   l = p*/q,  s = N/(N1+N2)
#            (1/N1) Σ_i    1   / (s1·l1_i + s2·r)
#
# and r → Z. Solved in logs so nothing overflows. Unlike the harmonic-mean
# estimator it has finite variance, and unlike Laplace it makes no Gaussianity
# assumption about the posterior — q only has to overlap it.


using Statistics: mean, std, cov
using LinearAlgebra: cholesky, logdet, Symmetric, I
using Random: AbstractRNG, MersenneTwister, randn, rand, rand!
using SpecialFunctions: loggamma

export bridge_evidence

# Memory the bootstrap's concurrent solves may hold, in bytes, unless
# `bootstrap_memory` says otherwise. Each solve holds 2 bytes per draw (see
# `_BridgeBootBuffers`): 20 MB at 9.8 million posterior draws, so this never
# binds below a hundred threads.
const _BRIDGE_BOOT_MEMORY = 2^31

"""
    bridge_evidence(target, chains; n_proposal, seed, proposal, max_iter, tol,
                    n_bootstrap, bootstrap_memory, show_progress, timings) -> NamedTuple

Log-evidence by optimal bridge sampling from an existing posterior sample.

Returns `(; log_z, se, n_post, n_prop, iters, converged, overlap, proposal)`.

`chains` must be the β=1 chain in the target's own parameter space (that is what
every sampler here stores). The target must be `unconstrained=true`: the
proposal is Gaussian on ℝⁿ, so on a bounded parametrisation most proposal draws
would land outside support and be wasted.

`proposal` is `:gaussian` (mean/covariance of the posterior draws) or
`:student` (the same moments with heavier tails, ν=4). Prefer `:student` when
the posterior is skewed or heavy-tailed — an over-narrow proposal is the one
failure mode that biases bridge sampling badly, because the region where q has
no mass never gets sampled.

`overlap` is the fraction of proposal draws with finite posterior density. It is
the diagnostic that matters: below ~0.1 the proposal is a poor match and the
estimate should not be trusted, however tight `se` looks.

`se` is a bootstrap standard error over both sample sets. It is a real sampling
error, not a placeholder — but it says nothing about proposal mismatch, which is
what `overlap` is for.

`use_lp`: when the chain has an `:lp` column holding this target's log
posterior in the bounded space (what `pt_emcee` stores), log p* at each draw is
that value plus the transform's log Jacobian, instead of a fresh evaluation of
the posterior at every draw. It is used only if 64 draws spread over the chain
agree with a full evaluation (`_bridge_stored_lp`); otherwise, or with
`use_lp = false`, every draw is evaluated.

The equation is solved by safeguarded Newton (`_bridge_iterate`), 4 to 8
passes over the draws.

Memory: the draws are not copied. They are read off the chain and mapped to
the target's space as each pass needs them (`_BridgeDraws`), the proposal's
mean and covariance come from one streaming pass (`_bridge_moments`), and
what is kept is one `Float64` per posterior draw (its log ratio).

The `n_bootstrap` resamples are solved in parallel, by as many workers as there
are threads and `bootstrap_memory` bytes allow (2 bytes per posterior draw
each, at least one worker; the budget is also capped at a quarter of
`Sys.total_memory()`). They draw from one `MersenneTwister(seed + 1)` in
replicate order, exactly as the serial loop did, so every number returned is
the same at any thread count and any worker count. Each starts Newton at the
primary root, which the resample's root is near. See `_bridge_bootstrap`.

`show_progress = true` prints a progress line for each stage that can run long
on a large chain: the posterior and proposal evaluations, the Newton
solve and the bootstrap.

`timings`, if a dictionary, receives each stage's wall time, allocation and
the process's peak RSS when it ended, the sample counts at each filter, the
iteration counts and the number of bootstrap workers. See `_bridge_stage!`.
"""
function bridge_evidence(target::NereusTarget, chains::MCMCChains.Chains;
                         n_proposal::Int = 20_000,
                         seed::Int = 1,
                         proposal::Symbol = :student,
                         max_iter::Int = 1_000,
                         tol::Real = 1e-10,
                         n_bootstrap::Int = 200,
                         bootstrap_memory::Integer = _BRIDGE_BOOT_MEMORY,
                         use_lp::Bool = true,
                         show_progress::Bool = false,
                         timings::Union{Nothing, AbstractDict} = nothing)
    d = length(target.params.layout.unfrozen_names)
    d >= 1 || return (; log_z = NaN, se = NaN, n_post = 0, n_prop = 0,
                        iters = 0, converged = false, overlap = NaN,
                        proposal = proposal)
    timings === nothing || (timings["threads"] = Threads.nthreads())

    # The draws' log ratios: one Float64 per posterior draw, the only
    # chain-length array the estimator keeps.
    rs = _bridge_ratios(target, chains, n_proposal, seed, proposal,
                        show_progress, timings; use_lp = use_lp)
    rs.status === :too_few &&
        return (; log_z = NaN, se = NaN, n_post = rs.n_post, n_prop = 0,
                  iters = 0, converged = false, overlap = NaN,
                  proposal = proposal)
    l1, l2, overlap = rs.l1, rs.l2, rs.overlap
    (isempty(l1) || isempty(l2)) &&
        return (; log_z = NaN, se = NaN, n_post = rs.n_post, n_prop = length(l2),
                  iters = 0, converged = false, overlap = overlap,
                  proposal = proposal)

    pb = ProgressBar("bridge solve"; enabled = show_progress)
    on_iter = (it, delta) -> update!(pb; n_done = it, fields = (:delta => delta,))
    log_r, iters, converged = _bridge_stage!(timings, "primary_solve") do
        _bridge_iterate(l1, l2, max_iter, Float64(tol); on_iter = on_iter)
    end
    update!(pb; n_done = iters, fields = (:converged => string(converged),), force = true)
    finish!(pb)
    if timings !== nothing
        timings["iters"] = iters
        timings["converged"] = converged
    end

    # --- bootstrap standard error -------------------------------------------
    # Only when the primary estimate CONVERGED. `se` is the sampling error OF
    # `log_r`, and every caller discards `log_r` outright when `converged` is
    # false (pt_emcee.jl gates on `b.converged && isfinite(b.log_z)`), so
    # bootstrapping a non-converged estimate prices 200 extra fixed-point solves
    # into an answer nobody reads -- and it is exactly the case where each of
    # them runs the full `max_iter`, because a fixed point that is not there at
    # 1000 iterations on the real data is not there on a resample of it either.
    # Measured on the HD 114762 joint fit (525k posterior draws, no fixed point):
    # 885 s of the 1064 s the sampler took, all of it thrown away.
    #
    # A root that IS there took the old fixed-point iteration hundreds of
    # steps when a few walkers sit far from the rest: their draws dominate the
    # denominator and each step moved log r by nearly the same amount as the
    # last. With 2% of 1M draws (18 parameters, 20k proposal draws) stuck 10σ
    # out, a solve took ~700 steps, against 5 without them. A step over 5.8M
    # draws costs ~40 ms, so 200 solves of 400 steps are an hour on one core
    # -- about what a production fit spent after its last plot, at one core,
    # with nothing printed. Hence the parallel bootstrap, and Newton's method
    # (`_bridge_iterate`), which takes 4 to 8 passes in both regimes -- and 3
    # or 4 for a replicate, started at the primary root `log_r`: F is strictly
    # decreasing, so the start changes only how many passes it takes.
    ses = Float64[]
    if n_bootstrap > 0 && converged && isfinite(log_r)
        n_workers = _bridge_boot_workers(length(l1), length(l2), n_bootstrap,
                                         bootstrap_memory)
        bpb = ProgressBar("bridge bootstrap"; total = n_bootstrap,
                          enabled = show_progress)
        bprog = _SharedProgress(bpb; fields = () -> (:workers => n_workers,))
        res, ok, its = _bridge_stage!(timings, "bootstrap") do
            _bridge_bootstrap(l1, l2, MersenneTwister(seed + 1), n_bootstrap,
                              max_iter, Float64(tol); workers = n_workers,
                              progress = bprog, start = log_r)
        end
        finish!(bpb)
        # In replicate order, as the serial loop pushed them: `std` sums in
        # that order.
        for k in 1:n_bootstrap
            ok[k] && isfinite(res[k]) && push!(ses, res[k])
        end
        if timings !== nothing
            timings["bootstrap_workers"] = n_workers
            timings["bootstrap_ok"] = length(ses)
            timings["bootstrap_iters_max"] = maximum(its)
            timings["bootstrap_iters_total"] = sum(its)
        end
    end
    se = length(ses) >= 2 ? std(ses) : NaN

    return (; log_z = log_r, se = se, n_post = length(l1), n_prop = length(l2),
              iters = iters, converged = converged, overlap = overlap,
              proposal = proposal)
end

"""
    _bridge_stage!(f, timings, name)

`f()`, and when `timings` is a dictionary, `timings[name]` = its wall time
(`seconds`), the bytes allocated meanwhile on every thread (`alloc_bytes`), the
time spent in GC (`gc_seconds`) and the process's peak RSS when it returned
(`maxrss_bytes`; a high-water mark, so a stage that raised it shows by how much).
"""
function _bridge_stage!(f, timings, name::AbstractString)
    timings === nothing && return f()
    s = @timed f()
    timings[name] = Dict{String, Any}("seconds" => s.time, "alloc_bytes" => Int(s.bytes),
                                      "gc_seconds" => s.gctime,
                                      "maxrss_bytes" => Int(Sys.maxrss()))
    return s.value
end

"""
    _bridge_ratios(target, chains, n_proposal, seed, proposal, show_progress,
                   timings) -> NamedTuple

The log ratios `l = log p* - log q` at the posterior draws (`l1`) and at
`n_proposal` draws from the proposal (`l2`), and `overlap`, for
`bridge_evidence`. `status` is `:too_few` when fewer than `2d + 2` draws are
finite in the target's space, and then nothing else is computed.

No copy of the posterior is made. The draws are read off the chain, mapped
to the target's space where each pass needs them (`_BridgeDraws`): once for the
proposal's mean and covariance (`_bridge_moments`), once for the log ratios.
The one chain-length array is the log ratios themselves, 8 bytes a draw. The
version before materialised the transformed draws as a `d × N` matrix and
`cov` made a centred copy of it: 1.2 GB at 5.8 million draws of 13
parameters, beside the 46 MB that are kept.
"""
function _bridge_ratios(target::NereusTarget, chains::MCMCChains.Chains,
                        n_proposal::Int, seed::Int, proposal::Symbol,
                        show_progress::Bool, timings; use_lp::Bool = true)
    pnames = target.params.layout.unfrozen_names
    d = length(pnames)

    # --- posterior draws, in the target's own space --------------------------
    # Chains are stored in BOUNDED space; the target's log-density expects the
    # UNCONSTRAINED parametrisation (and folds in the Jacobian). Map forward, as
    # mode_laplace_evidence does. Skipping this is not a small error: a Gaussian
    # fitted to bounded samples escapes the box on essentially every draw in 22
    # dimensions, and the estimator returns overlap = 0 with no usable output.
    D = _BridgeDraws(chains, pnames, target.transform)
    N = D.N
    mom = _bridge_stage!(() -> _bridge_moments(D), timings, "transform")
    keep, N1 = mom.keep, mom.n
    if timings !== nothing
        timings["n_draws"] = N
        timings["n_post"] = N1
        timings["n_proposal"] = n_proposal
    end
    N1 >= 2d + 2 || return (; status = :too_few, n_post = N1, l1 = Float64[],
                              l2 = Float64[], overlap = NaN)

    # The posterior through the workspace likelihoods, one evaluator (Theta +
    # PTWorkspace) per evaluation task. See `_bridge_logdensity!`.
    n_tasks = _bridge_ntasks(max(N, n_proposal))
    evs = [_BridgeEvaluator(target) for _ in 1:n_tasks]
    ys = [Vector{Float64}(undef, d) for _ in 1:n_tasks]     # a draw, per task
    zs = [Vector{Float64}(undef, d) for _ in 1:n_tasks]     # log q's scratch

    μ = mom.μ
    Σ = mom.Σ + 1e-10 * I(d)
    Lc = cholesky(Symmetric(Σ)).L
    ν  = 4.0
    lq = _BridgeLogq(μ, Lc, proposal, ν)

    rng = MersenneTwister(seed)
    # Student-t draw = Gaussian scaled by sqrt(nu / chi2_nu); chi2 with integer
    # nu is a sum of nu squared standard normals, so no extra dependency.
    nu_i = Int(ν)
    draw() = proposal === :gaussian ? μ .+ Lc * randn(rng, d) :
             μ .+ Lc * (randn(rng, d) / sqrt(sum(abs2, randn(rng, nu_i)) / ν))

    # --- log ratios l = log p* - log q on both sample sets --------------------
    # Each term is the full posterior. Evaluated at every posterior draw that
    # is N1 + `n_proposal` LIKELIHOOD EVALUATIONS, and N1 is the ENTIRE kept
    # chain: 150_000 draws for a 100-walker × 1500-post-burnin-step run,
    # 5.8 million for a production transit fit (89 s on 8 threads with 31,918
    # photometric points). So where the chain carries this target's log
    # posterior (`_bridge_stored_lp`), the posterior side reads it instead and
    # only the proposal draws are evaluated.
    #
    # Each task evaluates with its own Theta, workspace and draw buffers and
    # touches no shared state, so the evaluations thread. The PROPOSAL DRAWS do
    # not: they come off a single MersenneTwister, and drawing them out of
    # order would change the numbers. So they are drawn serially up front and
    # only the evaluations are spread out, each writing its own slot, and both
    # arrays are filtered in index order afterwards -- so what reaches
    # `_bridge_iterate` is the same at any thread count. See `_bridge_eval!`
    # for how the threads are used.
    #
    # The posterior side runs over every draw of the chain; a draw that is not
    # finite in the target's space (`keep`) gets NaN, which the filter drops,
    # so `l1` holds the kept draws' finite ratios in chain order, as before.
    jlp = use_lp ? _bridge_stored_lp(target, D, keep, N1, evs[1], ys[1], timings) :
                   nothing
    if timings !== nothing
        timings["lp_reused"] = jlp !== nothing
    end
    v1 = Vector{Float64}(undef, N)
    ppb = ProgressBar("bridge posterior"; total = N, enabled = show_progress)
    _bridge_stage!(timings, "posterior_evals") do
        if jlp === nothing
            _bridge_eval!(v1; progress = _SharedProgress(ppb)) do c, i
                keep[i] || return NaN
                y = _bridge_draw!(ys[c], D, i)
                _bridge_logdensity!(evs[c], y) - _bridge_logq!(zs[c], lq, y)
            end
        else
            pt = target.transform
            j = jlp::Int
            _bridge_eval!(v1; progress = _SharedProgress(ppb)) do c, i
                keep[i] || return NaN
                y = _bridge_draw!(ys[c], D, i)
                lj = pt === nothing ? 0.0 : Float64(transform_logabsdetjac_inv(y, pt))
                _bridge_value(D, j, i) + lj - _bridge_logq!(zs[c], lq, y)
            end
        end
    end
    finish!(ppb)
    l1 = _bridge_keep_finite!(v1)

    Yp = Matrix{Float64}(undef, d, n_proposal)
    @inbounds for i in 1:n_proposal
        Yp[:, i] = draw()
    end
    v2 = Vector{Float64}(undef, n_proposal)
    qpb = ProgressBar("bridge proposal"; total = n_proposal, enabled = show_progress)
    _bridge_stage!(timings, "proposal_evals") do
        _bridge_eval!(v2; progress = _SharedProgress(qpb)) do c, i
            _bridge_logdensity!(evs[c], @view Yp[:, i])
        end
    end
    finish!(qpb)
    l2 = Float64[]
    n_finite = 0
    z = zs[1]
    @inbounds for i in 1:n_proposal
        y = @view Yp[:, i]
        v = v2[i]
        if isfinite(v)
            n_finite += 1
            push!(l2, v - _bridge_logq!(z, lq, y))
        end
    end
    if timings !== nothing
        timings["n_post_finite"] = length(l1)
        timings["n_prop_finite"] = length(l2)
    end
    return (; status = :ok, n_post = N1, l1 = l1, l2 = l2,
              overlap = n_finite / n_proposal)
end

"""
    _bridge_keep_finite!(v) -> v

`v`'s finite entries, in order, moved to its front and the rest cut off: the
filter `[x for x in v if isfinite(x)]` without a second vector.
"""
function _bridge_keep_finite!(v::Vector{Float64})
    j = 0
    @inbounds for i in eachindex(v)
        x = v[i]
        isfinite(x) || continue
        j += 1
        v[j] = x
    end
    return resize!(v, j)
end

"""
    _BridgeDraws(chains, names, pt)

The draws of the parameters `names`, read straight from the chain's array
(`chains.value.data`, iteration × variable × chain) one draw at a time, with
no copy. Draw `i` is the `i`-th of `vec(Array(chains[:, name, :]))`:
iterations of chain 1, then of chain 2, and so on. `_bridge_draw!` gathers one
into a buffer and maps it to the target's unconstrained space when `pt` is a
transform, with `transform_forward!` -- the same numbers the `d × N` matrix
this replaces held. `lp` is the variable index of the chain's `:lp`, or
`nothing`.
"""
struct _BridgeDraws{A<:AbstractArray, T}
    data::A
    cols::Vector{Int}
    lp::Union{Nothing, Int}
    pt::T
    n_iter::Int
    N::Int
end

function _BridgeDraws(chains::MCMCChains.Chains, pnames, pt)
    data = chains.value.data
    vars = Base.names(chains)
    cols = Int[]
    for nm in pnames
        j = findfirst(==(Symbol(nm)), vars)
        j === nothing && throw(ArgumentError("bridge_evidence: no $(nm) in the chain"))
        push!(cols, j)
    end
    n_iter = size(data, 1)
    return _BridgeDraws(data, cols, findfirst(==(:lp), vars), pt, n_iter,
                        n_iter * size(data, 3))
end

"""
    _bridge_draw!(y, D, i) -> y

Draw `i` of `D` in the target's space, written into `y`.
"""
@inline function _bridge_draw!(y::Vector{Float64}, D::_BridgeDraws, i::Int)
    ch, it = divrem(i - 1, D.n_iter)
    @inbounds for j in eachindex(D.cols)
        y[j] = Float64(D.data[it + 1, D.cols[j], ch + 1])
    end
    D.pt === nothing || transform_forward!(y, y, D.pt)
    return y
end

"""
    _bridge_value(D, j, i) -> Float64

Variable `j` of the chain at draw `i`, as stored (bounded space).
"""
@inline function _bridge_value(D::_BridgeDraws, j::Int, i::Int)
    ch, it = divrem(i - 1, D.n_iter)
    return @inbounds Float64(D.data[it + 1, j, ch + 1])
end

# Draws per block of `_bridge_moments`. A multiple of 64, so each block owns
# whole words of the `keep` BitVector and the blocks can write it in parallel.
const _BRIDGE_MOMENT_BLOCK = 16_384

"""
    _bridge_moments(D) -> (; keep, n, μ, Σ)

Which draws of `D` are finite in the target's space (`keep`, a BitVector),
how many (`n`), and their mean `μ` and covariance `Σ` (normalised by `n - 1`,
as `cov`), in one pass over the chain with no copy of it.

The draws go in blocks of `_BRIDGE_MOMENT_BLOCK`, on all threads. Each block's
kept draws are centred on their own mean and their cross-products summed; the
blocks are then combined in block order by the pairwise update of Chan, Golub &
LeVeque (1979). Blocks are fixed by the draw count alone and combined in
order, so the result does not depend on the thread count. It matches
`mean`/`cov` of the `d × n` matrix to rounding (a few parts in 10^15).
"""
function _bridge_moments(D::_BridgeDraws)
    N, d = D.N, length(D.cols)
    B = _BRIDGE_MOMENT_BLOCK
    nb = cld(N, B)
    keep = falses(N)
    cnt = zeros(Int, nb)
    mus = zeros(d, nb)
    m2s = zeros(d, d, nb)                   # upper triangle used
    next = Threads.Atomic{Int}(1)
    Threads.@threads for _ in 1:max(1, min(_nthread_chunks(), nb))
        buf = Matrix{Float64}(undef, d, B)
        acc = Matrix{Float64}(undef, d, d)
        y = Vector{Float64}(undef, d)
        while true
            k = Threads.atomic_add!(next, 1)
            k > nb && break
            n = 0
            for i in ((k - 1) * B + 1):min(k * B, N)
                _bridge_draw!(y, D, i)
                all(isfinite, y) || continue
                keep[i] = true
                n += 1
                @inbounds for a in 1:d
                    buf[a, n] = y[a]
                end
            end
            cnt[k] = n
            n == 0 && continue
            @inbounds for a in 1:d
                s = 0.0
                for t in 1:n
                    s += buf[a, t]
                end
                mus[a, k] = s / n
            end
            fill!(acc, 0.0)
            @inbounds for t in 1:n
                for a in 1:d
                    buf[a, t] -= mus[a, k]
                end
                for b in 1:d, a in 1:b
                    acc[a, b] += buf[a, t] * buf[b, t]
                end
            end
            copyto!(view(m2s, :, :, k), acc)
        end
    end
    return (; keep, _bridge_merge_moments(cnt, mus, m2s)...)
end

"""
    _bridge_merge_moments(cnt, mus, m2s) -> (; n, μ, Σ)

`_bridge_moments`' blocks combined in block order: block `k` has `cnt[k]`
draws, mean `mus[:, k]` and centred cross-products `m2s[:, :, k]` (upper
triangle). A function of its own so that nothing it assigns is a variable the
threaded loop above could capture and share.
"""
function _bridge_merge_moments(cnt::Vector{Int}, mus::Matrix{Float64}, m2s::Array{Float64, 3})
    d, nb = size(mus)
    n = 0
    μ = zeros(d)
    m2 = zeros(d, d)
    δ = zeros(d)
    for k in 1:nb
        nk = cnt[k]
        nk == 0 && continue
        if n == 0
            μ .= view(mus, :, k)
            m2 .= view(m2s, :, :, k)
            n = nk
            continue
        end
        nn = n + nk
        f = Float64(n) * nk / nn
        r = nk / nn
        for a in 1:d
            δ[a] = mus[a, k] - μ[a]
        end
        for b in 1:d, a in 1:b
            m2[a, b] += m2s[a, b, k] + δ[a] * δ[b] * f
        end
        for a in 1:d
            μ[a] += δ[a] * r
        end
        n = nn
    end
    Σ = Matrix{Float64}(undef, d, d)
    for b in 1:d, a in 1:b
        Σ[a, b] = Σ[b, a] = m2[a, b] / (n - 1)
    end
    return (; n, μ, Σ)
end

"""
    _BridgeLogq(μ, Lc, proposal, ν)

The proposal's normalised log density, `:gaussian` or `:student` (ν degrees of
freedom), with mean `μ` and covariance `Lc Lc'`. Everything that does not
depend on the point -- `logdet(Lc)` and the normalising constants -- is
computed here once, grouped as the closure it replaces grouped it. `L` is
`Lc` as a plain matrix (lower triangle used).
"""
struct _BridgeLogq
    μ::Vector{Float64}
    L::Matrix{Float64}
    gaussian::Bool
    c0::Float64          # gaussian: 0.5 d log 2π
    ld::Float64          # gaussian: logdet(Lc)
    k::Float64           # student: every term but the one in q
    h::Float64           # student: (ν + d) / 2
    ν::Float64
end

function _BridgeLogq(μ::Vector{Float64}, Lc, proposal::Symbol, ν::Float64)
    d = length(μ)
    ld = logdet(Lc)
    c0 = 0.5 * d * log(2π)
    k = loggamma((ν + d) / 2) - loggamma(ν / 2) - 0.5 * d * log(ν * π) - ld
    return _BridgeLogq(μ, Matrix{Float64}(Lc), proposal === :gaussian, c0, ld, k,
                       0.5 * (ν + d), ν)
end

"""
    _bridge_logq!(z, lq, y) -> Float64

log q(y), with `z` (length d) as scratch: `z = Lc⁻¹ (y - μ)` by forward
substitution, column by column as BLAS's `trsv` does it, so it agrees with
`Lc \\ (y .- μ)` to rounding. Without allocating, and without BLAS: called
from 8 threads at once, as the posterior pass calls it, OpenBLAS's `trsv`
took 3.6 µs a call against 0.26 µs for this loop (d = 13, one BLAS thread or
eight alike), which was 3.4 s of a 5.8-million-draw pass.
"""
function _bridge_logq!(z::Vector{Float64}, lq::_BridgeLogq, y::AbstractVector)
    μ, L = lq.μ, lq.L
    d = length(z)
    @inbounds for i in 1:d
        z[i] = y[i] - μ[i]
    end
    @inbounds for j in 1:d
        zj = z[j] / L[j, j]
        z[j] = zj
        for i in (j + 1):d
            z[i] -= L[i, j] * zj
        end
    end
    q = sum(abs2, z)
    lq.gaussian && return -0.5 * q - lq.c0 - lq.ld
    return lq.k - lq.h * log1p(q / lq.ν)
end

# Draws at which `_bridge_stored_lp` compares the chain's `:lp` with a full
# evaluation, and how close the two must be: `atol + rtol * |log p*|` nats.
const _BRIDGE_LP_CHECK = 64
const _BRIDGE_LP_ATOL = 1e-6
const _BRIDGE_LP_RTOL = 1e-12

"""
    _bridge_stored_lp(target, D, keep, n_kept, ev, y, timings) -> Int or nothing

The variable index of the chain's `:lp` when that column is this target's log
posterior in the bounded space, else `nothing`. Then `lp + log|J|` at a draw
is log p* there, which is what `_bridge_logdensity!` computes. `keep` marks
the draws `bridge_evidence` keeps (`n_kept` of them); `y` is scratch.

Checked, not assumed. At `_BRIDGE_LP_CHECK` draws spread evenly over the kept
ones, `lp + log|J|` must match `_bridge_logdensity!` within `_BRIDGE_LP_ATOL`
nats plus `_BRIDGE_LP_RTOL` of its size, and must be finite exactly where it
is. pt_emcee's `:lp` is the log prior plus the log likelihood at β = 1, mapped
back from its flat scale to the bounded one, so it passes. A column that is a
log joint on another scale, a tempered density or a sum without the prior
misses by far more. The check draws no random numbers, so no stream changes.
The largest difference goes to `timings["lp_check_max_abs"]`.
"""
function _bridge_stored_lp(target::NereusTarget, D::_BridgeDraws, keep::BitVector,
                           n_kept::Int, ev, y::Vector{Float64}, timings)
    jlp = D.lp
    jlp === nothing && return nothing
    n_kept == 0 && return nothing
    pt = target.transform
    worst = 0.0
    ok = true
    ranks = unique(round.(Int, range(1, n_kept; length = min(n_kept, _BRIDGE_LP_CHECK))))
    for i in _bridge_kept_draws(keep, n_kept, ranks)
        _bridge_draw!(y, D, i)
        full = _bridge_logdensity!(ev, y)
        lj = pt === nothing ? 0.0 : Float64(transform_logabsdetjac_inv(y, pt))
        stored = _bridge_value(D, jlp, i) + lj
        if isfinite(full) != isfinite(stored)
            ok = false; worst = Inf; break
        end
        isfinite(full) || continue
        δ = abs(stored - full)
        worst = max(worst, δ)
        δ <= _BRIDGE_LP_ATOL + _BRIDGE_LP_RTOL * abs(full) || (ok = false; break)
    end
    timings === nothing || (timings["lp_check_max_abs"] = worst)
    return ok ? jlp : nothing
end

"""
    _bridge_kept_draws(keep, n_kept, ranks) -> Vector{Int}

The draw index of the `r`-th kept draw for each `r` in `ranks` (ascending).
"""
function _bridge_kept_draws(keep::BitVector, n_kept::Int, ranks::Vector{Int})
    n_kept == length(keep) && return copy(ranks)
    out = Vector{Int}(undef, length(ranks))
    r, k = 0, 1
    for i in eachindex(keep)
        k > length(ranks) && break
        keep[i] || continue
        r += 1
        while k <= length(ranks) && ranks[k] == r
            out[k] = i
            k += 1
        end
    end
    return out
end

# Block of indices a thread takes from the shared counter in `_bridge_eval!`.
# Small enough that the last blocks even out the finish across threads (a
# joint-fit evaluation is ~1 ms), large enough that the counter is never
# contended.
const _BRIDGE_EVAL_BLOCK = 16

"""
    _bridge_ntasks(n) -> Int

Number of tasks `_bridge_eval!` runs for `n` evaluations: one per thread, but
never more than there are blocks of work.
"""
_bridge_ntasks(n::Integer) = n <= 0 ? 0 : min(_nthread_chunks(), cld(n, _BRIDGE_EVAL_BLOCK))

"""
    _bridge_eval!(f, out) -> out

`out[i] = f(c, i)` for every index of `out`, on all threads, where `c` in
`1:_bridge_ntasks(length(out))` numbers the task making the call, so that `f`
can keep scratch per task.

One task per thread, each taking blocks of `_BRIDGE_EVAL_BLOCK` indices from a
shared counter until none are left, so a thread on a slow core or one that
draws cheap points (a proposal outside the support returns at the prior) does
not hold up the rest. Every `out[i]` is computed by one call and written once,
so the result does not depend on which thread computed it or on the thread
count.

When there is a task for every thread, each one is marked with
`_serial_inner_loops!`: every thread already has evaluations of its own, and
the photometry's threaded reduction (`transit_likelihood.jl`) would otherwise
spawn one task per thread on every call, which only queue behind the other
threads' evaluations and allocate. That reduction sums fixed chunks combined
in order, so running it serially gives the same bits.

`progress`, a `_SharedProgress`, is ticked by each block's size as it finishes.
"""
function _bridge_eval!(f::F, out::AbstractVector{Float64}; progress = nothing) where {F}
    n = length(out)
    n_tasks = _bridge_ntasks(n)
    n_tasks == 0 && return out
    fills_threads = n_tasks >= Threads.nthreads()
    next = Threads.Atomic{Int}(1)
    Threads.@threads for c in 1:n_tasks
        fills_threads && _serial_inner_loops!()
        while true
            lo = Threads.atomic_add!(next, _BRIDGE_EVAL_BLOCK)
            lo > n && break
            hi = min(lo + _BRIDGE_EVAL_BLOCK - 1, n)
            for i in lo:hi
                @inbounds out[i] = f(c, i)
            end
            progress === nothing || _tick!(progress, hi - lo + 1)
        end
    end
    return out
end

"""
    _BridgeEvaluator(target)

One evaluation task's scratch for `_bridge_logdensity!`: a `Theta` and a
`PTWorkspace`, as each slot of `pt_emcee` has, and whether the photometry can
go through the workspace method (`_bridge_phot_ws`).
"""
struct _BridgeEvaluator{TT<:NereusTarget}
    target::TT
    theta::Theta{Float64}
    ws::PTWorkspace
    phot_ws::Bool
end

function _BridgeEvaluator(target::NereusTarget)
    params, data = target.params, target.data
    ws = PTWorkspace(params, params.config.max_kplanet, length(params.config.noise_models);
                     n_obs = length(data.t_rv), n_phot = length(data.t_phot))
    return _BridgeEvaluator(target, Theta{Float64}(params), ws, _bridge_phot_ws(params))
end

"""
    _bridge_phot_ws(params) -> Bool

Whether the evaluator's photometry goes through the workspace method
`transit_log_likelihood(theta, data, ws)`. Not when a planet has gravity
darkening (`:GD`): those fits take `transit_log_likelihood(theta, data)`.

The workspace method had no gravity-darkened transit when the bridge first
used it, and it handed over to the other method only for TTVs, supersampled
exposures (then longer than 2 min, now longer than 3 min) and photometric noise
models. A gravity-darkened fit with no supersampled cadence therefore took the
plain transit there, and its log density did not depend on i_star or lambda. The workspace method now computes
each cadence's gravity-darkened flux with the same calls as the allocating
one. Gravity-darkened fits still take the allocating method here, which keeps
their photometry at the bits of `_logdensity_parts`; through the workspace it
would match to rounding, not always to the bit (the summation order on light
curves longer than 4096 cadences).
"""
_bridge_phot_ws(params) = !any(has_gd, params.config.planet_modes)

"""
    _bridge_e_clamped(theta) -> Bool

Whether `true_anomaly` (orbit.jl) clamps the eccentricity of any active planet,
that is whether some planet's e lies outside [0, 1) (or is NaN). The
allocating RV and transit methods take the true anomaly from `true_anomaly`;
the workspace photometry, and the workspace RV of a fit with no noise models,
compute cos f and sin f from E with the e they are given. Outside that interval
the two compute different models, not the same model with different rounding;
every likelihood rejects such an e, so in practice this sends only a NaN e to
the allocating methods. (The clamp was to [0, 0.9999] up to Nereus 0.8.3, and
then this test was what kept the bridge on one model above 0.9999.)

The planet blocks are abstractly typed, so `planet_e_w` is not inferred and its
result is boxed. A call allocates what `planet_e_w(theta, k)::Tuple{T, T}` does
for each planet, and nothing for the comparison: 64 bytes per planet. With
:sesinw that is all `planet_e_w` costs; with :ew `planet_e_w` costs 32 bytes
and the assertion boxes the tuple for the other 32.
"""
@inline function _bridge_e_clamped(theta::Theta{T}) where {T}
    for k in planet_indices(theta)
        # The planet blocks are abstractly typed, so `planet_e_w` is not
        # inferred; the assertion keeps that from spreading to the test below.
        e, _ = planet_e_w(theta, k)::Tuple{T, T}
        # `true_anomaly`'s own clamp: true exactly when it changes e, and for NaN.
        _anomaly_e(e) == e || return true
    end
    return false
end

"""
    _bridge_logdensity!(ev, y) -> Float64

The log-posterior at `y` in the target's own space, `sum(_logdensity_parts(target,
y))` with the same guards, but through the workspace likelihoods the samplers
draw with (`rv_log_likelihood(theta, data, ws)`, `transit_log_likelihood(theta,
data, ws)`) and the evaluator's own `Theta`, so that a call does not allocate
the likelihood's buffers afresh.

It takes the allocating methods where the workspace methods compute a
different model, and for gravity-darkened photometry:

- Gravity darkening: a fit with a :GD planet takes the allocating photometry
  (`_bridge_phot_ws`). The workspace transit had no gravity darkening when
  the bridge first used it; it has now, and the allocating method keeps these
  fits at the bits of `_logdensity_parts`. TTV fits, exposures longer than
  3 min and photometric noise models are handed over by the workspace method
  itself.
- Eccentricity outside [0, 1), or NaN (`_bridge_e_clamped`): the allocating
  methods take the true anomaly from `true_anomaly`, which clamps e into
  [0, 1); the workspace photometry, and the workspace RV of a fit with no
  noise models, use the given e. Every likelihood rejects e outside [0, 1),
  so this matters only for a NaN e. Up to Nereus 0.8.3 `true_anomaly` clamped
  at 0.9999, and above that the two differed by much more than rounding (on
  an RV + transit fit with a 20k-point light curve, up to 8.6 nats in the RV
  and 8.1e4 nats in the photometry); the clamp now leaves every e < 1 alone,
  and the two methods compute one model there.

With that it is the same function as `_logdensity_parts` up to rounding at
every point, but not to the bit:

- RV of a fit with no noise models at all: both RV methods choose their branch
  on `config.noise_models` as a whole, whatever channel each model is on. With
  none, the workspace method (`_rv_ll_no_noise(theta, data, ws)`) caches each
  planet's velocity curve and computes cos(f+ω) by the angle-sum identity, so
  its log L differs in the last bits at most points. Measured up to 2.4e-9
  nats on the HD 18599 white-noise fit without the floor, and 2.4e-7 nats
  (3e-15 relative) on prior draws of an RV + astrometry fit. With any noise
  model, on the RV or only on the photometry, the RV goes through
  `_rv_ll_with_noise`, which evaluates each cadence as the other method does,
  in the same order. That gives the same bits except where an active
  IndicatorFloor with `kernel = :qp` scores a channel, that is a floor
  channel that no active joint ActivityGP covers and that has indicator data
  and its amplitude and jitter slots. The workspace path builds that floor's
  kernel sines by the angle-difference identity, so the floor term, and with
  it the RV log L, differs in the last bits at most points. On 300 prior
  draws of each HD 18599 fit with such a floor the RV matched at 31 to 113 of
  them. With RV and photometry it differed by up to 1.5e-6 nats with activity
  decorrelation, 5.0e-7 with white noise, 4.5e-7 with the GP rotation and
  1.3e-7 with the error scale; with RV only, by up to 6.0e-7 with the GP
  rotation, 3.0e-7 with white noise, 2.8e-7 with the error scale and 1.8e-7
  with activity decorrelation. The floor term alone differed by the same
  amounts. At 200 points close to a high-posterior point of each fit the
  largest difference was 1.1e-10 nats. A floor whose every channel an active
  ActivityGP covers scores nothing in either method, so the RV gives the same
  bits: on the HD 18599 activity-GP fit, whose floor is active and has the
  ActivityGP's four channels, at every one of those 300 draws and 200
  points, and in the trans-dimensional job, whose menu always carries a `:qp`
  floor on the ActivityGP's channels, in every state with the ActivityGP on.
  In a state with the ActivityGP off the floor scores its channels, and the
  RV matched at 42 to 111 of 300 prior draws in each state the job can reach
  (floor alone, or with one of ErrorScale, CeleriteRotation or activity
  decorrelation), up to 5.0e-7 nats apart. With no `:qp` floor that scores a channel the RV gave the same bits
  on every fit measured (RM with tomography, and a CeleriteSHO on the
  photometry only, with and without gravity darkening).
- Photometry: the workspace method sums every cadence in one pass, where the
  other sums fixed chunks and then the chunk totals. Both take each cadence's
  sky separation from the same per-call orbit constants (`_sky_orbit`); when
  the figures here were measured the allocating method still took it from
  `sky_separation`, a different route. On the HD 18599 joint fits with a
  20k-point light curve the whole log density then differed by up to 5.4e-9
  nats near the reference point.

In the whole density, on 300 prior draws per HD 18599 joint fit (floor on,
RV and photometry together), the largest relative differences were 1.9e-11
on the white-noise fit, 2.1e-11 with the GP rotation, 1.8e-12 with the
activity GP and 4.7e-11 with activity decorrelation. On the three fits whose
floor scores its channels the RV floor term set those: at the draw that gave
each, the RV differed by 8.0e-8 nats (white noise), 6.8e-9 (GP rotation) and
1.9e-8 (activity decorrelation), the photometry by 7.3e-11, 1.3e-10 and
6.5e-10. With the activity GP, whose RV matches, the whole difference is the
photometry's.

All of these are sample maxima, not bounds. Relative figures are |difference|
/ max(1, |log p|), and rounding does not bound that ratio: where |log p| is
below 1, terms of thousands of nats cancel and leave their rounding. At such
points between the posterior and the prior of the HD 18599 fits it reached
3.6e-7. Relative to |log prior| + |log L_RV| + |log L_phot|, the
largest difference measured was 2.9e-11. On a two-planet RV + transit fit
(10k-point light curve, ρ⋆ and the mean anomaly as parameters, the Gladman
stability check) it was 1.9e-10 nats at a point near the best one where
log p = -18.2, and 1.2e-10 nats on a prior draw where log p = 2.46 (1.03e-11
and 4.7e-11 relative to max(1, |log p|)).

The same bits as `_logdensity_parts` come out for the tomogram, for the
photometry wherever it takes the allocating method, for the RV of a fit with
a noise model and no active `:qp` IndicatorFloor that scores a channel, and
for the whole density at a clamped e. A gravity-darkened fit, whose
photometry always takes the allocating method, is therefore bit-identical
whenever it has no RV data, or has a noise model on either channel and no
such floor. Only a gravity-darkened fit with RV data and either no noise
model at all or a `:qp` floor that scores a channel carries the workspace RV
rounding, like any other such fit: on a one-planet RVPM_GD fit the RV log L
matched at 72 of 420 points (up to 5.1e-11 nats apart); on a two-band,
two-planet one the whole density matched at 410 of 500 (up to 4.7e-10 nats,
1.4e-15 relative), and one bridge_evidence of five moved log Z by 4.5e-13.
"""
function _bridge_logdensity!(ev::_BridgeEvaluator, y::AbstractVector)
    target = ev.target
    data = target.data
    theta = ev.theta
    pt = target.transform
    lj = 0.0
    if pt === nothing
        @inbounds for i in eachindex(y)
            isfinite(y[i]) || return -Inf
        end
        set_unfrozen!(theta, y)
    else
        x = transform_inverse(y, pt)
        @inbounds for i in eachindex(x)
            isfinite(x[i]) || return -Inf
        end
        lj = Float64(transform_logabsdetjac_inv(y, pt))
        isfinite(lj) || return -Inf
        set_unfrozen!(theta, x)
    end
    lp = log_prior(theta)
    isfinite(lp) || return -Inf
    # Where `true_anomaly` clamps an eccentricity the workspace methods compute
    # another model, so both likelihoods take the allocating methods there.
    ws_ok = !_bridge_e_clamped(theta)
    ll = ws_ok ? rv_log_likelihood(theta, data, ev.ws) : rv_log_likelihood(theta, data)
    isfinite(ll) || return -Inf
    lt = ws_ok && ev.phot_ws ? transit_log_likelihood(theta, data, ev.ws) :
                               transit_log_likelihood(theta, data)
    ltomo = tomogram_log_likelihood(theta, data)
    isfinite(ltomo) || return -Inf
    isfinite(lt) || return -Inf
    # Grouped as `_logdensity_parts` groups them: (prior + Jacobian) + likelihood.
    pj = pt === nothing ? lp : lp + lj
    return Float64(pj) + Float64(ll + lt + ltomo)
end


"""
    _bridge_iterate(l1, l2, max_iter, tol; w1, w2, start, stop, on_iter)
        -> (log_r, iters, converged)

The root of the optimal-bridge equation, `x = g(x)` with `x = log r` and

    g(x) = log[(1/N2) Σ_j l2_j / (s1 l2_j + s2 e^x)] - log[(1/N1) Σ_i 1 / (s1 l1_i + s2 e^x)]

(the ratios in logs, as everywhere here). `l1` are log(p*/q) at the posterior
draws, `l2` the same at the proposal draws.

`w1` and `w2`, when given, are counts per entry (`UInt16`): entry `i` stands
for `w1[i]` draws, and N1 is their sum. That is a bootstrap resample without
copying it (`_bridge_bootstrap`). An entry with count 0 adds nothing. A NaN
entry, counted or not, stops the solve unconverged, as a NaN ratio always
has.

Solved by Newton's method on F(x) = g(x) - x, safeguarded by bisection. Each
pass computes g and g' together (`_bridge_g`) and allocates nothing. g' is a
difference of two weighted means of logistic weights in (0, 1), so it lies in
(-1, 1): F is strictly decreasing, has one root, and the Newton step
F / (1 - g') is always defined. Every pass narrows a bracket on that root, and
a Newton step that leaves the bracket is replaced by its midpoint. Converged
when a step, or the bracket, is below `tol`.

`start` is where Newton starts; NaN (the default) means halfway between the
means of `l1` and `l2`, a sane start in the right decade. The root does not
depend on it, only the number of passes does: a bootstrap replicate started
at the primary root, which its own is close to, takes 2 to 5 (3 or 4 on a
5.8-million-draw transit fit, against 7 from the default start).

This used to iterate x ← g(x) itself. That converges linearly at rate |g'|
near the root, and g' approaches 1 when a few draws sit far out in the tails:
5 passes on a well-mixed posterior, about 40 on a production-like transit fit,
and ~700 with 2% of the draws stuck 10σ out. Every pass is a sweep over all
the posterior draws, and the bootstrap repeats the solve 200 times. Newton
takes 4 to 8 passes in all of these cases. It finds the same root; the old
iteration stopped within its last step of the root, so the two agree to
about `tol / (1 - |g'|)`.

`stop`, a `Threads.Atomic{Bool}`, ends the solve early (unconverged) once set:
the bootstrap sets it when a worker fails. `on_iter(it, |Δ log r|)` is called
after each pass.
"""
function _bridge_iterate(l1::Vector{Float64}, l2::Vector{Float64},
                         max_iter::Int, tol::Float64;
                         w1::Union{Nothing, Vector{UInt16}} = nothing,
                         w2::Union{Nothing, Vector{UInt16}} = nothing,
                         start::Float64 = NaN,
                         stop::Union{Nothing, Threads.Atomic{Bool}} = nothing,
                         on_iter = nothing)
    n1, n2 = length(l1), length(l2)
    # Empty input is a caller error upstream (no finite draws on one side); return
    # a non-answer rather than throwing from inside a reduction.
    (n1 == 0 || n2 == 0) && return (NaN, 0, false)
    (w1 === nothing || length(w1) == n1) && (w2 === nothing || length(w2) == n2) ||
        throw(DimensionMismatch("bridge counts must match the ratio vectors"))
    N1, N2 = _bridge_wsum(w1, n1), _bridge_wsum(w2, n2)
    (N1 == 0 || N2 == 0) && return (NaN, 0, false)
    ls1 = log(N1 / (N1 + N2))
    ls2 = log(N2 / (N1 + N2))
    logn1, logn2 = log(N1), log(N2)
    # Where each side's largest term is: t1 falls as l1 rises and t2 rises
    # with l2 (see `_bridge_g`), so at the smallest l1 and the largest l2.
    a1 = _bridge_wextreme(min, l1, w1, Inf)
    a2 = _bridge_wextreme(max, l2, w2, -Inf)
    log_r = isnan(start) ? 0.5 * (_bridge_wmean(l1, w1) + _bridge_wmean(l2, w2)) : start
    lo, hi = -Inf, Inf                           # F(lo) > 0 > F(hi)
    iters = 0
    converged = false
    for it in 1:max_iter
        stop !== nothing && stop[] && break
        iters = it
        g, dg = _bridge_g(l1, l2, w1, w2, ls1, ls2, logn1, logn2, a1, a2, log_r)
        (isfinite(g) && isfinite(dg)) || break
        F = g - log_r
        if F == 0
            converged = true; break
        end
        F > 0 ? (lo = log_r) : (hi = log_r)
        new = log_r + F / (1 - dg)
        # Newton moves toward the side F points to, so it leaves the bracket
        # only through a bound already found, and then bisects. The one other
        # way out is an infinite step (g' rounding to 1) toward an open side:
        # then the plain fixed-point step g, which is finite.
        if !(lo < new < hi)
            new = isfinite(lo) && isfinite(hi) ? 0.5 * (lo + hi) : g
        end
        step = abs(new - log_r)
        on_iter === nothing || on_iter(it, step)
        if step < tol || hi - lo < tol
            log_r = new; converged = true; break
        end
        log_r = new
    end
    return log_r, iters, converged
end

# Counts: 1 per entry when there are none.
@inline _bridge_w(::Nothing, i::Int) = 1.0
@inline _bridge_w(w::Vector{UInt16}, i::Int) = @inbounds Float64(w[i])

_bridge_wsum(::Nothing, n::Int) = n
_bridge_wsum(w::Vector{UInt16}, ::Int) = Int(sum(w))

_bridge_wmean(l::Vector{Float64}, ::Nothing) = mean(l)
function _bridge_wmean(l::Vector{Float64}, w::Vector{UInt16})
    s, n = 0.0, 0
    @inbounds for i in eachindex(l)
        k = w[i]
        k == 0 && continue
        s += k * l[i]
        n += k
    end
    return s / n
end

# `op` (min or max) of the entries with a count; NaN wins, counted or not.
# Branch-free: a draw's count is 0 about a third of the time, at random.
function _bridge_wextreme(op::F, l::Vector{Float64}, w, init::Float64) where {F}
    a = init
    @inbounds for i in eachindex(l)
        v = l[i]
        a = op(a, ifelse((_bridge_w(w, i) == 0) & !isnan(v), init, v))
    end
    return a
end

# Below this, the l1 term at `a1` and the l2 term at `a2` have u (see
# `_bridge_g`) small enough that 1 + e^u is finite with room to spare, and
# `_bridge_g_sums` applies. Its terms that underflow are then below e^-109 of
# the largest: below the rounding of any sum.
const _BRIDGE_G_FAST = 600.0

"""
    _bridge_g(l1, l2, w1, w2, ls1, ls2, logn1, logn2, a1, a2, x) -> (g, g′)

One pass of `_bridge_iterate`: g(x) and its derivative. With c = ls2 + x, the
log terms are t2_j = l2_j - log(e^{ls1 + l2_j} + e^c) and
t1_i = -log(e^{ls1 + l1_i} + e^c), and g = logmeanexp(t2) - logmeanexp(t1),
each entry counted `w` times (once with no counts) and the means over
N = exp(logn). d t/dx is minus the logistic weight w = e^c / (e^{ls1 + l} + e^c),
which is e^{c + t2 - l2} and e^{c + t1}, so g′ = Σ p1 w1 - Σ p2 w2 with p the
softmax weights of each logsumexp.

The logsumexps are shifted by each side's largest term, so nothing
overflows. t1 is decreasing in l1 and t2 increasing in l2, so those are the
terms at `a1`, the smallest l1, and `a2`, the largest l2, and one pass does it
all: no second pass over the terms, no buffer to keep them in.

Each term is a logistic function of u = ls1 + l1_i - c (or u = c - ls1 - l2_j):
e^{t - m} = σ(-u) / σ(-u_max term) and the weight is σ(-u) (or 1 - σ(-u)),
with σ(-u) = 1 / (1 + e^u). So `_bridge_g_sums` spends one `exp` a draw
where the log-space form spends three and a `log1p`: 13 ms a pass over 5.8
million draws against 51 ms, to an ulp of g. Only when the largest terms sit
more than `_BRIDGE_G_FAST` nats out, far from any root, does
`_bridge_g_shifted` take the log-space form, where nothing can underflow.
"""
function _bridge_g(l1::Vector{Float64}, l2::Vector{Float64}, w1, w2,
                   ls1::Float64, ls2::Float64, logn1::Float64, logn2::Float64,
                   a1::Float64, a2::Float64, x::Float64)
    c = ls2 + x
    m2 = a2 - _logaddexp(ls1 + a2, c)
    m1 = -_logaddexp(ls1 + a1, c)
    (isfinite(m1) && isfinite(m2)) || return (NaN, NaN)
    u1 = ls1 + a1 - c
    u2 = c - ls1 - a2
    s1, v1, s2, v2 = u1 <= _BRIDGE_G_FAST && u2 <= _BRIDGE_G_FAST ?
        _bridge_g_sums(l1, l2, w1, w2, ls1, c, u1, u2) :
        _bridge_g_shifted(l1, l2, w1, w2, ls1, c, m1, m2)
    g = (m2 + log(s2) - logn2) - (m1 + log(s1) - logn1)
    return g, v1 / s1 - v2 / s2
end

# The sums of `_bridge_g` through σ(-u) = 1 / (1 + e^u), e^{t - m} = σ(-u)
# (1 + e^{u at the largest term}). An entry with count 0 is multiplied by 0
# rather than skipped: no branch to mispredict a third of the time.
function _bridge_g_sums(l1::Vector{Float64}, l2::Vector{Float64}, w1, w2,
                        ls1::Float64, c::Float64, u1::Float64, u2::Float64)
    S2 = 1 + exp(u2)
    b2 = c - ls1
    s2 = 0.0; v2 = 0.0
    @inbounds for j in eachindex(l2)
        σ = 1 / (1 + exp(b2 - l2[j]))
        e = _bridge_w(w2, j) * σ * S2
        s2 += e
        v2 += e * (1 - σ)
    end
    S1 = 1 + exp(u1)
    b1 = ls1 - c
    s1 = 0.0; v1 = 0.0
    @inbounds for i in eachindex(l1)
        σ = 1 / (1 + exp(b1 + l1[i]))
        e = _bridge_w(w1, i) * σ * S1
        s1 += e
        v1 += e * σ
    end
    return s1, v1, s2, v2
end

# The sums of `_bridge_g` in log space, shifted by the largest terms m1, m2.
function _bridge_g_shifted(l1::Vector{Float64}, l2::Vector{Float64}, w1, w2,
                           ls1::Float64, c::Float64, m1::Float64, m2::Float64)
    s2 = 0.0; v2 = 0.0
    @inbounds for j in eachindex(l2)
        k = _bridge_w(w2, j)
        k == 0 && continue
        l = l2[j]
        t = l - _logaddexp(ls1 + l, c)
        t == -Inf && continue            # no weight (an l2 of -Inf)
        e = k * exp(t - m2)
        s2 += e
        v2 += e * exp(c + t - l)
    end
    s1 = 0.0; v1 = 0.0
    @inbounds for i in eachindex(l1)
        k = _bridge_w(w1, i)
        k == 0 && continue
        t = -_logaddexp(ls1 + l1[i], c)
        t == -Inf && continue            # no weight (an l1 of +Inf)
        e = k * exp(t - m1)
        s1 += e
        v1 += e * exp(c + t)
    end
    return s1, v1, s2, v2
end

function _logsumexp(v::AbstractVector{<:Real})
    isempty(v) && return -Inf
    m = maximum(v)
    isfinite(m) || return m
    return m + log(sum(x -> exp(x - m), v))
end

# Indices drawn per `rand!` call by `_bridge_draw_counts!`: 128 KB of scratch
# per bootstrap worker.
const _BRIDGE_IDX_CHUNK = 16_384

"""
    _BridgeBootBuffers(n1, n2, rng)

One bootstrap worker's scratch: a resample's counts per posterior and per
proposal draw (`c1`, `c2`, `UInt16`), a chunk of `Int` indices for `rand!`,
and a generator to replay a replicate's draws on (`rng`, a copy of the shared
one). 2 bytes per posterior and per proposal draw, reused for every replicate
the worker solves. (Before: 16, a copy of the resampled ratios and a buffer
for a pass's terms.)
"""
struct _BridgeBootBuffers{R<:AbstractRNG}
    c1::Vector{UInt16}
    c2::Vector{UInt16}
    idx::Vector{Int}
    rng::R
end

_BridgeBootBuffers(n1::Int, n2::Int, rng::AbstractRNG) =
    _BridgeBootBuffers(zeros(UInt16, n1), zeros(UInt16, n2),
                       Vector{Int}(undef, min(_BRIDGE_IDX_CHUNK, max(n1, n2))), copy(rng))

_bridge_boot_bytes(n1::Int, n2::Int) =
    2 * (n1 + n2) + 8 * min(_BRIDGE_IDX_CHUNK, max(n1, n2))

"""
    _bridge_boot_workers(n1, n2, n_bootstrap, bootstrap_memory) -> Int

How many bootstrap replicates to solve at once: one per thread, no more than
there are replicates, and no more than fit in `bootstrap_memory` bytes -- or in
a quarter of `Sys.total_memory()` (the container's limit, where there is one),
whichever is less. Always at least one.
"""
function _bridge_boot_workers(n1::Int, n2::Int, n_bootstrap::Int, bootstrap_memory::Integer)
    budget = min(Int(bootstrap_memory), Int(Sys.total_memory() ÷ 4))
    return max(1, min(Threads.nthreads(), n_bootstrap, budget ÷ _bridge_boot_bytes(n1, n2)))
end

"""
    _bridge_skip_indices!(idx, rng, m, n)

Take `m` draws of `rand(rng, 1:n)` off `rng`, `length(idx)` at a time into
`idx`, and discard them: `rng` ends where `_bridge_draw_counts!` with the same
`m` and `n` leaves it.
"""
function _bridge_skip_indices!(idx::Vector{Int}, rng::AbstractRNG, m::Int, n::Int)
    lo = 1
    while lo <= m
        len = min(length(idx), m - lo + 1)
        rand!(rng, view(idx, 1:len), 1:n)
        lo += len
    end
    return rng
end

"""
    _bridge_draw_counts!(cnt, idx, rng, m) -> cnt

`m` draws of `rand(rng, 1:length(cnt))`, counted: `cnt[j]` is how often `j`
was drawn. They are drawn `length(idx)` at a time into `idx`. For a
`MersenneTwister` and an `Int` range, `rand!` on an array is a loop of one
`rand(rng, sampler)` per element (Random has no bulk method for ranges), so
drawing in chunks takes the same numbers off the generator as one
`rand(rng, 1:n, m)`. (test_bridge_bootstrap.jl checks it against the bulk
call.) A count past `typemax(UInt16)` throws; a draw of `n` from `n` reaches
13 at most in practice.
"""
function _bridge_draw_counts!(cnt::Vector{UInt16}, idx::Vector{Int}, rng::AbstractRNG,
                              m::Int)
    n = length(cnt)
    fill!(cnt, 0x0000)
    lo = 1
    while lo <= m
        len = min(length(idx), m - lo + 1)
        chunk = view(idx, 1:len)
        rand!(rng, chunk, 1:n)
        @inbounds for i in 1:len
            j = chunk[i]
            c = cnt[j]
            c == typemax(UInt16) && throw(OverflowError("bootstrap count over $(typemax(UInt16))"))
            cnt[j] = c + 0x0001
        end
        lo += len
    end
    return cnt
end

"""
    _bridge_bootstrap(l1, l2, rng, n_bootstrap, max_iter, tol;
                      workers = 1, progress = nothing, start = NaN,
                      solve = _bridge_solve_counts)
        -> (res, ok, iters)

The `n_bootstrap` bootstrap solves of `bridge_evidence`, by `workers` threads
at once. Replicate `k` resamples `l1` and `l2` with replacement and solves the
bridge equation on the resample, from `start` (see `_bridge_iterate`);
`res[k]`, `ok[k]` and `iters[k]` are what `solve` returned for it.

The resamples are the serial loop's, exactly: replicate `k` takes the `k`-th
pair of draws off `rng` -- `length(l1)` indices into `l1`, then `length(l2)`
into `l2`. A worker takes the next replicate number under one lock, copies
`rng`'s state into its own generator and moves `rng` past the replicate's
draws; it then draws them again from its copy, outside the lock, as counts per
entry (`_bridge_draw_counts!`). So the lock is held only to generate the
numbers, which is what the serial loop did too, and which worker solves a
replicate, and when, changes nothing; every result goes to the replicate's own
slot. The output is the same at any thread count, and the same as solving the
replicates one after another (`workers = 1`, which runs in the calling task).

A resample is held as counts, not as a copy of the ratios: each worker holds
one `_BridgeBootBuffers` (2 bytes per draw) for all the replicates it solves,
and nothing else is allocated per replicate. A pass still runs over every
draw, the third with count 0 adding nothing: cheaper than branching on them.
The solves themselves are serial: the parallelism is across replicates, one
per thread, so no thread is asked to split its own work.

What bounds it on a long chain is the lock. Generating one replicate's
indices takes about 20 ms at 5.8 million draws, one replicate after another,
so 200 replicates take 4 to 6 s at any thread count; the solves (3 or 4
passes of about 13 ms each, from the primary root) run beside that.

If a worker throws, it sets a flag that ends every other worker's solve at its
next solver pass and stops them taking replicates; the exception then
propagates, after all workers have returned (`Threads.@threads` waits for
them), so none is left running. An interrupt in the calling task sets the
same flag.

`progress`, a `_SharedProgress`, is ticked once per finished replicate.
`solve(l1, l2, c1, c2, max_iter, tol; start, stop)` exists for the tests.
"""
function _bridge_bootstrap(l1::Vector{Float64}, l2::Vector{Float64}, rng::AbstractRNG,
                           n_bootstrap::Int, max_iter::Int, tol::Float64;
                           workers::Int = 1, progress = nothing, start::Float64 = NaN,
                           solve::F = _bridge_solve_counts) where {F}
    n1, n2 = length(l1), length(l2)
    res = fill(NaN, max(n_bootstrap, 0))
    ok = fill(false, max(n_bootstrap, 0))      # not a BitVector: one Bool per slot
    its = zeros(Int, max(n_bootstrap, 0))
    n_bootstrap <= 0 && return res, ok, its
    (n1 > 0 && n2 > 0) || return res, ok, its

    next = Ref(0)                    # replicates handed out; guarded by `rng_lock`
    rng_lock = ReentrantLock()
    stop = Threads.Atomic{Bool}(false)
    function work()
        try
            buf = lock(() -> _BridgeBootBuffers(n1, n2, rng), rng_lock)
            while !stop[]
                k = lock(rng_lock) do
                    next[] < n_bootstrap || return 0
                    next[] += 1
                    copy!(buf.rng, rng)
                    _bridge_skip_indices!(buf.idx, rng, n1, n1)
                    _bridge_skip_indices!(buf.idx, rng, n2, n2)
                    next[]
                end
                k == 0 && break
                _bridge_draw_counts!(buf.c1, buf.idx, buf.rng, n1)
                _bridge_draw_counts!(buf.c2, buf.idx, buf.rng, n2)
                r, it, conv = solve(l1, l2, buf.c1, buf.c2, max_iter, tol;
                                    start = start, stop = stop)
                res[k] = r; its[k] = it; ok[k] = conv
                progress === nothing || _tick!(progress)
            end
        catch
            stop[] = true
            rethrow()
        end
        return nothing
    end

    w = clamp(workers, 1, n_bootstrap)
    if w == 1
        work()
    else
        try
            Threads.@threads for _ in 1:w
                work()
            end
        catch
            stop[] = true
            rethrow()
        end
    end
    return res, ok, its
end

# The bootstrap's solve: `_bridge_iterate` on the counts of a resample.
_bridge_solve_counts(l1, l2, c1, c2, max_iter, tol; start = NaN, stop = nothing) =
    _bridge_iterate(l1, l2, max_iter, tol; w1 = c1, w2 = c2, start = start, stop = stop)
