# Post-fit diagnostics on long chains: ESS and R-hat (`_diag_by_param`,
# fit_health, `convergence_report`) from Nereus' own rank-diagnostic kernel,
# the one-sort quantiles of `ParamStats`, chain columns read in one copy, and
# run_job's summary no longer building the blocks it threw away. Each must give
# exactly what MCMCDiagnosticTools or the code it replaced gave.
using Test, Nereus, Random, Statistics, MCMCChains
using Nereus: _diag_by_param, ParamStats, convergence_report, chain_convergence,
              _populate_summary!

# An (iter x param x chain) chain whose parameters differ in scale, tails and
# mixing, with an internals section that the diagnostics must skip.
function _diag_chain(; n_iter = 3_000, n_chain = 8, seed = 7)
    rng = MersenneTwister(seed)
    X = Array{Float64}(undef, n_iter, 6, n_chain)
    for c in 1:n_chain
        ar = 0.0
        for i in 1:n_iter
            ar = 0.95ar + randn(rng)                 # slow mixing
            X[i, 1, c] = ar
            X[i, 2, c] = 3.0 + 0.01randn(rng)        # tight
            X[i, 3, c] = exp(randn(rng))             # skewed
            X[i, 4, c] = randn(rng) + 0.3c           # chains disagree: R-hat > 1
            X[i, 5, c] = -1200.0 + randn(rng)        # an lp-like column
            X[i, 6, c] = rand(rng)                   # internals
        end
    end
    return Chains(X, [:a, :b, :c, :d, :lp, :acc], Dict(:internals => [:acc]))
end

@testset "post-fit diagnostics on long chains" begin
    ch = _diag_chain()
    Threads.nthreads() > 1 || @warn "one thread: the parallel path is not exercised"

    @testset "_diag_by_param: MCMCChains' table, row for row" begin
        for (f, kw) in ((MCMCChains.ess, (; kind = :bulk)), (MCMCChains.ess, (; kind = :tail)),
                        (MCMCChains.ess, (;)), (MCMCChains.rhat, (;)),
                        (MCMCChains.ess_rhat, (;)), (MCMCChains.ess_rhat, (; kind = :rank)))
            ref = f(ch; kw...)
            got = _diag_by_param(f, ch; kw...)
            @test got.name == ref.name
            @test keys(got.nt) == keys(ref.nt)
            @test got.nt.parameters == ref.nt.parameters == [:a, :b, :c, :d, :lp]
            for k in keys(ref.nt)
                @test isequal(collect(got.nt[k]), collect(ref.nt[k]))
            end
        end
        # Indexing as the call sites do.
        @test isequal(_diag_by_param(MCMCChains.ess, ch)[:, :ess], MCMCChains.ess(ch)[:, :ess])
        # One parameter, and an explicit `sections`: straight through to `f`.
        one = ch[:, [:a], :]
        @test isequal(_diag_by_param(MCMCChains.ess, one).nt.ess, MCMCChains.ess(one).nt.ess)
        s = _diag_by_param(MCMCChains.ess, ch; sections = :internals)
        @test s.nt.parameters == [:acc]
        # chain_convergence through it: R-hat flags the disagreeing parameter.
        cc = chain_convergence(ch)
        @test cc["per_parameter"]["d"]["rhat"] > 1.1
        @test cc["per_parameter"]["a"]["ess_bulk"] ==
              MCMCChains.ess(ch; kind = :bulk)[:, :ess][1]
    end

    @testset "ParamStats: one sort, the same quantiles" begin
        rng = MersenneTwister(3)
        for v in (randn(rng, 10_001), exp.(randn(rng, 20_000)), [1.0, 2.0], collect(1.0:7.0))
            p = ParamStats(v)
            q(x) = quantile(v, x)
            @test p.best === median(v)
            @test p.ci1 === (q(0.16), q(0.84))
            @test p.ci2 === (q(0.025), q(0.975))
            @test p.ci3 === (q(0.0015), q(0.9985))
            @test p.unc_lo === median(v) - q(0.16) && p.unc_hi === q(0.84) - median(v)
            # In place, on a copy: the same numbers.
            s = copy(v)
            p2 = Nereus.ParamStats!(s)
            @test issorted(s)
            for f in fieldnames(typeof(p))
                @test getfield(p2, f) === getfield(p, f)
            end
        end
        @test isequal(extrema([1.0, NaN, 2.0]), (NaN, NaN))   # science_derived relies on it
    end

    @testset "fit_* params summary: one sort, the same bounds" begin
        layout_only = (layout = (unfrozen_names = ["a", "c", "missing"],),)
        s = Nereus._summarise_params(ch, layout_only)
        @test sort(collect(keys(s))) == ["a", "c"]
        for nm in ("a", "c")
            v = vec(Array(ch[Symbol(nm)]))
            @test s[nm]["median"] === median(v)
            for (k, p) in (("lo16", 0.16), ("hi84", 0.84), ("lo2sig", 0.025),
                           ("hi2sig", 0.975), ("lo3sig", 0.0015), ("hi3sig", 0.9985))
                @test s[nm][k] === quantile(v, p)
            end
        end
    end

    @testset "convergence_report: per-parameter threads, the serial verdict" begin
        # Native multi-chain layout, n_walkers = number of chains.
        r = convergence_report(ch, 8; io = devnull, params = [:a, :b, :c, :d])
        @test !r.pass
        @test r.worst_rhat_param === :d
        # The same numbers as assessing each parameter alone.
        for p in (:a, :b, :c, :d)
            rp = convergence_report(ch, 8; io = devnull, params = [p])
            ref = MCMCChains.ess_rhat(MCMCChains.Chains(reshape(Array(ch[:, p, :]), :, 1, 8)))
            @test rp.worst_rhat === ref[1, :rhat]
            @test rp.min_ess === ref[1, :ess]
        end
    end

    @testset "_populate_summary!: no blocks that run_job replaces" begin
        res = (log_evidence = -12.5, n_evals = 1000, acceptance_within = [0.3, 0.4],
               acceptance_swap = [0.5])
        s = Dict{String, Any}()
        _populate_summary!(s, res, nothing, ch)
        @test s["log_z"] == -12.5 && s["n_evals"] == 1000
        @test !haskey(s, "params") && !haskey(s, "derived")
    end
end

# ---------------------------------------------------------------------------
# The rank-diagnostic kernel against MCMCDiagnosticTools itself.
# ---------------------------------------------------------------------------
const _MCDT_T = MCMCChains.MCMCDiagnosticTools

# The library's (bulk ESS, rank R-hat, tail ESS) for an (m × C) matrix of draws;
# NaN tail ESS where the library throws.
function _lib_split2(M)
    a = reshape(Matrix{Float64}(M), size(M, 1), size(M, 2), 1)
    er = _MCDT_T.ess_rhat(a; kind = :rank)
    et = try only(_MCDT_T.ess(a; kind = :tail)) catch; nothing end
    return (only(er.ess), only(er.rhat), et === nothing ? NaN : et), et !== nothing
end
# fit_health's: walkers split in half by `_ensemble_split`, then ess_rhat.
function _lib_ensemble(M)
    ch = Chains(reshape(Matrix{Float64}(M), size(M, 1), 1, size(M, 2)), [:p])
    t = MCMCChains.ess_rhat(Nereus._ensemble_split(ch); kind = :rank)
    return (only(collect(t.nt.ess)), only(collect(t.nt.rhat)))
end
_one(M, layout) = only(Nereus._rank_diag_batch([() -> M], layout; n_max = length(M)))

function _ar_draws(rng, m, C; ρ = 0.9, shift = 0.0)
    X = Matrix{Float64}(undef, m, C)
    for c in 1:C
        a = 0.0
        for i in 1:m
            a = ρ * a + randn(rng)
            X[i, c] = a + shift * c
        end
    end
    return X
end

@testset "rank diagnostics: MCMCDiagnosticTools' numbers, bit for bit" begin
    rng = MersenneTwister(11)
    cases = Any[]
    for (m, C) in ((20, 4), (21, 4), (22, 3), (64, 2), (65, 7), (1001, 5), (2000, 8),
                   (3000, 1), (2999, 1), (40, 1))
        push!(cases, ("ar $m×$C", _ar_draws(rng, m, C)))
    end
    push!(cases, ("chains disagree", _ar_draws(rng, 400, 6; ρ = 0.5, shift = 0.7)))
    push!(cases, ("ties", round.(_ar_draws(rng, 500, 4); digits = 1)))
    push!(cases, ("signed zeros", [isodd(i + c) ? -0.0 : 0.0 for i in 1:200, c in 1:4] .+
                                  [i > 150 ? 1.0 : 0.0 for i in 1:200, c in 1:4]))
    push!(cases, ("constant", fill(2.5, 300, 4)))
    push!(cases, ("two values", Float64.(rand(rng, 0:1, 300, 4))))
    push!(cases, ("short", _ar_draws(rng, 12, 4)))
    push!(cases, ("tiny", _ar_draws(rng, 3, 4)))
    with_nan = _ar_draws(rng, 300, 4); with_nan[17, 2] = NaN
    push!(cases, ("a NaN", with_nan))
    with_inf = _ar_draws(rng, 300, 4); with_inf[5, 1] = Inf
    push!(cases, ("an Inf", with_inf))

    for (label, M) in cases
        empty!(Nereus._RANK_DIAG_MEMO)
        r = _one(M, :split2)
        ref, tail_ok = _lib_split2(M)
        @test isequal((r.ess_bulk, r.rhat_rank, r.ess_tail), ref) || (@info label r ref; false)
        @test r.tail_ok == tail_ok
        if size(M, 1) >= 4
            e = _one(M, :ensemble)
            @test isequal((e.ess_bulk, e.rhat_rank), _lib_ensemble(M)) ||
                  (@info "ensemble $label" e _lib_ensemble(M); false)
            @test isnan(e.ess_tail)
        end
        # Remembered: the same numbers again, from the memo.
        @test isequal(_one(M, :split2), r)
    end
end

@testset "rank diagnostics: remembered by content, not by array" begin
    rng = MersenneTwister(5)
    M = _ar_draws(rng, 600, 4)
    empty!(Nereus._RANK_DIAG_MEMO)
    r1 = _one(M, :split2)
    n1 = length(Nereus._RANK_DIAG_MEMO)
    @test n1 == 2                          # even m: :split2 and :ensemble together
    @test isequal(_one(copy(M), :split2), r1)
    @test length(Nereus._RANK_DIAG_MEMO) == n1
    # One draw moved: a different key, the library's new numbers.
    M2 = copy(M); M2[300, 3] += 1e-9
    r2 = _one(M2, :split2)
    @test length(Nereus._RANK_DIAG_MEMO) == n1 + 2
    @test isequal((r2.ess_bulk, r2.rhat_rank, r2.ess_tail), _lib_split2(M2)[1])
    # In place: a mutated array is recomputed, not served stale.
    M[1, 1] = 50.0
    @test isequal(_one(M, :split2).ess_bulk, _lib_split2(M)[1][1])
end

@testset "rank diagnostics through MCMCChains' entry points" begin
    ch = _diag_chain()
    empty!(Nereus._RANK_DIAG_MEMO)
    for (f, kw) in ((MCMCChains.ess, (; kind = :bulk)), (MCMCChains.ess, (; kind = :tail)),
                    (MCMCChains.ess, (;)), (MCMCChains.rhat, (;)),
                    (MCMCChains.ess_rhat, (;)), (MCMCChains.ess_rhat, (; kind = :rank)))
        ref = f(ch; kw...)
        got = _diag_by_param(f, ch; kw...)
        @test got.name == ref.name
        @test keys(got.nt) == keys(ref.nt)
        for k in keys(ref.nt)
            @test isequal(collect(got.nt[k]), collect(ref.nt[k]))
        end
    end
    # A tail ESS the library refuses (NaN draws) is refused the same way.
    X = Array(ch.value.data); X[10, 1, 3] = NaN
    bad = Chains(X, names(ch), Dict(:internals => [:acc]))
    @test_throws ArgumentError MCMCChains.ess(bad; kind = :tail)
    @test_throws ArgumentError _diag_by_param(MCMCChains.ess, bad; kind = :tail)
    @test isequal(collect(_diag_by_param(MCMCChains.ess_rhat, bad).nt.rhat),
                  collect(MCMCChains.ess_rhat(bad).nt.rhat))
    # fit_health's ensemble split, read in place, odd and even lengths.
    for c in (ch, ch[1:2999, :, :])
        ps = [:a, :b, :c, :d]
        r = Nereus._rank_diag_chain(c, ps, :ensemble)
        t = MCMCChains.ess_rhat(Nereus._ensemble_split(c[:, ps, :]); kind = :rank)
        @test isequal([x.ess_bulk for x in r], collect(t.nt.ess))
        @test isequal([x.rhat_rank for x in r], collect(t.nt.rhat))
    end
end

@testset "convergence_report: masked and flat layouts, the old numbers" begin
    # The old per-parameter computation (library calls on per-walker matrices).
    function ref_cond(v, mask, nsteps, nw, widx)
        if mask === nothing
            M = Array{Float64,3}(undef, nsteps, 1, nw)
            for w in 1:nw; M[:, 1, w] = v[widx(w)]; end
            er = MCMCChains.ess_rhat(MCMCChains.Chains(M))
            return (er[1, :rhat], er[1, :ess])
        end
        colv = [(idx = widx(w); v[idx][mask[idx]]) for w in 1:nw]
        nz = filter(>(0), length.(colv))
        isempty(nz) && return (NaN, NaN)
        L = max(20, round(Int, median(nz)))
        kept = [c for c in colv if length(c) ≥ L]
        length(kept) < 4 && return (NaN, NaN)
        M = Array{Float64,3}(undef, L, 1, length(kept))
        for (j, c) in enumerate(kept); M[:, 1, j] = c[1:L]; end
        er = MCMCChains.ess_rhat(MCMCChains.Chains(M))
        return (er[1, :rhat], er[1, :ess])
    end
    rng = MersenneTwister(21)
    nw, ns = 12, 400
    K = _ar_draws(rng, ns, nw; ρ = 0.8) .+ 5
    act = Float64.(rand(rng, ns, nw) .< 0.7)
    act[:, 1:2] .= 0.0                                  # two walkers never on
    g = _ar_draws(rng, ns, nw; ρ = 0.3)
    X = Array{Float64,3}(undef, ns, 3, nw)
    X[:, 1, :] = K; X[:, 2, :] = act; X[:, 3, :] = g
    native = Chains(X, [:K_k1, :planet_active_1, :gamma])
    # The same draws as one flat chain with the walker as the fast index.
    flat = Chains(reshape(permutedims(X, (3, 1, 2)), ns * nw, 3, 1),
                  [:K_k1, :planet_active_1, :gamma])
    for (ch, walker_fast) in ((native, false), (flat, true))
        nflat = ns * nw
        widx(w) = walker_fast ? (w:nw:nflat) : (((w - 1) * ns + 1):(w * ns))
        for p in (:gamma, :K_k1)
            empty!(Nereus._RANK_DIAG_MEMO)
            v = vec(Array(ch[p]))
            mask = p === :K_k1 ? vec(Array(ch[:planet_active_1])) .> 0.5 : nothing
            r, e = ref_cond(v, mask, ns, nw, widx)
            rep = convergence_report(ch, nw; io = devnull, params = [p])
            @test rep.worst_rhat === r
            @test rep.min_ess === e
        end
    end
end

@testset "_draws: one copy, the same vector" begin
    for n_chain in (1, 5)
        ch = _diag_chain(; n_iter = 400, n_chain)
        for s in names(ch)
            ref = vec(Array(ch[s]))
            d = Nereus._draws(ch, s)
            @test typeof(d) === typeof(ref) && d == ref
            @test Nereus._draws(ch, String(s)) == ref
        end
    end
    ch = _diag_chain(; n_iter = 50, n_chain = 2)
    @test_throws Exception Nereus._draws(ch, :no_such_param)
    @test Nereus._draws(Dict(:a => [1.0 2.0; 3.0 4.0]), :a) == [1.0, 3.0, 2.0, 4.0]
end

@testset "multimodality check: the copies' numbers, read in place" begin
    # The check as it was: copies of each column, chain by chain.
    function ref_ratio(ch, p, active)
        mat = Array(ch[p])
        cols = haskey(active, p) ?
            [mat[reshape(active[p], size(mat))[:, k], k] for k in 1:size(mat, 2)] :
            [mat[:, k] for k in 1:size(mat, 2)]
        cols = filter(c -> length(c) >= 2, cols)
        length(cols) < 2 && return nothing
        between = std([median(c) for c in cols])
        within = mean([std(c) for c in cols])
        (between == 0.0 && within == 0.0) && return nothing
        return within <= 0.0 ? (between > 0.0 ? Inf : 0.0) : between / within
    end
    ch = _diag_chain(; n_iter = 501, n_chain = 6)
    ps = names(ch, :parameters)
    rng = MersenneTwister(3)
    n = size(ch, 1) * size(ch, 3)
    m = rand(rng, n) .> 0.4
    m[1:size(ch, 1)] .= false                  # one chain with no active draw
    m[size(ch, 1) + 1:size(ch, 1) + 1] .= true # and one with a single draw
    m[size(ch, 1) + 2:2size(ch, 1)] .= false
    for active in (Dict{Symbol, BitVector}(), Dict(ps[1] => m, ps[3] => m))
        # Every ratio above 0 is an offender, so `details` lists them all.
        c = Nereus._check_multimodality(ch, ps; mode_ratio_max = 0.0, active)
        got = Dict(c.details)
        for p in ps
            r = ref_ratio(ch, p, active)
            if r === nothing || !(r > 0.0)
                @test !haskey(got, string(p))
            else
                @test got[string(p)] === r
            end
        end
    end
end
