# Post-fit diagnostics on long chains: the per-parameter threading of ESS and
# R-hat (`_diag_by_param`), the one-sort quantiles of `ParamStats`, the
# per-parameter threading of `convergence_report`, and run_job's summary no
# longer building the blocks it threw away. Each must give exactly what the
# serial code gave.
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
