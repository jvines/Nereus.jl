# ess checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any step and continued with `resume = true` is
# bit-identical to one that ran straight through, so a run that has not mixed
# can simply be extended. ESS adapts nothing; what crosses a step is the RNG,
# the current draw with its log-likelihood and the draws kept so far, and the
# one change of phase is the end of burn-in. The split points below: inside
# burn-in (step 40 of 100), exactly at its end, a finished run extended past
# its own end, and the same for two chains.
using Test
using Nereus
using Random

Random.seed!(11)
const _ER_N = 60
const _ER_T = sort!(400 .* rand(_ER_N))
const _ER_RV = 40.0 .* sin.(2π .* _ER_T ./ 4.23) .+ 1.5 .* randn(_ER_N)

_er_target(rv = _ER_RV) = build_target(
    planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _ER_T, rv = rv, rv_err = fill(1.5, _ER_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),))
const _ER_TG = _er_target()
const _ER_KW = (n_burnin = 100, seed = 42)

_er_run(n_samples; kw...) = sample_ess(_ER_TG, _ER_TG.data; _ER_KW..., n_samples, kw...)
# A run killed after ESS step `k` (burn-in included), which `n_samples` cannot
# express inside burn-in: the loop stops at `k` and leaves the checkpoint an
# interval write at step `k` of a longer run would have left.
_er_stop(k, path; resume = false) = Nereus._sample_ess(_ER_TG, _ER_TG.data;
    _ER_KW..., n_steps = k, init = nothing, n_chains = 1, checkpoint = path,
    checkpoint_interval = 900.0, resume)
_draws(c) = c.value.data

function _same_chains(a, b)
    return _draws(a) == _draws(b) && a.value == b.value && a.name_map == b.name_map &&
           isequal(a.logevidence, b.logevidence) && a.info == b.info
end

@testset "ess resume" begin
    full = _er_run(300)

    # Killed inside burn-in, continued to its end, then past it.
    dir = mktempdir()
    path = joinpath(dir, "ess_state.jls")
    r40 = only(_er_stop(40, path))
    @test isfile(path) && !isfile(path * ".tmp")
    @test size(r40.draws, 1) == 0 && r40.n_post == 0
    r100 = only(_er_stop(100, path; resume = true))      # exactly the end of burn-in
    @test size(r100.draws, 1) == 0
    mid = _er_run(150; checkpoint = path, resume = true)    # a finished run ...
    ext = _er_run(300; checkpoint = path, resume = true)    # ... extended

    # Identical to the uninterrupted run, bit for bit.
    @test _same_chains(ext, full)
    # The finished 150-sample run kept the first of those draws.
    @test _draws(mid) == _draws(full)[1:size(_draws(mid), 1), :, :]
    @test size(_draws(mid), 1) > 0

    # A fresh run stopped exactly at the end of burn-in, then continued.
    path_b = joinpath(dir, "boundary.jls")
    _er_stop(100, path_b)
    @test _same_chains(_er_run(300; checkpoint = path_b, resume = true), full)

    # A run that wrote its checkpoint at every step ends with the same state.
    path_i = joinpath(dir, "interval.jls")
    _er_run(150; checkpoint = path_i, checkpoint_interval = 0.0)
    @test _same_chains(_er_run(300; checkpoint = path_i, resume = true), full)

    # Resuming at the checkpoint's own step runs nothing and returns the same.
    same = _er_run(300; checkpoint = path, resume = true)
    @test _same_chains(same, full)

    # Two chains: one state file each, both continued.
    full2 = _er_run(200; n_chains = 2)
    path2 = joinpath(dir, "ess2.jls")
    _er_run(60; n_chains = 2, checkpoint = path2)
    @test isfile(joinpath(dir, "ess2.chain1.jls")) && isfile(joinpath(dir, "ess2.chain2.jls"))
    @test !isfile(path2)
    ext2 = _er_run(200; n_chains = 2, checkpoint = path2, resume = true)
    @test size(_draws(ext2), 3) == 2
    @test _same_chains(ext2, full2)

    # Refusals: no checkpoint, a different run, a total shorter than the saved.
    @test_throws ArgumentError _er_run(300; resume = true)
    @test_throws ArgumentError _er_run(300; checkpoint = path * ".none", resume = true)
    for (kw, field) in (((seed = 43,), "seed"), ((n_burnin = 120,), "n_burnin"),
                        ((init = [4.23, 40.0, 0.0, 0.0, 3.0, 0.0, 2.0],), "init"))
        err = try _er_run(400; checkpoint = path, resume = true, kw...); nothing
              catch e; e end
        @test err isa ArgumentError && occursin(field, err.msg)
    end
    err = try (tg = _er_target(_ER_RV .+ 1.0);
               sample_ess(tg, tg.data; _ER_KW..., n_samples = 400,
                          checkpoint = path, resume = true)); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("data", err.msg)
    err = try _er_run(400; n_chains = 3, checkpoint = path2, resume = true); nothing
          catch e; e end
    @test err isa ArgumentError
    err = try _er_run(400; n_chains = 1, checkpoint = joinpath(dir, "ess2.chain1.jls"),
                      resume = true); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("n_chains", err.msg)
    @test_throws ArgumentError _er_run(250; checkpoint = path, resume = true)

    # run_job checkpoints by default into output_dir, and a job with
    # `"resume": true` and a larger `n_samples` continues it.
    out = mktempdir()
    job(kw) = Nereus._dispatch_sampler(
        Dict(:output_dir => out, :sampler => Dict(:name => "ess", :kwargs => kw)),
        _ER_TG, _ER_TG.data, 42)
    job(Dict("n_burnin" => 100, "n_samples" => 150))
    @test isfile(joinpath(out, "ess_state.jls"))
    res = job(Dict("n_burnin" => 100, "n_samples" => 300, "resume" => true))
    @test _same_chains(res.chains, full)
end
