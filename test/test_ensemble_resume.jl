# ensemble checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any step and continued with `resume = true` is
# bit-identical to one that ran straight through, so a run that has not mixed
# can simply be extended. The ensemble has no adaptation; what carries over
# between steps is the walkers, their log-densities, the RNG, the thinning phase
# and the burn-in cut. The split points below fall inside burn-in on a step
# between two thinned records (41), on the burn-in boundary (60), one odd step
# past it (63), and on a finished run extended past its own end (100 -> 150).
using Test
using Nereus
using Random

Random.seed!(7)
const _ES_N = 60
const _ES_T = sort!(400 .* rand(_ES_N))
const _ES_RV = 40.0 .* sin.(2π .* _ES_T ./ 4.23) .+ 1.5 .* randn(_ES_N)

_es_target(rv = _ES_RV) = build_target(
    planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _ES_T, rv = rv, rv_err = fill(1.5, _ES_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),))

# n_burnin counts thinned records: burn-in ends at step 60, the first kept
# record is step 62.
const _ES_KW = (n_walkers = 20, n_burnin = 30, thinning = 2, seed = 42,
                show_progress = false)

_es_run(n_steps; kw...) = sample_ensemble(_es_target(); _ES_KW..., n_steps, kw...)
_val(ch) = ch.value.data
const _ES_W = 20   # rows per kept record: one per walker

@testset "ensemble resume" begin
    # The loop is AffineInvariantMCMC.sample's, written out so it can stop and
    # carry on. Same draws, same arithmetic: the package's own chain, bit for bit.
    @testset "the step is AffineInvariantMCMC's" begin
        logp(v) = -0.5 * sum(abs2, v ./ (1.0:length(v)))
        for (nw, thin) in ((10, 1), (11, 3))
            x0 = randn(MersenneTwister(3), 4, nw)
            ref, refll = Nereus.AffineInvariantMCMC.sample(logp, nw, x0, 30, thin;
                                                           rng = MersenneTwister(5))
            rng = MersenneTwister(5)
            x = copy(x0)
            ll = [logp(x[:, j]) for j in 1:nw]
            dv = Nereus._stretch_divisions(nw)
            ch = similar(ref); chll = similar(refll)
            for i in 1:30
                Nereus._stretch_step!(logp, x, ll, rng, dv)
                if i % thin == 0
                    ch[:, :, i ÷ thin] = x
                    chll[:, i ÷ thin] = ll
                end
            end
            @test ch == ref
            @test chll == refll
        end
    end

    full = _es_run(150)

    # Checkpoints written along the way do not touch the chain.
    p0 = joinpath(mktempdir(), "ensemble_state.jls")
    @test _val(_es_run(150; checkpoint = p0, checkpoint_interval = 0)) == _val(full)

    path = joinpath(mktempdir(), "ensemble_state.jls")
    # Too short to keep a draw, so these refuse to summarise, after checkpointing.
    @test_throws ArgumentError _es_run(41; checkpoint = path)              # inside burn-in
    @test isfile(path) && !isfile(path * ".tmp")
    @test_throws ArgumentError _es_run(60; checkpoint = path, resume = true)  # its boundary
    one = _es_run(63; checkpoint = path, resume = true)                   # just past it
    mid = _es_run(100; checkpoint = path, resume = true)   # a finished run ...
    ext = _es_run(150; checkpoint = path, resume = true)   # ... extended

    # Identical to the uninterrupted run, bit for bit.
    @test _val(ext) == _val(full)
    @test names(ext) == names(full)
    @test _val(one) == _val(full)[1:_ES_W, :, :]
    # The finished 100-step run kept its 20 post-burn-in records.
    @test _val(mid) == _val(full)[1:20 * _ES_W, :, :]

    # Resuming at the checkpoint's own step runs nothing and returns the same.
    same = _es_run(150; checkpoint = path, resume = true)
    @test _val(same) == _val(full)

    # Refusals: no checkpoint, a different run, a total shorter than the saved.
    @test_throws ArgumentError _es_run(150; resume = true)
    @test_throws ArgumentError _es_run(150; checkpoint = path * ".none", resume = true)
    err = try _es_run(200; checkpoint = path, resume = true, n_walkers = 24); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("n_walkers", err.msg)
    err = try _es_run(200; checkpoint = path, resume = true, thinning = 1); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("thinning", err.msg)
    err = try sample_ensemble(_es_target(_ES_RV .+ 1.0); _ES_KW..., n_steps = 200,
                              checkpoint = path, resume = true); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("data", err.msg)
    # The walkers are stored in the sampling chart: a moved circular window is a
    # different run.
    tg = _es_target()
    i = only(Nereus.circular_indices(tg.params))
    Nereus.set_circular_window!(tg.params, i, 1.0; transforms = (tg.transform,))
    err = try sample_ensemble(tg; _ES_KW..., n_steps = 200, checkpoint = path,
                              resume = true); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("windows", err.msg)
    @test_throws ArgumentError _es_run(100; checkpoint = path, resume = true)

    # Multi-chain: one state file per chain, each continued on its own thread.
    full2 = _es_run(90; n_chains = 2)
    p2 = joinpath(mktempdir(), "ensemble_state.jls")
    _es_run(70; n_chains = 2, checkpoint = p2)
    @test all(isfile(Nereus._chain_checkpoint_path(p2, c)) for c in 1:2)
    ext2 = _es_run(90; n_chains = 2, checkpoint = p2, resume = true)
    @test _val(ext2) == _val(full2)
    err = try _es_run(90; checkpoint = Nereus._chain_checkpoint_path(p2, 1),
                      resume = true); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("n_chains", err.msg)

    # run_job checkpoints every sampler that declares `checkpoint`.
    @test any(m -> :checkpoint in Base.kwarg_decl(m), methods(sample_ensemble))
end
