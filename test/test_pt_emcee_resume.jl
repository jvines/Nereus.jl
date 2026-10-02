# pt_emcee checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any step and continued with `resume = true` is
# bit-identical to one that ran straight through, so a run that has not mixed
# can simply be extended. The split points below cross everything that carries
# state between steps: the burn-in re-cut of the circular Mo (steps 40 and 80),
# the ladder adaptation (step 50), the stranded-walker prune (step 50), the end
# of burn-in, and a finished run extended past its own end.
using Test
using Nereus
using Random

Random.seed!(7)
const _RS_N = 60
const _RS_T = sort!(400 .* rand(_RS_N))
const _RS_RV = 40.0 .* sin.(2π .* _RS_T ./ 4.23) .+ 1.5 .* randn(_RS_N)

# A fresh target per run: the re-cut moves circular windows on the target itself.
_rs_target(rv = _RS_RV) = build_target(
    planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _RS_T, rv = rv, rv_err = fill(1.5, _RS_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),))

const _RS_KW = (n_temps = 4, n_walkers = 20, n_burnin = 80, seed = 42,
                adapt_ladder = true, show_progress = false, bridge_headline = false)

_rs_run(n_steps; kw...) = (tg = _rs_target(); sample_pt_emcee(tg, tg.data; _RS_KW..., n_steps, kw...))
# (step, parameter, walker); `Array(chains)` would stack the walkers.
_cube(r) = r.chains.value.data

@testset "pt_emcee resume" begin
    full = _rs_run(150)

    path = joinpath(mktempdir(), "pt_emcee_state.jls")
    _rs_run(60; checkpoint = path)                    # stops inside burn-in
    @test isfile(path) && !isfile(path * ".tmp")
    mid = _rs_run(120; checkpoint = path, resume = true)   # a finished run ...
    ext = _rs_run(150; checkpoint = path, resume = true)   # ... extended

    # Identical to the uninterrupted run, bit for bit.
    @test _cube(ext) == _cube(full)
    @test ext.betas == full.betas
    @test ext.acceptance_within == full.acceptance_within
    @test ext.acceptance_swap == full.acceptance_swap
    @test ext.n_evals == full.n_evals
    @test ext.ladder.betas == full.ladder.betas
    @test ext.ladder.swap_rate == full.ladder.swap_rate
    @test isequal(ext.log_evidence, full.log_evidence)
    # The finished 120-step run kept the first 40 post-burn-in steps of it.
    @test _cube(mid) == _cube(full)[1:40, :, :]

    # Resuming at the checkpoint's own step runs nothing and returns the same.
    same = _rs_run(150; checkpoint = path, resume = true)
    @test _cube(same) == _cube(full)

    # Refusals: no checkpoint, a different run, a total shorter than the saved.
    @test_throws ArgumentError _rs_run(150; resume = true)
    @test_throws ArgumentError _rs_run(150; checkpoint = path * ".none", resume = true)
    err = try _rs_run(200; checkpoint = path, resume = true, n_walkers = 24); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("n_walkers", err.msg)
    err = try (tg = _rs_target(_RS_RV .+ 1.0);
               sample_pt_emcee(tg, tg.data; _RS_KW..., n_steps = 200,
                               checkpoint = path, resume = true)); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("data", err.msg)
    @test_throws ArgumentError _rs_run(100; checkpoint = path, resume = true)
end
