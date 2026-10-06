# transdim_pt_emcee checkpoints and resume (src/checkpoint.jl).
#
# The contract, as for pt_emcee: a run stopped at any step and continued with
# `resume = true` is bit-identical to one that ran straight through. The split
# points below cross everything that carries state between steps: the MoMS
# scale adaptation window and its accumulators (steps 50 and 100), the ladder
# adaptation (50, 100), the stranded-walker prune (50), the circular re-cut of
# Mo and the MoMS off-values it moves (50, 100), the end of burn-in (100), the
# informed-birth caches, and a finished run extended past its own end. A second
# target adds the noise menu: noise births, deaths and swaps and the
# planet<->activity swap.
#
# The informed-birth caches are process-global and keyed by RNG objectid, so a
# run's proposals depend on what earlier runs in the process left there
# (`reset_transdim_caches!`). Every run below starts from empty caches, as a
# fresh `run_job` process does.
using Test
using Nereus
using Nereus: Params, Data, InstrumentConfig, NereusTarget, TransDimConfig,
              NoiseModel, ActivityDecorrelation, RV_ONLY, reset_transdim_caches!
using Random
using Serialization

Random.seed!(11)
const _TR_N = 60
const _TR_T = sort!(300 .* rand(_TR_N))
const _TR_RV = 25.0 .* sin.(2π .* _TR_T ./ 6.1) .+ 12.0 .* sin.(2π .* _TR_T ./ 17.3) .+
               1.5 .* randn(_TR_N)

# A fresh target per run: the re-cut moves circular windows on the target itself.
# Two slots, so births happen from one-planet states too: their informed
# proposals come from residuals of the walker's planet as it stood when its
# cache entry was built, which is what makes those caches state.
_tr_planet() = (P = LogUniformPrior(2.0, 30.0), K = LogUniformPrior(1.0, 80.0),
                sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                Mo = UniformPrior(0.0, 2pi))
_tr_target(rv = _TR_RV) = build_target(
    planets = (b = _tr_planet(), c = _tr_planet()),
    rv = (SIM = (data = (t = _TR_T, rv = rv, rv_err = fill(1.5, _TR_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),))

const _TR_KW = (td = TransDimConfig(max_kplanet = 2), n_temps = 4, n_walkers = 20,
                n_burnin = 100, seed = 42, n_birth_tries = 3, n_birth_refine = 2,
                lambda_slide = 0.2, show_progress = false)

_tr_run(n_steps; kw...) = (reset_transdim_caches!(); tg = _tr_target();
                           sample_transdim_pt_emcee(tg, tg.data; _TR_KW..., n_steps, kw...))
# The raw draws: one row per (kept step, walker), walkers fastest.
_cube(r) = r.chains.value.data

# Every diagnostic the result carries, besides the draws.
function _same_run(a, b)
    @test _cube(a) == _cube(b)
    for f in (:acceptance_within, :acceptance_swap, :acceptance_transdim, :betas,
              :n_evals, :td_proposed, :td_accepted, :noise_td_proposed,
              :noise_td_accepted, :planet_birth_proposed, :planet_birth_accepted,
              :planet_death_proposed, :planet_death_accepted)
        @test getfield(a, f) == getfield(b, f)
    end
    @test isequal(a.log_evidence, b.log_evidence)
    for f in fieldnames(typeof(a.ladder))
        @test isequal(getfield(a.ladder, f), getfield(b.ladder, f))
    end
end

@testset "transdim_pt_emcee resume" begin
    full = _tr_run(160)

    path = joinpath(mktempdir(), "transdim_pt_emcee_state.jls")
    _tr_run(30; checkpoint = path)                       # inside burn-in
    @test isfile(path) && !isfile(path * ".tmp")
    # The informed births reached their caches, so the resume has them to carry.
    @test !isempty(open(deserialize, path).state.informed)
    _tr_run(100; checkpoint = path, resume = true)       # to the end of burn-in
    mid = _tr_run(130; checkpoint = path, resume = true) # a finished run ...
    ext = _tr_run(160; checkpoint = path, resume = true) # ... extended

    # Identical to the uninterrupted run, bit for bit.
    _same_run(ext, full)
    # The finished 130-step run kept the first 30 of its 60 post-burn-in steps.
    n_mid = size(_cube(full), 1) ÷ 60 * 30
    @test size(_cube(mid), 1) == n_mid
    @test _cube(mid) == _cube(full)[1:n_mid, :, :]

    # Resuming at the checkpoint's own step runs nothing and returns the same.
    same = _tr_run(160; checkpoint = path, resume = true)
    _same_run(same, full)

    # Refusals: no checkpoint, a different run, a total shorter than the saved.
    @test_throws ArgumentError _tr_run(160; resume = true)
    @test_throws ArgumentError _tr_run(160; checkpoint = path * ".none", resume = true)
    err = try _tr_run(200; checkpoint = path, resume = true, n_walkers = 32); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("n_walkers", err.msg)
    err = try _tr_run(200; checkpoint = path, resume = true,
                      td = TransDimConfig(max_kplanet = 2, transdim_fraction = 0.3)); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("td", err.msg)
    err = try (tg = _tr_target(_TR_RV .+ 1.0);
               sample_transdim_pt_emcee(tg, tg.data; _TR_KW..., n_steps = 200,
                                        checkpoint = path, resume = true)); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("data", err.msg)
    @test_throws ArgumentError _tr_run(100; checkpoint = path, resume = true)
end

@testset "transdim_pt_emcee resume with a noise menu" begin
    rng = MersenneTwister(20261002)
    N = 50
    t = sort(rand(rng, N)) .* 70.0
    act = 5.0 .* sin.(2π .* t ./ 8.0)
    rv = act .+ 12.0 .* sin.(2π .* t ./ 3.3) .+ 1.5 .* randn(rng, N)
    data = Data(t_rv = t, rv = rv, rv_err = fill(1.5, N), rv_inst = ones(Int, N),
                indicators = Dict("bis" => act .+ 2.0 .* randn(rng, N),
                                  "bis2" => act .+ 2.0 .* randn(rng, N)),
                indicator_errs = Dict("bis" => fill(1.0, N), "bis2" => fill(1.0, N)))
    ad  = ActivityDecorrelation(indicators = ["bis"])
    ad2 = ActivityDecorrelation(indicators = ["bis2"])
    td = TransDimConfig(; max_kplanet = 1, planets = true, noise = true,
                          toggleable = NoiseModel[ad, ad2],
                          noise_exclusion_groups = [NoiseModel[ad, ad2]])
    mk() = (p = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                       instruments = InstrumentConfig(rv = ["I1"]), data = data,
                       M_s = 1.0, noise_models = NoiseModel[ad, ad2],
                       transdim_noise = true, stability = :none);
            NereusTarget(p, data; unconstrained = false))
    run(n_steps; kw...) = (reset_transdim_caches!(); tg = mk();
        sample_transdim_pt_emcee(tg, data; td, n_temps = 4, n_walkers = 16,
                                 n_burnin = 40, seed = 3, n_birth_tries = 2,
                                 n_birth_refine = 2, show_progress = false,
                                 n_steps, kw...))

    full = run(80)
    path = joinpath(mktempdir(), "transdim_pt_emcee_state.jls")
    run(20; checkpoint = path)
    ext = run(80; checkpoint = path, resume = true)
    _same_run(ext, full)
    @test sum(full.noise_td_proposed) > 0
end
