# rjmcmc checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any iteration and continued with
# `resume = true` is bit-identical to one that ran straight through, so a run
# that has not mixed can simply be extended. The split points below cross
# everything that carries state between iterations: the RWM scale adaptation
# (all of warmup), the circular warmup trace (recorded over iterations
# 201-400), the re-cut that ends it (400), the informed-birth periodogram
# caches, the noise toggle, and a finished run extended past its own end.
#
# `n_samples` counts only the draws after warmup, so no finished run ends
# inside warmup; a killed one does, and `_RJMCMC_STOP_AT` makes the kill.
using Test
using Nereus
using Nereus: Params, Data, InstrumentConfig, NereusTarget, TransDimConfig,
              NoiseModel, ActivityDecorrelation
using Random
using Serialization

const _RJ_N = 50
const _RJ_RNG = MersenneTwister(99)
const _RJ_T = sort!(80.0 .* rand(_RJ_RNG, _RJ_N))
const _RJ_ACT = 3.0 .* sin.(2π .* _RJ_T ./ 11.0)
# The phase puts Mo's posterior across the 0/2π seam, so the re-cut moves it.
const _RJ_RV = 30.0 .* sin.(2π .* _RJ_T ./ 4.23 .+ 6.0) .+ _RJ_ACT .+
               1.5 .* randn(_RJ_RNG, _RJ_N)
const _RJ_BIS = _RJ_ACT .+ randn(_RJ_RNG, _RJ_N)

# A fresh target per run: the re-cut moves circular windows on the target
# itself. And a fresh `td` with it: toggleable noise models are matched to the
# target's by identity.
function _rj_setup(rv; td_kw = (;), params_kw = (;), data_kw = (;))
    data = Data(; t_rv = _RJ_T, rv = rv, rv_err = fill(1.5, _RJ_N),
                  rv_inst = ones(Int, _RJ_N), indicators = Dict("bis" => _RJ_BIS),
                  indicator_errs = Dict("bis" => fill(1.0, _RJ_N)), data_kw...)
    ad = ActivityDecorrelation(indicators = ["bis"])
    params = Params(; max_kplanet = 2, planet_modes = fill(RV_ONLY, 2),
                      instruments = InstrumentConfig(rv = ["I1"]), data = data,
                      M_s = 1.0, noise_models = NoiseModel[ad], transdim_noise = true,
                      priors = Dict{String,PriorSpec}(
                          "P_k1" => LogUniformPrior(2.0, 20.0),
                          "P_k2" => LogUniformPrior(2.0, 20.0),
                          "K_k1" => LogUniformPrior(1.0, 100.0),
                          "K_k2" => LogUniformPrior(1.0, 100.0)), params_kw...)
    td = TransDimConfig(; max_kplanet = 2, noise = true, toggleable = NoiseModel[ad],
                          transdim_fraction = 0.3, alias_jump_fraction = 0.2, td_kw...)
    return NereusTarget(params, data; unconstrained = false), data, td
end

const _RJ_KW = (n_warmup = 400, seed = 5, within_model = :rwm, show_progress = false)

# The informed-birth caches are process-global: every run starts them cold, as
# a fresh process does.
function _rj_run(n_samples; rv = _RJ_RV, td_kw = (;), params_kw = (;), data_kw = (;),
                 kw...)
    tg, data, td = _rj_setup(rv; td_kw, params_kw, data_kw)
    Nereus.reset_transdim_caches!()
    chains, n_evals = sample_rjmcmc(tg, data; td, _RJ_KW..., n_samples, kw...)
    return (; chains, n_evals, windows = Nereus.circular_windows(tg.params))
end

# A run killed after iteration `at`, its checkpoint written.
function _rj_killed(at, n_samples; kw...)
    Nereus._RJMCMC_STOP_AT[] = at
    try
        _rj_run(n_samples; kw...)
        return false
    catch e
        e isa InterruptException || rethrow()
        return true
    finally
        Nereus._RJMCMC_STOP_AT[] = 0
    end
end

_iter(path) = open(deserialize, path).state.chains[1].iter
_cube(r) = r.chains.value.data

@testset "rjmcmc resume" begin
    @test :checkpoint in Base.kwarg_decl(first(methods(sample_rjmcmc)))  # run_job's default

    full = _rj_run(600)
    # Everything the splits must carry was exercised.
    @test full.windows != Nereus.circular_windows(_rj_setup(_RJ_RV)[1].params)
    @test 0 < sum(full.chains[:n_planets]) < 2 * 600
    @test 0 < sum(full.chains[:noise_active_1]) < 600

    path = joinpath(mktempdir(), "rjmcmc_state.jls")
    @test _rj_killed(150, 600; checkpoint = path)      # warmup, before the trace
    @test isfile(path) && !isfile(path * ".tmp") && _iter(path) == 150
    @test _rj_killed(300, 600; checkpoint = path, resume = true)   # inside the trace
    @test _iter(path) == 300
    @test _rj_killed(399, 600; checkpoint = path, resume = true)
    # The state a straight run saves on the eve of the re-cut, the trace and
    # the RWM adaptation included, is the state the split run saves there.
    straight = joinpath(mktempdir(), "rjmcmc_state.jls")
    @test _rj_killed(399, 600; checkpoint = straight)
    a, b = (open(deserialize, p).state.chains[1] for p in (straight, path))
    @test a.circ.tick == b.circ.tick == 199 && a.circ.draws == b.circ.draws
    for f in (:values, :log_pi, :log_L, :n_evals, :widths)
        @test getfield(a, f) == getfield(b, f)
    end
    for f in (:rwm_sigmas, :rwm_attempts, :rwm_accepts)
        @test getfield(a.ws, f) == getfield(b.ws, f)
    end
    @test _rj_killed(400, 600; checkpoint = path, resume = true)   # the re-cut
    @test _iter(path) == 400
    mid = _rj_run(250; checkpoint = path, resume = true)   # a finished run ...
    ext = _rj_run(600; checkpoint = path, resume = true)   # ... extended

    # Identical to the uninterrupted run, bit for bit.
    @test _cube(ext) == _cube(full)
    @test names(ext.chains) == names(full.chains)
    @test ext.n_evals == full.n_evals
    @test ext.windows == full.windows
    # The finished 250-sample run kept the first 250 draws of it.
    @test _cube(mid) == _cube(full)[1:250, :, :]

    # Resuming at the checkpoint's own iteration runs nothing and returns the same.
    same = _rj_run(600; checkpoint = path, resume = true)
    @test _cube(same) == _cube(full)
    @test same.n_evals == full.n_evals

    # Refusals: no checkpoint, a different run, a total shorter than the saved.
    @test_throws ArgumentError _rj_run(600; resume = true)
    @test_throws ArgumentError _rj_run(600; checkpoint = path * ".none", resume = true)
    for (kw, field) in (((n_warmup = 300,), "n_warmup"), ((seed = 6,), "seed"),
                        ((within_model = :slice,), "within_model"),
                        ((td_kw = (alias_jump_fraction = 0.1,),), "td_alias_jump_fraction"),
                        ((rv = _RJ_RV .+ 1.0,), "data"),
                        # Model settings that change no name and no prior, and
                        # a scalar of the data: each changes the posterior.
                        ((params_kw = (stability = :none,),), "model"),
                        ((params_kw = (M_s = 0.8,),), "model"),
                        ((data_kw = (t_ref = 10.0,),), "data"))
        err = try _rj_run(800; checkpoint = path, resume = true, kw...); nothing
              catch e; e end
        @test err isa ArgumentError && occursin(field, err.msg)
    end
    @test_throws ArgumentError _rj_run(500; checkpoint = path, resume = true)
end

@testset "rjmcmc resume, two chains" begin
    kw = (n_chains = 2, within_model = :slice)
    full = _rj_run(400; kw...)                       # 200 draws per chain

    path = joinpath(mktempdir(), "rjmcmc_state.jls")
    _rj_run(200; kw..., checkpoint = path)           # finished at 100 per chain
    ext = _rj_run(400; kw..., checkpoint = path, resume = true)
    @test _cube(ext) == _cube(full)
    @test ext.n_evals == full.n_evals

    # A chain the checkpoint holds nothing for (killed before its first
    # checkpoint) starts from its seed, which is where it was.
    raw = open(deserialize, path)
    Nereus.write_checkpoint(path, "rjmcmc", raw.fingerprint,
                            (; chains = Any[raw.state.chains[1], nothing]))
    redo = _rj_run(400; kw..., checkpoint = path, resume = true)
    @test _cube(redo) == _cube(full)
    @test redo.n_evals == full.n_evals
end

@testset "rjmcmc fingerprint reads the whole model" begin
    fp(tg, data) = Nereus.run_fingerprint(tg.params, data)
    # A target rebuilt from scratch, new vectors and Dicts throughout, matches.
    @test fp(_rj_setup(_RJ_RV)[1:2]...) == fp(_rj_setup(_RJ_RV)[1:2]...)

    # Base.hash reads only a sample of an array of 8192 entries or more. Find a
    # point of a 20000-point light curve it skips: changing that point must
    # still change the fingerprint.
    n = 20_000
    t = collect(range(0.0, 27.0; length = n))
    f = 1.0 .+ 1e-4 .* sin.(t)
    h0 = hash(f)
    i = findfirst(1:n) do i
        old = f[i]
        f[i] = old + 1e-6
        missed = hash(f) == h0
        f[i] = old
        missed
    end
    @test i !== nothing
    g = copy(f)
    g[i] += 1e-6
    phot(flux) = Data(; t_rv = _RJ_T, rv = _RJ_RV, rv_err = fill(1.5, _RJ_N),
                        t_phot = t, flux = flux, flux_err = fill(1e-4, n))
    tg = _rj_setup(_RJ_RV)[1]
    @test fp(tg, phot(f)).data != fp(tg, phot(g)).data
    @test fp(tg, phot(f)).data == fp(tg, phot(copy(f))).data
end
