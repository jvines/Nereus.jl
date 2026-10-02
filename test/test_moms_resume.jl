# moms checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any iteration and continued with
# `resume = true` is bit-identical to one that ran straight through, so a run
# that has not mixed can simply be extended. The split points below cross
# everything that carries state between iterations: the MoMS scale adaptation
# (windows of 50 iterations), the RWM scale adaptation, the circular warmup
# trace (iterations 61-120) and its re-cut at the end of warmup, the end of
# warmup itself, and a finished run extended past its own end. A mid-warmup
# stop is what a kill leaves behind; the `_stop_after` hook makes one at a
# chosen iteration.
using Test
using Nereus
using Random
using Nereus: ActivityDecorrelation, Data, InstrumentConfig, NereusTarget, NoiseModel,
              Params, PlanetDataSources, PriorSpec

Random.seed!(11)
const _MR_N = 50
const _MR_T = sort!(300 .* rand(_MR_N))
const _MR_ERR = 1.5 .* randn(_MR_N)

# Two same-mode slots, so births permute and the two Mo share a window. Tight
# priors on P, K and the eccentricity let a birth land on the planet; with ω
# pinned, Mo is set by the data, and the data put it on the seam at 0 = 2π, so
# the re-cut at the end of warmup moves the window.
_mr_planet() = (P = UniformPrior(4.225, 4.235), K = UniformPrior(35.0, 45.0),
                sesinw = UniformPrior(0.25, 0.35), secosw = UniformPrior(0.35, 0.45),
                Mo = UniformPrior(0.0, 2pi))
# A fresh target per run: the re-cut moves circular windows on the target itself.
# `tkw` are further build_target settings.
_mr_target(rv = zeros(_MR_N); tkw...) = build_target(; planets = (b = _mr_planet(), c = _mr_planet()),
    rv = (SIM = (data = (t = _MR_T, rv = rv, rv_err = fill(1.5, _MR_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),), tkw...)
const _MR_RV = let (e, ω) = Nereus.sesinw_to_ew(0.3, 0.4)
    Nereus.rv_keplerian.(_MR_T, 4.23, 40.0, e, ω, 0.0, _mr_target().data.t_ref) .+ _MR_ERR
end

const _MR_TD = TransDimConfig(max_kplanet = 2, transdim_fraction = 0.3)
# init_scale = 1.5: births spread over the whole Mo window, not just its middle.
const _MR_KW = (n_warmup = 120, seed = 5, init_scale = 1.5, show_progress = false)

# (chains, evals, strategy, the target's circular windows afterwards)
function _mr_run(n_samples; td = _MR_TD, rv = _MR_RV, tkw = (;), kw...)
    tg = _mr_target(rv; tkw...)
    ch, ev, st = sample_moms(tg, tg.data; td, _MR_KW..., n_samples, kw...)
    return (chains = ch, evals = ev, strategy = st,
            windows = Nereus.circular_windows(tg.params))
end
_draws(r) = r.chains.value.data

function _same(a, b)
    @test _draws(a) == _draws(b)
    @test names(a.chains) == names(b.chains)
    @test a.evals == b.evals
    @test a.strategy.scales == b.strategy.scales
    @test a.strategy.off_values == b.strategy.off_values
    @test a.windows == b.windows
end

@testset "moms resume" begin
    @testset "within_model = $wm" for wm in (:slice, :rwm)
        full = _mr_run(200; within_model = wm)
        # The re-cut moved a window, so the restore of windows is exercised.
        @test full.windows != Nereus.circular_windows(_mr_target().params)

        path = joinpath(mktempdir(), "moms_state.jls")
        ck(n; kw...) = _mr_run(n; within_model = wm, checkpoint = path, kw...)
        r40 = ck(200; _stop_after = 40)            # first adaptation window
        @test isfile(path) && !isfile(path * ".tmp")
        @test size(_draws(r40), 1) == 0
        ck(200; resume = true, _stop_after = 90)   # circular trace, 2nd window
        wu = ck(0; resume = true)                  # end of warmup, after the re-cut
        @test size(_draws(wu), 1) == 0
        mid = ck(100; resume = true)               # a finished run ...
        ext = ck(200; resume = true)               # ... extended
        _same(ext, full)
        # The finished 100-draw run kept the first 100 draws of it.
        @test _draws(mid) == _draws(full)[1:100, :, :]

        # Resuming at the checkpoint's own iteration runs nothing and returns the same.
        _same(ck(200; resume = true), full)
        # The whole saved state, not just what reaches the draws, is the same at
        # a common iteration inside the circular trace: the re-cut is coarse
        # enough to come out the same from part of the trace, the state is not.
        straight = joinpath(mktempdir(), "moms_state.jls")
        _mr_run(200; within_model = wm, checkpoint = straight, _stop_after = 100)
        ck(200; _stop_after = 40)
        ck(200; resume = true, _stop_after = 90)
        ck(200; resume = true, _stop_after = 100)
        a, b = Nereus.deserialize(straight).state, Nereus.deserialize(path).state
        @test a.iter == b.iter == 100
        @test b.circ !== nothing && !isempty(b.circ.draws[1])
        for k in keys(a)
            @test (k, isequal(a[k], b[k])) == (k, true)   # names the field that differs
        end

        # Writing the state at every iteration leaves the chain as it was.
        _same(_mr_run(200; within_model = wm, checkpoint = joinpath(mktempdir(), "s.jls"),
                      checkpoint_interval = 0), full)
    end

    @testset "n_chains = 2" begin
        full = _mr_run(200; n_chains = 2)
        path = joinpath(mktempdir(), "moms_state.jls")
        _mr_run(200; n_chains = 2, checkpoint = path, _stop_after = 90)
        @test isfile(joinpath(dirname(path), "moms_state_chain1.jls"))
        @test isfile(joinpath(dirname(path), "moms_state_chain2.jls"))
        @test !isfile(path)
        _mr_run(100; n_chains = 2, checkpoint = path, resume = true)
        ext = _mr_run(200; n_chains = 2, checkpoint = path, resume = true)
        @test _draws(ext) == _draws(full)
        @test ext.evals == full.evals
        @test ext.strategy.scales == full.strategy.scales
    end

    @testset "noise toggles and swaps" begin
        # Two exchangeable activity models in one exclusion group, no planets:
        # the noise birth, death and within-group swap carry the state.
        rng = MersenneTwister(20260810)
        n = 60
        t = sort(rand(rng, n)) .* 70.0
        act = sin.(2π .* t ./ 8.0)   # weak: neither model is certain, so they toggle
        data = Data(t_rv = t, rv = act .+ 1.5 .* randn(rng, n), rv_err = fill(1.5, n),
                    rv_inst = ones(Int, n),
                    indicators = Dict("bis" => act .+ 2.0 .* randn(rng, n),
                                      "bis2" => act .+ 2.0 .* randn(rng, n)),
                    indicator_errs = Dict("bis" => fill(1.0, n), "bis2" => fill(1.0, n)))
        ad, rot = (ActivityDecorrelation(indicators = [s]) for s in ("bis", "bis2"))
        td_n = TransDimConfig(; max_kplanet = 0, planets = false, noise = true,
                                toggleable = NoiseModel[ad, rot],
                                noise_exclusion_groups = [NoiseModel[ad, rot]],
                                transdim_fraction = 0.5)
        nrun(n_samples; td = td_n, kw...) = sample_moms(
            NereusTarget(Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                    instruments = InstrumentConfig(rv = ["I1"]), data = data,
                    priors = Dict{String,PriorSpec}("gamma_I1" => UniformPrior(-20.0, 20.0),
                                                    "sigma_I1" => LogUniformPrior(0.2, 12.0)),
                    noise_models = NoiseModel[ad, rot], transdim_noise = true,
                    stability = :none), data),
            data; td, n_warmup = 100, n_samples, seed = 1, show_progress = false, kw...)
        full = nrun(200)
        on = full[1].value.data[:, end-1:end, 1]
        @test any(c -> 0 < sum(c) < length(c), eachcol(on))   # the toggles moved
        path = joinpath(mktempdir(), "moms_state.jls")
        nrun(200; checkpoint = path, _stop_after = 60)
        nrun(50; checkpoint = path, resume = true)
        ext = nrun(200; checkpoint = path, resume = true)
        @test ext[1].value.data == full[1].value.data
        @test ext[2] == full[2]
        # Another exclusion grouping is another run.
        td2 = TransDimConfig(; max_kplanet = 0, planets = false, noise = true,
                               toggleable = NoiseModel[ad, rot], transdim_fraction = 0.5)
        err = try nrun(300; checkpoint = path, resume = true, td = td2); nothing
              catch e; e end
        @test err isa ArgumentError && occursin("noise_exclusion_groups", err.msg)
        # So is another setting of the process-wide informed noise births. The
        # AD one changes every ActivityDecorrelation birth (an OLS draw and its
        # Hastings density instead of a prior draw), so these chains part.
        for (sw, field) in ((Nereus.AD_INFORMED_BIRTH, "ad_informed_birth"),
                            (Nereus.GP_INFORMED_BIRTH, "gp_informed_birth"))
            was = sw[]
            err = try
                sw[] = !was
                nrun(300; checkpoint = path, resume = true); nothing
            catch e; e
            finally
                sw[] = was
            end
            @test err isa ArgumentError && occursin(field, err.msg)
        end
        let was = Nereus.AD_INFORMED_BIRTH[]
            off = try
                Nereus.AD_INFORMED_BIRTH[] = false
                nrun(200)
            finally
                Nereus.AD_INFORMED_BIRTH[] = was
            end
            @test off[1].value.data != full[1].value.data
        end
        # The noise models are in the fingerprint by content: a deserialized
        # copy of the model and data, new objects throughout, matches.
        tg = NereusTarget(Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                instruments = InstrumentConfig(rv = ["I1"]), data = data,
                priors = Dict{String,PriorSpec}("gamma_I1" => UniformPrior(-20.0, 20.0),
                                                "sigma_I1" => LogUniformPrior(0.2, 12.0)),
                noise_models = NoiseModel[ad, rot], transdim_noise = true,
                stability = :none), data)
        io = IOBuffer()
        Nereus.serialize(io, (tg.params, data, td_n))
        p2, d2, td2c = Nereus.deserialize(seekstart(io))
        fp(p, d, t) = Nereus._moms_fingerprint(p, d, t; seed = 1)
        @test isequal(fp(tg.params, data, td_n), fp(p2, d2, td2c))
    end

    @testset "run_job checkpoints by default" begin
        # The cfg as run_job hands it over: JSON-native values, an output_dir,
        # and no `checkpoint`, which run_job then puts in output_dir.
        dir = mktempdir()
        cfg(kw) = Dict(:sampler => Dict(:name => "moms", :kwargs => kw),
                       :transdim => Dict(:max_kplanet => 2, :transdim_fraction => 0.3),
                       :output_dir => dir)
        kw = Dict(:n_warmup => 120, :init_scale => 1.5, :show_progress => false,
                  :within_model => "slice")
        job(extra) = (tg = _mr_target(_MR_RV);
                      Nereus._dispatch_sampler(cfg(merge(kw, extra)), tg, tg.data, 5))
        full = job(Dict(:n_samples => 200))
        @test isfile(joinpath(dir, "moms_state.jls"))
        job(Dict(:n_samples => 100))
        ext = job(Dict(:n_samples => 200, :resume => true))
        @test ext.chains.value.data == full.chains.value.data
        @test ext.n_evals == full.n_evals
    end

    @testset "refusals" begin
        path = joinpath(mktempdir(), "moms_state.jls")
        _mr_run(100; checkpoint = path)
        # No checkpoint.
        @test_throws ArgumentError _mr_run(200; resume = true)
        @test_throws ArgumentError _mr_run(200; checkpoint = path * ".none", resume = true)
        # A different run: each names the field that differs.
        for (kw, field) in (((n_warmup = 150,), "n_warmup"), ((seed = 6,), "seed"),
                            ((init_scale = 0.5,), "init_scale"),
                            ((within_model = :rwm,), "within_model"),
                            ((target_birth_accept = 0.3,), "target_birth_accept"),
                            ((td = TransDimConfig(max_kplanet = 2,
                                                  transdim_fraction = 0.4),),
                             "transdim_fraction"),
                            ((rv = _MR_RV .+ 1.0,), "data"),
                            # Model settings outside the names, priors and data.
                            ((tkw = (M_s = 1.0, stability = :gladman),), "config_stability"),
                            ((tkw = (M_s = 1.0,),), "config_M_s"))
            err = try _mr_run(200; checkpoint = path, resume = true, kw...); nothing
                  catch e; e end
            @test err isa ArgumentError && occursin(field, err.msg)
        end
        # Gladman stability is a different chain from no stability check, so a
        # resume across them would match neither.
        let tkw(s) = (M_s = 1.0, stability = s)
            @test _draws(_mr_run(100; tkw = tkw(:none))) !=
                  _draws(_mr_run(100; tkw = tkw(:gladman)))
        end
        # Data changed where Base's `hash` does not look: an element in the
        # middle of an array of 8192 or more, of which it reads a few dozen.
        # The `data` field of `run_fingerprint` passes it; `data_content` does not.
        let n = 9000, t = collect(range(0.0, 300.0; length = n)),
            rv0 = 10.0 .* randn(MersenneTwister(3), n)
            bumped(i) = (r = copy(rv0); r[i] += 1.0; r)
            i = findfirst(i -> hash(bumped(i)) == hash(rv0), 1:n)
            @test i !== nothing
            mk(rv) = build_target(planets = (b = _mr_planet(),),
                rv = (SIM = (data = (t = t, rv = rv, rv_err = fill(1.5, n)),
                             sigma = LogUniformPrior(0.5, 10.0)),))
            t0, t1 = mk(rv0), mk(bumped(i))
            @test Nereus.run_fingerprint(t0.params, t0.data).data ==
                  Nereus.run_fingerprint(t1.params, t1.data).data
            p3 = joinpath(mktempdir(), "moms_state.jls")
            go(tg; kw...) = sample_moms(tg, tg.data; td = TransDimConfig(max_kplanet = 1),
                n_warmup = 4, n_samples = 0, seed = 1, show_progress = false,
                checkpoint = p3, kw...)
            go(t0)
            err = try go(t1; n_samples = 2, resume = true); nothing catch e; e end
            @test err isa ArgumentError && occursin("data_content", err.msg)
            @test size(go(t0; n_samples = 2, resume = true)[1].value.data, 1) == 2
        end
        # Several chains look for one file each, which a one-chain run never wrote.
        @test_throws ArgumentError _mr_run(200; checkpoint = path, resume = true,
                                           n_chains = 2)
        # A total shorter than what the checkpoint has already kept.
        @test_throws ArgumentError _mr_run(50; checkpoint = path, resume = true)
        # Informed births cannot be resumed, so nothing is written for them.
        p2 = joinpath(mktempdir(), "moms_state.jls")
        _mr_run(20; checkpoint = p2, informed_birth_fraction = 0.5)
        @test !isfile(p2)
        @test_throws ArgumentError _mr_run(20; checkpoint = p2, resume = true,
                                           informed_birth_fraction = 0.5)
    end
end
