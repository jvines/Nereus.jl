# assess_fit — post-fit "silent-wrong" guard. Verifies the three real
# failure modes the validation matrix caught all get flagged:
#   (a) clean unimodal Gaussian chain          => overall :ok
#   (b) bimodal chain (two disjoint modes)     => :fail w/ multimodality msg
#   (c) chain railed against a prior bound      => :fail w/ rail msg
# plus log-posterior sanity, ensemble mode, and the MAP-consistency
# convenience method.

using Nereus, Random, Test, MCMCChains
using Random: MersenneTwister

# Locate a check by name in a FitHealthReport.
_find_check(r, name) = r.checks[findfirst(c -> c.name === name, r.checks)]

@testset "assess_fit — silent-wrong guard" begin

    @testset "(a) clean unimodal Gaussian => :ok" begin
        rng = MersenneTwister(7)
        # 4 well-mixed chains around the same mode for two params.
        arr = zeros(Float64, 2000, 2, 4)
        for k in 1:4
            arr[:, 1, k] .= 10.0 .+ 1.0 .* randn(rng, 2000)   # K ~ N(10,1)
            arr[:, 2, k] .= 4.14 .+ 0.01 .* randn(rng, 2000)  # P ~ N(4.14,0.01)
        end
        ch = Chains(arr, [:K, :P])
        r = assess_fit(ch)

        @test r isa FitHealthReport
        @test r.overall === :ok
        @test _find_check(r, :convergence).status === :ok
        @test _find_check(r, :multimodality).status === :ok
        # show() must not throw and should print the OK icon.
        io = IOBuffer()
        show(io, MIME"text/plain"(), r)
        @test occursin("✅", String(take!(io)))
    end

    @testset "(b) bimodal — two disjoint modes => :fail (multimodality)" begin
        rng = MersenneTwister(8)
        # Two chains frozen at -8, two at +8: the nuts failure mode.
        arr = zeros(Float64, 2000, 1, 4)
        arr[:, 1, 1] .= -8.0 .+ 0.3 .* randn(rng, 2000)
        arr[:, 1, 2] .= -8.0 .+ 0.3 .* randn(rng, 2000)
        arr[:, 1, 3] .=  8.0 .+ 0.3 .* randn(rng, 2000)
        arr[:, 1, 4] .=  8.0 .+ 0.3 .* randn(rng, 2000)
        ch = Chains(arr, [:K])
        r = assess_fit(ch)

        @test r.overall === :fail
        mm = _find_check(r, :multimodality)
        @test mm.status === :fail
        @test occursin("disjoint modes", mm.message)
        @test occursin("merged CI is not a posterior", mm.message)
        # The merged marginal brackets 0 even though no chain visits it.
        @test !isempty(mm.details)
        # LOUD banner on overall fail.
        io = IOBuffer()
        show(io, MIME"text/plain"(), r)
        out = String(take!(io))
        @test occursin("DO NOT TRUST", out)
        @test occursin("❌", out)
    end

    @testset "(c) railed against prior bound => :fail (rail)" begin
        rng = MersenneTwister(9)
        # Eccentricity piled against the hard lower bound at 0.
        arr = zeros(Float64, 2000, 1, 4)
        for k in 1:4
            arr[:, 1, k] .= abs.(0.002 .* randn(rng, 2000))   # ~0, bound = 0
        end
        ch = Chains(arr, [:e])
        r = assess_fit(ch; prior_bounds = Dict(:e => (0.0, 1.0)))

        @test r.overall === :fail
        rail = _find_check(r, :prior_rail)
        @test rail.status === :fail
        @test occursin("rail", lowercase(rail.message))
        @test occursin("e", rail.message)
        # Without prior_bounds the rail check is a no-op (:ok), so the
        # rail itself — not convergence — must be what fails here.
        r_nobounds = assess_fit(ch)
        @test _find_check(r_nobounds, :prior_rail).status === :ok
    end

    @testset "prior-edge mass (no median rail) still flags" begin
        rng = MersenneTwister(10)
        # Median is safely interior, but a fat tail piles >5% of draws
        # against the upper bound at 1.0.
        arr = zeros(Float64, 4000, 1, 4)
        for k in 1:4
            v = 0.6 .+ 0.5 .* randn(rng, 4000)
            clamp!(v, 0.0, 1.0)
            arr[:, 1, k] .= v
        end
        ch = Chains(arr, [:e])
        r = assess_fit(ch; prior_bounds = Dict(:e => (0.0, 1.0)))
        @test _find_check(r, :prior_rail).status === :fail
    end

    @testset "log-scale priors are measured in log space, not linearly" begin
        rng = MersenneTwister(21)
        # The reported false positive: P = 12.3 d on LogUniform(0.1, 3000). A
        # linear 1% margin is 30 d, so any P under 30 d "railed at lower bound".
        arr = zeros(Float64, 2000, 1, 4)
        for k in 1:4
            arr[:, 1, k] .= 12.3 .+ 0.002 .* randn(rng, 2000)
        end
        ch = Chains(arr, [:P])
        pb = Dict(:P => (0.1, 3000.0))
        @test _find_check(assess_fit(ch; prior_bounds = pb), :prior_rail).status === :fail
        r = assess_fit(ch; prior_bounds = pb, log_scale = Dict(:P => 0.0))
        @test _find_check(r, :prior_rail).status === :ok

        # A real rail on that prior is still one: draws piled on P = 0.1 d.
        for k in 1:4
            arr[:, 1, k] .= 0.1 .+ abs.(0.0005 .* randn(rng, 2000))
        end
        rail = _find_check(assess_fit(Chains(arr, [:P]); prior_bounds = pb,
                                      log_scale = Dict(:P => 0.0)), :prior_rail)
        @test rail.status === :fail
        @test occursin("lower bound", rail.message)

        # ModJeffreys is flat in log(x + knee): K = 12 m/s on ModJeffreys(1, 1000)
        # is interior there, and within 1% of the linear span (10 m/s) is not.
        for k in 1:4
            arr[:, 1, k] .= 12.0 .+ 0.5 .* randn(rng, 2000)
        end
        chK = Chains(arr, [:K])
        pbK = Dict(:K => (0.0, 1000.0))
        r = assess_fit(chK; prior_bounds = pbK, log_scale = Dict(:K => 1.0))
        @test _find_check(r, :prior_rail).status === :ok
    end

    @testset "trans-dim: a component is assessed on its active draws only" begin
        # One flattened chain, as trans-dim samplers return. P2 exists in the
        # first ~half of the draws (a planet at 40 d); elsewhere its slot is
        # parked at the prior floor. Unmasked, the parked values rail and wreck
        # R-hat; masked, the planet is a clean posterior.
        rng = MersenneTwister(31)
        n = 20_000
        on = rand(rng, n) .< 0.5
        P2 = ifelse.(on, 40.0 .+ 0.05 .* randn(rng, n), 0.1)
        ch = Chains(reshape(P2, n, 1, 1), [:P2])
        pb = Dict(:P2 => (0.1, 3000.0)); ls = Dict(:P2 => 0.0)
        r0 = assess_fit(ch; prior_bounds = pb, log_scale = ls, ensemble = true)
        @test _find_check(r0, :prior_rail).status === :fail
        r = assess_fit(ch; prior_bounds = pb, log_scale = ls, ensemble = true,
                       active = Dict(:P2 => on))
        @test _find_check(r, :prior_rail).status === :ok
        @test _find_check(r, :convergence).status === :ok
        @test_throws ArgumentError assess_fit(ch; active = Dict(:P2 => on[1:10]))
    end

    @testset "log_scale_shift: the scale each prior is flat on" begin
        @test Nereus.log_scale_shift(LogUniformPrior(0.1, 3000.0)) == 0.0
        @test Nereus.log_scale_shift(ModJeffreysPrior(2.0, 100.0)) == 2.0
        @test Nereus.log_scale_shift(UniformPrior(0.0, 1.0)) === nothing
        @test Nereus.log_scale_shift(NormalPrior(0.0, 1.0, -5.0, 5.0)) === nothing
    end

    @testset "science-table rail flag uses the same scale" begin
        # 3σ CI of P = 12.3 d against LogUniform(0.1, 3000).
        @test Nereus._railed([12.29, 12.31], 0.1, 3000.0) == (true, "lower")
        @test Nereus._railed([12.29, 12.31], 0.1, 3000.0, 0.0) == (false, "")
        @test Nereus._railed([0.1001, 0.2], 0.1, 3000.0, 0.0) == (true, "lower")
        @test Nereus._railed([2000.0, 2999.0], 0.1, 3000.0, 0.0) == (true, "upper")
    end

    # A jitter can be zero -- the reported errors may carry all the scatter --
    # so a posterior piled there is never cause for alarm. Every default
    # fit of an easy 12.3 d RV target (errors exact, no excess) printed "FIT
    # HEALTH: FAIL -- DO NOT TRUST THIS POSTERIOR" for sigma_HARPS at 0.
    @testset "a jitter at zero is not a rail" begin
        rng = MersenneTwister(41)
        arr = zeros(Float64, 2000, 1, 4)
        for k in 1:4
            arr[:, 1, k] .= abs.(0.7 .* randn(rng, 2000))   # half-normal at 0
        end
        ch = Chains(arr, [:sigma_HARPS])
        pb = Dict(:sigma_HARPS => (0.0, 50.0))
        @test _find_check(assess_fit(ch; prior_bounds = pb), :prior_rail).status === :fail
        r = assess_fit(ch; prior_bounds = pb, jitter = ["sigma_HARPS"])
        @test r.overall === :ok
        rail = _find_check(r, :prior_rail)
        @test rail.status === :ok
        @test occursin("jitter at zero", rail.message)
        @test occursin("sigma_HARPS", rail.message)

        # On a log-scale prior the floor stands in for zero: the same.
        for k in 1:4
            arr[:, 1, k] .= 0.01 .* exp.(abs.(0.3 .* randn(rng, 2000)))
        end
        chl = Chains(arr, [:gp_act_jit_bis])
        pbl = Dict(:gp_act_jit_bis => (0.01, 10.0)); lsl = Dict(:gp_act_jit_bis => 0.0)
        @test _find_check(assess_fit(chl; prior_bounds = pbl, log_scale = lsl),
                          :prior_rail).status === :fail
        @test _find_check(assess_fit(chl; prior_bounds = pbl, log_scale = lsl,
                                     jitter = [:gp_act_jit_bis]), :prior_rail).status === :ok

        # The upper bound is still a rail: a noise excess the prior truncated.
        for k in 1:4
            arr[:, 1, k] .= 50.0 .- abs.(0.3 .* randn(rng, 2000))
        end
        up = _find_check(assess_fit(Chains(arr, [:sigma_HARPS]); prior_bounds = pb,
                                    jitter = [:sigma_HARPS]), :prior_rail)
        @test up.status === :fail
        @test occursin("upper bound", up.message)

        # Only jitters: an eccentricity at 0 next to one still fails.
        mix = zeros(Float64, 2000, 2, 4)
        for k in 1:4
            mix[:, 1, k] .= abs.(0.002 .* randn(rng, 2000))
            mix[:, 2, k] .= abs.(0.7 .* randn(rng, 2000))
        end
        mixed = _find_check(assess_fit(Chains(mix, [:e, :sigma_HARPS]);
                                       prior_bounds = merge(pb, Dict(:e => (0.0, 1.0))),
                                       jitter = [:sigma_HARPS]), :prior_rail)
        @test mixed.status === :fail
        @test occursin("e:", mixed.message) && !occursin("sigma_HARPS", mixed.message)

        # Science tables: no "railed" flag (and no † footnote) at the floor,
        # still one at the ceiling.
        @test Nereus._railed([0.0, 2.1], 0.0, 50.0) == (true, "lower")
        @test Nereus._railed([0.0, 2.1], 0.0, 50.0; floor_ok = true) == (false, "")
        @test Nereus._railed([40.0, 50.0], 0.0, 50.0; floor_ok = true) == (true, "upper")
        @test Nereus._railed([0.0, 50.0], 0.0, 50.0; floor_ok = true) == (true, "upper")
    end

    @testset "jitter_names: the jitter terms, and only those" begin
        t = collect(55000.0:10.0:55300.0)
        blk = (P = LogUniformPrior(12.0, 12.6), K = LogUniformPrior(1.0, 100.0),
               sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
               Mo = UniformPrior(0.0, 2π))
        tg = build_target(planets = (b = blk,),
                          rv = (HARPS = (data = (t = t, rv = zeros(length(t)),
                                                 rv_err = fill(2.0, length(t))),
                                         sigma = LogUniformPrior(0.01, 10.0)),
                                ESPRESSO = (data = (t = t, rv = zeros(length(t)),
                                                    rv_err = fill(0.5, length(t))),
                                            sigma = LogUniformPrior(0.01, 10.0))))
        jn = Nereus.jitter_names(tg.params)
        @test jn == Set(["sigma_HARPS", "sigma_ESPRESSO"])

        # Noise-model jitters: the white-noise floors, not the slope jit_act_*
        # nor the GP / floor amplitudes.
        n = 25
        d = Data(; t_rv = collect(1.0:n), rv = zeros(n), rv_err = ones(n),
                   rv_inst = ones(Int, n),
                   indicators = Dict("bis" => zeros(n), "log_rhk" => zeros(n)),
                   indicator_errs = Dict("bis" => fill(0.01, n),
                                         "log_rhk" => fill(0.01, n)))
        p = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = InstrumentConfig(rv = ["SIM"]), data = d, M_s = 1.0,
                     noise_models = NoiseModel[
                         ActivityGP(channels = [:bis], marginalize_indicators = false),
                         ActivityJitter(indicator = "log_rhk"),
                         IndicatorFloor(channels = [:bis], kernel = :qp)])
        @test Nereus.jitter_names(p) ==
              Set(["sigma_SIM", "gp_act_jit_bis", "jit_base_log_rhk_SIM", "ind_floor_bis_jit"])
        # IndicatorFloor's default :white kernel names its floor bare,
        # `ind_floor_<ch>`; NightlyOffset's night_sigma is a per-night offset
        # scale, not white noise.
        pw = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                      instruments = InstrumentConfig(rv = ["SIM"]), data = d, M_s = 1.0,
                      noise_models = NoiseModel[
                          IndicatorFloor(channels = [:bis, :log_rhk]),
                          NightlyOffset(instruments = ["SIM"])])
        @test any(startswith("night_sigma"), pw.layout.names)
        @test Nereus.jitter_names(pw) == Set(["sigma_SIM", "ind_floor_bis", "ind_floor_log_rhk"])
    end

    # The MAP's own rail flag: api.jl turns a railed MAP into a failed fit. Its
    # physical-floor exemption covered a lower bound of exactly 0 only, so a
    # jitter on LogUniform(0.01, 10) at its floor failed the fit.
    @testset "MAP: a jitter at its floor is not railed" begin
        rng = MersenneTwister(43)
        N = 60
        t = sort(55000 .+ 200 .* rand(rng, N))
        blk = (P = LogUniformPrior(12.0, 12.6), K = LogUniformPrior(1.0, 100.0),
               sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
               Mo = UniformPrior(0.0, 2π))
        mk(rv) = build_target(planets = (b = blk,),
                              rv = (HARPS = (data = (t = t, rv = rv, rv_err = fill(2.0, N)),
                                             sigma = LogUniformPrior(0.01, 10.0)),))
        tg0 = mk(zeros(N))
        th = Theta{Float64}(tg0.params)
        for (nm, v) in ("P_k1" => 12.3, "K_k1" => 25.0, "sesinw_k1" => 0.3,
                        "secosw_k1" => 0.4, "Mo_k1" => 2.0, "sigma_HARPS" => 0.01)
            set_param!(th, nm, v)
        end
        model, _ = rv_predictions(th, tg0.data)
        # 1.5 m/s of scatter under 2 m/s errors: no excess, the jitter's
        # posterior mode is its floor.
        tg = mk(model .+ 1.5 .* randn(rng, N))
        names_ = tg.params.layout.unfrozen_names
        x = [let i = findfirst(==(nm), tg0.params.layout.names)
                 th.values[i]
             end for nm in names_]
        j = findfirst(==("sigma_HARPS"), names_)
        @test x[j] == 0.01
        # sample_map runs this on its optimum; before, sigma_HARPS was flagged
        # ("the posterior wants to leave the box ... widen the offending prior").
        @test !(j in Nereus._detect_railed(x, tg, 1e-3))
    end

    @testset "log-posterior sanity catches corrupt :lp" begin
        rng = MersenneTwister(11)
        arr = zeros(Float64, 1000, 2, 3)
        for k in 1:3
            arr[:, 1, k] .= 1.0 .+ randn(rng, 1000)
            arr[:, 2, k] .= 2.0 .+ randn(rng, 1000)
        end
        # Corrupt log-posterior column: absurdly negative values.
        lp = fill(-1e9, 1000, 1, 3)
        ch = Chains(cat(arr, lp; dims = 2), [:a, :b, :lp])
        r = assess_fit(ch)
        lpc = _find_check(r, :logpost)
        @test lpc.status === :fail
        @test occursin("corrupt log-posterior", lpc.message)
        @test r.overall === :fail

        # Sane lp => :ok and lp is not convergence-tested as a parameter.
        lp_ok = -50.0 .+ randn(rng, 1000, 1, 3)
        ch_ok = Chains(cat(arr, lp_ok; dims = 2), [:a, :b, :lp])
        r_ok = assess_fit(ch_ok)
        @test _find_check(r_ok, :logpost).status === :ok
        @test r_ok.overall === :ok
    end

    @testset "ensemble-aware convergence on a single ensemble" begin
        rng = MersenneTwister(12)
        # One ensemble of 20 walkers, each its own short trace — well
        # mixed within and across walkers.
        arr = zeros(Float64, 1000, 1, 20)
        for w in 1:20
            arr[:, 1, w] .= 5.0 .+ 1.0 .* randn(rng, 1000)
        end
        ch = Chains(arr, [:K])
        r = assess_fit(ch; ensemble = true)
        @test _find_check(r, :convergence).status === :ok
        @test r.overall === :ok
    end

    @testset "MAP-consistency convenience method" begin
        rng = MersenneTwister(13)
        arr = zeros(Float64, 2000, 2, 4)
        for k in 1:4
            arr[:, 1, k] .= 10.0 .+ 1.0 .* randn(rng, 2000)
            arr[:, 2, k] .= 4.14 .+ 0.01 .* randn(rng, 2000)
        end
        ch = Chains(arr, [:K, :P])

        # MAP inside the bulk => :ok.
        r_in = assess_fit(Dict(:K => 10.1, :P => 4.141), ch; nsigma = 4.0)
        @test _find_check(r_in, :map_consistency).status === :ok
        @test r_in.overall === :ok

        # MAP far outside the bulk (railed/ spurious mode) => :fail.
        r_out = assess_fit(Dict(:K => 30.0, :P => 4.14), ch; nsigma = 4.0)
        mc = _find_check(r_out, :map_consistency)
        @test mc.status === :fail
        @test occursin("outside posterior bulk", mc.message)
        @test r_out.overall === :fail
    end
end
