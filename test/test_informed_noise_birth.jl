# Periodogram-informed birth for period-bearing noise kernels.
#
# The proposal is only useful if it is EXACT: an informed birth whose reverse
# density is not the same function evaluated at the same point injects a bias
# into the Hastings ratio, and the occupancy silently stops being P(M|D) --
# which is the whole quantity the trans-dim noise machinery exists to report.
# So the test is detailed balance, not "does it find the period".
using Test
using Nereus
using Nereus: Params, Data, InstrumentConfig, NereusTarget, Theta, TransDimState,
              TransDimConfig, CeleriteRotation, CeleriteRotationFM17, CeleriteSHO,
              MaternGP, PlanetDataSources, NoiseModel,
              propose_noise_birth, propose_noise_death,
              _gp_hints, _gp_period_param, _gp_amp_param, _gp_informed_draw,
              _gp_informed_logq, _gp_logP_of, _gp_val_of,
              activate_noise!, is_noise_active, noise_param_names
using Random, Statistics

@testset "informed noise birth" begin
# The hint is OFF by default (it degraded mixing); these tests exercise it
# explicitly, so turn it on for the duration.
Nereus.GP_INFORMED_BIRTH[] = true
try

    # Rotation-modulated RV so the periodogram has a real peak to find.
    P_ROT, N, SIG = 8.0, 70, 1.5
    rng0 = MersenneTwister(11)
    tt = sort(rand(rng0, N)) .* 80.0
    act = 5.0 .* (sin.(2π .* tt ./ P_ROT) .+ 0.4 .* sin.(4π .* tt ./ P_ROT .+ 0.5))
    rv = act .+ SIG .* randn(rng0, N)
    data = Data(t_rv = tt, rv = rv, rv_err = fill(SIG, N), rv_inst = ones(Int, N))

    function build(nm)
        params = Params(max_kplanet = 0, planet_modes = PlanetDataSources[],
                        instruments = InstrumentConfig(rv = ["A"]), data = data,
                        noise_models = [nm], transdim_noise = true)
        th = Theta{Float64}(params)
        th.td = TransDimState(max_planets = 0, n_noise = 1)
        return params, th
    end

    # ---- coordinate round-trip, including the Jacobian -------------------
    @testset "period coordinate transforms" begin
        for (kind, v) in ((:log, 8.0), (:ident, log(8.0)),
                          (:omega, log(2π / 8.0)))
            lp, logJ = _gp_logP_of(v, Val(kind))
            @test lp ≈ log(8.0) atol = 1e-12
            @test _gp_val_of(lp, Val(kind)) ≈ v atol = 1e-12
            # analytic Jacobian vs finite difference of logP(v)
            h = 1e-6
            lp2, _ = _gp_logP_of(v + h, Val(kind))
            @test exp(logJ) ≈ abs((lp2 - lp) / h) rtol = 1e-4
        end
    end

    # ---- which models get a hint ----------------------------------------
    @testset "period-bearing models only" begin
        @test _gp_period_param(CeleriteRotation(channel = :rv)) !== nothing
        @test _gp_period_param(CeleriteRotationFM17()) !== nothing
        @test _gp_period_param(CeleriteSHO()) !== nothing
        # a correlation length is not a period: MaternGP gets no period hint,
        # but its amplitude IS informable
        @test _gp_period_param(MaternGP()) === nothing
        @test _gp_amp_param(MaternGP()) !== nothing
        # SHO's S0 is not an amplitude on its own (k(0)=S0*w0*Q)
        @test _gp_amp_param(CeleriteSHO()) === nothing
        @test _gp_amp_param(CeleriteRotation(channel=:rv)) !== nothing
    end

    # ---- DETAILED BALANCE: the reason this exists ------------------------
    # One toggleable model ⇒ every combinatorial term in both log_q_ratios is
    # log(1) = 0, so an exact reverse density forces log_q_death = −log_q_birth.
    @testset "birth/death reversibility" begin
        for nm in (CeleriteRotation(channel = :rv), CeleriteRotationFM17(),
                   CeleriteSHO(), MaternGP())
            params, th = build(nm)
            tog = NoiseModel[nm]
            nbirths = 0
            for trial in 1:12
                rng = MersenneTwister(4000 + trial)
                born, lqb = propose_noise_birth(th, rng, tog; data = data)
                isfinite(lqb) || continue
                @test is_noise_active(born.td, 1)
                back, lqd = propose_noise_death(born, MersenneTwister(7), tog;
                                                data = data)
                @test isfinite(lqd)
                @test !is_noise_active(back.td, 1)
                @test lqd ≈ -lqb atol = 1e-9
                nbirths += 1
            end
            @test nbirths > 0
        end
    end

    # ---- the hint actually points at the rotation period -----------------
    @testset "amplitude hint tracks the data scatter" begin
        nm = CeleriteRotation(channel = :rv)
        params, th = build(nm)
        hs = _gp_hints(th, data, nm, params.layout, params.config.instruments)
        amp = hs[end]
        @test amp.slot == params.layout.name_to_idx["gp_sigma"]
        # one synthetic peak, sitting at the robust scatter of the RV
        @test length(amp.mu) == 1
        @test exp(amp.mu[1]) ≈ 1.4826*median(abs.(rv .- median(rv))) rtol = 1e-8
        rng = MersenneTwister(5)
        draws = [_gp_informed_draw(rng, amp) for _ in 1:400]
        @test all(isfinite, _gp_informed_logq.(draws, Ref(amp)))
        @test median(draws) > 0
    end

    @testset "hint locates the rotation peak" begin
        nm = CeleriteRotation(channel = :rv)
        params, th = build(nm)
        hs = _gp_hints(th, data, nm, params.layout, params.config.instruments)
        @test length(hs) == 2          # period AND amplitude
        h = hs[1]
        best = h.mu[argmax(h.w)]
        # the strongest peak sits at P_rot or its first harmonic (the injected
        # signal has a 2:1 harmonic, which a periodogram legitimately favours)
        @test min(abs(exp(best) - P_ROT), abs(exp(best) - P_ROT / 2)) < 1.0
        # and draws concentrate there rather than spreading over the prior
        rng = MersenneTwister(3)
        draws = [_gp_informed_draw(rng, h) for _ in 1:400]
        near = count(d -> min(abs(d - P_ROT), abs(d - P_ROT / 2)) < 1.5, draws)
        @test near > 400 * 0.3      # α = 0.7 informed, rest uniform
        # every draw must have finite density under the same mixture
        @test all(isfinite, _gp_informed_logq.(draws, Ref(h)))
    end

    # ---- a model with no data hint must fall back, not crash -------------
    @testset "falls back to the prior without data" begin
        nm = CeleriteRotation(channel = :rv)
        params, th = build(nm)
        @test isempty(_gp_hints(th, nothing, nm, params.layout,
                                params.config.instruments))
        born, lqb = propose_noise_birth(th, MersenneTwister(1), NoiseModel[nm];
                                        data = nothing)
        @test isfinite(lqb)
        back, lqd = propose_noise_death(born, MersenneTwister(2), NoiseModel[nm];
                                        data = nothing)
        @test lqd ≈ -lqb atol = 1e-9
    end

finally
    Nereus.GP_INFORMED_BIRTH[] = false
end
end
