# The transit-phase time anchor, time = :phic: φ_c = 2π (Tc − t_ref_c) / P.
#
# What it is for: a Tc prior U(t_min, t_max) holds baseline/P copies of the
# orbit, and an ensemble cannot move between them. φ_c is Tc in phase units,
# so its window is one period at every P. The assertions pin that it reads
# exactly like Tc at the matching value, repeats every 2π where Mo does, keeps
# its epoch with the parametrization, and reports Tc per draw in one copy.

using Nereus, MCMCChains, Random, Statistics, Test
using Nereus: phic_to_tc, tc_to_phic, anchor_to_tp, anchor_to_tc, tc_to_tp

include(joinpath(@__DIR__, "fixtures", "transit_gate_target.jl"))

@testset "transit phase anchor (phic)" begin

    @testset "conversions" begin
        rng = MersenneTwister(1)
        for _ in 1:200
            P = 5 * exp(randn(rng)); t_ref = 2.46e6
            Tc = t_ref + 100 * randn(rng); t_ref_c = t_ref + 50 * randn(rng)
            e = 0.9 * rand(rng); ω = 2π * rand(rng) - π; Mo = 2π * rand(rng)
            φ = tc_to_phic(Tc, P, t_ref_c)
            @test phic_to_tc(φ, P, t_ref_c) ≈ Tc atol = 1e-8
            @test phic_to_tc(φ + 2π, P, t_ref_c) ≈ Tc + P atol = 1e-8
            @test anchor_to_tc(:phic, φ, P, e, ω, t_ref, t_ref_c) === phic_to_tc(φ, P, t_ref_c)
            @test anchor_to_tp(:phic, φ, P, e, ω, t_ref, t_ref_c) ===
                  tc_to_tp(phic_to_tc(φ, P, t_ref_c), P, e, ω)
            # The other anchors are the expressions they replaced, bit for bit;
            # t_ref_c is never read by them.
            @test anchor_to_tp(:Mo, Mo, P, e, ω, t_ref, NaN) === t_ref - Mo * P / 2π
            @test anchor_to_tp(:Tp, Tc, P, e, ω, t_ref, NaN) === Tc
            @test anchor_to_tp(:Tc, Tc, P, e, ω, t_ref, NaN) === tc_to_tp(Tc, P, e, ω)
            @test anchor_to_tc(:Tc, Tc, P, e, ω, t_ref, NaN) === Tc
        end
        @test_throws ArgumentError anchor_to_tp(:Tx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    end

    @testset "ParametrizationConfig" begin
        pc = ParametrizationConfig(time = :phic)
        @test pc.time === :phic && isnan(pc.t_ref_c)
        @test ParametrizationConfig(time = :phic, t_ref_c = 5.0).t_ref_c == 5.0
        @test isnan(ParametrizationConfig().t_ref_c)
        @test_throws ArgumentError ParametrizationConfig(time = :phase)
        @test_throws ArgumentError ParametrizationConfig(time = :phic, t_ref_c = Inf)
    end

    # Photometry from the transit fixture (P = 4.137 d, a 9 d sector and three
    # ground nights), plus RVs, so an RVPM planet runs both likelihoods.
    phot = transit_gate_data()
    rng = MersenneTwister(4)
    t_rv = sort(TG_BJD0 .+ 300 .* rand(rng, 25))
    data = Data(; t_rv = t_rv, rv = 30 .* sin.(2π .* (t_rv .- TG_TRUTH.Tc) ./ TG_TRUTH.P) .+
                                  3 .* randn(rng, 25),
                  rv_err = fill(3.0, 25), t_phot = phot.t_phot, flux = phot.flux,
                  flux_err = phot.flux_err, phot_inst = phot.phot_inst)
    ic = InstrumentConfig(rv = ["HARPS"], pm = ["TESS", "GROUND"])
    base = Dict{String, PriorSpec}(
        "P_k1" => UniformPrior(3.9, 4.4), "K_k1" => UniformPrior(1.0, 100.0),
        "b_k1" => UniformPrior(0.0, 1.3), "rr_k1" => UniformPrior(0.005, 0.3),
        "sesinw_k1" => UniformPrior(-0.5, 0.5), "secosw_k1" => UniformPrior(-0.5, 0.5),
        "rho_s" => LogUniformPrior(0.02, 30.0))
    mk(time; modes = [RVPM], d = data, priors = Dict{String, PriorSpec}(), kw...) =
        Params(; max_kplanet = 1, planet_modes = modes, instruments = ic, data = d,
                 parametrization = ParametrizationConfig(; time, use_rho_s = true, kw...),
                 priors = merge(base, priors), stability = :none)
    pT = mk(:Tc; priors = Dict{String, PriorSpec}(
        "Tc_k1" => UniformPrior(TG_BJD0 - 300.0, TG_BJD0 + 300.0)))
    pφ = mk(:phic)
    t_ref_c = pφ.config.parametrization.t_ref_c

    @testset "epoch, slot and default prior" begin
        # The middle of the photometry, not the RV median the Mo anchor uses.
        @test t_ref_c == median(data.t_phot)
        @test data.t_ref == median(t_rv) && t_ref_c != data.t_ref
        @test "phic_k1" in pφ.layout.unfrozen_names
        @test !("Tc_k1" in pφ.layout.unfrozen_names)
        @test Nereus.bounds(pφ.config.priors["phic_k1"]) == (-Float64(π), Float64(π))
        @test mk(:phic; t_ref_c = TG_BJD0).config.parametrization.t_ref_c == TG_BJD0
        # Without photometry it falls back to the data reference epoch.
        rvd = Data(; t_rv = t_rv, rv = data.rv, rv_err = data.rv_err)
        prv = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                       instruments = InstrumentConfig(rv = ["HARPS"]), data = rvd,
                       parametrization = ParametrizationConfig(time = :phic))
        @test prv.config.parametrization.t_ref_c == rvd.t_ref
        # Measured from the data being fitted, so it cannot be guessed without.
        @test_throws ArgumentError Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
            instruments = InstrumentConfig(rv = ["HARPS"]),
            parametrization = ParametrizationConfig(time = :phic))
    end

    @testset "circular where Mo is" begin
        @test Nereus.circular_names(pφ) == Set(["phic_k1"])
        @test isempty(Nereus.circular_names(pT))
        # A 2π shift renumbers the transits under TTVs, and moves the O'Neil
        # prior's linear mean anomaly: not periodic there.
        @test !Nereus.is_circular("phic_k1", UniformPrior(-π, π),
                                  (; planet_modes = [PM_TTV]))
        @test !Nereus.is_circular("phic_k1", UniformPrior(-π, π),
            (; planet_modes = [RV_ONLY],
               parametrization = ParametrizationConfig(time = :phic, obs_prior = true)))
        # Two periods are the user's walls, as for Mo.
        @test !Nereus.is_circular("phic_k1", UniformPrior(-2π, 2π), pφ.config)
    end

    # Points drawn from the Tc model's prior, with φ_c set to the same transit.
    function pairs_(n; seed = 7)
        r = MersenneTwister(seed)
        out = Tuple{Theta{Float64}, Theta{Float64}}[]
        while length(out) < n
            thT = Theta{Float64}(pT); thφ = Theta{Float64}(pφ)
            for (j, nm) in enumerate(pT.layout.unfrozen_names)
                set_param!(thT, nm, Nereus.quantile(pT.layout.unfrozen_priors[j], rand(r)))
            end
            isfinite(Nereus.log_prior(thT)) || continue
            for nm in pT.layout.unfrozen_names
                nm == "Tc_k1" || set_param!(thφ, nm, get_param(thT, nm))
            end
            set_param!(thφ, "phic_k1",
                       tc_to_phic(get_param(thT, "Tc_k1"), get_param(thT, "P_k1"), t_ref_c))
            push!(out, (thT, thφ))
        end
        return out
    end

    @testset "reads exactly like Tc" begin
        n_rv, n_ph = length(data.t_rv), length(data.t_phot)
        wsT = Nereus.PTWorkspace(pT, 1, 0; n_obs = n_rv, n_phot = n_ph)
        wsφ = Nereus.PTWorkspace(pφ, 1, 0; n_obs = n_rv, n_phot = n_ph)
        for (thT, thφ) in pairs_(30)
            for f in (Nereus.rv_log_likelihood, Nereus.transit_log_likelihood)
                a, b = f(thT, data), f(thφ, data)
                @test isfinite(a) && isapprox(a, b; rtol = 1e-9)
            end
            a, b = Nereus.rv_log_likelihood(thT, data, wsT),
                   Nereus.rv_log_likelihood(thφ, data, wsφ)
            @test isfinite(a) && isapprox(a, b; rtol = 1e-9)
            a, b = Nereus.transit_log_likelihood(thT, data, wsT),
                   Nereus.transit_log_likelihood(thφ, data, wsφ)
            @test isfinite(a) && isapprox(a, b; rtol = 1e-9)
            # The decoders every other consumer goes through.
            P = get_param(thT, "P_k1"); e, ω = Nereus.planet_e_w(thT, 1)
            TpT, TcT = Nereus._planet_tp_tc(thT, 1, data.t_ref, P, e, ω)
            Tpφ, Tcφ = Nereus._planet_tp_tc(thφ, 1, data.t_ref, P, e, ω)
            @test Tcφ ≈ TcT atol = 1e-8
            @test Tpφ ≈ TpT atol = 1e-8
            @test Nereus._planet_Tc(thφ, data, 1) ≈ TcT atol = 1e-8
            @test Nereus._planet_Tp(thφ, data, 1) ≈ TpT atol = 1e-8
        end
    end

    @testset "repeats every 2π" begin
        for (_, thφ) in pairs_(10; seed = 8)
            a = Nereus.transit_log_likelihood(thφ, data) + Nereus.rv_log_likelihood(thφ, data)
            φ = get_param(thφ, "phic_k1")
            for s in (-2π, 2π)
                set_param!(thφ, "phic_k1", φ + s)
                b = Nereus.transit_log_likelihood(thφ, data) +
                    Nereus.rv_log_likelihood(thφ, data)
                @test isapprox(a, b; rtol = 1e-9)
            end
        end
    end

    @testset "TTV diagnostics read Tc under both anchors" begin
        for (thT, thφ) in pairs_(5; seed = 10)
            @test Nereus._ttv_Tc(thφ.values, pφ, 1) ≈ Nereus._ttv_Tc(thT.values, pT, 1) atol = 1e-8
        end
        # Mo needs the data's t_ref, which these diagnostics do not take: a
        # clear error, not a missing-key one.
        pM = mk(:Mo)
        @test_throws ArgumentError Nereus._ttv_Tc(Theta{Float64}(pM).values, pM, 1)
    end

    @testset "run_job config" begin
        verrs(cfg) = try
            Nereus._validate_config(cfg); ""
        catch err
            sprint(showerror, err)
        end
        sel = phot.phot_inst .== 1
        dcfg = Dict{String, Any}("transit_photometry" => [Dict{String, Any}(
            "instrument" => "TESS",
            "values" => Dict{String, Any}("bjd" => phot.t_phot[sel], "flux" => phot.flux[sel],
                                          "flux_err" => phot.flux_err[sel]))])
        par = Dict{String, Any}("time" => "phic", "use_rho_s" => true)
        cfg = Dict{String, Any}("star" => Dict{String, Any}("M_s" => 1.0, "R_s" => 1.0),
            "priors" => Dict{String, Any}(), "noise_models" => Any[], "data" => dcfg,
            "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["PM_ONLY"],
                                         "parametrization" => par))
        @test !occursin("parametrization.time", verrs(cfg))
        d, irv, ipm = Nereus._build_data(dcfg)
        for (t_ref_c_cfg, want) in ((nothing, median(d.t_phot)), (TG_BJD0 + 2.0, TG_BJD0 + 2.0))
            t_ref_c_cfg === nothing || (par["t_ref_c"] = t_ref_c_cfg)
            p, _, _ = Nereus._build_model(cfg, d, Nereus._build_star(cfg["star"]), irv, ipm)
            @test p.config.parametrization.time === :phic
            @test p.config.parametrization.t_ref_c == want
            @test "phic_k1" in p.layout.unfrozen_names
        end
        par["time"] = "phase"
        @test occursin("unknown parametrization.time=`phase`", verrs(cfg))
    end

    @testset "derived Tc: one copy, each draw's own P" begin
        # φ_c ~ N(π − 0.01, 0.02) folded into [−π, π): split at the seam, the
        # shape an unrelabelled chain has. Tc must still come back contiguous.
        r = MersenneTwister(9)
        nm = pφ.layout.unfrozen_names
        n, nc = 400, 2
        X = zeros(n, length(nm) + 1, nc)
        for c in 1:nc, s in 1:n, (j, name) in enumerate(nm)
            X[s, j, c] = name == "phic_k1" ? rem2pi(π - 0.01 + 0.02 * randn(r), RoundNearest) :
                         name == "P_k1" ? 4.137 + 1e-3 * randn(r) :
                         Nereus.quantile(pφ.layout.unfrozen_priors[j], 0.25 + 0.5 * rand(r))
        end
        ch = Chains(X, vcat(Symbol.(nm), :lp))
        ph = vec(Array(ch[:phic_k1])); P = vec(Array(ch[:P_k1]))
        @test any(<(0), ph) && any(>(0), ph)                           # really split
        Tc = compute_derived(ch, pφ)[1].values["Tc"]
        @test maximum(Tc) - minimum(Tc) < 0.5                          # one copy, not two
        # Each draw is its own φ_c's transit, up to whole periods of its own P.
        @test all(abs.(rem.(Tc .- phic_to_tc.(ph, P, t_ref_c), P, Ref(RoundNearest))) .< 1e-6)
        @test median(Tc) ≈ t_ref_c + (π - 0.01) * 4.137 / 2π atol = 0.01
        # The Tc fit reports no derived Tc: it samples it.
        chT = Chains(X, vcat(Symbol.(replace(nm, "phic_k1" => "Tc_k1")), :lp))
        @test !haskey(compute_derived(chT, pT)[1].values, "Tc")
    end
end
