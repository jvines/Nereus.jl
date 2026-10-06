# Every RV zero point has a proper prior, in every configuration.
#
# The offset is the one parameter no fit can do without and no user thinks to
# set, so it is the one that must never be left to chance: a `gamma_<INST>`
# with no prior, an unbounded one, or one that does not reach the instrument's
# actual zero point fails in ways that look like a bad fit rather than a bad
# setup. This builds each configuration through the entry point a user reaches
# it by and checks the prior every offset ends up with.
#
# "Proper" here means: a prior exists, it is a normalisable density on a finite
# interval, the offset is sampled (not silently frozen), and the zero point that
# generated the data lies inside it with room to spare.

using Test
using Nereus
using Nereus: Params, Data, InstrumentConfig, ParametrizationConfig, PriorSpec,
               build_target, UniformPrior, NormalPrior, LogUniformPrior, SinePrior,
               HGCAData, mjd_epochs, RMNight, TomoNight, obliquity_data,
               obliquity_params, joint_obliquity_logpost, rm_anomaly, is_fixed
using Distributions: logpdf
using Random, Statistics

@testset "RV offset priors" begin
    rng = MersenneTwister(7)
    P, K, Tc = 3.52, 60.0, 2459000.3
    orb(t) = -K .* sin.(2π .* (t .- Tc) ./ P)

    # An absolute-velocity instrument, a differential one, a second absolute one
    # with a different zero point, and two in-transit nights.
    truth = Dict("HARPS" => 31_250.0, "HIRES" => -3.0, "CORALIE" => 31_190.0,
                 "N1" => -12_400.0, "N2" => 8.0)
    function season(name, n)
        t = sort(2458900.0 .+ 300.0 .* rand(rng, n))
        (t = t, rv = truth[name] .+ orb(t) .+ 3 .* randn(rng, n), rv_err = fill(3.0, n))
    end
    function night(name, k)
        tc = Tc + k * P
        t = collect(range(tc - 0.12, tc + 0.12; length = 45))
        rm = 40 .* sin.(2π .* (t .- tc) ./ 0.2) .* (abs.(t .- tc) .< 0.07)
        (t = t, rv = truth[name] .+ orb(t) .+ rm .+ 4 .* randn(rng, 45), rv_err = fill(4.0, 45))
    end
    rvd = Dict("HARPS" => season("HARPS", 40), "HIRES" => season("HIRES", 25),
               "CORALIE" => season("CORALIE", 30), "N1" => night("N1", 0),
               "N2" => night("N2", 3))
    tph = collect(range(Tc - 0.25, Tc + 0.25; length = 200))
    phot = (t = tph, flux = 1 .+ 2e-4 .* randn(rng, 200), flux_err = fill(2e-4, 200))
    hgca() = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                      pmra = (5.0, 4.95, 4.9), pmdec = (-3.0, -3.05, -3.1),
                      sigma_pmra = (0.2, 0.2, 0.2), sigma_pmdec = (0.2, 0.2, 0.2),
                      plx = 25.0, plx_err = 0.05, hip_id = 1)

    function pack(names)
        t = Float64[]; v = Float64[]; e = Float64[]; inst = Int[]
        for (i, n) in enumerate(names)
            d = rvd[n]
            append!(t, d.t); append!(v, d.rv); append!(e, d.rv_err)
            append!(inst, fill(i, length(d.t)))
        end
        return t, v, e, inst
    end

    # The check itself. `members` maps a gamma label to the instruments that
    # share it; by default a label is its own instrument.
    function check_offsets(params; members = Dict{String, Vector{String}}())
        lay = params.layout
        slots = [n for n in lay.names if startswith(n, "gamma_")]
        @test !isempty(slots)
        for g in slots
            @test haskey(params.config.priors, g)
            ps = params.config.priors[g]
            @test !is_fixed(ps)                              # sampled, not frozen
            @test g in lay.unfrozen_names
            @test isfinite(ps.lo) && isfinite(ps.hi) && ps.lo < ps.hi
            @test isfinite(logpdf(ps.dist, (ps.lo + ps.hi) / 2))
            label = replace(g, "gamma_" => "")
            for m in get(members, label, [label])
                haskey(truth, m) || continue
                @test ps.lo + 30 < truth[m] < ps.hi - 30     # inside, with room
            end
        end
    end

    function direct(names, mode; with_phot = false, with_astrom = false, kw...)
        t, v, e, inst = pack(names)
        dkw = Dict{Symbol, Any}(:t_rv => t, :rv => v, :rv_err => e, :rv_inst => inst)
        with_phot && merge!(dkw, Dict(:t_phot => phot.t, :flux => phot.flux,
            :flux_err => phot.flux_err, :phot_inst => ones(Int, length(phot.t))))
        with_astrom && (dkw[:hgca] = hgca())
        return Params(; max_kplanet = 1, planet_modes = [mode],
                      instruments = InstrumentConfig(rv = collect(String, names),
                                                     pm = with_phot ? ["TESS"] : String[]),
                      data = Data(; dkw...), M_s = 1.0, R_s = 1.0, kw...)
    end

    seasons = ["HARPS", "HIRES", "CORALIE"]
    with_nights = ["HARPS", "N1", "N2"]

    @testset "RV only" begin
        check_offsets(direct(seasons, Nereus.RV_ONLY))
        check_offsets(direct(seasons, Nereus.RV_ONLY; trend_order = 2))
        check_offsets(direct(seasons, Nereus.RV_ONLY;
                             sharing = Dict(:gamma => [["HARPS", "CORALIE"]]));
                      members = Dict("HARPS+CORALIE" => ["HARPS", "CORALIE"]))
        check_offsets(direct(["HARPS"], Nereus.BINARY))
    end

    @testset "RV + transit" begin
        check_offsets(direct(seasons, Nereus.RVPM; with_phot = true))
        check_offsets(direct(with_nights, Nereus.RVPM_TTV; with_phot = true))
    end

    @testset "RV + astrometry" begin
        msec = ParametrizationConfig(mass = :M_sec_driven)
        check_offsets(direct(seasons, Nereus.RVAS; with_astrom = true,
                             parametrization = msec))
        check_offsets(direct(seasons, Nereus.RVPMAS; with_phot = true,
                             with_astrom = true, parametrization = msec))
    end

    @testset "RM, every kernel, and gravity darkening" begin
        for mode in (Nereus.RVPM_RM, Nereus.RVPM_RM_R, Nereus.RVPM_RM_A,
                     Nereus.RVPM_RM_GD, Nereus.RVPM_GD)
            check_offsets(direct(with_nights, mode; with_phot = true))
        end
    end

    @testset "build_target (fit_rv, fit_joint)" begin
        blk = (P = LogUniformPrior(1.0, 100.0), K = LogUniformPrior(1.0, 500.0),
               sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
               Mo = UniformPrior(0.0, 2π))
        rvnt = NamedTuple{Tuple(Symbol.(seasons))}(Tuple((data = rvd[n],) for n in seasons))
        check_offsets(build_target(planets = (b = blk,), rv = rvnt).params)

        ablk = (a = LogUniformPrior(1.0, 100.0), M_sec = LogUniformPrior(0.001, 0.5),
                sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                Mo = UniformPrior(0.0, 2π), inc = SinePrior(), Omega = UniformPrior(0.0, 2π))
        check_offsets(build_target(M_pri = NormalPrior(0.856, 0.014), planets = (b = ablk,),
                                   rv = rvnt, hgca = hgca()).params)

        chans = [Dict("source" => "RV", "data" => Dict(n => rvd[n] for n in seasons)),
                 Dict("source" => "PM", "data" => Dict("TESS" => phot))]
        tblk = merge(blk, (b = UniformPrior(0.0, 1.0), rr = UniformPrior(0.01, 0.3)))
        check_offsets(Nereus._target_from(chans, (b = tblk,); M_s = 1.0, R_s = 1.0).params)
    end

    @testset "job config (fit_rm, run_job)" begin
        rmpar = Dict{String, Any}("mass" => "K_driven", "time" => "Tc", "ew" => "sesinw",
                                  "geom" => "b_rr", "use_rho_s" => true)
        function from_config(modes, names; par = Dict{String, Any}(), with_phot = true)
            cfg = Dict{String, Any}(
                "star"  => Dict{String, Any}("M_s" => 1.0, "R_s" => 1.0),
                "data"  => Nereus._data_block(; rv = Dict(n => rvd[n] for n in names),
                                              phot = with_phot ? Dict("TESS" => phot) : nothing),
                "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => modes,
                                             "parametrization" => par, "stability" => "none"),
                "priors" => Dict{String, Any}())
            data, irv, ipm = Nereus._build_data(cfg["data"])
            params, _, _ = Nereus._build_model(cfg, data, Nereus._build_star(cfg["star"]),
                                               irv, ipm)
            return params
        end
        for m in ("RVPM_RM", "RVPM_RM_R", "RVPM_RM_A")
            check_offsets(from_config([m], with_nights; par = rmpar))
        end
        check_offsets(from_config(["RVPM_RM_A"], ["N1", "N2"]; par = rmpar))   # nights only
        check_offsets(from_config(["RV_ONLY"], seasons; with_phot = false))
        check_offsets(from_config(["RVPM"], seasons))
    end

    nights = [RMNight(tag, rvd[tag].t, rvd[tag].rv, rvd[tag].rv_err, 8000.0, Tc + k * P)
              for (tag, k) in ("N1" => 0, "N2" => 3)]

    @testset "obliquity_params: one night, one instrument, one offset" begin
        data, names = obliquity_data(nights)
        for arome in (false, true)
            check_offsets(obliquity_params(data, names; P = P, Tc = Tc, b = (0.3, 0.05),
                                           a_Rs = (8.0, 0.3), rr = (0.1, 0.01),
                                           vsini = (10_000.0, 2_000.0), arome = arome))
        end
    end

    # The bespoke joint sampler carries its own parameter vector and its own
    # priors, and its per-night offset had none: the log-posterior was flat in
    # gamma out to any value at all.
    @testset "joint_obliquity_logpost: the per-night offset is bounded" begin
        aRs, inc, rr, vs, σ0 = 6.81, deg2rad(83.6), 0.116, 25_900.0, 15_700.0
        t = collect(range(-0.10, 0.10; length = 41))
        γ = -12_400.0
        v = γ .+ rm_anomaly(t, 0.0, P, aRs, inc, deg2rad(-55.0), vs, σ0; rr = rr) .+
            15 .* randn(rng, 41)
        n1 = RMNight("N1", t, v, fill(15.0, 41), σ0, 0.0)
        b = aRs * cos(inc)
        kw = (vsini_mu = 25.9, vsini_sd = 1.0, b_mu = b, b_sd = 0.05, a_mu = aRs,
              a_sd = 0.2, rr = rr, u1 = 0.23, u2 = 0.15, use_tomogram = false,
              use_rv_gp = false)
        lp(γv, nt) = joint_obliquity_logpost(
            [deg2rad(-55.0), 25.9, b, aRs, 0.0, γv, 2.0, 0.0, 1.0, 1.0],
            TomoNight[], Vector{Float64}[], [nt], P; kw...)

        @test isfinite(lp(γ, n1))
        lo, hi = Nereus._rm_gamma_bounds([n1])[1]
        @test isfinite(lo) && isfinite(hi) && lo + 30 < γ < hi - 30
        @test lp(lo - 1.0, n1) == -Inf
        @test lp(hi + 1.0, n1) == -Inf
        @test lp(γ + 1e6, n1) == -Inf            # was a finite number

        # The bounds are data-driven, so moving the night's zero point moves
        # them with it and the posterior at the true offset does not change.
        shifted = RMNight("N1", t, v .+ 5.0e4, fill(15.0, 41), σ0, 0.0)
        @test lp(γ + 5.0e4, shifted) ≈ lp(γ, n1) rtol = 1e-9

        # One rule, not two: the framework path gives that night the same prior.
        data, names = obliquity_data([n1])
        p = obliquity_params(data, names; P = P, Tc = 0.0, b = (b, 0.05),
                             a_Rs = (aRs, 0.2), rr = (rr, 0.01), vsini = (vs, 1000.0))
        ps = p.config.priors["gamma_N1"]
        @test ps.lo ≈ lo && ps.hi ≈ hi
    end

    # Under analytic marginalisation the offset is not a parameter, so a prior
    # handed to it cannot be honoured. It used to be accepted, which unfroze the
    # slot and sampled a dimension the likelihood never read.
    @testset "marginalize_gamma refuses a prior it cannot use" begin
        t, v, e, inst = pack(["HARPS"])
        d = Data(t_rv = t, rv = v, rv_err = e, rv_inst = inst)
        mk(pr) = Params(; max_kplanet = 1, planet_modes = [Nereus.RV_ONLY],
                        instruments = InstrumentConfig(rv = ["HARPS"], pm = String[]),
                        data = d, priors = pr,
                        parametrization = ParametrizationConfig(marginalize_gamma = true))
        @test_throws ArgumentError mk(Dict{String, PriorSpec}(
            "gamma_HARPS" => UniformPrior(31_000.0, 31_500.0)))
        p = mk(Dict{String, PriorSpec}())
        @test !("gamma_HARPS" in p.layout.unfrozen_names)
    end
end
