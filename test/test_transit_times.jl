# Measured mid-transit times (`data.transit_times`): the loader and its
# validation, the likelihood against the model's own transit times, and fits:
# a linear ephemeris from transit times alone, a sinusoidal TTV through PM_TTV,
# and the period width of an RV fit with and without the transit times.

using Nereus, MCMCChains, Random, Statistics, Test
using Nereus: transit_times_model, anchor_to_tp, tp_to_tc, tc_to_tp

const TT_P = 2.7182818
const TT_T0 = 2_458_000.25
const TT_MIN = 1 / 1440                       # one minute, days

# A `data.transit_times` block in the job-config format.
function _tt_block(ep, tc, err; planet = 1, time_system = "BJD_TDB", source = nothing,
                   ref = Dict("t0" => TT_T0, "period" => TT_P, "epoch_zero" => 0),
                   extra...)
    vals = Dict{String, Any}("epoch" => collect(ep), "tc" => collect(tc),
                             "tc_err" => collect(err))
    vals["source"] = source === nothing ? ["SYN:$i" for i in eachindex(tc)] : source
    b = Dict{String, Any}("planet" => planet, "name" => "test b",
                          "time_system" => time_system, "values" => vals,
                          "reference_ephemeris" => ref)
    for (k, v) in extra
        b[String(k)] = v
    end
    return b
end

# A linear ephemeris observed at `n` random epochs over `span` cycles.
function _tt_linear(rng; n = 40, span = 2000, err = 0.5 * TT_MIN, noise = true)
    ep = sort(unique(rand(rng, 0:span, n)))
    σ = fill(err, length(ep))
    tc = TT_T0 .+ TT_P .* ep .+ (noise ? σ .* randn(rng, length(ep)) : 0.0)
    return ep, tc, σ
end

# n_steps counts the burn-in.
_tt_sampler(; n_walkers = 32, n_steps = 1500, n_burnin = 1000) =
    Dict("name" => "pt_emcee", "kwargs" => Dict("show_progress" => false,
         "n_temps" => 2, "n_walkers" => n_walkers, "n_steps" => n_steps,
         "n_burnin" => n_burnin))

const _TT_OUT = Dict("plots" => String[], "ppc" => false, "detection_limits" => false,
                     "loo" => false)

# The transit geometry and the eccentricity held fixed: transit times say nothing
# about them, and freeing them only slows the small fits below.
const _TT_FIXED_GEOM = Dict(
    "sesinw_k1" => Dict("type" => "fixed", "args" => [0.0]),
    "secosw_k1" => Dict("type" => "fixed", "args" => [0.0]),
    "b_k1" => Dict("type" => "fixed", "args" => [0.3]),
    "rr_k1" => Dict("type" => "fixed", "args" => [0.1]))

function _tt_draws(out, name)
    ch = Nereus.load_chains(joinpath(out, "chains.nc"))
    ch = ch isa Tuple ? first(ch) : ch
    return vec(Array(ch[Symbol(name)]))
end

_tt_err(f) = try
    f(); nothing
catch e
    e
end

# Weighted least-squares line tc = a + b·x: (a, b, σ_a, σ_b).
function _tt_wls(x, y, σ)
    w = 1 ./ σ .^ 2
    S, Sx, Sy, Sxx, Sxy = sum(w), sum(w .* x), sum(w .* y), sum(w .* x .^ 2),
                          sum(w .* x .* y)
    D = S * Sxx - Sx^2
    return ((Sxx * Sy - Sx * Sxy) / D, (S * Sxy - Sx * Sy) / D, sqrt(Sxx / D), sqrt(S / D))
end

@testset "transit times" begin

    @testset "config validation" begin
        ep, tc, σ = _tt_linear(MersenneTwister(1); n = 10)
        cfg(blocks; modes = ["PM_ONLY"], max_k = 1) = Dict{String, Any}(
            "output_dir" => mktempdir(), "sampler" => _tt_sampler(),
            "model" => Dict{String, Any}("max_kplanet" => max_k, "planet_modes" => modes),
            "data" => Dict{String, Any}("transit_times" => blocks))
        msg(c) = (e = _tt_err(() -> Nereus._validate_config(c)); e === nothing ? "" : e.msg)

        # Transit times alone are a complete config.
        @test Nereus._validate_config(cfg([_tt_block(ep, tc, σ)])) === nothing

        m = msg(cfg([_tt_block(ep, tc, σ; time_system = "BJD_UTC")]))
        @test occursin("time_system \"BJD_UTC\"", m) && occursin("BJD_TDB", m)
        m = msg(cfg([_tt_block(ep, tc[1:end-1], σ)]))
        @test occursin("different lengths", m) && occursin("tc = $(length(tc) - 1)", m)
        m = msg(cfg([_tt_block(ep, tc, σ; source = ["a"])]))
        @test occursin("different lengths", m) && occursin("source = 1", m)
        m = msg(cfg([_tt_block(ep, tc, σ; planet = 3)]))
        @test occursin("planet 3 does not exist", m)
        m = msg(cfg([_tt_block(ep, tc, σ; planet = 0)]))
        @test occursin("planet 0 does not exist", m)
        bad = copy(σ); bad[2] = 0.0
        m = msg(cfg([_tt_block(ep, tc, bad)]))
        @test occursin("tc_err must be finite and > 0", m) && occursin("[2]", m)
        m = msg(cfg([_tt_block(ep, tc, σ), _tt_block(ep, tc, σ)]))
        @test occursin("second block for planet 1", m)
        # An RV-only planet predicts no transit: the message names the modes.
        m = msg(cfg([_tt_block(ep, tc, σ)]; modes = ["RV_ONLY"]))
        @test occursin("RV_ONLY", m) && occursin("RVPM", m) && occursin("PM_ONLY", m)
        # ... also when planet_modes is omitted (every planet RV_ONLY).
        c = cfg([_tt_block(ep, tc, σ)]); delete!(c["model"], "planet_modes")
        @test occursin("RVPM", msg(c))
        m = msg(cfg([_tt_block(ep, tc, σ; photometry_overlap = "sometimes")]))
        @test occursin("photometry_overlap", m)
        c = cfg([_tt_block(ep, tc, σ)]); c["data"]["transit_times"] = Dict("planet" => 1)
        @test occursin("list of per-planet blocks", msg(c))
        # The TTV slot keys map planets to a count or a list of transits.
        c = cfg([_tt_block(ep, tc, σ)]; modes = ["PM_TTV"])
        c["model"]["ttv_n_transits"] = Dict(); c["model"]["ttv_epochs"] = Dict("1" => [0, 4])
        @test Nereus._validate_config(c) === nothing
        c["model"]["ttv_epochs"] = [0, 4]
        @test occursin("model.ttv_epochs must map planet indices", msg(c))
        c["model"]["ttv_epochs"] = Dict("1" => [0.5])
        @test occursin("model.ttv_epochs must map planet indices", msg(c))
        c["model"]["ttv_epochs"] = Dict("1" => [0, 4]); c["model"]["ttv_n_transits"] = Dict("b" => 3)
        @test occursin("model.ttv_n_transits must map planet indices", msg(c))
    end

    @testset "TransitTimeSet / Data validation" begin
        ep, tc, σ = _tt_linear(MersenneTwister(2); n = 12)
        @test_throws ArgumentError TransitTimeSet(; planet = 1, epoch = ep, tc, tc_err = σ,
                                                  time_system = "HJD")
        @test_throws ArgumentError TransitTimeSet(; planet = 1, epoch = ep[1:end-1], tc,
                                                  tc_err = σ)
        @test_throws ArgumentError TransitTimeSet(; planet = 1, epoch = ep, tc,
                                                  tc_err = -σ)
        @test_throws ArgumentError TransitTimeSet(; planet = 1, epoch = ep .+ 0.5, tc,
                                                  tc_err = σ)
        # A mislabelled point: epoch off by one.
        ep2 = copy(ep); ep2[5] += 1
        e = _tt_err(() -> TransitTimeSet(; planet = 1, epoch = ep2, tc, tc_err = σ))
        @test e isa ArgumentError && occursin("epoch $(ep2[5])", e.msg)
        # Accepted: JSON-style Float epochs, a reference ephemeris as a Dict.
        s = TransitTimeSet(; planet = 1, epoch = Float64.(ep), tc, tc_err = σ,
                           reference = Dict("t0" => TT_T0, "period" => TT_P))
        @test s.epoch == ep && s.reference.epoch_zero == 0
        # One set per planet.
        @test_throws ArgumentError Data(; transit_times = [s, s])
        # Transit times alone are data; t_ref falls back to their median.
        d = Data(; transit_times = [s])
        @test n_transit_times(d) == length(tc)
        @test d.t_ref ≈ median(tc)
        # φ_c's epoch without photometry: the 1/tc_err²-weighted mean.
        w = 1 ./ σ .^ 2
        @test Nereus.phic_reference_epoch(d) ≈ sum(w .* tc) / sum(w) rtol = 0 atol = 1e-6
        # A planet that is not in the model, and one whose mode has no transit.
        ic = InstrumentConfig()
        s2 = TransitTimeSet(; planet = 2, epoch = ep, tc, tc_err = σ)
        e = _tt_err(() -> Params(; max_kplanet = 1, planet_modes = [PM_ONLY],
                                 instruments = ic, data = Data(; transit_times = [s2]),
                                 stability = :none))
        @test e isa ArgumentError && occursin("planet 2", e.msg)
        e = _tt_err(() -> Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                                 instruments = ic, data = d, stability = :none))
        @test e isa ArgumentError && occursin("RVPM", e.msg)
    end

    @testset "use mask, exclude_sources, photometry overlap" begin
        ep, tc, σ = _tt_linear(MersenneTwister(3); n = 15, span = 300)
        src = [isodd(i) ? "ETD:$i" : "TESS:S$i" for i in eachindex(tc)]
        use = trues(length(tc)); use[3] = false
        s = TransitTimeSet(; planet = 1, epoch = ep, tc, tc_err = σ, source = src, use,
                           exclude_sources = ["ETD"])
        @test s.excluded[3] == "use"
        @test all(s.excluded[i] == "exclude_sources" for i in 1:2:length(tc) if i != 3)
        @test s.used == [i for i in 2:2:length(tc)]
        # Photometry over two transits: those two leave the likelihood, for a
        # planet the photometry models, unless the block keeps them.
        i1, i2 = 4, 9
        tph = vcat(collect(range(tc[i1] - 0.1, tc[i1] + 0.1; length = 50)),
                   collect(range(tc[i2] - 0.1, tc[i2] + 0.1; length = 50)))
        dc = Dict{String, Any}(
            "transit_photometry" => [Dict("instrument" => "TESS", "values" => Dict(
                "bjd" => tph, "flux" => ones(100), "flux_err" => fill(1e-3, 100)))],
            "transit_times" => [_tt_block(ep, tc, σ)])
        d, = @test_logs (:warn, r"inside the fitted photometry") match_mode = :any Nereus._build_data(
            dc; pm_planets = Set([1]))
        tt = transit_time_set(d, 1)
        @test findall(==("photometry_overlap"), tt.excluded) == [i1, i2]
        @test n_transit_times(d) == length(tc) - 2
        dc["transit_times"] = [_tt_block(ep, tc, σ; photometry_overlap = "keep")]
        d, = Nereus._build_data(dc; pm_planets = Set([1]))
        @test n_transit_times(d) == length(tc)
        d, = Nereus._build_data(dc; pm_planets = Set{Int}())   # planet not modelled by it
        @test n_transit_times(d) == length(tc)
    end

    @testset "likelihood: the model's transit times" begin
        ep, tc, σ = _tt_linear(MersenneTwister(4); n = 25, span = 900, noise = false)
        ic = InstrumentConfig()
        # Exactly zero without transit times.
        drv = Data(; t_rv = [1.0, 2.0, 3.0], rv = zeros(3), rv_err = ones(3))
        prv = Params(; max_kplanet = 1, planet_modes = [RV_ONLY], instruments =
                     InstrumentConfig(rv = ["X"]), data = drv, stability = :none)
        @test transit_times_log_likelihood(Theta{Float64}(prv), drv) === 0.0

        # Under every time anchor, eccentric: the model's mid-transit times are
        # the inferior conjunctions (f = π/2 − ω) of the orbit, Tc + n·P.
        e, ω = 0.45, 2.1
        for anchor in (:phic, :Tc, :Mo, :Tp)
            s = TransitTimeSet(; planet = 1, epoch = ep, tc, tc_err = σ)
            d = Data(; transit_times = [s])
            pc = ParametrizationConfig(time = anchor)
            p = Params(; max_kplanet = 1, planet_modes = [PM_ONLY], instruments = ic,
                       data = d, stability = :none, parametrization = pc)
            th = Theta{Float64}(p)
            Nereus.set_param!(th, "P_k1", TT_P)
            Nereus.set_param!(th, "sesinw_k1", sqrt(e) * sin(ω))
            Nereus.set_param!(th, "secosw_k1", sqrt(e) * cos(ω))
            # The anchor value that puts a transit at TT_T0 + 300 P.
            Tc = TT_T0 + 300 * TT_P
            Tp = tc_to_tp(Tc, TT_P, e, ω)
            t_ref, trc = d.t_ref, p.config.parametrization.t_ref_c
            v = anchor === :Tc ? Tc : anchor === :Tp ? Tp :
                anchor === :Mo ? mod(2π * (t_ref - Tp) / TT_P, 2π) :
                mod(2π * (Tc - trc) / TT_P + π, 2π) - π
            Nereus.set_param!(th, (anchor === :Mo ? "Mo" : string(anchor)) * "_k1", v)
            m = only(transit_times_model(th, d))
            @test maximum(abs.(m.residual)) < 1e-7
            # f + ω = π/2 at the model time, by Kepler's equation solved here.
            Tp_th = anchor_to_tp(anchor, v, TT_P, e, ω, t_ref, trc)
            M = 2π * (m.model[1] - Tp_th) / TT_P
            E = M
            for _ in 1:60
                E -= (E - e * sin(E) - M) / (1 - e * cos(E))
            end
            f = 2atan(sqrt((1 + e) / (1 - e)) * tan(E / 2))
            @test abs(mod(f + ω - π / 2 + π, 2π) - π) < 1e-8
            # ln L at zero residuals.
            @test transit_times_log_likelihood(th, d) ≈ -sum(log.(2π .* σ .^ 2)) / 2
            # The labels: shifting every epoch changes nothing.
            ds = Data(; transit_times = [TransitTimeSet(; planet = 1, epoch = ep .+ 1000,
                                                        tc, tc_err = σ)])
            @test transit_times_log_likelihood(th, ds) == transit_times_log_likelihood(th, d)
        end

        s = TransitTimeSet(; planet = 1, epoch = ep, tc, tc_err = σ)
        d = Data(; transit_times = [s])
        p = Params(; max_kplanet = 1, planet_modes = [PM_ONLY], instruments = ic,
                   data = d, stability = :none, parametrization =
                   ParametrizationConfig(time = :Tc))
        th = Theta{Float64}(p)
        for (nm, v) in ("P_k1" => TT_P, "Tc_k1" => TT_T0 + 450 * TT_P,
                        "sesinw_k1" => 0.0, "secosw_k1" => 0.0)
            Nereus.set_param!(th, nm, v)
        end
        ll0 = transit_times_log_likelihood(th, d)
        # The Gaussian: a 2 min shift costs Σ (2 min / σ)² / 2 (to the
        # precision of a BJD in Float64, ~1e-9 d on a 2 min residual).
        Nereus.set_param!(th, "Tc_k1", TT_T0 + 450 * TT_P + 2TT_MIN)
        @test transit_times_log_likelihood(th, d) ≈ ll0 - sum((2TT_MIN ./ σ) .^ 2) / 2 atol = 1e-3
        # The guard: a period that drifts more than P/4 from a point over the
        # data's span refuses the labels.
        Nereus.set_param!(th, "Tc_k1", TT_T0 + 450 * TT_P)
        Nereus.set_param!(th, "P_k1", TT_P + 0.3 * TT_P / 450)
        @test transit_times_log_likelihood(th, d) == -Inf
        Nereus.set_param!(th, "P_k1", TT_P + 0.2 * TT_P / 900)
        @test isfinite(transit_times_log_likelihood(th, d))
        # Excluded points leave ln L as if they were absent.
        Nereus.set_param!(th, "P_k1", TT_P + 1e-6)
        use = trues(length(tc)); use[[2, 7]] .= false
        du = Data(; transit_times = [TransitTimeSet(; planet = 1, epoch = ep, tc,
                                                    tc_err = σ, use)])
        keep = findall(use)
        dk = Data(; transit_times = [TransitTimeSet(; planet = 1, epoch = ep[keep],
                                                    tc = tc[keep], tc_err = σ[keep])])
        @test transit_times_log_likelihood(th, du) ≈ transit_times_log_likelihood(th, dk)
        # The jitter: off by default, on through the block or tt_jitter, a
        # LogUniform prior, added in quadrature.
        @test !any(startswith("sigma_tt"), p.layout.unfrozen_names)
        dj = Data(; transit_times = [TransitTimeSet(; planet = 1, epoch = ep, tc,
                                                    tc_err = σ, jitter = true)])
        pj = Params(; max_kplanet = 1, planet_modes = [PM_ONLY], instruments = ic,
                    data = dj, stability = :none,
                    parametrization = ParametrizationConfig(time = :Tc))
        @test "sigma_tt_k1" in pj.layout.unfrozen_names
        pr = pj.config.priors["sigma_tt_k1"]
        @test pr.dist isa Nereus.Distributions.LogUniform
        @test pr.lo < 0.01 * σ[1] && pr.hi >= 10 * σ[1]
        thj = Theta{Float64}(pj)
        for (nm, v) in ("P_k1" => TT_P, "Tc_k1" => TT_T0 + 450 * TT_P + 2TT_MIN,
                        "sesinw_k1" => 0.0, "secosw_k1" => 0.0, "sigma_tt_k1" => 3TT_MIN)
            Nereus.set_param!(thj, nm, v)
        end
        v = σ .^ 2 .+ (3TT_MIN)^2
        @test transit_times_log_likelihood(thj, dj) ≈
              -sum((2TT_MIN)^2 ./ v .+ log.(2π .* v)) / 2 atol = 1e-4
        @test Nereus.sci_param_unit("sigma_tt_k1", pj) == ("d", false)
        # The total adds it.
        @test rv_log_likelihood(thj, dj) ≈ transit_times_log_likelihood(thj, dj)
    end

    @testset "likelihood: N-body TTVs from the photometric solver" begin
        # RV for the masses, transit times for planet 1, no photometry.
        t_rv = collect(range(2_450_000.0, 2_450_060.0; length = 10))
        ep = collect(0:4:120)
        tc0 = 2_450_001.3 .+ 4.0 .* ep
        σ = fill(TT_MIN, length(ep))
        data = Data(; t_rv, rv = zeros(10), rv_err = ones(10),
                    transit_times = [TransitTimeSet(; planet = 1, epoch = ep, tc = tc0,
                                                    tc_err = σ)])
        params = Params(; max_kplanet = 2, planet_modes = [RVPM_TTV_NB, RVPM_TTV_NB],
                        instruments = InstrumentConfig(rv = ["HARPS"]), data,
                        parametrization = ParametrizationConfig(time = :Tc),
                        M_s = 1.0, R_s = 1.0, stability = :none)
        th = Theta{Float64}(params)
        orb = ((P = 4.0, K = 4.0, e = 0.05, ω = 0.7, Tc = 2_450_001.3, b = 0.2, rr = 0.1),
               (P = 8.1, K = 3.0, e = 0.04, ω = -1.2, Tc = 2_450_002.9, b = 0.0, rr = 0.08))
        for (k, o) in enumerate(orb)
            for (nm, v) in ("P" => o.P, "K" => o.K, "Tc" => o.Tc, "b" => o.b, "rr" => o.rr,
                            "sesinw" => sqrt(o.e) * sin(o.ω), "secosw" => sqrt(o.e) * cos(o.ω))
                Nereus.set_param!(th, "$(nm)_k$k", v)
            end
        end
        Nereus.set_param!(th, "sigma_HARPS", 1.0)
        m = only(transit_times_model(th, data))
        lin = orb[1].Tc .+ (ep .+ m.delta) .* orb[1].P
        δ = m.model .- lin
        @test all(isfinite, δ)
        @test maximum(abs.(δ)) > 10 / 86400          # real TTVs, seconds or more
        # The same offsets the photometric path reads, over the same span.
        p_idx = planet_indices(th)
        st = Nereus._tt_ttv_state(th, data, p_idx)
        @test st.n0s[1] <= 0 && length(st.δts[1]) >= 121
        @test δ ≈ [st.δts[1][n - st.n0s[1] + 1] for n in ep .+ m.delta]
        # Transit times at the model's own times: zero residuals.
        dm = Data(; t_rv, rv = zeros(10), rv_err = ones(10),
                  transit_times = [TransitTimeSet(; planet = 1, epoch = ep, tc = m.model,
                                                  tc_err = σ)])
        @test transit_times_log_likelihood(th, dm) ≈ -sum(log.(2π .* σ .^ 2)) / 2
    end

    @testset "a prior object in the priors block passes validation" begin
        ep, tc, σ = _tt_linear(MersenneTwister(6); n = 10, span = 100)
        @test Nereus._validate_config(Dict{String, Any}(
            "output_dir" => mktempdir(), "sampler" => _tt_sampler(),
            "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["PM_ONLY"]),
            "priors" => Dict{String, Any}("P_k1" => Nereus.UniformPrior(9.0, 11.0)),
            "data" => Dict{String, Any}("transit_times" => [_tt_block(ep, tc, σ)]))) === nothing
    end

    @testset "PM_TTV_NB: not exported, not a config mode, refused by Params" begin
        @test !(:PM_TTV_NB in names(Nereus))
        ep, tc, σ = _tt_linear(MersenneTwister(5); n = 10, span = 100)
        d = Data(; transit_times = [TransitTimeSet(; planet = 1, epoch = ep, tc,
                                                   tc_err = σ)])
        nb = Nereus.PM_TTV_NB
        e = _tt_err(() -> Params(; max_kplanet = 2, planet_modes = [nb, nb],
                                 instruments = InstrumentConfig(), data = d,
                                 M_s = 1.0, R_s = 1.0, stability = :none))
        @test e isa ArgumentError && occursin("RVPM_TTV_NB", e.msg) &&
              occursin("PM_TTV", e.msg) && occursin("planet 1", e.msg)
        e = _tt_err(() -> Nereus._validate_config(Dict{String, Any}(
            "output_dir" => mktempdir(), "sampler" => _tt_sampler(),
            "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["PM_TTV_NB"]),
            "data" => Dict{String, Any}("transit_times" => [_tt_block(ep, tc, σ)]))))
        @test e isa ArgumentError && occursin("unknown planet_mode `PM_TTV_NB`", e.msg)
        @test !("PM_TTV_NB" in Nereus._KNOWN_PLANET_MODES) &&
              !haskey(Nereus._PLANET_MODES, "PM_TTV_NB")
    end

    @testset "fit: linear ephemeris from transit times alone" begin
        rng = MersenneTwister(2026)
        ep, tc, σ = _tt_linear(rng; n = 40, span = 2000)
        out = mktempdir()
        cfg = Dict{String, Any}("seed" => 5, "output_dir" => out,
            "data" => Dict("transit_times" => [_tt_block(ep, tc, σ)]),
            "model" => Dict("max_kplanet" => 1, "planet_modes" => ["PM_ONLY"]),
            "priors" => _TT_FIXED_GEOM,
            "sampler" => _tt_sampler(n_walkers = 16, n_steps = 1500, n_burnin = 1000),
            "output" => merge(_TT_OUT, Dict("plots" => ["ttv_oc"])))
        s = Nereus.run_job(cfg)
        @test s["status"] == "ok"
        blk = s["transit_times"]
        @test blk["status"] == "ok"
        pl = only(blk["planets"])
        eph = pl["ephemeris"]
        σt = (eph["t0_err_minus"] + eph["t0_err_plus"]) / 2
        σP = (eph["period_err_minus"] + eph["period_err_plus"]) / 2
        # The truth within 1σ.
        @test abs(eph["t0"] - TT_T0) < σt
        @test abs(eph["period"] - TT_P) < σP
        # The posterior is the weighted least-squares solution (a linear model,
        # flat priors): same centre, same width.
        a, b, σa, σb = _tt_wls(Float64.(ep), tc .- TT_T0, σ)
        @test abs(eph["t0"] - (TT_T0 + a)) < 0.25σa
        @test abs(eph["period"] - b) < 0.25σb
        @test isapprox(σt, σa; rtol = 0.25) && isapprox(σP, σb; rtol = 0.25)
        @info "transit times, linear ephemeris" t0_minus_truth = eph["t0"] - TT_T0 σt σa P_minus_truth = eph["period"] - TT_P σP σb red_chi2 = pl["red_chi2"]
        # The O−C block.
        oc = pl["oc"]
        @test pl["n_points"] == length(tc) && pl["n_excluded"] == 0
        @test oc["epoch"] == ep && oc["tc"] == tc && oc["source"][1] == "SYN:1"
        @test oc["residual"] ≈ oc["tc"] .- oc["model"]
        @test 0.5 < pl["red_chi2"] < 1.8
        @test pl["sigma_tt"] === nothing && eph["epoch"] == 0
        @test haskey(s["figures"], "models/ttv_oc")
    end

    @testset "fit: a sinusoidal TTV through PM_TTV" begin
        rng = MersenneTwister(7)
        N = 30
        ep = collect(0:N-1)
        A, Pttv, φ = 5TT_MIN, 7.3, 0.4
        sinus = A .* sin.(2π .* ep ./ Pttv .+ φ)
        σ = fill(0.5TT_MIN, N)
        tc = TT_T0 .+ TT_P .* ep .+ sinus .+ σ .* randn(rng, N)
        out = mktempdir()
        pri = merge(_TT_FIXED_GEOM, Dict(
            "ttv_k1_t$i" => Dict("type" => "truncnorm", "args" => [0.0, 10TT_MIN, -0.05, 0.05])
            for i in 1:N))
        cfg = Dict{String, Any}("seed" => 9, "output_dir" => out,
            "data" => Dict("transit_times" => [_tt_block(ep, tc, σ)]),
            # The default anchor: φ_c's epoch, and Tc, mid-data. The offsets are
            # numbered from the first observed transit, so slot i is epoch i − 1
            # wherever Tc sits.
            "model" => Dict("max_kplanet" => 1, "planet_modes" => ["PM_TTV"],
                            "ttv_n_transits" => Dict("1" => N)),
            "priors" => pri,
            "sampler" => _tt_sampler(n_walkers = 96, n_steps = 6000, n_burnin = 3000),
            "output" => _TT_OUT)
        s = Nereus.run_job(cfg)
        @test s["status"] == "ok"
        δ = [median(_tt_draws(out, "ttv_k1_t$i")) for i in 1:N]
        # The offsets carry the TTV less a linear ephemeris, which Tc and P
        # absorb: compare them with a sinusoid plus a line.
        X = hcat(ones(N), ep, sin.(2π .* ep ./ Pttv), cos.(2π .* ep ./ Pttv))
        c = X \ δ
        amp, ph = hypot(c[3], c[4]), atan(c[4], c[3])
        @info "transit times, PM_TTV sinusoid" amp_min = amp / TT_MIN A_min = A / TT_MIN phase = ph φ
        @test isapprox(amp, A; rtol = 0.1)
        @test abs(mod(ph - φ + π, 2π) - π) < 0.1
        # The residual about that fit is the measurement noise, not the TTV.
        @test std(δ .- X * c) < 1.5 * 0.5TT_MIN
        pl = only(s["transit_times"]["planets"])
        @test 0.0 < pl["red_chi2"] < 2.0
    end

    @testset "fit: RV + transit times narrow the period" begin
        rng = MersenneTwister(11)
        t0 = 2_460_000.0
        P, Tc, K = 4.137, t0 + 1.31, 25.0
        trv = sort(t0 .+ 60 .* rand(rng, 30))
        rv = K .* sin.(2π .* (trv .- Tc) ./ P .+ π) .+ 3.0 .* randn(rng, 30)
        ep = sort(unique(rand(rng, 0:400, 15)))
        σ = fill(TT_MIN, length(ep))
        tc = Tc .+ P .* ep .+ σ .* randn(rng, length(ep))
        function fit_P(with_tt)
            out = mktempdir()
            data = Dict{String, Any}("rv" => Dict("values" => Dict("bjd" => trv,
                "rv" => rv, "rv_err" => fill(3.0, 30), "instrument" => fill("HARPS", 30))))
            pri = copy(_TT_FIXED_GEOM)
            if with_tt
                data["transit_times"] = [_tt_block(ep, tc, σ; ref = nothing)]
            else
                pri["P_k1"] = Dict("type" => "uniform", "args" => [4.0, 4.3])
            end
            s = Nereus.run_job(Dict{String, Any}("seed" => 3, "output_dir" => out,
                "star" => Dict("M_s" => 1.0, "R_s" => 1.0), "data" => data,
                "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RVPM"]),
                "priors" => pri, "sampler" => _tt_sampler(), "output" => _TT_OUT))
            return s, _tt_draws(out, "P_k1")
        end
        s0, P0 = fit_P(false)
        s1, P1 = fit_P(true)
        @test s0["status"] == "ok" && s1["status"] == "ok"
        @test !haskey(s0, "transit_times") && haskey(s1, "transit_times")
        @info "transit times, period width" std_P_rv = std(P0) std_P_rv_tt = std(P1) ratio = std(P1) / std(P0)
        @test std(P1) < 0.05 * std(P0)
        @test abs(median(P1) - P) < 4 * std(P1)
        @test abs(median(P0) - P) < 4 * std(P0)
    end

    @testset "fit_ttv: transit times alone" begin
        rng = MersenneTwister(12)
        ep = sort(setdiff(0:23, [3, 9, 15, 17]))
        σ = fill(0.5TT_MIN, length(ep))
        tc = TT_T0 .+ TT_P .* ep .+ σ .* randn(rng, length(ep))
        out = mktempdir()
        r = Nereus.fit_ttv(; transit_times = [_tt_block(ep, tc, σ)], planets = 1,
                           priors = _TT_FIXED_GEOM, output_dir = out,
                           engine = Dict("engine" => "pt_emcee", "options" => Dict(
                               "show_progress" => false, "n_temps" => 2,
                               "n_walkers" => 64, "n_steps" => 600, "n_burnin" => 300)),
                           output = Dict("ppc" => false, "detection_limits" => false,
                                         "loo" => false))
        @test r.summary["status"] == "ok"
        @test r.summary["transit_times"]["status"] == "ok"
        @test only(r.summary["transit_times"]["planets"])["n_points"] == length(tc)
        # run_job's default: one free offset per transit time, on its transit.
        ch = Nereus.load_chains(joinpath(out, "chains.nc"))
        ch = ch isa Tuple ? first(ch) : ch
        @test Symbol("ttv_k1_t$(length(ep))") in names(ch)
        @test !(Symbol("ttv_k1_t$(length(ep) + 1)") in names(ch))
        sl = only(r.summary["ttv_slots"]["planets"])
        @test sl["source"] == "default"
        @test sl["epochs"] == ep .- ep[1]
        @test sl["transit_times_epoch"] == ep
        @test sl["slots"][end] == "ttv_k1_t$(length(ep))"

        # The keys: explicit (one Int, or a Dict), else left to run_job's
        # default; refused for a planet with photometry and no transit times.
        blk = _tt_block(ep, tc, σ)
        m = Nereus._fit_ttv_slots!(Dict{String, Any}(), 1, nothing, nothing, [blk], false)
        @test !haskey(m, "ttv_n_transits") && !haskey(m, "ttv_epochs")
        m = Nereus._fit_ttv_slots!(Dict{String, Any}(), 2, 7, nothing, [blk], false)
        @test m["ttv_n_transits"] == Dict("1" => 7, "2" => 7)
        m = Nereus._fit_ttv_slots!(Dict{String, Any}(), 1, Dict(1 => 30), nothing, [blk], false)
        @test m["ttv_n_transits"] == Dict("1" => 30)
        m = Nereus._fit_ttv_slots!(Dict{String, Any}(), 1, nothing, Dict(1 => [0, 5]),
                                   nothing, true)
        @test m["ttv_epochs"] == Dict("1" => [0, 5])
        e = _tt_err(() -> Nereus._fit_ttv_slots!(Dict{String, Any}(), 1, nothing, nothing,
                                                 nothing, true))
        @test e isa ArgumentError && occursin("ttv_n_transits = Dict(1 => N)", e.msg) &&
              occursin("ttv_epochs", e.msg)
        e = _tt_err(() -> Nereus._fit_ttv_slots!(Dict{String, Any}(), 2, nothing, nothing,
                                                 [blk], true))
        @test e isa ArgumentError && occursin("planet(s) 2", e.msg)
        tph = collect(range(TT_T0 - 2TT_P - 0.1, TT_T0 - 2TT_P + 0.1; length = 50))
        e = _tt_err(() -> Nereus.fit_ttv(; phot = Dict("TESS" => (t = tph,
                         flux = ones(50), flux_err = fill(1e-3, 50))), planets = 1))
        @test e isa ArgumentError && occursin("ttv_n_transits", e.msg)
        # N-body: no offsets, and the masses need RV.
        e = _tt_err(() -> Nereus.fit_ttv(; transit_times = [blk], nbody = true))
        @test e isa ArgumentError && occursin("needs rv", e.msg)
        e = _tt_err(() -> Nereus.fit_ttv(; transit_times = [blk], nbody = true,
                                         rv = Dict("H" => (t = [1.0, 2.0], rv = [0.0, 0.0],
                                                           rv_err = [1.0, 1.0])),
                                         ttv_n_transits = 5))
        @test e isa ArgumentError && occursin("Drop them", e.msg)
        # The engine's options reach the sampler.
        @test Nereus._sampler_block(nothing) == Dict{String, Any}()
        @test Nereus._sampler_block(Dict("engine" => "pt_emcee",
                                         "options" => Dict("n_steps" => 5))) ==
              Dict{String, Any}("name" => "pt_emcee", "kwargs" => Dict("n_steps" => 5))
    end

    @testset "ttv_epochs: slots on listed transits" begin
        # The slot lookup.
        @test Nereus.ttv_slot(7, 3, [0, 7, 30]) == 2
        @test Nereus.ttv_slot(30, 3, [0, 7, 30]) == 3
        @test Nereus.ttv_slot(8, 3, [0, 7, 30]) == 0
        @test Nereus.ttv_slot(-1, 3, [0, 7, 30]) == 0
        @test Nereus.ttv_slot(2, 3, Int[]) == 3
        @test Nereus.ttv_slot(3, 3, Int[]) == 0
        @test Nereus.ttv_slot(-1, 3, Int[]) == 0

        ep = [100, 107, 130, 150]
        tc = TT_T0 .+ TT_P .* (ep .- 100)
        σ = fill(TT_MIN, 4)
        data = Data(; transit_times = [TransitTimeSet(; planet = 1, epoch = ep, tc, tc_err = σ)])
        mk(; kw...) = Params(; max_kplanet = 1, planet_modes = [PM_TTV],
                               instruments = InstrumentConfig(), data,
                               parametrization = ParametrizationConfig(time = :Tc),
                               stability = :none, kw...)
        p = mk(ttv_epochs = Dict(1 => [0, 7, 30]))
        @test [n for n in p.layout.unfrozen_names if startswith(n, "ttv_")] ==
              ["ttv_k1_t1", "ttv_k1_t2", "ttv_k1_t3"]
        @test p.config.ttv_n_transits[1] == 3
        @test Nereus.ttv_slot_epochs(p.config, 1) == [0, 7, 30]
        @test Nereus.ttv_name_epoch(p.config, "ttv_k1_t2") == 7
        @test Nereus.ttv_name_epoch(p.config, "ttv_k1_t4") === nothing
        @test Nereus.ttv_name_epoch(p.config, "P_k1") === nothing
        # Offsets on the three listed transits; the fourth has none.
        th = Theta{Float64}(p)
        Nereus.set_param!(th, "P_k1", TT_P); Nereus.set_param!(th, "Tc_k1", TT_T0)
        Nereus.set_param!(th, "sesinw_k1", 0.0); Nereus.set_param!(th, "secosw_k1", 0.0)
        off = [3.0, -2.0, 4.0] .* TT_MIN
        for i in 1:3
            Nereus.set_param!(th, "ttv_k1_t$i", off[i])
        end
        m = only(transit_times_model(th, data))
        @test m.model .- tc ≈ [off; 0.0] atol = 1e-9
        # ttv_n_transits = N is ttv_epochs = 0:N-1.
        q = mk(ttv_n_transits = Dict(1 => 3))
        @test Nereus.ttv_slot_epochs(q.config, 1) == [0, 1, 2]
        @test Nereus.ttv_name_epoch(q.config, "ttv_k1_t3") == 2
        # Refused: both keys, unsorted, repeated, before the first transit.
        for (kw, msg) in ((Dict(:ttv_n_transits => Dict(1 => 3),
                                :ttv_epochs => Dict(1 => [0, 7])), "in both"),
                          (Dict(:ttv_epochs => Dict(1 => [7, 0])), "strictly increasing"),
                          (Dict(:ttv_epochs => Dict(1 => [0, 7, 7])), "strictly increasing"),
                          (Dict(:ttv_epochs => Dict(1 => [-1, 7])), "first observed"))
            e = _tt_err(() -> mk(; kw...))
            @test e isa ArgumentError && occursin(msg, e.msg)
        end
        # The fingerprint of a model without listed epochs is the old one.
        @test isempty(q.config.ttv_epochs)
        @test Nereus._hash_skip(q.config, :ttv_epochs)
        @test !Nereus._hash_skip(p.config, :ttv_epochs)
    end

    @testset "runner default: offsets on the observed transits" begin
        ep = [0, 1, 2, 5, 8, 9, 13, 20]
        σ = fill(0.5TT_MIN, length(ep))
        tc = TT_T0 .+ TT_P .* ep
        blk = _tt_block(ep, tc, σ)
        use = trues(length(ep)); use[end] = false
        blk_u = deepcopy(blk); blk_u["values"]["use"] = use
        # A light curve two transits before epoch 0 (one transit window), and
        # one spanning epochs 3 and 4 (the transit at 5 has a transit time).
        tph = vcat(collect(range(TT_T0 - 2TT_P - 0.1, TT_T0 - 2TT_P + 0.1; length = 50)),
                   collect(range(TT_T0 + 2.6TT_P, TT_T0 + 4.4TT_P; length = 300)))
        phot = [Dict("instrument" => "TESS", "values" => Dict("bjd" => tph,
                     "flux" => ones(length(tph)), "flux_err" => fill(1e-3, length(tph))))]
        data_of(dc) = first(Nereus._build_data(dc; pm_planets = [1]))
        mdl = Dict{String, Any}("planet_modes" => ["PM_TTV"])
        # Transit times alone: the transits of those in the likelihood.
        d1 = data_of(Dict{String, Any}("transit_times" => [blk_u]))
        c, e = @test_logs (:info, r"7 free transit-time offsets") Nereus._ttv_slot_kwargs(mdl, d1, [PM_TTV])
        @test isempty(c) && e[1] == ep[1:end-1]
        # With photometry: its transit windows too, numbered from the first.
        d2 = data_of(Dict{String, Any}("transit_times" => [blk], "transit_photometry" => phot))
        c, e = @test_logs (:info, r"3 more in the photometry") Nereus._ttv_slot_kwargs(mdl, d2, [PM_TTV])
        @test e[1] == sort(vcat([-2, 3, 4], ep)) .+ 2
        # An empty dict names no planet ("ttv_n_transits": {}).
        c, e = @test_logs (:info,) Nereus._ttv_slot_kwargs(
            Dict{String, Any}("ttv_n_transits" => Dict(), "ttv_epochs" => Dict()), d1, [PM_TTV])
        @test e[1] == ep[1:end-1]
        # A planet the config names keeps what it says, with no message.
        c, e = @test_logs Nereus._ttv_slot_kwargs(
            Dict{String, Any}("ttv_n_transits" => Dict("1" => 4)), d1, [PM_TTV])
        @test c == Dict(1 => 4) && isempty(e)
        c, e = @test_logs Nereus._ttv_slot_kwargs(
            Dict{String, Any}("ttv_epochs" => Dict("1" => [0, 3])), d1, [PM_TTV])
        @test isempty(c) && e[1] == [0, 3]
        # Photometry only: a linear ephemeris, said so.
        d3 = data_of(Dict{String, Any}("transit_photometry" => phot))
        c, e = @test_logs (:warn, r"linear ephemeris.*ttv_n_transits.*ttv_epochs") Nereus._ttv_slot_kwargs(mdl, d3, [PM_TTV])
        @test isempty(c) && isempty(e)
        # No analytic TTVs, no default.
        c, e = @test_logs Nereus._ttv_slot_kwargs(mdl, d1, [PM_ONLY])
        @test isempty(c) && isempty(e)
    end

    @testset "fit: ttv_n_transits and ttv_epochs = 0:N-1 give the same draws" begin
        rng = MersenneTwister(21)
        N = 8
        ep = collect(0:N-1)
        σ = fill(0.5TT_MIN, N)
        tc = TT_T0 .+ TT_P .* ep .+ 2TT_MIN .* sin.(ep) .+ σ .* randn(rng, N)
        pri = merge(_TT_FIXED_GEOM, Dict(
            "ttv_k1_t$i" => Dict("type" => "truncnorm", "args" => [0.0, 10TT_MIN, -0.05, 0.05])
            for i in 1:N))
        function run(slots)
            out = mktempdir()
            s = Nereus.run_job(Dict{String, Any}("seed" => 4, "output_dir" => out,
                "data" => Dict("transit_times" => [_tt_block(ep, tc, σ)]),
                "model" => merge(Dict{String, Any}("max_kplanet" => 1,
                                                   "planet_modes" => ["PM_TTV"]), slots),
                "priors" => pri,
                "sampler" => _tt_sampler(n_walkers = 32, n_steps = 400, n_burnin = 200),
                "output" => _TT_OUT))
            return s, out
        end
        s1, o1 = run(Dict("ttv_n_transits" => Dict("1" => N)))
        s2, o2 = run(Dict("ttv_epochs" => Dict("1" => ep)))
        @test s1["status"] == "ok" && s2["status"] == "ok"
        for nm in vcat(["P_k1", "phic_k1"], ["ttv_k1_t$i" for i in 1:N])
            @test _tt_draws(o1, nm) == _tt_draws(o2, nm)
        end
        @test s1["ttv_slots"]["planets"][1]["source"] == "ttv_n_transits"
        @test s2["ttv_slots"]["planets"][1]["source"] == "ttv_epochs"
        @test s1["ttv_slots"]["planets"][1]["epochs"] == s2["ttv_slots"]["planets"][1]["epochs"] == ep
        # Both keys for one planet: refused.
        e = _tt_err(() -> run(Dict("ttv_n_transits" => Dict("1" => N),
                                   "ttv_epochs" => Dict("1" => ep))))
        @test e isa ArgumentError && occursin("in both", e.msg)
    end

    @testset "fit: sparse transit times, one offset per observed transit" begin
        # 40 transit times over 3000 epochs (an ExoClock-like record), a
        # sinusoidal TTV on them. With neither ttv key, PM_TTV gets 40 slots on
        # the observed transits, not 3000.
        rng = MersenneTwister(33)
        ep = Int[]
        while length(ep) < 40
            ep = sort(unique(vcat(ep, rand(rng, 0:3000, 40 - length(ep)))))
        end
        A, Pttv, φ = 5TT_MIN, 137.3, 0.9
        inj = A .* sin.(2π .* ep ./ Pttv .+ φ)
        σ = fill(0.5TT_MIN, 40)
        tc = TT_T0 .+ TT_P .* ep .+ inj .+ σ .* randn(rng, 40)
        out = mktempdir()
        pri = merge(_TT_FIXED_GEOM, Dict(
            "ttv_k1_t$i" => Dict("type" => "truncnorm", "args" => [0.0, 10TT_MIN, -0.05, 0.05])
            for i in 1:40))
        cfg = Dict{String, Any}("seed" => 5, "output_dir" => out,
            "data" => Dict("transit_times" => [_tt_block(ep, tc, σ)]),
            "model" => Dict("max_kplanet" => 1, "planet_modes" => ["PM_TTV"],
                            "ttv_n_transits" => Dict()),
            "priors" => pri,
            "sampler" => _tt_sampler(n_walkers = 128, n_steps = 6000, n_burnin = 3000),
            "output" => _TT_OUT)
        s = @test_logs (:info, r"40 free transit-time offsets") match_mode = :any Nereus.run_job(cfg)
        @test s["status"] == "ok"
        sl = only(s["ttv_slots"]["planets"])
        @test sl["epochs"] == ep .- ep[1] && sl["transit_times_epoch"] == ep
        @test sl["source"] == "default"
        ch = Nereus.load_chains(joinpath(out, "chains.nc"))
        ch = ch isa Tuple ? first(ch) : ch
        @test !(Symbol("ttv_k1_t41") in names(ch))
        δ = [median(_tt_draws(out, "ttv_k1_t$i")) for i in 1:40]
        # The offsets carry the TTV less a linear ephemeris (Tc and P absorb
        # it): a sinusoid plus a line, at the observed epochs.
        X = hcat(ones(40), ep, sin.(2π .* ep ./ Pttv), cos.(2π .* ep ./ Pttv))
        c = X \ δ
        amp, ph = hypot(c[3], c[4]), atan(c[4], c[3])
        @info "transit times, sparse PM_TTV" amp_min = amp / TT_MIN A_min = A / TT_MIN phase = ph φ
        @test isapprox(amp, A; rtol = 0.1)
        @test abs(mod(ph - φ + π, 2π) - π) < 0.1
        @test std(δ .- X * c) < 1.5 * 0.5TT_MIN
        @test 0.0 < only(s["transit_times"]["planets"])["red_chi2"] < 2.0
    end
end
