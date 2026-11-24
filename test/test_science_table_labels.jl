# Units and LaTeX symbols in the science tables.
#
# Every failure below shipped in a real table. A gaia4 astrometric fit wrote
#
#     $\mathrm{inc}\,\mathrm{k1}$ & $2.114$          & deg    <- radians, labelled deg
#     $\omega\,\mathrm{k1}$       & $176.5^{+178.5}$ & deg    <- that is Omega, not omega
#     $\mathrm{M}\,\mathrm{sec}\,\mathrm{k1}$ & ...  &        <- no unit at all
#     $\mathrm{plx}$              & $13.624$         &        <- no unit at all
#
# for an inclination of 121.11 deg and a longitude of the ascending node. None
# of it is catchable by a reader of the table, which is what makes it worth a
# test rather than a comment.

using Test, Nereus, Random, MCMCChains

@testset "science table units" begin
    # inc and i are SAMPLED IN RADIANS and reported in degrees, so the
    # convert-from-radians flag must be set. `("deg", false)` means "call it
    # degrees and never convert".
    for nm in ("inc", "i")
        unit, is_rad = Nereus._SCI_UNITS[nm]
        @test unit == "deg"
        @test is_rad
    end
    # inc_deg is a DERIVED key, already in degrees: it must NOT be converted.
    @test Nereus._SCI_UNITS["inc_deg"] == ("deg", false)

    # Angles that were already right stay right.
    for nm in ("w", "omega", "Mo", "M0", "Omega", "lambda")
        @test Nereus._SCI_UNITS[nm] == ("deg", true)
    end

    # Sampled in astrometry, absent from the registry, so both published as
    # dimensionless.
    @test Nereus._SCI_UNITS["plx"]   == ("mas", false)
    @test Nereus._SCI_UNITS["M_sec"] == ("M_sun", false)
end

@testset "science table LaTeX symbols" begin
    # The ascending node is a CAPITAL omega. The greek lookup lowercased every
    # token first, so Omega_k1 was published as \omega -- the argument of
    # periastron, a different angle, in the same table.
    @test Nereus._texsym("Omega_k1") == "\${\\Omega}_{1}\$"
    @test occursin("\\Omega", Nereus._texsub("Omega_k1"))
    # ... without breaking the lowercase one.
    @test occursin("\\omega", Nereus._texsub("omega_k1"))
    @test !occursin("\\Omega", Nereus._texsub("omega_k1"))

    # The symbol table keyed only the DERIVED spellings (a_au, inc_deg), so
    # the FITTED table fell through to \mathrm{} for half its rows.
    @test Nereus._texsym("a_k1")     == "\${a}_{1}\$"
    @test Nereus._texsym("inc_k1")   == "\${i}_{1}\$"
    @test Nereus._texsym("M_sec_k1") == "\${M_{\\rm sec}}_{1}\$"
    @test Nereus._texsym("plx")      == "\$\\varpi\$"

    # Untouched: the ones that were already correct.
    @test Nereus._texsym("sesinw_k1") == "\${\\sqrt{e}\\sin\\omega}_{1}\$"
    @test Nereus._texsym("Mo_k1")     == "\${M_0}_{1}\$"
    @test Nereus._texsym("gamma_HARPS") == "\$\\gamma_{\\mathrm{HARPS}}\$"

    # The tell-tale of a missing symbol: the planet suffix leaking into roman
    # text as \mathrm{k1} instead of becoming the subscript _{1}.
    for nm in ("a_k1", "M_sec_k1", "inc_k1", "Omega_k1", "sesinw_k1", "Mo_k1")
        @test !occursin("\\mathrm{k", Nereus._texsym(nm))
    end
    # And no row may carry a raw underscore, which breaks math mode.
    for nm in ("a_k1", "M_sec_k1", "inc_k1", "Omega_k1", "plx", "rho_s",
               "sigma_HIRES", "gamma_Lick", "sesinw_k1")
        body = replace(Nereus._texsym(nm), "\\_" => "")   # escaped ones are fine
        @test !occursin(r"_[A-Za-z0-9]{2,}", body)
    end
end

@testset "science table rounding follows the smaller error" begin
    # Rounding to the larger error destroys the precise side of an asymmetric
    # interval: 0.484 +0.082 -1.027 became 0.5 +0.1 -1.0.
    v, lo, hi = Nereus._fmt3(0.4836, -1.027, 0.082)
    @test v == "0.484" && lo == "1.027" && hi == "0.082"
end

# rho_s is SAMPLED IN SOLAR UNITS, ρ★/ρ☉ (`rho_s_to_a_Rs`), and the fitted table
# labelled it g/cm3: an NGTS-33 summary read 0.532 g/cm³ for a 0.532 ρ☉ star,
# which is 0.750 g/cm³. The fitted value stays in the units it was sampled and
# its prior written in, now labelled so; the g/cm³ value is the derived
# rho_star_transit, ρ☉ × 1.411.
@testset "fitted stellar density: solar units, and the g/cm3 one derived" begin
    n = 400
    data = Data(; t_phot = collect(range(0.0, 20.0; length = n)), flux = ones(n),
                  flux_err = fill(3e-4, n), phot_inst = ones(Int, n))
    # ρ★ sampled as such (density = :rho_star); the default, where it is
    # derived from the sampled ρ_circ, is checked at the end.
    mkp(density) = Params(; max_kplanet = 1, planet_modes = [PM_ONLY],
                      instruments = InstrumentConfig(pm = ["TESS"]), data = data,
                      parametrization = ParametrizationConfig(; time = :Tc,
                                                              use_rho_s = true, density),
                      priors = Dict("P_k1" => UniformPrior(2.9, 3.1),
                                    "Tc_k1" => UniformPrior(0.9, 1.1)))
    params = mkp(:rho_star)
    rng = MersenneTwister(2)
    m = 2000
    val = Dict("P_k1" => 3.0, "sesinw_k1" => 0.0, "secosw_k1" => 0.0,
               "Tc_k1" => 1.0, "b_k1" => 0.3, "rr_k1" => 0.1,
               "offset_TESS" => 0.0, "jitter_TESS" => 1e-3, "q1_TESS" => 0.4,
               "q2_TESS" => 0.3, "rho_s" => 0.532)
    nm = params.layout.unfrozen_names
    @test Set(nm) == Set(keys(val))
    chains = Chains(hcat((val[k] .+ 1e-4 .* randn(rng, m) for k in nm)...), Symbol.(nm))
    out = mktempdir()
    sci = Nereus.science_summary(out, chains, params, data)

    rho = sci["fitted"]["parameters"]["rho_s"]
    @test rho["unit"] == "rho_sun"
    @test rho["value"] ≈ 0.532 rtol = 1e-3
    der = sci["derived"]["parameters"]["rho_star_transit_k1"]
    @test der["unit"] == "g/cm3"
    @test der["value"] ≈ 0.532 * Nereus.RHO_SUN_CGS rtol = 1e-3        # 0.750 g/cm³
    # ... and so in every table written
    dat = read(joinpath(out, "tables", "fitted.dat"), String)
    @test occursin(r"rho_s\s.*rho_sun", dat)
    @test !occursin(r"rho_s\s.*g/cm3", dat)
    tex = read(joinpath(out, "tables", "fitted.tex"), String)
    @test occursin("\$\\rho_\\odot\$", tex)
    @test occursin(r"\nrho_s,[^\n]*,rho_sun,", read(joinpath(out, "tables", "fitted.csv"), String))

    # density = :rho_circ: ρ★ is derived from the sampled ρ_circ = ρ★ g³ (here
    # e ≈ 0, so ρ_circ ≈ ρ★) and reported where it was, in the same units.
    pc = mkp(:rho_circ)
    nmc = pc.layout.unfrozen_names
    @test "rho_circ_k1" in nmc && !("rho_s" in nmc)
    valc = merge(val, Dict("rho_circ_k1" => 0.532))
    chc = Nereus.with_density_columns(
        Chains(hcat((valc[k] .+ 1e-4 .* randn(rng, m) for k in nmc)...), Symbol.(nmc)), pc)
    scc = Nereus.science_summary(mktempdir(), chc, pc, data)
    @test scc["fitted"]["parameters"]["rho_s"]["unit"] == "rho_sun"
    @test scc["fitted"]["parameters"]["rho_s"]["value"] ≈ 0.532 rtol = 1e-3
    @test scc["fitted"]["parameters"]["rho_circ_k1"]["unit"] == "rho_sun"
end

# Every parameter name a layout can hold resolves to an explicit unit rule.
# "" is a rule (dimensionless); falling through every rule is a gap -- which is
# how dvdt, i★ (radians!), M_pri, v sin i★, the ARoME β_p, the tomography line
# widths and every GP/activity hyperparameter reached the tables with no unit,
# and how a shared photometric jitter `jitter_TESS+K2` was labelled m/s.
include(joinpath(@__DIR__, "fixtures", "ccf_synth.jl"))

function _unit_layouts()
    N = Nereus
    rng = MersenneTwister(11); n = 30
    t = sort(100 .* rand(rng, n))
    ind = Data(t_rv = t, rv = randn(rng, n), rv_err = ones(n),
               rv_inst = [i <= 15 ? 1 : 2 for i in 1:n],
               indicators = Dict("bis" => randn(rng, n), "fwhm" => randn(rng, n),
                                 "log_rhk" => randn(rng, n)))
    rv2 = InstrumentConfig(rv = ["HARPS", "FEROS"])
    rvp(nms; kw...) = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                             instruments = rv2, stability = :none,
                             noise_models = NoiseModel[nms...], kw...)
    grid = collect(range(-70.0, 70.0; length = 30)); Tc = 2459000.0
    tt = collect(range(Tc - 0.05, Tc + 0.05; length = 8))
    tomo = Data(t_rv = [Tc - 0.4, Tc, Tc + 0.4], rv = [0.0, 5.0, -3.0],
                rv_err = fill(3.0, 3), rv_inst = [1, 1, 1], t_phot = tt,
                flux = fill(1.0, length(tt)), flux_err = fill(1e-3, length(tt)),
                phot_inst = ones(Int, length(tt)),
                tomo = [N.TomoNight("HARPS", tt, 0.01 .* randn(rng, length(tt), 30),
                                    grid, Tc)])
    tomop(nms) = Params(parametrization = ParametrizationConfig(time = :Mo),
                        max_kplanet = 1, planet_modes = [RVPM_RM],
                        instruments = InstrumentConfig(rv = ["I1"], pm = ["TESS"]),
                        data = tomo, stability = :none, M_s = 1.0, R_s = 1.0,
                        noise_models = NoiseModel[nms...])
    return [
        "RV + trend + rotation GP, shared γ" => Params(max_kplanet = 1,
            planet_modes = [RV_ONLY], trend_order = 2, stability = :none,
            instruments = InstrumentConfig(rv = ["HARPS", "FEROS", "PFS"]),
            sharing = Dict(:gamma => [["HARPS", "FEROS"]]),
            noise_models = NoiseModel[CeleriteRotation()]),
        "FM17" => rvp([CeleriteRotationFM17()]),
        "activity decorrelation" => rvp([ActivityDecorrelation(indicators = ["bis", "fwhm"],
                                                               derivative = true)]; data = ind),
        "activity jitter" => rvp([ActivityJitter(indicator = "log_rhk")]; data = ind),
        "MA + AR" => rvp([MAModel(order = 2), ARModel(order = 1, per_instrument = true)];
                         data = ind),
        "error scale" => rvp([ErrorScale()]), "nightly" => rvp([NightlyOffset()]),
        "harmonic" => rvp([HarmonicBlock()]), "Student-t" => rvp([StudentT()]),
        "restricted SHO/Matern + photometric SHO" => Params(max_kplanet = 1,
            planet_modes = [RVPM], stability = :none,
            instruments = InstrumentConfig(rv = ["HARPS", "FEROS"], pm = ["TESS"]),
            noise_models = NoiseModel[CeleriteSHO(instruments = ["HARPS"]),
                MaternGP(instruments = ["FEROS"]), CeleriteSHO(channel = :phot),
                NightlyOffset(channel = :phot), HarmonicBlock(channel = :phot)]),
        "activity GP + indicator floors" => rvp([ActivityGP(channels = [:bis, :fwhm, :logrhk, :halpha],
                                                            use_derivative = true),
                                                 IndicatorFloor(kernel = :qp),
                                                 IndicatorFloor(channels = [:bis], kernel = :white)];
                                                transdim_noise = true),
        "transit, shared jitter/offset group, baseline, ρ★" => Params(max_kplanet = 1,
            planet_modes = [PM_ONLY], phot_trend_order = 2, stability = :none,
            instruments = InstrumentConfig(pm = ["TESS", "K2"]),
            sharing = Dict(:pm_jitter => [["TESS", "K2"]], :pm_offset => [["TESS", "K2"]]),
            parametrization = ParametrizationConfig(use_rho_s = true, density = :rho_star,
                                                    time = :Tc)),
        "joint, r1/r2, a/R★ sampled" => Params(max_kplanet = 1, planet_modes = [RVPM],
            instruments = InstrumentConfig(rv = ["H"], pm = ["TESS"]), stability = :none,
            parametrization = ParametrizationConfig(geom = :r1r2, sample_a_Rs = true,
                                                    density = :rho_star, ew = :esinw,
                                                    time = :Tp)),
        "joint, ρ_circ" => Params(max_kplanet = 1, planet_modes = [RVPM], stability = :none,
            instruments = InstrumentConfig(rv = ["H"], pm = ["TESS"]),
            parametrization = ParametrizationConfig(use_rho_s = true)),
        "ARoME RM + gravity darkening" => Params(max_kplanet = 2, M_s = 1.0,
            planet_modes = [RVPM_RM_A, N.RVPM_GD], stability = :none,
            instruments = InstrumentConfig(rv = ["H"], pm = ["TESS"])),
        "reloaded RM" => Params(max_kplanet = 1, planet_modes = [RVPM_RM_R], stability = :none,
            instruments = InstrumentConfig(rv = ["H"], pm = ["TESS"])),
        "tomography" => tomop([]),
        "tomography, Matérn" => tomop([MaternGP(channel = :tomo, instruments = ["HARPS"])]),
        "tomography, SHO" => tomop([CeleriteSHO(channel = :tomo, instruments = ["HARPS"])]),
        "RV + astrometry, imager + IAD jitter" => Params(max_kplanet = 1, planet_modes = [RVAS],
            instruments = InstrumentConfig(rv = ["H"], as = ["GPI"], iad = ["HIP"]),
            stability = :none, M_s = 1.0, parametrization = ParametrizationConfig(time = :Mo)),
        "astrometry, M_sec sampled" => Params(max_kplanet = 1, planet_modes = [RVAS],
            instruments = InstrumentConfig(rv = ["H"]), stability = :none, M_s = 1.0,
            parametrization = ParametrizationConfig(time = :Tp, mass = :M_sec_driven)),
        "RV + transit + astrometry" => Params(max_kplanet = 1, planet_modes = [RVPMAS],
            instruments = InstrumentConfig(rv = ["H"], pm = ["TESS"]), stability = :none,
            M_s = 1.0),
        "SB2 + astrometry" => Params(max_kplanet = 1, planet_modes = [BINARY], M_s = 1.0,
            instruments = InstrumentConfig(rv = ["A"]), stability = :none,
            parametrization = ParametrizationConfig(time = :Tp)),
        "SB2" => Params(max_kplanet = 1, planet_modes = [BINARY_RV], stability = :none,
            instruments = InstrumentConfig(rv = ["A"]),
            parametrization = ParametrizationConfig(time = :Tp)),
        "TTV" => Params(max_kplanet = 1, planet_modes = [RVPM_TTV],
            ttv_n_transits = Dict(1 => 3), stability = :none,
            instruments = InstrumentConfig(rv = ["H"], pm = ["TESS"])),
        "CCF components" => first(cs_model(; synth = (; n_epochs = 6))),
    ]
end

@testset "every layout name has a unit, rendered on every path" begin
    for (family, p) in _unit_layouts()
        @testset "$family" begin
            for nm in p.layout.names
                nm == "n_p" && continue
                known = Nereus.sci_unit_known(nm, p)
                known || @info "no unit rule" family nm
                @test known
                u, _ = Nereus.sci_param_unit(nm, p)
                # the .tex tables and the figure labels render the same unit
                # from one table; no unit falls back to escaped text
                @test haskey(Nereus._UNIT_FORMS, u)
                @test Nereus._texunit(u) == Nereus._UNIT_FORMS[u][1]
                @test Nereus._sci_unit_latex(u) == Nereus._UNIT_FORMS[u][2]
            end
        end
    end
    # every unit of the registries has its renderings
    for u in vcat([u for (u, _) in values(Nereus._SCI_UNITS)],
                  [u for (u, _) in values(Nereus._SCI_SYSTEM_UNITS)],
                  [u for (_, u) in Nereus._SCI_PREFIX_UNITS],
                  collect(values(Nereus._DERIVED_UNITS)))
        @test haskey(Nereus._UNIT_FORMS, u)
    end
end

@testset "units that were missing or wrong" begin
    L = Dict(_unit_layouts())
    u(f, nm) = Nereus.sci_param_unit(nm, L[f])
    # sampled in radians and published as raw radians with no unit
    @test u("ARoME RM + gravity darkening", "i_star") == ("deg", true)
    @test u("ARoME RM + gravity darkening", "v_sin_i_star") == ("m/s", false)
    @test u("ARoME RM + gravity darkening", "beta_p") == ("m/s", false)
    @test u("ARoME RM + gravity darkening", "sigma_ccf") == ("m/s", false)
    @test u("RV + astrometry, imager + IAD jitter", "M_pri") == ("M_sun", false)
    @test u("tomography", "tomo_sigma_line_HARPS") == ("km/s", false)
    @test u("tomography", "tomo_ell_v_HARPS") == ("km/s", false)
    @test u("tomography, Matérn", "matern_rho_tomo_HARPS") == ("d", false)
    @test u("tomography, Matérn", "matern_sigma_tomo_HARPS") == ("", false)
    # a shared photometric group was read as an RV instrument: m/s
    @test u("transit, shared jitter/offset group, baseline, ρ★", "jitter_TESS+K2") == ("", false)
    @test u("RV + trend + rotation GP, shared γ", "gamma_HARPS+FEROS") == ("m/s", false)
    # GP and activity hyperparameters, by the channel they act on
    @test u("RV + trend + rotation GP, shared γ", "gp_sigma") == ("m/s", false)
    @test u("RV + trend + rotation GP, shared γ", "gp_period") == ("d", false)
    @test u("restricted SHO/Matern + photometric SHO", "gp_log_S0_HARPS") ==
          ("ln(m2/s2*day)", false)
    @test u("restricted SHO/Matern + photometric SHO", "gp_log_S0_phot") == ("ln(day)", false)
    @test u("restricted SHO/Matern + photometric SHO", "gp_log_omega0_phot") ==
          ("ln(1/day)", false)
    @test u("restricted SHO/Matern + photometric SHO", "matern_sigma_FEROS") == ("m/s", false)
    @test u("restricted SHO/Matern + photometric SHO", "harm_amp_TESS_phot") == ("", false)
    @test u("FM17", "gp_log_amp") == ("ln(m2/s2)", false)
    @test u("FM17", "gp_log_timescale") == ("ln(day)", false)
    @test u("activity decorrelation", "C_bis_HARPS") == ("m/s", false)
    @test u("activity decorrelation", "Cdot_bis_HARPS") == ("m/s*day", false)
    @test u("activity jitter", "jit_base_log_rhk_HARPS") == ("m/s", false)
    @test u("MA + AR", "ma_beta_1") == ("d", false)
    @test u("nightly", "night_sigma_HARPS") == ("m/s", false)
    @test u("harmonic", "harm_period") == ("d", false)
    @test u("activity GP + indicator floors", "Vc") == ("m/s", false)
    @test u("activity GP + indicator floors", "gp_act_period") == ("d", false)
    @test u("activity GP + indicator floors", "ind_floor_lambda_e") == ("d", false)
    @test u("CCF components", "ccf_jit_SIM_1") == ("", false)

    # derived keys that came out without a unit
    for (k, unit) in ("lambda_deg" => "deg", "lambda_abs_deg" => "deg",
                      "i_star_deg" => "deg", "psi_deg" => "deg",
                      "M_A" => "M_sun", "M_B" => "M_sun",
                      "M_A_sini3" => "M_sun", "M_B_sini3" => "M_sun",
                      "u1_TESS" => "", "Tsec" => "BJD", "g_p" => "m/s2")
        @test Nereus._derived_unit_lookup(k) == unit
    end
    @test Nereus._derived_unit_lookup("no_such_key") === nothing
end

# i★ end to end: the chain holds radians, every table reports degrees.
@testset "i_star is reported in degrees" begin
    p = Dict(_unit_layouts())["ARoME RM + gravity darkening"]
    nm = p.layout.unfrozen_names
    rng = MersenneTwister(3)
    mid(x) = (b = Nereus.bounds(p.config.priors[x]);
              isfinite(b[1]) && isfinite(b[2]) ? (b[1] + b[2]) / 2 : 1.0)
    vals = Dict(x => (x == "i_star" ? deg2rad(60.0) : mid(x)) for x in nm)
    m = 500
    ch = Chains(hcat((vals[x] .+ 1e-6 .* randn(rng, m) for x in nm)...), Symbol.(nm))
    e = Dict(first(Nereus.science_fitted(ch, p)))
    @test e["i_star"]["unit"] == "deg"
    @test e["i_star"]["value"] ≈ 60.0 rtol = 1e-4
    @test e["v_sin_i_star"]["unit"] == "m/s"
end
