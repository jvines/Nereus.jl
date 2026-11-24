# A trans-dimensional noise menu written in a JSON job config.
#
# `transdim.toggleable` and `transdim.noise_exclusion_groups` name entries of
# the top-level `noise_models` array, by 1-based index or by the entry's
# `label`. Before, run_job refused any non-empty `toggleable` ("JSON-specified
# toggleable models aren't supported via run_job"), so the only JSON route to a
# noise menu was the RV-oriented `noise_menu` block: a per-light-curve
# photometric menu {white | MaternGP | CeleriteSHO} could not be stated.
using Test, Random, Statistics
using JSON3
using Nereus
using Nereus: NoiseModel, MaternGP, CeleriteSHO, ErrorScale, NightlyOffset,
              TransDimConfig

const _JM_T0 = 2_460_000.0
const _JM_P, _JM_TC = 3.3, _JM_T0 + 1.2

# Two light curves of one transiting planet: a 30-min space sector and three
# ground nights, the ground one with a smooth correlated trend on top.
function _jm_lightcurves(; seed = 11)
    rng = MersenneTwister(seed)
    tA = collect(range(_JM_T0, _JM_T0 + 8.0; step = 30 / 1440))
    tB = Float64[]
    for c in (1.2, 4.5, 7.8)
        append!(tB, range(_JM_T0 + c - 0.15, _JM_T0 + c + 0.15; step = 6 / 1440))
    end
    dip(t) = (ph = mod(t - _JM_TC + _JM_P / 2, _JM_P) - _JM_P / 2;
              1.0 - 0.005 * exp(-0.5 * (ph / 0.035)^2))
    fA = dip.(tA) .+ 5e-4 .* randn(rng, length(tA))
    fB = dip.(tB) .+ 1e-3 .* randn(rng, length(tB)) .+ 1.5e-3 .* sin.(2π .* tB ./ 0.4)
    lc(t, f, s) = Dict{String, Any}("bjd" => t, "flux" => f, "flux_err" => fill(s, length(t)))
    return [Dict{String, Any}("instrument" => "SPACE", "values" => lc(tA, fA, 5e-4)),
            Dict{String, Any}("instrument" => "GROUND", "values" => lc(tB, fB, 1e-3))]
end

_jm_u(lo, hi) = Dict{String, Any}("type" => "UniformPrior", "args" => [lo, hi])
_jm_lu(lo, hi) = Dict{String, Any}("type" => "LogUniformPrior", "args" => [lo, hi])

# The per-light-curve menu, in the JSON shape: noise_models[1:2] are SPACE's
# two GPs, [3:4] GROUND's; each pair is one exclusion group (white = both off).
function _jm_noise_models()
    nm(kind, inst, label) = Dict{String, Any}("kind" => kind, "channel" => "phot",
                                              "instruments" => [inst], "label" => label)
    return Any[nm("MaternGP", "SPACE", "space_matern"), nm("CeleriteSHO", "SPACE", "space_sho"),
               nm("MaternGP", "GROUND", "ground_matern"), nm("CeleriteSHO", "GROUND", "ground_sho")]
end

function _jm_cfg(; toggleable = Any[1, 2, 3, 4], groups = Any[Any[1, 2], Any[3, 4]],
                 sampler = Dict{String, Any}("name" => "transdim_pt_emcee",
                     "kwargs" => Dict{String, Any}("n_temps" => 2, "n_walkers" => 8,
                         "n_steps" => 300, "n_burnin" => 150, "show_progress" => false)))
    td = Dict{String, Any}("max_kplanet" => 1, "planets" => false, "noise" => true,
                           "toggleable" => toggleable)
    groups === nothing || (td["noise_exclusion_groups"] = groups)
    return Dict{String, Any}("seed" => 3, "output_dir" => mktempdir(),
        "star" => Dict{String, Any}("M_s" => 1.0, "R_s" => 1.0),
        "priors" => Dict{String, Any}(
            "P_k1" => _jm_u(_JM_P - 0.02, _JM_P + 0.02),
            "rr_k1" => _jm_u(0.02, 0.15), "b_k1" => _jm_u(0.0, 0.9),
            "matern_sigma_phot_SPACE" => _jm_lu(1e-5, 1e-2),
            "matern_rho_phot_SPACE" => _jm_lu(0.02, 5.0),
            "matern_sigma_phot_GROUND" => _jm_lu(1e-5, 1e-2),
            "matern_rho_phot_GROUND" => _jm_lu(0.02, 5.0)),
        "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["PM_ONLY"]),
        "data" => Dict{String, Any}("transit_photometry" => _jm_lightcurves()),
        "noise_models" => _jm_noise_models(),
        "transdim" => td,
        "sampler" => sampler,
        "output" => Dict{String, Any}("plots" => String[], "ppc" => false,
                                      "detection_limits" => false, "loo" => false))
end

# What a JSON file delivers: JSON3 arrays and Int64 indices, not Julia Vectors.
_jm_json(cfg) = JSON3.read(JSON3.write(cfg), Dict{String, Any})

_jm_verrs(cfg) = try
    Nereus._validate_config(cfg); ""
catch err
    sprint(showerror, err)
end

@testset "run_job: JSON noise menu by index / label" begin

    @testset "validator: references into noise_models" begin
        @test _jm_verrs(_jm_cfg()) == ""
        @test _jm_verrs(_jm_json(_jm_cfg())) == ""
        # labels are as good as indices, and may be mixed
        @test _jm_verrs(_jm_cfg(toggleable = Any["space_matern", "space_sho", 3, 4],
                                groups = Any[Any["space_matern", 2], Any[3, "ground_sho"]])) == ""
        e = _jm_verrs(_jm_cfg(toggleable = Any[0, 2]))
        @test occursin("transdim.toggleable[1] = 0", e) && occursin("1..4", e)
        e = _jm_verrs(_jm_cfg(toggleable = Any[1, 5]))
        @test occursin("transdim.toggleable[2] = 5", e) && occursin("1..4", e)
        e = _jm_verrs(_jm_cfg(toggleable = Any[1, 2.5]))
        @test occursin("transdim.toggleable[2]", e) && occursin("integer", e)
        e = _jm_verrs(_jm_cfg(toggleable = Any[1, "nope"]))
        @test occursin("\"nope\"", e) && occursin("no noise_models entry has that label", e)
        e = _jm_verrs(_jm_cfg(toggleable = Any[1, 1, 2]))
        @test occursin("noise_models[1] listed twice", e)
        # a group member that is not toggleable
        e = _jm_verrs(_jm_cfg(toggleable = Any[1, 2, 3], groups = Any[Any[1, 2], Any[3, 4]]))
        @test occursin("transdim.noise_exclusion_groups[2]", e) &&
              occursin("noise_models[4]", e) && occursin("not in transdim.toggleable", e)
        e = _jm_verrs(_jm_cfg(groups = Any[Any[1, 2], Any[3]]))
        @test occursin("transdim.noise_exclusion_groups[2] has 1 member", e)
        e = _jm_verrs(_jm_cfg(groups = Any[Any[1, 9]]))
        @test occursin("transdim.noise_exclusion_groups[1][2] = 9", e)
        # references with no noise_models at all
        c = _jm_cfg(); delete!(c, "noise_models")
        @test occursin("there is no `noise_models` array", _jm_verrs(c))
        # labels: strings, unique
        c = _jm_cfg(); c["noise_models"][2]["label"] = "space_matern"
        @test occursin("label \"space_matern\" is used by noise_models[1] and [2]", _jm_verrs(c))
        c = _jm_cfg(); c["noise_models"][2]["label"] = 7
        @test occursin("noise_models[2].label must be a string", _jm_verrs(c))
        # noise_menu with these keys still refused
        c = _jm_cfg(); delete!(c, "noise_models"); c["noise_menu"] = Dict{String, Any}()
        @test occursin("cannot be set with `noise_menu`", _jm_verrs(c))
    end

    @testset "resolved models are the params' own objects" begin
        cfg = _jm_json(_jm_cfg(toggleable = Any["space_matern", 2, 3, "ground_sho"]))
        d, irv, ipm = Nereus._build_data(cfg["data"])
        p, tg, menu = Nereus._build_model(cfg, d, Nereus._build_star(cfg["star"]), irv, ipm)
        @test menu === nothing
        nms = p.config.noise_models
        @test length(nms) == 4
        @test nms[1] isa MaternGP && nms[2] isa CeleriteSHO
        @test all(m -> Nereus.noise_channel(m) === :phot, nms)
        td = Nereus._build_transdim(cfg["transdim"]; noise_models = nms,
                                    noise_models_cfg = cfg["noise_models"])
        @test td.noise
        @test length(td.toggleable) == 4 && all(td.toggleable[i] === nms[i] for i in 1:4)
        @test length(td.noise_exclusion_groups) == 2
        @test td.noise_exclusion_groups[1][1] === nms[1] && td.noise_exclusion_groups[1][2] === nms[2]
        @test td.noise_exclusion_groups[2][1] === nms[3] && td.noise_exclusion_groups[2][2] === nms[4]
        # integer references need the model list
        @test_throws ArgumentError Nereus._build_transdim(cfg["transdim"])
        # a menu block still refuses them
        @test_throws ErrorException Nereus._build_transdim(cfg["transdim"];
            menu = (noise_models = nms, toggleable = NoiseModel[], exclusion_groups = Vector{NoiseModel}[]))
    end

    @testset "a per-light-curve GP scores its own light curve, and only it" begin
        cfg = _jm_json(_jm_cfg())
        d, irv, ipm = Nereus._build_data(cfg["data"])
        p, _, _ = Nereus._build_model(cfg, d, Nereus._build_star(cfg["star"]), irv, ipm)
        nms = p.config.noise_models
        th = Nereus.Theta{Float64}(p)
        # every free parameter at its prior median, then the transit and GPs
        Nereus.set_unfrozen!(th, [Nereus.quantile(pr, 0.5) for pr in p.layout.unfrozen_priors])
        phic = Nereus.tc_to_phic(_JM_TC, _JM_P, p.config.parametrization.t_ref_c)
        vals = Dict("P_k1" => _JM_P, "phic_k1" => phic, "b_k1" => 0.3, "rr_k1" => 0.07,
                    "matern_sigma_phot_SPACE" => 1e-3, "matern_rho_phot_SPACE" => 0.3,
                    "matern_sigma_phot_GROUND" => 2e-3, "matern_rho_phot_GROUND" => 0.3,
                    "gp_log_S0_phot_SPACE" => -16.0, "gp_log_Q_phot_SPACE" => 0.0,
                    "gp_log_omega0_phot_SPACE" => 2.0, "gp_log_S0_phot_GROUND" => -15.0,
                    "gp_log_Q_phot_GROUND" => 0.0, "gp_log_omega0_phot_GROUND" => 2.0)
        for (k, v) in vals
            haskey(p.layout.name_to_idx, k) && Nereus.set_param!(th, k, v)
        end
        ws = Nereus.PTWorkspace(p, p.config.max_kplanet, length(nms);
                                n_obs = length(d.t_rv), n_phot = length(d.t_phot))
        function ll(on)
            th.td = Nereus.TransDimState(max_planets = 1, n_noise = length(nms))
            Nereus.activate_planet!(th.td, 1)
            foreach(j -> Nereus.activate_noise!(th.td, j), on)
            a = Nereus.transit_log_likelihood(th, d)
            b = Nereus.transit_log_likelihood(th, d, ws)
            @test a ≈ b rtol = 1e-10
            return a
        end
        white = ll(Int[])
        @test isfinite(white)
        # Each GP alone moves the likelihood; SPACE's and GROUND's add up, so
        # each touches its own light curve only.
        for (s, g) in ((1, 3), (1, 4), (2, 3), (2, 4))
            ds, dg = ll([s]) - white, ll([g]) - white
            @test abs(ds) > 1e-6 && abs(dg) > 1e-6
            @test ll([s, g]) - white ≈ ds + dg rtol = 1e-8
        end
    end

    @testset "co-activity that the likelihood cannot score is refused" begin
        # Two GPs on the same light curve, both toggleable, no group: both on
        # would score the light curve twice.
        cfg = _jm_json(_jm_cfg(groups = Any[Any[1, 2]]))
        d, irv, ipm = Nereus._build_data(cfg["data"])
        p, _, _ = Nereus._build_model(cfg, d, Nereus._build_star(cfg["star"]), irv, ipm)
        err = try
            Nereus._build_transdim(cfg["transdim"]; noise_models = p.config.noise_models,
                                   noise_models_cfg = cfg["noise_models"]); nothing
        catch e; e; end
        @test err isa ArgumentError
        @test occursin("GROUND", sprint(showerror, err)) &&
              occursin("noise_exclusion_groups", sprint(showerror, err))
        # NightlyOffset always on beside per-light-curve GPs on the same channel
        nm = [MaternGP(channel = :phot, instruments = ["A"]),
              CeleriteSHO(channel = :phot, instruments = ["A"]),
              NightlyOffset(channel = :phot)]
        @test_throws ArgumentError Nereus.validate_noise_coactivity(nm, [1, 2], [[1, 2]])
        @test Nereus.validate_noise_coactivity(nm[1:2], [1, 2], [[1, 2]]) === nothing
        # fixed-dimension: all on, so the same composition is refused up front
        @test_throws ArgumentError Nereus.validate_noise_models(NoiseModel[nm[1], nm[3]])
    end

    @testset "noise models that would do nothing on photometry are refused" begin
        # ErrorScale has no channel: it rescales RV errors only.
        c = _jm_cfg(); c["noise_models"] = Any[Dict{String, Any}("kind" => "ErrorScale",
                                                                 "channel" => "phot")]
        c["transdim"]["toggleable"] = Any[1]; delete!(c["transdim"], "noise_exclusion_groups")
        d, irv, ipm = Nereus._build_data(c["data"])
        err = try Nereus._build_model(c, d, Nereus._build_star(c["star"]), irv, ipm); nothing
              catch e; e; end
        @test err !== nothing && occursin("ErrorScale", sprint(showerror, err)) &&
              occursin("channel", sprint(showerror, err))
        # ...and naming a light curve as its instrument is no RV instrument
        c["noise_models"] = Any[Dict{String, Any}("kind" => "ErrorScale",
                                                  "instruments" => ["SPACE"])]
        err = try Nereus._build_model(c, d, Nereus._build_star(c["star"]), irv, ipm); nothing
              catch e; e; end
        @test err !== nothing && occursin("SPACE", sprint(showerror, err))
        # a phot GP naming an instrument the data does not have
        c["noise_models"] = Any[Dict{String, Any}("kind" => "MaternGP", "channel" => "phot",
                                                  "instruments" => ["TESS"])]
        err = try Nereus._build_model(c, d, Nereus._build_star(c["star"]), irv, ipm); nothing
              catch e; e; end
        @test err !== nothing && occursin("TESS", sprint(showerror, err))
    end

    @testset "transdim_pt_emcee over a per-light-curve menu" begin
        cfg = _jm_json(_jm_cfg())
        s = Nereus.run_job(cfg)
        @test s["status"] == "ok"
        ch, _ = Nereus.load_chains(joinpath(cfg["output_dir"], "chains.nc"))
        cn = names(ch, :parameters)
        cols = [Symbol("noise_active_$j") for j in 1:4]
        @test all(in(cn), cols)
        a = [vec(Array(ch[c])) .> 0.5 for c in cols]
        @test all(.!(a[1] .& a[2]))          # never two GPs on SPACE
        @test all(.!(a[3] .& a[4]))          # never two GPs on GROUND
        # the sampler moved through the menu, not stuck in its initial state
        @test length(unique(zip(a...))) > 1
        ms = s["model_selection"]["noise_models"]
        @test length(ms) == 4
        @test all(v -> 0.0 <= v["occupancy"] <= 1.0, values(ms))
        # per-instrument labels, not "GP-Matern" and "GP-Matern#3"
        @test Set(keys(ms)) == Set(["GP-Matern[SPACE]", "GP-SHO[SPACE]",
                                    "GP-Matern[GROUND]", "GP-SHO[GROUND]"])
    end
end
