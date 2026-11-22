# CCF profiles as Gaussian line components (src/ccf_data.jl,
# src/ccf_likelihood.jl): the container, the layout, the likelihood against a
# hand computation, each component type, gradients, recovery of injected
# blends, the edge cases, a joint fit that shares the orbit, checkpoints, and
# the evidence estimators. Every sampler through run_job is in
# test_ccf_runjob_samplers.jl; the NOI-101822 parity in
# test_ccf_parity_noi101822.jl.
using Test
using Nereus
using Random
using Statistics
using LinearAlgebra
using ForwardDiff
using JSON3
using LogDensityProblems
@isdefined(cs_ccfs) || include(joinpath(@__DIR__, "fixtures", "ccf_synth.jl"))

const _FWHM = 2sqrt(2log(2))

# ln L at the truth computed by hand from the fixture's model.
function _hand_ll(t, v, y, truth, jit)
    ll = 0.0
    for n in eachindex(t)
        m = fill(truth.cont[n], length(v[n]))
        for (_, pl, γ, K, fw, d) in truth.comps
            c = pl == 0 ? γ : γ + K * cs_kep(t[n])
            m .-= truth.scale[n] * d .* exp.(-((v[n] .- c) ./ (fw / _FWHM)) .^ 2 ./ 2)
        end
        ok = isfinite.(y[n])
        r = (y[n] .- m)[ok]
        ll -= sum(r .^ 2 ./ jit^2 .+ log(2π * jit^2)) / 2
    end
    return ll
end

# Bounded-space log-posterior Hessian → 1σ of each unfrozen parameter.
function _sigmas(target, x)
    f(z) = Nereus.logdensity_bounded(target, z)
    H = ForwardDiff.hessian(f, x)
    return sqrt.(diag(inv(Symmetric(-H))))
end

_mid(q) = isa(q.dist, Nereus.LogUniform) ? sqrt(q.lo * q.hi) : (q.lo + q.hi) / 2

# Starts that know nothing about the answer beyond what a look at the CCFs
# gives, on a grid, best optimum kept -- the way the reference analysis fits:
# continuum and depth scales 1, jitter 0.01, FWHM 10, depth 0.08; a static
# component at the line centre (γ = 10); a tied one at γ ∈ {5, 15} with
# K ∈ {−15, 15}. One start is not enough: from a single start LBFGS lands in a
# component-swapped or railed basin often enough to fail a test.
function _grid_start(p, γt, Kt)
    th = Nereus.Theta{Float64}(p); L = p.layout.name_to_idx
    for (nm, i) in L
        startswith(nm, "ccf_cont_") && (th.values[i] = 1.0)
        startswith(nm, "ccf_scale_") && (th.values[i] = 1.0)
        startswith(nm, "ccf_jit_") && (th.values[i] = 0.01)
        startswith(nm, "ccf_gamma_") && (th.values[i] = 10.0)
        startswith(nm, "ccf_K_") && (th.values[i] = Kt)
        startswith(nm, "ccf_fwhm_") && (th.values[i] = 10.0)
        startswith(nm, "ccf_depth_") && (th.values[i] = 0.08)
    end
    for (nm, i) in L
        startswith(nm, "ccf_K_") && (th.values[L[replace(nm, "ccf_K_" => "ccf_gamma_")]] = γt)
    end
    return th.values[p.layout.unfrozen_idx]
end

function _map(p, data; kw...)
    tg = NereusTarget(p, data)
    best = nothing
    for γt in (5.0, 15.0), Kt in (-15.0, 15.0)
        init = Nereus.transform_forward(_grid_start(p, γt, Kt), tg.transform)
        r = Nereus.sample_map(tg; init, n_starts = 1)
        (best === nothing || r.log_posterior > best.log_posterior) && (best = r)
    end
    return best, tg
end

@testset "CCF line components" begin

    @testset "container: validation and options" begin
        t, v, y, _ = cs_ccfs(n_epochs = 3)
        comps = cs_components()
        s = CCFSet(; t, v, ccf = y, components = comps)
        @test s.tag == "CCF" && s.labels == ["1", "2", "3"]
        @test all(n -> 0.9 < median(s.y[n][1:5]) < 1.1, 1:3)       # edge-normalised
        @test all(e -> all(==(0.0), e), s.err)
        @test_throws ArgumentError CCFComponent("")
        @test_throws ArgumentError CCFComponent("a_b")               # '_' splits names
        @test_throws ArgumentError CCFComponent("A"; planet = -1)
        @test_throws ArgumentError CCFSet(; t, v, ccf = y, components = CCFComponent[])
        @test_throws ArgumentError CCFSet(; t, v, ccf = y,
                                          components = [CCFComponent("A"), CCFComponent("A")])
        @test_throws ArgumentError CCFSet(; t, v = v[1:2], ccf = y, components = comps)
        @test_throws ArgumentError CCFSet(; t, v, ccf = y, components = comps, jitter = :x)
        @test_throws ArgumentError CCFSet(; t, v, ccf = y, components = comps, ref_epoch = 4)
        @test_throws ArgumentError CCFSet(; t, v, ccf = y, components = comps,
                                          labels = ["a", "a", "b"])
        @test_throws ArgumentError CCFSet(; t, v = [reverse(x) for x in v], ccf = y,
                                          components = comps)
        @test_throws ArgumentError Data(; ccf = [s, s])                # tags collide
        # CCF data alone is data; t_ref is the median epoch
        d = Data(; ccf = [s])
        @test d.t_ref ≈ median(t)
        # Edge-normalisation divides errors by the same factor
        e = [fill(0.01, length(x)) for x in v]
        s2 = CCFSet(; t, v, ccf = [2 .* x for x in y], err = e, components = comps)
        @test s2.err[1] ≈ e[1] ./ s2.norm[1]
        @test s2.norm[1] ≈ 2 * s.norm[1]
    end

    @testset "edge cases: NaN pixels and unequal grids" begin
        t, v, y, truth = cs_ccfs(n_epochs = 4, unequal = true, nan = true)
        s = CCFSet(; t, v, ccf = y, components = cs_components(), normalize_edges = 0)
        @test length(s.v[2]) == length(v[2]) - 3          # NaN pixels dropped
        @test all(isfinite, s.y[2])
        @test length(s.v[1]) != length(s.v[2])            # grids differ, both kept
        # JSON `null` is a masked pixel too
        @test Nereus._ccf_float_vec(Any[1.0, nothing, 2]) |> x -> isnan(x[2]) && x[3] == 2.0
        yy = deepcopy(y); yy[3] .= NaN
        @test_throws ArgumentError CCFSet(; t, v, ccf = yy, components = cs_components())
    end

    @testset "layout: names per component type and option" begin
        p, data, _ = cs_model()
        n = p.layout.names
        @test "ccf_gamma_SIM_A" in n && !("ccf_K_SIM_A" in n)         # static: no K
        @test all(x -> x in n, ("ccf_gamma_SIM_B", "ccf_K_SIM_B", "ccf_fwhm_SIM_B",
                                "ccf_depth_SIM_B"))
        @test !("ccf_scale_SIM_1" in n) && "ccf_scale_SIM_2" in n       # ref epoch
        @test count(startswith("ccf_jit_SIM_"), n) == 14
        @test count(startswith("ccf_cont_SIM_"), n) == 14
        @test n[p.layout.systemic.ccf[1].K[2]] == "ccf_K_SIM_B"
        @test p.layout.systemic.ccf[1].K[1] == 0
        @test issubset(ccf_param_names(data.ccf[1]), n)
        @test all(x -> x in Nereus.jitter_names(p), filter(startswith("ccf_jit_"), n))
        p2, _, _ = cs_model(; continuum = :shared, jitter = :shared, depth_scale = :none)
        n2 = p2.layout.names
        @test "ccf_cont_SIM" in n2 && "ccf_jit_SIM" in n2
        @test !any(startswith("ccf_scale_"), n2)
        @test length(unique(p2.layout.systemic.ccf[1].jit)) == 1
        p3, _, _ = cs_model(; ref_epoch = 5)
        @test "ccf_scale_SIM_1" in p3.layout.names && !("ccf_scale_SIM_5" in p3.layout.names)
        # A component on a planet the model does not have
        t, v, y, _ = cs_ccfs(n_epochs = 3)
        bad = Data(; ccf = [CCFSet(; t, v, ccf = y, components = [CCFComponent("X"; planet = 2)])])
        @test_throws ArgumentError Params(; max_kplanet = 1, planet_modes = [Nereus.RV_ONLY],
            data = bad, priors = cs_orbit_priors(), parametrization = ParametrizationConfig(time = :Tc))
        # default priors: K either sign, γ over the grid, depth ≤ 1 when normalised
        ps = p.config.priors
        @test ps["ccf_K_SIM_B"].lo < 0 < ps["ccf_K_SIM_B"].hi
        @test ps["ccf_gamma_SIM_A"].lo ≈ -40 && ps["ccf_gamma_SIM_A"].hi ≈ 60
        @test ps["ccf_jit_SIM_1"].lo > 0
    end

    @testset "likelihood equals the hand computation (static + tied)" begin
        for synth in ((;), (unequal = true, nan = true))
            p, data, truth = cs_model(; synth)
            th = cs_truth_theta(p, truth)
            t, v, y, _ = cs_ccfs(; synth...)
            # rtol 1e-9: the hand model solves Kepler to machine precision, Nereus to
            # its kepler_solve tol (1e-10 in E), a few 1e-9 km/s in velocity
            @test ccf_log_likelihood(th, data) ≈ _hand_ll(t, v, y, truth, truth.noise) rtol = 1e-9
        end
    end

    @testset "component types" begin
        # static only: velocities constant, independent of the orbit
        st = [("S", 0, 5.0, 0.0, 14.0, 0.1)]
        p, data, truth = cs_model(; comps = st)
        th = cs_truth_theta(p, truth)
        @test all(==(5.0), ccf_component_velocities(th, data)[1])
        ll0 = ccf_log_likelihood(th, data)
        th.values[p.layout.name_to_idx["P_k1"]] = 3.0
        @test ccf_log_likelihood(th, data) == ll0
        # tied only: velocity = γ + K g(t) with g the RV model's unit Keplerian
        tied = [("T", 1, -3.0, 25.0, 14.0, 0.1)]
        p, data, truth = cs_model(; comps = tied)
        th = cs_truth_theta(p, truth)
        vel = ccf_component_velocities(th, data)[1][:, 1]
        # atol 1e-7 km/s: kepler_solve's tol (1e-10 in E) against a converged Newton
        @test vel ≈ -3.0 .+ 25.0 .* cs_kep.(data.ccf[1].t) atol = 1e-7
        # and agrees with the RV model's own sign: rv_keplerian with K = 1
        o = Nereus._ccf_orbit(th, data, 1)
        @test vel ≈ -3.0 .+ 25.0 .* [Nereus.rv_keplerian(x, o[1], 1.0, o[2], o[3], 0.0, o[4])
                                    for x in data.ccf[1].t]
        # K < 0 mirrors the curve about γ
        th.values[p.layout.name_to_idx["ccf_K_SIM_T"]] = -25.0
        @test ccf_component_velocities(th, data)[1][:, 1] ≈ -3.0 .- 25.0 .* cs_kep.(data.ccf[1].t) atol = 1e-7
        # a tied component follows the orbit: moving P changes ln L
        ll0 = ccf_log_likelihood(th, data)
        th.values[p.layout.name_to_idx["P_k1"]] = 4.3
        @test ccf_log_likelihood(th, data) != ll0
        # unphysical width or depth scale: -Inf
        th = cs_truth_theta(p, truth)
        th.values[p.layout.name_to_idx["ccf_fwhm_SIM_T"]] = 0.0
        @test ccf_log_likelihood(th, data) == -Inf
    end

    @testset "model and residuals" begin
        p, data, truth = cs_model()
        th = cs_truth_theta(p, truth)
        m = ccf_model(th, data)[1]
        @test m.tag == "SIM" && length(m.model) == 14
        @test std(reduce(vcat, m.residual)) ≈ truth.noise rtol = 0.1
        @test all(s -> all(≈(truth.noise), s), m.sigma)
    end

    @testset "posterior: CCF term in every path; gradients" begin
        p, data, truth = cs_model(; jitter = :shared)
        th = cs_truth_theta(p, truth)
        x = th.values[p.layout.unfrozen_idx]
        tb = NereusTarget(p, data; unconstrained = false)
        tu = NereusTarget(p, data)
        lp = Nereus.log_prior(th)
        @test LogDensityProblems.logdensity(tb, x) ≈ lp + ccf_log_likelihood(th, data) +
              Nereus._rv_log_likelihood_core(th, data)
        @test Nereus.logdensity_bounded(tu, x) ≈ LogDensityProblems.logdensity(tb, x)
        pj, ll = Nereus._logdensity_parts(tb, x)
        @test ll ≈ ccf_log_likelihood(th, data) + Nereus._rv_log_likelihood_core(th, data)
        # ForwardDiff through the CCF term matches central differences
        f(z) = LogDensityProblems.logdensity(tb, z)
        g = ForwardDiff.gradient(f, x)
        h = 1e-6
        for i in (1, 2, 3, 5, length(x))
            e = zeros(length(x)); e[i] = h
            @test g[i] ≈ (f(x .+ e) - f(x .- e)) / 2h rtol = 1e-4
        end
        # Enzyme would drop the CCF term: refused
        @test_throws ArgumentError Nereus.EnzymeGradientConfig(tu)
    end

    @testset "recovery of an injected blend (static A + tied B)" begin
        p, data, truth = cs_model(; jitter = :shared,
                                  synth = (noise = 0.004, seed = 3, n_epochs = 8))
        r, tg = _map(p, data)
        @test r.converged
        σ = _sigmas(NereusTarget(p, data; unconstrained = false), r.x_map)
        get_(nm) = (i = findfirst(==(nm), r.param_names); (r.x_map[i], σ[i]))
        for (nm, _, γ, K, fw, d) in truth.comps
            for (par, tv) in (("gamma", γ), ("fwhm", fw), ("depth", d))
                est, s = get_("ccf_$(par)_SIM_$nm")
                @test abs(est - tv) < 4s
            end
            if nm == "B"
                est, s = get_("ccf_K_SIM_B")
                @test abs(est - K) < 4s
                @test s < 0.5                        # K_B is measured, not prior-bound
            end
        end
        @test abs(get_("ccf_jit_SIM")[1] - truth.noise) < 4get_("ccf_jit_SIM")[2]
    end

    @testset "N = 3 components and one component" begin
        three = [("A", 0, 10.0, 0.0, 12.0, 0.15), ("B", 1, 8.0, 20.0, 18.0, 0.06),
                 ("C", 1, 12.0, -30.0, 8.0, 0.04)]
        p, data, truth = cs_model(; comps = three, jitter = :shared, depth_scale = :none,
                                  synth = (scale_sd = 0.0, seed = 4))
        th = cs_truth_theta(p, truth)
        t, v, y, _ = cs_ccfs(; comps = three, scale_sd = 0.0, seed = 4)
        @test ccf_log_likelihood(th, data) ≈ _hand_ll(t, v, y, truth, truth.noise) rtol = 1e-9
        # MAP from the truth stays at the truth (each component identified)
        tb = NereusTarget(p, data)
        x0 = th.values[p.layout.unfrozen_idx]
        r = Nereus.sample_map(tb; init = Nereus.transform_forward(x0, tb.transform),
                              n_starts = 1)
        σ = _sigmas(NereusTarget(p, data; unconstrained = false), r.x_map)
        iK = findfirst(==("ccf_K_SIM_C"), r.param_names)
        @test abs(r.x_map[iK] + 30.0) < 4σ[iK]
        # one component: the blend is a worse fit than two by far
        p1, d1, _ = cs_model(; fit = [CS_BLEND[2]], jitter = :shared, continuum = :shared,
                             depth_scale = :none, synth = (seed = 3, n_epochs = 8,
                             scale_sd = 0.0, cont_sd = 0.0))
        @test length(p1.layout.systemic.ccf[1].gamma) == 1
        r1, _ = _map(p1, d1)
        p2, d2, _ = cs_model(; jitter = :shared, continuum = :shared, depth_scale = :none,
                             synth = (seed = 3, n_epochs = 8, scale_sd = 0.0, cont_sd = 0.0))
        r2, _ = _map(p2, d2)
        @test r2.log_posterior - r1.log_posterior > 50
    end

    @testset "run_job config: inline values and CCF dump files" begin
        dir = mktempdir()
        t, v, y, _ = cs_ccfs(n_epochs = 6)
        eps = Any[]
        for n in 1:6
            f = joinpath(dir, "ccf_$n.json")
            # the database dump form: arrays as JSON strings, an id
            write(f, JSON3.write(Dict("id" => 100 + n, "velocities" => JSON3.write(v[n]),
                                      "ccf_values" => JSON3.write(y[n]))))
            push!(eps, Dict("bjd" => t[n], "file" => f))
        end
        blk = Dict{String, Any}("tag" => "SIM", "epochs" => eps, "ref_epoch" => "103",
                                "components" => [Dict("name" => "A"),
                                                 Dict("name" => "B", "planet" => 1)])
        sets = Nereus._parse_ccf_blocks(blk)
        @test sets[1].labels == string.(101:106) && sets[1].ref_epoch == 3
        id, vv, yy = load_ccf_json(eps[1]["file"])
        @test id == 101 && vv == v[1] && yy == y[1]
        # The whole config survives a JSON round trip unchanged in meaning
        inline = cs_ccf_block(synth = (n_epochs = 6,))
        cfg = Dict{String, Any}("data" => Dict("ccf" => [inline]))
        rt = JSON3.read(JSON3.write(cfg), Dict{String, Any})
        a = Nereus._parse_ccf_blocks(cfg["data"]["ccf"])[1]
        b = Nereus._parse_ccf_blocks(rt["data"]["ccf"])[1]
        @test a.y == b.y && a.t == b.t && [c.planet for c in b.components] == [0, 1]
        @test_throws ArgumentError Nereus._parse_ccf_blocks(Dict("epochs" => eps))
        bad = deepcopy(blk); bad["ref_epoch"] = "nope"
        @test_throws ArgumentError Nereus._parse_ccf_blocks(bad)
    end

    @testset "run_job: CCF-only fit, summary and chains" begin
        out = mktempdir()
        cfg = Dict{String, Any}("output_dir" => out, "seed" => 3,
            "data" => Dict{String, Any}("ccf" => [cs_ccf_block(
                synth = (n_epochs = 8,), extra = Dict{String, Any}("jitter" => "shared"))]),
            "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"]),
            "priors" => cs_prior_cfg(),
            "sampler" => Dict{String, Any}("name" => "pt_emcee", "kwargs" => Dict(
                "n_temps" => 3, "n_walkers" => 64, "n_steps" => 60, "n_burnin" => 40,
                "show_progress" => false)),
            "output" => Dict{String, Any}("plots" => String[], "ppc" => false,
                                          "detection_limits" => false))
        s = Nereus.run_job(cfg)
        @test s["status"] == "ok"
        @test isfile(joinpath(out, "chains.nc"))
        @test isfile(joinpath(out, "summary.json"))
    end

    @testset "joint CCF + RV + transit fit shares the orbit" begin
        rng = MersenneTwister(11)
        # RVs and a transit of the same planet
        trv = sort!(CS_TC .+ 60 .* rand(rng, 25))
        Kp = 50.0
        rv = Kp .* cs_kep.(trv) .+ 3.0 .* randn(rng, 25)
        tph = collect(range(CS_TC - 0.2, CS_TC + 0.2; length = 300))
        cfg_data = Dict{String, Any}(
            "rv" => Dict("values" => Dict("bjd" => trv, "rv" => rv, "rv_err" => fill(3.0, 25),
                                          "instrument" => fill("SIM", 25))),
            "transit_photometry" => [Dict("instrument" => "PHOT", "values" => Dict(
                "bjd" => tph, "flux" => ones(300) .+ 1e-3 .* randn(rng, 300),
                "flux_err" => fill(1e-3, 300)))],
            "ccf" => [cs_ccf_block(synth = (n_epochs = 8,),
                                   extra = Dict{String, Any}("jitter" => "shared"))])
        cfg = Dict{String, Any}("data" => cfg_data,
            "star" => Dict("M_s" => 1.0, "R_s" => 1.0),
            "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["RVPM"],
                                         "parametrization" => Dict("time" => "Tc")))
        data, rvn, pmn, asn, iadn, _ = Nereus._build_data(cfg["data"])
        @test length(data.ccf) == 1 && Nereus.n_rv(data) == 25 && Nereus.n_phot(data) == 300
        p, tg, _ = Nereus._build_model(cfg, data, Nereus._build_star(cfg["star"]), rvn, pmn)
        n = p.layout.names
        @test "P_k1" in n && "K_k1" in n && "ccf_K_SIM_B" in n && "b_k1" in n
        # one P slot read by all three terms
        th = Nereus.Theta{Float64}(p)
        th.values[p.layout.unfrozen_idx] .= [_mid(q) for q in p.layout.unfrozen_priors]
        L = p.layout.name_to_idx
        th.values[L["P_k1"]] = CS_P; th.values[L["Tc_k1"]] = CS_TC
        th.values[L["sesinw_k1"]] = CS_SESINW; th.values[L["secosw_k1"]] = CS_SECOSW
        th.values[L["K_k1"]] = Kp
        ref = cs_truth_theta(p, (; cs_ccfs(n_epochs = 8)[4]..., noise = 0.004))
        for nm in n
            startswith(nm, "ccf_") && (th.values[L[nm]] = ref.values[L[nm]])
        end
        lc0 = ccf_log_likelihood(th, data)
        lr0 = Nereus._rv_log_likelihood_core(th, data)
        lt0 = Nereus.transit_log_likelihood(th, data)
        @test Nereus.rv_log_likelihood(th, data) ≈ lr0 + lc0
        th.values[L["P_k1"]] = CS_P + 0.01
        @test ccf_log_likelihood(th, data) < lc0          # CCF sees the shared P
        @test Nereus._rv_log_likelihood_core(th, data) != lr0
        @test Nereus.transit_log_likelihood(th, data) != lt0
        # the joint target's posterior is the sum of the pieces
        th.values[L["P_k1"]] = CS_P
        x = th.values[p.layout.unfrozen_idx]
        @test Nereus.logdensity_bounded(tg, x) ≈ Nereus.log_prior(th) + lr0 + lc0 + lt0
    end

    @testset "checkpoint and resume on a CCF target" begin
        p, data, _ = cs_model(; jitter = :shared, continuum = :shared, depth_scale = :none,
                              synth = (n_epochs = 6, scale_sd = 0.0, cont_sd = 0.0))
        kw = (n_temps = 3, n_walkers = 32, n_burnin = 30, seed = 9, show_progress = false,
              bridge_headline = false)
        tgf() = NereusTarget(p, data; unconstrained = false)
        full = sample_pt_emcee(tgf(), data; kw..., n_steps = 80)
        path = joinpath(mktempdir(), "ccf_state.jls")
        sample_pt_emcee(tgf(), data; kw..., n_steps = 40, checkpoint = path)
        @test isfile(path)
        res = sample_pt_emcee(tgf(), data; kw..., n_steps = 80, checkpoint = path,
                              resume = true)
        @test res.chains.value.data == full.chains.value.data
        # a different CCF set is a different run: refused
        p2, d2, _ = cs_model(; jitter = :shared, continuum = :shared, depth_scale = :none,
                             synth = (n_epochs = 6, scale_sd = 0.0, cont_sd = 0.0, seed = 99))
        @test_throws Exception sample_pt_emcee(NereusTarget(p2, d2; unconstrained = false), d2;
                                               kw..., n_steps = 80, checkpoint = path,
                                               resume = true)
        # data without CCFs hashes as before the field existed
        dn = Data(; t_rv = [1.0, 2.0], rv = [0.0, 1.0], rv_err = [1.0, 1.0])
        @test Nereus._hash_skip(dn, :ccf) && !Nereus._hash_skip(data, :ccf)
    end

    @testset "evidence: nested and bridge on CCF targets, 2 vs 1 components" begin
        common = (jitter = :shared, continuum = :shared, depth_scale = :none)
        synth = (n_epochs = 6, scale_sd = 0.0, cont_sd = 0.0, seed = 5)
        lz = Dict{Int, Float64}()
        for (nc, comps) in ((1, [CS_BLEND[2]]), (2, CS_BLEND))
            pri = Dict{String, Nereus.PriorSpec}()
            for c in comps
                pri["ccf_gamma_SIM_$(c[1])"] = UniformPrior(-10.0, 30.0)
                pri["ccf_fwhm_SIM_$(c[1])"] = UniformPrior(4.0, 40.0)
                pri["ccf_depth_SIM_$(c[1])"] = UniformPrior(0.0, 0.5)
                c[2] == 1 && (pri["ccf_K_SIM_$(c[1])"] = UniformPrior(-40.0, 40.0))
            end
            pri["ccf_cont_SIM"] = UniformPrior(0.95, 1.05)
            pri["ccf_jit_SIM"] = LogUniformPrior(1e-4, 0.1)
            p, data, _ = cs_model(; fit = comps, priors = pri, synth, common...)
            ch, lzn, _ = sample_nested(NereusTarget(p, data; unconstrained = false), data;
                                       n_live = 150, dlogz = 0.5, seed = 1,
                                       show_progress = false)
            @test isfinite(lzn)
            lz[nc] = lzn
            if nc == 2
                b = bridge_evidence(NereusTarget(p, data; unconstrained = false), ch)
                lb = b.log_z
                @test isfinite(lb)
                @test abs(lb - lzn) < 5          # two estimators agree
            end
        end
        @test lz[2] - lz[1] > 10                            # the blend is detected
    end
end
