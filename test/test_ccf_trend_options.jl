# CCF line components: a time trend on a component's γ (`CCFComponent(...;
# trend)`, `trend_t_ref`), one width for every component (`shared_fwhm`) and
# the N_eff weighting of correlated pixels (`pixel_correlation = :neff`). The
# defaults leave the 0.9.6 model as it was; test_ccf_components.jl and the
# NOI-101822 parity cover that model itself.
using Test
using Nereus
using Random
using Statistics
using LinearAlgebra
using ForwardDiff
using JSON3
using NCDatasets
using LogDensityProblems
@isdefined(cs_ccfs) || include(joinpath(@__DIR__, "fixtures", "ccf_synth.jl"))

const _TFW = 2sqrt(2log(2))
const _TSIMPLE = (jitter = :shared, continuum = :shared, depth_scale = :none)
const _TSYNTH = (scale_sd = 0.0, cont_sd = 0.0)

# Per-epoch ln L at the truth by hand, trends included.
function _hand_ll_epochs(t, v, y, truth, jit)
    out = Float64[]
    for n in eachindex(t)
        m = fill(truth.cont[n], length(v[n]))
        for (nm, pl, γ, K, fw, d) in truth.comps
            c = pl == 0 ? γ : γ + K * cs_kep(t[n])
            if haskey(truth.trend, nm)
                dt = t[n] - truth.t_ref
                c += truth.trend[nm][1] * dt + truth.trend[nm][2] * dt^2 / 2
            end
            m .-= truth.scale[n] * d .* exp.(-((v[n] .- c) ./ (fw / _TFW)) .^ 2 ./ 2)
        end
        ok = isfinite.(y[n])
        r = (y[n] .- m)[ok]
        push!(out, -sum(r .^ 2 ./ jit^2 .+ log(2π * jit^2)) / 2)
    end
    return out
end

# The truth Theta of the fixture, with each component's trend terms set.
function _trend_theta(p, truth)
    th = cs_truth_theta(p, truth)
    L = p.layout.name_to_idx
    for (nm, (g1, g2)) in truth.trend
        haskey(L, "ccf_gdot_SIM_$nm") && (th.values[L["ccf_gdot_SIM_$nm"]] = g1)
        haskey(L, "ccf_gddot_SIM_$nm") && (th.values[L["ccf_gddot_SIM_$nm"]] = g2)
    end
    return th
end

function _tsigmas(p, data, x)
    tb = NereusTarget(p, data; unconstrained = false)
    H = ForwardDiff.hessian(z -> LogDensityProblems.logdensity(tb, z), x)
    return sqrt.(diag(inv(Symmetric(-H))))
end

# MAP from `th` (unconstrained target); returns (result, 1σ per unfrozen param).
function _tmap(p, data, th)
    tg = NereusTarget(p, data)
    x0 = th.values[p.layout.unfrozen_idx]
    r = Nereus.sample_map(tg; init = Nereus.transform_forward(x0, tg.transform), n_starts = 1)
    return r, _tsigmas(p, data, r.x_map)
end

@testset "CCF trends, shared widths, N_eff" begin

    @testset "defaults: the 0.9.6 layout, unit weights, hash unchanged" begin
        p, data, _ = cs_model(; _TSIMPLE...)
        s = data.ccf[1]
        @test all(c -> c.trend == 0, s.components)
        @test !s.shared_fwhm && s.pixel_correlation === :independent
        @test isnan(s.resolution_kms) && s.trend_t_ref ≈ mean(s.t)
        @test ccf_param_names(s) == ["ccf_gamma_SIM_A", "ccf_fwhm_SIM_A", "ccf_depth_SIM_A",
                                     "ccf_gamma_SIM_B", "ccf_K_SIM_B", "ccf_fwhm_SIM_B",
                                     "ccf_depth_SIM_B", "ccf_cont_SIM", "ccf_jit_SIM"]
        ix = p.layout.systemic.ccf[1]
        @test all(==(0), ix.gdot) && all(==(0), ix.gddot)
        @test all(==(1.0), ix.weight) && isnan(ix.resolution)
        @test length(unique(ix.fwhm)) == 2
        # the new fields stay out of the checkpoint hash at their defaults
        for f in (:trend_t_ref, :shared_fwhm, :pixel_correlation, :resolution_kms)
            @test Nereus._hash_skip(s, f)
        end
        @test Nereus._hash_skip(s.components[1], :trend)
        # and enter it once they change the model
        t, v, y, _ = cs_ccfs(n_epochs = 4)
        s2 = CCFSet(; t, v, ccf = y, shared_fwhm = true, pixel_correlation = :neff,
                    resolution_kms = 5.0,
                    components = [CCFComponent("A"; trend = 1), CCFComponent("B"; planet = 1)])
        for f in (:trend_t_ref, :shared_fwhm, :pixel_correlation, :resolution_kms)
            @test !Nereus._hash_skip(s2, f)
        end
        @test !Nereus._hash_skip(s2.components[1], :trend)
        @test Nereus._hash_skip(s2.components[2], :trend)
    end

    @testset "trend: validation" begin
        @test_throws ArgumentError CCFComponent("A"; trend = 3)
        @test_throws ArgumentError CCFComponent("A"; trend = -1)
        @test CCFComponent("A"; planet = 1, trend = 2).trend == 2
        mk(n, tr; kw...) = (d = cs_ccfs(n_epochs = n);
            CCFSet(; t = d[1], v = d[2], ccf = d[3],
                   components = [CCFComponent("A"; trend = tr)], kw...))
        @test_throws ArgumentError mk(2, 1)            # trend + 2 = 3 epochs needed
        @test mk(3, 1).components[1].trend == 1
        @test_throws ArgumentError mk(3, 2)
        @test mk(4, 2).components[1].trend == 2
        @test mk(2, 0).components[1].trend == 0        # no trend: any number of epochs
        t, v, y, _ = cs_ccfs(n_epochs = 3)
        @test_throws ArgumentError CCFSet(; t = fill(t[1], 3), v, ccf = y,
                                          components = [CCFComponent("A"; trend = 1)])
        @test_throws ArgumentError mk(3, 1; trend_t_ref = NaN)
        @test mk(3, 1; trend_t_ref = 2459510.0).trend_t_ref == 2459510.0
        @test mk(3, 1).trend_t_ref ≈ mean(cs_ccfs(n_epochs = 3)[1])
    end

    @testset "trend: names, default priors, velocities, likelihood" begin
        tr = Dict("A" => (0.05, 0.0), "B" => (-0.03, 0.004))
        comps_fit = [CCFComponent("A"; trend = 1), CCFComponent("B"; planet = 1, trend = 2)]
        t, v, y, truth = cs_ccfs(; n_epochs = 10, trend = tr, _TSYNTH...)
        s = CCFSet(; tag = "SIM", t, v, ccf = y, components = comps_fit, normalize_edges = 0,
                   _TSIMPLE...)
        @test ccf_param_names(s)[1:8] == ["ccf_gamma_SIM_A", "ccf_gdot_SIM_A", "ccf_fwhm_SIM_A",
                                          "ccf_depth_SIM_A", "ccf_gamma_SIM_B", "ccf_gdot_SIM_B",
                                          "ccf_gddot_SIM_B", "ccf_K_SIM_B"]
        data = Data(; ccf = [s])
        p = Params(; max_kplanet = 1, planet_modes = [Nereus.RV_ONLY], data,
                   priors = cs_orbit_priors(), parametrization = ParametrizationConfig(time = :Tc))
        @test issubset(ccf_param_names(s), p.layout.names)
        # priors: ±span/baseline, ±span/baseline²
        span = 100.0; B = maximum(t) - minimum(t)
        pr = p.config.priors
        @test pr["ccf_gdot_SIM_A"].lo ≈ -span / B && pr["ccf_gdot_SIM_A"].hi ≈ span / B
        @test pr["ccf_gddot_SIM_B"].lo ≈ -span / B^2 && pr["ccf_gddot_SIM_B"].hi ≈ span / B^2
        @test !haskey(pr, "ccf_gddot_SIM_A")
        th = _trend_theta(p, truth)
        vel = ccf_component_velocities(th, data)[1]
        dt = t .- mean(t)
        @test vel[:, 1] ≈ 10.0 .+ 0.05 .* dt
        @test vel[:, 2] ≈ 8.0 .- 0.03 .* dt .+ 0.002 .* dt .^ 2 .+ 20.0 .* cs_kep.(t) atol = 1e-7
        @test ccf_model(th, data)[1].velocities == vel
        @test ccf_log_likelihood(th, data) ≈ sum(_hand_ll_epochs(t, v, y, truth, truth.noise)) rtol = 1e-9
        # a trend term at zero is the model without it
        d0 = Data(; ccf = [CCFSet(; tag = "SIM", t, v, ccf = y, normalize_edges = 0,
                                  components = cs_components(), _TSIMPLE...)])
        p0 = Params(; max_kplanet = 1, planet_modes = [Nereus.RV_ONLY], data = d0,
                    priors = cs_orbit_priors(), parametrization = ParametrizationConfig(time = :Tc))
        th0 = cs_truth_theta(p0, truth)
        th.values[p.layout.name_to_idx["ccf_gdot_SIM_A"]] = 0.0
        th.values[p.layout.name_to_idx["ccf_gdot_SIM_B"]] = 0.0
        th.values[p.layout.name_to_idx["ccf_gddot_SIM_B"]] = 0.0
        @test ccf_log_likelihood(th, data) == ccf_log_likelihood(th0, d0)
        # ForwardDiff through the trend terms matches central differences
        th = _trend_theta(p, truth)
        x = th.values[p.layout.unfrozen_idx]
        tb = NereusTarget(p, data; unconstrained = false)
        f(z) = LogDensityProblems.logdensity(tb, z)
        g = ForwardDiff.gradient(f, x)
        for nm in ("ccf_gdot_SIM_A", "ccf_gdot_SIM_B", "ccf_gddot_SIM_B")
            i = findfirst(==(nm), p.layout.unfrozen_names)
            h = 1e-7; e = zeros(length(x)); e[i] = h
            @test g[i] ≈ (f(x .+ e) - f(x .- e)) / 2h rtol = 1e-4
        end
    end

    @testset "trend: recovery of a known drift" begin
        tr = Dict("A" => (0.05, 0.0), "B" => (-0.03, 0.004))
        synth = (n_epochs = 10, seed = 3, trend = tr, _TSYNTH...)
        fit = [CCFComponent("A"; trend = 1), CCFComponent("B"; planet = 1, trend = 2)]
        t, v, y, truth = cs_ccfs(; synth...)
        s = CCFSet(; tag = "SIM", t, v, ccf = y, components = fit, normalize_edges = 0,
                   _TSIMPLE...)
        data = Data(; ccf = [s])
        p = Params(; max_kplanet = 1, planet_modes = [Nereus.RV_ONLY], data,
                   priors = cs_orbit_priors(), parametrization = ParametrizationConfig(time = :Tc))
        # start at the truth with no drift at all: the fit has to find it
        th = cs_truth_theta(p, truth)
        r, σ = _tmap(p, data, th)
        @test r.converged
        est(nm) = (i = findfirst(==(nm), r.param_names); (r.x_map[i], σ[i]))
        for (nm, tv) in (("ccf_gdot_SIM_A", 0.05), ("ccf_gdot_SIM_B", -0.03),
                         ("ccf_gddot_SIM_B", 0.004), ("ccf_gamma_SIM_A", 10.0),
                         ("ccf_gamma_SIM_B", 8.0), ("ccf_K_SIM_B", 20.0))
            e, sg = est(nm)
            @test abs(e - tv) < 4sg
        end
        @test est("ccf_gdot_SIM_A")[2] < 0.005           # measured, not prior-bound
        # the same CCFs without trends fit far worse
        p0, d0, _ = cs_model(; synth, _TSIMPLE...)
        r0, _ = _tmap(p0, d0, cs_truth_theta(p0, truth))
        @test r.log_posterior - r0.log_posterior > 50
        # and data without a drift give γ̇ consistent with zero
        t2, v2, y2, truth2 = cs_ccfs(; n_epochs = 10, seed = 3, _TSYNTH...)
        s2 = CCFSet(; tag = "SIM", t = t2, v = v2, ccf = y2, normalize_edges = 0,
                    components = [CCFComponent("A"; trend = 1), CCFComponent("B"; planet = 1)],
                    _TSIMPLE...)
        d2 = Data(; ccf = [s2])
        p2 = Params(; max_kplanet = 1, planet_modes = [Nereus.RV_ONLY], data = d2,
                    priors = cs_orbit_priors(), parametrization = ParametrizationConfig(time = :Tc))
        r2, σ2 = _tmap(p2, d2, cs_truth_theta(p2, truth2))
        i = findfirst(==("ccf_gdot_SIM_A"), r2.param_names)
        @test abs(r2.x_map[i]) < 4σ2[i]
    end

    @testset "shared_fwhm: one width for every component" begin
        p, data, truth = cs_model(; shared_fwhm = true, _TSIMPLE..., synth = (n_epochs = 6,))
        n = p.layout.names
        @test "ccf_fwhm_SIM" in n && !any(x -> startswith(x, "ccf_fwhm_SIM_"), n)
        ix = p.layout.systemic.ccf[1]
        @test ix.fwhm[1] == ix.fwhm[2] == p.layout.name_to_idx["ccf_fwhm_SIM"]
        @test ccf_param_names(data.ccf[1]) ==
              ["ccf_gamma_SIM_A", "ccf_depth_SIM_A", "ccf_gamma_SIM_B", "ccf_K_SIM_B",
               "ccf_depth_SIM_B", "ccf_fwhm_SIM", "ccf_cont_SIM", "ccf_jit_SIM"]
        @test p.config.priors["ccf_fwhm_SIM"].lo ≈ 1.0      # 2 grid steps, as per component
        # the same likelihood as separate widths set equal
        pu, du, _ = cs_model(; _TSIMPLE..., synth = (n_epochs = 6,))
        for w in (9.0, 14.0)
            th = cs_truth_theta(p, truth); th.values[p.layout.name_to_idx["ccf_fwhm_SIM"]] = w
            thu = cs_truth_theta(pu, truth)
            thu.values[pu.layout.name_to_idx["ccf_fwhm_SIM_A"]] = w
            thu.values[pu.layout.name_to_idx["ccf_fwhm_SIM_B"]] = w
            @test ccf_log_likelihood(th, data) == ccf_log_likelihood(thu, du)
            @test ccf_model(th, data)[1].model == ccf_model(thu, du)[1].model
        end
        # recovery when the lines share one width
        eq = [("A", 0, 10.0, 0.0, 14.0, 0.15), ("B", 1, 8.0, 20.0, 14.0, 0.06)]
        ps, ds, ts = cs_model(; comps = eq, shared_fwhm = true, _TSIMPLE...,
                              synth = (n_epochs = 8, seed = 3, _TSYNTH...))
        th = cs_truth_theta(ps, ts); th.values[ps.layout.name_to_idx["ccf_fwhm_SIM"]] = 12.0
        r, σ = _tmap(ps, ds, th)
        i = findfirst(==("ccf_fwhm_SIM"), r.param_names)
        @test abs(r.x_map[i] - 14.0) < 4σ[i]
    end

    @testset "pixel_correlation = :neff" begin
        t, v, y, truth = cs_ccfs(; n_epochs = 6, unequal = true, _TSYNTH...)
        # validation
        @test_throws ArgumentError CCFSet(; t, v, ccf = y, components = cs_components(),
                                          pixel_correlation = :banded)
        @test_throws ArgumentError CCFSet(; t, v, ccf = y, components = cs_components(),
                                          resolution_kms = 5.0)        # only with :neff
        @test_throws ArgumentError CCFSet(; t, v, ccf = y, components = cs_components(),
                                          pixel_correlation = :neff, resolution_kms = 0.0)
        s = CCFSet(; tag = "SIM", t, v, ccf = y, components = cs_components(),
                   normalize_edges = 0, pixel_correlation = :neff, resolution_kms = 6.0,
                   _TSIMPLE...)
        # the factor: (velocity span / resolution) / N_pix, ≤ 1
        w = ccf_epoch_weights(s, 6.0)
        @test w ≈ [((x[end] - x[1]) / 6.0) / length(x) for x in s.v]
        @test w[1] ≈ (100.0 / 6.0) / 201                     # the 0.5 km/s grid
        @test w[2] ≈ (99.2 / 6.0) / 125                      # the 0.8 km/s grid
        @test all(==(1.0), ccf_epoch_weights(s, 0.1))        # finer than a pixel: capped
        @test_throws ArgumentError ccf_epoch_weights(s, -1.0)
        data = Data(; ccf = [s])
        mkp(d; priors = Dict{String, Nereus.PriorSpec}()) =
            Params(; max_kplanet = 1, planet_modes = [Nereus.RV_ONLY], data = d,
                   priors = merge(cs_orbit_priors(), priors),
                   parametrization = ParametrizationConfig(time = :Tc))
        p = mkp(data)
        ix = p.layout.systemic.ccf[1]
        @test ix.resolution == 6.0 && ix.weight ≈ w
        # ln L = Σ_n w_n ln L_n
        th = cs_truth_theta(p, truth)
        lln = _hand_ll_epochs(t, v, y, truth, truth.noise)
        @test ccf_log_likelihood(th, data) ≈ sum(w .* lln) rtol = 1e-9
        # the resolution element from the narrowest FWHM prior when not given
        sd = CCFSet(; tag = "SIM", t, v, ccf = y, components = cs_components(),
                    normalize_edges = 0, pixel_correlation = :neff, _TSIMPLE...)
        dd = Data(; ccf = [sd])
        pd = mkp(dd; priors = Dict{String, Nereus.PriorSpec}(
            "ccf_fwhm_SIM_A" => UniformPrior(7.0, 40.0), "ccf_fwhm_SIM_B" => UniformPrior(5.0, 40.0)))
        @test pd.layout.systemic.ccf[1].resolution == 5.0
        @test pd.layout.systemic.ccf[1].weight ≈ ccf_epoch_weights(sd, 5.0)
        # default FWHM priors: 2 grid steps of the finest grid
        @test mkp(dd).layout.systemic.ccf[1].resolution ≈ 1.0
        # a fixed width is its own lower bound; with shared_fwhm the one prior
        sh = CCFSet(; tag = "SIM", t, v, ccf = y, components = cs_components(),
                    normalize_edges = 0, pixel_correlation = :neff, shared_fwhm = true,
                    _TSIMPLE...)
        psh = mkp(Data(; ccf = [sh]); priors = Dict{String, Nereus.PriorSpec}(
            "ccf_fwhm_SIM" => FixedPrior(13.0)))
        @test psh.layout.systemic.ccf[1].resolution == 13.0
        # a prior not bounded away from zero cannot give one
        @test_throws ArgumentError mkp(dd; priors = Dict{String, Nereus.PriorSpec}(
            "ccf_fwhm_SIM_A" => UniformPrior(0.0, 40.0)))
    end

    @testset "run_job config: options parsed, outputs carry the new parameters" begin
        blk = cs_ccf_block(; synth = (n_epochs = 6, trend = Dict("A" => (0.05, 0.0)), _TSYNTH...),
                           extra = Dict{String, Any}("jitter" => "shared", "continuum" => "shared",
                                                     "depth_scale" => "none",
                                                     "shared_fwhm" => true,
                                                     "pixel_correlation" => "neff",
                                                     "resolution_kms" => 6.2,
                                                     "trend_t_ref" => 2459520.0))
        blk["components"][1]["trend"] = 1
        rt = JSON3.read(JSON3.write(Dict("ccf" => [blk])), Dict{String, Any})
        s = Nereus._parse_ccf_blocks(rt["ccf"])[1]
        @test [c.trend for c in s.components] == [1, 0]
        @test s.shared_fwhm && s.pixel_correlation === :neff
        @test s.resolution_kms == 6.2 && s.trend_t_ref == 2459520.0
        # defaults through the config
        s0 = Nereus._parse_ccf_blocks(cs_ccf_block(synth = (n_epochs = 3,)))[1]
        @test !s0.shared_fwhm && s0.pixel_correlation === :independent &&
              isnan(s0.resolution_kms) && all(c -> c.trend == 0, s0.components)
        bad = deepcopy(blk); bad["pixel_correlation"] = "banded"
        @test_throws ArgumentError Nereus._parse_ccf_blocks(bad)
        bad = deepcopy(blk); bad["components"][1]["trend"] = 3
        @test_throws ArgumentError Nereus._parse_ccf_blocks(bad)
        short = cs_ccf_block(; synth = (n_epochs = 2,))
        short["components"][1]["trend"] = 1
        @test_throws ArgumentError Nereus._parse_ccf_blocks(short)
        # a short run: chains.nc and summary.json carry the trend and the shared width
        out = mktempdir()
        delete!(blk, "trend_t_ref")
        cfg = Dict{String, Any}("output_dir" => out, "seed" => 3,
            "data" => Dict{String, Any}("ccf" => [blk]),
            "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"]),
            "priors" => cs_prior_cfg(),
            "sampler" => Dict{String, Any}("name" => "pt_emcee", "kwargs" => Dict(
                "n_temps" => 2, "n_walkers" => 48, "n_steps" => 30, "n_burnin" => 10,
                "show_progress" => false)),
            "output" => Dict{String, Any}("plots" => String[], "ppc" => false,
                                          "detection_limits" => false))
        sm = run_job(cfg)
        @test sm["status"] == "ok"
        NCDataset(joinpath(out, "chains.nc")) do ds
            @test haskey(ds, "ccf_gdot_SIM_A") && haskey(ds, "ccf_fwhm_SIM")
            @test !haskey(ds, "ccf_fwhm_SIM_A") && !haskey(ds, "ccf_gdot_SIM_B")
            @test all(isfinite, vec(Array(ds["ccf_gdot_SIM_A"])))
        end
        js = read(joinpath(out, "summary.json"), String)
        @test occursin("\"ccf_gdot_SIM_A\"", js) && occursin("\"ccf_fwhm_SIM\"", js)
    end
end
