# Correctness gates for the parametric (non-GP) correlated-noise menu:
# ErrorScale, NightlyOffset, HarmonicBlock (src/noise/parametric_noise.jl).
using Nereus, Test, Random, LinearAlgebra
import ForwardDiff
using Nereus: ErrorScale, NightlyOffset, HarmonicBlock, StudentT, MaternGP,
               _woodbury_white_ll, _woodbury_celerite_ll, _woodbury_matern_ll,
               dense_additive_ll, _additive_factor, _nightly_factor, _harmonic_factor,
               _night_group_ids, error_scale_factor,
               _studentt_diag_ll, validate_noise_models,
               default_noise_menu, SSMatern32, _kfuncs, gp_log_likelihood

# Two-instrument RV dataset whose epochs cluster into nights (intra-night gaps
# ≪ 0.5 d, inter-night gaps ≫ 0.5 d) so nightly grouping is non-trivial.
function _clustered_dataset(rng; n_nights = 8, per_night = 3)
    t = Float64[]; inst = Int[]
    for m in 1:2
        base = 100.0 * m                      # offset the two instruments in time
        for g in 1:n_nights
            night0 = base + 3.0 * g + 0.3 * m
            for _ in 1:per_night
                push!(t, night0 + 0.05 * rand(rng))   # < 0.5 d within a night
                push!(inst, m)
            end
        end
    end
    perm = sortperm(t)
    return t[perm], inst[perm]
end

function _build(noise_models, t, inst; extra_truth = Dict{String,Float64}())
    n = length(t)
    data = Nereus.Data(; t_rv = t, rv = zeros(n), rv_err = fill(0.5, n),
                          rv_inst = inst)
    ic = Nereus.InstrumentConfig(rv = ["A", "B"])
    p = Nereus.Params(; max_kplanet = 0, planet_modes = Nereus.PlanetDataSources[],
                         instruments = ic, data = data, M_s = 1.0,
                         noise_models = Nereus.NoiseModel[noise_models...])
    th = Nereus.Theta{Float64}(p)
    for (nm, v) in extra_truth
        haskey(p.layout.name_to_idx, nm) && Nereus.set_param!(th, nm, v)
    end
    return p, th, data
end

@testset "parametric noise menu" begin
    rng = MersenneTwister(2026_0720)
    t, inst = _clustered_dataset(rng)
    n = length(t)
    inst_names = ["A", "B"]
    two_pi = 2π
    y   = randn(rng, n)
    var = fill(0.25, n) .+ 0.1 .* rand(rng, n)

    @testset "gate 1 — NightlyOffset dense-oracle parity (white base)" begin
        m = NightlyOffset(gap = 0.5)
        _, th, _ = _build([m], t, inst;
            extra_truth = Dict("night_sigma_A" => 0.8, "night_sigma_B" => 1.3))
        F = _additive_factor([m], th, t, inst, inst_names)
        # one column per night-group; each instrument gets n_nights groups.
        ids, gi = _night_group_ids(m, t, inst, inst_names)
        @test size(F, 2) == length(gi)
        @test maximum(ids) == length(gi)
        ss = _woodbury_white_ll(y, var, F, two_pi)
        de = dense_additive_ll(y, var, t, inst, inst_names, [m], th)
        @test isfinite(ss)
        @test abs(ss - de) / max(abs(de), 1) < 1e-9
    end

    @testset "gate 2 — HarmonicBlock dense-oracle parity (white base)" begin
        m = HarmonicBlock(nharm = 3)
        _, th, _ = _build([m], t, inst;
            extra_truth = Dict("harm_period" => 11.0,
                               "harm_amp_A" => 1.4, "harm_amp_B" => 0.7))
        F = _additive_factor([m], th, t, inst, inst_names)
        @test size(F, 2) == 2 * 3                 # rank 2·nharm, shared latent
        ss = _woodbury_white_ll(y, var, F, two_pi)
        de = dense_additive_ll(y, var, t, inst, inst_names, [m], th)
        @test isfinite(ss)
        @test abs(ss - de) / max(abs(de), 1) < 1e-9
    end

    @testset "gate 3 — GP-base Woodbury parity (NightlyOffset + CeleriteSHO)" begin
        base = Nereus.CeleriteSHO()
        m = NightlyOffset(gap = 0.5)
        _, th, _ = _build([base, m], t, inst;
            extra_truth = Dict("gp_log_S0" => 0.5, "gp_log_Q" => 1.0,
                               "gp_log_omega0" => 0.3,
                               "night_sigma_A" => 0.6, "night_sigma_B" => 0.9))
        F = _additive_factor([m], th, t, inst, inst_names)
        ss = _woodbury_celerite_ll(y, var, t, th, base, F, two_pi)
        de = dense_additive_ll(y, var, t, inst, inst_names, [m], th; base_nm = base)
        @test isfinite(ss)
        @test abs(ss - de) / max(abs(de), 1) < 1e-8
    end

    @testset "gate 4 — combined terms + empty factor reduces to base" begin
        # NightlyOffset + HarmonicBlock together, white base.
        mn = NightlyOffset(gap = 0.5); mh = HarmonicBlock(nharm = 2)
        _, th, _ = _build([mn, mh], t, inst;
            extra_truth = Dict("night_sigma_A" => 0.5, "night_sigma_B" => 0.7,
                               "harm_period" => 9.0,
                               "harm_amp_A" => 1.0, "harm_amp_B" => 0.5))
        F = _additive_factor([mn, mh], th, t, inst, inst_names)
        @test size(F, 2) == length(_night_group_ids(mn, t, inst, inst_names)[2]) + 2 * 2
        ss = _woodbury_white_ll(y, var, F, two_pi)
        de = dense_additive_ll(y, var, t, inst, inst_names, [mn, mh], th)
        @test abs(ss - de) / max(abs(de), 1) < 1e-9
        # empty factor ⇒ plain diagonal Gaussian
        F0 = zeros(n, 0)
        ll0 = _woodbury_white_ll(y, var, F0, two_pi)
        ref = -0.5 * (sum(y .^ 2 ./ var) + sum(log, var) + n * log(two_pi))
        @test abs(ll0 - ref) < 1e-10
    end

    @testset "gate 5 — ForwardDiff = finite-diff (additive params)" begin
        mn = NightlyOffset(gap = 0.5); mh = HarmonicBlock(nharm = 2)
        p, th, _ = _build([mn, mh], t, inst)
        pnames = ["night_sigma_A", "night_sigma_B", "harm_period",
                  "harm_amp_A", "harm_amp_B"]
        idx = [p.layout.name_to_idx[nm] for nm in pnames]
        p0 = [0.5, 0.7, 9.0, 1.0, 0.5]
        function f(v)
            th2 = Nereus.Theta{eltype(v)}(p)
            for (k, ix) in enumerate(idx)
                th2.values[ix] = v[k]
            end
            F = _additive_factor([mn, mh], th2, t, inst, inst_names)
            _woodbury_white_ll(y, var, F, eltype(v)(2π))
        end
        gad = ForwardDiff.gradient(f, p0)
        gfd = map(eachindex(p0)) do i
            δ = 1e-6 * max(abs(p0[i]), 1); pp = copy(p0); pm = copy(p0)
            pp[i] += δ; pm[i] -= δ; (f(pp) - f(pm)) / (2δ)
        end
        @test maximum(abs.(gad .- gfd) ./ (abs.(gfd) .+ 1)) < 1e-5
    end

    @testset "gate 6 — ErrorScale replaces jitter, end-to-end" begin
        # f = 1 must equal the plain white LL; f ≠ 1 must scale the formal error.
        m = ErrorScale()
        p, th, data = _build([m], t, inst)
        # give the two instruments a systemic + jitter so the baseline isn't trivial
        for (nm, v) in (("gamma_A", 0.0), ("gamma_B", 0.0),
                        ("sigma_A", 0.3), ("sigma_B", 0.3),
                        ("errscale_A", 1.0), ("errscale_B", 1.0))
            haskey(p.layout.name_to_idx, nm) && Nereus.set_param!(th, nm, v)
        end
        yobs = randn(MersenneTwister(7), n)
        data2 = Nereus.Data(; t_rv = t, rv = yobs, rv_err = fill(0.5, n), rv_inst = inst)
        ll_f1 = Nereus.rv_log_likelihood(th, data2)
        # with errscale, jitter is REPLACED: var = f²·σ_formal² (no +σ_jit²).
        ref_f1 = -0.5 * sum(@. log(2π * 0.25) + yobs^2 / 0.25)   # f=1 ⇒ var = 0.5²
        @test abs(ll_f1 - ref_f1) < 1e-8
        Nereus.set_param!(th, "errscale_A", 2.0)
        Nereus.set_param!(th, "errscale_B", 2.0)
        ll_f2 = Nereus.rv_log_likelihood(th, data2)
        ref_f2 = -0.5 * sum(@. log(2π * 1.0) + yobs^2 / 1.0)     # f=2 ⇒ var = (2·0.5)²=1
        @test abs(ll_f2 - ref_f2) < 1e-8
        @test error_scale_factor(th, m, 1) == 4.0                # f²
    end

    @testset "gate 7 — end-to-end routing through rv_log_likelihood" begin
        yobs = randn(MersenneTwister(11), n)
        data2 = Nereus.Data(; t_rv = t, rv = yobs, rv_err = fill(0.5, n), rv_inst = inst)
        for m in (NightlyOffset(gap = 0.5), HarmonicBlock(nharm = 3))
            p, th, _ = _build([m], t, inst)
            for nm in p.layout.unfrozen_names
                v = startswith(nm, "night_sigma") ? 0.7 :
                    nm == "harm_period"           ? 12.0 :
                    startswith(nm, "harm_amp")    ? 1.0 : nothing
                v === nothing || Nereus.set_param!(th, nm, v)
            end
            ll = Nereus.rv_log_likelihood(th, data2)
            ll_white = -0.5 * sum(@. log(2π * 0.25) + yobs^2 / 0.25)
            @test isfinite(ll)
            @test ll != ll_white                  # the additive term changed Σ
        end
    end

    @testset "gate 8 — StudentT closed-form + Gaussian limit" begin
        yobs = randn(MersenneTwister(13), n)
        data2 = Nereus.Data(; t_rv = t, rv = yobs, rv_err = fill(0.5, n), rv_inst = inst)
        p, th, _ = _build([StudentT()], t, inst)
        for (nm, v) in (("gamma_A", 0.0), ("gamma_B", 0.0),
                        ("sigma_A", 0.2), ("sigma_B", 0.2), ("studentt_nu", 4.0))
            haskey(p.layout.name_to_idx, nm) && Nereus.set_param!(th, nm, v)
        end
        var = fill(0.25 + 0.04, n)                 # obs_err² + jitter² = 0.5²+0.2²
        ll = Nereus.rv_log_likelihood(th, data2)
        @test abs(ll - _studentt_diag_ll(yobs, var, 4.0)) < 1e-8
        # ν → large ⇒ Student-t collapses onto the Gaussian white LL.
        Nereus.set_param!(th, "studentt_nu", 1e6)
        ll_big = Nereus.rv_log_likelihood(th, data2)
        ll_gauss = -0.5 * sum(@. log(2π * var) + yobs^2 / var)
        @test abs(ll_big - ll_gauss) < 1e-3
        # heavier tails than Gaussian ⇒ finite, and different at moderate ν.
        @test isfinite(ll) && ll != ll_gauss
    end

    @testset "gate 9 — StudentT hard incompatibility guard" begin
        # construction rejects Student-t + any covariance/additive/sequential
        @test_throws ArgumentError validate_noise_models(
            Nereus.NoiseModel[StudentT(), Nereus.CeleriteSHO()])
        @test_throws ArgumentError validate_noise_models(
            Nereus.NoiseModel[StudentT(), NightlyOffset()])
        # white-only combos are fine
        validate_noise_models(Nereus.NoiseModel[StudentT(), ErrorScale()])
        @test true
    end

    @testset "gate 10 — StudentT ForwardDiff through ν (loggamma)" begin
        yobs = randn(MersenneTwister(17), n)
        var = fill(0.3, n)
        f = ν -> _studentt_diag_ll(yobs, var, ν[1])
        gad = ForwardDiff.gradient(f, [5.0])[1]
        gfd = (f([5.0 + 1e-6]) - f([5.0 - 1e-6])) / 2e-6
        @test abs(gad - gfd) / (abs(gfd) + 1) < 1e-6
    end

    @testset "gate 11 — labeled ActivityDecorrelation = separate hypotheses" begin
        # A labeled model must (a) score identically to the unlabeled model at
        # the same coefficient, (b) coexist with a second labeled model without
        # a param-name collision (the linear-vs-FF′ exclusion-group setup).
        m = 25
        tt = collect(1.0:m); bis = 0.1 .* sin.(tt ./ 3)
        yb = 0.4 .* randn(MersenneTwister(3), m)
        d = Nereus.Data(; t_rv = tt, rv = yb, rv_err = fill(0.5, m),
                          rv_inst = ones(Int, m),
                          indicators = Dict("bis" => bis),
                          indicator_errs = Dict("bis" => fill(0.01, m)))
        ic = Nereus.InstrumentConfig(rv = ["HARPS"])
        function score(ad, cname, cval)
            p = Nereus.Params(; max_kplanet = 0, planet_modes = Nereus.PlanetDataSources[],
                                 instruments = ic, data = d, M_s = 1.0,
                                 noise_models = Nereus.NoiseModel[ad])
            th = Nereus.Theta{Float64}(p)
            Nereus.set_param!(th, "gamma_HARPS", 0.0)
            Nereus.set_param!(th, "sigma_HARPS", 0.1)
            Nereus.set_param!(th, cname, cval)
            return Nereus.rv_log_likelihood(th, d)
        end
        ll_bare = score(Nereus.ActivityDecorrelation(indicators = ["bis"]),
                        "C_bis_HARPS", 0.35)
        ll_lab  = score(Nereus.ActivityDecorrelation(indicators = ["bis"], label = "lin"),
                        "C_bis_HARPS_lin", 0.35)
        @test abs(ll_bare - ll_lab) < 1e-10       # label is cosmetic for scoring

        # two labeled models coexist with disjoint params (no collision)
        adL = Nereus.ActivityDecorrelation(indicators = ["bis"], derivative = false, label = "lin")
        adF = Nereus.ActivityDecorrelation(indicators = ["bis"], derivative = true,  label = "ffp")
        p2 = Nereus.Params(; max_kplanet = 0, planet_modes = Nereus.PlanetDataSources[],
                              instruments = ic, data = d, M_s = 1.0,
                              noise_models = Nereus.NoiseModel[adL, adF])
        for nm in ("C_bis_HARPS_lin", "C_bis_HARPS_ffp", "Cdot_bis_HARPS_ffp")
            @test haskey(p2.layout.name_to_idx, nm)
        end
        @test !haskey(p2.layout.name_to_idx, "C_bis_HARPS")   # no bare (collidable) name
    end

    @testset "gate 12 — MaternGP standalone + additive Woodbury parity" begin
        σt, ρt = 0.8, 12.0
        kern = SSMatern32(σt, ρt)
        Kd = [_kfuncs(kern, t[i] - t[j])[1] for i in 1:n, j in 1:n]  # k(τ) value block
        # standalone: gp_log_likelihood(MaternGP) == dense N(0, K + diag var)
        _, th, _ = _build([MaternGP()], t, inst;
            extra_truth = Dict("matern_sigma" => σt, "matern_rho" => ρt))
        ll = gp_log_likelihood(y, var, t, th, MaternGP())
        Σ = Kd + Diagonal(var)
        C = cholesky(Symmetric(Σ))
        ll_dense = -0.5 * (dot(y, C \ y) + logdet(C) + n * log(2π))
        @test isfinite(ll) && abs(ll - ll_dense) / max(abs(ll_dense), 1) < 1e-8

        # + NightlyOffset via semiseparable-base Woodbury == dense (K + FFᵀ + diag)
        mn = NightlyOffset(gap = 0.5)
        _, th2, _ = _build([MaternGP(), mn], t, inst;
            extra_truth = Dict("matern_sigma" => σt, "matern_rho" => ρt,
                               "night_sigma_A" => 0.6, "night_sigma_B" => 0.9))
        F = _additive_factor([mn], th2, t, inst, inst_names)
        ll_w = _woodbury_matern_ll(y, var, t, th2, MaternGP(), F, 2π)
        Σ2 = Kd + F * F' + Diagonal(var)
        C2 = cholesky(Symmetric(Σ2))
        ll_d2 = -0.5 * (dot(y, C2 \ y) + logdet(C2) + n * log(2π))
        @test isfinite(ll_w) && abs(ll_w - ll_d2) / max(abs(ll_d2), 1) < 1e-8
    end

    @testset "gate 13 — default_noise_menu structure + validity" begin
        # (a) no indicators ⇒ no ActivityGP / AD / floor
        dp = Nereus.Data(; t_rv = t, rv = zeros(n), rv_err = fill(0.5, n), rv_inst = inst)
        m0 = default_noise_menu(dp)
        names0 = string.(typeof.(m0.toggleable))
        @test !any(occursin.("ActivityGP", names0))
        @test !any(occursin.("MaternGP", names0))   # free-period interpolator: off by default
        @test any(occursin.("CeleriteRotation", names0))  # anchored rotation is the GP slot
        @test any(occursin.("ErrorScale", names0))
        @test !any(occursin.("StudentT", names0))   # trimmed from default (opt-in)
        # every exclusion-group member is in toggleable (the TransDimConfig invariant)
        for g in m0.exclusion_groups, mem in g
            @test mem in m0.toggleable
        end
        # WHITE competes with CORRELATED — it must NEVER compose. ErrorScale has to
        # sit in the SAME exclusion group as the covariance GP: when they were
        # separate axes the fit did both, inflating FEROS ×10 to discard it while a
        # rotation GP injected at P_rot/2 ≈ P_orb and ran K from ~9 to ~17.
        gi0 = findfirst(g -> any(m -> m isa ErrorScale, g), m0.exclusion_groups)
        @test gi0 !== nothing
        @test any(m -> m isa Nereus.CeleriteRotation, m0.exclusion_groups[gi0])

        # (b) with BIS indicator ⇒ ActivityGP + labeled ADs + always-on floor
        bis = 0.1 .* sin.(t ./ 5)
        di = Nereus.Data(; t_rv = t, rv = zeros(n), rv_err = fill(0.5, n), rv_inst = inst,
                           indicators = Dict("bis" => bis),
                           indicator_errs = Dict("bis" => fill(0.01, n)))
        mi = default_noise_menu(di)
        ti = string.(typeof.(mi.toggleable))
        @test any(occursin.("ActivityGP", ti))
        @test count(occursin.("ActivityDecorrelation", ti)) == 1       # linear only (FF′ opt-in)
        @test any(m -> m isa Nereus.IndicatorFloor, mi.noise_models)  # honest-occupancy floor
        @test !any(m -> m isa Nereus.IndicatorFloor, mi.toggleable)   # always on, not toggled
        for g in mi.exclusion_groups, mem in g
            @test mem in mi.toggleable
        end
        # Single noise-treatment group: white (ErrorScale) vs correlated GPs vs the
        # indicator decorrelation all compete for ONE winner.
        gii = findfirst(g -> any(m -> m isa ErrorScale, g), mi.exclusion_groups)
        @test gii !== nothing
        let grp = mi.exclusion_groups[gii]
            @test any(m -> m isa Nereus.CeleriteRotation, grp)
            @test any(m -> m isa Nereus.ActivityDecorrelation, grp)
            @test any(m -> m isa Nereus.ActivityGP, grp)
        end
        # Params builds from the menu (labeled ADs don't collide; layout OK)
        p = Nereus.Params(; max_kplanet = 0, planet_modes = Nereus.PlanetDataSources[],
                             instruments = Nereus.InstrumentConfig(rv = ["A", "B"]),
                             data = di, M_s = 1.0, noise_models = mi.noise_models,
                             transdim_noise = true)
        @test length(p.layout.unfrozen_names) > 0

        # (c) USER-SELECTED indicator subset: menu members must use ONLY the
        # selected channels (leak surface control — a non-correlating channel
        # can absorb planet amplitude through its AD coefficient).
        d2 = Nereus.Data(; t_rv = t, rv = zeros(n), rv_err = fill(0.5, n), rv_inst = inst,
                            indicators = Dict("bis" => bis, "fwhm" => bis .* 2),
                            indicator_errs = Dict("bis" => fill(0.01, n),
                                                  "fwhm" => fill(0.01, n)))
        ms = default_noise_menu(d2; indicators = ["bis"])
        adx = only(filter(m -> m isa Nereus.ActivityDecorrelation, ms.toggleable))
        @test adx.indicators == ["bis"]
        agx = only(filter(m -> m isa Nereus.ActivityGP, ms.toggleable))
        @test agx.channels == [:bis]
        flx = only(filter(m -> m isa Nereus.IndicatorFloor, ms.noise_models))
        @test flx.channels == [:bis]
    end

    @testset "gate 14 — run_job noise_menu block (exoautomata path)" begin
        # _build_model with a `noise_menu` block: menu built server-side with the
        # selected indicators, models wired into params, transdim_noise set, and
        # _build_transdim consumes the SAME menu objects (identity, not copies).
        n2 = 30; t2 = collect(range(0, 40; length = n2))
        d3 = Nereus.Data(; t_rv = t2, rv = zeros(n2), rv_err = fill(0.5, n2),
                            rv_inst = ones(Int, n2),
                            indicators = Dict("bis" => 0.1 .* sin.(t2 ./ 3),
                                              "fwhm" => 0.2 .* cos.(t2 ./ 5)),
                            indicator_errs = Dict("bis" => fill(0.01, n2),
                                                  "fwhm" => fill(0.01, n2)))
        cfg = Dict{String, Any}(
            "model" => Dict{String, Any}("max_kplanet" => 0,
                                          "planet_modes" => String[]),
            "noise_menu" => Dict{String, Any}("indicators" => ["bis"]))
        star = (; M_s = 1.0, R_s = nothing)
        pj, tj, menu = Nereus._build_model(cfg, d3, star, ["A"], String[])
        @test menu !== nothing
        # constructors may copy the OUTER vector; what the sampler needs is
        # ELEMENT identity (findfirst(==(nm),...) matches by egal on Vector fields)
        @test all(a === b for (a, b) in zip(pj.config.noise_models, menu.noise_models))
        @test length(pj.config.noise_models) == length(menu.noise_models)
        adj = only(filter(m -> m isa Nereus.ActivityDecorrelation, menu.toggleable))
        @test adj.indicators == ["bis"]
        td = Nereus._build_transdim(Dict{String, Any}("max_kplanet" => 0); menu = menu)
        @test td.noise                      # menu flips the default on
        @test all(a === b for (a, b) in zip(td.toggleable, menu.toggleable))
        @test length(td.toggleable) == length(menu.toggleable)
        @test all(all(a === b for (a, b) in zip(g1, g2))
                  for (g1, g2) in zip(td.noise_exclusion_groups, menu.exclusion_groups))
        # unknown channel → hard error
        bad = Dict{String, Any}(
            "model" => Dict{String, Any}("max_kplanet" => 0, "planet_modes" => String[]),
            "noise_menu" => Dict{String, Any}("indicators" => ["s_index"]))
        @test_throws ErrorException Nereus._build_model(bad, d3, star, ["A"], String[])
        # menu + explicit noise_models → hard error
        both = Dict{String, Any}(
            "model" => Dict{String, Any}("max_kplanet" => 0, "planet_modes" => String[]),
            "noise_models" => [Dict{String, Any}("kind" => "ErrorScale")],
            "noise_menu" => Dict{String, Any}())
        @test_throws ErrorException Nereus._build_model(both, d3, star, ["A"], String[])
        # menu + JSON toggleable → hard error
        @test_throws ErrorException Nereus._build_transdim(
            Dict{String, Any}("max_kplanet" => 0,
                              "toggleable" => [Dict("kind" => "ErrorScale")]); menu = menu)
    end
end
