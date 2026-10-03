# ActivityDecorrelation / ActivityJitter / ErrorScale with their layout slots
# resolved once (RVModifierSlots) instead of a name built and looked up per
# point: the per-point terms must equal the reference functions bit for bit,
# and the workspace, generic and rv_predictions paths must agree.
using Nereus, Test, Random
using Nereus: Theta, Params, Data, InstrumentConfig, PlanetDataSources, NoiseModel,
              ActivityDecorrelation, ActivityJitter, ErrorScale, CeleriteRotation,
              IndicatorFloor, PTWorkspace, RVModifierSlots, rv_log_likelihood,
              rv_predictions, apply_activity_decorrelation, apply_activity_jitter,
              error_scale_factor, _errorscale_covers, _modifier_slots!,
              _ad_term, _aj_variance, _es_variance, is_noise_model_active

function _mod_data(rng; n = 40)
    t = sort(1000.0 .+ 200 .* rand(rng, n))
    inst = [1 + mod(i, 3) for i in 1:n]
    ind = Dict{String,Vector{Float64}}(c => randn(rng, n) for c in ("bis", "fwhm", "log_rhk"))
    ind["bis"][5] = NaN          # skipped by ActivityDecorrelation
    errs = Dict{String,Vector{Float64}}(c => 0.1 .+ rand(rng, n) for c in ("bis", "fwhm"))
    Data(; t_rv = t, rv = 5 .* randn(rng, n), rv_err = 1 .+ rand(rng, n), rv_inst = inst,
           indicators = ind, indicator_errs = errs)
end

function _mod_params(d, noise; max_kplanet = 1)
    Params(; max_kplanet, planet_modes = fill(Nereus.RV_ONLY, max_kplanet),
             instruments = InstrumentConfig(rv = ["A", "B", "C"]), data = d, M_s = 1.0,
             noise_models = NoiseModel[noise...])
end

_mod_ws(p, d) = PTWorkspace(p, p.config.max_kplanet, length(p.config.noise_models);
                            n_obs = length(d.t_rv))

function _mod_draws(p, n, rng)
    [[Nereus.quantile(ps, rand(rng)) for ps in p.layout.unfrozen_priors] for _ in 1:n]
end

function _mod_set!(th, p, x)
    for (j, idx) in enumerate(p.layout.unfrozen_idx)
        th.values[idx] = x[j]
    end
    th
end

@testset "RV modifier slots resolved once" begin
    rng = MersenneTwister(77)
    d = _mod_data(rng)
    menus = (
        "AD per instrument + ErrorScale" =>
            [ActivityDecorrelation(indicators = ["bis", "fwhm", "log_rhk"]), ErrorScale()],
        "AD shared, derivative, labelled + ErrorScale on two instruments" =>
            [ActivityDecorrelation(indicators = ["bis", "fwhm"], per_instrument = false,
                                   derivative = true, label = "ff"),
             ErrorScale(instruments = ["A", "C"])],
        "two AD models + ActivityJitter" =>
            [ActivityDecorrelation(indicators = ["fwhm"], derivative = true),
             ActivityDecorrelation(indicators = ["bis"], label = "two"),
             ActivityJitter(indicator = "log_rhk")],
        "ErrorScale + ActivityJitter (last one wins), GP and floor" =>
            [ErrorScale(instruments = ["B"]), ActivityJitter(indicator = "fwhm"),
             CeleriteRotation(),
             IndicatorFloor(channels = [:fwhm, :log_rhk], kernel = :qp)],
    )
    @testset "$label" for (label, noise) in menus
        p = _mod_params(d, noise)
        th = Theta{Float64}(p)
        ws = _mod_ws(p, d)
        sl = RVModifierSlots()
        nm_list = p.config.noise_models
        nfin = 0
        for x in _mod_draws(p, 60, rng)
            _mod_set!(th, p, x)
            _modifier_slots!(sl, th, d)
            ok = true
            for i in eachindex(d.t_rv), (m, nm) in enumerate(nm_list)
                ins = d.rv_inst[i]
                pred = 0.37 * i
                e2 = d.rv_err[i]^2
                if nm isa ActivityDecorrelation
                    ok &= _ad_term(pred, th, d, nm, sl, m, ins, i) ===
                          apply_activity_decorrelation(pred, th, d, nm, ins, i)
                elseif nm isa ActivityJitter
                    ok &= _aj_variance(e2, th, d, nm, sl, m, ins, i) ===
                          apply_activity_jitter(e2, th, d, nm, ins, i)
                elseif nm isa ErrorScale
                    ok &= sl.es_cov[m][ins] == _errorscale_covers(th, nm, ins)
                    _errorscale_covers(th, nm, ins) &&
                        (ok &= _es_variance(d.rv_err[i], th, nm, sl, m, ins) ===
                               error_scale_factor(th, nm, ins) * d.rv_err[i] * d.rv_err[i])
                end
            end
            @test ok
            a = rv_log_likelihood(th, d, ws)
            b = rv_log_likelihood(th, d)
            @test a === b
            isfinite(a) && (nfin += 1)
        end
        @test nfin > 10
        # rv_predictions resolves its own slots: same predictions as the
        # per-point reference, assembled here by hand for one draw.
        pr, va = rv_predictions(th, d)
        @test length(pr) == length(d.t_rv) && all(isfinite, va)
    end

    @testset "rv_predictions matches the per-point reference" begin
        noise = [ActivityDecorrelation(indicators = ["bis", "fwhm"], derivative = true),
                 ActivityJitter(indicator = "log_rhk"), ErrorScale(instruments = ["C"])]
        p = _mod_params(d, noise; max_kplanet = 0)
        th = Theta{Float64}(p)
        for x in _mod_draws(p, 10, rng)
            _mod_set!(th, p, x)
            pr, va = rv_predictions(th, d)
            for i in eachindex(d.t_rv)
                ins = d.rv_inst[i]
                dt = d.t_rv[i] - d.t_ref
                ref = Nereus.rv_gamma(th, ins) + 0.0 * dt + 0.0 * dt * dt
                ref = apply_activity_decorrelation(ref, th, d, noise[1], ins, i)
                s = Nereus.rv_sigma(th, ins)
                v = d.rv_err[i] * d.rv_err[i] + s * s
                v = apply_activity_jitter(d.rv_err[i] * d.rv_err[i], th, d, noise[2], ins, i)
                if _errorscale_covers(th, noise[3], ins)
                    v = error_scale_factor(th, noise[3], ins) * d.rv_err[i] * d.rv_err[i]
                end
                @test pr[i] === ref
                @test va[i] === v
            end
        end
    end

    @testset "slots follow the layout and the data" begin
        noise1 = [ActivityDecorrelation(indicators = ["bis", "fwhm"]), ErrorScale()]
        noise2 = [ActivityJitter(indicator = "bis"),
                  ActivityDecorrelation(indicators = ["log_rhk"], per_instrument = false)]
        p1 = _mod_params(d, noise1)
        p2 = _mod_params(d, noise2)
        d2 = _mod_data(MersenneTwister(3))
        ws = _mod_ws(p1, d)
        th1 = Theta{Float64}(p1)
        th2 = Theta{Float64}(p2)
        for (p, th, dd) in ((p1, th1, d), (p2, th2, d), (p1, th1, d2), (p1, th1, d))
            for x in _mod_draws(p, 10, rng)
                _mod_set!(th, p, x)
                @test rv_log_likelihood(th, dd, ws) === rv_log_likelihood(th, dd)
            end
        end
    end

    @testset "no per-point allocation" begin
        noise = [ActivityDecorrelation(indicators = ["bis", "fwhm", "log_rhk"], derivative = true),
                 ErrorScale()]
        p = _mod_params(d, noise)
        th = Theta{Float64}(p)
        ws = _mod_ws(p, d)
        xs = _mod_draws(p, 3, rng)
        _mod_set!(th, p, xs[1]); rv_log_likelihood(th, d, ws)
        _mod_set!(th, p, xs[2])
        # The old path built 2-3 strings per (point, indicator): ~40 points ×
        # 3 indicators here, tens of kB. What is left is per call, not per point.
        @test (@allocated rv_log_likelihood(th, d, ws)) < 1024
    end
end
