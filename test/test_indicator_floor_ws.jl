# IndicatorFloor (:qp) on the PTWorkspace path: the workspace method must
# return exactly the bits of the generic method (which stays the reference,
# and the ForwardDiff path), and must not allocate per call.
using Nereus, Test, Random
import ForwardDiff
using Nereus: Theta, Params, Data, InstrumentConfig, PlanetDataSources, NoiseModel,
              IndicatorFloor, ActivityGP, CeleriteRotation, PTWorkspace,
              indicator_floor_log_likelihood, rv_log_likelihood, set_param!

function _floor_data(rng; n = 36, channels = ("bis", "fwhm", "logrhk"),
                     errs_for = ("bis", "fwhm"), duplicate = false)
    t = sort(2000.0 .+ 300 .* rand(rng, n))
    duplicate && (t[2] = t[1])
    inst = [isodd(i) ? 1 : 2 for i in 1:n]
    ind = Dict{String,Vector{Float64}}(c => randn(rng, n) for c in channels)
    errs = Dict{String,Vector{Float64}}(c => 0.1 .+ 0.2 .* rand(rng, n) for c in errs_for)
    return Data(; t_rv = t, rv = 3 .* randn(rng, n), rv_err = 1 .+ rand(rng, n),
                  rv_inst = inst, indicators = ind, indicator_errs = errs,
                  normalize_indicators = false)
end

function _floor_params(data, noise)
    Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
             instruments = InstrumentConfig(rv = ["A", "B"]), data = data, M_s = 1.0,
             noise_models = NoiseModel[noise...])
end

_ws(p, d) = PTWorkspace(p, p.config.max_kplanet, length(p.config.noise_models);
                        n_obs = length(d.t_rv))

# Raw prior draws (every coordinate in its prior), as a sampler would visit.
function _prior_draws(p, n, rng)
    L = p.layout
    [[Nereus.quantile(ps, rand(rng)) for ps in L.unfrozen_priors] for _ in 1:n]
end

function _set!(th, p, x)
    for (j, idx) in enumerate(p.layout.unfrozen_idx)
        th.values[idx] = x[j]
    end
    th
end

# Allocation probe behind a function barrier (concrete argument types).
_floor_ws_alloc(th, d, ws) = @allocated indicator_floor_log_likelihood(th, d, ws)

@testset "IndicatorFloor :qp workspace path" begin
    rng = MersenneTwister(20261003)
    # :halpha is configured but absent from the data (skipped); :logrhk has
    # no errors (err² = 0 branch).
    floor = IndicatorFloor(channels = [:bis, :fwhm, :halpha, :logrhk], kernel = :qp)
    d = _floor_data(rng)

    @testset "bit-identical to the generic method: $label" for (label, noise) in (
            ("floor alone", [floor]),
            ("with CeleriteRotation", [CeleriteRotation(), floor]),
            ("AGP covering bis", [ActivityGP(channels = [:bis], marginalize_indicators = false),
                                  floor]))
        p = _floor_params(d, noise)
        th = Theta{Float64}(p)
        ws = _ws(p, d)
        nfin = 0
        for x in _prior_draws(p, 150, rng)
            _set!(th, p, x)
            a = indicator_floor_log_likelihood(th, d)
            b = indicator_floor_log_likelihood(th, d, ws)
            @test a === b
            isfinite(a) && (nfin += 1)
            @test rv_log_likelihood(th, d) === rv_log_likelihood(th, d, ws)
        end
        @test nfin > 50
    end

    @testset "rejections match" begin
        p = _floor_params(d, [floor])
        th = Theta{Float64}(p)
        ws = _ws(p, d)
        _set!(th, p, _prior_draws(p, 1, rng)[1])
        for (nm, v) in (("ind_floor_bis_amp", -1.0), ("ind_floor_fwhm_jit", 0.0),
                        ("ind_floor_period", -2.0), ("ind_floor_lambda_p", 0.0))
            old = th.values[p.layout.name_to_idx[nm]]
            set_param!(th, nm, v)
            @test indicator_floor_log_likelihood(th, d, ws) === -Inf
            @test indicator_floor_log_likelihood(th, d) === -Inf
            set_param!(th, nm, old)
        end
        # Duplicate epochs, no errors and a vanishing jitter: Σ is singular
        # in floating point. Both paths must agree on whatever LAPACK says.
        dd = _floor_data(MersenneTwister(5); errs_for = (), duplicate = true)
        pd = _floor_params(dd, [floor])
        thd = Theta{Float64}(pd)
        wsd = _ws(pd, dd)
        _set!(thd, pd, _prior_draws(pd, 1, rng)[1])
        set_param!(thd, "ind_floor_bis_jit", 1e-12)
        set_param!(thd, "ind_floor_bis_amp", 1e6)
        @test indicator_floor_log_likelihood(thd, dd, wsd) ===
              indicator_floor_log_likelihood(thd, dd)
    end

    @testset "slots follow the layout the workspace is used with" begin
        # A workspace resolved against one layout, then handed a Theta built
        # on another (different channels, so different slots): it must
        # re-resolve, not reuse stale slots.
        p1 = _floor_params(d, [floor])
        p2 = _floor_params(d, [CeleriteRotation(),
                               IndicatorFloor(channels = [:logrhk, :bis], kernel = :qp)])
        ws = _ws(p1, d)
        th1 = Theta{Float64}(p1)
        th2 = Theta{Float64}(p2)
        for (p, th) in ((p1, th1), (p2, th2), (p1, th1))
            for x in _prior_draws(p, 20, rng)
                _set!(th, p, x)
                @test indicator_floor_log_likelihood(th, d, ws) ===
                      indicator_floor_log_likelihood(th, d)
            end
        end
    end

    @testset "no allocation per call" begin
        p = _floor_params(d, [CeleriteRotation(), floor])
        th = Theta{Float64}(p)
        ws = _ws(p, d)
        xs = _prior_draws(p, 5, rng)
        _set!(th, p, xs[1])
        indicator_floor_log_likelihood(th, d, ws)
        _set!(th, p, xs[2])
        @test _floor_ws_alloc(th, d, ws) == 0
    end

    @testset "duals take the generic method" begin
        p = _floor_params(d, [floor])
        ws = _ws(p, d)
        x = _prior_draws(p, 1, rng)[1]
        function f(v, use_ws)
            th = Theta{eltype(v)}(p)
            for (j, idx) in enumerate(p.layout.unfrozen_idx)
                th.values[idx] = v[j]
            end
            use_ws ? indicator_floor_log_likelihood(th, d, ws) :
                     indicator_floor_log_likelihood(th, d)
        end
        @test ForwardDiff.gradient(v -> f(v, true), x) ==
              ForwardDiff.gradient(v -> f(v, false), x)
    end
end
