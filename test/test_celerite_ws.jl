# CeleriteRotation on the PTWorkspace path: same bits as the allocating
# method, no allocation per call; and the shared coefficient helpers give
# the values of the formulas they replaced.
using Nereus, Test, Random
import ForwardDiff
using Nereus: Theta, Params, Data, InstrumentConfig, PlanetDataSources, NoiseModel,
              CeleriteRotation, CeleriteSHO, IndicatorFloor, PTWorkspace,
              gp_log_likelihood, rv_log_likelihood, sho_coefficients,
              rotation_coefficients, celerite_loglike, indicator_floor_log_likelihood,
              CeleriteWork, _rotation_coefficients!

# sho_coefficients as it was written before the scalar helper.
function _sho_reference(S0, Q, ω0)
    T = promote_type(typeof(S0), typeof(Q), typeof(ω0))
    eps = T(1e-5)
    half = T(0.5)
    if Q < half
        f = sqrt(max(1 - 4Q^2, eps))
        a = half * S0 * ω0 * Q
        c = half * ω0 / Q
        return (T[a * (1 + 1/f), a * (1 - 1/f)], T[c * (1 - f), c * (1 + f)],
                T[], T[], T[], T[])
    else
        f = sqrt(max(4Q^2 - 1, eps))
        a = S0 * ω0 * Q
        c = half * ω0 / Q
        return (T[], T[], T[a], T[a/f], T[c], T[c * f])
    end
end

function _gp_data(rng; n = 50)
    t = sort(3000.0 .+ 400 .* rand(rng, n))
    ind = Dict{String,Vector{Float64}}("bis" => randn(rng, n), "fwhm" => randn(rng, n))
    errs = Dict{String,Vector{Float64}}("bis" => 0.1 .+ rand(rng, n))
    Data(; t_rv = t, rv = 4 .* randn(rng, n), rv_err = 1 .+ rand(rng, n),
           rv_inst = [isodd(i) ? 1 : 2 for i in 1:n], indicators = ind,
           indicator_errs = errs)
end

_gp_params(d, noise) = Params(; max_kplanet = 1, planet_modes = [Nereus.RV_ONLY],
                              instruments = InstrumentConfig(rv = ["A", "B"]), data = d,
                              M_s = 1.0, noise_models = NoiseModel[noise...])

_gp_ws(p, d) = PTWorkspace(p, p.config.max_kplanet, length(p.config.noise_models);
                           n_obs = length(d.t_rv))

_gp_draws(p, n, rng) =
    [[Nereus.quantile(ps, rand(rng)) for ps in p.layout.unfrozen_priors] for _ in 1:n]

function _gp_set!(th, p, x)
    for (j, idx) in enumerate(p.layout.unfrozen_idx)
        th.values[idx] = x[j]
    end
    th
end

_same_bits(a, b) = length(a) == length(b) &&
                    all(reinterpret(UInt64, a) .== reinterpret(UInt64, b))

# Allocation probes behind a function barrier (concrete argument types).
_alloc_gp(r, v, t, th, nm, ws) = @allocated gp_log_likelihood(r, v, t, th, nm, ws)
_alloc_floor(th, d, ws) = @allocated indicator_floor_log_likelihood(th, d, ws)

@testset "CeleriteRotation workspace path" begin
    rng = MersenneTwister(4242)

    @testset "coefficient helpers keep the old values" begin
        for _ in 1:500
            S0 = exp(4 * randn(rng)); ω0 = exp(2 * randn(rng))
            Q = rand(rng) < 0.5 ? 0.5 * rand(rng) + 1e-3 : 0.5 + 10 * rand(rng)
            @test all(map(_same_bits, sho_coefficients(S0, Q, ω0), _sho_reference(S0, Q, ω0)))
        end
        cw = CeleriteWork()
        for _ in 1:500
            σ = exp(randn(rng)); P = 1 + 50 * rand(rng); Q0 = 0.1 + 5 * rand(rng)
            dQ = 3 * rand(rng); fr = rand(rng)
            ref = rotation_coefficients(σ, P, Q0, dQ, fr)
            _rotation_coefficients!(cw, σ, P, Q0, dQ, fr)
            got = (cw.ar, cw.cr, cw.ac, cw.bc, cw.cc, cw.dc)
            @test all(map(_same_bits, got, ref))
        end
    end

    d = _gp_data(rng)
    floor = IndicatorFloor(channels = [:bis, :fwhm], kernel = :qp)
    @testset "bit-identical to the allocating method: $label" for (label, noise) in (
            ("rotation", [CeleriteRotation()]),
            ("rotation + floor", [CeleriteRotation(), floor]))
        p = _gp_params(d, noise)
        th = Theta{Float64}(p)
        ws = _gp_ws(p, d)
        nm = p.config.noise_models[1]
        r = randn(rng, length(d.t_rv))
        v = 0.5 .+ rand(rng, length(d.t_rv))
        nfin = 0
        for x in _gp_draws(p, 120, rng)
            _gp_set!(th, p, x)
            a = gp_log_likelihood(r, v, d.t_rv, th, nm)
            @test a === gp_log_likelihood(r, v, d.t_rv, th, nm, ws)
            isfinite(a) && (nfin += 1)
            # Everything but the floor is bit-identical; the floor's workspace
            # kernel uses the angle-difference sine (tolerance: see
            # test_indicator_floor_ws.jl).
            @test Nereus._rv_log_likelihood_core(th, d) ===
                  Nereus._rv_log_likelihood_core(th, d, ws)
            fa = indicator_floor_log_likelihood(th, d)
            fb = indicator_floor_log_likelihood(th, d, ws)
            @test fa === fb || abs(fa - fb) <= 1e-6 * max(1.0, abs(fa))
        end
        @test nfin > 50
        @test _alloc_gp(r, v, d.t_rv, th, nm, ws) == 0
        @test _alloc_floor(th, d, ws) == 0
    end

    @testset "other kernels and duals take the allocating method" begin
        p = _gp_params(d, [CeleriteSHO()])
        th = Theta{Float64}(p)
        ws = _gp_ws(p, d)
        r = randn(rng, length(d.t_rv)); v = 0.5 .+ rand(rng, length(d.t_rv))
        _gp_set!(th, p, _gp_draws(p, 1, rng)[1])
        nm = p.config.noise_models[1]
        @test gp_log_likelihood(r, v, d.t_rv, th, nm, ws) ===
              gp_log_likelihood(r, v, d.t_rv, th, nm)

        pr = _gp_params(d, [CeleriteRotation()])
        wsr = _gp_ws(pr, d)
        x = _gp_draws(pr, 1, rng)[1]
        nmr = pr.config.noise_models[1]
        function f(xv, use_ws)
            T = eltype(xv)
            t = Theta{T}(pr)
            for (j, idx) in enumerate(pr.layout.unfrozen_idx)
                t.values[idx] = xv[j]
            end
            use_ws ? gp_log_likelihood(T.(r), T.(v), d.t_rv, t, nmr, wsr) :
                     gp_log_likelihood(T.(r), T.(v), d.t_rv, t, nmr)
        end
        @test ForwardDiff.gradient(z -> f(z, true), x) ==
              ForwardDiff.gradient(z -> f(z, false), x)
    end

    @testset "buffers follow the number of points" begin
        # One workspace, GP evaluated on a shorter series and back: the
        # buffers are resized, never read stale.
        p = _gp_params(d, [CeleriteRotation()])
        th = Theta{Float64}(p)
        ws = _gp_ws(p, d)
        nm = p.config.noise_models[1]
        _gp_set!(th, p, _gp_draws(p, 1, rng)[1])
        for n in (length(d.t_rv), 17, length(d.t_rv))
            r = randn(rng, n); v = 0.5 .+ rand(rng, n); t = d.t_rv[1:n]
            @test gp_log_likelihood(r, v, t, th, nm, ws) ===
                  gp_log_likelihood(r, v, t, th, nm)
        end
    end
end
