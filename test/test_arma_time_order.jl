# AR/MA lags are previous observations IN TIME. Taken in storage order, RVs
# stored instrument by instrument put an earlier lag later in time, dt < 0, and
# exp(-dt/β) blows the correction up. The lags now run in time order, so data
# stored in any order give the likelihood of the same data sorted.
using Nereus, Test, Random
using Nereus: Theta, Params, Data, InstrumentConfig, PlanetDataSources, NoiseModel,
              MAModel, ARModel, apply_ma!, apply_ar!

function _arma_params(t, inst, rv, err, noise)
    d = Data(; t_rv = t, rv = rv, rv_err = err, rv_inst = inst)
    p = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
               instruments = InstrumentConfig(rv = ["A", "B"]), data = d,
               M_s = 1.0, noise_models = NoiseModel[noise...])
    return p, d
end

const _ARMA = Dict("ma_omega_1" => 0.6, "ma_beta_1" => 4.0,
                   "ma_omega_2" => 0.2, "ma_beta_2" => 9.0,
                   "ar_phi_1" => 0.5, "ar_alpha_1" => 3.0,
                   "ma_omega_1_A" => 0.5, "ma_beta_1_A" => 5.0,
                   "ma_omega_1_B" => 0.7, "ma_beta_1_B" => 2.0,
                   "gamma_A" => 1.0, "gamma_B" => -2.0, "sigma_A" => 0.8, "sigma_B" => 1.2)

function _arma_theta(p)
    th = Theta{Float64}(p)
    for (k, v) in _ARMA
        haskey(p.layout.name_to_idx, k) && Nereus.set_param!(th, k, v)
    end
    return th
end

@testset "AR/MA lags run in time order" begin
    rng = MersenneTwister(31)
    nA, nB = 25, 20
    tA = sort(500 .+ 60 .* rand(rng, nA)); tB = sort(500 .+ 60 .* rand(rng, nB))
    t = vcat(tA, tB); inst = vcat(fill(1, nA), fill(2, nB))
    n = length(t)
    @test !issorted(t)
    o = sortperm(t)
    rv = 3 .* randn(rng, n); err = 0.5 .+ rand(rng, n)

    @testset "$label" for (label, noise) in (
            ("MA(2), global", [MAModel(order = 2)]),
            ("MA(1), per instrument", [MAModel(order = 1, per_instrument = true)]),
            ("AR(1), global", [ARModel(order = 1)]),
            ("AR(1) + MA(1)", [ARModel(order = 1), MAModel(order = 1)]))
        p, d = _arma_params(t, inst, rv, err, noise)
        ps, ds = _arma_params(t[o], inst[o], rv[o], err[o], noise)
        th = _arma_theta(p); ths = _arma_theta(ps)

        ll = Nereus._rv_log_likelihood_core(th, d)
        @test isfinite(ll)
        @test ll ≈ Nereus._rv_log_likelihood_core(ths, ds) rtol = 1e-12

        # The transforms themselves: the sorted result, in the caller's order.
        for nm in noise
            f! = nm isa MAModel ? apply_ma! : apply_ar!
            x = randn(rng, n)
            xs = f!(x[o], t[o], inst[o], ths, nm)
            y = f!(copy(x), t, inst, th, nm)
            @test y[o] == xs
        end
    end
end
