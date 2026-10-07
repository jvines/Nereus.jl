# A photometry-channel noise model must reach the photometry likelihood
# whether or not a transit is in the data.
#
# The transit likelihood scores the light curve with a white-noise sum unless
# a phot noise model asks for the serial path through `_eval_channel_likelihood`.
# It used to ask only for AR/MA and GPs, so an additive covariance
# (HarmonicBlock -- the external-comb pulsator model is a photometry model --
# or NightlyOffset) or a Student-t likelihood on :phot was silently dropped
# whenever a planet transited, and the fit scored white noise instead. The
# transit-free path already routed every phot noise model; now both do.
using Nereus, Test, Random
using Nereus: HarmonicBlock, NightlyOffset, StudentT, NoiseModel

function _transit_target(noise; seed = 7)
    rng = MersenneTwister(seed)
    n = 1500
    t = collect(range(0.0, 12.0; length = n))
    P0, T0, dur, dep = 3.11, 1.2, 0.11, 0.004
    ph = @. abs(mod(t - T0 + P0 / 2, P0) - P0 / 2)
    flux = 1.0 .+ 3e-4 .* randn(rng, n) .+ 4e-4 .* sin.(2π .* 1.5 .* t)
    flux[ph .< dur / 2] .-= dep
    ferr = fill(3e-4, n)
    tg = build_target(
        planets = (b = (P  = UniformPrior(P0 - 0.05, P0 + 0.05),
                        Tc = UniformPrior(T0 - 0.05, T0 + 0.05),
                        b  = UniformPrior(0.0, 0.9),
                        rr = UniformPrior(0.01, 0.2),
                        sesinw = UniformPrior(-1.0, 1.0),
                        secosw = UniformPrior(-1.0, 1.0)),),
        phot = (TESS = (data = (t = t, flux = flux, flux_err = ferr),),),
        noise_models = NoiseModel[noise...],
        M_s = 1.0, R_s = 1.0)
    th = Nereus.Theta{Float64}(tg.params)
    truth = Dict("P_k1" => P0, "Tc_k1" => T0, "b_k1" => 0.3, "rr_k1" => 0.063,
                 "harm_amp_TESS_phot" => 5e-4, "night_sigma_TESS_phot" => 2e-4,
                 "studentt_nu" => 4.0)
    for nm in tg.params.layout.unfrozen_names
        Nereus.set_param!(th, nm, get(truth, nm, 0.0))
    end
    return tg, th
end

# What the photometry likelihood must be: the channel evaluator on the
# residuals of the photometric model.
function _expected(th, data)
    preds, vars = Nereus.phot_predictions(th, data)
    r = data.flux .- preds
    return Nereus._eval_channel_likelihood(th, r, vars, data.t_phot, data.phot_inst,
                                           :phot, 2π)
end

function _white(th, data)
    preds, vars = Nereus.phot_predictions(th, data)
    r = data.flux .- preds
    return sum(@. -0.5 * (log(2π * vars) + r^2 / vars))
end

@testset "phot noise models reach the transit likelihood: $label" for (label, noise) in (
        ("HarmonicBlock, external comb", [HarmonicBlock(channel = :phot, freqs = [1.5, 3.0])]),
        ("NightlyOffset", [NightlyOffset(channel = :phot, gap = 0.5)]),
        ("Student-t", [StudentT(channel = :phot)]))
    tg, th = _transit_target(noise)
    d = tg.data
    exp_ll = _expected(th, d)
    @test isfinite(exp_ll)
    @test abs(exp_ll - _white(th, d)) > 1.0          # the model matters here

    ll = Nereus.transit_log_likelihood(th, d)
    @test ll ≈ exp_ll rtol = 1e-10

    ws = Nereus.PTWorkspace(tg.params, tg.params.config.max_kplanet,
                            length(tg.params.config.noise_models);
                            n_obs = length(d.t_rv), n_phot = length(d.t_phot))
    @test Nereus.transit_log_likelihood(th, d, ws) ≈ exp_ll rtol = 1e-10
end

# A light curve fitted with no transiting planet (a noise-only fit, as
# detrend_gp builds) has no limb-darkening or dilution slots. phot_predictions
# read them before its no-transit fast path and crashed with a BoundsError, so
# every consumer of the photometric model -- LOO, PPC, residual plots -- failed
# on such a fit although its likelihood was fine.
@testset "phot_predictions with no transiting planet" begin
    rng = MersenneTwister(5)
    n = 60
    t = sort!(10 .* rand(rng, n))
    flux = 1 .+ 5e-4 .* randn(rng, n)
    d = Nereus.Data(; t_phot = t, flux = flux, flux_err = fill(5e-4, n),
                    phot_inst = ones(Int, n))
    p = Nereus.Params(; max_kplanet = 0, planet_modes = Nereus.PlanetDataSources[],
                      instruments = Nereus.InstrumentConfig(String[], ["TESS"]), data = d,
                      M_s = 1.0, R_s = 1.0)
    th = Nereus.Theta{Float64}(p)
    Nereus.set_param!(th, "offset_TESS", 2e-4); Nereus.set_param!(th, "jitter_TESS", 1e-4)
    preds, vars = Nereus.phot_predictions(th, d)
    @test length(preds) == n
    @test all(==(vars[1]), vars)
    # The model the likelihood scores.
    @test sum(@. -0.5 * (log(2π * vars) + (flux - preds)^2 / vars)) ≈
          Nereus.transit_log_likelihood(th, d) rtol = 1e-12
end
