# The workspace likelihoods (what pt_emcee and the other samplers draw with)
# and the allocating ones are one model, up to rounding, at every eccentricity
# the likelihood accepts, e < 1.
#
# They were not above e = 0.9999. `true_anomaly` clamped e to 0.9999, while
# the workspace photometry and the workspace RV of a fit with no noise models
# take cos f and sin f from E with the e they are given. Kepler's equation and
# r/a always used the given e, so above 0.9999 the allocating methods computed
# a true anomaly from another orbit than the rest of their model, and the two
# disagreed by up to ~10 nats in the RV and ~1e4 nats in the photometry.

using Test
using Nereus
using Random: MersenneTwister

const _HE_NRV = 30
const _HE_NPH = 6000
const _HE_P, _HE_TC = 3.11, 1.2

function _he_target(; rv_noise::Bool = false)
    rng = MersenneTwister(17)
    t_rv = sort!(60 .* rand(rng, _HE_NRV))
    rv = 30.0 .* sin.(2π .* (t_rv .- _HE_TC) ./ _HE_P) .+ 2.0 .* randn(rng, _HE_NRV)
    t_ph = collect(range(0.0, 20.0; length = _HE_NPH))
    ph = @. abs(mod(t_ph - _HE_TC + _HE_P / 2, _HE_P) - _HE_P / 2)
    flux = 1.0 .+ 3e-4 .* randn(rng, _HE_NPH); flux[ph .< 0.05] .-= 0.006
    return build_target(
        planets = (b = (P = UniformPrior(3.0, 3.2), Tc = UniformPrior(1.1, 1.3),
                        K = UniformPrior(5.0, 60.0), b = UniformPrior(0.0, 0.9),
                        rr = UniformPrior(0.02, 0.15), sesinw = UniformPrior(-1.0, 1.0),
                        secosw = UniformPrior(-1.0, 1.0)),),
        rv = (SIM = (data = (t = t_rv, rv = rv, rv_err = fill(2.0, _HE_NRV)),
                     sigma = LogUniformPrior(0.5, 10.0)),),
        phot = (TESS = (data = (t = t_ph, flux = flux, flux_err = fill(3e-4, _HE_NPH)),),),
        noise_models = rv_noise ? [ErrorScale()] : nothing,
        M_s = 1.0, R_s = 1.0)
end

@testset "workspace and allocating likelihoods agree up to e = 0.99999" begin
    tg = _he_target()
    params, data = tg.params, tg.data
    ix = params.layout.name_to_idx
    ws = Nereus.PTWorkspace(params, 1, length(params.config.noise_models);
                            n_obs = length(data.t_rv), n_phot = length(data.t_phot))
    th = Theta{Float64}(params)
    base = Dict("P_k1" => _HE_P, "Tc_k1" => _HE_TC, "K_k1" => 30.0, "b_k1" => 0.3,
                "rr_k1" => 0.075, "sigma_SIM" => 2.0)
    for (k, v) in base
        th.values[ix[k]] = v
    end
    worst_rv = 0.0; worst_ph = 0.0; n = 0
    for e in (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99993, 0.99995, 0.99998, 0.99999),
        ω in (-π / 2, -π / 2 + 1e-3, -π / 2 - 0.02, 0.0, 1.1, π / 2, -2.5)
        th.values[ix["sesinw_k1"]] = sqrt(e) * sin(ω)
        th.values[ix["secosw_k1"]] = sqrt(e) * cos(ω)
        e_got, _ = Nereus.planet_e_w(th, 1)
        @test e_got < 1
        rv_a = rv_log_likelihood(th, data)
        rv_w = rv_log_likelihood(th, data, ws)
        ph_a = transit_log_likelihood(th, data)
        ph_w = transit_log_likelihood(th, data, ws)
        @test isfinite(rv_a) && isfinite(ph_a)
        worst_rv = max(worst_rv, abs(rv_a - rv_w) / max(1.0, abs(rv_a)))
        worst_ph = max(worst_ph, abs(ph_a - ph_w) / max(1.0, abs(ph_a)))
        n += 1
    end
    @info "workspace vs allocating, max relative |Δ log L|, $n orbits" worst_rv worst_ph
    @test worst_rv < 1e-11
    @test worst_ph < 1e-11

    # With a noise model the RV takes another workspace route (`_refresh_kepler!`
    # and `true_anomaly`); it must give the same function too.
    tgn = _he_target(; rv_noise = true)
    pn, dn = tgn.params, tgn.data
    ixn = pn.layout.name_to_idx
    wsn = Nereus.PTWorkspace(pn, 1, length(pn.config.noise_models);
                             n_obs = length(dn.t_rv), n_phot = length(dn.t_phot))
    thn = Theta{Float64}(pn)
    for (k, v) in base
        thn.values[ixn[k]] = v
    end
    for nm in keys(ixn)
        startswith(nm, "errscale") && (thn.values[ixn[nm]] = 1.3)
    end
    for e in (0.9999, 0.99995, 0.99999), ω in (-π / 2, 0.4)
        thn.values[ixn["sesinw_k1"]] = sqrt(e) * sin(ω)
        thn.values[ixn["secosw_k1"]] = sqrt(e) * cos(ω)
        a = rv_log_likelihood(thn, dn)
        w = rv_log_likelihood(thn, dn, wsn)
        @test isfinite(a)
        @test abs(a - w) <= 1e-11 * max(1.0, abs(a))
    end
end
