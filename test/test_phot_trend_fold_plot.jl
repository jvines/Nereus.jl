# A phase fold of a light curve that carries a baseline trend.
#
# The fold evaluates its model and credible bands on t0 + phase × P, at the
# reference epoch, not at the cadences. A trend evaluated there is extrapolated
# from the instrument's own night to an epoch that can be hundreds of days away:
# with each trend in its own group's time (x = (t − t̄) / (span / 2)) a single
# night's linear term reaches x ~ 10⁴ and the folded model lands at flux ≈ −7.
# The fold must show the transit with the trend taken out: the model and bands
# without the trend, and the folded cadences detrended by the fitted trend.
using Test
using Nereus
using MCMCChains
using Statistics: mean

function _trend_fold_fixture()
    tS = collect(0.0:0.004:30.0)                        # survey, flat
    tN = collect(range(400.0, 400.25; length = 80))     # one night, far from t0
    t = vcat(tS, tN)
    inst = vcat(fill(1, length(tS)), fill(2, length(tN)))
    data = Data(; t_phot = t, flux = ones(length(t)), flux_err = fill(1e-3, length(t)),
                  phot_inst = inst)
    params = Params(; parametrization = ParametrizationConfig(time = :Tc), max_kplanet = 1,
                      planet_modes = [PM_ONLY], instruments = InstrumentConfig(pm = ["SURVEY", "NIGHT"]),
                      data = data, M_s = 1.0, R_s = 1.0, stability = :none,
                      phot_trend_order = Dict("NIGHT" => 1))
    vals = Dict("P_k1" => 2.83, "sesinw_k1" => 0.0, "secosw_k1" => 0.0, "Tc_k1" => 1.0,
                "b_k1" => 0.3, "rr_k1" => 0.1,
                "offset_SURVEY" => 0.0, "jitter_SURVEY" => 1e-4, "q1_SURVEY" => 0.4, "q2_SURVEY" => 0.3,
                "offset_NIGHT" => 0.0, "jitter_NIGHT" => 1e-4, "q1_NIGHT" => 0.4, "q2_NIGHT" => 0.3,
                "phot_c1_NIGHT" => 0.003, "rho_circ_k1" => 1.0, "eta_k1" => 0.0)
    theta = Theta{Float64}(params)
    nm = params.layout.unfrozen_names
    for (k, v) in vals
        k in nm && set_param!(theta, k, v)
    end
    for x in nm
        haskey(vals, x) || (vals[x] = Nereus.get_param(theta, x))
    end
    data.flux .= Nereus.phot_predictions(theta, data)[1]
    cols = vcat(Symbol.(nm), [:lp])
    arr = zeros(8, length(cols), 1)
    for i in 1:8
        arr[i, 1:length(nm), 1] = [vals[x] * (1 + 1e-6 * i) for x in nm]
        arr[i, end, 1] = -0.1 * i
    end
    chains = Nereus.with_density_columns(Chains(arr, cols), params)
    return params, data, theta, chains
end

@testset "phase fold of a trended light curve" begin
    params, data, theta, chains = _trend_fold_fixture()
    night = findall(==(2), data.phot_inst)
    tN = data.t_phot[night]

    # at the cadences the full model still carries the trend, as the likelihood does
    m_full = Nereus.compute_transit_model_on_grid(theta, data, tN, 2)
    @test m_full ≈ Nereus.phot_predictions(theta, data)[1][night] rtol = 1e-12
    # without it: the transit model alone, 1 + offset out of transit
    m_flat = Nereus.compute_transit_model_on_grid(theta, data, tN, 2; trend = false)
    @test maximum(m_flat) ≈ 1.0 atol = 1e-12
    @test maximum(abs.(m_full .- m_flat)) > 0.002        # the trend is there to remove

    # the folded band, at the reference epoch far from the night: a transit of
    # depth ~ rr², not the trend extrapolated over 400 d
    ph = collect(range(-0.05, 0.05; length = 201))
    ci = Nereus.compute_transit_ci_bands(chains, params, data, ph, 2; planet = 1,
                                         fold_phase = ph, n_draws = 8, trend = false)
    @test all(0.98 .< ci.median .<= 1.0 + 1e-9)
    @test minimum(ci.median) < 0.995                     # the dip is in the band
    @test maximum(ci.median) ≈ 1.0 atol = 1e-6

    # the fold renders, and its panel is framed on the transit, not on −7
    out = mktempdir()
    @test plot_pm_phasefold(chains, params, data; planet = 1, output = out) === nothing
    png = joinpath(out, "models", "Transit_phasefold_K1_NIGHT.png")
    @test isfile(png)
    # 3:2, wider than tall (PNG header: width, height as big-endian UInt32 at bytes 17-24)
    w, h = open(png) do io; skip(io, 16); (ntoh(read(io, UInt32)), ntoh(read(io, UInt32))); end
    @test w / h ≈ 1.5
    # the folded cadences are detrended: they sit on the transit model, not on the slope
    flux, model = Nereus._pm_fold_data(theta, data, data.phot_inst .== 2, 2, 1)
    @test model ≈ m_flat rtol = 1e-12
    @test maximum(abs.(flux .- model)) < 1e-9          # noiseless data: residuals vanish
    @test maximum(abs.(flux .- data.flux[night])) > 0.002
end
