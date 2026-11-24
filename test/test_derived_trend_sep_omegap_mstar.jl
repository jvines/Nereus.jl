# omega_p, RV-trend companion bounds, sky separation/PA at an epoch, and the
# stellar mass and log g from rho_star -- the derived rows added together.

using Nereus, Test, MCMCChains, Random, Statistics

function _dt_chains(params, n, gen)
    nm = params.layout.unfrozen_names
    arr = zeros(n, length(nm) + 1, 1)
    for i in 1:n, (j, x) in enumerate(nm)
        arr[i, j, 1] = gen(x, i)
    end
    return Chains(arr, vcat(Symbol.(nm), :lp))
end

function _dt_theta(params, chains, i)
    th = Theta{Float64}(params)
    for nm in params.layout.unfrozen_names
        set_param!(th, nm, Nereus._draws(chains, Symbol(nm))[i])
    end
    return th
end

@testset "omega_p" begin
    data = Data(t_rv = [0.0, 100.0, 200.0], rv = [0.0, 1.0, 0.0], rv_err = [1.0, 1.0, 1.0])
    params = Params(max_kplanet = 1, planet_modes = [RV_ONLY],
                    instruments = InstrumentConfig(rv = ["X"]), data = data,
                    stability = :none, M_s = 1.0,
                    priors = Dict("P_k1" => LogUniformPrior(10.0, 100.0),
                                  "K_k1" => LogUniformPrior(1.0, 100.0)))
    for (w_deg, expect) in ((100.0, 280.0), (-170.0, 10.0), (0.0, 180.0), (-10.0, 170.0))
        w = deg2rad(w_deg)
        gen = (x, i) -> x == "sesinw_k1" ? 0.5 * sin(w) : x == "secosw_k1" ? 0.5 * cos(w) :
                        x == "P_k1" ? 50.0 : x == "K_k1" ? 10.0 : 0.5
        chains = _dt_chains(params, 5, gen)
        d = only(compute_derived(chains, params; M_s = 1.0)).values
        @test median(d["omega_p_deg"]) ≈ expect atol = 1e-8
        @test 0 <= median(d["omega_p_deg"]) < 360
        # always omega + 180 (mod 360)
        dd = mod(d["omega_p_deg"][1] - d["omega_deg"][1] - 180, 360)
        @test min(dd, 360 - dd) < 1e-8
    end
end

@testset "RV-trend bound" begin
    t = [0.0, 1000.0, 2000.0, 3650.0]
    data = Data(t_rv = t, rv = zeros(4), rv_err = ones(4))
    params = Params(max_kplanet = 1, planet_modes = [RV_ONLY],
                    instruments = InstrumentConfig(rv = ["X"]), data = data,
                    stability = :none, M_s = 1.0, trend_order = 1,
                    priors = Dict("P_k1" => LogUniformPrior(10.0, 100.0),
                                  "K_k1" => LogUniformPrior(1.0, 100.0)))
    gen = (x, i) -> x == "dvdt" ? -0.002 * i : x == "P_k1" ? 50.0 :
                    x == "K_k1" ? 10.0 : 0.3
    chains = _dt_chains(params, 4, gen)
    T = 3650.0
    dv = abs.(-0.002 .* (1:4))                       # m/s/day

    s = compute_derived_system(chains, params; M_s = 1.0, data = data)
    @test all(==(T), s["trend_P_min"])
    @test s["trend_K_min"] ≈ dv .* T ./ (2π) rtol = 1e-12
    # hand calc, draw 1: K = 0.002 * 3650 / 2π = 1.16183 m/s
    @test s["trend_K_min"][1] ≈ 1.1618 rtol = 1e-4
    # exact mass function at P = T, circular: matches the exported msini
    @test s["trend_Msini_min"] ≈ [msini(1.0, K, T, 0.0) for K in s["trend_K_min"]] rtol = 1e-12
    # small-companion closed form: M sin i = K sqrt(1-e^2) (P/2πG)^(1/3) M_s^(2/3)
    G = Nereus.G_CGS
    msini_cf = 1.16183 * (T * 86400 / (2π * G))^(1/3) * (Nereus.M_SUN_G)^(2/3) / Nereus.M_JUP_G * 1e2^(-1)
    @test s["trend_Msini_min"][1] ≈ msini_cf rtol = 2e-3
    @test !haskey(s, "trend_Mc_min_torres")

    # Torres (1999): 0.0145 M_J (d/pc)^2 (rho/arcsec)^2 |dv/dt|[m/s/yr]
    s2 = compute_derived_system(chains, params; M_s = 1.0, data = data,
                                distance_pc = 20.0, trend_sep_arcsec = 2.0)
    @test s2["trend_Mc_min_torres"][1] ≈ 0.0145 * 20.0^2 * 2.0^2 * 0.002 * 365.25 rtol = 1e-12
    @test s2["trend_Mc_min_torres"][1] ≈ 0.0145 * 1600 * 0.73050 rtol = 1e-9

    # no trend in the model -> no rows
    p0 = Params(max_kplanet = 1, planet_modes = [RV_ONLY],
                instruments = InstrumentConfig(rv = ["X"]), data = data,
                stability = :none, M_s = 1.0,
                priors = Dict("P_k1" => LogUniformPrior(10.0, 100.0),
                              "K_k1" => LogUniformPrior(1.0, 100.0)))
    c0 = _dt_chains(p0, 3, (x, i) -> x == "P_k1" ? 50.0 : x == "K_k1" ? 10.0 : 0.3)
    @test isempty(compute_derived_system(c0, p0; M_s = 1.0, data = data))
end

@testset "stellar mass and log g from rho_star" begin
    data = Data(t_rv = [0.0, 100.0], rv = [0.0, 0.0], rv_err = [1.0, 1.0])
    params = Params(max_kplanet = 1, planet_modes = [RV_ONLY],
                    instruments = InstrumentConfig(rv = ["X"]), data = data,
                    stability = :none, M_s = 1.0,
                    priors = Dict("P_k1" => LogUniformPrior(10.0, 100.0),
                                  "K_k1" => LogUniformPrior(1.0, 100.0)))
    chains = _dt_chains(params, 3, (x, i) -> x == "P_k1" ? 50.0 : x == "K_k1" ? 10.0 : 0.3)
    s = compute_derived_system(chains, params; R_s = 1.0,
                               rho_star = fill(Nereus.RHO_SUN_CGS, 3))
    @test all(≈(1.0; rtol = 1e-12), s["M_s_rho"])
    @test all(≈(4.438; atol = 2e-3), s["logg_s"])
    # a 0.5 rho_sun, 1.2 R_sun star: M = 0.5 * 1.2^3 = 0.864
    s2 = compute_derived_system(chains, params; R_s = 1.2,
                                rho_star = fill(0.5 * Nereus.RHO_SUN_CGS, 3))
    @test s2["M_s_rho"][1] ≈ 0.864 rtol = 1e-12
    @test s2["logg_s"][1] ≈ log10(Nereus.G_CGS * 0.864 * Nereus.M_SUN_G / (1.2 * Nereus.R_SUN_CM)^2) rtol = 1e-12
    # no R_s -> no rows; no rho -> no rows
    @test !haskey(compute_derived_system(chains, params; rho_star = fill(1.0, 3)), "M_s_rho")
    @test !haskey(compute_derived_system(chains, params; R_s = 1.0), "logg_s")
    # first transiting planet's rho_star_transit
    dp = [Nereus.DerivedParams("planet_1", Dict("ecc" => [0.1])),
          Nereus.DerivedParams("planet_2", Dict("rho_star_transit" => [1.4, 1.5]))]
    @test Nereus._first_rho_transit(dp) == [1.4, 1.5]
end

@testset "separation and PA at an epoch" begin
    relast = RelAstromData(t = [55000.0, 56000.0, 57000.0],
                           ra_off = [10.0, 12.0, 8.0], dec_off = [-5.0, -3.0, -7.0],
                           ra_err = [1.0, 1.0, 1.0], dec_err = [1.0, 1.0, 1.0])
    hgca = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                    pmra = (0., 0., 0.), pmdec = (0., 0., 0.),
                    sigma_pmra = (0.5, 0.02, 0.03), sigma_pmdec = (0.4, 0.02, 0.03),
                    plx = 10.0, plx_err = 0.1, hip_id = 1)
    data = Data(t_rv = [55000.0, 55100.0], rv = [0.0, 0.0], rv_err = [1.0, 1.0],
                relastrom = relast, hgca = hgca)
    params = Params(max_kplanet = 1, planet_modes = [RVAS],
                    instruments = InstrumentConfig(rv = ["X"]), data = data,
                    stability = :none, M_s = 1.0,
                    priors = Dict("P_k1" => LogUniformPrior(100.0, 20000.0),
                                  "K_k1" => LogUniformPrior(1.0, 500.0)),
                    parametrization = ParametrizationConfig(time = :Mo, mass = :K_driven))
    gen = (x, i) -> x == "P_k1" ? 3000.0 + 200.0i : x == "K_k1" ? 20.0 + 3.0i :
                    x == "sesinw_k1" ? 0.1 : x == "secosw_k1" ? 0.2 :
                    x == "inc_k1" ? 1.0 + 0.1i : x == "Omega_k1" ? 2.0 :
                    x == "Mo_k1" ? 1.0 : x == "plx" ? 10.0 + 0.5i :
                    x == "M_pri" ? 0.9 + 0.05i : x == "n_p" ? 1.0 : 0.5
    n = 5
    chains = _dt_chains(params, n, gen)

    # default epoch: the data's reference epoch
    d0 = only(compute_derived(chains, params; M_s = 1.0, t_ref = data.t_ref)).values
    d1 = only(compute_derived(chains, params; M_s = 1.0, t_ref = data.t_ref,
                              separation_epoch = 56500.0)).values
    @test haskey(d0, "sep_mas") && haskey(d0, "pa_deg")
    for (d, t_ep) in ((d0, data.t_ref), (d1, 56500.0)), i in 1:n
        th = _dt_theta(params, chains, i)
        orb, _ = Nereus._planet_orbit(th, 1, astrom_M_pri(th), astrom_plx(th), data.t_ref)
        dra, ddec = Nereus.relastrom_offset(orb, t_ep)
        @test d["sep_mas"][i] ≈ hypot(dra, ddec) rtol = 1e-10
        @test mod(d["pa_deg"][i], 360) ≈ mod(rad2deg(atan(dra, ddec)), 360) atol = 1e-8
    end
    @test d0["sep_mas"] != d1["sep_mas"]
    # bounded by the apocentre: a(1+e) (mas)
    @test all(d1["sep_mas"] .<= d1["a_mas"] .* 1.5)

    # no t_ref (and no epoch) -> skipped silently
    @test !haskey(only(compute_derived(chains, params; M_s = 1.0)).values, "sep_mas")
end
