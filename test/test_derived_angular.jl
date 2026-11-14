# Derived P, K and the orbit as an angle on the sky, for fits that do not
# sample them: P under :a_driven, K under :a_driven / :M_sec_driven, and the
# semi-major axis of the relative orbit and of the host's reflex orbit (mas,
# μas) for any planet whose block carries astrometry.

using Nereus, Test, MCMCChains, Random

const _DA_YEAR = Nereus.KEPLER_YEAR_DAYS

function _da_setup(mass::Symbol; planet_modes = [RVAS])
    instruments = InstrumentConfig(rv = ["X"])
    relast = RelAstromData(t = [55000.0, 56000.0, 57000.0],
                           ra_off = [10.0, 12.0, 8.0], dec_off = [-5.0, -3.0, -7.0],
                           ra_err = [1.0, 1.0, 1.0], dec_err = [1.0, 1.0, 1.0])
    hgca = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                    pmra = (0., 0., 0.), pmdec = (0., 0., 0.),
                    sigma_pmra = (0.5, 0.02, 0.03), sigma_pmdec = (0.4, 0.02, 0.03),
                    plx = 10.0, plx_err = 0.1, hip_id = 1)
    data = Data(t_rv = [55000.0, 55100.0], rv = [0.0, 0.0], rv_err = [1.0, 1.0],
                relastrom = relast, hgca = hgca)
    pri = mass === :K_driven ?
          Dict("P_k1" => LogUniformPrior(100.0, 20000.0),
               "K_k1" => LogUniformPrior(1.0, 500.0)) :
          mass === :a_driven ? Dict("a_k1" => LogUniformPrior(0.5, 1000.0)) :
                               Dict{String, PriorSpec}()
    params = Params(max_kplanet = 1, planet_modes = planet_modes,
                    instruments = instruments, data = data, stability = :none,
                    M_s = 1.0, priors = pri,
                    parametrization = ParametrizationConfig(time = :Mo, mass = mass))
    return params
end

# A Chains object over the unfrozen names, each draw filled by `gen(name, i)`.
function _da_chains(params, n, gen)
    nm = params.layout.unfrozen_names
    arr = zeros(n, length(nm) + 1, 1)
    for i in 1:n, (j, x) in enumerate(nm)
        arr[i, j, 1] = gen(x, i)
    end
    return Chains(arr, vcat(Symbol.(nm), :lp))
end

function _da_gen(mass)
    return function (x, i)
        x == "a_k1" && return 3.0 + 0.5i
        x == "P_k1" && return 3000.0 + 200.0i
        x == "K_k1" && return 20.0 + 3.0i
        x == "M_sec_k1" && return 0.02 + 0.004i
        x == "sesinw_k1" && return 0.1
        x == "secosw_k1" && return 0.2
        x == "inc_k1" && return 1.0 + 0.1i
        x == "Omega_k1" && return 2.0
        x == "Mo_k1" && return 1.0
        x == "plx" && return 10.0 + 0.5i
        x == "M_pri" && return 0.9 + 0.05i
        x == "n_p" && return 1.0
        return 0.5
    end
end

function _da_theta(params, chains, i)
    th = Theta{Float64}(params)
    for nm in params.layout.unfrozen_names
        set_param!(th, nm, Nereus._draws(chains, Symbol(nm))[i])
    end
    return th
end

@testset "derived P, K and angular orbit size" begin
    n = 6

    @testset ":a_driven astrometric planet" begin
        params = _da_setup(:a_driven)
        chains = _da_chains(params, n, _da_gen(:a_driven))
        d = only(compute_derived(chains, params; M_s = 1.0)).values
        @test haskey(d, "P_days") && haskey(d, "K") &&
              haskey(d, "a_mas") && haskey(d, "alpha_star_uas")
        for i in 1:n
            th = _da_theta(params, chains, i)
            @test d["P_days"][i] ≈ planet_P(th, 1) rtol = 1e-12
            @test d["K"][i] ≈ planet_K(th, 1) rtol = 1e-12
            a = th.values[params.layout.name_to_idx["a_k1"]]
            ms = th.values[params.layout.name_to_idx["M_sec_k1"]]
            mp = astrom_M_pri(th); plx = astrom_plx(th)
            @test d["a_mas"][i] ≈ a * plx rtol = 1e-12
            @test d["alpha_star_uas"][i] ≈ a * ms / (mp + ms) * plx * 1000 rtol = 1e-12
            # P from Kepler III with the per-draw masses, in the Kepler year
            @test d["P_days"][i] ≈ sqrt(a^3 / (mp + ms)) * _DA_YEAR rtol = 1e-12
        end
        # M_pri and plx really vary per draw (not stale from the first draw)
        @test length(unique(d["a_mas"])) == n
    end

    @testset ":K_driven astrometric planet" begin
        params = _da_setup(:K_driven)
        chains = _da_chains(params, n, _da_gen(:K_driven))
        d = only(compute_derived(chains, params; M_s = 1.0)).values
        @test !haskey(d, "K") && !haskey(d, "P_days")
        for i in 1:n
            th = _da_theta(params, chains, i)
            ms = planet_M_sec(th, 1)
            a = planet_a(th, 1)
            mp = astrom_M_pri(th); plx = astrom_plx(th)
            @test ms > 0
            @test d["alpha_star_uas"][i] ≈ a * ms / (mp + ms) * plx * 1000 rtol = 1e-12
            @test d["a_mas"][i] ≈ a * plx rtol = 1e-12
        end
    end

    @testset ":M_sec_driven RV-only planet gets K, no angles" begin
        params = _da_setup(:M_sec_driven; planet_modes = [RV_ONLY])
        chains = _da_chains(params, n, (x, i) -> x == "P_k1" ? 3000.0 + 100i :
                            x == "M_sec_k1" ? 0.002 * i : _da_gen(:M_sec_driven)(x, i))
        d = only(compute_derived(chains, params; M_s = 1.0)).values
        @test haskey(d, "K") && !haskey(d, "a_mas") && !haskey(d, "alpha_star_uas")
        for i in 1:n
            @test d["K"][i] ≈ planet_K(_da_theta(params, chains, i), 1) rtol = 1e-12
        end
    end

    @testset "a throwing or non-finite draw is NaN" begin
        params = _da_setup(:a_driven)
        chains = _da_chains(params, n, (x, i) -> x == "plx" && i == 2 ? NaN :
                            _da_gen(:a_driven)(x, i))
        d = only(compute_derived(chains, params; M_s = 1.0)).values
        @test isnan(d["a_mas"][2]) && isfinite(d["a_mas"][1])
    end

    @testset "science_derived rows and units" begin
        params = _da_setup(:a_driven)
        chains = _da_chains(params, 50, (x, i) -> _da_gen(:a_driven)(x, mod1(i, 6)))
        entries, _ = Nereus.science_derived(chains, params; M_s = 1.0)
        e = Dict(entries)
        for (key, unit) in ("P_days_k1" => "d", "P_yr_k1" => "yr", "K_k1" => "m/s",
                            "a_mas_k1" => "mas", "alpha_star_uas_k1" => "μas")
            @test haskey(e, key)
            @test e[key]["unit"] == unit
        end
        @test e["P_yr_k1"]["value"] ≈ e["P_days_k1"]["value"] / 365.25 rtol = 1e-9
        @test Nereus._texsym("alpha_star_uas_k1") == "\${\\alpha_\\star}_{1}\$"
        @test Nereus._texsym("P_days_k1") == "\${P}_{1}\$"
        @test Nereus._texunit("μas") == "\$\\mu\$as"
        @test Nereus._texunit("mas") == "mas"

        # sampled P: P_yr still reported, no P_days
        params = _da_setup(:K_driven)
        chains = _da_chains(params, 50, (x, i) -> _da_gen(:K_driven)(x, mod1(i, 6)))
        e = Dict(first(Nereus.science_derived(chains, params; M_s = 1.0)))
        @test haskey(e, "P_yr_k1") && !haskey(e, "P_days_k1") && !haskey(e, "K_k1")
        @test haskey(e, "alpha_star_uas_k1")
    end
end

# No m << M approximation in the derived masses and semi-major axes.
@testset "derived masses and a use the exact mass function" begin
    MS_MJ = Nereus.M_SUN_G / Nereus.M_JUP_G
    # K that a companion of mass m (M_sun) at sin i = 1 induces on a star M
    K_of(m, M, P, e) = cbrt(m^3 / (M + m)^2 /
                            (Nereus._F_M_FACTOR * P * (1 - e^2)^1.5))

    # The M_p << M closed form, which Nereus no longer contains: kept here only
    # to size the error the exact inversion removes. In M_sun.
    lowmass(M, K, P, e) = cbrt(Nereus.mass_function(K, P, e) * M^2)

    @testset "brown dwarf about an M dwarf: the closed form is ~9% off" begin
        M, m, P, e = 0.5, 0.08, 400.0, 0.2
        K = K_of(m, M, P, e)
        @test msini(M, K, P, e) / MS_MJ ≈ m rtol = 1e-10
        @test abs(lowmass(M, K, P, e) - m) / m > 0.09
        # at an inclination the true mass is the same inversion at sin i
        inc = 40.0
        @test planet_mass(M, K, P, e, 90.0) ≈ msini(M, K, P, e) rtol = 1e-12
        @test planet_mass(M, K, P, e, inc) > msini(M, K, P, e)
        # a: total mass in Kepler III
        @test semimajor_axis_au(P, M, m) ≈ semimajor_axis_au(P, M) * cbrt((M + m) / M) rtol = 1e-12
    end

    @testset "a Jupiter about a Sun-like star moves by < 0.1%" begin
        M, m, P, e = 1.0, 9.5e-4, 4332.0, 0.05
        K = K_of(m, M, P, e)
        @test msini(M, K, P, e) / MS_MJ ≈ m rtol = 1e-10
        @test abs(msini(M, K, P, e) / MS_MJ / lowmass(M, K, P, e) - 1) < 1e-3
        @test abs(semimajor_axis_au(P, M, m) / semimajor_axis_au(P, M) - 1) < 1e-3
    end

    @testset "round trip K -> msini -> K through the model's planet_K" begin
        params = _da_setup(:M_sec_driven; planet_modes = [RV_ONLY])
        th = Theta{Float64}(params)
        for (nm, v) in ("n_p" => 1.0, "P_k1" => 400.0, "M_sec_k1" => 0.08,
                        "sesinw_k1" => 0.3, "secosw_k1" => 0.2, "Mo_k1" => 1.0)
            set_param!(th, nm, v)
        end
        e = sum(abs2, (0.3, 0.2))
        K = planet_K(th, 1)
        M = Nereus.astrom_M_pri(th)
        @test msini(M, K, 400.0, e) / MS_MJ ≈ 0.08 rtol = 1e-10
        @test K_of(msini(M, K, 400.0, e) / MS_MJ, M, 400.0, e) ≈ K rtol = 1e-10
    end

    @testset "compute_derived: RV-only K_driven table" begin
        params = _da_setup(:K_driven; planet_modes = [RV_ONLY])
        chains = _da_chains(params, 6, (x, i) -> x == "P_k1" ? 300.0 + 40i :
                            x == "K_k1" ? 3000.0 + 100i : _da_gen(:K_driven)(x, i))
        ms = 0.5
        d = only(compute_derived(chains, params; M_s = ms)).values
        for i in 1:6
            P = Nereus._draws(chains, :P_k1)[i]; K = Nereus._draws(chains, :K_k1)[i]
            e = 0.1^2 + 0.2^2
            m = msec_from_K(K, P, e, 1.0, ms)
            @test d["msini_jup"][i] ≈ m * MS_MJ rtol = 1e-12
            @test d["a_au"][i] ≈ semimajor_axis_au(P, ms, m) rtol = 1e-12
            @test d["a_au"][i] > semimajor_axis_au(P, ms)
        end
    end

    @testset "compute_derived: astrometric K_driven uses the model's mass and a" begin
        params = _da_setup(:K_driven)
        chains = _da_chains(params, 6, _da_gen(:K_driven))
        d = only(compute_derived(chains, params; M_s = 1.0)).values
        for i in 1:6
            th = _da_theta(params, chains, i)
            @test d["mass_jup"][i] ≈ planet_M_sec(th, 1) * MS_MJ rtol = 1e-12
            @test d["a_au"][i] ≈ planet_a(th, 1) rtol = 1e-12
            @test d["a_au"][i] * astrom_plx(th) ≈ d["a_mas"][i] rtol = 1e-12
            @test d["msini_jup"][i] <= d["mass_jup"][i] * (1 + 1e-9)
        end
    end
end
