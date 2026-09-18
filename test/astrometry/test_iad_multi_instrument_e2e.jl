# End-to-end: a mock two-mission source, from FILE to recovered mass.
#
# The unit tests next door exercise the design algebra directly. This one
# goes through the real path a user takes — write a van Leeuwen 2007
# residual record, parse it with the production parser, merge it with a
# Gaia DR4-style full-abscissa set, and profile the likelihood in companion
# mass — because that path has three places to get the conventions wrong
# (the header's catalogue solution, the O-C vs full abscissa distinction,
# and the per-mission frame offset) and none of them are visible from a
# hand-built `IADData`.

using Random: MersenneTwister
using Printf: @printf, @sprintf

@testset "astrometry — multi-instrument end to end" begin
    # ------------------------------------------------------------ truth
    M_pri, M_sec = 1.05, 0.045        # M_sun (~47 M_Jup brown dwarf)
    P_yr = 42.0; P_d = P_yr * 365.25
    e, ω, Ω = 0.35, 0.9, 2.1
    inc = deg2rad(63.0)
    plx_cat = 24.0                    # mas
    pmra_cat, pmdec_cat = 150.0, -95.0
    tp = jyear_to_mjd(1985.0)
    orb = Nereus.build_orbit(P_d, e, ω, Ω, inc, M_pri, M_sec, tp, plx_cat)

    HIP_EPOCH, DR4_EPOCH = 1991.25, 2017.5
    rng = MersenneTwister(20260914)

    # ------------------------------- Hipparcos: a real-format residual record
    n_h = 52; σ_h = 2.2
    jyr_h = collect(range(1990.05, 1993.15, length = n_h))
    t_h   = jyear_to_mjd.(jyr_h)
    ψ_h   = 2π .* rand(rng, n_h)
    epoch_h = jyr_h .- HIP_EPOCH
    parf_h  = sin.(2π .* (jyr_h .- HIP_EPOCH) .- ψ_h)
    res_h = Vector{Float64}(undef, n_h)
    for j in 1:n_h
        Δra, Δdec = Nereus.star_reflex_offset(orb, t_h[j], M_sec)
        res_h[j] = Nereus.along_scan_projection(Δra, Δdec, ψ_h[j]) + σ_h * randn(rng)
    end

    hip_path = tempname() * ".d"
    open(hip_path, "w") do io
        println(io, "# This file contains residual records, extracted from the Hipparcos 2")
        println(io, "# Interactive Data Access Tool (2014). For more information, see:")
        println(io, "# https://www.cosmos.esa.int/web/hipparcos/interactive-data-access")
        println(io, "# https://www.cosmos.esa.int/web/hipparcos/catalogues")
        println(io, "#")
        println(io, "# HIP    MCE    NRES NC isol_n SCE  F2     F1")
        @printf(io, "# %6d %6d %3d  %1d  %3d    %1d    %5.2f  %2d \n",
                99999, 99999, n_h, 1, 5, 0, 1.10, 0)
        println(io, "# Hp      B-V    VarAnn NOB NR")
        @printf(io, "# %6.4f  %5.3f  %1d      %3d %2d \n", 7.42, 0.64, 0, n_h, 0)
        println(io, "# RAdeg        DEdeg        Plx      pm_RA    pm_DE    e_RA   e_DE   e_Plx  e_pmRA e_pmDE dpmRA  dpmDE  e_dpmRA  e_dpmDE  ddpmRA  ddpmDE  e_ddpmRA  e_ddpmDE  upsRA   upsDE   e_upsRA  e_upsDE  var   ")
        @printf(io, "# %.8f  %.8f   %.2f    %.2f   %.2f  %.2f   %.2f   %.2f   %.2f   %.2f   ---    ---    ---      ---      ---     ---     ---       ---       ---     ---     ---      ---      ---   \n",
                132.11223344, -17.55667788, plx_cat, pmra_cat, pmdec_cat,
                0.71, 0.52, 0.83, 0.79, 0.58)
        println(io, "#")
        println(io, "# IORB   EPOCH    PARF    CPSI    SPSI     RES   SRES")
        for j in 1:n_h
            @printf(io, "  %4d %7.4f %7.4f %7.4f %7.4f %7.2f %6.2f\n",
                    100 + j, epoch_h[j], parf_h[j],
                    sin(ψ_h[j]), cos(ψ_h[j]), res_h[j], σ_h)
        end
    end
    hip = Nereus._parse_van_leeuwen_iad(hip_path)
    rm(hip_path)

    @testset "parser reads the catalogue solution off header line 11" begin
        @test n_iad(hip) == n_h
        @test hip.abscissa_kind == [:residual]
        # (Δα₀, Δδ₀) are zero BY CONSTRUCTION: the residuals are measured
        # from (RAdeg, DEdeg), so that position IS the tangent-plane origin.
        # Putting the header's absolute degrees here would inject ~1e9 mas.
        @test hip.ref_params[1] == (0.0, 0.0, plx_cat, pmra_cat, pmdec_cat)
        @test n_iad_inst(hip) == 1
    end

    # ------------------------------------- Gaia DR4: full abscissae, own frame
    n_g = 210; σ_g = 0.12
    jyr_g = collect(range(2014.60, 2017.45, length = n_g))
    t_g  = jyear_to_mjd.(jyr_g)
    ψ_g  = 2π .* rand(rng, n_g)
    pm_g = jyr_g .- DR4_EPOCH
    parf_g = sin.(2π .* (jyr_g .- DR4_EPOCH) .- ψ_g)
    zp_g = (3.5, -2.0)                     # a genuine inter-mission frame offset
    absc_g = Vector{Float64}(undef, n_g)
    for j in 1:n_g
        Δra, Δdec = Nereus.star_reflex_offset(orb, t_g[j], M_sec)
        sky = zp_g[1]*sin(ψ_g[j]) + zp_g[2]*cos(ψ_g[j]) + plx_cat*parf_g[j] +
              pmra_cat*sin(ψ_g[j])*pm_g[j] + pmdec_cat*cos(ψ_g[j])*pm_g[j]
        absc_g[j] = sky + Nereus.along_scan_projection(Δra, Δdec, ψ_g[j]) +
                    σ_g * randn(rng)
    end
    gaia = IADData(t = t_g, abscissa = absc_g, abscissa_err = fill(σ_g, n_g),
                   psi = ψ_g, parallax_factor = parf_g, pm_factor = pm_g)
    joint = merge_iad(hip, gaia)

    @testset "merge carries both missions" begin
        @test n_iad(joint) == n_h + n_g
        @test n_iad_inst(joint) == 2
        @test count(==(1), joint.inst) == n_h
        @test count(==(2), joint.inst) == n_g
        @test joint.abscissa_kind == [:residual, :absolute]
    end

    # -------------------------------------------------- mass recovery
    a_true = ((M_pri + M_sec) * P_yr^2)^(1/3)
    function _profile(iad)
        data = Data(t_rv = [t_h[1], t_g[end]], rv = [0.0, 0.0],
                    rv_err = [1.0, 1.0], iad = iad)
        params = Params(max_kplanet = 1, planet_modes = [RVAS],
            instruments = InstrumentConfig(rv = ["X"]), data = data,
            stability = :none, M_s = M_pri,
            parametrization = ParametrizationConfig(mass = :a_driven),
            priors = Dict{String, PriorSpec}(
                "n_p" => FixedPrior(1.0), "a_k1" => LogUniformPrior(0.5, 200.0),
                "M_sec_k1" => LogUniformPrior(0.0005, 1.0),
                "sesinw_k1" => UniformPrior(-1.0, 1.0),
                "secosw_k1" => UniformPrior(-1.0, 1.0),
                "Mo_k1" => UniformPrior(0.0, 2π), "inc_k1" => SinePrior(),
                "Omega_k1" => UniformPrior(0.0, 2π),
                "sigma_X" => LogUniformPrior(0.1, 10.0),
                "M_pri" => FixedPrior(M_pri)))
        theta = Theta(params)
        set_param!(theta, "n_p", 1); set_param!(theta, "a_k1", a_true)
        set_param!(theta, "sesinw_k1", sqrt(e)*sin(ω))
        set_param!(theta, "secosw_k1", sqrt(e)*cos(ω))
        set_param!(theta, "Mo_k1", mod(2π*((t_h[1]+t_g[end])/2 - tp)/P_d, 2π))
        set_param!(theta, "inc_k1", inc); set_param!(theta, "Omega_k1", Ω)
        set_param!(theta, "plx", plx_cat)
        set_param!(theta, "gamma_X", 0.0); set_param!(theta, "sigma_X", 1.0)
        grid = collect(range(0.002, 0.12, length = 201))
        ll = map(grid) do m
            set_param!(theta, "M_sec_k1", m)
            iad_log_likelihood(theta, data)
        end
        imax = argmax(ll)
        lo = findlast(k -> k <= imax && ll[k] < ll[imax] - 0.5, eachindex(grid))
        hi = findfirst(k -> k >= imax && ll[k] < ll[imax] - 0.5, eachindex(grid))
        σm = (lo === nothing || hi === nothing) ? NaN : (grid[hi] - grid[lo]) / 2
        return grid[imax], σm
    end

    m_h, s_h = _profile(hip)
    m_g, s_g = _profile(gaia)
    m_j, s_j = _profile(joint)

    @testset "joint recovers the injected mass, and beats either mission" begin
        # Hipparcos alone says nothing: a 42-yr orbit over a 3.1-yr baseline
        # IS a linear sky path, and the marginalisation eats it. No 1-sigma
        # interval exists inside the grid.
        @test isnan(s_h)
        # Gaia alone detects it, but only just.
        @test isfinite(s_g)
        @test abs(m_g - M_sec) < 3 * s_g
        # Joint: unbiased and strictly tighter. Measured on this seed,
        # 0.052 +/- 0.0069 against 0.064 +/- 0.0164 for Gaia alone.
        @test isfinite(s_j)
        @test abs(m_j - M_sec) < 3 * s_j
        @test s_j < s_g
        @test s_g / s_j > 1.5
    end
end


# ---------------------------------------------------------------------
# Multi-instrument RV *composed with* multi-instrument astrometry.
#
# `rv_inst` and the astrometry instrument indices are independent axes, and
# every other test in this suite pins one of them at a single instrument.
# This is the configuration the whole change exists to serve: several
# spectrographs, two astrometric missions, and two imagers, all constraining
# one orbit through one `Data`.
# ---------------------------------------------------------------------
@testset "astrometry — multi-instrument RV + astrometry compose" begin
    rng = MersenneTwister(7)
    M_pri, M_sec = 1.05, 0.045
    P_yr = 42.0; P_d = P_yr * 365.25
    e, ω, Ω, inc = 0.35, 0.9, 2.1, deg2rad(63.0)
    plx_cat, pmra_cat, pmdec_cat = 24.0, 150.0, -95.0
    tp = jyear_to_mjd(1985.0)
    orb = Nereus.build_orbit(P_d, e, ω, Ω, inc, M_pri, M_sec, tp, plx_cat)
    a_true = ((M_pri + M_sec) * P_yr^2)^(1/3)

    function _iad_block(jyr, σ, refepoch; absolute, zp = (0.0, 0.0))
        n = length(jyr); t = jyear_to_mjd.(jyr); ψ = 2π .* rand(rng, n)
        pm = jyr .- refepoch
        pf = sin.(2π .* (jyr .- refepoch) .- ψ)
        a = Vector{Float64}(undef, n)
        for j in 1:n
            Δra, Δdec = Nereus.star_reflex_offset(orb, t[j], M_sec)
            refl = Nereus.along_scan_projection(Δra, Δdec, ψ[j])
            sky = absolute ?
                (zp[1]*sin(ψ[j]) + zp[2]*cos(ψ[j]) + plx_cat*pf[j] +
                 pmra_cat*sin(ψ[j])*pm[j] + pmdec_cat*cos(ψ[j])*pm[j]) : 0.0
            a[j] = sky + refl + σ * randn(rng)
        end
        return IADData(t = t, abscissa = a, abscissa_err = fill(σ, n), psi = ψ,
                       parallax_factor = pf, pm_factor = pm,
                       ref_params = absolute ? nothing :
                                    [(0.0, 0.0, plx_cat, pmra_cat, pmdec_cat)],
                       abscissa_kind = absolute ? :absolute : :residual)
    end
    iad = merge_iad(_iad_block(range(1990.05, 1993.15, length = 52), 2.2, 1991.25;
                               absolute = false),
                    _iad_block(range(2014.60, 2017.45, length = 210), 0.12, 2017.5;
                               absolute = true, zp = (3.5, -2.0)))

    function _imager(jyr, σ)
        t = jyear_to_mjd.(jyr); ra = Float64[]; dec = Float64[]
        for tj in t
            Δra, Δdec = Nereus.relastrom_offset(orb, tj)
            push!(ra, Δra + σ*randn(rng)); push!(dec, Δdec + σ*randn(rng))
        end
        RelAstromData(t = t, ra_off = ra, dec_off = dec,
                      ra_err = fill(σ, length(t)), dec_err = fill(σ, length(t)))
    end
    relast = merge_relast(_imager([2014.1, 2015.2, 2016.3], 3.0),
                          _imager([2017.4, 2018.5, 2019.6, 2020.7], 1.5))

    t_h = collect(jyear_to_mjd(2010.0):40.0:jyear_to_mjd(2016.0))
    t_k = collect(jyear_to_mjd(2013.0):55.0:jyear_to_mjd(2021.0))
    t_rv    = vcat(t_h, t_k)
    rv_inst = vcat(fill(1, length(t_h)), fill(2, length(t_k)))
    rv_err  = vcat(fill(2.0, length(t_h)), fill(4.0, length(t_k)))
    γ_true  = (10.0, -35.0)

    instruments = InstrumentConfig(rv = ["HARPS", "HIRES"], as = ["GPI", "SPHERE"])
    PR = Dict{String, PriorSpec}(
        "n_p"=>FixedPrior(1.0), "a_k1"=>LogUniformPrior(0.5, 200.0),
        "M_sec_k1"=>LogUniformPrior(0.0005, 1.0),
        "sesinw_k1"=>UniformPrior(-1.0,1.0), "secosw_k1"=>UniformPrior(-1.0,1.0),
        "Mo_k1"=>UniformPrior(0.0,2π), "inc_k1"=>SinePrior(),
        "Omega_k1"=>UniformPrior(0.0,2π), "M_pri"=>FixedPrior(M_pri),
        "gamma_HARPS"=>UniformPrior(-500.0,500.0),
        "gamma_HIRES"=>UniformPrior(-500.0,500.0),
        "sigma_HARPS"=>LogUniformPrior(0.1,50.0),
        "sigma_HIRES"=>LogUniformPrior(0.1,50.0),
        "sigma_as_GPI"=>LogUniformPrior(1e-3,50.0),
        "sigma_as_SPHERE"=>LogUniformPrior(1e-3,50.0))

    function _build(rv_vals)
        data = Data(t_rv = t_rv, rv = rv_vals, rv_err = rv_err, rv_inst = rv_inst,
                    iad = iad, relastrom = relast)
        params = Params(max_kplanet = 1, planet_modes = [RVAS],
            instruments = instruments, data = data, stability = :none, M_s = M_pri,
            parametrization = ParametrizationConfig(mass = :a_driven), priors = PR)
        th = Theta(params)
        set_param!(th,"n_p",1); set_param!(th,"a_k1",a_true)
        set_param!(th,"M_sec_k1",M_sec)
        set_param!(th,"sesinw_k1",sqrt(e)*sin(ω))
        set_param!(th,"secosw_k1",sqrt(e)*cos(ω))
        set_param!(th,"Mo_k1", mod(2π*(data.t_ref - tp)/P_d, 2π))
        set_param!(th,"inc_k1",inc); set_param!(th,"Omega_k1",Ω)
        set_param!(th,"plx",plx_cat)
        set_param!(th,"gamma_HARPS",γ_true[1]); set_param!(th,"gamma_HIRES",γ_true[2])
        set_param!(th,"sigma_HARPS",1.0); set_param!(th,"sigma_HIRES",1.0)
        set_param!(th,"sigma_as_GPI",0.1); set_param!(th,"sigma_as_SPHERE",0.1)
        return th, data, params
    end
    # Bootstrap RV data from the truth model so the two data types describe
    # one star rather than two unrelated signals.
    th0, d0, _ = _build(zeros(length(t_rv)))
    pred, _ = Nereus.rv_predictions(th0, d0)
    th, data, params = _build(pred .+ rv_err .* randn(rng, length(t_rv)))

    @testset "one Data carries every instrument axis at once" begin
        @test maximum(data.rv_inst) == 2
        @test n_iad_inst(data.iad) == 2
        @test n_relast_inst(data.relastrom) == 2
        for nm in ("gamma_HARPS", "gamma_HIRES", "sigma_HARPS", "sigma_HIRES",
                   "sigma_as_GPI", "sigma_as_SPHERE", "plx", "M_pri")
            @test nm in params.layout.names
        end
    end

    ll_tot = rv_log_likelihood(th, data)
    @testset "the combined likelihood is finite and additive" begin
        @test isfinite(ll_tot)
        @test astrom_log_likelihood(th, data) ≈
              iad_log_likelihood(th, data) + relastrom_log_likelihood(th, data) rtol = 1e-12
    end

    @testset "every per-instrument parameter moves the likelihood" begin
        function bump(nm, v)
            idx = params.layout.name_to_idx[nm]
            old = th.values[idx]
            set_param!(th, nm, v); new = rv_log_likelihood(th, data)
            set_param!(th, nm, old)
            return new - ll_tot
        end
        d_gh = bump("gamma_HARPS", γ_true[1] + 30.0)
        d_gk = bump("gamma_HIRES", γ_true[2] + 30.0)
        d_sh = bump("sigma_HARPS", 20.0)
        d_sk = bump("sigma_HIRES", 20.0)
        d_ag = bump("sigma_as_GPI", 30.0)
        d_as = bump("sigma_as_SPHERE", 30.0)
        for d in (d_gh, d_gk, d_sh, d_sk, d_ag, d_as)
            @test isfinite(d) && d < 0            # truth is the better point
        end
        # Distinct per instrument — a shared slot would give identical deltas.
        @test d_gh != d_gk
        @test d_sh != d_sk
        @test d_ag != d_as
    end

    @testset "RV sharpens the astrometric mass" begin
        function profile(f)
            grid = collect(range(0.005, 0.12, length = 221))
            ll = map(grid) do m
                set_param!(th, "M_sec_k1", m); f()
            end
            set_param!(th, "M_sec_k1", M_sec)
            i = argmax(ll)
            lo = findlast(k -> k <= i && ll[k] < ll[i] - 0.5, eachindex(grid))
            hi = findfirst(k -> k >= i && ll[k] < ll[i] - 0.5, eachindex(grid))
            σ = (lo === nothing || hi === nothing) ? NaN : (grid[hi] - grid[lo]) / 2
            return grid[i], σ
        end
        m_as, s_as = profile(() -> astrom_log_likelihood(th, data))
        m_all, s_all = profile(() -> rv_log_likelihood(th, data))
        @test isfinite(s_as) && isfinite(s_all)
        @test abs(m_as  - M_sec) < 3 * s_as
        @test abs(m_all - M_sec) < 3 * s_all
        # Measured: 0.0505 +/- 0.0058 astrometry-only, 0.0447 +/- 0.0005 joint.
        @test s_all < s_as
        @test s_as / s_all > 3
    end
end
