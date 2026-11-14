# The Gaia-epoch models fit the GOST transits inside the data release's own
# window, and nothing else. A GOST plan is fetched over the whole mission (to
# 2025 by default), while the catalogue quantities the likelihoods model were
# measured from the DR2 window (G23H's DR2 epoch) or the DR3 window (HGCA's
# Gaia epoch and Gaia position, G23H's DR3 epoch, the published Gaia DR3
# five-parameter solution). Fitting through every scan averaged the reflex
# over years the catalogue never saw.

using Nereus, Test, LinearAlgebra

const W2 = Nereus.GAIA_DR2_WINDOW
const W3 = Nereus.GAIA_DR3_WINDOW

# Transits inside DR2, inside DR3 but after DR2, and after DR3.
const T_DR2  = collect(range(W2[1] + 5, W2[2] - 5, length = 24))
const T_DR3X = collect(range(W2[2] + 5, W3[2] - 5, length = 10))
const T_LATE = collect(range(W3[2] + 20, 60600.0, length = 60))

_gw_gost(t) = GOSTData(t = t, psi = mod.(2π .* (1:length(t)) ./ 7.3, 2π),
                       parallax_factor = sin.(2π .* (t .- 57000.0) ./ 365.25))

const G_ALL  = _gw_gost(vcat(T_DR2, T_DR3X, T_LATE))
const G_DR3  = Nereus._gost_window(G_ALL, W3)        # what DR3 saw
const G_NOX  = _gw_gost(vcat(T_DR2, T_LATE))         # DR2 + late, no DR3-only

# A 4 yr orbit: short enough that which transits enter moves the fitted PM.
const M_SEC = 0.01
_gw_orbit() = Nereus.build_orbit(4 * 365.25, 0.4, 0.7, 1.1, deg2rad(55), 1.0,
                                 M_SEC, 57300.0, 25.0)

const HGCA = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                      pmra = (5.0, 4.95, 4.9), pmdec = (-3.0, -3.05, -3.1),
                      sigma_pmra = (0.5, 0.02, 0.03), sigma_pmdec = (0.4, 0.02, 0.03),
                      plx = 25.0, plx_err = 0.05, hip_id = 1)

const G23H = G23HData(epochs = mjd_epochs((1991.25, 2004.6, 2015.5, 2015.75, 2016.0)),
                      pmra = (5.0, 4.95, 4.9, 4.9, 4.9), pmdec = (-3.0, -3.05, -3.1, -3.1, -3.1),
                      cov = Matrix(Diagonal(fill(0.05^2, 10))),
                      plx = 25.0, plx_err = 0.05, hip_id = 1)

function _gw_iad()
    n = 40
    t = collect(range(jyear_to_mjd(1990.0), jyear_to_mjd(1993.0), length = n))
    psi = mod.(2π .* (1:n) ./ 5.1, 2π)
    plxf = sin.(2π .* (t .- t[1]) ./ 365.25 .- psi)
    IADData(t = t, abscissa = 25.0 .* plxf, abscissa_err = fill(1.0, n), psi = psi,
            parallax_factor = plxf, pm_factor = (t .- jyear_to_mjd(1991.25)) ./ 365.25)
end

const GAIA = GaiaDR3Data(params = (0.0, 0.0, 25.0, 0.3, -0.2),
                         cov = Matrix(Diagonal([0.05, 0.05, 0.02, 0.02, 0.02] .^ 2)),
                         t_ref = jyear_to_mjd(2016.0))

function _gw_setup(; gost, kw...)
    data = Data(; t_rv = [57000.0, 57300.0], rv = [10.0, -10.0], rv_err = [1.0, 1.0],
                gost = gost, kw...)
    params = Params(parametrization = ParametrizationConfig(time = :Mo), max_kplanet = 1, planet_modes = [RVAS],
                    instruments = InstrumentConfig(rv = ["X"]), data = data,
                    stability = :none, M_s = 1.0)
    theta = Theta(params)
    set_param!(theta, "n_p", 1)
    set_param!(theta, "P_k1", 4 * 365.25); set_param!(theta, "K_k1", 300.0)
    set_param!(theta, "sesinw_k1", sqrt(0.4) * sin(0.7))
    set_param!(theta, "secosw_k1", sqrt(0.4) * cos(0.7))
    set_param!(theta, "Mo_k1", 1.0)
    set_param!(theta, "inc_k1", deg2rad(55)); set_param!(theta, "Omega_k1", 1.1)
    set_param!(theta, "plx", 25.0); set_param!(theta, "sigma_X", 1.0)
    return data, theta
end

function _gw_orbit_of(theta, data)
    M_pri = astrom_M_pri(theta); plx = astrom_plx(theta)
    orb, M_sec = Nereus._planet_orbit(theta, 1, M_pri, plx, data.t_ref)
    return orb, M_sec, M_pri, plx
end

@testset "GOST: Gaia-epoch models use the data release's window" begin

    @testset "windowed fit == fit through the window's transits" begin
        orb = _gw_orbit(); tr = jyear_to_mjd(2016.0)
        @test gost_5param_fit(orb, G_ALL, M_SEC; t_ref = tr, window = W3) ==
              gost_5param_fit(orb, G_DR3, M_SEC; t_ref = tr)
        @test gost_5param_fit(orb, G_ALL, M_SEC; t_ref = tr, window = W2) ==
              gost_5param_fit(orb, Nereus._gost_window(G_ALL, W2), M_SEC; t_ref = tr)
        # No window: every transit, exactly as before the keyword existed.
        @test gost_5param_fit(orb, G_ALL, M_SEC; window = nothing) ==
              gost_5param_fit(orb, G_ALL, M_SEC)
        # Default t_ref is the mean of the transits USED.
        @test gost_5param_fit(orb, G_ALL, M_SEC; window = W3) ≈
              gost_5param_fit(orb, G_ALL, M_SEC; window = W3,
                              t_ref = sum(G_DR3.t) / length(G_DR3.t)) rtol = 1e-12
        @test collect(gost_window_avg_pm(orb, G_ALL, M_SEC; window = W3)) ==
              gost_5param_fit(orb, G_ALL, M_SEC; window = W3)[4:5]
        # Below five transits in the window: zeros, as for a short plan.
        @test gost_5param_fit(orb, _gw_gost(T_LATE), M_SEC; window = W3) == zeros(5)
        # The window matters for this orbit, by far more than the HGCA error.
        μ3 = collect(gost_window_avg_pm(orb, G_ALL, M_SEC; window = W3))
        μa = collect(gost_window_avg_pm(orb, G_ALL, M_SEC))
        @test norm(μ3 .- μa) > 0.1
    end

    @testset "HGCA: Gaia PM and position from the DR3-window fit" begin
        d_all, th_all = _gw_setup(gost = G_ALL, hgca = HGCA)
        d_dr3, th_dr3 = _gw_setup(gost = G_DR3, hgca = HGCA)
        # Transits after DR3 do not enter at all.
        @test hgca_log_likelihood(th_all, d_all) == hgca_log_likelihood(th_dr3, d_dr3)

        orb, M_sec, M_pri, plx = _gw_orbit_of(th_all, d_all)
        pmra, pmdec = Nereus._hgca_model_pm(th_all, HGCA, d_all, M_pri, plx, d_all.t_ref)
        q = gost_5param_fit(orb, G_ALL, M_sec; t_ref = HGCA.epochs[3], window = W3)
        @test (pmra[3], pmdec[3]) == (q[4], q[5])
        # No Hipparcos scans in this fixture: the Hipparcos end is the
        # instantaneous fallback (test_hip_epoch_models.jl covers the fit).
        oH = Nereus.star_reflex_offset(orb, HGCA.epochs[1], M_sec)
        dt = (HGCA.epochs[3] - HGCA.epochs[1]) / 365.25
        @test pmra[2]  == (q[1] - oH[1]) / dt
        @test pmdec[2] == (q[2] - oH[2]) / dt
        @test (pmra[1], pmdec[1]) == Nereus.star_reflex_pm(orb, HGCA.epochs[1], M_sec)
    end

    @testset "G23H: DR2 and DR3 epochs each through their own window" begin
        d_all, th_all = _gw_setup(gost = G_ALL, g23h = G23H)
        d_dr3, th_dr3 = _gw_setup(gost = G_DR3, g23h = G23H)
        @test g23h_log_likelihood(th_all, d_all) == g23h_log_likelihood(th_dr3, d_dr3)

        orb, M_sec, M_pri, plx = _gw_orbit_of(th_all, d_all)
        pmra, pmdec = Nereus._g23h_model_pm(th_all, G23H, d_all, M_pri, plx, d_all.t_ref)
        # Each release's fit is referred to its catalogue epoch, which the
        # positional differences need; the proper motion is the same fit's, to
        # rounding (it used to be compared bit for bit with the fit at the
        # transits' mean epoch, `gost_window_avg_pm`).
        q3 = gost_5param_fit(orb, G_ALL, M_sec; t_ref = G23H.epochs[5], window = W3)
        q2 = gost_5param_fit(orb, G_ALL, M_sec; t_ref = G23H.epochs[3], window = W2)
        @test (pmra[5], pmdec[5]) == (q3[4], q3[5])
        @test (pmra[3], pmdec[3]) == (q2[4], q2[5])
        @test collect((pmra[5], pmdec[5])) ≈
              collect(gost_window_avg_pm(orb, G_ALL, M_sec; window = W3)) rtol = 1e-10
        # No Hipparcos scans here: the Hipparcos epoch is the instantaneous
        # reflex (with a warning). Hip→Gaia and DR3−DR2 are positional
        # differences over the epoch differences, not instantaneous proper
        # motions at 2004.6 and 2015.75 as they used to be.
        @test (pmra[1], pmdec[1]) == Nereus.star_reflex_pm(orb, G23H.epochs[1], M_sec)
        oH = Nereus.star_reflex_offset(orb, G23H.epochs[1], M_sec)
        @test pmra[2]  == (q3[1] - oH[1]) / ((G23H.epochs[5] - G23H.epochs[1]) / 365.25)
        @test pmdec[2] == (q3[2] - oH[2]) / ((G23H.epochs[5] - G23H.epochs[1]) / 365.25)
        @test pmra[4]  == (q3[1] - q2[1]) / ((G23H.epochs[5] - G23H.epochs[3]) / 365.25)
        @test pmdec[4] == (q3[2] - q2[2]) / ((G23H.epochs[5] - G23H.epochs[3]) / 365.25)

        # Transits after DR2 but inside DR3 move the DR3 epoch, not the DR2 one.
        d_nox, th_nox = _gw_setup(gost = G_NOX, g23h = G23H)
        pr, pd = Nereus._g23h_model_pm(th_nox, G23H, d_nox, M_pri, plx, d_nox.t_ref)
        @test (pr[3], pd[3]) == (pmra[3], pmdec[3])
        @test (pr[5], pd[5]) != (pmra[5], pmdec[5])
    end

    @testset "Gaia DR3 + IAD joint: the published solution's window only" begin
        iad = _gw_iad()
        d_all, th_all = _gw_setup(gost = G_ALL, iad = iad, gaia_dr3 = GAIA)
        d_dr3, th_dr3 = _gw_setup(gost = G_DR3, iad = iad, gaia_dr3 = GAIA)
        ll = iad_log_likelihood(th_all, d_all)
        @test isfinite(ll)
        @test ll == iad_log_likelihood(th_dr3, d_dr3)
    end

    @testset "Data refuses a plan that misses the window it needs" begin
        late = _gw_gost(T_LATE)
        @test_throws ArgumentError _gw_setup(gost = late, hgca = HGCA)
        @test_throws ArgumentError _gw_setup(gost = late, iad = _gw_iad(), gaia_dr3 = GAIA)
        # G23H also needs the DR2 window.
        no_dr2 = _gw_gost(vcat(T_DR3X, T_LATE))
        err = try _gw_setup(gost = no_dr2, g23h = G23H); nothing catch e; e end
        @test err isa ArgumentError && occursin("DR2", err.msg)
        @test _gw_setup(gost = no_dr2, hgca = HGCA) isa Tuple   # HGCA needs DR3 only
        # GOST with nothing to model is inert, not an error.
        inert = @test_logs (:warn, r"INERT") match_mode = :any Data(;
            t_rv = [57000.0, 57300.0], rv = [1.0, -1.0], rv_err = [1.0, 1.0], gost = late)
        @test inert isa Data
    end
end
