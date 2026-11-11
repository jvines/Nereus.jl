# Every HGCA and G23H model quantity is the catalogue's own measurement of the
# reflex, at the catalogue's own epochs:
#
#   Hipparcos PM         — a weighted five-parameter fit through the Hipparcos
#                          IAD scans inside HIPPARCOS_WINDOW;
#   Gaia DR2 / DR3 PM    — the same fit through the GOST transits inside the
#                          release's window;
#   Hipparcos–Gaia       — (Gaia position − Hipparcos position) / Δt;
#   G23H DR3 − DR2       — (DR3 position − DR2 position) / Δt;
#
# every position at the catalogue's central epoch of that coordinate, every Δt
# the difference of those epochs. The Hipparcos epoch used to be the
# instantaneous reflex at 1991.25, the G23H Hip→Gaia and DR3−DR2 epochs
# instantaneous proper motions at 2004.6 and 2015.75.

using Nereus, Test, LinearAlgebra, ForwardDiff, FITSIO

const HE_WH = Nereus.HIPPARCOS_WINDOW
const HE_W2 = Nereus.GAIA_DR2_WINDOW
const HE_W3 = Nereus.GAIA_DR3_WINDOW
const HE_YR = 365.25

# ---- scan geometry ---------------------------------------------------------

# A Hipparcos-like IAD: 40 epochs across the mission, three abscissae each
# (same epoch, slightly different angle, different errors), so 120 scans.
function _he_hip(; t0 = jyear_to_mjd(1989.95), t1 = jyear_to_mjd(1993.10), n_ep = 40,
                 extra_t = Float64[])
    tep = collect(range(t0, t1, length = n_ep))
    t = Float64[]; psi = Float64[]; err = Float64[]
    for (k, te) in enumerate(tep), m in 1:3
        push!(t, te)
        push!(psi, mod(2π * k / 6.7 + 0.01 * m, 2π))
        push!(err, 1.0 + 0.6 * mod(k + m, 4))
    end
    for te in extra_t                       # scans outside the mission, if asked
        push!(t, te); push!(psi, 0.3 + te / 1e4); push!(err, 1.2)
    end
    plxf = sin.(2π .* (t .- t[1]) ./ HE_YR .- psi)
    IADData(t = t, abscissa = zeros(length(t)), abscissa_err = err, psi = psi,
            parallax_factor = plxf, pm_factor = (t .- jyear_to_mjd(1991.25)) ./ HE_YR)
end

# Transits inside DR2, inside DR3 after DR2, and after DR3.
const HE_T2  = collect(range(HE_W2[1] + 5, HE_W2[2] - 5, length = 24))
const HE_T3X = collect(range(HE_W2[2] + 5, HE_W3[2] - 5, length = 10))
const HE_TL = collect(range(HE_W3[2] + 20, 60600.0, length = 40))
_he_gost(t) = GOSTData(t = t, psi = mod.(2π .* (1:length(t)) ./ 7.3, 2π),
                       parallax_factor = sin.(2π .* (t .- 57000.0) ./ HE_YR))
const HE_GOST = _he_gost(vcat(HE_T2, HE_T3X, HE_TL))
const HE_HIP  = _he_hip()

# ---- catalogue rows --------------------------------------------------------

# Per-coordinate central epochs that differ between RA and Dec, as real ones do
# (HIP 103836: Hipparcos 1991.20 / 1991.39, Gaia 2015.96 / 2016.11).
const HE_HGRA  = mjd_epochs((1991.20, 2003.58, 2015.96))
const HE_HGDEC = mjd_epochs((1991.39, 2003.75, 2016.11))
_he_hgca(; ra = HE_HGRA, dec = HE_HGDEC) =
    HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
             pmra = (5.0, 4.95, 4.9), pmdec = (-3.0, -3.05, -3.1),
             sigma_pmra = (0.5, 0.02, 0.03), sigma_pmdec = (0.4, 0.02, 0.03),
             plx = 25.0, plx_err = 0.05, hip_id = 1, epochs_ra = ra, epochs_dec = dec)
const HE_GRA  = mjd_epochs((1991.20, 2003.6, 2015.42, 2015.69, 2015.96))
const HE_GDEC = mjd_epochs((1991.39, 2003.7, 2015.58, 2015.85, 2016.11))
_he_g23h(; ra = HE_GRA, dec = HE_GDEC) =
    G23HData(epochs = mjd_epochs((1991.25, 2004.6, 2015.5, 2015.75, 2016.0)),
             pmra = (5.0, 4.95, 4.9, 4.9, 4.9), pmdec = (-3.0, -3.05, -3.1, -3.1, -3.1),
             cov = Matrix(Diagonal(fill(0.05^2, 10))),
             plx = 25.0, plx_err = 0.05, hip_id = 1, epochs_ra = ra, epochs_dec = dec)

# ---- a target with one astrometric companion -------------------------------

# M_sec-driven, so the period and the true mass are set directly: the
# long-period limit holds the mass, not K, fixed.
function _he_setup(; P_yr = 4.0, M_sec = 0.02, data_kw...)
    data = Data(; t_rv = [57000.0, 57300.0], rv = [10.0, -10.0], rv_err = [1.0, 1.0],
                data_kw...)
    params = Params(max_kplanet = 1, planet_modes = [RVAS],
                    instruments = InstrumentConfig(rv = ["X"]), data = data,
                    stability = :none, M_s = 1.0,
                    parametrization = ParametrizationConfig(mass = :M_sec_driven))
    θ = Theta(params)
    set_param!(θ, "n_p", 1)
    set_param!(θ, "P_k1", P_yr * HE_YR)
    set_param!(θ, "M_sec_k1", M_sec)
    set_param!(θ, "sesinw_k1", sqrt(0.4) * sin(0.7))
    set_param!(θ, "secosw_k1", sqrt(0.4) * cos(0.7))
    set_param!(θ, "Mo_k1", 1.0)
    set_param!(θ, "inc_k1", deg2rad(55)); set_param!(θ, "Omega_k1", 1.1)
    set_param!(θ, "plx", 25.0); set_param!(θ, "sigma_X", 1.0)
    return data, θ
end

function _he_orbit(θ, data)
    M_pri = astrom_M_pri(θ); plx = astrom_plx(θ)
    orb, M_sec = Nereus._planet_orbit(θ, 1, M_pri, plx, data.t_ref)
    return orb, M_sec, M_pri, plx
end

# Quiet construction: the instantaneous-fallback warning is tested on its own.
_he_quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# A linear sky path standing in for an orbit: position (a, d) at t0, proper
# motion (μa, μd). `_scan_5param_accumulate` only asks a kernel for its offset.
struct _HeLinPath
    a::Float64; d::Float64; μa::Float64; μd::Float64; t0::Float64
end
Nereus._reflex_offset(k::_HeLinPath, t::Real) =
    (k.a + k.μa * (t - k.t0) / HE_YR, k.d + k.μd * (t - k.t0) / HE_YR)
Nereus._kernel_eltype(::_HeLinPath) = Float64

# Hand-built weighted least squares: the fit written out with an explicit
# design matrix, against the along-scan projection `_iad_residuals!` uses.
function _he_wls(t, s, c, f, w, η, t_ra, t_dec)
    X = hcat(s, c, f, s .* (t .- t_ra) ./ HE_YR, c .* (t .- t_dec) ./ HE_YR)
    W = Diagonal(w)
    return (X' * W * X) \ (X' * W * η)
end

@testset "HGCA / G23H: catalogue epochs modelled as measured" begin

    @testset "convention: a pure linear motion is recovered exactly" begin
        path = _HeLinPath(3.7, -1.9, 12.5, -7.25, jyear_to_mjd(1991.25))
        tra, tdec = HE_HGRA[1], HE_HGDEC[1]
        q = Nereus._scan_5param_accumulate(path, HE_HIP, 0.0, tra, tdec, HE_WH[1], HE_WH[2])
        @test q[1] ≈ path.a + path.μa * (tra - path.t0) / HE_YR atol = 1e-10
        @test q[2] ≈ path.d + path.μd * (tdec - path.t0) / HE_YR atol = 1e-10
        @test abs(q[3]) < 1e-10
        @test q[4] ≈ path.μa atol = 1e-10
        @test q[5] ≈ path.μd atol = 1e-10
        # Gaia: the same through a GOST plan's DR3 window.
        path = _HeLinPath(-0.8, 2.2, -4.0, 9.0, jyear_to_mjd(2016.0))
        q = Nereus._scan_5param_accumulate(path, HE_GOST, 0.0, HE_GRA[5], HE_GDEC[5], HE_W3[1], HE_W3[2])
        @test q[1] ≈ path.a + path.μa * (HE_GRA[5] - path.t0) / HE_YR atol = 1e-10
        @test q[2] ≈ path.d + path.μd * (HE_GDEC[5] - path.t0) / HE_YR atol = 1e-10
        @test q[4] ≈ path.μa atol = 1e-10
        @test q[5] ≈ path.μd atol = 1e-10

        # ... and the projection is the IAD forward model's: the Hipparcos fit
        # of an orbit equals a least-squares fit, weighted by 1/σ², to exactly
        # the reflex `_iad_residuals!` subtracts from each abscissa.
        orb, M_sec = Nereus.build_orbit(4 * HE_YR, 0.4, 0.7, 1.1, deg2rad(55), 1.0, 0.02,
                                    48600.0, 25.0), 0.02
        r = zeros(n_iad(HE_HIP))
        Nereus._iad_residuals!(r, HE_HIP, [orb], [M_sec], 0.0)  # abscissae 0, no parallax term
        q = scan_5param_fit(orb, HE_HIP, M_sec; t_ref = (tra, tdec), window = HE_WH)
        @test q ≈ _he_wls(HE_HIP.t, HE_HIP.sinpsi, HE_HIP.cospsi, HE_HIP.parallax_factor, HE_HIP.weight,
                       -r, tra, tdec) rtol = 1e-9
        # GOST: uniform weights, the projection taken from ψ.
        k = findall(t -> HE_W3[1] <= t <= HE_W3[2], HE_GOST.t)
        g = HE_GOST
        η = [Nereus.along_scan_projection(Nereus.star_reflex_offset(orb, g.t[j], M_sec)...,
                                      g.psi[j]) for j in k]
        q = gost_5param_fit(orb, g, M_sec; t_ref = (HE_GRA[5], HE_GDEC[5]), window = HE_W3)
        @test q ≈ _he_wls(g.t[k], sin.(g.psi[k]), cos.(g.psi[k]), g.parallax_factor[k],
                       ones(length(k)), η, HE_GRA[5], HE_GDEC[5]) rtol = 1e-9
        # One reference epoch for both coordinates is still the GOST fit as it
        # always was, to the bit.
        @test gost_5param_fit(orb, g, M_sec; t_ref = 57388.0, window = HE_W3) ==
              gost_5param_fit(orb, g, M_sec; t_ref = (57388.0, 57388.0), window = HE_W3)
        # The proper motion does not depend on the reference epochs.
        q1 = scan_5param_fit(orb, HE_HIP, M_sec; t_ref = (tra, tdec), window = HE_WH)
        q2 = scan_5param_fit(orb, HE_HIP, M_sec; window = HE_WH)
        @test q1[4:5] ≈ q2[4:5] rtol = 1e-10
    end

    @testset "HGCA: each quantity is the hand-built fit" begin
        hg = _he_hgca()
        d, θ = _he_quiet(() -> _he_setup(hgca = hg, gost = HE_GOST, hip_scans = HE_HIP))
        @test d.hip_scans === HE_HIP
        orb, M_sec, M_pri, plx = _he_orbit(θ, d)
        pmra, pmdec = Nereus._hgca_model_pm(θ, hg, d, M_pri, plx, d.t_ref)
        H = scan_5param_fit(orb, HE_HIP, M_sec; t_ref = (HE_HGRA[1], HE_HGDEC[1]), window = HE_WH)
        G = gost_5param_fit(orb, HE_GOST, M_sec; t_ref = (HE_HGRA[3], HE_HGDEC[3]), window = HE_W3)
        @test (pmra[1], pmdec[1]) == (H[4], H[5])
        @test pmra[2]  == (G[1] - H[1]) / ((HE_HGRA[3] - HE_HGRA[1]) / HE_YR)
        @test pmdec[2] == (G[2] - H[2]) / ((HE_HGDEC[3] - HE_HGDEC[1]) / HE_YR)
        @test (pmra[3], pmdec[3]) == (G[4], G[5])
        # The Dec end of the difference really is at the Dec epochs.
        @test pmdec[2] != (G[2] - H[2]) / ((HE_HGRA[3] - HE_HGRA[1]) / HE_YR)
    end

    @testset "G23H: each of the five is the hand-built fit" begin
        g = _he_g23h()
        d, θ = _he_quiet(() -> _he_setup(g23h = g, gost = HE_GOST, hip_scans = HE_HIP))
        orb, M_sec, M_pri, plx = _he_orbit(θ, d)
        pmra, pmdec = Nereus._g23h_model_pm(θ, g, d, M_pri, plx, d.t_ref)
        H  = scan_5param_fit(orb, HE_HIP, M_sec;  t_ref = (HE_GRA[1], HE_GDEC[1]), window = HE_WH)
        D2 = gost_5param_fit(orb, HE_GOST, M_sec; t_ref = (HE_GRA[3], HE_GDEC[3]), window = HE_W2)
        D3 = gost_5param_fit(orb, HE_GOST, M_sec; t_ref = (HE_GRA[5], HE_GDEC[5]), window = HE_W3)
        @test (pmra[1], pmdec[1]) == (H[4], H[5])
        @test pmra[2]  == (D3[1] - H[1]) / ((HE_GRA[5] - HE_GRA[1]) / HE_YR)
        @test pmdec[2] == (D3[2] - H[2]) / ((HE_GDEC[5] - HE_GDEC[1]) / HE_YR)
        @test (pmra[3], pmdec[3]) == (D2[4], D2[5])
        @test pmra[4]  == (D3[1] - D2[1]) / ((HE_GRA[5] - HE_GRA[3]) / HE_YR)
        @test pmdec[4] == (D3[2] - D2[2]) / ((HE_GDEC[5] - HE_GDEC[3]) / HE_YR)
        @test (pmra[5], pmdec[5]) == (D3[4], D3[5])
        # The nominal DR3 − DR2 divisor is the 0.5 yr between J2016.0 and J2015.5.
        gn = _he_g23h(ra = g.epochs, dec = g.epochs)
        dn, θn = _he_quiet(() -> _he_setup(g23h = gn, gost = HE_GOST, hip_scans = HE_HIP))
        pr, pd = Nereus._g23h_model_pm(θn, gn, dn, M_pri, plx, dn.t_ref)
        D2n = gost_5param_fit(orb, HE_GOST, M_sec; t_ref = gn.epochs[3], window = HE_W2)
        D3n = gost_5param_fit(orb, HE_GOST, M_sec; t_ref = gn.epochs[5], window = HE_W3)
        @test (gn.epochs[5] - gn.epochs[3]) / HE_YR == 0.5
        @test pr[4] ≈ (D3n[1] - D2n[1]) / 0.5 rtol = 1e-12
    end

    @testset "DR3 − DR2 depends on which transits fall in each window" begin
        g = _he_g23h()
        model(gost) = begin
            d, θ = _he_quiet(() -> _he_setup(g23h = g, gost = gost, hip_scans = HE_HIP))
            _, _, M_pri, plx = _he_orbit(θ, d)
            Nereus._g23h_model_pm(θ, g, d, M_pri, plx, d.t_ref)
        end
        all3 = model(HE_GOST)
        # Transits after DR3 enter nothing.
        @test model(_he_gost(vcat(HE_T2, HE_T3X))) == all3
        # Without the DR3-only transits the DR2 epoch is unchanged and the DR3
        # position, so DR3 − DR2 and DR3, move.
        nox = model(_he_gost(vcat(HE_T2, HE_TL)))
        @test (nox[1][3], nox[2][3]) == (all3[1][3], all3[2][3])
        @test abs(nox[1][4] - all3[1][4]) + abs(nox[2][4] - all3[2][4]) > 1e-3
        @test (nox[1][5], nox[2][5]) != (all3[1][5], all3[2][5])
        # Thinning the DR2 window moves the DR2 epoch and DR3 − DR2.
        thin = model(_he_gost(vcat(HE_T2[1:2:end], HE_T3X, HE_TL)))
        @test (thin[1][3], thin[2][3]) != (all3[1][3], all3[2][3])
        @test (thin[1][4], thin[2][4]) != (all3[1][4], all3[2][4])
    end

    # The old model of each quantity, written out: instantaneous reflex at the
    # Hipparcos epochs; G23H Hip→Gaia and DR3−DR2 as instantaneous PMs.
    function _old_vs_new(P_yr; M_sec = 0.02)
        hg = _he_hgca(); g = _he_g23h()
        d, θ = _he_quiet(() -> _he_setup(; P_yr, M_sec, hgca = hg, g23h = g, gost = HE_GOST,
                                     hip_scans = HE_HIP))
        orb, Ms, M_pri, plx = _he_orbit(θ, d)
        hn = Nereus._hgca_model_pm(θ, hg, d, M_pri, plx, d.t_ref)
        gn = Nereus._g23h_model_pm(θ, g, d, M_pri, plx, d.t_ref)
        # HGCA: the old Hipparcos epoch, the Gaia end through GOST as before.
        μH = (Nereus.star_reflex_pm(orb, HE_HGRA[1], Ms)[1], Nereus.star_reflex_pm(orb, HE_HGDEC[1], Ms)[2])
        oH = (Nereus.star_reflex_offset(orb, HE_HGRA[1], Ms)[1],
              Nereus.star_reflex_offset(orb, HE_HGDEC[1], Ms)[2])
        G = gost_5param_fit(orb, HE_GOST, Ms; t_ref = (HE_HGRA[3], HE_HGDEC[3]), window = HE_W3)
        ho = ([μH[1], (G[1] - oH[1]) / ((HE_HGRA[3] - HE_HGRA[1]) / HE_YR), G[4]],
              [μH[2], (G[2] - oH[2]) / ((HE_HGDEC[3] - HE_HGDEC[1]) / HE_YR), G[5]])
        # G23H: the old instantaneous PMs at Hipparcos, 2004.6 and 2015.75.
        go = (copy(gn[1]), copy(gn[2]))
        for ie in (1, 2, 4)
            μ = Nereus.star_reflex_pm(orb, g.epochs[ie], Ms)
            go[1][ie] = μ[1]; go[2][ie] = μ[2]
        end
        return hn, ho, gn, go
    end
    σ_hg = (_he_hgca().cov_ep[1][1, 1]^0.5, _he_hgca().cov_ep[2][1, 1]^0.5)  # Hip, HG

    @testset "long-period limit: the new values approach the instantaneous ones" begin
        Δ(a, b, ie) = hypot(a[1][ie] - b[1][ie], a[2][ie] - b[2][ie])
        rows = [(P, _old_vs_new(P)...) for P in (60.0, 250.0, 1000.0)]
        for ie in (1, 2)        # HGCA Hipparcos, Hipparcos–Gaia
            d = [Δ(r[2], r[3], ie) for r in rows]
            @test d[1] > d[2] > d[3]
        end
        for ie in (1, 2, 4)     # G23H Hipparcos, Hip→Gaia, DR3−DR2
            d = [Δ(r[4], r[5], ie) for r in rows]
            @test d[1] > d[2] > d[3]
        end
        # At P = 1000 yr (300× the Hipparcos mission, 40× the baseline) the
        # difference is a small fraction of the catalogue errors.
        _, hn, ho, gn, go = rows[3]
        @test Δ(hn, ho, 1) < 0.01 * σ_hg[1]
        @test Δ(hn, ho, 2) < 0.01 * σ_hg[2]
        @test Δ(gn, go, 1) < 0.01 * σ_hg[1]
        @test Δ(gn, go, 2) < 0.05 * 0.05          # G23H σ = 0.05 mas/yr here
        @test Δ(gn, go, 4) < 0.05 * 0.05
    end

    @testset "P of a few years: they differ by far more than the errors" begin
        # A 0.15 Msun companion on a 3 yr orbit at 40 pc: the Hipparcos fit
        # averages a reflex the mission spans about once.
        hn, ho, gn, go = _old_vs_new(3.0; M_sec = 0.15)
        Δ(a, b, ie) = hypot(a[1][ie] - b[1][ie], a[2][ie] - b[2][ie])
        @test Δ(hn, ho, 1) > 5 * σ_hg[1]          # Hipparcos: fit vs instantaneous
        @test Δ(hn, ho, 2) > 5 * σ_hg[2]          # Hipparcos–Gaia
        @test Δ(gn, go, 2) > 5 * 0.05             # G23H Hip→Gaia
        @test Δ(gn, go, 4) > 5 * 0.05             # G23H DR3−DR2
    end

    @testset "scans outside the Hipparcos mission do not enter" begin
        hg = _he_hgca()
        out = _he_hip(extra_t = vcat(collect(range(HE_WH[1] - 400, HE_WH[1] - 5, length = 6)),
                                     collect(range(57000.0, 57800.0, length = 30))))
        d1, θ1 = _he_quiet(() -> _he_setup(hgca = hg, gost = HE_GOST, hip_scans = HE_HIP))
        d2, θ2 = _he_quiet(() -> _he_setup(hgca = hg, gost = HE_GOST, hip_scans = out))
        @test hgca_log_likelihood(θ1, d1) == hgca_log_likelihood(θ2, d2)
        # A merged Hipparcos + Gaia-epoch IAD: the Gaia instrument's transits
        # stay out of the Hipparcos fit, which uses the IAD's own scans.
        gaia = IADData(t = collect(range(57000.0, 57800.0, length = 30)),
                       abscissa = zeros(30), abscissa_err = fill(0.1, 30),
                       psi = mod.(2π .* (1:30) ./ 5.3, 2π),
                       parallax_factor = sin.(2π .* (1:30) ./ 11))
        merged = merge_iad(HE_HIP, gaia)
        d3, θ3 = _he_quiet(() -> _he_setup(hgca = hg, gost = HE_GOST, iad = merged))
        @test d3.hip_scans === merged
        _, _, M_pri, plx = _he_orbit(θ1, d1)
        @test Nereus._hgca_model_pm(θ3, hg, d3, M_pri, plx, d3.t_ref) ==
              Nereus._hgca_model_pm(θ1, hg, d1, M_pri, plx, d1.t_ref)
    end

    @testset "where the scan geometry comes from" begin
        hg = _he_hgca()
        # With Hipparcos IAD in the fit, its own scans; nothing else needed.
        d, _ = _he_quiet(() -> _he_setup(hgca = hg, iad = HE_HIP))
        @test d.hip_scans === HE_HIP
        # The same scans passed twice is fine; different ones are refused.
        d, _ = _he_quiet(() -> _he_setup(hgca = hg, iad = HE_HIP, hip_scans = HE_HIP))
        @test d.hip_scans === HE_HIP
        @test_throws ArgumentError _he_quiet(() -> _he_setup(hgca = hg, iad = HE_HIP,
                                                         hip_scans = _he_hip(n_ep = 30)))
        # Fewer than five scans in the mission cannot be fitted.
        few = _he_hip(t0 = HE_WH[1] + 10, t1 = HE_WH[1] + 10, n_ep = 1)   # 3 scans
        @test_throws ArgumentError _he_quiet(() -> _he_setup(hgca = hg, hip_scans = few))
        # Given without a catalogue row it does nothing, and says so.
        @test_logs (:warn, r"INERT") match_mode = :any _he_setup(hip_scans = HE_HIP)
    end

    @testset "missing scan geometry warns and falls back to instantaneous" begin
        hg = _he_hgca(); g = _he_g23h()
        d, θ = @test_logs (:warn, r"INSTANTANEOUS") match_mode = :any _he_setup(
            hgca = hg, g23h = g, gost = HE_GOST)
        @test d.hip_scans === nothing
        orb, Ms, M_pri, plx = _he_orbit(θ, d)
        pmra, pmdec = Nereus._hgca_model_pm(θ, hg, d, M_pri, plx, d.t_ref)
        @test pmra[1]  == Nereus.star_reflex_pm(orb, HE_HGRA[1], Ms)[1]
        @test pmdec[1] == Nereus.star_reflex_pm(orb, HE_HGDEC[1], Ms)[2]
        G = gost_5param_fit(orb, HE_GOST, Ms; t_ref = (HE_HGRA[3], HE_HGDEC[3]), window = HE_W3)
        @test pmra[2] == (G[1] - Nereus.star_reflex_offset(orb, HE_HGRA[1], Ms)[1]) /
                         ((HE_HGRA[3] - HE_HGRA[1]) / HE_YR)
        @test pmdec[2] == (G[2] - Nereus.star_reflex_offset(orb, HE_HGDEC[1], Ms)[2]) /
                          ((HE_HGDEC[3] - HE_HGDEC[1]) / HE_YR)
        # G23H without GOST either: every epoch instantaneous, but the two
        # differences stay positional differences.
        d, θ = _he_quiet(() -> _he_setup(g23h = g))
        pr, pd = Nereus._g23h_model_pm(θ, g, d, M_pri, plx, d.t_ref)
        off(t, i) = Nereus.star_reflex_offset(orb, t, Ms)[i]
        @test pr[4] == (off(HE_GRA[5], 1) - off(HE_GRA[3], 1)) / ((HE_GRA[5] - HE_GRA[3]) / HE_YR)
        @test pd[2] == (off(HE_GDEC[5], 2) - off(HE_GDEC[1], 2)) / ((HE_GDEC[5] - HE_GDEC[1]) / HE_YR)
        @test pr[3] == Nereus.star_reflex_pm(orb, HE_GRA[3], Ms)[1]
    end

    @testset "run_job / fit_* obtain the scans by hip_id" begin
        hg = _he_hgca()
        calls = Int[]
        fake(h) = (push!(calls, h); HE_HIP)
        # Automatic: an HGCA row with a Hipparcos ID and no Hipparcos IAD.
        @test Nereus._hip_scans_for(nothing, nothing, hg, nothing; fetch = fake) === HE_HIP
        @test calls == [1]
        @test Nereus._hip_scans_for(true, nothing, nothing, _he_g23h(); fetch = fake) === HE_HIP
        # Nothing to fetch: the IAD has the scans, there is no row, or `false`.
        @test Nereus._hip_scans_for(nothing, HE_HIP, hg, nothing; fetch = fake) === nothing
        @test Nereus._hip_scans_for(nothing, nothing, nothing, nothing; fetch = fake) === nothing
        @test Nereus._hip_scans_for(false, nothing, hg, nothing; fetch = fake) === nothing
        @test length(calls) == 2
        # Explicit forms.
        @test Nereus._hip_scans_for(HE_HIP, nothing, hg, nothing; fetch = fake) === HE_HIP
        @test Nereus._hip_scans_for(Dict("hip_id" => 7), nothing, hg, nothing;
                                fetch = fake) === HE_HIP
        @test calls[end] == 7
        v = Nereus._hip_scans_for(Dict("values" => Dict("t" => HE_HIP.t, "psi" => HE_HIP.psi,
                                                    "parallax_factor" => HE_HIP.parallax_factor,
                                                    "abscissa_err" => HE_HIP.abscissa_err)),
                              nothing, hg, nothing; fetch = fake)
        @test v.t == HE_HIP.t && v.sinpsi == HE_HIP.sinpsi && v.weight == HE_HIP.weight
        # A failed fetch is an error that says how to proceed, not a fallback.
        boom(h) = error("offline")
        err = try Nereus._hip_scans_for(nothing, nothing, hg, nothing; fetch = boom); nothing
              catch e; e end
        @test err isa ErrorException && occursin("hip_scans: false", err.msg)

        # fit_* channels: `"hip_scans"` reaches Data.
        rv = Dict("source" => "RV", "data" => Dict("RV" => (t = [58000.0, 58100.0, 58200.0],
                  rv = [0.0, 1.0, 2.0], rv_err = [1.0, 1.0, 1.0])))
        pri = Dict("plx" => Dict("type" => "normal", "mu" => 25.0, "sigma" => 0.05),
                   "M_pri" => Dict("type" => "normal", "mu" => 1.0, "sigma" => 0.05))
        as(; kw...) = Dict{String, Any}("source" => "AS", (String(k) => v for (k, v) in kw)...)
        tgt = _he_quiet(() -> Nereus._target_from([rv, as(hgca = hg, hip_scans = HE_HIP)], 1; priors = pri))
        @test tgt.data.hip_scans === HE_HIP
        tgt = _he_quiet(() -> Nereus._target_from([rv, as(hgca = hg, hip_scans = false)], 1; priors = pri))
        @test tgt.data.hip_scans === nothing
        @test_throws ArgumentError Nereus._target_from(
            [rv, as(hgca = hg, hip_scans = HE_HIP), as(hip_scans = false)], 1; priors = pri)
        # run_job: `data.hip_scans`.
        cfg = Dict{String, Any}(
            "hgca" => Dict("values" => Dict("t" => collect(hg.epochs),
                "epochs_ra" => collect(hg.epochs_ra), "epochs_dec" => collect(hg.epochs_dec),
                "pmra" => collect(hg.pmra), "pmdec" => collect(hg.pmdec),
                "cov_ep" => [[c[1, :], c[2, :]] for c in hg.cov_ep],
                "plx" => hg.plx, "plx_err" => hg.plx_err, "hip_id" => hg.hip_id)),
            "hip_scans" => Dict("values" => Dict("t" => HE_HIP.t, "psi" => HE_HIP.psi,
                "parallax_factor" => HE_HIP.parallax_factor, "abscissa_err" => HE_HIP.abscissa_err)),
            "rv" => Dict("values" => Dict("bjd" => [58000.0, 58100.0], "rv" => [0.0, 1.0],
                                          "rv_err" => [1.0, 1.0], "instrument" => ["X", "X"])))
        data = _he_quiet(() -> first(Nereus._build_data(cfg)))
        @test data.hip_scans.t == HE_HIP.t
        @test data.hgca.epochs_ra == hg.epochs_ra && data.hgca.epochs_dec == hg.epochs_dec
        cfg["hip_scans"] = false
        @test _he_quiet(() -> first(Nereus._build_data(cfg))).hip_scans === nothing
    end

    @testset "the catalogue's per-coordinate epochs are carried" begin
        # load_hgca_row on a one-row HGCA table with distinct RA / Dec epochs.
        mktempdir() do dir
            path = joinpath(dir, "hgca.fits")
            cols = Dict{String, Any}(
                "hip_id" => [103836], "epoch_ra_hip" => [1991.203517190048],
                "epoch_dec_hip" => [1991.394347505262],
                "epoch_ra_gaia" => Float32[2015.9588623046875],
                "epoch_dec_gaia" => Float32[2016.105712890625],
                "parallax_gaia" => [27.0], "parallax_gaia_error" => [0.02])
            for (k, v) in (("pmra_hip", 1.0), ("pmdec_hip", 2.0), ("pmra_hg", 3.0),
                           ("pmdec_hg", 4.0), ("pmra_gaia", 5.0), ("pmdec_gaia", 6.0),
                           ("pmra_hip_error", 0.5), ("pmdec_hip_error", 0.4),
                           ("pmra_hg_error", 0.02), ("pmdec_hg_error", 0.03),
                           ("pmra_gaia_error", 0.04), ("pmdec_gaia_error", 0.05),
                           ("pmra_pmdec_hip", 0.1), ("pmra_pmdec_hg", -0.2),
                           ("pmra_pmdec_gaia", 0.3))
                cols[k] = [v]
            end
            FITS(path, "w") do f
                write(f, collect(keys(cols)), collect(values(cols)))
            end
            h = load_hgca_row(path, 103836)
            @test h.epochs_ra == mjd_epochs((1991.203517190048,
                                              (1991.203517190048 + 2015.9588623046875) / 2,
                                              2015.9588623046875))
            @test h.epochs_dec == mjd_epochs((1991.394347505262,
                                               (1991.394347505262 + 2016.105712890625) / 2,
                                               2016.105712890625))
            @test h.epochs == h.epochs_ra          # the nominal tuple is unchanged
            @test h.epochs_ra[1] != h.epochs_dec[1]
        end
        # The struct's own fields round-trip through a client field dict.
        hg = _he_hgca()
        fd = Dict{String, Any}(String(f) => getfield(hg, f) for f in fieldnames(HGCAData))
        back = Nereus._parse_hgca_block(fd)
        @test back.epochs_ra == hg.epochs_ra && back.epochs_dec == hg.epochs_dec
        # Old field dicts (no per-coordinate epochs) take `epochs` for both.
        delete!(fd, "epochs_ra"); delete!(fd, "epochs_dec")
        back = Nereus._parse_hgca_block(fd)
        @test back.epochs_ra == hg.epochs && back.epochs_dec == hg.epochs
        # G23H keeps them too, and defaults to `epochs`.
        @test _he_g23h().epochs_dec == HE_GDEC
        @test _he_g23h(ra = nothing, dec = nothing).epochs_ra == _he_g23h().epochs
        @test_throws ArgumentError _he_hgca(ra = (HE_HGRA[1], HE_HGRA[2], HE_HGRA[1]))
    end

    @testset "ForwardDiff through the Hipparcos fit" begin
        orb_of(Ms) = Nereus.build_orbit(4 * HE_YR, 0.4, 0.7, 1.1, deg2rad(55), 1.0, Ms,
                                    48600.0, 25.0)
        f(Ms) = scan_5param_fit(orb_of(Ms), HE_HIP, Ms; t_ref = (HE_HGRA[1], HE_HGDEC[1]),
                                window = HE_WH)[4]
        h = 1e-6
        @test ForwardDiff.derivative(f, 0.02) ≈ (f(0.02 + h) - f(0.02 - h)) / 2h rtol = 1e-6
        hg = _he_hgca()
        d, θ = _he_quiet(() -> _he_setup(hgca = hg, gost = HE_GOST, hip_scans = HE_HIP))
        ℓ(x) = (t = Theta{eltype(x)}(θ.params); Nereus.set_unfrozen!(t, x);
                hgca_log_likelihood(t, d))
        x0 = Nereus.unfrozen_values(θ)
        gr = ForwardDiff.gradient(ℓ, x0)
        @test all(isfinite, gr) && any(!=(0), gr)
    end
end
