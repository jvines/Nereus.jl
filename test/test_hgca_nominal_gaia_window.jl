# HGCA / G23H without a GOST plan: the Gaia epoch is averaged over a uniform
# nominal scan grid across the release's observing window instead of falling
# back to the instantaneous reflex (wrong whenever the period is not much longer
# than the ~2.8 yr DR3 baseline). A supplied GOST plan is used unchanged.
using Test, Nereus

@testset "no-GOST Gaia epoch: jittered nominal window" begin
    # A 400 d companion: shorter than the DR3 window (1038 d).
    orb = Nereus.build_orbit(400.0, 0.3, 0.7, 1.1, 1.2, 1.0, 0.05, 56000.0, 30.0)
    M_sec = 0.05
    win = Nereus.GAIA_DR3_WINDOW
    t_ra = t_dec = 0.5 * (win[1] + win[2])

    # What it used to be: the instantaneous reflex.
    inst = (Nereus.star_reflex_offset(orb, t_ra, M_sec)...,
            Nereus.star_reflex_pm(orb, t_ra, M_sec)...)
    new = Nereus._catalogue_epoch(orb, M_sec, nothing, win, t_ra, t_dec)

    # The no-GOST path equals averaging over the uniform grid, by the same code
    # path as a GOST plan (scan_5param_fit through synthetic scans).
    g = Nereus._nominal_gaia_scans(win)
    @test all(win[1] .<= g.t .<= win[2])
    ts = unique(g.t)
    @test length(ts) > 100
    h = Nereus.NOMINAL_GAIA_SCAN_STEP_DAYS   # one epoch inside each bin, jittered
    @test all(floor.(Int, (ts .- win[1]) ./ h) .== 0:(length(ts) - 1))
    @test any(abs.(diff(ts) .- h) .> 0.5)
    q = Nereus.scan_5param_fit(orb, g, M_sec; t_ref = (t_ra, t_dec), window = win)
    @test collect(new) == [q[1], q[2], q[4], q[5]]

    # ... and that differs materially from the instantaneous value here.
    @test maximum(abs.(collect(new) .- collect(inst)) ./ (abs.(collect(inst)) .+ 1e-12)) > 0.05

    # A long period is unaffected to first order: the average over the window of
    # a near-linear reflex is the reflex at its centre.
    orbL = Nereus.build_orbit(40000.0, 0.1, 0.7, 1.1, 1.2, 1.0, 0.05, 56000.0, 30.0)
    newL  = Nereus._catalogue_epoch(orbL, M_sec, nothing, win, t_ra, t_dec)
    instL = (Nereus.star_reflex_offset(orbL, t_ra, M_sec)...,
             Nereus.star_reflex_pm(orbL, t_ra, M_sec)...)
    @test isapprox(collect(newL), collect(instL); rtol = 0.02, atol = 1e-3)

    # Hipparcos without scans still falls back to the instantaneous reflex.
    hip = Nereus._catalogue_epoch(orb, M_sec, nothing, Nereus.HIPPARCOS_WINDOW,
                                  48500.0, 48500.0)
    @test collect(hip) == collect((Nereus.star_reflex_offset(orb, 48500.0, M_sec)...,
                                   Nereus.star_reflex_pm(orb, 48500.0, M_sec)...))

    # A supplied GOST plan is used exactly as before.
    gost = GOSTData(t = [56900.0, 57000.0, 57100.0, 57250.0, 57400.0, 57550.0, 57700.0],
                    psi = [0.1, 1.2, 2.3, 0.6, 1.9, 2.8, 0.4],
                    parallax_factor = [0.5, -0.3, 0.8, -0.6, 0.2, 0.9, -0.1])
    with = Nereus._catalogue_epoch(orb, M_sec, gost, win, t_ra, t_dec)
    qg = Nereus.scan_5param_fit(orb, gost, M_sec; t_ref = (t_ra, t_dec), window = win)
    @test collect(with) == [qg[1], qg[2], qg[4], qg[5]]
    @test collect(with) != collect(new)
end

@testset "Data announces the nominal window when GOST is absent" begin
    hgca = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                    pmra = (10.0, 11.0, 12.5), pmdec = (-3.0, -2.5, -2.0),
                    sigma_pmra = (0.5, 0.02, 0.03), sigma_pmdec = (0.4, 0.02, 0.03),
                    corr_pmra_pmdec = (0.2, -0.1, 0.3),
                    plx = 50.0, plx_err = 0.05, hip_id = 85653)
    mk() = Data(t_rv = [58000.0, 58100.0, 58200.0], rv = [0.0, 1.0, 2.0],
                rv_err = [1.0, 1.0, 1.0], hgca = hgca)
    @test_logs (:info, r"NOMINAL") match_mode = :any mk()
end

@testset "no aliasing: a P near 10/n d companion gives no spurious PM" begin
    win = Nereus.GAIA_DR3_WINDOW
    t_mid = 0.5 * (win[1] + win[2])
    # Dense 0.1 d uniform reference (RA and Dec scans at every epoch).
    function grid(ts)
        t = Float64[]; psi = Float64[]; pf = Float64[]
        for x in ts
            ph = 2π * (x - win[1]) / 365.25
            push!(t, x); push!(psi, π / 2); push!(pf, sin(ph))
            push!(t, x); push!(psi, 0.0);   push!(pf, cos(ph))
        end
        GOSTData(; t, psi, parallax_factor = pf)
    end
    ref = grid(win[1]:0.1:win[2])
    old = grid(range(win[1], win[2]; length = ceil(Int, (win[2] - win[1]) / 10.0) + 1))
    M = 0.05
    worst_new = 0.0; worst_old = 0.0
    for P in (10.05, 5.02, 9.97, 3.34)
        orb = Nereus.build_orbit(P, 0.1, 0.7, 1.1, 1.2, 1.0, M, 56000.0, 30.0)
        fit(g) = Nereus.scan_5param_fit(orb, g, M; t_ref = (t_mid, t_mid), window = win)[[4, 5]]
        r = fit(ref)
        n = fit(Nereus._nominal_gaia_scans(win))
        o = fit(old)
        worst_new = max(worst_new, maximum(abs.(n .- r)))
        worst_old = max(worst_old, maximum(abs.(o .- r)))
    end
    
    # Scale: the reflex amplitude itself (mas) of this companion.
    @test worst_new < 0.1 * worst_old       # the old uniform grid aliases
    @test worst_new < 0.01                  # mas/yr (measured 0.0037 vs 0.082 uniform)
    # Deterministic: identical on a rebuild.
    Nereus._NOMINAL_GAIA_SCANS |> empty!
    @test Nereus._nominal_gaia_scans(win).t == Nereus._nominal_gaia_scans(win).t
end
