# The PM-anomaly figure shows Gaia DR2 and DR3 as GOST projects them.
#
# A GOST scan plan covers the whole mission (the fetcher's default runs to
# 2025), but DR2 and DR3 were fitted to 22 and 34 months of it. Each release's
# projection goes through the scans inside its own window, drawn in its own
# colour and marker.

using Nereus, Test, MCMCChains

const _Mk = Nereus.CairoMakie.Makie

@testset "PM anomaly: Gaia DR2/DR3 projected through their GOST windows" begin
    # A scan plan from before DR2 to the end of the extended mission.
    tg = collect(range(56800.0, 60600.0; length = 160))
    gost = GOSTData(; t = tg, psi = 2π .* mod.((1:160) .* 0.618034, 1.0),
                    parallax_factor = sin.(tg ./ 58.0))

    w2 = Nereus._gost_window(gost, Nereus.GAIA_DR2_WINDOW)
    w3 = Nereus._gost_window(gost, Nereus.GAIA_DR3_WINDOW)
    inwin(t, w) = w[1] <= t <= w[2]
    @test n_gost(w2) == count(t -> inwin(t, Nereus.GAIA_DR2_WINDOW), tg)
    @test n_gost(w3) == count(t -> inwin(t, Nereus.GAIA_DR3_WINDOW), tg)
    @test all(t -> inwin(t, Nereus.GAIA_DR3_WINDOW), w3.t)
    @test n_gost(w2) < n_gost(w3) < n_gost(gost)
    # too few scans for a five-parameter fit: no projection, not a zero one
    @test Nereus._gost_window(gost, (50000.0, 50030.0)) === nothing

    hg = HGCAData(; epochs = mjd_epochs((1991.25, 2004.0, 2016.0)),
                  pmra = (0.0, 0.0, 0.0), pmdec = (0.0, 0.0, 0.0),
                  sigma_pmra = (0.5, 0.02, 0.05), sigma_pmdec = (0.5, 0.02, 0.05),
                  plx = 20.0, plx_err = 0.05, hip_id = 1)
    pl = (a = LogUniformPrior(0.3, 40.0), M_sec = LogUniformPrior(0.001, 0.5),
          sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
          inc = SinePrior(), Omega = UniformPrior(0.0, 2pi),
          Mo = UniformPrior(0.0, 2pi))
    target = build_target(M_pri = 1.0, planets = (b = pl,), hgca = hg, gost = gost,
                          plx = NormalPrior(20.0, 0.05), M_s = 1.0)
    # P ≈ 8 yr: a 34-month average and a 10-yr one differ by mas/yr.
    truth = Dict("a_k1" => 4.0, "M_sec_k1" => 0.05, "sesinw_k1" => 0.3,
                 "secosw_k1" => 0.2, "inc_k1" => 1.1, "Omega_k1" => 2.0,
                 "Mo_k1" => 1.0, "plx" => 20.0)
    nm = target.params.layout.unfrozen_names
    arr = zeros(3, length(nm) + 1, 1)
    for i in 1:3
        arr[i, 1:end-1, 1] = [truth[x] for x in nm]
        arr[i, end, 1] = -10.0 * i
    end
    chains = Chains(arr, vcat(Symbol.(nm), :lp))

    θ = Theta{Float64}(target.params)
    for (k, v) in truth
        set_param!(θ, k, v)
    end
    _, orbs, Ms = Nereus._iad_active_orbits(θ, astrom_M_pri(θ), astrom_plx(θ),
                                            target.data.t_ref)
    want(gw) = Nereus._dr_reflex_pm(orbs, Ms, gw)
    dr2, dr3, full = want(w2), want(w3), want(gost)
    # the fixture can tell the windows apart
    @test abs(dr3[1] - full[1]) + abs(dr3[2] - full[2]) > 0.1

    fig = plot_pm_anomaly(chains, target.params, target.data; planet_idx = 1,
                          n_draws = 5)
    axes = [c for c in fig.content if c isa Nereus.Axis]
    @test length(axes) == 2
    # a projection is the one single-point scatter at its release's epoch
    proj(ax, ep) = only(p for p in ax.scene.plots
                        if p isa _Mk.Scatter && length(p[1][]) == 1 &&
                           abs(p[1][][1][1] - jyear_to_mjd(ep)) < 0.5)
    for (ax, comp) in zip(axes, (1, 2)), (val, ep) in ((dr2, 2015.5), (dr3, 2016.0))
        y = proj(ax, ep)[1][][1][2]
        @test y ≈ val[comp] atol = 1e-4 * (1 + abs(val[comp]))
    end
    # each release in its own colour and marker, labelled once
    p2 = proj(axes[1], 2015.5); p3 = proj(axes[1], 2016.0)
    @test p2[:color][] != p3[:color][]
    @test repr(p2[:marker][]) != repr(p3[:marker][])
    @test p2[:label][] == "Gaia DR2 (GOST)" && p3[:label][] == "Gaia DR3 (GOST)"
end
