# The astrometry model figure: the reflex orbit on the sky with the epoch
# astrometry of every mission drawn as BINNED ABSCISSA LINES along their scan
# axes, a zoom on the most precise mission, and each mission's along-scan O−C
# against epoch -- astroEMPEROR's `astrometry_model`, with the catalogue layer
# (DR2/DR3 positions and proper-motion wedges, the GOST model) replaced by the
# intermediate astrometric data itself.
#
# Pinned here: missions are told apart by epoch; Hipparcos records of one
# satellite orbit, which differ in scan angle by a few 1e-3 rad, bin into one
# line while a real change of scan direction never does; the O−C drawn are the
# likelihood's own and vanish on noiseless data; the combined figure and every
# panel on its own are written; and the runner knows the figure by name.

using Nereus, Test, Random, MCMCChains, Statistics
using Nereus: IADData

# Two missions on one source. Instrument 1 is Hipparcos-like: ~1990-1993,
# 2-3 abscissa records per satellite orbit at scan angles a few 1e-3 rad apart,
# mas-level errors. Instrument 2 is Gaia-like: 2014-2019, field-of-view transits
# of 9 CCD abscissae at one scan angle. `gaia_only` drops instrument 1.
function _two_mission_target(; noise = true, seed = 5, gaia_only = false,
                               σ_hip = 1.5, σ_gaia = 0.08)
    rng = MersenneTwister(seed)
    t, psi, inst, σ = Float64[], Float64[], Int[], Float64[]
    if !gaia_only
        for tk in sort(47900.0 .+ 1150 .* rand(rng, 35))
            ψ = 2π * rand(rng)
            for c in 0:rand(rng, 1:2)
                push!(t, tk); push!(psi, ψ + 2e-3 * c); push!(inst, 1); push!(σ, σ_hip)
            end
        end
    end
    g = gaia_only ? 1 : 2
    for tk in sort(56900.0 .+ 1800 .* rand(rng, 40))
        ψ = 2π * rand(rng)
        for c in 0:8
            push!(t, tk + c * 5.6e-5); push!(psi, ψ); push!(inst, g); push!(σ, σ_gaia)
        end
    end
    n = length(t)
    plxf = sin.(2π .* (t .- 57000.0) ./ 365.25 .- psi)
    pmf  = (t .- 57388.5) ./ 365.25
    n_inst = gaia_only ? 1 : 2
    iad = IADData(t = t, abscissa = zeros(n), abscissa_err = σ, psi = psi,
                  parallax_factor = plxf, pm_factor = pmf, inst = inst,
                  ref_params = [ntuple(_ -> 0.0, 5) for _ in 1:n_inst],
                  abscissa_kind = fill(:absolute, n_inst))
    pl = (a = LogUniformPrior(0.3, 6.0), M_sec = LogUniformPrior(0.001, 0.05),
          sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
          inc = SinePrior(), Omega = UniformPrior(0.0, 2pi),
          Mo = UniformPrior(0.0, 2pi))
    target = build_target(M_pri = 0.644, planets = (b = pl,), iad = iad,
                          plx = NormalPrior(13.6, 0.02), M_s = 0.644)
    truth = Dict("a_k1" => 2.4, "M_sec_k1" => 0.011, "sesinw_k1" => 0.45,
                 "secosw_k1" => 0.3, "inc_k1" => 2.1, "Omega_k1" => 3.0,
                 "Mo_k1" => 1.0, "plx" => 13.6)
    theta = Theta{Float64}(target.params)
    for (k, v) in truth
        set_param!(theta, k, v)
    end
    _, orbs, M_secs = Nereus._iad_active_orbits(theta, 0.644, 13.6, target.data.t_ref)
    for j in 1:n
        reflex = Nereus.along_scan_projection(
            Nereus.star_reflex_offset(orbs[1], t[j], M_secs[1])..., psi[j])
        zp = inst[j] == 1 ? (1.3, -0.7) : (-4.0, 2.5)
        target.data.iad.abscissa[j] = reflex + 13.6 * plxf[j] +
            zp[1] * sin(psi[j]) + zp[2] * cos(psi[j]) +
            40.0 * sin(psi[j]) * pmf[j] - 25.0 * cos(psi[j]) * pmf[j] +
            (noise ? σ[j] * randn(rng) : 0.0)
    end
    nm = target.params.layout.unfrozen_names
    rows = [[truth[x] for x in nm]]
    for _ in 1:5
        push!(rows, [truth[x] * (1 + 0.05 * randn(rng)) for x in nm])
    end
    arr = zeros(length(rows), length(nm) + 1, 1)
    for (i, r) in enumerate(rows)
        arr[i, 1:length(nm), 1] = r
        arr[i, end, 1] = i == 1 ? 0.0 : -100.0 * i
    end
    return target, theta, Chains(arr, vcat(Symbol.(nm), :lp))
end

_am_files(dir) = isdir(joinpath(dir, "models")) ?
    sort(filter(f -> startswith(f, "astrometry_model"), readdir(joinpath(dir, "models")))) :
    String[]

@testset "astrometry model figure" begin

    @testset "missions are named by epoch" begin
        target, _, _ = _two_mission_target()
        iad = target.data.iad
        @test Nereus._iad_mission_names(iad) == ["Hipparcos", "Gaia"]
        @test Nereus._iad_mission_names(iad; names = ["HIP", "DR4"]) == ["HIP", "DR4"]
        @test_throws ArgumentError Nereus._iad_mission_names(iad; names = ["only one"])
        g, _, _ = _two_mission_target(gaia_only = true)
        @test Nereus._iad_mission_names(g.data.iad) == ["Gaia"]
        @test Nereus._mjd_to_jyear(51544.5) == 2000.0
        @test Nereus._mjd_to_jyear(57388.5) ≈ 2016.0
    end

    @testset "abscissae bin per epoch within a scan-angle tolerance" begin
        target, _, _ = _two_mission_target()
        iad = target.data.iad
        e = zeros(n_iad(iad))
        n_hip_epochs = length(unique(iad.t[iad.inst .== 1]))
        n_hip_recs   = count(==(1), iad.inst)
        @test n_hip_recs > n_hip_epochs
        # The default tolerance is one Gaia FoV transit's: Hipparcos records of
        # one orbit, 2e-3 rad apart, stay separate.
        strict = Nereus._iad_normal_points(iad, e, 0.01)
        @test count(==(1), strict.inst) > n_hip_epochs
        # The figure's tolerance bins them: one line per satellite orbit.
        loose = Nereus._iad_normal_points(iad, e, 0.01; psi_tol = 0.02)
        @test count(==(1), loose.inst) == n_hip_epochs
        @test count(==(2), loose.inst) == 40                 # one per FoV transit
        @test sum(loose.size) == n_iad(iad)
        # Binning shrinks the error bar as 1/√n and never mixes instruments.
        g = findfirst(i -> loose.inst[i] == 2, eachindex(loose.inst))
        @test loose.σ[g] ≈ 0.08 / sqrt(9)
    end

    @testset "what is drawn is the likelihood's O−C" begin
        target, theta, chains = _two_mission_target(noise = false)
        S = Nereus._astrometry_model_scene(chains, target.params, target.data, 1)
        @test S !== nothing
        @test S.oc ≈ Nereus._iad_oc(theta, target.data).oc atol = 1e-12
        # Noiseless data: every binned abscissa sits on the model orbit.
        @test maximum(abs, S.np.e) < 1e-6
        @test maximum(hypot.(S.xn .- S.xm, S.yn .- S.ym)) < 1e-6
        @test S.names == ["Hipparcos", "Gaia"]
        @test S.zoom == 2                       # the more precise mission

        # With noise the binned O−C scatter as their own error bars say.
        target, theta, chains = _two_mission_target()
        S = Nereus._astrometry_model_scene(chains, target.params, target.data, 1)
        for m in 1:2
            sel = S.np.inst .== m
            @test 0.5 < std(S.np.e[sel] ./ S.np.σ[sel]) < 1.5
        end
        # The binned abscissa is drawn AT model + (O−C)·u, along the scan axis.
        g = 1
        @test S.xn[g] ≈ S.xm[g] + S.np.e[g] * S.np.ux[g]
        @test S.yn[g] ≈ S.ym[g] + S.np.e[g] * S.np.uy[g]
    end

    @testset "combined figure and every panel on its own" begin
        target, _, chains = _two_mission_target()
        out = mktempdir()
        plot_astrometry_model(chains, target.params, target.data; output = out)
        @test _am_files(out) == ["astrometry_model_K1.png",
                                 "astrometry_model_K1_oc_gaia.png",
                                 "astrometry_model_K1_oc_hipparcos.png",
                                 "astrometry_model_K1_sky.png",
                                 "astrometry_model_K1_sky_gaia.png"]
        # panels = false: the combined figure alone.
        out2 = mktempdir()
        plot_astrometry_model(chains, target.params, target.data; output = out2,
                              panels = false)
        @test _am_files(out2) == ["astrometry_model_K1.png"]

        # One mission: there is nothing to zoom on, so no zoom panel.
        g, _, gch = _two_mission_target(gaia_only = true)
        out3 = mktempdir()
        plot_astrometry_model(gch, g.params, g.data; output = out3)
        @test _am_files(out3) == ["astrometry_model_K1.png",
                                  "astrometry_model_K1_oc_gaia.png",
                                  "astrometry_model_K1_sky.png"]
    end

    @testset "sky panels keep East to the left" begin
        sky(fig) = [c for c in fig.content
                    if c isa Nereus.Axis && occursin("ΔRA", string(c.xlabel[]))]
        target, _, chains = _two_mission_target()
        fig = plot_astrometry_model(chains, target.params, target.data)
        @test length(sky(fig)) == 2            # (a) and the zoom
        @test all(a -> a.xreversed[], sky(fig))
        g, _, gch = _two_mission_target(gaia_only = true)
        fig = plot_astrometry_model(gch, g.params, g.data)
        @test length(sky(fig)) == 1
        @test all(a -> a.xreversed[], sky(fig))
    end

    @testset "nothing to draw writes nothing" begin
        target, _, chains = _two_mission_target()
        out = mktempdir()
        # slot 2 does not exist
        plot_astrometry_model(chains, target.params, target.data; planet_idx = 2,
                              output = out)
        @test isempty(_am_files(out))
        # no IAD at all
        t = collect(0.0:5.0:300.0)
        rvt = build_target(
            planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                            sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                            Mo = UniformPrior(0.0, 2pi)),),
            rv = (SIM = (data = (t = t, rv = sin.(t), rv_err = fill(1.0, length(t))),
                         sigma = LogUniformPrior(0.5, 10.0)),))
        @test Nereus._astrometry_model_scene(chains, rvt.params, rvt.data, 1) === nothing
    end

    @testset "the runner knows it by name" begin
        target, _, chains = _two_mission_target()
        out = mktempdir()
        @test Nereus._dispatch_plot("astrometry_model", chains, target.params, target.data,
                                    out, Dict{Symbol,Any}()) == "models/astrometry_model_K*.png"
        @test length(_am_files(out)) == 5
        @test "astrometry_model" in Nereus._auto_plot_kinds(chains, target.params, target.data)
        @test "astrometry_model" in Nereus._KNOWN_PLOTS

        # plot_kwargs reach it: panels = false through the runner.
        out2 = mktempdir()
        cfg = Dict("output" => Dict("plots" => ["astrometry_model"],
                                    "plot_kwargs" => Dict("panels" => false),
                                    "show_progress" => false))
        made = Nereus._make_plots(cfg, chains, target.params, target.data, out2)
        @test made == ["models/astrometry_model_K*.png"]
        @test _am_files(joinpath(out2, "plots")) == ["astrometry_model_K1.png"]
    end
end
