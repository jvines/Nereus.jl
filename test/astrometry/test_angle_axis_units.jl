# Astrometric angle axes read in the unit their extent calls for.
#
# Data and limits stay in mas. Once a figure is drawn, `_angle_units!` looks at
# a group's final axis limits: under 1 mas the group is read in μas (tick labels
# ×1000, axis label μas), otherwise in mas. A group that shows one quantity (a
# sky panel's x and y, the RA and Dec panels of a proper-motion figure, the
# panels of an IAD-residual figure) shares one unit.

using Nereus, Test, Random, MCMCChains
using Nereus: IADData, Axis, Figure, limits!
const _UMk = Nereus.CairoMakie.Makie

# Two-mission IAD (Hipparcos-like, Gaia-like) plus HGCA and GOST on one source.
# `a` [AU] and `M_sec` [Msun] scale the astrometric signal: a Jupiter at 3 AU
# around a 1 Msun star at 50 pc reflexes by tens of μas; 0.05 Msun at 4 AU by mas.
function _units_target(a, M_sec; plx = 20.0, seed = 3)
    rng = MersenneTwister(seed)
    t, psi, inst, σ = Float64[], Float64[], Int[], Float64[]
    for tk in sort(47900.0 .+ 1150 .* rand(rng, 30))
        ψ = 2π * rand(rng)
        for c in 0:1
            push!(t, tk); push!(psi, ψ + 2e-3 * c); push!(inst, 1); push!(σ, 1.0)
        end
    end
    for tk in sort(56900.0 .+ 1800 .* rand(rng, 30))
        ψ = 2π * rand(rng)
        for c in 0:8
            push!(t, tk + c * 5.6e-5); push!(psi, ψ); push!(inst, 2); push!(σ, 0.05)
        end
    end
    n = length(t)
    plxf = sin.(2π .* (t .- 57000.0) ./ 365.25 .- psi)
    pmf  = (t .- 57388.5) ./ 365.25
    iad = IADData(t = t, abscissa = zeros(n), abscissa_err = σ, psi = psi,
                  parallax_factor = plxf, pm_factor = pmf, inst = inst,
                  ref_params = [ntuple(_ -> 0.0, 5) for _ in 1:2],
                  abscissa_kind = fill(:absolute, 2))
    hg = HGCAData(; epochs = mjd_epochs((1991.25, 2004.0, 2016.0)),
                  pmra = (0.0, 0.0, 0.0), pmdec = (0.0, 0.0, 0.0),
                  sigma_pmra = (0.5, 0.02, 0.05), sigma_pmdec = (0.5, 0.02, 0.05),
                  plx = plx, plx_err = 0.05, hip_id = 1)
    tg = collect(range(mjd_epochs((2014.7,))[1], mjd_epochs((2017.3,))[1]; length = 40))
    gost = GOSTData(; t = tg, psi = 2π .* ((1:40) ./ 41), parallax_factor = sin.(tg ./ 58.0))
    pl = (a = LogUniformPrior(0.3, 40.0), M_sec = LogUniformPrior(0.0005, 0.5),
          sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
          inc = SinePrior(), Omega = UniformPrior(0.0, 2pi),
          Mo = UniformPrior(0.0, 2pi))
    target = build_target(M_pri = 1.0, planets = (b = pl,), iad = iad, hgca = hg,
                          gost = gost, plx = NormalPrior(plx, 0.05), M_s = 1.0)
    truth = Dict("a_k1" => a, "M_sec_k1" => M_sec, "sesinw_k1" => 0.3,
                 "secosw_k1" => 0.2, "inc_k1" => 1.1, "Omega_k1" => 2.0,
                 "Mo_k1" => 1.0, "plx" => plx)
    θ = Theta{Float64}(target.params)
    for (k, v) in truth
        set_param!(θ, k, v)
    end
    _, orbs, Ms = Nereus._iad_active_orbits(θ, 1.0, plx, target.data.t_ref)
    for j in 1:n
        reflex = Nereus.along_scan_projection(
            Nereus.star_reflex_offset(orbs[1], t[j], Ms[1])..., psi[j])
        target.data.iad.abscissa[j] = reflex + plx * plxf[j] + σ[j] * randn(rng)
    end
    μra, μdec = Nereus._hgca_model_pm(θ, target.data.hgca, target.data, 1.0, plx,
                                      target.data.t_ref)
    hg2 = HGCAData(; epochs = mjd_epochs((1991.25, 2004.0, 2016.0)),
                   pmra = Tuple(μra .+ 3.0), pmdec = Tuple(μdec .- 2.0),
                   sigma_pmra = (0.5, 0.02, 0.05), sigma_pmdec = (0.5, 0.02, 0.05),
                   plx = plx, plx_err = 0.05, hip_id = 1)
    target = build_target(M_pri = 1.0, planets = (b = pl,), iad = target.data.iad,
                          hgca = hg2, gost = gost, plx = NormalPrior(plx, 0.05), M_s = 1.0)
    nm = target.params.layout.unfrozen_names
    arr = zeros(3, length(nm) + 1, 1)
    for i in 1:3
        arr[i, 1:end-1, 1] = [truth[x] for x in nm]
        arr[i, end, 1] = i == 1 ? 0.0 : -10.0 * i
    end
    return target, Chains(arr, vcat(Symbol.(nm), :lp))
end

_axes(fig) = [c for c in fig.content if c isa Axis]
_str(x) = x isa AbstractString ? String(x) : join(_str.(x.children))   # rich text
_ylab(ax) = _str(ax.ylabel[])
_xlab(ax) = _str(ax.xlabel[])
# the labels an axis would draw on its y ticks
function _yticklabels(ax)
    r = ax.finallimits[]
    vals = collect(range(r.origin[2], r.origin[2] + r.widths[2]; length = 5))
    f = ax.ytickformat[]
    return f === _UMk.automatic ? nothing : f(vals)
end

@testset "angle axis units" begin
    @testset "helper: sub-mas axis is read in μas, labels clean" begin
        fig = Figure(); ax = Axis(fig[1, 1]); limits!(ax, -1, 1, -0.3, 0.3)
        u = Nereus._angle_units!([ax]; dims = :y, label = u -> "Δ ($u)")
        @test u == "μas"
        @test ax.ylabel[] == "Δ (μas)"
        @test ax.xlabel[] != "Δ (μas)"
        labs = ax.ytickformat[]([-0.2, 0.0, 0.002, 0.1, 0.2])
        @test labs == ["-200", "0", "2", "100", "200"] || labs == ["−200", "0", "2", "100", "200"]
        @test all(l -> !occursin("0000", l), labs)
        @test ax.xtickformat[] === _UMk.automatic       # x not in dims
    end

    @testset "helper: mas-scale axis is untouched" begin
        fig = Figure(); ax = Axis(fig[1, 1]); limits!(ax, -1, 1, -5, 5)
        @test Nereus._angle_units!([ax]; dims = :y, label = u -> "Δ ($u)") == "mas"
        @test ax.ylabel[] == "Δ (mas)"
        @test ax.ytickformat[] === _UMk.automatic
    end

    @testset "helper: group shares the unit of its largest axis" begin
        fig = Figure()
        a1 = Axis(fig[1, 1]); a2 = Axis(fig[2, 1])
        limits!(a1, 0, 1, -0.05, 0.05); limits!(a2, 0, 1, -3, 3)
        @test Nereus._angle_units!([a1, a2]; dims = :y, label = u -> "r ($u)") == "mas"
        @test a1.ylabel[] == "r (mas)" && a2.ylabel[] == "r (mas)"
        b1 = Axis(fig[3, 1]); b2 = Axis(fig[4, 1])
        limits!(b1, 0, 1, -0.05, 0.05); limits!(b2, 0, 1, -0.4, 0.4)
        @test Nereus._angle_units!([b1, b2]; dims = :y, label = u -> "r ($u)") == "μas"
        @test b1.ylabel[] == "r (μas)" && b2.ylabel[] == "r (μas)"
    end

    @testset "helper: a sky panel's x and y share one unit, flipped x included" begin
        fig = Figure(); ax = Axis(fig[1, 1]); Nereus._flip_xaxis!(ax)
        Nereus._sky_limits!(ax, -0.02, 0.02, -0.5, 0.5)   # y alone is 0.5 mas, x 0.02
        @test Nereus._sky_units!(ax) == "μas"
        @test _xlab(ax) == "ΔRA·cos δ (μas)" && _ylab(ax) == "Δδ (μas)"
        fig = Figure(); ax = Axis(fig[1, 1]); Nereus._flip_xaxis!(ax)
        Nereus._sky_limits!(ax, -0.02, 0.02, -3.0, 3.0)   # y is large, x small
        @test Nereus._sky_units!(ax) == "mas"
        @test _xlab(ax) == "ΔRA·cos δ (mas)"
    end

    @testset "plot_pm_anomaly: μas/yr for a Jupiter at 50 pc, mas/yr for a brown dwarf" begin
        target, chains = _units_target(3.0, 0.001)
        fig = plot_pm_anomaly(chains, target.params, target.data; planet_idx = 1, n_draws = 3)
        ax = _axes(fig)
        @test length(ax) == 2
        @test all(a -> occursin("μas/yr", _ylab(a)), ax)
        @test _yticklabels(ax[1]) !== nothing

        target, chains = _units_target(4.0, 0.05)
        fig = plot_pm_anomaly(chains, target.params, target.data; planet_idx = 1, n_draws = 3)
        ax = _axes(fig)
        @test all(a -> occursin("(mas/yr)", _ylab(a)) && !occursin("μas", _ylab(a)), ax)
        @test all(a -> a.ytickformat[] === _UMk.automatic, ax)
    end

    @testset "plot_iad_residuals: shared unit across both panels" begin
        # the catalogue solution is marginalised, so residuals are the noise: mas
        target, chains = _units_target(4.0, 0.05)
        fig = plot_iad_residuals(chains, target.params, target.data)
        ax = _axes(fig)
        @test length(ax) == 2
        @test _ylab(ax[1]) == _ylab(ax[2])
        @test occursin("abscissa residual (", _ylab(ax[1]))
    end

    @testset "plot_astrometry_model: O−C per mission, sky panel in the orbit's unit" begin
        target, chains = _units_target(3.0, 0.001)
        fig = plot_astrometry_model(chains, target.params, target.data; planet_idx = 1)
        ax = _axes(fig)
        sky = [a for a in ax if occursin("ΔRA", _xlab(a))]
        @test !isempty(sky)
        @test all(a -> occursin("(μas)", _xlab(a)) && occursin("(μas)", _ylab(a)), sky)
        oc = [a for a in ax if startswith(_ylab(a), "O−C")]
        @test length(oc) == 2
        @test all(a -> occursin(r"\((mas|μas)\)", _ylab(a)), oc)
        # Gaia's abscissae are 0.05 mas: its strip is read in μas, Hipparcos' in mas
        @test any(a -> _ylab(a) == "O−C (μas)", oc)
        @test any(a -> _ylab(a) == "O−C (mas)", oc)
    end

    @testset "plot_pm_residuals: HGCA Δμ panels share a unit" begin
        target, chains = _units_target(3.0, 0.001)
        fig = plot_pm_residuals(chains, target.params, target.data; planet_idx = 1)
        ax = _axes(fig)
        @test length(ax) == 2
        units = [match(r"\((μas|mas)/yr\)", _ylab(a))[1] for a in ax]
        @test units[1] == units[2]
    end
end
