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

# RV + relative astrometry + HGCA (`kind = :rel`) or RV + G23H (`kind = :g23h`) on
# one source. `a` is the relative semi-major axis [AU]: a·plx is the relative
# orbit on the sky. `σ` is the astrometric error (mas for relAST, mas/yr for
# G23H); `bump` adds a mas/yr offset to the DR2 epoch of G23H.
function _units_target2(a, M_sec; kind = :rel, plx = 20.0, σ = 0.05, bump = 0.0, seed = 4)
    rng = MersenneTwister(seed)
    pl = (a = LogUniformPrior(0.005, 40.0), M_sec = LogUniformPrior(0.0005, 0.5),
          sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
          inc = SinePrior(), Omega = UniformPrior(0.0, 2pi), Mo = UniformPrior(0.0, 2pi))
    P_d = 365.25 * sqrt(a^3 / (1.0 + M_sec))
    t_rv = sort(55000.0 .+ 4 * P_d .* rand(rng, 24))
    rvdat = (t = t_rv, rv = 5 .* sin.(2π .* t_rv ./ P_d) .+ randn(rng, 24),
             rv_err = fill(1.0, 24))
    rv = (X = (data = rvdat, sigma = LogUniformPrior(0.1, 10.0)),)
    epochs5 = mjd_epochs((1991.25, 2004.6, 2015.0, 2015.5, 2016.0))
    cov5(σ) = (c = zeros(10, 10); for i in 1:10; c[i, i] = σ^2; end; c)
    g23h0 = G23HData(; epochs = epochs5, pmra = ntuple(_ -> 0.0, 5),
                     pmdec = ntuple(_ -> 0.0, 5), cov = cov5(max(σ, 1e-3)),
                     plx = plx, plx_err = 0.05, hip_id = 1)
    hg = HGCAData(; epochs = mjd_epochs((1991.25, 2004.0, 2016.0)),
                  pmra = (0.0, 0.0, 0.0), pmdec = (0.0, 0.0, 0.0),
                  sigma_pmra = (0.5, 0.02, 0.05), sigma_pmdec = (0.5, 0.02, 0.05),
                  plx = plx, plx_err = 0.05, hip_id = 1)
    mk(; relast = nothing, g23h = nothing) =
        build_target(M_pri = 1.0, planets = (b = pl,), rv = rv, relAST = relast,
                     hgca = kind === :rel ? hg : nothing, g23h = g23h,
                     plx = NormalPrior(plx, 0.05), M_s = 1.0)
    truth = Dict("a_k1" => a, "M_sec_k1" => M_sec, "sesinw_k1" => 0.3,
                 "secosw_k1" => 0.2, "inc_k1" => 1.1, "Omega_k1" => 2.0,
                 "Mo_k1" => 1.0, "plx" => plx)
    t0 = mk(g23h = kind === :g23h ? g23h0 : nothing,
            relast = kind === :rel ? RelAstromData(t = [55000.0, 55001.0],
                ra_off = [0.0, 0.0], dec_off = [0.0, 0.0],
                ra_err = [σ, σ], dec_err = [σ, σ]) : nothing)
    θ = Theta{Float64}(t0.params)
    for (k, v) in truth
        set_param!(θ, k, v)
    end
    if kind === :rel
        orb, _ = Nereus._planet_orbit(θ, 1, 1.0, plx, t0.data.t_ref)
        tr = sort(55000.0 .+ 4 * P_d .* rand(rng, 12))
        off = [Nereus.relastrom_offset(orb, t) for t in tr]
        relast = RelAstromData(t = tr, ra_off = [o[1] + σ * randn(rng) for o in off],
                               dec_off = [o[2] + σ * randn(rng) for o in off],
                               ra_err = fill(σ, 12), dec_err = fill(σ, 12))
        target = mk(relast = relast)
    else
        μra, μdec = Nereus._g23h_model_pm(θ, g23h0, t0.data, 1.0, plx, t0.data.t_ref)
        bumps = (0.0, 0.0, bump, 0.0, 0.0)
        g = G23HData(; epochs = epochs5,
                     pmra = ntuple(i -> μra[i] + 3.0 + bumps[i] + σ * randn(rng), 5),
                     pmdec = ntuple(i -> μdec[i] - 2.0 + σ * randn(rng), 5),
                     cov = cov5(σ), plx = plx, plx_err = 0.05, hip_id = 1)
        target = mk(g23h = g)
    end
    nm = target.params.layout.unfrozen_names
    arr = zeros(3, length(nm) + 1, 1)
    for i in 1:3
        arr[i, 1:end-1, 1] = [get(truth, x, 1.0) for x in nm]
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

    # tick labels an axis would draw: nothing while the format is Makie's default
    function _clean_ticks(ax, dim)
        r = ax.finallimits[]
        i = dim === :x ? 1 : 2
        vals = collect(range(r.origin[i], r.origin[i] + r.widths[i]; length = 5))
        f = dim === :x ? ax.xtickformat[] : ax.ytickformat[]
        f === _UMk.automatic && return true
        labs = f(vals)
        return all(l -> !occursin(r"0{5,}|9{5,}", l), labs)
    end

    @testset "plot_g23h_residuals: μas/yr for a quiet source, mas/yr with a mas/yr offset" begin
        target, chains = _units_target2(3.0, 0.001; kind = :g23h, σ = 0.004)
        fig = plot_g23h_residuals(chains, target.params, target.data)
        ax = _axes(fig)
        @test length(ax) == 2
        @test all(a -> occursin("(μas/yr)", _ylab(a)), ax)
        @test all(a -> _clean_ticks(a, :y), ax)
        @test all(a -> a.ytickformat[] !== _UMk.automatic, ax)

        target, chains = _units_target2(4.0, 0.05; kind = :g23h, σ = 0.3, bump = 4.0)
        fig = plot_g23h_residuals(chains, target.params, target.data)
        ax = _axes(fig)
        @test all(a -> occursin("(mas/yr)", _ylab(a)) && !occursin("μas", _ylab(a)), ax)
        @test all(a -> a.ytickformat[] === _UMk.automatic, ax)
    end

    @testset "plot_relastrom_timeseries / _residuals: mas for a wide orbit, μas for a tiny one" begin
        for (a, σ, unit) in ((4.0, 0.5, "mas"), (0.02, 0.01, "μas"))
            target, chains = _units_target2(a, 0.001; kind = :rel, σ = σ)
            fig = plot_relastrom_timeseries(chains, target.params, target.data;
                                            n_draws = 3)
            ax = _axes(fig)
            sep = first(a for a in ax if startswith(_ylab(a), "Separation"))
            @test _ylab(sep) == "Separation ($unit)"
            @test _clean_ticks(sep, :y)
            @test any(a -> _ylab(a) == "PA (deg)", ax)      # degrees untouched
            fig = plot_relastrom_residuals(chains, target.params, target.data)
            sep = first(a for a in _axes(fig) if startswith(_ylab(a), "Δρ"))
            @test occursin(r"Δρ \((mas|μas)\)", _ylab(sep))
            @test _clean_ticks(sep, :y)
        end
    end

    @testset "plot_rv_astrom_phasefold and plot_orbit_skyplane: sky unit follows the orbit" begin
        for (a, unit) in ((0.02, "μas"), (4.0, "mas"))
            target, chains = _units_target2(a, 0.001; kind = :rel, σ = 0.01)
            fig = plot_rv_astrom_phasefold(chains, target.params, target.data;
                                           n_draws = 5)
            sky = first(x for x in _axes(fig) if occursin("ΔRA", _xlab(x)))
            @test _xlab(sky) == "ΔRA·cos δ ($unit)" && _ylab(sky) == "Δδ ($unit)"
            @test _clean_ticks(sky, :x) && _clean_ticks(sky, :y)
            @test !any(x -> occursin("(μas)", _ylab(x)) || occursin("(mas)", _ylab(x)),
                       [x for x in _axes(fig) if x !== sky])      # RV panel untouched

            fig = plot_orbit_skyplane(chains, target.params, target.data)
            sky = only(_axes(fig))
            @test _xlab(sky) == "ΔRA·cos δ ($unit)" && _ylab(sky) == "Δδ ($unit)"
            @test _clean_ticks(sky, :x) && _clean_ticks(sky, :y)
        end
        # the same orbit as an absolute-astrometry planet (no relAST)
        target, chains = _units_target(0.02, 0.001)
        sky = only(_axes(plot_orbit_skyplane(chains, target.params, target.data)))
        @test _xlab(sky) == "ΔRA·cos δ (μas)"
    end
end
