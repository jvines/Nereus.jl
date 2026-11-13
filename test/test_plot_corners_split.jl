# run_job's "corner" draws one corner per planet slot and one of the noise model
# (`plot_corners`, src/plotting/diagnostics.jl), never the all-parameter corner:
# its n² panels were most of the plotting stage's memory. Pinned here: which
# parameters go in which corner, the draws a trans-dimensional planet's corner
# uses, the files run_job writes and advertises, and the axes (unit labels,
# angles in degrees).

using Nereus, Test, Random, MCMCChains
using CairoMakie

const _SC_MAKIE = Nereus.CairoMakie.Makie

# Two RV planets, two instruments, an FM17 rotation GP.
function _sc_rv_params(; max_k = 2)
    rng = MersenneTwister(2)
    n = 40
    t = sort(300.0 .* rand(rng, n))
    data = Data(; t_rv = t, rv = randn(rng, n), rv_err = fill(2.0, n),
                  rv_inst = [isodd(i) ? 1 : 2 for i in 1:n])
    params = Params(; parametrization = ParametrizationConfig(time = :Mo),
                      max_kplanet = max_k, planet_modes = fill(RV_ONLY, max_k),
                      instruments = InstrumentConfig(rv = ["A", "B"]),
                      noise_models = NoiseModel[CeleriteRotationFM17()],
                      data = data, M_s = 1.0)
    return params, data
end

# Draws of every reported parameter scattered round `centre(name)`, plus `extra`
# columns (trans-dim bookkeeping) and an `lp`.
function _sc_chain(params, n; extra = Pair{Symbol, Vector{Float64}}[],
                   centre = _ -> 1.0, seed = 3)
    rng = MersenneTwister(seed)
    nm = Nereus.reported_param_names(params)
    cols = Symbol[Symbol.(nm); first.(extra); :lp]
    arr = zeros(n, length(cols), 1)
    for (j, x) in enumerate(nm)
        c = centre(x)
        arr[:, j, 1] .= c .+ 0.01 .* max(abs(c), 0.1) .* randn(rng, n)
    end
    for (j, (_, v)) in enumerate(extra)
        arr[:, length(nm) + j, 1] .= v
    end
    arr[:, end, 1] .= -0.5 .* randn(rng, n) .^ 2
    return Chains(arr, cols)
end

_sc_stems(groups) = [g.stem for g in groups]
_sc_group(groups, stem) = groups[findfirst(g -> g.stem == stem, groups)]

@testset "split corners" begin
    @testset "one corner per planet and one of the noise model" begin
        params, data = _sc_rv_params()
        ch = _sc_chain(params, 2_000)
        groups = Nereus._corner_groups(ch, params)
        @test _sc_stems(groups) == ["corner_K1", "corner_K2", "corner_noise"]
        for k in 1:2
            g = _sc_group(groups, "corner_K$k")
            @test g.names == ["P_k$k", "K_k$k", "sesinw_k$k", "secosw_k$k", "Mo_k$k"]
            @test g.draws === nothing                    # fixed dimension: every draw
        end
        g = _sc_group(groups, "corner_noise")
        @test g.names == ["sigma_A", "sigma_B", "gp_log_amp", "gp_log_timescale",
                          "gp_log_period", "gp_log_factor"]
        @test g.draws === nothing
        # Offsets are in no corner.
        @test !any(g -> any(startswith("gamma"), g.names), groups)
    end

    @testset "a model without two noise parameters has no noise corner" begin
        rng = MersenneTwister(1)
        data = Data(; t_rv = collect(1.0:30.0), rv = randn(rng, 30),
                      rv_err = fill(2.0, 30), rv_inst = ones(Int, 30))
        params = Params(; parametrization = ParametrizationConfig(time = :Mo),
                          max_kplanet = 1, planet_modes = [RV_ONLY],
                          instruments = InstrumentConfig(rv = ["A"]),
                          data = data, M_s = 1.0)
        @test _sc_stems(Nereus._corner_groups(_sc_chain(params, 500), params)) ==
              ["corner_K1"]
    end

    @testset "rho_circ and eta go in their planet's corner" begin
        rng = MersenneTwister(5)
        t = collect(0.0:0.01:20.0)
        data = Data(; t_phot = t, flux = 1.0 .+ 1e-4 .* randn(rng, length(t)),
                      flux_err = fill(1e-4, length(t)), phot_inst = ones(Int, length(t)))
        params = Params(; parametrization = ParametrizationConfig(time = :Mo,
                                                                    density = :rho_circ),
                          max_kplanet = 2, planet_modes = fill(PM_ONLY, 2),
                          instruments = InstrumentConfig(pm = ["TESS"]),
                          data = data, M_s = 1.0, R_s = 1.0)
        @test Nereus.density_active(params)
        groups = Nereus._corner_groups(_sc_chain(params, 500), params)
        owner = Nereus.density_planet_names(params)
        @test !isempty(owner)
        for (n, k) in owner
            @test n in _sc_group(groups, "corner_K$k").names
        end
        for k in 1:2
            @test "rho_circ_k$k" in _sc_group(groups, "corner_K$k").names
        end
        # Limb darkening and the flux offset are in no corner, and a single
        # photometric jitter makes no noise corner.
        cornered = reduce(vcat, [g.names for g in groups])
        @test !any(n -> any(p -> startswith(n, p), ("q1_", "q2_", "offset_")), cornered)
        @test _sc_stems(groups) == ["corner_K1", "corner_K2"]
    end

    @testset "a trans-dim planet's corner uses the draws it is live in" begin
        params, data = _sc_rv_params()
        n = 400
        # Slot 1 is live in the first 100 draws and parked at K = 30 m/s in the
        # rest; slot 2 is live throughout.
        on1 = Float64[i <= 100 for i in 1:n]
        centre(x) = x == "K_k1" ? 5.0 : x == "K_k2" ? 8.0 : 1.0
        ch = _sc_chain(params, n; centre,
                       extra = [:n_planets => 1.0 .+ on1, :planet_active_1 => on1,
                                :planet_active_2 => ones(n)])
        Kc = findfirst(==(:K_k1), names(ch))
        ch.value.data[101:end, Kc, 1] .= 30.0
        groups = Nereus._corner_groups(ch, params)
        @test _sc_group(groups, "corner_K1").draws == collect(1:100)
        @test _sc_group(groups, "corner_K2").draws === nothing     # live in every draw
        figs = plot_corners(ch, params)
        ax = [c for c in figs["corner_K1"].content if c isa Axis]
        ys = [p[2] for a in ax if a.ylabel[] == Nereus._sci_label("K_k1", params)
              for pl in a.scene.plots if pl isa _SC_MAKIE.Hexbin for p in pl[1][]]
        @test !isempty(ys)
        @test maximum(ys) < 6.0                          # not the parked K = 30

        # A slot never live gets no corner.
        ch0 = _sc_chain(params, n; extra = [:n_planets => ones(n),
                                            :planet_active_1 => zeros(n),
                                            :planet_active_2 => ones(n)])
        @test _sc_stems(Nereus._corner_groups(ch0, params)) == ["corner_K2", "corner_noise"]
    end

    @testset "toggled noise models: the modal pattern's draws and parameters" begin
        params, data = _sc_rv_params()
        n = 300
        for (frac_on, gp_in) in ((0.3, false), (0.7, true))
            on = Float64[i <= frac_on * n for i in 1:n]
            ch = _sc_chain(params, n; extra = [:n_planets => fill(2.0, n),
                                               :planet_active_1 => ones(n),
                                               :planet_active_2 => ones(n),
                                               :noise_active_1 => on])
            g = _sc_group(Nereus._corner_groups(ch, params), "corner_noise")
            @test ("gp_log_amp" in g.names) == gp_in
            @test issubset(["sigma_A", "sigma_B"], g.names)
            @test g.draws == findall(==(gp_in ? 1.0 : 0.0), on)
        end
    end

    @testset "axes carry units; angles are drawn in degrees" begin
        params, data = _sc_rv_params()
        centre(x) = startswith(x, "Mo_") ? 0.7 : startswith(x, "P_") ? 11.3 : 1.0
        ch = _sc_chain(params, 2_000; centre)
        fig = plot_corner(ch, params; params_to_plot = ["P_k1", "Mo_k1"])
        ax = only(c for c in fig.content if c isa Axis)
        @test ax.xlabel[] == Nereus._sci_label("P_k1", params)
        @test ax.ylabel[] == Nereus._sci_label("Mo_k1", params)
        @test occursin("(\\mathrm{d})", String(ax.xlabel[]))
        @test occursin("(\\mathrm{deg})", String(ax.ylabel[]))
        ys = [p[2] for pl in ax.scene.plots if pl isa _SC_MAKIE.Hexbin for p in pl[1][]]
        @test all(y -> 35 < y < 45, ys)                  # 0.7 rad = 40.1 deg
        # Thinned, the pinned limits are in degrees too.
        fig = plot_corner(ch, params; params_to_plot = ["P_k1", "Mo_k1"], max_draws = 100)
        ax = only(c for c in fig.content if c isa Axis)
        lim = Nereus._corner_lims(Nereus._draws(ch, :Mo_k1), rad2deg).lims
        save(tempname() * ".png", fig)
        o, w = ax.finallimits[].origin[2], ax.finallimits[].widths[2]
        @test o ≈ lim.low && o + w ≈ lim.high
    end

    @testset "run_job writes and advertises the split corners" begin
        params, data = _sc_rv_params()
        ch = _sc_chain(params, 2_000)
        out = mktempdir()
        got = Nereus._make_plots(Dict("output" => Dict("plots" => ["corner"],
                                                       "show_progress" => false)),
                                 ch, params, data, out)
        @test got == ["corner_*.png"]
        pngs = sort(filter(endswith(".png"), readdir(joinpath(out, "plots"))))
        @test pngs == ["corner_K1.png", "corner_K2.png", "corner_noise.png"]
    end
end
