# Instrument colors are pastelised samples of the cool colormap, spread over
# the number of instruments in the plot.
using Nereus, Test

@testset "inst_color" begin
    for n in (1, 2, 5)
        ref = Nereus.cool_pastel(max(n, 2))
        for i in 1:n
            @test Nereus.inst_color(i, n) == ref[i]
        end
    end
    @test Nereus.inst_color(1, 1) == Nereus.cool_pastel(2)[1]
    @test length(unique(Nereus.inst_color(i, 5) for i in 1:5)) == 5
end

# The plots that call inst_color must actually use cool_pastel(n) entries.
using Random, MCMCChains
using CairoMakie: Figure, Axis, Colorant, RGB
const _MK = Nereus.CairoMakie.Makie

# every RGB (alpha dropped) any plot of the figure was given as `color`
function _used_colors(fig)
    out = Set{NTuple{3, Float32}}()
    function visit(p)
        if haskey(p.attributes, :color)
            c = p.color[]
            c isa Tuple && (c = c[1])
            c isa Colorant && (r = RGB(c); push!(out, (Float32(r.r), Float32(r.g), Float32(r.b))))
        end
        foreach(visit, p.plots)
    end
    for ax in filter(c -> c isa Axis, fig.content), p in ax.scene.plots
        visit(p)
    end
    return out
end
_rgb3(c) = (r = RGB(c); (Float32(r.r), Float32(r.g), Float32(r.b)))

@testset "plot_transit_phasefold sector colors" begin
    rng = MersenneTwister(5)
    P, t0, dur = 3.0, 0.5, 0.15
    t = collect(range(0.0, 30.0; length = 1500))
    flux = 1 .+ 1e-4 .* randn(rng, length(t))
    ph = Nereus.phase_fold(t, P, t0)
    flux[abs.(ph) .< dur / (2P)] .-= 5e-3
    result = (flux_detrended = flux, flux_err_detrended = fill(1e-4, length(t)),
              periods = [P], t0s = [t0], durations = [dur], snr = [20.0])
    for ns in (2, 3)
        sid = [min(ns, 1 + floor(Int, (ti / 30.0) * ns)) for ti in t]
        names = ["S$k" for k in 1:ns]
        fig = Nereus.plot_transit_phasefold(t, result; sector_id = sid,
                                            sector_names = names, save_pdf = false)
        @test fig isa Figure
        used = _used_colors(fig)
        for c in Nereus.cool_pastel(ns)
            @test _rgb3(c) in used
        end
    end
end

@testset "plot_rm instrument colors" begin
    rng = MersenneTwister(9)
    P, K, Tc = 3.5, 30.0, 2459000.3
    t = collect(range(Tc - 0.2, Tc + 0.2; length = 40)); n = length(t)
    data = Data(; t_rv = t, rv = zeros(n), rv_err = fill(3.0, n),
                  rv_inst = [isodd(i) ? 1 : 2 for i in 1:n])
    params = Params(; parametrization = ParametrizationConfig(time = :Tc), max_kplanet = 1,
                      planet_modes = [Nereus.RVPM_RM],
                      instruments = InstrumentConfig(rv = ["A", "B"]),
                      data = data, M_s = 1.0, R_s = 1.0)
    val = Dict("P_k1" => P, "K_k1" => K, "Tc_k1" => Tc, "b_k1" => 0.3, "rr_k1" => 0.1,
               "lambda_k1" => 0.5, "gamma_A" => 0.0, "gamma_B" => 0.0, "sigma_A" => 1.0,
               "sigma_B" => 1.0, "rho_circ_k1" => 1.4, "eta_k1" => 0.0,
               "v_sin_i_star" => 5000.0)
    nm = params.layout.unfrozen_names
    cols = vcat(Symbol.(nm), [:lp])
    arr = zeros(30, length(cols), 1)
    for i in 1:30
        arr[i, 1:length(nm), 1] = [val[x] + 1e-6 * randn(rng) for x in nm]
        arr[i, end, 1] = -0.1 * i
    end
    chains = Chains(arr, cols)
    fig = Nereus.plot_rm(chains, params, data; n_draws = 20)
    @test fig isa Figure
    used = _used_colors(fig)
    for c in Nereus.cool_pastel(2)
        @test _rgb3(c) in used
    end
end
