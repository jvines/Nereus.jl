# The workspace refresh used to find the cadences inside the transit window by
# testing all n_obs of them (15-18 us per refresh on 20k cadences). It now
# binary-searches a time order of the cadences (built once per workspace) and
# applies the same test only near each transit centre. The index SET must be
# exactly the full scan's, whatever the cadences look like: unsorted across
# instruments, repeated times, BJD or small magnitudes, windows that are empty,
# tiny, wider than a quarter period, negative or NaN, centres far from the data.
using Test
using Nereus
using Random
using Nereus: _phot_window_indices!, PhotDataCache, _transit_window_halfwidth, tp_to_tc

include(joinpath(@__DIR__, "fixtures", "transit_gate_target.jl"))

function tws_scan(t, Tc, P, hw)
    out = Int[]
    for i in eachindex(t)
        Δt = t[i] - Tc
        Δt -= P * round(Δt / P)
        abs(Δt) <= hw && push!(out, i)
    end
    return out
end
tws_data(t) = Data(; t_phot = t, flux = ones(length(t)), flux_err = fill(1e-3, length(t)),
                     phot_inst = ones(Int, length(t)))

@testset "transit window by binary search" begin
    @testset "index set === full scan" begin
        rng = MersenneTwister(99)
        n_case = 0; n_nonempty = 0
        for trial in 1:3000
            t0 = rand(rng, (0.0, 1500.0, 2_460_000.0))
            n = rand(rng, (1, 2, 7, 300, 4000))
            # instruments appended one after the other: globally unsorted
            parts = [t0 .+ sort!(rand(rng, n) .* rand(rng, (0.5, 30.0, 400.0)))
                     for _ in 1:rand(rng, 1:3)]
            t = reduce(vcat, parts)
            rand(rng) < 0.2 && (t[rand(rng, 1:length(t))] = t[1])      # repeated time
            data = tws_data(t)
            cache = PhotDataCache()
            for _ in 1:8
                P  = exp(log(0.05) + rand(rng) * (log(500) - log(0.05)))
                Tc = t0 + (rand(rng) - 0.3) * rand(rng, (1.0, 50.0, 5000.0))
                hw = rand(rng, (0.0, -0.01, NaN, Inf, P / 4, P / 4 * (1 - 1e-12), P / 2,
                                rand(rng) * 0.3 * P, rand(rng) * 1e-3 * P, 1e-9))
                # boundary cadences: put some exactly on Tc + mP ± hw
                if isfinite(hw) && rand(rng) < 0.3
                    m = round((t[1] - Tc) / P)
                    t2 = copy(t); t2[1] = Tc + m * P + hw; t2[end] = Tc + (m + 1) * P - hw
                    d2 = tws_data(t2)
                    got = sort!(_phot_window_indices!(Int[], d2, PhotDataCache(), Tc, P, hw))
                    @test got == tws_scan(t2, Tc, P, hw)
                end
                got = _phot_window_indices!(Int[], data, cache, Tc, P, hw)
                ref = tws_scan(t, Tc, P, hw)
                @test length(got) == length(unique(got))
                @test sort(got) == ref
                n_case += 1; n_nonempty += !isempty(ref)
            end
        end
        @info "window search vs scan" n_case n_nonempty
        @test n_nonempty > 2000
        # A non-finite cadence time: the full scan, unchanged.
        t = [2_460_000.0, NaN, 2_460_000.5, 2_460_001.0]
        @test _phot_window_indices!(Int[], tws_data(t), PhotDataCache(), 2_460_000.5, 1.0, 0.1) ==
              tws_scan(t, 2_460_000.5, 1.0, 0.1) == [3]
    end

    @testset "workspace refresh uses the same cadences" begin
        tg = transit_gate_target()
        data = tg.data
        @test !issorted(data.t_phot)                 # two instruments, concatenated
        th = Nereus.Theta{Float64}(tg.params); ws = transit_gate_ws(tg)
        n_cmp = 0
        for v in transit_gate_points(tg, 300; seed = 51)
            th.values .= v
            isfinite(Nereus.transit_log_likelihood(th, data, ws)) || continue
            P, e, ω, Tp = ws.transit_Ps[1], ws.transit_es[1], ws.transit_ws[1], ws.transit_Tps[1]
            rr, aR, b = ws.transit_rrs[1], ws.transit_a_Rs[1], ws.transit_bs[1]
            b < 1 + rr || continue
            Tc = tp_to_tc(Tp, P, e, ω)
            hw = _transit_window_halfwidth(P, e, ω, rr, aR, max(abs(Tc), abs(Tp)))
            @test sort(ws.transit_in_idx[1]) == tws_scan(data.t_phot, Tc, P, hw)
            n_cmp += 1
        end
        @test n_cmp > 100
        @test ws.phot_data.order_built && ws.phot_data.sortable
        @test issorted(ws.phot_data.t_sorted)
        @test data.t_phot[ws.phot_data.perm] == ws.phot_data.t_sorted
    end
end
