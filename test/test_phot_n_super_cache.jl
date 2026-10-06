# `_phot_n_super(data)` reads every exposure time (about 20 us on 20k cadences),
# and the workspace transit likelihood asked for it on every call. It is now
# computed once per workspace (`ws.phot_data`, filled on first use).
#
# Bit-identity with the code before the cache: the cached value is the same Int
# the scan returns, and with it the workspace likelihood takes the same branch.
# On the supersampled branch it returns exactly what the non-workspace method
# returns, as it did before (it called that method).
using Test
using Nereus
using Nereus: _phot_n_super, PhotDataCache

include(joinpath(@__DIR__, "fixtures", "transit_gate_target.jl"))

@testset "photometry supersampling factor cached per workspace" begin
    @testset "cached value is the scan's value" begin
        t = collect(range(0.0, 1.0; length = 50))
        mk(ex) = Data(; t_phot = t, flux = ones(50), flux_err = fill(1e-3, 50),
                        phot_inst = ones(Int, 50), exposure_times = ex)
        for ex in (Float64[], fill(120 / 86_400, 50), fill(1800 / 86_400, 50),
                   [isodd(i) ? 600 / 86_400 : NaN for i in 1:50],
                   [isodd(i) ? -1.0 : 200 / 86_400 for i in 1:50],
                   fill(Inf, 50))
            d = mk(ex)
            c = PhotDataCache()
            @test c.n_super == 0                      # 0 = not computed yet
            @test _phot_n_super(d, c) === _phot_n_super(d)
            @test c.n_super === _phot_n_super(d) >= 1
        end
        # Filled once: later calls do not read the exposures again.
        d = mk(fill(120 / 86_400, 50))
        c = PhotDataCache()
        @test _phot_n_super(d, c) == 1
        d.exposure_times .= 1800 / 86_400              # would give 30 if rescanned
        @test _phot_n_super(d, c) == 1
        @test _phot_n_super(d) == 30
    end

    for exposure in (false, true)
        @testset "workspace likelihood, exposure = $exposure" begin
            tg = transit_gate_target(; exposure)
            data = tg.data
            pts = transit_gate_points(tg, 60; seed = 21)
            th = Nereus.Theta{Float64}(tg.params)
            ws = transit_gate_ws(tg)
            @test ws.phot_data.n_super == 0
            vals = Float64[]
            for v in pts
                th.values .= v
                push!(vals, Nereus.transit_log_likelihood(th, data, ws))
            end
            @test ws.phot_data.n_super == _phot_n_super(data) == (exposure ? 30 : 1)
            # A second workspace, filled from the first point on, gives the same bits.
            ws2 = transit_gate_ws(tg)
            for (v, ref) in zip(pts, vals)
                th.values .= v
                @test Nereus.transit_log_likelihood(th, data, ws2) === ref
            end
            if exposure
                # Supersampled: the workspace method hands over to the
                # non-workspace one, exactly as before.
                for (v, ref) in zip(pts, vals)
                    th.values .= v
                    @test Nereus.transit_log_likelihood(th, data) === ref
                end
            end
            # Not in a checkpoint: a restored workspace rebuilds it on first use.
            snap = Nereus._ws_snapshot(ws)
            @test !haskey(snap, :phot_data)
            ws3 = transit_gate_ws(tg)
            Nereus._ws_restore!(ws3, snap)
            @test ws3.phot_data.n_super == 0
            th.values .= pts[end]
            @test Nereus.transit_log_likelihood(th, data, ws3) === vals[end]
            @test ws3.phot_data.n_super == ws.phot_data.n_super
        end
    end
end
