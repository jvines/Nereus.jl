# Relative astrometry as the Python client sends it: a `values` block inside an
# astrometry channel, the same block a run_job config takes.

using Nereus, Test

@testset "fit_* channels: relative astrometry from a values block" begin
    rel = Dict("values" => Dict("t" => [58768.0, 58800.0], "ra_off" => [100.9, 99.0],
                                "dec_off" => [-55.3, -57.0], "ra_err" => [1.0, 1.1],
                                "dec_err" => [0.9, 1.0], "corr" => [0.3, -0.2],
                                "planet_idx" => [2, 2]))
    rv = Dict("source" => "RV", "data" => Dict("RV" => (t = [58000.0, 58100.0, 58200.0],
                                                     rv = [0.0, 1.0, 2.0], rv_err = [1.0, 1.0, 1.0])))
    priors = Dict("plx" => Dict("type" => "normal", "mu" => 13.7, "sigma" => 0.03),
                  "M_pri" => Dict("type" => "normal", "mu" => 1.5, "sigma" => 0.04))
    tgt = Nereus._target_from([rv, Dict("source" => "AS", "relast" => rel)], 2; priors)
    ra = tgt.data.relastrom
    @test ra isa RelAstromData
    @test ra.planet_idx == [2, 2]
    @test ra.corr == [0.3, -0.2]
    @test ra.ra_off == [100.9, 99.0]

    # Two channels' relative astrometry merge rather than the last one winning.
    rel2 = Dict("values" => Dict("t" => [59000.0], "ra_off" => [90.0], "dec_off" => [-60.0],
                                 "ra_err" => [1.0], "dec_err" => [1.0], "planet_idx" => [2]))
    tgt2 = Nereus._target_from([rv, Dict("source" => "AS", "relast" => rel),
                                Dict("source" => "AS", "relast" => rel2)], 2; priors)
    @test length(tgt2.data.relastrom.t) == 3
end
