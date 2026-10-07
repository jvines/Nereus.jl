# The progress bar's elapsed/ETA formatter rounded only the remainder after
# taking whole minutes, so 1919.6 s printed as "31m60s" (seen on a live
# pt_emcee bar), 3599.6 s as "59m60s" and 59.6 s as "60s".

using Nereus, Test
const tf = Nereus._time_fmt

@testset "progress time format carries rounding into the next unit" begin
    @test tf(1919.6) == "32m00s"
    @test tf(3599.6) == "1h00m"
    @test tf(59.6) == "1m00s"
    @test tf(0.4) == "0s"
    @test tf(-3.0) == "0s"
    @test tf(125.0) == "2m05s"
    @test tf(7322.0) == "2h02m"
    # no field ever reads 60
    @test all(s -> !occursin(r"(^|[^0-9])60[sm]", tf(s)), 0.0:0.1:7300.0)
end
