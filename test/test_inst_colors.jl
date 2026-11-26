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
