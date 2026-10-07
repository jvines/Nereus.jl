# The periodogram figure's vertical layout when no peak is significant.
#
# The y-range was set from the highest power alone, so the FAP lines were in
# view only by chance; the peak labels were stacked from that same height, so
# nearby peaks climbed out of the axis and the top label was clipped; and the
# leaders ran solid and dark from the peak tip up through the FAP lines, where
# they read as the spike itself crossing the 1% line. On the residuals of a
# 36-point RV fit the 0.1% line was off the axis and two peaks at FAP 0.85 and
# 0.98 looked like detections.

using Nereus, Test
using CairoMakie: Axis, Lines

# Three insignificant peaks close together in log frequency (labels stagger),
# under four FAP lines the highest of which is well above the spectrum.
function _insignificant_pgram()
    f = collect(range(0.002, 2.0; length = 4000))
    pwr = fill(0.05, length(f))
    pk = PgramPeak[]
    for (fi, h) in ((1.380, 0.37), (1.721, 0.35), (1.760, 0.31))
        i = argmin(abs.(f .- fi))
        pwr[i] = h
        push!(pk, PgramPeak(1 / f[i], f[i], h, 0.9))
    end
    GLSPgram(f, 1 ./ f, pwr, [0.1, 0.05, 0.01, 0.001], [0.47, 0.50, 0.55, 0.65],
             36, pk)
end

@testset "periodogram: FAP lines in view, labels above them and unclipped" begin
    pg = _insignificant_pgram()
    fig = plot_periodogram(pg)
    ax = first(filter(c -> c isa Axis, fig.content))
    lo, hi = ax.limits[][2]
    ref = maximum(pg.fap_thresholds)
    @test hi > ref                                   # every FAP line drawn in view
    leaders = [p[1][] for p in ax.scene.plots
               if p isa Lines && length(p[1][]) == 2 && p[1][][1][1] == p[1][][2][1]]
    @test length(leaders) == length(pg.peaks)
    tops = [l[2][2] for l in leaders]
    @test all(>(ref), tops)                          # labels sit above the FAP lines
    @test maximum(tops) + 0.2 * ref <= hi            # and the highest is not clipped
    @test length(unique(tops)) == 3                  # the close peaks staggered
    for l in leaders                                 # leaders start above their peak
        x = l[1][1]
        pk = pg.peaks[argmin([abs(log10(p.frequency) - x) for p in pg.peaks])]
        @test l[1][2] > pk.power
    end
end
