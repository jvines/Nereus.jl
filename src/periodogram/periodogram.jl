# Periodogram module entry — loads every piece in dependency order.
#
# Public surface (re-exported from Nereus.jl top-level):
#   - Types:        Periodogram, GLSPgram, BGLSPgram, SBGLSPgram, L1Pgram,
#                    PgramSet, PgramPeak
#   - Method calls: gls_periodogram, bgls_periodogram, sbgls_periodogram,
#                    l1_periodogram_compute
#   - High-level:   find_rv_planets
#   - Plotting:     plot_periodogram, plot_periodograms_stacked,
#                    plot_sbgls_heatmap

include("types.jl")
include("peaks.jl")
include("gls.jl")
include("bgls.jl")
include("l1.jl")
include("api.jl")
include("plots.jl")
