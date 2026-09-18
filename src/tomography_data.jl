# Tomographic data container.
#
# Lives here rather than in tomography.jl because `Data` needs the type and is
# constructed long before the tomographic likelihood is defined — the same
# reason RelAstromData/IADData sit in astrometry/data.jl. Struct only, no
# science.

"""
    TomoNight(tag, t, R, grid, Tc)

One transit's residual map, ready for `tomogram_bayes`. `R` is (n_exposure ×
n_velocity) as returned by `tomogram_residuals`, `t` the BJDs, `grid` the
velocity grid [km/s], `Tc` that night's mid-transit time.
"""
struct TomoNight
    tag::String
    t::Vector{Float64}
    R::Matrix{Float64}
    grid::Vector{Float64}
    Tc::Float64
end

n_tomo(nights) = length(nights)
