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

# ---------------------------------------------------------------------
# Scratch for the residual-map likelihood
# ---------------------------------------------------------------------
#
# `tomogram_log_likelihood` allocated 0.94 MB per call on the NGTS-33 maps
# (37, 33 and 30 exposures x 57 velocities): the shadow and residual matrices,
# both kernel factors, the copies `eigen` makes, its LAPACK workspace, and the
# two products. Under pt_emcee on 4 threads that was a stop-the-world
# collection every ~9 ms, 9-10% of the wall time, each one first waiting for
# the other threads to come back from LAPACK (up to 1.3 ms); the collections
# come more often the more threads allocate. A sampler owns one of these per
# slot and the likelihood writes into it; the arithmetic, and so every bit of
# the result, is the allocating path's. Structs only here (PTWorkspace holds
# one); the code is in tomography.jl.

using LinearAlgebra: BlasInt

"""
    SymEigenWork(n)

Buffers for one `dsyevr` call on an n x n symmetric matrix, as
`LinearAlgebra.LAPACK.syevr!('V', 'A', 'U', A, 0, 0, 0, 0, -1)` makes them:
the input copy `A` (overwritten), eigenvalues `W`, eigenvectors `Z`, and the
LAPACK workspaces at the sizes LAPACK's own query returns (`lwork = 0` until
the first call queries them).
"""
mutable struct SymEigenWork
    n::Int
    A::Matrix{Float64}
    W::Vector{Float64}
    Z::Matrix{Float64}
    isuppz::Vector{BlasInt}
    work::Vector{Float64}
    iwork::Vector{BlasInt}
    lwork::BlasInt
    liwork::BlasInt
end

SymEigenWork(n::Int) = SymEigenWork(n, zeros(n, n), zeros(n), zeros(n, n),
                                    zeros(BlasInt, 2n), Float64[], BlasInt[],
                                    BlasInt(0), BlasInt(0))

"""
    KernelDistances(x)

The pairs (i <= j) of a coordinate vector grouped by `abs(x[i] - x[j])`, bit for
bit: `cls[p]` is the group of the p-th upper-triangle entry in column order and
`d[c]` that group's distance. A stationary kernel then needs one evaluation per
group -- 57 instead of 1653 on a uniform 57-point velocity grid, whose
differences are exact -- and gives every entry exactly the value a direct
evaluation would. `val` holds the per-group values of the current call.
"""
struct KernelDistances
    n::Int
    cls::Vector{Int32}
    d::Vector{Float64}
    val::Vector{Float64}
end

function KernelDistances(x::AbstractVector{Float64})
    n = length(x)
    seen = Dict{UInt64,Int32}()
    d = Float64[]
    cls = Vector{Int32}(undef, n * (n + 1) ÷ 2)
    p = 0
    @inbounds for j in 1:n, i in 1:j
        δ = abs(x[i] - x[j])
        c = get!(seen, reinterpret(UInt64, δ)) do
            push!(d, δ)
            Int32(length(d))
        end
        p += 1
        cls[p] = c
    end
    return KernelDistances(n, cls, d, zeros(length(d)))
end

"""
Per-map scratch: shadow, residuals, both kernel factors and their
eigensystems, and the product U_t' r U_v.
"""
struct TomoNightWork
    M::Matrix{Float64}          # shadow model, n_t x n_v
    r::Matrix{Float64}          # residuals R - alpha M
    Kt::Matrix{Float64}         # temporal factor K_t (upper triangle)
    Kv::Matrix{Float64}         # velocity factor K_v (upper triangle)
    et::SymEigenWork
    ev::SymEigenWork
    Y::Matrix{Float64}          # U_t' r
    Z::Matrix{Float64}          # U_t' r U_v
    dt::Vector{Float64}
    dv::Vector{Float64}
    tdist::KernelDistances      # exposure-time pairs (days)
    vdist::KernelDistances      # velocity-grid pairs (km/s)
end

"""
    TomoWorkspace

Scratch for `tomogram_log_likelihood(theta, data, ws)`, one per sampler slot.
Built on first use for one `Params` and one set of maps, and rebuilt if
either changes.
"""
mutable struct TomoWorkspace
    params::Any
    nights_src::Vector{TomoNight}
    nights::Vector{TomoNightWork}
end
