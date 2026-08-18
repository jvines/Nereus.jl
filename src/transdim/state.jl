# TransDimState — active masks for variable-dimension models.
#
# The parameter vector stays fixed-size (allocated for max_kplanet).
# Dimension changes are mask flips, not vector resizes. This preserves:
#   - ForwardDiff compatibility (fixed Dual vector length)
#   - Pre-computed indices (PackedPriors, PackedTransforms, PlanetBlock slots)
#   - Layout stability (no rebuilding ParamsLayout at runtime)

"""
    TransDimState(; max_planets, n_noise=0)

Mutable state tracking which planets and noise components are currently
active. Birth = flip a bit to true + increment counter. Death = reverse.

# Fields
- `n_planets_active::Int`   — count of active planets (0 ≤ n ≤ max_planets)
- `planet_active::BitVector` — length max_planets
- `noise_active::BitVector`  — length n_noise
"""
mutable struct TransDimState
    n_planets_active::Int
    planet_active::BitVector
    noise_active::BitVector
    # Per-planet ASTROMETRIC coupling. Whether companion k's reflex is fitted to
    # the astrometry, independently of whether the companion exists at all.
    #
    # This is a SEPARATE question from planet_active, and the distinction is the
    # point: the RV can establish a companion beyond doubt while the astrometry
    # says nothing about it. Fixing the coupling on makes every fit report an
    # inclination — and therefore a "dynamical mass" — whether or not the
    # astrometry constrained one, which is how prior volume gets published as a
    # measurement.
    #
    # The layout is built at the MAXIMAL mode (inc/Omega allocated) and masked,
    # exactly as planets and noise are, because a Julia layout cannot change
    # block type at run time. When the flag is off those slots carry their
    # priors, which integrate to 1 and so leave the marginal likelihood
    # untouched — the same property that makes use_tomogram and the noise
    # toggles evidence-comparable.
    as_active::BitVector
end

function TransDimState(; max_planets::Int, n_noise::Int=0)
    max_planets >= 0 || throw(ArgumentError("max_planets must be ≥ 0"))
    n_noise >= 0 || throw(ArgumentError("n_noise must be ≥ 0"))
    # as_active defaults TRUE: a planet in an astrometry-bearing mode is
    # coupled unless the sampler decides otherwise, so behaviour is unchanged
    # for every existing fit that never touches the mask.
    TransDimState(0, falses(max_planets), falses(n_noise), trues(max_planets))
end

# --- Planet mask operations -------------------------------------------

"""
    activate_planet!(tds, k) -> tds

Activate planet slot `k`. No-op if already active.
"""
function activate_planet!(tds::TransDimState, k::Int)
    @boundscheck 1 ≤ k ≤ length(tds.planet_active) || throw(BoundsError(tds.planet_active, k))
    tds.planet_active[k] && return tds
    tds.planet_active[k] = true
    tds.n_planets_active += 1
    return tds
end

"""
    deactivate_planet!(tds, k) -> tds

Deactivate planet slot `k`. No-op if already inactive.
"""
function deactivate_planet!(tds::TransDimState, k::Int)
    @boundscheck 1 ≤ k ≤ length(tds.planet_active) || throw(BoundsError(tds.planet_active, k))
    tds.planet_active[k] || return tds
    tds.planet_active[k] = false
    tds.n_planets_active -= 1
    return tds
end

"""
    active_planets(tds) -> Vector{Int}

Return indices of all active planet slots. Allocates.
"""
active_planets(tds::TransDimState) = findall(tds.planet_active)

"""
    n_active_planets(tds) -> Int

Number of active planets (cached, no scan).
"""
n_active_planets(tds::TransDimState) = tds.n_planets_active

"""
    first_inactive_planet(tds) -> Union{Int, Nothing}

Return index of first inactive planet slot, or `nothing` if all active.
"""
first_inactive_planet(tds::TransDimState) = findfirst(.!tds.planet_active)

"""
    group_first_inactive(tds, modes) -> Vector{Int}

First inactive slot of EACH planet-mode group (groups = distinct values in
`modes`, e.g. RVPM slots vs an RV_ONLY slot). Birth moves pick uniformly
among these so a heterogeneous slot menu has no unreachable groups: with
strict `first_inactive_planet` ordering, an RV_ONLY long-period slot behind
RVPM slots is only reachable while a lower slot is transiently occupied —
on WASP-47 this deadlocked c (slot 4) against e (slot 3): e's proposals
were starved by unmodelled-c RV aliases, and c couldn't enter until slot 3
was filled. Empty when all slots are active. For a homogeneous menu this
reduces to `[first_inactive_planet(tds)]` (one group → exact no-op).
"""
function group_first_inactive(tds::TransDimState, modes)
    out = Int[]
    seen = Set{Any}()
    @inbounds for k in eachindex(tds.planet_active)
        tds.planet_active[k] && continue
        m = modes[k]
        m in seen && continue
        push!(seen, m)
        push!(out, k)
    end
    return out
end

# --- Noise mask operations --------------------------------------------

"""
    activate_noise!(tds, i) -> tds
"""
function activate_noise!(tds::TransDimState, i::Int)
    @boundscheck 1 ≤ i ≤ length(tds.noise_active) || throw(BoundsError(tds.noise_active, i))
    tds.noise_active[i] = true
    return tds
end

"""
    deactivate_noise!(tds, i) -> tds
"""
function deactivate_noise!(tds::TransDimState, i::Int)
    @boundscheck 1 ≤ i ≤ length(tds.noise_active) || throw(BoundsError(tds.noise_active, i))
    tds.noise_active[i] = false
    return tds
end

"""
    active_noise(tds) -> Vector{Int}
"""
active_noise(tds::TransDimState) = findall(tds.noise_active)

"""
    is_noise_active(tds, i) -> Bool
"""
is_noise_active(tds::TransDimState, i::Int) = tds.noise_active[i]

# --- Copy ---------------------------------------------------------------

Base.copy(tds::TransDimState) =
    TransDimState(tds.n_planets_active, copy(tds.planet_active),
                  copy(tds.noise_active), copy(tds.as_active))

"""
    copy_into!(dst::TransDimState, src::TransDimState) -> dst

In-place copy of TransDimState fields from `src` into `dst`.
No allocations.
"""
function copy_into!(dst::TransDimState, src::TransDimState)
    dst.n_planets_active = src.n_planets_active
    dst.planet_active .= src.planet_active
    dst.noise_active .= src.noise_active
    dst.as_active .= src.as_active
    return dst
end

# --- Likelihood evaluation counter ------------------------------------

"""
    LikelihoodCounter

Mutable counter for total likelihood evaluations.
"""
mutable struct LikelihoodCounter
    count::Int
end

LikelihoodCounter() = LikelihoodCounter(0)

# --- Slot membership queries ------------------------------------------

"""
    planet_slot_indices(block::PlanetBlock) -> Vector{Int}

Return all theta slot indices owned by a planet block.

For astrometric block subtypes (`RVASBlock`, `RVPMASBlock`) the
returned vector also contains the inclination and longitude-of-node
slots. Note that `block.K` is always present on AS-bearing blocks but
its physical meaning depends on the active mass parametrization:
under `:K_driven` it is the RV semi-amplitude, while under
`:M_sec_driven` and `:a_driven` the same slot stores the secondary
mass `M_sec`. Birth/death proposals do not need to special-case the
parametrization here — they draw from whichever prior is attached to
the slot.
"""
planet_slot_indices(b::RVOnlyBlock) = [b.P, b.K, b.e1, b.e2, b.t]
planet_slot_indices(b::PMOnlyBlock) = [b.P, b.e1, b.e2, b.t, b.b, b.r]
planet_slot_indices(b::RVPMBlock) = [b.P, b.K, b.e1, b.e2, b.t, b.b, b.r]
planet_slot_indices(b::RVASBlock) = [b.P, b.K, b.e1, b.e2, b.t, b.inc, b.Omega]
planet_slot_indices(b::RVPMASBlock) =
    [b.P, b.K, b.e1, b.e2, b.t, b.b, b.r, b.Omega]
# SB2 binary: two amplitudes; inc/Omega only when astrometric (BINARY).
planet_slot_indices(b::SB2Block) = b.inc == 0 ?
    [b.P, b.KA, b.KB, b.e1, b.e2, b.t] :
    [b.P, b.KA, b.KB, b.e1, b.e2, b.t, b.inc, b.Omega]

# --- Astrometric-coupling mask ----------------------------------------

"""
    activate_as!(tds, k)  /  deactivate_as!(tds, k)  /  is_as_active(tds, k)

Couple or decouple companion `k`'s reflex from the astrometry.

Separate from `planet_active` on purpose: a companion the RV has established
beyond doubt may still be astrometrically undetected, and forcing the coupling
on makes the fit report an inclination — hence a "dynamical mass" — whether or
not the astrometry constrained one.
"""
function activate_as!(tds::TransDimState, k::Int)
    @boundscheck 1 ≤ k ≤ length(tds.as_active) || throw(BoundsError(tds.as_active, k))
    tds.as_active[k] = true; return tds
end
function deactivate_as!(tds::TransDimState, k::Int)
    @boundscheck 1 ≤ k ≤ length(tds.as_active) || throw(BoundsError(tds.as_active, k))
    tds.as_active[k] = false; return tds
end
@inline is_as_active(tds::TransDimState, k::Int) =
    k ≤ length(tds.as_active) ? tds.as_active[k] : true

"""
    is_planet_as_active(theta, k) -> Bool

Astrometric coupling for planet `k`, defaulting to TRUE when there is no
trans-dim state or the slot is untracked — the canonical check, mirroring
`is_noise_model_active`. Reading the mask directly gets the fixed-dim case wrong.
"""
@inline function is_planet_as_active(theta, k::Int)
    theta.td === nothing && return true
    return is_as_active(theta.td, k)
end
