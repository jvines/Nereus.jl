# ======================================================================
# State proposals for the noise-model swap
# ======================================================================
#
# `propose_noise_swap` exchanges the active member A of an exclusion group for
# another member B by drawing only B's OWN parameters (from the prior, the GP
# hints, or the OLS fit of the decorrelation). Every parameter the states share
# -- per-instrument jitters, offsets, trend, an always-on indicator term --
# keeps the value it had under A. Those values are tuned to A: a decorrelation
# state with the jitters of a GP state, or a GP state with the jitters of a
# white model, lands hundreds to thousands of nats below the incumbent. On
# HD 18599 (four-member RV menu) the median Δln L of a cold-chain swap was
# -2800 to -85000 for every pair, and the cold chain accepted ~80 swaps in 10^6
# proposals, so the member probabilities depended on the seed.
#
# A `NoiseStateProposal` is an independence proposal for the whole state of a
# member: its own parameters plus the shared ones, a multivariate normal fitted
# to that member's own draws (log scale for positive-support parameters). The
# sampler learns one per member during burn-in from the coldest temperature
# where the member has enough draws, and freezes them at the end of burn-in;
# the production swap is then an ordinary RJMCMC move with a fixed proposal:
#   log α = β Δln L + Δln π + ln q_A(x_A) - ln q_B(x_B') + ln(G / G_rev).
# =====================================================================

struct NoiseStateProposal
    slots::Vector{Int}            # theta slots: the member's own + the shared
    logt::BitVector               # log-transformed (positive support)
    lo::Vector{Float64}
    hi::Vector{Float64}
    mu::Vector{Float64}
    L::Matrix{Float64}            # lower Cholesky factor of the (inflated) covariance
    logdet::Float64               # log|L|
end

"""
    noise_state_slots(params, nm_idx, toggleable_idx) -> (slots, upos)

Theta slots (and their positions among the unfrozen parameters) of the state of
noise model `nm_idx`: its own unfrozen parameters plus the shared ones in its
scope -- the per-instrument parameters of the instruments it acts on (all of its
channel's when it names none), the RV trend for an RV-channel member, and the
parameters of always-on noise models on its channel. Planet parameters and other
toggleable models' parameters never enter (see `noise_state_planet_slots`).
"""
function noise_state_slots(params::Params, nm_idx::Int, toggleable_idx::Vector{Int})
    layout = params.layout; ic = params.config.instruments
    nms = params.config.noise_models
    planet = Set{Int}()
    for b in layout.planet_blocks, f in fieldnames(typeof(b))
        v = getfield(b, f)
        v isa Integer && v > 0 && push!(planet, Int(v))
        v isa AbstractVector{<:Integer} && union!(planet, Int.(v))
    end
    slots_of(j) = Int[layout.name_to_idx[n] for n in noise_param_names(nms[j], ic)
                      if haskey(layout.name_to_idx, n)]
    member = Set{Int}(); for j in toggleable_idx; union!(member, slots_of(j)); end
    own = Set(slots_of(nm_idx))
    # shared parameters in the member's scope only: the per-instrument ones
    # (jitter, offset, ...) of the instruments it acts on, and the parameters of
    # always-on noise models on its channel. A menu per light curve would
    # otherwise redraw every other light curve's jitter in each swap.
    nm = nms[nm_idx]; ch = noise_channel(nm)
    insts = hasfield(typeof(nm), :instruments) && !isempty(getfield(nm, :instruments)) ?
            String.(getfield(nm, :instruments)) :
            (ch === :rv ? ic.rv_names : ch === :phot ? ic.pm_names : String[])
    always = Set{Int}()
    for j in eachindex(nms)
        (j in toggleable_idx || noise_channel(nms[j]) !== ch) && continue
        union!(always, slots_of(j))
    end
    # the RV trend is shared by every RV model and is what an activity model
    # trades against: a rotation GP absorbs a slow drift that a decorrelation
    # leaves to `dvdt`, so a decorrelation state proposed with the GP's trend
    # lands far below its own optimum (HD 18599: the fixed-fit AD optimum is 74
    # nats above the GP-Rot walkers in log L, the proposals never got within 70)
    trend = ch === :rv ? ("dvdt", "d2vdt2") : ()
    in_scope(s) = s in always || layout.names[s] in trend ||
                  any(i -> endswith(layout.names[s], "_" * i), insts)
    upos = Int[]; slots = Int[]
    for (j, s) in enumerate(layout.unfrozen_idx)
        (s in own || (!(s in planet) && !(s in member) && in_scope(s))) || continue
        s == layout.n_p_idx && continue
        push!(slots, s); push!(upos, j)
    end
    return slots, upos
end

"""
    noise_state_group_slots(params, members, toggleable_idx) -> Dict{Int, Vector{Int}}

State slots of every member of one exclusion group, with ONE shared set for the
group: the union of the members' shared slots. The swap from A to B redraws the
slots of B's proposal and is reversed by redrawing those of A's, so the two
must replace the same shared parameters or the move is not reversible (members
acting on different instruments would otherwise have different shared sets).
"""
function noise_state_group_slots(params::Params, members::Vector{Int},
                                 toggleable_idx::Vector{Int})
    nms = params.config.noise_models; ic = params.config.instruments
    layout = params.layout
    own(j) = Set{Int}(layout.name_to_idx[n] for n in noise_param_names(nms[j], ic)
                      if haskey(layout.name_to_idx, n))
    base = Dict(j => noise_state_slots(params, j, toggleable_idx)[1] for j in members)
    shared = Set{Int}()
    for j in members; union!(shared, setdiff(base[j], own(j))); end
    pos = Dict(s => u for (u, s) in enumerate(layout.unfrozen_idx))
    return Dict(j => sort!(collect(union(base[j], shared)); by = s -> pos[s])
                for j in members)
end

"""
    noise_state_planet_slots(params, members) -> Vector{Tuple{Int,Int}}

`(k, slot)` of the unfrozen RV semi-amplitude of each planet `k`, when any member
of the group acts on the RV channel; empty otherwise. An activity model trades
amplitude with the planets: on HD 18599 the GP-Rot walkers carry K = 15 m/s on
the inner planet where the decorrelation's optimum has 8.3, and a decorrelation
state proposed with the GP's K stays ~100 nats below its optimum. These slots
join a member's state only in the proposals kept per planet configuration, and
only for the planets that configuration has on (an inactive planet's parameters
have no prior density, so they are left alone).
"""
function noise_state_planet_slots(params::Params, members::Vector{Int})
    any(j -> noise_channel(params.config.noise_models[j]) === :rv, members) ||
        return Tuple{Int,Int}[]
    layout = params.layout; unf = Set(layout.unfrozen_idx)
    out = Tuple{Int,Int}[]
    for (k, b) in enumerate(layout.planet_blocks)
        hasfield(typeof(b), :K) || continue
        s = getfield(b, :K)
        (s isa Integer && s > 0 && s in unf) && push!(out, (k, Int(s)))
    end
    return out
end

"""
    fit_noise_state_proposal(X, slots, lo, hi; inflate = 1.5, min_draws = 0)
        -> Union{NoiseStateProposal, Nothing}

Multivariate normal fitted to the rows of `X` (draws of `slots`), on the log
scale for parameters whose lower bound is positive, with its covariance
inflated by `inflate^2`. `nothing` when there are fewer than `min_draws` rows,
fewer rows than 2 × dimension, fewer distinct rows than max(2 × dimension + 2,
50), a parameter without spread, or a covariance that is not positive definite.
"""
function fit_noise_state_proposal(X::AbstractMatrix{<:Real}, slots::Vector{Int},
                                  lo::Vector{Float64}, hi::Vector{Float64};
                                  inflate::Real = 1.5, min_draws::Int = 0)
    n, d = size(X)
    (n < max(min_draws, 2d, 10)) && return nothing
    # Rows repeat when walkers do not move between collections; a few stuck
    # walkers then give a covariance of zero width in some directions, and a
    # proposal that rejects every move out of their own state (measured on HD
    # 18599: log q = -1e10 at the walkers' own values). Count distinct rows.
    length(unique(eachrow(X))) < max(2d + 2, 50) && return nothing
    logt = BitVector(lo .> 0)
    Y = Matrix{Float64}(X)
    for k in 1:d
        logt[k] || continue
        any(<=(0), view(Y, :, k)) && return nothing
        Y[:, k] .= log.(view(Y, :, k))
    end
    mu = vec(sum(Y; dims = 1)) ./ n
    Yc = Y .- mu'
    S = (Yc' * Yc) ./ (n - 1)
    sc = Float64(inflate)^2
    vars = diag(S)
    any(<=(0), vars) && return nothing
    for k in 1:d; S[k, k] += 1e-10 * max(S[k, k], 1e-12); end
    C = cholesky(Symmetric(S .* sc); check = false)
    issuccess(C) || return nothing
    Lm = Matrix(C.L)
    return NoiseStateProposal(copy(slots), logt, copy(lo), copy(hi), mu, Lm,
                              sum(log, diag(Lm)))
end

# log density of the proposal at the values of its slots in `vals`, on the
# original scale (Jacobian of the log transform included); -Inf outside bounds.
function noise_state_logq(p::NoiseStateProposal, vals::AbstractVector)
    d = length(p.slots)
    y = Vector{Float64}(undef, d); jac = 0.0
    @inbounds for k in 1:d
        x = Float64(vals[p.slots[k]])
        (p.lo[k] <= x <= p.hi[k]) || return -Inf
        if p.logt[k]
            x > 0 || return -Inf
            y[k] = log(x); jac += y[k]
        else
            y[k] = x
        end
    end
    z = LowerTriangular(p.L) \ (y .- p.mu)
    return -0.5 * dot(z, z) - p.logdet - 0.5 * d * log(2π) - jac
end

# Draw the proposal's slots into `vals`; returns the log density of the draw,
# or -Inf when it falls outside a prior bound (the move is then rejected, which
# leaves detailed balance intact: the target is zero there).
function noise_state_draw!(vals::AbstractVector, p::NoiseStateProposal, rng::AbstractRNG)
    d = length(p.slots)
    y = p.mu .+ p.L * randn(rng, d)
    @inbounds for k in 1:d
        x = p.logt[k] ? exp(y[k]) : y[k]
        (p.lo[k] <= x <= p.hi[k]) || return -Inf
        vals[p.slots[k]] = x
    end
    return noise_state_logq(p, vals)
end

"""
    propose_noise_swap_state(theta, rng, group, toggleable, props)
        -> (new_theta, log_q_ratio)

`propose_noise_swap` with state proposals: the active member A of `group` is
exchanged for a uniformly chosen inactive member B, and B's state (its own and
the shared parameters) is drawn from `props[B]`. The reverse move draws A's
state from `props[A]`. Returns `(theta, -Inf)` when the move does not apply or
a member has no proposal (`props[j] === nothing`).
"""
function propose_noise_swap_state(theta::Theta{T}, rng::AbstractRNG,
                                  group::Vector{NoiseModel},
                                  toggleable::Vector{NoiseModel},
                                  props::AbstractVector) where {T}
    nm_models = theta.params.config.noise_models
    members = Tuple{NoiseModel,Int}[]
    for nm in group
        nm in toggleable || continue
        j = findfirst(==(nm), nm_models); j === nothing && continue
        push!(members, (nm, j))
    end
    length(members) ≥ 2 || return (theta, convert(T, -Inf))
    active = [j for (nm, j) in members if is_noise_active(theta.td, j)]
    length(active) == 1 || return (theta, convert(T, -Inf))
    ai = active[1]
    inactive = [j for (nm, j) in members if !is_noise_active(theta.td, j)]
    isempty(inactive) && return (theta, convert(T, -Inf))
    G = length(inactive)
    bi = inactive[rand(rng, 1:G)]
    pA = props[ai]; pB = props[bi]
    (pA === nothing || pB === nothing) && return (theta, convert(T, -Inf))
    lq_rev = noise_state_logq(pA, theta.values)
    isfinite(lq_rev) || return (theta, convert(T, -Inf))
    nv = copy(theta.values); ntd = copy(theta.td)
    lq_fwd = noise_state_draw!(nv, pB, rng)
    isfinite(lq_fwd) || return (theta, convert(T, -Inf))
    deactivate_noise!(ntd, ai); activate_noise!(ntd, bi)
    nt = Theta{T}(theta.params, nv; td = ntd)
    G_rev = count(((nm, j),) -> !is_noise_active(ntd, j), members)
    return (nt, convert(T, (log(G) - log(max(G_rev, 1))) + lq_rev - lq_fwd))
end

"""
    NoiseStateEnsemble

The fixed-member ensemble behind a seeded proposal: walker positions (full theta
value vectors) and the trans-dim state they are evaluated under. Kept between
refits so each refit continues the same run instead of starting over.
"""
mutable struct NoiseStateEnsemble
    X::Vector{Vector{Float64}}
    td::TransDimState
    n_done::Int
end

"""
    run_noise_state_ensemble!(ens, logp, slots, lo, hi, rng; n_steps, stretch_a = 2.0)
        -> Matrix{Float64}

Affine-invariant stretch moves (Goodman & Weare 2010) on the member's state slots
for every walker of `ens`, on the scale of `fit_noise_state_proposal` (log for
positive support, Jacobian included), for `n_steps` steps. `logp(values, k)` is
the β = 1 log posterior with the member on; `k` is a workspace index (walkers of
one half-ensemble are evaluated in parallel, dealt round-robin to chunks, and `k`
is the chunk number 1..nchunks, never a thread id). Returns the draws of the last
half of the steps, rows on the original scale.
"""
function run_noise_state_ensemble!(ens::NoiseStateEnsemble, logp, slots::Vector{Int},
                                   lo::Vector{Float64}, hi::Vector{Float64},
                                   rng::AbstractRNG; n_steps::Int, stretch_a::Real = 2.0,
                                   nchunks::Int = 1)
    d = length(slots); nw = length(ens.X); logt = BitVector(lo .> 0)
    fwd(x, k) = logt[k] ? log(x) : x
    bwd(y, k) = logt[k] ? exp(y) : y
    Y = [Float64[fwd(clamp(x[slots[k]], lo[k], hi[k]), k) for k in 1:d] for x in ens.X]
    jac(y) = sum((y[k] for k in 1:d if logt[k]); init = 0.0)
    function lpy(y, base, c)
        v = copy(base)
        for k in 1:d
            x = bwd(y[k], k)
            (lo[k] <= x <= hi[k]) || return -Inf
            v[slots[k]] = x
        end
        l = logp(v, c)
        return isfinite(l) ? l + jac(y) : -Inf
    end
    L = Vector{Float64}(undef, nw)
    Threads.@threads :static for c in 1:nchunks
        for i in c:nchunks:nw
            L[i] = lpy(Y[i], ens.X[i], c)
        end
    end
    keep = Vector{Vector{Float64}}()
    half = [1:2:nw, 2:2:nw]
    a = Float64(stretch_a)
    for step in 1:n_steps
        for h in 1:2
            act = collect(half[h]); oth = collect(half[3 - h])
            # draws made serially so the run is reproducible whatever the threads
            zs = [((a - 1) * rand(rng) + 1)^2 / a for _ in act]
            js = [oth[rand(rng, 1:length(oth))] for _ in act]
            us = rand(rng, length(act))
            Threads.@threads :static for c in 1:nchunks
                for ii in c:nchunks:length(act)
                    i = act[ii]; z = zs[ii]; yj = Y[js[ii]]
                    yp = yj .+ z .* (Y[i] .- yj)
                    lp = lpy(yp, ens.X[i], c)
                    if isfinite(lp) && log(us[ii]) < (d - 1) * log(z) + lp - L[i]
                        Y[i] = yp; L[i] = lp
                    end
                end
            end
        end
        if step > n_steps ÷ 2
            for i in 1:nw
                push!(keep, Float64[bwd(Y[i][k], k) for k in 1:d])
            end
        end
    end
    for i in 1:nw, k in 1:d
        ens.X[i][slots[k]] = bwd(Y[i][k], k)
    end
    ens.n_done += n_steps
    return reduce(hcat, keep)'
end
