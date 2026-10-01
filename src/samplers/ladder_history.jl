# What a tempered ensemble run did to its own ladder, step by step.
#
# `sample_pt_emcee` and `sample_transdim_pt_emcee` report the run-total swap
# acceptance and the final β. Both are summaries of a trajectory: a ladder
# that was still moving when burn-in ended, a pair whose exchange died half
# way through, a hot rung that stopped carrying states anywhere -- none of
# that is visible in a total. This is the trajectory, in the three quantities
# reddemcee records (Peña & Jenkins 2026, A&A 706 A323) and EMPEROR draws as
# `rates`: the temperatures, the swap acceptance and the swap mean distance.
# `plot_ladder_rates` and `plot_beta_ladder` (src/plotting/ladder_plots.jl)
# read it.
#
# Recording draws no random numbers and changes no state, so a run with it is
# the run without it.

export LadderHistory

"""
    LadderHistory

Per-step history of a tempering ladder, returned as `result.ladder` by
[`sample_pt_emcee`](@ref) and [`sample_transdim_pt_emcee`](@ref).

Rungs are ordered as `result.betas` is: cold (`β = 1`) first. A "pair" `k`
is the adjacent rungs `(k, k + 1)`.

# Fields
- `betas::Matrix{Float64}` — `(n_steps, n_temps)`. The ladder at the END of
  each step, i.e. after that step's adaptation, so the last row is
  `result.betas`. Constant unless `adapt_ladder = true`, and then only up to
  `n_burnin`.
- `swap_rate::Matrix{Float64}` — `(n_steps, n_temps - 1)`. Fraction of the
  step's swap proposals for each pair that were accepted. Every step proposes
  the same number per pair, so the column means are `result.acceptance_swap`.
- `swap_distance::Matrix{Float64}` — `(n_steps, n_temps - 1)`. The swap mean
  distance (SMD) of reddemcee: the mean, over the step's swap proposals for
  the pair, of the distance the exchange moved the walker, counting a rejected
  proposal as zero. The distance is Euclidean with every dimension divided by
  its prior width on the scale the walkers move on (`_swap_distance_scale`),
  so it is dimensionless and at most `√n_dim`. It is what a swap acceptance
  cannot say: a pair can exchange often and still carry nothing, because the
  two rungs already sit on the same states.
- `mean_logL::Vector{Float64}` — `⟨log L⟩` at each rung over the post-burn-in
  steps: the integrand of thermodynamic integration, and the one array every
  tempered evidence estimator in `result.evidence` reads. `NaN` at a rung
  with no finite post-burn-in draw.
- `n_burnin::Int` — steps before recording of the chain (and of `mean_logL`)
  began.

Angles are not wrapped in `swap_distance`: two walkers on either side of a
seam count as far apart, as they do in reddemcee.
"""
struct LadderHistory
    betas::Matrix{Float64}
    swap_rate::Matrix{Float64}
    swap_distance::Matrix{Float64}
    mean_logL::Vector{Float64}
    n_burnin::Int
end

function Base.show(io::IO, L::LadderHistory)
    print(io, "LadderHistory(", size(L.betas, 1), " steps × ",
          size(L.betas, 2), " temperatures)")
end

"""
    _swap_distance_scale(priors, flat_shift) -> Vector{Float64}

Width of each parameter's prior on the scale the ensemble moves on — the `D`
that reddemcee divides a swap's displacement by, so that a period spanning
three decades and an eccentricity spanning one unit both count for one.

A prior the walkers move in `log(x + s)` (`flat_shift[d] = s`; see
`log_scale_shift`) is measured there. A prior with no hard bounds has no
width, so it gets the central 99.73% of its mass, ±3σ for a Normal.
"""
function _swap_distance_scale(priors, flat_shift)
    D = Vector{Float64}(undef, length(priors))
    for (d, ps) in enumerate(priors)
        lo, hi = bounds(ps)
        if !(isfinite(lo) && isfinite(hi))
            q_lo, q_hi = quantile(ps.dist, 0.00135), quantile(ps.dist, 0.99865)
            isfinite(lo) || (lo = q_lo)
            isfinite(hi) || (hi = q_hi)
        end
        s = flat_shift[d]
        w = s === nothing ? hi - lo : log(hi + s) - log(lo + s)
        D[d] = (isfinite(w) && w > 0) ? w : 1.0
    end
    return D
end

"""
Distance between walker `w` at rung `t` and walker `w2` at rung `t + 1`, each
dimension in units of `scale`. `state` holds the walkers on their move scale.
"""
@inline function _swap_distance(state::AbstractArray{Float64,3}, t::Int, w::Int,
                                w2::Int, scale::Vector{Float64})
    acc = 0.0
    @inbounds for d in eachindex(scale)
        δ = (state[t, w, d] - state[t + 1, w2, d]) / scale[d]
        acc += δ * δ
    end
    return sqrt(acc)
end

"""
The trans-dim form. `state` is in `x`, so a log-scale dimension is mapped here;
and only dimensions ACTIVE IN BOTH walkers count -- a parked slot sits at its
off-value, which is a bookkeeping position, not somewhere the walker is.
`owner[d]` is the planet slot (`> 0`), noise toggle (`< 0`) or nothing (`0`)
that dimension `d` belongs to.
"""
@inline function _swap_distance(state::AbstractArray{Float64,3}, t::Int, w::Int,
                                w2::Int, scale::Vector{Float64},
                                flat_shift::Vector{Union{Nothing,Float64}},
                                owner::Vector{Int}, a::TransDimState,
                                b::TransDimState)
    acc = 0.0
    @inbounds for d in eachindex(scale)
        o = owner[d]
        if o > 0
            (a.planet_active[o] && b.planet_active[o]) || continue
        elseif o < 0
            (a.noise_active[-o] && b.noise_active[-o]) || continue
        end
        x, y = state[t, w, d], state[t + 1, w2, d]
        s = flat_shift[d]
        if s !== nothing
            (x + s > 0 && y + s > 0) || continue
            x, y = log(x + s), log(y + s)
        end
        δ = (x - y) / scale[d]
        acc += δ * δ
    end
    return sqrt(acc)
end

"""Per-rung `⟨log L⟩` from the evidence accumulator; `NaN` where it is empty."""
_mean_logL(acc::EvidenceAccumulator) =
    Float64[acc.n[k] > 0 ? acc.sum_logL[k] / acc.n[k] : NaN
            for k in eachindex(acc.n)]

"""
Close a run's history: keep the `n_done` steps actually taken (a
run-until-converged stop leaves the rest of the preallocated rows unwritten).
"""
function _ladder_history(β_hist::Matrix{Float64}, swap_hist::Matrix{Float64},
                         smd_hist::Matrix{Float64}, acc::EvidenceAccumulator,
                         n_done::Int, n_burnin::Int)
    return LadderHistory(β_hist[1:n_done, :], swap_hist[1:n_done, :],
                         smd_hist[1:n_done, :], _mean_logL(acc),
                         min(n_burnin, n_done))
end
