# ------------------------------------------------------------------
# Scouted starts for the local samplers.
#
# rjmcmc and moms (with a fixed set of planets) and NUTS started cold each
# began a chain at ONE prior draw and moved it with a local kernel only: slice
# or random-walk updates of one coordinate at a time, or Hamiltonian
# trajectories. A draw that lands outside the basin of a narrow mode climbs to
# the nearest plateau instead and stays there for the whole run. Measured on an
# RV + Hipparcos-IAD companion (circular, 0.05 mas per scan, a on
# LogUniform(0.8, 2), M_sec on LogUniform(0.003, 0.03)): a chain that started
# at the wrong period settled with M_sec on the lower edge of its prior, where
# any orbital signal only hurts the fit, 23,000 to 46,000 nats below the mode.
# At i = 45° 3 of 8 single chains of each of rjmcmc, moms and cold NUTS did,
# and of four chains one or two; the run reported their pooled draws as the
# posterior: i = 36.7 ± 8.1° against 44.6 ± 0.9°.
#
# Nothing inside such a chain can find the mode: it has no point of reference.
# So a chain now starts from the best-supported of several. `_scout_starts`
# draws `n_scouts` prior points as the chain would its one, moves each with
# the sampler's own slice updates until it stops climbing, and then asks of
# every scout the question `_pt_prune_stranded!` asks of a walker: can the
# region it settled in carry posterior mass? With A the scouts near the best
# one and Z_A its Laplace volume (the scouts' covariance, or with too few of
# them the axis widths at the top of the best one's mode, `_pt_axis_widths`),
#     Z_{L<c} = ∫_{L<c} π L ≤ c   with   log c = log Z_A - tau,
# so a scout with log L below c sits where the whole region of that likelihood
# weighs under e^-tau of A, and is dropped. The chain starts at one of the
# rest, picked at random: a genuinely multimodal posterior still sends
# different chains to different modes, and a weak signal whose plateau carries
# the mass keeps its plateau scouts, both as with the pruning. On the target
# above, 5 of 16 scouts settled on the plateau (log L ≈ -23,000), the cut came
# out at log L ≈ 148 and the other 11 sat at 192-195.
#
# The ensemble (src/samplers/ensemble.jl) takes the survivors as up to half
# its walkers: its burn-in pruning moves stranded walkers onto those that
# found the mode, and had nothing to move them to when none had. One chain of
# four at i = 5° sat at i = 161°, 1,500 nats below the mode, for the whole run.
# Cold pt_hmc and pathfinder_init start from them through `_cold_starts`
# (src/samplers/nuts.jl): pt_hmc freezes each chain's kernel where its warm-up
# ended, and a β = 1 warm-up on the plateau left a kernel that could not move
# at the mode; Pathfinder's 8 L-BFGS runs all stopped on such plateaus.
#
# A chain is never started from a point the bound did not clear, but nothing
# guarantees that some scout found the mode; with p the chance that one prior
# draw does not, all of them miss with probability p^n_scouts (p ≈ 0.4 above,
# 1e-6 at 16 scouts). The cost is the scouts' sweeps, 20-50 slice sweeps each
# there: ~5 % of an 8000 + 4000 rjmcmc run.
# ------------------------------------------------------------------

"""
    _scout_survivors(theta, data, rng, widths; draw!, n_scouts = 16, tau = 10.0,
                     max_sweeps = 100, ctr = nothing) -> Vector{Vector{Float64}}

The scouts the bound keeps (see the note above), each a full `theta.values`,
in a random order. `theta` gives the trans-dimensional state the chain starts
in, which every scout keeps; only its active unfrozen coordinates move.
`draw!(th, rng)` writes a prior draw into `th` (the sampler's own
initialisation). `widths` are the slice step-out widths, one per unfrozen
coordinate (`_prior_widths`). Each scout runs blocks of 10 slice sweeps
(`_within_model_move!`) until a block's best log posterior rose by less than
d/2 over the previous block's (d the number of active coordinates), at least
2 blocks and at most `max_sweeps` sweeps. Evaluations go to `ctr`. Empty if
no draw had a finite density.
"""
function _scout_survivors(theta::Theta, data::Data, rng::AbstractRNG,
                          widths::Vector{Float64}; draw!, n_scouts::Int = 16,
                          tau::Float64 = 10.0, max_sweeps::Int = 100,
                          ctr::Union{LikelihoodCounter,Nothing} = nothing)
    params = theta.params
    act, pos = _active_unfrozen(params, theta.td)
    d = length(act)
    sc = Theta{Float64}(params, copy(theta.values);
                        td = theta.td === nothing ? nothing : copy(theta.td))
    vals = Vector{Vector{Float64}}()
    lps = Float64[]
    lls = Float64[]
    for _ in 1:n_scouts
        lp, ll = -Inf, -Inf
        for _ in 1:100
            sc.values .= theta.values
            draw!(sc, rng)
            lp = log_prior(sc)
            ll = isfinite(lp) ? _eval_ll(sc, data, ctr) : -Inf
            isfinite(lp) && isfinite(ll) && break
        end
        isfinite(lp) && isfinite(ll) || continue
        prev = -Inf
        for blk in 1:(d == 0 ? 0 : cld(max_sweeps, 10))
            top = -Inf
            for _ in 1:10
                _within_model_move!(sc, data, rng, 1.0, widths, lp, ll;
                                    ctr = ctr) do nlp, nll
                    lp, ll = nlp, nll
                end
                top = max(top, lp + ll)
            end
            blk >= 2 && top < prev + d / 2 && break
            prev = top
        end
        push!(vals, copy(sc.values))
        push!(lps, lp)
        push!(lls, ll)
    end
    isempty(vals) && return vals
    keep = d == 0 ? collect(eachindex(vals)) :
           _scouts_supported(sc, data, vals, lps, lls, act, pos, widths, tau, ctr)
    return vals[keep[randperm(rng, length(keep))]]
end

"""
    _scout_starts(theta, data, rng, widths; draw!, n_starts = 1, kwargs...)
        -> Vector{Vector{Float64}}

`n_starts` start points (full `theta.values`): the survivors of
[`_scout_survivors`](@ref) in their random order, repeated if there are fewer
of them than `n_starts`; with none, plain draws of `draw!`.
"""
function _scout_starts(theta::Theta, data::Data, rng::AbstractRNG,
                       widths::Vector{Float64}; draw!, n_starts::Int = 1, kwargs...)
    surv = _scout_survivors(theta, data, rng, widths; draw!, kwargs...)
    isempty(surv) || return [copy(surv[mod1(c, length(surv))]) for c in 1:n_starts]
    th = Theta{Float64}(theta.params, copy(theta.values);
                        td = theta.td === nothing ? nothing : copy(theta.td))
    return [(th.values .= theta.values; draw!(th, rng); copy(th.values))
            for _ in 1:n_starts]
end

# The scouts that may carry posterior mass: those whose log L is at least
# log Z_A - tau (the cut of `_pt_prune_stranded!` at β = 1, over the active
# coordinates `act`, theta slots, at unfrozen positions `pos`). All of them
# when no volume can be measured.
function _scouts_supported(sc::Theta, data::Data, vals, lps, lls, act, pos,
                           widths, tau, ctr)
    n = length(vals)
    d = length(act)
    ℓ = lps .+ lls
    b = argmax(ℓ)
    best = ℓ[b]
    allow = quantile(Chisq(d), 1 - 1e-6) / 2
    A = findall(>=(best - allow), ℓ)
    seam = Set(seam_indices(sc.params))
    circ = Int[k for k in 1:d if pos[k] in seam]
    state = Array{Float64,3}(undef, 1, n, d)
    for s in 1:n, k in 1:d
        state[1, s, k] = vals[s][act[k]]
    end
    yb = state[1, b, :]
    lv = length(A) >= d + 2 ? _pt_half_logdet_cov(state, 1, A, yb, circ) : nothing
    if lv === nothing
        # Too few scouts near the best to give a covariance: the volume from
        # the density about the top of the best one's mode, as the pruning
        # does for a rung with few walkers in A.
        sc.values .= vals[b]
        ℓat = y -> begin
            @inbounds for k in 1:d
                sc.values[act[k]] = y[k]
            end
            v = _tempered_lp(sc, data, 1.0; ctr = ctr)
            isfinite(v) ? v : -Inf
        end
        h0 = Float64[0.1 * widths[p] for p in pos]
        σ = _pt_axis_widths(ℓat, yb, best, h0)
        σ === nothing && return collect(1:n)
        yc, ℓc = _pt_climb(ℓat, yb, best, σ)
        if ℓc > best
            σc = _pt_axis_widths(ℓat, yc, ℓc, h0)
            σc === nothing || ((best, σ) = (ℓc, σc))
        end
        lv = sum(log, σ)
    end
    cut = best + 0.5 * d * log(2π) + lv - tau
    return union!([b], findall(>=(cut), lls))
end
