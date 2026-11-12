# Differential-evolution move for the affine-invariant ensembles (pt_emcee,
# transdim_pt_emcee's within-model step, pt_whitening, the `ensemble` engine and
# population annealing's stretch mutation).
#
#     x' = x_i + γ (x_j - x_k) + ε,     j ≠ k from the complementary half
#
# ter Braak 2006 (Stat. Comput. 16, 239); Nelson, Ford & Payne 2014 (ApJS 210,
# 11) for its use on exoplanet orbits; emcee's `DEMove`. j and k come from the
# half of the rung the stretch move draws its partner from, which no walker
# writes while this half moves, so given that half the proposal is symmetric:
# (j, k) and (k, j) are equally likely and γ, ε are drawn independently of
# both, so the increment is as likely as its negative. The Metropolis-Hastings
# ratio is the (tempered) target ratio alone -- no z^(d-1) factor.
#
# Why the stretch move needs it. A stretch proposal lies on the line through
# the walker and one partner, at least halfway out from the partner. A walker
# in a thin curved filament proposes along a chord that leaves the filament
# almost always, and the filament keeps whatever share of walkers burn-in
# left there: on the default transit-only fit (b = 0.3, e = 0, phic anchor) one
# seed in six started half its walkers on the constant-duration ridge at
# e > 0.5, ω ≈ -28°, and drained it from 51 % to 40 % over the 1000 recorded
# steps (31 % to 14 % at four times the length), while nested sampling puts
# 1.4 % there. R-hat stayed at 1.0002: the walkers drift together. The DE
# difference x_j - x_k of two walkers in the filament points along it, so a
# walker in it steps along the filament, and one in the bulk can be carried
# by a bulk-to-filament difference (γ = 1) and back.
#
# γ is γ₀ = 2.38/√(2m), the optimal random-walk scale for a Gaussian in the m
# coordinates the move changes, jittered by 10 % so the step length is not one
# value; with probability 0.1 it is 1, which carries a walker by the full
# separation of two others -- from one mode to another when they sit in
# different modes. ε is Gaussian with 1e-4 of the complementary half's spread
# along each coordinate (per rung): far below any step that matters, but it
# lets the move leave the affine hull of the ensemble, which neither it nor the
# stretch move could otherwise do (copied walkers leave a rung rank-deficient).
# γ is not tuned. Over 18 paired runs (transit-only, two-planet RV and RV +
# Hipparcos ridge fits, two seeds each, at shares 0.2, 0.3 and 0.5), tuning it
# per rung in burn-in toward 25 % acceptance changed the ESS per likelihood
# evaluation by a factor 1.007 ± 0.016; the DE-snooker (ter Braak & Vrugt 2008)
# on a further 10 % of the walker updates, by 0.996 ± 0.026.
#
# The default share, `DE_MOVE_DEFAULT` = 0.5: on those fits it gave 1.09 ± 0.02
# times the stretch-only ESS per evaluation (the seed-3 transit fit left out,
# whose stretch-only run sat on the filament), 0.1-0.3 gave 1.01-1.03, within
# noise of 0.5, and it drained a stranded ensemble fastest. Even so a stranded
# ensemble drains slowly: the seed-3 run above, continued from the end of its
# stretch-only burn-in with 48 % of its walkers on the filament, still kept 25 %
# there on average over the second thousand steps (0.1: 29 %, 0.2: 28 %, 0.3:
# 28 %; stretch only: 34 %). The move's work is done during burn-in, where it
# keeps the filament from filling.
#
# Coordinates are the stretch move's, exactly. Each sampler proposes on the
# scale its stretch move does (pt_emcee and transdim: a LogUniform /
# ModJeffreys parameter as log(x + s); the `ensemble` engine: the
# unconstrained coordinates of the target; pt_whitening and population
# annealing: bounded x), with the same Jacobian where the target is written on
# another scale (transdim keeps walkers in x: Π (x'+s)/(x+s)). A full-circle or
# polar angle is a plain coordinate in its chart there too: the difference is
# taken in the chart, a proposal past a wall is outside the support and
# rejected, and the re-cuts that move a chart during burn-in relabel every
# walker between steps, never inside one. Not wrapping the difference onto the
# circle is what keeps the move symmetric on the chart -- the same reason the
# stretch move is not re-wrapped (`WrappedUniform`). Coordinates the stretch
# leaves (a `burnin_hold` coordinate, every walker sharing it; a trans-dim
# walker's parked slots) the DE move leaves too.


using Random: AbstractRNG

"""
    DE_MOVE_DEFAULT

The default `de_move` of every ensemble sampler: the probability, per walker
and step, of a differential-evolution move in place of the stretch move. See
src/samplers/de_move.jl for the move, and `sample_pt_emcee` for the default.
"""
const DE_MOVE_DEFAULT = 0.5

# γ jitter (relative), the probability of γ = 1, and ε relative to the spread.
const _DE_GAMMA_JITTER = 0.1
const _DE_MODE_HOP = 0.1
const _DE_EPS = 1e-4

# The checkpoint fingerprint's share: nothing at 0, the stretch-only kernel every
# checkpoint written before the move existed ran.
_de_fp(de_move::Float64) = de_move == 0 ? (;) : (; de_move)

"""
    _de_check(de_move) -> Float64

`de_move` as a probability, or an `ArgumentError`.
"""
function _de_check(de_move::Real)
    p = Float64(de_move)
    0 <= p <= 1 || throw(ArgumentError(
        "de_move is a probability (per walker and step, of a differential-" *
        "evolution move in place of the stretch move); got $de_move"))
    return p
end

"""
    _de_chosen(rng, p_de) -> Bool

Whether this walker update is a DE move (probability `p_de`) rather than a
stretch. Draws nothing at `p_de = 0`, so a run at `de_move = 0` draws exactly
what the stretch-only kernel did.
"""
@inline _de_chosen(rng::AbstractRNG, p_de::Float64) = p_de > 0 && rand(rng) < p_de

"""
    _de_pair(rng, lo, hi) -> (j, k)

Two distinct walkers, uniform over ordered pairs from `lo:hi`.
"""
@inline function _de_pair(rng::AbstractRNG, lo::Int, hi::Int)
    j = rand(rng, lo:hi)
    k = rand(rng, lo:(hi - 1))
    k >= j && (k += 1)
    return j, k
end

# The optimal random-walk scale for a Gaussian in the m coordinates a move
# changes (ter Braak 2006).
@inline _de_gamma0(m::Int) = 2.38 / sqrt(2 * max(m, 1))

"""
    _de_gamma(rng, base) -> γ

The DE scale of one proposal: 1 with probability `_DE_MODE_HOP`, else `base`
jittered by `_DE_GAMMA_JITTER`.
"""
@inline _de_gamma(rng::AbstractRNG, base::Float64) =
    rand(rng) < _DE_MODE_HOP ? 1.0 : base * (1 + _DE_GAMMA_JITTER * randn(rng))

# Acceptance per rung from the move's tallies; NaN where none was proposed.
_de_acceptance(acc::AbstractVector{<:Integer}, prop::AbstractVector{<:Integer}) =
    Float64[p > 0 ? a / p : NaN for (a, p) in zip(acc, prop)]

"""
    _de_spread(x) -> Float64

The standard deviation of the finite values of `x` (0 for fewer than two, or
when they are all equal): the scale of ε along one coordinate.
"""
function _de_spread(x::AbstractVector{Float64})
    n = 0; s = 0.0; lo = Inf; hi = -Inf
    @inbounds for v in x
        isfinite(v) || continue
        n += 1; s += v
        lo = min(lo, v); hi = max(hi, v)
    end
    (n < 2 || lo == hi) && return 0.0
    μ = s / n; ss = 0.0
    @inbounds for v in x
        isfinite(v) && (ss += (v - μ)^2)
    end
    return sqrt(ss / (n - 1))
end

"""
    _de_spread!(scale, S, lo, hi)

`scale[t, d]` = `_de_spread` of coordinate `d` over walkers `lo:hi` of rung `t`
of `S[t, w, d]`.
"""
function _de_spread!(scale::AbstractMatrix{Float64}, S::AbstractArray{Float64,3},
                     lo::Int, hi::Int)
    @inbounds for d in axes(S, 3), t in axes(S, 1)
        scale[t, d] = _de_spread(view(S, t, lo:hi, d))
    end
    return scale
end

"""
    _de_fill!(buf, S, t, w, j, k, γ, scale, held, rng)

The DE proposal for walker `w` of rung `t` of `S[t, w, d]`, into `buf`:
x_w + γ (x_j - x_k) + ε, with ε_d ~ N(0, (`_DE_EPS` scale[t, d])²); a
coordinate with `held[d]` stays.
"""
@inline function _de_fill!(buf::AbstractVector{Float64}, S::AbstractArray{Float64,3},
                           t::Int, w::Int, j::Int, k::Int, γ::Float64,
                           scale::AbstractMatrix{Float64}, held::AbstractVector{Bool},
                           rng::AbstractRNG)
    @inbounds for d in eachindex(buf)
        buf[d] = held[d] ? S[t, w, d] :
                 S[t, w, d] + γ * (S[t, j, d] - S[t, k, d]) +
                 _DE_EPS * scale[t, d] * randn(rng)
    end
    return buf
end

# The `ensemble` engine's layout, x[d, w], one ensemble: the same proposals.
@inline function _de_fill!(buf::AbstractVector{Float64}, X::AbstractMatrix{Float64},
                           w::Int, j::Int, k::Int, γ::Float64,
                           scale::AbstractVector{Float64}, rng::AbstractRNG)
    @inbounds for d in eachindex(buf)
        buf[d] = X[d, w] + γ * (X[d, j] - X[d, k]) + _DE_EPS * scale[d] * randn(rng)
    end
    return buf
end
