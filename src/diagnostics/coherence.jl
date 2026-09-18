# Coherence discriminant — candidate-vetting test 4 for the RV catalog gate.
#
# Question: is a periodic signal at period `P` consistent with a strictly
# COHERENT Keplerian (a planet — phase-stable over the whole baseline), or is it
# forced to a finite quality factor Q (a finite-lived, INCOHERENT signal — stellar
# rotation / spot evolution, which decoheres over a few rotations)?
#
# Method: fit a celerite stochastically-driven simple-harmonic-oscillator (SHO) GP
# at ω0 = 2π/P and PROFILE the O(n) marginal likelihood over the quality factor Q
# (reusing `sho_coefficients` + `celerite_loglike`). The SHO covariance stays
# oscillatory over ~Q cycles, so:
#   - a genuinely coherent signal is fit *better and better* as Q → ∞ (the marginal
#     likelihood rises monotonically with Q),
#   - a signal that decorrelates after a few cycles is fit *best at finite Q* and
#     high-Q models are penalised (they predict long-lag correlation that isn't there).
# A signal coherent across the observed span needs Q ≳ N_cycles = baseline/P. If the
# data force Q well below that, the signal is finite-lived → activity.
#
# ASYMMETRIC VETO (documented, deliberate): we can DETECT finite coherence (activity),
# we CANNOT prove Q → ∞ (a planet). And it is baseline-limited — coherence longer than
# the observed span is unmeasurable, so on short baselines the test is toothless and
# the gate must lean on the indicator / evidence vetoes instead.

using Optim: optimize, NelderMead, Options, minimizer, minimum as _optmin
using Statistics: median

"""
    CoherenceResult

Output of [`coherence_discriminant`](@ref).

# Fields
- `P::Float64`          — candidate period tested
- `verdict::Symbol`     — `:coherent` (planet-consistent), `:incoherent` (activity),
                          or `:no_signal` (no significant oscillatory power at `P`)
- `coherent::Bool`      — `verdict === :coherent`
- `Q_ml::Float64`       — maximum-likelihood quality factor
- `Q_coh::Float64`      — coherence needed over the baseline (≈ N_cycles = baseline/P)
- `dlogL_coh::Float64`  — max profile logL at Q ≥ Q_coh minus the global max (≤ 0;
                          ≈ 0 ⇒ high-Q not disfavoured ⇒ coherent; ≪ 0 ⇒ finite-Q forced)
- `score::Float64`      — soft [0,1] coherence score (profile mass above `Q_coh`)
- `Q_grid`, `logL_prof` — the profiled Q grid and profile log-likelihood
"""
struct CoherenceResult
    P::Float64
    verdict::Symbol
    coherent::Bool
    Q_ml::Float64
    Q_coh::Float64
    dlogL_coh::Float64
    score::Float64
    Q_grid::Vector{Float64}
    logL_prof::Vector{Float64}
end

"""
    coherence_discriminant(t, y, yerr, P; Δtol=2.0, n_Q=30, Q_min=0.3,
                            Q_max_factor=50.0, no_signal_frac=1e-2) -> CoherenceResult

Coherence test at period `P`. `y` should be the RV RESIDUAL with the other
planets (and γ/trend) removed — this test asks only about the signal AT `P`.

`Δtol` is the profile-likelihood tolerance (in nats) below which a high-Q
(coherent) fit is deemed "not significantly disfavoured" — 2 ≈ 2σ. `Q_coh` is
set internally to the number of cycles spanned by the data; a signal must stay
coherent over that many cycles to look planet-like.
"""
function coherence_discriminant(t::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                                 yerr::AbstractVector{<:Real}, P::Real;
                                 Δtol::Real = 2.0, n_Q::Int = 30, Q_min::Real = 0.3,
                                 Q_max_factor::Real = 50.0, no_signal_frac::Real = 1e-2)
    # celerite requires time-sorted input; carry y/yerr along.
    ord = sortperm(collect(Float64, t))
    tv  = Float64.(collect(t))[ord]
    yv  = Float64.(collect(y))[ord]
    e2  = (Float64.(collect(yerr))[ord]) .^ 2
    n   = length(tv)
    n >= 8 || throw(ArgumentError("coherence_discriminant needs ≥ 8 points (got $n)"))

    # de-mean (inverse-variance weighted) so the zero-mean GP is honest
    wmean = sum(yv ./ e2) / sum(1.0 ./ e2)
    yv .-= wmean
    y_var = sum(abs2, yv) / n

    baseline = tv[end] - tv[1]
    P > 0 || throw(ArgumentError("P must be > 0"))
    N_cyc = baseline / P
    Q_coh = max(N_cyc, 1.0)
    ω0    = 2π / P

    # Inner problem: at fixed Q, optimise (log S0, log σ_wn) — the oscillator power
    # and an extra white jitter — maximising the SHO marginal likelihood.
    function neg_ll(Q::Float64, p)
        S0  = exp(p[1])
        wn2 = exp(2 * p[2])
        (isfinite(S0) && isfinite(wn2)) || return 1e12
        ar, cr, ac, bc, cc, dc = sho_coefficients(S0, Q, ω0)
        return -celerite_loglike(tv, yv, e2 .+ wn2, ar, cr, ac, bc, cc, dc)
    end

    # Q grid from over-damped (red-noise) to well above the coherence threshold.
    Qs = exp.(range(log(Q_min), log(Q_coh * Q_max_factor); length = n_Q))
    logL = fill(-Inf, n_Q)
    S0ml = zeros(n_Q)
    # warm-start init: SHO power ≈ data variance, white jitter ≈ median error
    p0 = [log(max(y_var, 1e-8)), 0.5 * log(max(median(e2), 1e-8))]
    opts = Options(iterations = 200, g_tol = 1e-8)
    for i in 1:n_Q
        res = optimize(p -> neg_ll(Qs[i], p), p0, NelderMead(), opts)
        logL[i] = -_optmin(res)
        pml = minimizer(res)
        S0ml[i] = exp(pml[1])
        p0 = pml                      # warm-start the next Q
    end

    Lmax, imax = findmax(logL)
    Q_ml = Qs[imax]

    hi = findall(>=(Q_coh), Qs)
    dlogL_coh = isempty(hi) ? -Inf : maximum(@view logL[hi]) - Lmax
    coherent  = dlogL_coh >= -Δtol

    # soft score: profile-normalised likelihood mass above Q_coh (∝-marginal proxy)
    w = exp.(logL .- Lmax); w ./= sum(w)
    score = sum(@view w[hi])

    # no-signal guard: the ML oscillator variance (k(0) ≈ S0·ω0·Q) is negligible
    # relative to the data scatter ⇒ there is nothing at P to judge.
    sho_var = S0ml[imax] * ω0 * Q_ml
    verdict = sho_var < no_signal_frac * y_var ? :no_signal :
              (coherent ? :coherent : :incoherent)

    return CoherenceResult(Float64(P), verdict, verdict === :coherent, Q_ml, Q_coh,
                           dlogL_coh, score, Qs, logL)
end
