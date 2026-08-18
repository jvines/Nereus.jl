#!/usr/bin/env julia
# Validate the SHO-Q coherence discriminant (candidate-vetting test 4).
#
# Same period P for all three cases — the ONLY difference is coherence:
#   1. COHERENT (planet):   fixed-phase sinusoid over the whole baseline → high Q.
#   2. INCOHERENT (rotation): sinusoid whose phase is reshuffled each observing
#      season (spots reforming at random longitudes) → coherent within a season,
#      decoheres across seasons → finite low Q.
#   3. WHITE noise only → no significant power at P → :no_signal.
# The discriminant must call (1) :coherent and (2) :incoherent from identical
# periods and sampling — that's the whole point.

using Nereus, Statistics, Printf, Random
Random.seed!(20260711)

const P      = 30.0                 # days (rotation-like)
const AMP    = 5.0                  # m/s
const SIG    = 1.5                  # m/s white noise
const NSEAS  = 12                   # observing seasons (years)
const NPTS_S = 16                   # points per season
const SEASON_LEN = 130.0            # days observed per season (~4 cycles of P)
const YEAR   = 365.25

# time sampling: NPTS_S points in a ~130 d window each year, 12 years
t = Float64[]
season_id = Int[]
for k in 0:(NSEAS-1)
    t0 = k*YEAR + 20.0
    append!(t, sort!(t0 .+ SEASON_LEN .* rand(NPTS_S)))
    append!(season_id, fill(k+1, NPTS_S))
end
n = length(t)
baseline = maximum(t) - minimum(t)
@printf("sampling: %d pts, %d seasons, baseline %.0f d = %.0f cycles of P=%.0f d (Q_coh≈%.0f)\n",
        n, NSEAS, baseline, baseline/P, P, baseline/P)

# (1) coherent — fixed phase
φ0 = 1.1
y_coh = AMP .* sin.(2π .* t ./ P .+ φ0) .+ SIG .* randn(n)

# (2) incoherent — phase reshuffled every ~3 cycles (short-lived spots): a
# genuinely finite-coherence signal. Block = 90 d ≈ 3 cycles of P.
const BLOCK = 90.0
blk(ti) = floor(Int, ti / BLOCK) + 1
φ_blk = Dict(b => 2π*rand() for b in unique(blk.(t)))
y_inc = [AMP*sin(2π*t[i]/P + φ_blk[blk(t[i])]) for i in 1:n] .+ SIG .* randn(n)

# (3) white noise only
y_wn = SIG .* randn(n)
yerr = fill(SIG, n)

println("\n=== coherence_discriminant at the TRUE period P=$P d ===")
rc = coherence_discriminant(t, y_coh, yerr, P)
ri = coherence_discriminant(t, y_inc, yerr, P)
rw = coherence_discriminant(t, y_wn,  yerr, P)

report(tag, r) = @printf("%-26s verdict=%-11s Q_ml=%9.2f  Q_coh=%6.1f  ΔlogL_coh=%9.2f  score=%.3f\n",
                         tag, r.verdict, r.Q_ml, r.Q_coh, r.dlogL_coh, r.score)
report("coherent (planet)",  rc)
report("incoherent (rotation)", ri)
report("white noise",        rw)

ok = true
chk(n,c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
println()
chk("coherent → :coherent",              rc.verdict === :coherent)
chk("coherent Q_ml ≥ Q_coh (unbounded)", rc.Q_ml >= rc.Q_coh)
chk("coherent high-Q not disfavoured",   rc.dlogL_coh > -2.0)
chk("incoherent → :incoherent",          ri.verdict === :incoherent)
chk("incoherent Q_ml < Q_coh (forced below coherence)", ri.Q_ml < ri.Q_coh)
chk("incoherent high-Q disfavoured (Δ < −Δtol)", ri.dlogL_coh < -2.0)
chk("white noise → :no_signal",          rw.verdict === :no_signal)
chk("coherent score > incoherent score", rc.score > ri.score)

# robustness: run 3 more noise seeds, coherent/incoherent verdicts must be stable
nseed_ok = 0
for s in 1:3
    Random.seed!(1000+s)
    yc = AMP .* sin.(2π .* t ./ P .+ φ0) .+ SIG .* randn(n)
    φb = Dict(b => 2π*rand() for b in unique(blk.(t)))
    yi = [AMP*sin(2π*t[i]/P + φb[blk(t[i])]) for i in 1:n] .+ SIG .* randn(n)
    (coherence_discriminant(t,yc,yerr,P).verdict === :coherent &&
     coherence_discriminant(t,yi,yerr,P).verdict === :incoherent) && (global nseed_ok += 1)
end
chk("verdicts stable over 3 extra seeds", nseed_ok == 3)

println(ok ? "\n✅ COHERENCE DISCRIMINANT VALIDATION PASS" : "\n❌ COHERENCE — FAILURES ABOVE")
