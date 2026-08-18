#!/usr/bin/env julia
# HOW MUCH OF THE LADDER VERDICT IS THE PRIOR?
#
# The 5p/7p/9p evidences are exact, but "exact" is not "assumption-free": the
# acceleration and jerk coefficients need PROPER priors or the Bayes factors are
# undefined, and the width of those priors enters the answer directly. For a
# well-measured coefficient the Occam penalty for including it scales as the
# log of the prior width, so widening sigma_accel by a decade costs ~log(10) per
# added parameter — ~4.6 nats for the two acceleration terms. That is a factor
# of 100 in odds PER DECADE.
#
# So the honest question is not "is the default right" but "over what range of
# prior width does each source keep its verdict, and is the default inside it".
# Sources with large evidence gaps will not care. Near-ties will flip. Near-ties
# are exactly where the method claims to add value over Gaia's cascade, so this
# is where the claim is weakest and has to be stated.
#
# Two parts:
#   1. real DR4 sources — verdict vs prior width, and how wide the stable range is
#   2. synthetic with KNOWN injected acceleration — where does recovery break
using Nereus, Printf, Statistics, Random
using Nereus: ladder_probabilities, astrom_logZ, astrom_design, n_iad, IADData

# Prior with the first five held fixed and the accel/jerk widths scanned.
mkprior(s_acc) = o -> begin
    σ = Float64[5.0, 5.0, 5.0, 5.0, 5.0]
    o >= 7 && (push!(σ, s_acc); push!(σ, s_acc))
    o >= 9 && (push!(σ, s_acc / 3); push!(σ, s_acc / 3))   # jerk ~ accel/T
    σ
end

const DEFAULT_SACC = 2 * 5.0 / 10.0^2          # = 0.1 mas/yr², the shipped default
const SCAN = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]

# ---------- 1. real sources ----------
srcs = read_gaia_epoch_votable(fetch_gaia_dr4_prerelease())
ids  = sort(collect(keys(srcs)))

println("VERDICT vs ACCELERATION PRIOR WIDTH (mas/yr²), real DR4 sources")
println("default is $(DEFAULT_SACC).  entries are best-rung(P(best))\n")
@printf("%-20s %6s", "source_id", "χ²/dof")
for s in SCAN; @printf(" %12s", @sprintf("%.0e", s)); end
println("   stable?")

function row(sid)
    iad = srcs[sid].iad
    n_iad(iad) > 20 || return nothing
    verdicts = String[]; bests = Int[]
    c2 = NaN
    for s_acc in SCAN
        r = ladder_probabilities(iad; prior = mkprior(s_acc))
        push!(bests, r.best)
        push!(verdicts, @sprintf("%d(%.2f)", r.best, maximum(r.prob)))
        c2 = minimum(r.chi2_reduced)
    end
    return (verdicts, bests, c2)
end

nflip = 0; ntot = 0
for sid in ids
    v = row(sid); v === nothing && continue
    verdicts, bests, c2 = v
    # Only count a flip between CONFIDENT verdicts. When the prior is tight
    # enough that the extra coefficients cannot move, the rungs collapse to the
    # same model, the evidences converge and probability splits ~1/3 each — the
    # argmax is then arbitrary noise, not instability. That is the P~0.33
    # signature in the table.
    conf = [b for (b, v) in zip(bests, verdicts) if
            parse(Float64, split(split(v, "(")[2], ")")[1]) > 0.5]
    stable = length(unique(conf)) <= 1
    global ntot += 1; stable || (global nflip += 1)
    @printf("%-20d %6.2f", sid, c2)
    for x in verdicts; @printf(" %12s", x); end
    println(stable ? "   yes" : "   NO — FLIPS")
end
@printf("\n%d/%d sources change their preferred rung across 5 decades of prior width\n",
        nflip, ntot)

# ---------- 2. synthetic, known truth ----------
function synth(; accel, n = 600, σ = 0.15, seed = 4)
    rng = MersenneTwister(seed)
    t = collect(range(0.0, 2000.0; length = n))
    dt = (t .- mean(t)) ./ 365.25
    psi = 2π .* rand(rng, n)
    pf = cos.(2π .* t ./ 365.25)
    w = similar(dt)
    for j in 1:n
        s, c = sincos(psi[j])
        w[j] = 0.3s - 0.2c + 1.5pf[j] + 2.0*s*dt[j] - 1.0*c*dt[j] +
               accel*s*dt[j]^2/2 + σ*randn(rng)
    end
    IADData(t=t, abscissa=w, abscissa_err=fill(σ,n), psi=psi,
            parallax_factor=pf, pm_factor=dt)
end

println("\nSYNTHETIC: injected acceleration recovered? (P(7p or 9p))")
@printf("%-14s", "accel true")
for s in SCAN; @printf(" %12s", @sprintf("%.0e", s)); end
println()
for a in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.2, 1.0)
    iad = synth(accel = a)
    @printf("%-14.2f", a)
    for s_acc in SCAN
        r = ladder_probabilities(iad; prior = mkprior(s_acc))
        @printf(" %12.3f", r.prob[2] + r.prob[3])
    end
    println()
end

println("""
\nReading it: each column is a prior width, each row a true acceleration. The
top row (accel = 0) is the FALSE-POSITIVE rate — it must stay near zero at every
prior width, and a prior far too tight is what would break it. Lower rows are
detections. The prior width at which a genuine acceleration stops being
detected is the method's real sensitivity limit, and it is a CHOICE, not a
property of the data.""")
