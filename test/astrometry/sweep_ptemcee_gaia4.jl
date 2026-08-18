# How lean can ptemcee go on the Gaia-4 astrometry-only fit and still get the
# same recovery (P,M,e,i) AND the same TI log-evidence (~675)? The 150k-sample
# baseline is 100 walkers × 1500 kept steps — the real cost is
# n_temps × n_walkers × n_steps likelihood evaluations. Sweep walkers/steps down
# and watch recovery, logZ, min swap-acceptance, #evals, and wall-clock.

using Nereus, MCMCChains, Printf
using Statistics: median

const SID = 1457486023639239296
const M_PRI = 0.644
const TRUTH = (P = 571.3, M = 11.8, e = 0.338, i = 116.9)

xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, SID)

build() = build_target(
    M_pri = M_PRI,
    planets = (b = (
        a = LogUniformPrior(0.3, 4.0), M_sec = LogUniformPrior(0.001, 0.05),
        sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
        Mo = UniformPrior(0.0, 2π), inc = SinePrior(), Omega = UniformPrior(0.0, 2π),
    ),),
    iad = src.iad, plx = NormalPrior(13.628, 0.021), M_s = M_PRI,
)

function summ(chains)
    a = vec(Array(chains[:, :a_k1, :])); Ms = vec(Array(chains[:, :M_sec_k1, :]))
    ses = vec(Array(chains[:, :sesinw_k1, :])); sec = vec(Array(chains[:, :secosw_k1, :]))
    inc = vec(Array(chains[:, :inc_k1, :]))
    P = [365.25 * sqrt(a[j]^3 / (M_PRI + Ms[j])) for j in eachindex(a)]
    (P = median(P), M = median(Ms .* 1047.57), e = median(ses .^ 2 .+ sec .^ 2),
     i = median(rad2deg.(inc)))
end

# (n_temps, n_walkers, n_steps, n_burnin)
CONFIGS = [
    (8, 100, 2500, 1000),   # baseline (the 150k run)
    (8,  50, 1500,  500),
    (8,  30, 1200,  400),
    (8,  24, 1000,  400),
    (8,  20,  800,  300),
    (6,  30, 1200,  400),   # also thin the ladder
]

@printf("%-4s %-4s %-6s %-6s | %8s %6s %6s %7s | %8s %8s %6s %8s\n",
        "nT", "nW", "steps", "burn", "P_d", "M_J", "e", "i_deg",
        "logZ", "kept", "min", "Nevals")
@printf("%-4s %-4s %-6s %-6s | %8.1f %6.1f %6.3f %7.1f | %8s\n",
        "", "", "", "TRUTH", TRUTH.P, TRUTH.M, TRUTH.e, TRUTH.i, "—")
println("-"^92)
for (nT, nW, nS, nB) in CONFIGS
    tgt = build()
    t0 = time()
    r = try
        sample_ptemcee(tgt, tgt.data; n_temps = nT, n_walkers = nW,
                       n_steps = nS, n_burnin = nB, seed = 42, show_progress = false)
    catch err
        @printf("%-4d %-4d %-6d %-6d | FAILED: %s\n", nT, nW, nS, nB, sprint(showerror, err))
        continue
    end
    dt = time() - t0
    r === nothing && (@printf("%-4d %-4d %-6d %-6d | returned nothing (too few samples)\n", nT, nW, nS, nB); continue)
    s = summ(r.chains)
    kept = length(vec(Array(r.chains[:, :a_k1, :])))
    @printf("%-4d %-4d %-6d %-6d | %8.1f %6.1f %6.3f %7.1f | %8.1f %8d %6.1f %8d\n",
            nT, nW, nS, nB, s.P, s.M, s.e, s.i, r.log_evidence, kept, dt / 60, r.n_evals)
end
println("-"^92)
println("target recovery: P≈578 M≈10.8 e≈0.40 i≈121 (astrometry-only value all samplers agree on)")
println("logZ consensus ≈ 675 (ptemcee TI / nested / HMC SS⁺). Find the smallest #evals that holds both.")
