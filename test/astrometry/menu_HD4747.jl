# Real-target recovery menu — HD 4747 B (RV + HGCA + relAST, build_target API).
#
# Scores the full ORBIT (a, e, i, M_sec) against Peretti+ 2019 (A&A 631, A107),
# the definitive combined HIRES+CORALIE RV + NACO/NIRC2/SPHERE imaging fit:
#   a = 10.01 ± 0.21 AU, e = 0.7362 ± 0.0025, i = 47.3 ± 1.6°, M_B = 70.0 ± 1.6 MJup.
# (In-repo builder only checked M_sec = 65.3 ± 3.3 from Brandt+ 2019 — an
# M_sec-only check is a false-positive risk, so we validate the orbit too.)
# relAST is a short ~9-yr arc (2008-2017) of a ~33-yr orbit, so a is the least
# constrained → wider tolerance there. i may land at the 180-i reflection.

using Nereus
using MCMCChains
using Statistics: median, std, quantile
using Printf

const RESULTS = joinpath(@__DIR__, "menu_HD4747_results.txt")
logline(s) = (open(RESULTS, "a") do io; println(io, s); end; println(s); flush(stdout))

println("=" ^ 70); println("HD 4747 — sampler recovery menu"); println("=" ^ 70)

DATADIR  = joinpath(@__DIR__, "..", "..", "..", "data", "HD4747")
HGCAFILE = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062", "HGCA_vEDR3.fits")
hgca   = load_hgca_row(HGCAFILE, 3850)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD4747_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD4747_relAST.txt"))
println("Loaded: $(length(rvdat.t)) RV, $(n_relast(relast)) relAST, HGCA HIP 3850")

make_target() = build_target(
    M_pri = 0.856,                                # Crepp+ 2016
    planets = (b = (
        a      = LogUniformPrior(1.0, 100.0),
        M_sec  = LogUniformPrior(0.001, 0.5),
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        Mo     = UniformPrior(0.0, 2π),
        inc    = SinePrior(),
        Omega  = UniformPrior(0.0, 2π),
    ),),
    rv = (HIRES = (data = rvdat, sigma = LogUniformPrior(0.1, 30.0)),),
    relAST = relast,
    hgca   = hgca,
)

# Peretti+ 2019 combined-fit orbit
const REF = (a=10.01, a_sig=0.21, e=0.7362, i_deg=47.3, M_J=70.0, M_J_sig=1.6)

function score_chains(name, chains, logZ, secs)
    a_v   = vec(Array(chains[:, :a_k1, :]))
    M_sec = vec(Array(chains[:, :M_sec_k1, :]))
    ses_v = vec(Array(chains[:, :sesinw_k1, :]))
    sec_v = vec(Array(chains[:, :secosw_k1, :]))
    inc_v = vec(Array(chains[:, :inc_k1, :]))
    e_arr = ses_v.^2 .+ sec_v.^2
    M_J   = M_sec .* 1047.57
    a_med = quantile(a_v, 0.50); e_med = quantile(e_arr, 0.50)
    i_med = rad2deg(quantile(inc_v, 0.50)); M_med = quantile(M_J, 0.50)
    M_sig = (quantile(M_J, 0.84) - quantile(M_J, 0.16)) / 2
    # i may rail to the 180-i reflection (relAST orientation ambiguity)
    i_off = min(abs(i_med - REF.i_deg), abs(i_med - (180 - REF.i_deg)))
    ok_a = abs(a_med - REF.a) < 3.0           # short arc → loose
    ok_e = abs(e_med - REF.e) < 0.12
    ok_i = i_off < 10.0
    ok_M = abs(M_med - REF.M_J) / sqrt(M_sig^2 + REF.M_J_sig^2) < 3.0
    verdict = (ok_a && ok_e && ok_i && ok_M) ? "✅ RECOVERED" : "⚠ MISMATCH"
    @sprintf("%-9s %.0fs logZ=%-9.1f a=%.2f/%.2f e=%.3f/%.3f i=%.1f/%.0f M=%.1f/%.1f  a:%s e:%s i:%s M:%s  %s",
             name, secs, logZ, a_med, REF.a, e_med, REF.e, i_med, REF.i_deg,
             M_med, REF.M_J, ok_a ? "✓" : "✗", ok_e ? "✓" : "✗",
             ok_i ? "✓" : "✗", ok_M ? "✓" : "✗", verdict)
end

function run_one(name)
    target = make_target(); data = target.data
    t0 = time()
    try
        local chains, logZ
        if name == "ptemcee"
            res = sample_ptemcee(target, data; n_temps = 16, n_walkers = 150,
                                  n_steps = 12000, n_burnin = 5000,
                                  init_strategy = :prior, show_progress = false)
            chains = res.chains; logZ = res.log_evidence
        elseif name == "pt"
            chains, logZ = sample_pt(target; n_rounds = 13, n_chains = 12,
                                      show_report = false)
        elseif name == "nested"
            chains, logZ = sample_nested(target, data; n_live = 1500, dlogz = 0.5)
        else
            error("unknown sampler $name")
        end
        logline(score_chains(name, chains, logZ, time() - t0))
    catch e
        logline(@sprintf("%-9s %.0fs FAILED: %s", name, time() - t0,
                         sprint(showerror, e)))
    end
end

logline("# HD 4747 menu  (ref Peretti+2019: a=10.01 e=0.736 i=47.3 M=70.0 MJ)")
for name in ["ptemcee", "pt", "nested"]
    println("\n--- $name ---")
    run_one(name)
end
println("\nDone. Verdicts in $RESULTS")
