# Real-target recovery menu — HD 159062 (RV + HGCA + relAST, a-driven RVAS).
#
# Runs the workhorse sampler menu on the SAME model + SAME orbit scorer as
# fit_HD159062_warm.jl and checks each against the orvara/Brandt ground truth.
# This is the astrometry leg of the real-target recovery menu (after 51 Peg /
# GJ 876 RV legs). Validates the ORBIT (a, e, i), not just M_sec — the RV+HGCA
# mass function pins M_sec even when the orbit is wrong (the gamma-prior bug).
#
# Each sampler's verdict is appended to RESULTS as it finishes, so a slow
# nested run timing out still leaves the ptemcee/pt verdicts on disk.

using Nereus
using MCMCChains
using Statistics: median, std, quantile
using Printf

const RESULTS = joinpath(@__DIR__, "menu_HD159062_results.txt")
logline(s) = (open(RESULTS, "a") do io; println(io, s); end; println(s); flush(stdout))

println("=" ^ 70); println("HD 159062 — sampler recovery menu"); println("=" ^ 70)

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)
println("Loaded: $(length(rvdat.t)) RV, $(n_relast(relast)) relAST, HGCA HIP 85653")

const M_PRI = 0.81
make_params() = Params(
    max_kplanet = 1, planet_modes = [RVAS],
    instruments = InstrumentConfig(rv = ["HIRES"]),
    data = data, stability = :none, M_s = M_PRI,
    parametrization = ParametrizationConfig(mass = :a_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"        => FixedPrior(1.0),
        "a_k1"       => LogUniformPrior(5.0, 1500.0),
        "M_sec_k1"   => LogUniformPrior(0.05, 1.5),
        "sesinw_k1"  => UniformPrior(-1.0, 1.0),
        "secosw_k1"  => UniformPrior(-1.0, 1.0),
        "Mo_k1"      => UniformPrior(0.0, 2π),
        "inc_k1"     => SinePrior(),
        "Omega_k1"   => UniformPrior(0.0, 2π),
        "sigma_HIRES"=> LogUniformPrior(1e-3, 1e3),
        "M_pri"      => NormalPrior(0.81, 0.04, 0.5, 1.2),
    ),
)

# orvara chain (HD159062B tutorial; matches Brandt+ 2021)
const REF = (a=58.0, e=0.11, i_deg=62.0, M_sec=0.608, M_sec_sig=0.008)

# Score a chains object against the orvara orbit. Returns a one-line verdict.
function score_chains(name, chains, logZ, secs)
    a_v   = vec(Array(chains[:, :a_k1, :]))
    M_sec = vec(Array(chains[:, :M_sec_k1, :]))
    ses_v = vec(Array(chains[:, :sesinw_k1, :]))
    sec_v = vec(Array(chains[:, :secosw_k1, :]))
    inc_v = vec(Array(chains[:, :inc_k1, :]))
    e_arr = ses_v.^2 .+ sec_v.^2
    a_med = quantile(a_v, 0.50); e_med = quantile(e_arr, 0.50)
    i_med = rad2deg(quantile(inc_v, 0.50)); M_med = quantile(M_sec, 0.50)
    M_sig = (quantile(M_sec, 0.84) - quantile(M_sec, 0.16)) / 2
    ok_a = abs(a_med - REF.a) < 15.0
    ok_e = abs(e_med - REF.e) < 0.15
    ok_i = abs(i_med - REF.i_deg) < 8.0
    ok_M = abs(M_med - REF.M_sec) / sqrt(M_sig^2 + REF.M_sec_sig^2) < 3.0
    verdict = (ok_a && ok_e && ok_i && ok_M) ? "✅ RECOVERED" : "⚠ MISMATCH"
    @sprintf("%-9s %.0fs logZ=%-9.1f a=%.1f/%.0f e=%.3f/%.2f i=%.1f/%.0f M=%.4f/%.4f  a:%s e:%s i:%s M:%s  %s",
             name, secs, logZ, a_med, REF.a, e_med, REF.e, i_med, REF.i_deg,
             M_med, REF.M_sec, ok_a ? "✓" : "✗", ok_e ? "✓" : "✗",
             ok_i ? "✓" : "✗", ok_M ? "✓" : "✗", verdict)
end

function run_one(name)
    params = make_params(); target = NereusTarget(params, data)
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

logline("# HD 159062 menu  (ref orvara/Brandt: a=58 e=0.11 i=62 M_sec=0.608)")
for name in ["ptemcee", "pt", "nested"]
    println("\n--- $name ---")
    run_one(name)
end
println("\nDone. Verdicts in $RESULTS")
