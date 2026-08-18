# HD 159062 with sample_pt_warm: Pathfinder-seeded multi-init PT.
#
# Same a-driven priors as fit_HD159062_PT.jl, but the PT replicas are
# initialized at Pathfinder draws rather than fresh prior samples.
# Tests whether multi-basin warm-up escapes the short-period local mode
# the prior-init PT settles in.

using Nereus
using MCMCChains
using Statistics: median, std, quantile
using Printf

println("=" ^ 70)
println("HD 159062 — Pathfinder-warm PT")
println("=" ^ 70)
println()

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)
println("Loaded: $(length(rvdat.t)) RV, $(n_relast(relast)) relAST, HGCA HIP 85653")

const M_PRI = 0.81
params = Params(
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
println("Free params: $(n_unfrozen(params))")
target = NereusTarget(params, data)

println("\nRunning sample_pt_warm: 16 Pathfinder runs → 14 rounds × 10 chains")
t0 = time()
chains, log_Z, pf = sample_pt_warm(target;
                                    n_pathfinder_runs = 16,
                                    n_rounds = 14,
                                    n_chains = 10,
                                    seed = 42,
                                    show_report = false)
println(@sprintf("Done in %.1f min, log Z = %.2f", (time()-t0)/60, log_Z))
# Fixed-dim :auto now routes to prior-init ptemcee (no Pathfinder), so pf is
# nothing; the :pigeons/:nereus opt-in paths still return a Pathfinder result.
if pf === nothing
    println("Backend: prior-init ptemcee (no Pathfinder warm-start)")
else
    println(@sprintf("Pathfinder Pareto k = %.2f (>0.7 = approximation poor)",
                     pf.psis_result.pareto_shape))
end
println()

a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
Mp_v  = vec(Array(chains[:, :M_pri, :]))
ses_v = vec(Array(chains[:, :sesinw_k1, :]))
sec_v = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
n = length(a_v)
e_arr = ses_v.^2 .+ sec_v.^2
P_yr  = [sqrt(a_v[j]^3 / (Mp_v[j] + M_sec[j])) for j in 1:n]

println("Posteriors:")
for (k, v) in pairs((a_au=a_v, P_yr=P_yr, M_pri=Mp_v, M_sec=M_sec,
                     e=e_arr, i_deg=rad2deg.(inc_v)))
    q16, q50, q84 = quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
    @printf("  %-10s = %12.4f  [+%.4f, -%.4f]\n", string(k), q50, q84-q50, q50-q16)
end
println()

# Reference = orvara chain (HD159062B tutorial; matches Brandt+ 2021):
#   a = 58 AU, P ≈ 354 yr, e = 0.11, i = 62°, M_sec = 0.608 M_sun.
# NOTE: validate the ORBIT (a, e, i), not just M_sec — the RV+HGCA mass
# function pins M_sec even when the orbit is wrong, so an M_sec-only check
# is a false positive (this is exactly how the gamma-prior bug hid).
const REF = (a=58.0, e=0.11, i_deg=62.0, M_sec=0.608, M_sec_sig=0.008)
a_med, e_med    = quantile(a_v, 0.50), quantile(e_arr, 0.50)
i_med, M_med    = rad2deg(quantile(inc_v, 0.50)), quantile(M_sec, 0.50)
M_sig           = (quantile(M_sec, 0.84) - quantile(M_sec, 0.16)) / 2
@printf("\nReference (orvara/Brandt): a=%.0f AU  e=%.2f  i=%.0f°  M_sec=%.4f M_sun\n",
        REF.a, REF.e, REF.i_deg, REF.M_sec)
@printf("Nereus median:            a=%.1f AU  e=%.3f  i=%.1f°  M_sec=%.4f M_sun\n",
        a_med, e_med, i_med, M_med)
ok_a   = abs(a_med - REF.a)       < 15.0
ok_e   = abs(e_med - REF.e)       < 0.15
ok_i   = abs(i_med - REF.i_deg)   < 8.0
ok_M   = abs(M_med - REF.M_sec) / sqrt(M_sig^2 + REF.M_sec_sig^2) < 3.0
@printf("  a:%s  e:%s  i:%s  M_sec:%s\n",
        ok_a ? "✓" : "✗", ok_e ? "✓" : "✗", ok_i ? "✓" : "✗", ok_M ? "✓" : "✗")
println((ok_a && ok_e && ok_i && ok_M) ?
        "✅ VALIDATED — orbit (a,e,i) AND mass match orvara/Brandt" :
        "⚠ NOT validated — orbit and/or mass disagree with orvara/Brandt")
