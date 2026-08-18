# Sampler head-to-head on ONE common problem: the Gaia-4 b orbit from DR4
# epoch astrometry alone (8 params, fixed-dim, differentiable). Runs Pigeons PT,
# in-house ptemcee, nested sampling, and PT-HMC (NUTS) on the SAME target and
# compares recovery (P, M, e, i), log-evidence, and wall-clock.
#
# Truth (Stefansson+ 2024): P=571.3 d, M=11.8 M_J, e=0.338, i=116.9°.
# Not iso-compute — each sampler uses sensible moderate settings; wall-clock is
# reported alongside so the trade-off is explicit.
#
#   env NEREUS_GAIA_DR4_XML=<path>   reuse a local VOTable

using Nereus, MCMCChains, Printf
using Statistics: median, quantile

const SID = 1457486023639239296
const M_PRI = 0.644
const TRUTH = (P = 571.3, M = 11.8, e = 0.338, i = 116.9)

xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, SID)

build() = build_target(
    M_pri = M_PRI,
    planets = (b = (
        a      = LogUniformPrior(0.3, 4.0),
        M_sec  = LogUniformPrior(0.001, 0.05),
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        Mo     = UniformPrior(0.0, 2π),
        inc    = SinePrior(),
        Omega  = UniformPrior(0.0, 2π),
    ),),
    iad = src.iad,
    plx = NormalPrior(13.628, 0.021),
    M_s = M_PRI,
)

# chains → (P_days, M_J, e, i_deg) posterior medians
function summarize(chains)
    a   = vec(Array(chains[:, :a_k1, :]))
    Ms  = vec(Array(chains[:, :M_sec_k1, :]))
    ses = vec(Array(chains[:, :sesinw_k1, :]))
    sec = vec(Array(chains[:, :secosw_k1, :]))
    inc = vec(Array(chains[:, :inc_k1, :]))
    P   = [365.25 * sqrt(a[j]^3 / (M_PRI + Ms[j])) for j in eachindex(a)]
    (P = median(P), M = median(Ms .* 1047.57),
     e = median(ses .^ 2 .+ sec .^ 2), i = median(rad2deg.(inc)),
     nP = length(P))
end

results = Tuple{String, Any, Float64, Float64, Int}[]   # (name, summary|err, logZ, secs, nsamp)

function run!(name, f)
    @printf(">>> %-10s ... ", name); flush(stdout)
    t0 = time()
    try
        chains, logZ = f()
        s = summarize(chains); dt = time() - t0
        @printf("done %.1f min  logZ=%.1f\n", dt / 60, logZ)
        push!(results, (name, s, logZ, dt, s.nP))
    catch err
        dt = time() - t0
        @printf("FAILED (%s)\n", sprint(showerror, err))
        push!(results, (name, err, NaN, dt, 0))
    end
end

run!("Pigeons", () -> sample_pt(build(); n_rounds = 10, n_chains = 8, seed = 42,
                                show_report = false))
run!("ptemcee", () -> let tgt = build()
        r = sample_ptemcee(tgt, tgt.data; n_temps = 8, n_walkers = 100,
                           n_steps = 2500, n_burnin = 1000, seed = 42)
        (r.chains, r.log_evidence)
    end)
run!("nested", () -> let tgt = build()
        sample_nested(tgt, tgt.data; n_live = 800, bounds = :multi,
                      proposal = :rslice, seed = 42)
    end)
run!("PT-HMC", () -> let (c, lz, _) = sample_pt_hmc(build(); n_temps = 10,
                                                    n_sweeps = 800, n_warmup = 300, seed = 42)
        (c, lz)
    end)

# --- comparison table --------------------------------------------------------
println("\n", "="^78)
@printf("%-9s %8s %8s %7s %8s %9s %7s %7s\n",
        "sampler", "P_d", "M_J", "e", "i_deg", "logZ", "min", "Nsamp")
@printf("%-9s %8.1f %8.1f %7.3f %8.1f %9s %7s %7s\n",
        "TRUTH", TRUTH.P, TRUTH.M, TRUTH.e, TRUTH.i, "—", "—", "—")
println("-"^78)
for (name, s, logZ, dt, n) in results
    if s isa NamedTuple
        @printf("%-9s %8.1f %8.1f %7.3f %8.1f %9.1f %7.1f %7d\n",
                name, s.P, s.M, s.e, s.i, logZ, dt / 60, n)
    else
        @printf("%-9s   FAILED: %s\n", name, sprint(showerror, s))
    end
end
println("="^78)
println("(recovery = do P/M/e/i match TRUTH; logZ comparable only within an estimator family;")
println(" wall-clock is NOT iso-compute — see per-sampler settings in the script)")
