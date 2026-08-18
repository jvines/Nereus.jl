#!/usr/bin/env julia
# Fit-output PLOT validation. Drive a synthetic RV+transit fit through run_job
# (the production dispatcher) requesting the full science + diagnostic plot
# suite, and assert: (a) the injected orbit is recovered, and (b) every
# requested plot is actually emitted (path exists, non-empty). The PNGs are
# then eyeballed — this is the visual half of "validate everything". One fit
# exercises rv_phasefold / rv_timeseries / ppc / trace / corner / posteriors.
using Nereus, Printf, Random

out = joinpath(@__DIR__, "..", "..", "results", "runjob_plot_check")
rm(out; force = true, recursive = true); mkpath(out)
rng = MersenneTwister(3)

# --- synthetic RV + transit, one planet, known truth ------------------------
# e≠0 (e=0.2, ω=1.0 rad): at e=0 sesinw/secosw are DEGENERATE (ω undefined), so
# NUTS can't mix them and the whole chain fails to converge (and the predictive
# CI band goes degenerate). Injecting a real eccentricity makes e/ω identifiable
# → the fit converges. ([[e=0 ω degeneracy]])
P0, Tc0, K0, rr0, b0 = 3.0, 1.0, 12.0, 0.10, 0.30
sesinw0, secosw0 = 0.3763, 0.2416        # e = 0.20, ω = 1.0 rad
jit0 = 2.0                               # real RV jitter (keeps sigma off the funnel)
trv = sort(rand(rng, 50) .* 40.0)
tph = sort(vcat(collect(range(Tc0 - 0.12, Tc0 + 0.12; length = 60)),
                collect(range(Tc0 + P0 - 0.12, Tc0 + P0 + 0.12; length = 60))))
σrv, σf = 2.0, 3e-4
tgt0 = build_target(M_s = 1.0, R_s = 1.0,
    planets = (b = (P = P0, Tc = Tc0, K = K0, sesinw = sesinw0, secosw = secosw0, b = b0, rr = rr0),),
    rv = (HARPS = (data = (t = trv, rv = zeros(length(trv)), rv_err = fill(σrv, length(trv))),
                   gamma = 0.0, sigma = 1.0),),
    phot = (TESS = (data = (t = tph, flux = ones(length(tph)), flux_err = fill(σf, length(tph))),
                    jitter = 1e-4, offset = 0.0, q1 = 0.3, q2 = 0.2),))
th0 = Nereus.Theta{Float64}(tgt0.params)
rvp, _ = Nereus.rv_predictions(th0, tgt0.data)
phf, _ = Nereus.phot_predictions(th0, tgt0.data)
rv_obs = rvp .+ σrv .* randn(rng, length(trv)) .+ jit0 .* randn(rng, length(trv))
fl_obs = phf .+ σf .* randn(rng, length(tph))
@printf("synthetic: %d RV + %d phot cadences | inject P=%.2f K=%.1f rr=%.2f\n",
        length(trv), length(tph), P0, K0, rr0)

requested = ["rv_phasefold", "rv_timeseries", "pm_phasefold", "trace", "corner", "posteriors_parameters"]
cfg = Dict(
  "version" => "1.0", "seed" => 1, "output_dir" => out,
  "star" => Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 5800.0),
  "data" => Dict(
     "rv" => Dict("values" => Dict("bjd" => trv, "rv" => rv_obs,
                  "rv_err" => fill(σrv, length(trv)), "instrument" => fill("HARPS", length(trv)))),
     "transit_photometry" => [Dict(
         "values" => Dict("bjd" => tph, "flux" => fl_obs, "flux_err" => fill(σf, length(tph))),
         "instrument" => "TESS", "exposure_time" => 120.0)]),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RVPM"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tc", "ew" => "sesinw",
                               "geom" => "b_rr", "marginalize_gamma" => true),
     "stability" => "none"),
  "priors" => Dict(
     "P_k1"  => Dict("type" => "NormalPrior",  "args" => [P0, 0.01, P0 - 0.2, P0 + 0.2]),
     "Tc_k1" => Dict("type" => "NormalPrior",  "args" => [Tc0, 0.02, Tc0 - 0.2, Tc0 + 0.2]),
     "K_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 40.0]),
     "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  # ptemcee (ensemble PT) — robust to the correlated K/e/b/LD geometry that
  # NUTS couldn't mix (Rhat 2.3-2.7 even after γ-marg + jitter + target_accept).
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
      "n_temps" => 8, "n_walkers" => 60, "n_steps" => 3500, "n_burnin" => 2000,
      "show_progress" => false)),
  "output" => Dict("plots" => requested, "save_pdf" => false),
)

@printf("\nrun_job (synthetic RV+transit, nuts)...\n")
s = Nereus.run_job(cfg)

ok = true
chk(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))

# (a) recovery
fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
K  = get(get(fp, "K_k1", Dict()), "value", NaN)
rr = get(get(fp, "rr_k1", Dict()), "value", NaN)
@printf("\nrecovery: K=%.2f m/s (truth %.1f) | rr=%.4f (truth %.4f)\n", K, K0, rr, rr0)
chk("status ok", s["status"] == "ok")
chk("K recovered within 3 m/s",  isfinite(K) && abs(K - K0) < 3.0)
chk("rr recovered within 0.012", isfinite(rr) && abs(rr - rr0) < 0.012)

# (c) the fit actually CONVERGED — assess_fit on the saved chains (a tight K
# median can hide unconverged/multimodal chains; gate on convergence directly).
ch, _ = Nereus.load_chains(joinpath(out, "chains.nc"))   # now a real cube (iter,param,walker)
hr = assess_fit(ch)   # walkers are already the chain axis → standard multi-chain Rhat
cstat(name) = (i = findfirst(c -> c.name === name, hr.checks); i === nothing ? :absent : hr.checks[i].status)
@printf("\nconvergence: assess_fit overall=%s | convergence=%s | multimodality=%s\n",
        hr.overall, cstat(:convergence), cstat(:multimodality))
chk("fit converged (assess_fit :convergence ok)", cstat(:convergence) === :ok)
chk("fit not flagged multimodal",                 cstat(:multimodality) !== :fail)

# (b) the suite is emitted in run_job's real layout (named by content)
exists_nonempty(rel) = (f = joinpath(out, rel); isfile(f) && filesize(f) > 1000)
count_png(reldir) = (d = joinpath(out, reldir); isdir(d) ?
    sum(endswith(f, ".png") for (r, _, fs) in walkdir(d) for f in fs; init = 0) : 0)
println("\n=== emitted plot suite ===")
for (label, rel) in [
        ("RV phase-fold",  "plots/models/RV_phasefold_P1.png"),
        ("RV timeseries",  "plots/models/rv_timeseries.png"),
        ("transit fold",   "plots/models/Transit_phasefold_P1_TESS.png"),
        ("corner",         "plots/corner.png"),
        ("PPC",            "ppc.png")]
    @printf("  %-16s %s\n", label, exists_nonempty(rel) ? "✓" : "MISSING")
    chk("$label emitted", exists_nonempty(rel))
end
ntr, npost = count_png("plots/traces"), count_png("plots/posteriors")
@printf("  traces=%d  posterior-param plots=%d\n", ntr, npost)
chk("trace per parameter (≥10)",            ntr >= 10)
chk("posterior parameter plots (≥10)",      npost >= 10)
# science tables — the output contract, in every advertised format
for fmt in ("csv", "json", "tex", "ecsv", "dat")
    chk("fitted+derived tables .$fmt", exists_nonempty("tables/fitted.$fmt") &&
                                       exists_nonempty("tables/derived.$fmt"))
end

println(ok ? "\n✅ RUN_JOB PLOT-SUITE VALIDATION PASS" : "\n❌ RUN_JOB PLOT-SUITE — FAILURES ABOVE")
@printf("plots dir: %s\n", joinpath(out, "plots"))
