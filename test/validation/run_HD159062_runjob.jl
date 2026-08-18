#!/usr/bin/env julia
# HD 159062 (RV + relative astrometry + HGCA) END-TO-END through run_job — the
# real-data RVAS merge-path check. Loads the orvara-format data with the orvara
# loaders, passes everything INLINE to run_job, fits with pt_warm (the canonical
# recipe that escapes the short-period local mode), renders the orbit plots, and
# compares the recovered orbit to the orvara/Brandt ground truth.
using Nereus, Printf, Statistics

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
@printf("Loaded HD159062: %d RV, %d relAST epochs, HGCA HIP 85653\n",
        length(rvdat.t), length(relast.t))

out = joinpath(@__DIR__, "..", "..", "results", "HD159062_runjob"); mkpath(out)
cfg = Dict(
  "version" => "1.0", "seed" => 42, "output_dir" => out,
  "star" => Dict("M_s" => 0.81),
  "data" => Dict(
     "rv" => Dict("values" => Dict(
        "bjd" => rvdat.t, "rv" => rvdat.rv, "rv_err" => rvdat.rv_err,
        "instrument" => fill("HIRES", length(rvdat.t)))),
     "relastrom" => Dict("values" => Dict(
        "t" => relast.t, "ra_off" => relast.ra_off, "dec_off" => relast.dec_off,
        "ra_err" => relast.ra_err, "dec_err" => relast.dec_err)),
     "hgca" => Dict("values" => Dict(
        "t" => collect(hgca.epochs), "pmra" => collect(hgca.pmra),
        "pmdec" => collect(hgca.pmdec),
        "cov_ep" => [hgca.cov_ep[1], hgca.cov_ep[2], hgca.cov_ep[3]],
        "plx" => hgca.plx, "plx_err" => hgca.plx_err, "hip_id" => hgca.hip_id))),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RVAS"],
     "parametrization" => Dict("mass" => "a_driven", "time" => "Mo", "ew" => "sesinw"),
     "stability" => "none"),
  "priors" => Dict(
     "a_k1"     => Dict("type" => "LogUniformPrior", "args" => [5.0, 1500.0]),
     "M_sec_k1" => Dict("type" => "LogUniformPrior", "args" => [0.05, 1.5]),
     "M_pri"    => Dict("type" => "NormalPrior",     "args" => [0.81, 0.04, 0.5, 1.2])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
     "n_temps" => 6, "n_walkers" => 30, "n_steps" => 800, "n_burnin" => 400,
     "show_progress" => false)),
  "output" => Dict("plots" => ["rv_timeseries", "rv_phasefold", "orbit_skyplane",
                               "relastrom_residuals", "rv_astrom_phasefold"],
                   "save_pdf" => false),
)

@printf("\nrun_job RVAS (real HD159062, pt_warm)...\n"); t0 = time()
s = Nereus.run_job(cfg)
@printf("done %.1f min  status=%s\n", (time() - t0) / 60, s["status"])

# --- recovery vs orvara/Brandt ground truth -----------------------------------
fp = get(s, "fitted",  Dict("parameters" => Dict()))["parameters"]
dp = get(s, "derived", Dict("parameters" => Dict()))["parameters"]
gv(k, b) = get(get(b, k, Dict()), "value", NaN)
a    = gv("a_k1", fp)
Msec = gv("M_sec_k1", fp)
ecc  = gv("ecc_k1", dp)
inc  = gv("inc_k1", fp)               # radians
@printf("\n=== HD159062 recovery (orvara/Brandt: a=58 AU, e=0.11, i=62°, M_sec=0.608 M_sun) ===\n")
@printf("  a_k1      = %8.2f AU\n", a)
@printf("  ecc_k1    = %8.3f\n", ecc)
@printf("  inc_k1    = %8.1f°\n", rad2deg(inc))
@printf("  M_sec_k1  = %8.4f M_sun\n", Msec)
oka = abs(a - 58) < 15; oke = abs(ecc - 0.11) < 0.15
oki = abs(rad2deg(inc) - 62) < 10 || abs(180 - rad2deg(inc) - 62) < 10
okm = abs(Msec - 0.608) < 0.10
@printf("  match: a:%s e:%s i:%s M_sec:%s\n", oka ? "✓" : "✗", oke ? "✓" : "✗",
        oki ? "✓" : "✗", okm ? "✓" : "✗")
println((oka && oke && oki && okm) ? "\n✅ HD159062 RVAS via run_job RECOVERED" :
        "\n⚠ orbit disagrees with orvara (sampler may have stuck in the short-period mode)")
@printf("\nfigures: %s\n", sort(collect(keys(get(s, "figures", Dict())))))
@printf("plots dir: %s\n", joinpath(out, "plots"))
