#!/usr/bin/env julia
# Photometry OUTPUT-CONTRACT + recovery test via run_job. Guards the blocker that
# run_job's photometry path crashed on wrong Data kwargs, and exercises the new
# transit observables (T14, depth_ppm, rho_star_transit) + photometric units +
# dilution-at-default (D=0 ⇒ no bias). Synthesizes a transit with Nereus's own
# forward model, then fits it through the documented entry point.
using Nereus, Printf, Random

# --- synthesize a clean transit at known truth via the forward model ----------
P0, Tc0, rr0, b0 = 3.0, 1.0, 0.10, 0.30
rng = MersenneTwister(11)
t = sort(vcat(collect(range(Tc0 - 0.13, Tc0 + 0.13; length = 70)),
              collect(range(Tc0 + P0 - 0.13, Tc0 + P0 + 0.13; length = 70))))
σf = 3e-4
tgt0 = build_target(M_s = 1.0, R_s = 1.0,
    planets = (b = (P = P0, Tc = Tc0, sesinw = 0.0, secosw = 0.0, b = b0, rr = rr0),),
    phot = (TESS = (data = (t = t, flux = ones(length(t)), flux_err = fill(σf, length(t))),
                    jitter = 1e-4, offset = 0.0, q1 = 0.3, q2 = 0.2),))
th0 = Nereus.Theta{Float64}(tgt0.params)
model, _ = Nereus.phot_predictions(th0, tgt0.data)
flux_obs = model .+ σf .* randn(rng, length(t))
@printf("synthetic transit: %d cadences, depth≈%.0f ppm\n",
        length(t), (1 - minimum(model)) * 1e6)

out_dir = mktempdir()
cfg = Dict(
  "version" => "1.0", "seed" => 1, "output_dir" => out_dir,
  "star" => Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 5800.0),
  "data" => Dict("transit_photometry" => [Dict(
      "values" => Dict("bjd" => t, "flux" => flux_obs, "flux_err" => fill(σf, length(t))),
      "instrument" => "TESS", "exposure_time" => 120.0)]),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["PM_ONLY"],
      "parametrization" => Dict("time" => "Tc", "ew" => "sesinw", "geom" => "b_rr"),
      "stability" => "none"),
  "priors" => Dict(
      "P_k1"  => Dict("type" => "NormalPrior",  "args" => [P0, 0.01, P0 - 0.2, P0 + 0.2]),
      "Tc_k1" => Dict("type" => "NormalPrior",  "args" => [Tc0, 0.02, Tc0 - 0.2, Tc0 + 0.2]),
      "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
      "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "nuts", "kwargs" => Dict(
      "n_chains" => 2, "n_samples" => 300, "n_warmup" => 300, "show_progress" => false)),
  "output" => Dict("plots" => ["pm_phasefold", "pm_timeseries"], "save_pdf" => false),
)

@printf("\nrun_job (PM_ONLY transit through the documented entry point)...\n")
s = Nereus.run_job(cfg)

ok = true
check(name, cond) = (global ok; ok &= cond; @printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name))
println("=== photometry output contract ===")
check("status ok (blocker: run_job phot no longer crashes)", s["status"] == "ok")
fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
dp = get(s, "derived", Dict("parameters" => Dict()))["parameters"]
check("fitted has rr_k1", haskey(fp, "rr_k1"))
check("fitted has b_k1",  haskey(fp, "b_k1"))
check("fitted jitter_TESS dimensionless (not m/s)",
      get(get(fp, "jitter_TESS", Dict()), "unit", "m/s") == "")
# new derived observables
check("derived has depth_ppm_k1",        haskey(dp, "depth_ppm_k1"))
check("derived depth_ppm unit = ppm",    get(get(dp, "depth_ppm_k1", Dict()), "unit", "") == "ppm")
check("derived has T14_k1",              haskey(dp, "T14_k1"))
check("derived T14 unit = hr",           get(get(dp, "T14_k1", Dict()), "unit", "") == "hr")
check("derived has rho_star_transit_k1", haskey(dp, "rho_star_transit_k1"))
check("derived rho_star_transit g/cm3",  get(get(dp, "rho_star_transit_k1", Dict()), "unit", "") == "g/cm3")

# recovery (dilution at default D=0 ⇒ unbiased)
rr = get(get(fp, "rr_k1", Dict()), "value", NaN)
bb = get(get(fp, "b_k1",  Dict()), "value", NaN)
d14 = get(get(dp, "T14_k1", Dict()), "value", NaN)
dep = get(get(dp, "depth_ppm_k1", Dict()), "value", NaN)
@printf("\n recovery: rr=%.4f (truth %.4f)  b=%.3f (truth %.3f)  T14=%.2f hr  depth=%.0f ppm\n",
        rr, rr0, bb, b0, d14, dep)
check("rr recovered within 0.01 (no dilution bias)", isfinite(rr) && abs(rr - rr0) < 0.01)

println(ok ? "\n✅ PHOT CONTRACT + RECOVERY PASS" : "\n❌ PHOT CONTRACT INCOMPLETE")
