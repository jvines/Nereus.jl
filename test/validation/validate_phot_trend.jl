#!/usr/bin/env julia
# Photometric baseline-TREND recovery via run_job. Synthesizes a transit on a
# sloped baseline (continuum = 1 + offset + c1·x, x normalized time) using
# Nereus's own forward model, then fits with phot_trend_order=1 and checks that
# BOTH the transit (rr) and the trend slope (phot_c1) are recovered — i.e. the
# trend is wired into the layout, likelihood, output, and is identifiable.
using Nereus, Printf, Random

P0, Tc0, rr0, b0, C1 = 3.0, 1.0, 0.10, 0.30, 0.02   # C1 = true baseline slope
rng = MersenneTwister(5)
# Cadence over ~2.5 days × 3 transits with out-of-transit baseline to pin the trend.
tph = sort(collect(range(0.4, 7.6; length = 280)))
σf = 3e-4

# Truth target WITH a linear baseline; set the trend coeff to C1 by hand.
tgt0 = build_target(M_s = 1.0, R_s = 1.0, phot_trend_order = 1,
    planets = (b = (P = P0, Tc = Tc0, sesinw = 0.0, secosw = 0.0, b = b0, rr = rr0),),
    phot = (TESS = (data = (t = tph, flux = ones(length(tph)), flux_err = fill(σf, length(tph))),
                    jitter = 1e-4, offset = 0.0, q1 = 0.3, q2 = 0.2),))
th0 = Nereus.Theta{Float64}(tgt0.params)
Nereus.set_param!(th0, "phot_c1_TESS", C1)
model, _ = Nereus.phot_predictions(th0, tgt0.data)
flux_obs = model .+ σf .* randn(rng, length(tph))
@printf("synthetic: %d cadences, baseline drift span %.0f ppm, depth %.0f ppm\n",
        length(tph), (maximum(model) - minimum(model[model .> 0.99])) * 1e6, rr0^2 * 1e6)

out = joinpath(@__DIR__, "..", "..", "results", "phot_trend_check"); mkpath(out)
cfg = Dict(
  "version" => "1.0", "seed" => 1, "output_dir" => out,
  "star" => Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 5800.0),
  "data" => Dict("transit_photometry" => [Dict(
      "values" => Dict("bjd" => tph, "flux" => flux_obs, "flux_err" => fill(σf, length(tph))),
      "instrument" => "TESS", "exposure_time" => 120.0)]),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["PM_ONLY"],
      "parametrization" => Dict("time" => "Tc", "ew" => "sesinw", "geom" => "b_rr"),
      "phot_trend_order" => 1, "stability" => "none"),
  "priors" => Dict(
      "P_k1"  => Dict("type" => "NormalPrior",  "args" => [P0, 0.01, P0 - 0.2, P0 + 0.2]),
      "Tc_k1" => Dict("type" => "NormalPrior",  "args" => [Tc0, 0.02, Tc0 - 0.2, Tc0 + 0.2]),
      "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
      "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "nuts", "kwargs" => Dict(
      "n_chains" => 2, "n_samples" => 400, "n_warmup" => 400, "show_progress" => false)),
  "output" => Dict("plots" => ["pm_timeseries"], "save_pdf" => false),
)

@printf("\nrun_job PM_ONLY + phot_trend_order=1...\n")
s = Nereus.run_job(cfg)

ok = true
check(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
println("=== phot trend contract ===")
check("status ok", s["status"] == "ok")
check("fitted has rr_k1", haskey(fp, "rr_k1"))
check("fitted has phot_c1_TESS (trend param wired)", haskey(fp, "phot_c1_TESS"))
check("phot_c1_TESS dimensionless", get(get(fp, "phot_c1_TESS", Dict()), "unit", "x") == "")

rr = get(get(fp, "rr_k1", Dict()), "value", NaN)
c1 = get(get(fp, "phot_c1_TESS", Dict()), "value", NaN)
c1lo = get(get(fp, "phot_c1_TESS", Dict()), "err_lo", NaN)
c1hi = get(get(fp, "phot_c1_TESS", Dict()), "err_hi", NaN)
@printf("\n recovery: rr=%.4f (truth %.4f)   phot_c1=%.4f -%.4f/+%.4f (truth %.4f)\n",
        rr, rr0, c1, c1lo, c1hi, C1)
check("rr recovered within 0.012", isfinite(rr) && abs(rr - rr0) < 0.012)
check("phot_c1 recovered within 3σ of truth",
      isfinite(c1) && abs(c1 - C1) < 3 * max(c1lo, c1hi, 1e-3))

println(ok ? "\n✅ PHOT TREND RECOVERY PASS" : "\n❌ PHOT TREND FAIL")
