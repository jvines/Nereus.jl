#!/usr/bin/env julia
# Quadratic photometric-trend recovery (phot_trend_order=2). Injects a transit on
# a CURVED baseline (continuum = 1 + offset + c1·x + c2·x², x normalized) and
# checks that BOTH trend coefficients AND the transit are recovered — i.e. the
# linear and quadratic terms are separately identifiable, not degenerate.
using Nereus, Printf, Random

P0, Tc0, rr0, b0 = 3.0, 1.0, 0.10, 0.30
C1, C2 = 0.02, -0.03                       # true linear + quadratic coeffs
rng = MersenneTwister(8)
tph = sort(collect(range(0.4, 7.6; length = 320)))
σf = 3e-4

tgt0 = build_target(M_s = 1.0, R_s = 1.0, phot_trend_order = 2,
    planets = (b = (P = P0, Tc = Tc0, sesinw = 0.0, secosw = 0.0, b = b0, rr = rr0),),
    phot = (TESS = (data = (t = tph, flux = ones(length(tph)), flux_err = fill(σf, length(tph))),
                    jitter = 1e-4, offset = 0.0, q1 = 0.3, q2 = 0.2),))
th0 = Nereus.Theta{Float64}(tgt0.params)
Nereus.set_param!(th0, "phot_c1_TESS", C1)
Nereus.set_param!(th0, "phot_c2_TESS", C2)
model, _ = Nereus.phot_predictions(th0, tgt0.data)
flux_obs = model .+ σf .* randn(rng, length(tph))
oot = model .> 0.99
@printf("synthetic: %d cadences, curved baseline range %.4f..%.4f, depth %.0f ppm\n",
        length(tph), minimum(model[oot]), maximum(model[oot]), rr0^2 * 1e6)

out = mktempdir()
cfg = Dict(
  "version" => "1.0", "seed" => 1, "output_dir" => out,
  "star" => Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 5800.0),
  "data" => Dict("transit_photometry" => [Dict(
      "values" => Dict("bjd" => tph, "flux" => flux_obs, "flux_err" => fill(σf, length(tph))),
      "instrument" => "TESS", "exposure_time" => 120.0)]),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["PM_ONLY"],
      "parametrization" => Dict("time" => "Tc", "ew" => "sesinw", "geom" => "b_rr"),
      "phot_trend_order" => 2, "stability" => "none"),
  "priors" => Dict(
      "P_k1"  => Dict("type" => "NormalPrior",  "args" => [P0, 0.01, P0 - 0.2, P0 + 0.2]),
      "Tc_k1" => Dict("type" => "NormalPrior",  "args" => [Tc0, 0.02, Tc0 - 0.2, Tc0 + 0.2]),
      "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
      "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "nuts", "kwargs" => Dict(
      "n_chains" => 2, "n_samples" => 400, "n_warmup" => 400, "show_progress" => false)),
  "output" => Dict("plots" => String[], "save_pdf" => false),
)

@printf("\nrun_job PM_ONLY + phot_trend_order=2...\n")
s = Nereus.run_job(cfg)

ok = true
check(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
println("=== quadratic phot trend contract ===")
check("status ok", s["status"] == "ok")
check("fitted has phot_c1_TESS", haskey(fp, "phot_c1_TESS"))
check("fitted has phot_c2_TESS (quadratic term wired)", haskey(fp, "phot_c2_TESS"))

gv(k) = get(get(fp, k, Dict()), "value", NaN)
ge(k) = max(get(get(fp, k, Dict()), "err_lo", NaN), get(get(fp, k, Dict()), "err_hi", NaN), 1e-3)
rr, c1, c2 = gv("rr_k1"), gv("phot_c1_TESS"), gv("phot_c2_TESS")
@printf("\n recovery: rr=%.4f (truth %.4f)  c1=%.4f (truth %.4f)  c2=%.4f (truth %.4f)\n",
        rr, rr0, c1, C1, c2, C2)
check("rr recovered within 0.012", isfinite(rr) && abs(rr - rr0) < 0.012)
check("c1 recovered within 3σ", isfinite(c1) && abs(c1 - C1) < 3 * ge("phot_c1_TESS"))
check("c2 recovered within 3σ", isfinite(c2) && abs(c2 - C2) < 3 * ge("phot_c2_TESS"))

println(ok ? "\n✅ QUADRATIC TREND RECOVERY PASS" : "\n❌ QUADRATIC TREND FAIL")
