#!/usr/bin/env julia
# End-to-end validation of the planet-fitting OUTPUT CONTRACT via run_job:
# a fast synthetic 1-planet RV fit, then assert the returned summary carries
# the full deliverable — conditioned fitted/derived (units + asymmetric err),
# model_selection, run_info (R̂/ESS, priors, provenance, git), the multi-format
# tables manifest, and the figures manifest. This is the exoautomata API shape.
using Nereus, Printf, Random

rng = MersenneTwister(7); n = 40
t   = sort(rand(rng, n) .* 60.0)
rv  = 9.0 .* sin.(2π .* t ./ 7.3 .+ 0.6) .+ 1.2 .* randn(rng, n)
out_dir = mktempdir()

cfg = Dict(
  "version" => "1.0", "seed" => 1, "output_dir" => out_dir,
  "star" => Dict("M_s" => 0.9, "R_s" => 0.88, "T_eff" => 5200.0),
  "data" => Dict("rv" => Dict("values" => Dict(
      "bjd" => t, "rv" => rv, "rv_err" => fill(1.2, n),
      "instrument" => fill("SIM", n)))),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"],
      "parametrization" => Dict("time" => "Mo", "ew" => "sesinw"),
      "stability" => "none"),
  "priors" => Dict("P_k1" => Dict("type" => "LogUniformPrior", "args" => [2.0, 30.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
      "n_temps" => 8, "n_walkers" => 60, "n_steps" => 1500, "n_burnin" => 600,
      "show_progress" => false)),
  "output" => Dict("plots" => ["rv_timeseries", "rv_phasefold",
      "posteriors_histograms", "traces_grouped"], "save_pdf" => false),
)

s = Nereus.run_job(cfg)

ok = true
check(name, cond) = (global ok; ok &= cond; @printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name))
println("=== output contract ===")
check("status ok", s["status"] == "ok")
for blk in ("fitted", "derived", "model_selection", "run_info", "tables", "figures")
    check("has $blk", haskey(s, blk))
end
check("fitted has K_k1", haskey(s["fitted"]["parameters"], "K_k1"))
check("fitted K_k1 has unit m/s", get(get(s["fitted"]["parameters"], "K_k1", Dict()), "unit", "") == "m/s")
check("fitted K_k1 asymmetric err", haskey(s["fitted"]["parameters"]["K_k1"], "err_lo") &&
                                     haskey(s["fitted"]["parameters"]["K_k1"], "err_hi"))
check("fitted K_k1 3σ CI", length(get(s["fitted"]["parameters"]["K_k1"], "ci3", [])) == 2)
check("fitted conditioning label", haskey(s["fitted"]["conditioning"], "n_planets"))
check("derived has ecc_k1", haskey(s["derived"]["parameters"], "ecc_k1"))
check("derived has msini_earth_k1", haskey(s["derived"]["parameters"], "msini_earth_k1"))
check("run_info stellar propagated", haskey(s["run_info"], "stellar"))
check("run_info convergence", haskey(s["run_info"], "convergence"))
check("run_info priors", haskey(s["run_info"], "priors"))
check("run_info provenance", haskey(s["run_info"]["data_provenance"], "rv"))
check("run_info git_hash", haskey(s["run_info"], "git_hash"))
check("tables fitted 5 formats", length(s["tables"]["fitted"]) == 5)
check("figures populated", !isempty(s["figures"]))
check("figures has tree-relative keys", any(k -> occursin("/", k), keys(s["figures"])))
check("figures has a traces group", any(k -> startswith(k, "traces/"), keys(s["figures"])))
check("figures has posteriors histograms", any(k -> startswith(k, "posteriors/histograms/"), keys(s["figures"])))
check("legacy unconditioned params dropped", !haskey(s, "params"))

K = s["fitted"]["parameters"]["K_k1"]
@printf("\nK_k1 = %.3f -%.3f/+%.3f m/s  (injected 9.0)\n", K["value"], K["err_lo"], K["err_hi"])
@printf("ecc_k1 = %.3f\n", s["derived"]["parameters"]["ecc_k1"]["value"])
@printf("tables.fitted: %s\n", sort(collect(keys(s["tables"]["fitted"]))))
@printf("figures: %s\n", sort(collect(keys(s["figures"]))))
@printf("convergence: %s\n", s["run_info"]["convergence"])
println(ok ? "\n✅ OUTPUT CONTRACT VALIDATED" : "\n❌ CONTRACT INCOMPLETE")
