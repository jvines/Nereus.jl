#!/usr/bin/env julia
# JOINT RV+transit (RVPM) through run_job — the headline planet-fitting case,
# untested until now. Guards that the blocker fix + the new transit-model changes
# play correctly with the combined RV+phot likelihood and that the contract carries
# both RV-derived (msini) and transit-derived (rr, T14, rho_star_transit) science.
using Nereus, Printf, Random

rng = MersenneTwister(3)
P0, Tc0, K0, rr0, b0 = 3.0, 1.0, 12.0, 0.10, 0.30
trv = sort(rand(rng, 32) .* 40.0)
tph = sort(vcat(collect(range(Tc0 - 0.12, Tc0 + 0.12; length = 60)),
                collect(range(Tc0 + P0 - 0.12, Tc0 + P0 + 0.12; length = 60))))
σrv, σf = 2.0, 3e-4

tgt0 = build_target(M_s = 1.0, R_s = 1.0,
    planets = (b = (P = P0, Tc = Tc0, K = K0, sesinw = 0.0, secosw = 0.0, b = b0, rr = rr0),),
    rv = (HARPS = (data = (t = trv, rv = zeros(length(trv)), rv_err = fill(σrv, length(trv))),
                   gamma = 0.0, sigma = 1.0),),
    phot = (TESS = (data = (t = tph, flux = ones(length(tph)), flux_err = fill(σf, length(tph))),
                    jitter = 1e-4, offset = 0.0, q1 = 0.3, q2 = 0.2),))
th0 = Nereus.Theta{Float64}(tgt0.params)
rvp, _ = Nereus.rv_predictions(th0, tgt0.data)
phf, _ = Nereus.phot_predictions(th0, tgt0.data)
rv_obs = rvp .+ σrv .* randn(rng, length(trv))
fl_obs = phf .+ σf .* randn(rng, length(tph))
@printf("RVPM synthetic: %d RV + %d phot cadences\n", length(trv), length(tph))

out = mktempdir()
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
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tc", "ew" => "sesinw", "geom" => "b_rr"),
     "stability" => "none"),
  "priors" => Dict(
     "P_k1"  => Dict("type" => "NormalPrior",  "args" => [P0, 0.01, P0 - 0.2, P0 + 0.2]),
     "Tc_k1" => Dict("type" => "NormalPrior",  "args" => [Tc0, 0.02, Tc0 - 0.2, Tc0 + 0.2]),
     "K_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 40.0]),
     "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "nuts", "kwargs" => Dict(
      "n_chains" => 2, "n_samples" => 350, "n_warmup" => 350, "show_progress" => false)),
  "output" => Dict("plots" => ["rv_phasefold", "pm_phasefold"], "save_pdf" => false),
)

@printf("\nrun_job RVPM (joint RV + transit)...\n")
s = Nereus.run_job(cfg)

ok = true
check(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
dp = get(s, "derived", Dict("parameters" => Dict()))["parameters"]
println("=== RVPM joint contract ===")
check("status ok", s["status"] == "ok")
check("fitted has K_k1 (RV)",  haskey(fp, "K_k1"))
check("fitted K_k1 unit m/s",  get(get(fp, "K_k1", Dict()), "unit", "") == "m/s")
check("fitted has rr_k1 (transit)", haskey(fp, "rr_k1"))
check("fitted jitter_TESS dimensionless", get(get(fp, "jitter_TESS", Dict()), "unit", "x") == "")
check("derived has msini_earth_k1 (RV)",  haskey(dp, "msini_earth_k1"))
check("derived has T14_k1 (transit)",     haskey(dp, "T14_k1"))
check("derived has rho_star_transit_k1",  haskey(dp, "rho_star_transit_k1"))

K  = get(get(fp, "K_k1", Dict()), "value", NaN)
rr = get(get(fp, "rr_k1", Dict()), "value", NaN)
@printf("\n recovery: K=%.2f m/s (truth %.1f)  rr=%.4f (truth %.4f)\n", K, K0, rr, rr0)
check("K recovered within 3 m/s",  isfinite(K) && abs(K - K0) < 3.0)
check("rr recovered within 0.012", isfinite(rr) && abs(rr - rr0) < 0.012)

println(ok ? "\n✅ RVPM JOINT CONTRACT + RECOVERY PASS" : "\n❌ RVPM JOINT FAIL")
