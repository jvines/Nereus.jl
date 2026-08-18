#!/usr/bin/env julia
# Rossiter-McLaughlin plot group (rm_anomaly) through run_job. A transiting hot
# Jupiter on a MISALIGNED orbit (inject v·sin i_* and λ): RV is sampled densely
# IN-TRANSIT (a spectroscopic transit) carrying the RM anomaly, plus out-of-transit
# RV for the orbit, plus photometry pinning the transit geometry. Fit RVPM_RM and
# assert: v·sin i_* and λ are recovered, and the rm_anomaly plot emits. The truth
# RV (Keplerian + RM) is built via rv_predictions, which now includes the RM term.
using Nereus, Printf, Random, Statistics

out = joinpath(@__DIR__, "..", "..", "results", "runjob_rm_check")
const REPLOT = get(ENV, "RM_REPLOT", "0") == "1"
REPLOT || (rm(out; force = true, recursive = true); mkpath(out))
rng = MersenneTwister(73)

# --- truth: misaligned hot Jupiter ------------------------------------------
P0, Tc0, K0, rr0, b0 = 3.5, 1.0, 50.0, 0.12, 0.30
sesinw0, secosw0 = 0.30, 0.18                  # e≈0.12
vsini0, lam0 = 6000.0, deg2rad(35.0)           # 6 km/s, λ=35°
σrv, σf = 6.0, 3e-4

# RV: 3 densely-sampled spectroscopic transits + out-of-transit orbit coverage
tdur = 0.12                                     # ~transit duration (d)
trv = Float64[]
for n in 0:2
    c = Tc0 + n * P0
    append!(trv, collect(range(c - 0.75 * tdur, c + 0.75 * tdur; length = 22)))   # in-transit
end
append!(trv, sort(rand(rng, 22) .* (3 * P0)))   # out-of-transit orbit anchors
sort!(trv)
# Photometry across each transit to pin rr/b
tph = Float64[]
for n in 0:2
    c = Tc0 + n * P0
    append!(tph, collect(range(c - 0.13, c + 0.13; length = 45)))
end
sort!(tph)

# Build params/data from a placeholder cfg, set truth, forward-model (incl. RM).
cfg = Dict(
  "version" => "1.0", "seed" => 1, "output_dir" => out,
  "star" => Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 6200.0),
  "data" => Dict(
     "rv" => Dict("values" => Dict("bjd" => trv, "rv" => zeros(length(trv)),
                  "rv_err" => fill(σrv, length(trv)), "instrument" => fill("HARPS", length(trv)))),
     "transit_photometry" => [Dict(
         "values" => Dict("bjd" => tph, "flux" => ones(length(tph)), "flux_err" => fill(σf, length(tph))),
         "instrument" => "TESS", "exposure_time" => 120.0)]),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RVPM_RM"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tc", "ew" => "sesinw", "geom" => "b_rr"),
     "stability" => "none"),
  "priors" => Dict(
     "P_k1"           => Dict("type" => "NormalPrior",  "args" => [P0, 0.01, P0 - 0.1, P0 + 0.1]),
     "Tc_k1"          => Dict("type" => "NormalPrior",  "args" => [Tc0, 0.01, Tc0 - 0.1, Tc0 + 0.1]),
     "K_k1"           => Dict("type" => "UniformPrior", "args" => [0.0, 120.0]),
     "rr_k1"          => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "b_k1"           => Dict("type" => "UniformPrior", "args" => [0.0, 0.9]),
     "v_sin_i_star"   => Dict("type" => "NormalPrior",  "args" => [vsini0, 1500.0, 500.0, 20000.0]),
     "lambda_k1"      => Dict("type" => "UniformPrior", "args" => [-π, π])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
      "n_temps" => 8, "n_walkers" => 70, "n_steps" => 3000, "n_burnin" => 1800,
      "show_progress" => false)),
  "output" => Dict("plots" => ["rm_anomaly", "rv_phasefold", "corner"], "save_pdf" => false),
)

data0, nrv, npm = Nereus._build_data(cfg["data"])
params0, _ = Nereus._build_model(cfg, data0, Nereus._build_star(cfg["star"]), nrv, npm)
thT = Nereus.Theta{Float64}(params0)
for (nm, v) in (("P_k1", P0), ("Tc_k1", Tc0), ("K_k1", K0), ("rr_k1", rr0), ("b_k1", b0),
                ("sesinw_k1", sesinw0), ("secosw_k1", secosw0),
                ("v_sin_i_star", vsini0), ("lambda_k1", lam0),
                ("q1_TESS", 0.3), ("q2_TESS", 0.2), ("gamma_HARPS", 0.0), ("sigma_HARPS", 1.0))
    try Nereus.set_param!(thT, nm, v) catch end
end
# RM-free baseline (v·sini=0) to report the injected RM amplitude
th0 = Nereus.Theta{Float64}(params0, copy(thT.values); td = thT.td)
th0.values[params0.layout.systemic.v_sin_i_star] = 0.0
rvp, _  = Nereus.rv_predictions(thT, data0)       # Keplerian + RM (rv_predictions now includes RM)
rv0, _  = Nereus.rv_predictions(th0, data0)        # Keplerian only
phf, _  = Nereus.phot_predictions(thT, data0)
n_intr  = count(t -> abs(mod(t - Tc0 + P0 / 2, P0) - P0 / 2) < 0.75 * tdur, trv)
rv_obs  = rvp .+ σrv .* randn(rng, length(trv))
fl_obs  = phf .+ σf .* randn(rng, length(tph))
@printf("RM synthetic: %d RV (%d in-transit) + %d phot | v·sini=%.0f m/s, λ=%.0f°, RM amp≈%.0f m/s (K=%.0f)\n",
        length(trv), n_intr, length(tph), vsini0, rad2deg(lam0), maximum(abs.(rvp .- rv0)), K0)
cfg["data"]["rv"]["values"]["rv"] = rv_obs
cfg["data"]["transit_photometry"][1]["values"]["flux"] = fl_obs

ok = true
chk(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
if REPLOT
    @printf("\n[replot] re-dispatching rm_anomaly from saved chains...\n")
    data, nr, np_ = Nereus._build_data(cfg["data"])
    params, _ = Nereus._build_model(cfg, data, Nereus._build_star(cfg["star"]), nr, np_)
    ch, _ = Nereus.load_chains(joinpath(out, "chains.nc"))
    Nereus._dispatch_plot("rm_anomaly", ch, params, data, joinpath(out, "plots"), Dict{Symbol,Any}())
    Nereus._dispatch_plot("rv_phasefold", ch, params, data, joinpath(out, "plots"), Dict{Symbol,Any}())
else
    @printf("\nrun_job RVPM_RM (in-transit RV + transit photometry, ptemcee)...\n"); t0 = time()
    s = Nereus.run_job(cfg)
    @printf("done %.1f min  status=%s\n", (time() - t0) / 60, s["status"])
    fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
    dp = get(s, "derived", Dict("parameters" => Dict()))["parameters"]
    gv(k, b) = get(get(b, k, Dict()), "value", NaN)
    vs = gv("v_sin_i_star", fp); lam = gv("lambda_k1", fp); K = gv("K_k1", fp)
    @printf("\nrecovery: v·sini=%.0f m/s (truth %.0f) | λ=%.1f° (truth %.0f°) | K=%.1f (truth %.0f)\n",
            vs, vsini0, rad2deg(lam), rad2deg(lam0), K, K0)
    chk("status ok", s["status"] == "ok")
    chk("v·sin i_* recovered (within 25%)", isfinite(vs) && abs(vs - vsini0) / vsini0 < 0.25)
    chk("λ recovered (within 12°)",         isfinite(lam) && abs(rad2deg(lam) - rad2deg(lam0)) < 12)
end

exists(rel) = (f = joinpath(out, rel); isfile(f) && filesize(f) > 1000)
chk("rm_anomaly plot emitted", exists(joinpath("plots", "models", "rm_anomaly_K1.png")))
println(ok ? "\n✅ RUN_JOB RM (rm_anomaly) PASS" : "\n❌ RUN_JOB RM — FAILURES ABOVE")
@printf("plot: %s\n", joinpath(out, "plots", "models", "rm_anomaly_K1.png"))
