#!/usr/bin/env julia
# Trans-dimensional plot group. A strong synthetic 1-planet RV+transit through
# run_job with sample_transdim_ptemcee exploring planet count 0..2, and assert:
# the occupancy concentrates on N_p = 1 (model selection recovers the truth),
# the science is recovered at the modal count, and the transdim_occupancy plot
# emits. The occupancy = P(N_p | data) plot is the headline trans-dim output.
using Nereus, Printf, Random

out = joinpath(@__DIR__, "..", "..", "results", "runjob_transdim_check")
rm(out; force = true, recursive = true); mkpath(out)
rng = MersenneTwister(7)

# --- strong synthetic 1-planet RV+transit (e=0.2 off the origin) -------------
P0, Tc0, K0, rr0, b0 = 3.0, 1.0, 14.0, 0.10, 0.30
sesinw0, secosw0 = 0.3763, 0.2416         # e=0.20
jit0, σrv, σf = 2.0, 2.0, 3e-4
trv = sort(rand(rng, 60) .* 40.0)
tph = sort(vcat(collect(range(Tc0 - 0.12, Tc0 + 0.12; length = 60)),
                collect(range(Tc0 + P0 - 0.12, Tc0 + P0 + 0.12; length = 60))))
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
@printf("trans-dim synthetic: %d RV + %d phot | 1 planet (P=%.1f K=%.0f), search N_p 0..2\n",
        length(trv), length(tph), P0, K0)

cfg = Dict(
  "version" => "1.0", "seed" => 1, "output_dir" => out,
  "star" => Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 5800.0),
  "data" => Dict(
     "rv" => Dict("values" => Dict("bjd" => trv, "rv" => rv_obs,
                  "rv_err" => fill(σrv, length(trv)), "instrument" => fill("HARPS", length(trv)))),
     "transit_photometry" => [Dict(
         "values" => Dict("bjd" => tph, "flux" => fl_obs, "flux_err" => fill(σf, length(tph))),
         "instrument" => "TESS", "exposure_time" => 120.0)]),
  "model" => Dict("max_kplanet" => 2, "planet_modes" => ["RVPM", "RVPM"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tc", "ew" => "sesinw",
                               "geom" => "b_rr", "marginalize_gamma" => true),
     "stability" => "none"),
  # broad-ish search priors so births must actually FIND the planet
  "priors" => Dict(
     "P_k1"  => Dict("type" => "UniformPrior", "args" => [2.5, 3.5]),
     "P_k2"  => Dict("type" => "UniformPrior", "args" => [2.5, 3.5]),
     "K_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 40.0]),
     "K_k2"  => Dict("type" => "UniformPrior", "args" => [0.0, 40.0]),
     "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "rr_k2" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "Tc_k1" => Dict("type" => "NormalPrior",  "args" => [Tc0, 0.05, Tc0 - 0.3, Tc0 + 0.3]),
     "Tc_k2" => Dict("type" => "NormalPrior",  "args" => [Tc0, 0.05, Tc0 - 0.3, Tc0 + 0.3]),
     "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0]),
     "b_k2"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  "transdim" => Dict("max_kplanet" => 2, "planets" => true, "noise" => false,
                     "transdim_fraction" => 0.3, "birth_strategies" => ["PriorBirth"]),
  "sampler" => Dict("name" => "transdim_ptemcee", "kwargs" => Dict(
      "n_temps" => 8, "n_walkers" => 80, "n_steps" => 500, "n_burnin" => 300,
      "informed_birth_fraction" => 0.5, "n_birth_tries" => 5, "show_progress" => false)),
  "output" => Dict("plots" => ["transdim_occupancy"], "save_pdf" => false),
)

@printf("\nrun_job trans-dim (transdim_ptemcee, N_p 0..2)...\n")
s = Nereus.run_job(cfg)

ok = true
chk(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))

# occupancy from the saved chain
ch, _ = Nereus.load_chains(joinpath(out, "chains.nc"))
np = Int.(round.(Float64.(vec(Array(ch[:n_planets])))))
occ = [count(==(k), np) / length(np) for k in 0:2]
@printf("\noccupancy  P(N_p=0)=%.3f  P(N_p=1)=%.3f  P(N_p=2)=%.3f  (modal=%d)\n",
        occ[1], occ[2], occ[3], argmax(occ) - 1)
chk("status ok", s["status"] == "ok")
chk("occupancy concentrates on N_p=1 (>0.6)", occ[2] > 0.6)
chk("N_p=1 is the modal count",               argmax(occ) == 2)

# the occupancy plot emitted
occ_png = joinpath(out, "plots", "transdim", "occupancy.png")
chk("transdim_occupancy plot emitted", isfile(occ_png) && filesize(occ_png) > 1000)

println(ok ? "\n✅ RUN_JOB TRANS-DIM OCCUPANCY PASS" : "\n❌ RUN_JOB TRANS-DIM — FAILURES ABOVE")
@printf("plot: %s\n", occ_png)
