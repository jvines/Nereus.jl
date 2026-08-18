#!/usr/bin/env julia
# Activity-GP plot group through run_job (now that ActivityGP is wired into the
# noise registry). Synthetic: a shared rotation activity G(t) + its derivative,
# projected to RV (Vc,Vr), BIS (Bc,Br), FWHM (Fc,Fr) FF'-style, plus a planet in
# the RV. Fit with an ActivityGP noise model on channels [bis,fwhm]; assert the
# planet is recovered through the activity and the activity-GP plots emit.
using Nereus, Printf, Random

out = joinpath(@__DIR__, "..", "..", "results", "runjob_agp_check")
const REPLOT = get(ENV, "AGP_REPLOT", "0") == "1"   # re-render plots from saved chains, no re-fit
REPLOT || (rm(out; force = true, recursive = true); mkpath(out))
rng = MersenneTwister(11)

# --- planet (via forward model) + shared rotation activity --------------------
P0, Tc0, K0 = 4.05, 1.0, 6.0
sesinw0, secosw0 = 0.30, 0.18          # e≈0.12
Prot = 23.0
trv = sort(rand(rng, 55) .* 90.0)      # ~4 rotation cycles
σrv, σbis, σfwhm = 1.5, 3.0, 2.0
tgt0 = build_target(M_s = 1.0,
    planets = (b = (P = P0, Tc = Tc0, K = K0, sesinw = sesinw0, secosw = secosw0),),
    rv = (HARPS = (data = (t = trv, rv = zeros(length(trv)), rv_err = fill(σrv, length(trv))),
                   gamma = 0.0, sigma = 1.0),))
th0 = Nereus.Theta{Float64}(tgt0.params)
rvp, _ = Nereus.rv_predictions(th0, tgt0.data)     # planet RV
G  = sin.(2π .* trv ./ Prot)                         # latent rotation proxy
Gd = (2π / Prot) .* cos.(2π .* trv ./ Prot)          # its derivative
Vc, Vr = 8.0, 2.5;  Bc, Br = 15.0, 4.0;  Fc, Fr = 10.0, 3.0
rv_obs   = rvp .+ Vc .* G .+ Vr .* Gd .+ σrv .* randn(rng, length(trv))
bis_obs  =        Bc .* G .+ Br .* Gd .+ σbis .* randn(rng, length(trv))
fwhm_obs =        Fc .* G .+ Fr .* Gd .+ σfwhm .* randn(rng, length(trv))
@printf("AGP synthetic: %d RV+BIS+FWHM | planet K=%.0f, rotation P=%.0f d (RV act amp≈%.0f m/s)\n",
        length(trv), K0, Prot, Vc)

cfg = Dict(
  "version" => "1.0", "seed" => 1, "output_dir" => out,
  "star" => Dict("M_s" => 1.0),
  "data" => Dict(
     "rv" => Dict("values" => Dict("bjd" => trv, "rv" => rv_obs,
                  "rv_err" => fill(σrv, length(trv)), "instrument" => fill("HARPS", length(trv)),
                  "bis" => bis_obs, "bis_err" => fill(σbis, length(trv)),
                  "fwhm" => fwhm_obs, "fwhm_err" => fill(σfwhm, length(trv))))),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Tc", "ew" => "sesinw"),
     "stability" => "none"),
  "priors" => Dict(
     "P_k1" => Dict("type" => "NormalPrior", "args" => [P0, 0.02, P0 - 0.3, P0 + 0.3]),
     "K_k1" => Dict("type" => "UniformPrior", "args" => [0.0, 30.0])),
  "noise_models" => [Dict("kind" => "ActivityGP", "instruments" => ["HARPS"],
      "kwargs" => Dict("channels" => ["bis", "fwhm"], "use_derivative" => true))],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
      "n_temps" => 8, "n_walkers" => 80, "n_steps" => 1500, "n_burnin" => 1000,
      "show_progress" => false)),
  "output" => Dict("plots" => ["activity_gp_decomposition", "activity_gp_latent",
                               "rv_timeseries", "rv_components", "corner"],
                   "save_pdf" => false),
)

if REPLOT
    # Rebuild params/data from the same cfg (cheap, no sampler) and re-render the
    # two AGP plots from the saved chains — for iterating on plot styling.
    data, nrv, npm = Nereus._build_data(cfg["data"])
    star = Nereus._build_star(cfg["star"])
    params, _ = Nereus._build_model(cfg, data, star, nrv, npm)
    ch, _ = Nereus.load_chains(joinpath(out, "chains.nc"))
    mkpath(joinpath(out, "plots"))
    Nereus.plot_activity_gp_latent(ch, params, data;
        filename = joinpath(out, "plots", "activity_gp_latent.png"))
    Nereus.plot_activity_gp_decomposition(ch, params, data;
        filename = joinpath(out, "plots", "activity_gp_decomposition.png"))
    Nereus.plot_rv_timeseries(ch, params, data; output = joinpath(out, "plots"))
    Nereus.plot_rv_components(ch, params, data; output = joinpath(out, "plots"))
    println("re-plotted AGP + rv_timeseries (decorr) + rv_components (decomp)"); exit(0)
end

@printf("\nrun_job AGP (RV + BIS + FWHM, ActivityGP)...\n")
s = Nereus.run_job(cfg)

ok = true
chk(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
K = get(get(fp, "K_k1", Dict()), "value", NaN)
@printf("\nrecovery: K=%.2f m/s (truth %.1f) through %.0f m/s of rotation activity\n", K, K0, Vc)
chk("status ok", s["status"] == "ok")
chk("K recovered within 3 m/s (planet survives the activity GP)", isfinite(K) && abs(K - K0) < 3.0)

exists(rel) = (f = joinpath(out, rel); isfile(f) && filesize(f) > 1000)
hits = filter(exists, [joinpath(r, f) for (r, _, fs) in walkdir(out) for f in fs
                       if occursin("activity", lowercase(f)) || occursin("gp", lowercase(f))]) .|>
       (p -> relpath(p, out))
println("\nactivity-GP plots emitted:"); foreach(p -> println("  ", p), hits)
chk("an activity_gp_decomposition plot emitted",
    any(p -> occursin("decomp", lowercase(p)), hits))
chk("an activity_gp_latent plot emitted",
    any(p -> occursin("latent", lowercase(p)), hits))
chk("rv_components plot emitted (component decomposition)",
    exists(joinpath("plots", "models", "rv_components.png")))
chk("rv_timeseries plot still emitted (unchanged decorr view)",
    exists(joinpath("plots", "models", "rv_timeseries.png")))

println(ok ? "\n✅ RUN_JOB ACTIVITY-GP PLOTS PASS" : "\n❌ RUN_JOB ACTIVITY-GP — FAILURES ABOVE")
@printf("plots dir: %s\n", joinpath(out, "plots"))
