#!/usr/bin/env julia
# TTV plot group (ttv_oc) through run_job. A single transiting planet with an
# INJECTED sinusoidal transit-timing variation: photometry is generated at
# per-transit-shifted times so transit n lands at Tc+n·P+Δ_n. The fit uses a
# linear ephemeris (no perturber), so the measured per-transit Tc's — and hence
# the O−C diagram — must recover the injected Δ_n. Asserts the orbit is
# recovered, the measured O−C correlates with the injection, and ttv_oc emits.
using Nereus, Printf, Random, Statistics

out = joinpath(@__DIR__, "..", "..", "results", "runjob_ttv_check")
const REPLOT = get(ENV, "TTV_REPLOT", "0") == "1"   # re-dispatch ttv_oc from saved chains, no re-fit
REPLOT || (rm(out; force = true, recursive = true); mkpath(out))
rng = MersenneTwister(31)

# --- 1 transiting planet (e≠0 off-origin) + injected sinusoidal TTV ----------
P0, Tc0, K0, rr0, b0 = 3.6, 1.5, 12.0, 0.10, 0.30
sesinw0, secosw0 = 0.30, 0.18                    # e≈0.12
σrv, σf = 2.0, 2.5e-4
Ntr = 12                                          # transits 0..11
A_ttv, Nsuper = 0.012, 5.0                        # 0.012 d ≈ 17 min, 5-transit super-period
Δ(t) = A_ttv * sin(2π * round((t - Tc0) / P0) / Nsuper)

tph = Float64[]                                   # transit windows, native cadence
for n in 0:(Ntr - 1)
    c = Tc0 + n * P0
    append!(tph, collect(range(c - 0.13, c + 0.13; length = 42)))
end
sort!(tph)
trv = sort(rand(rng, 45) .* (Ntr * P0))

# generate the LC at TTV-shifted times so transit n appears at Tc+n·P+Δ_n
t_shift = tph .- Δ.(tph)
tgt0 = build_target(M_s = 1.0, R_s = 1.0,
    planets = (b = (P = P0, Tc = Tc0, K = K0, sesinw = sesinw0, secosw = secosw0, b = b0, rr = rr0),),
    rv = (HARPS = (data = (t = trv, rv = zeros(length(trv)), rv_err = fill(σrv, length(trv))),
                   gamma = 0.0, sigma = 1.0),),
    phot = (TESS = (data = (t = t_shift, flux = ones(length(t_shift)), flux_err = fill(σf, length(t_shift))),
                    jitter = 1e-4, offset = 0.0, q1 = 0.3, q2 = 0.2),))
th0 = Nereus.Theta{Float64}(tgt0.params)
rvp, _ = Nereus.rv_predictions(th0, tgt0.data)
phf, _ = Nereus.phot_predictions(th0, tgt0.data)         # evaluated at shifted times
rv_obs = rvp .+ σrv .* randn(rng, length(trv))
fl_obs = phf .+ σf .* randn(rng, length(tph))             # assigned to ORIGINAL tph
@printf("TTV synthetic: %d transits, %d phot + %d RV | inj. TTV amp=%.0f min, super-period=%.0f transits\n",
        Ntr, length(tph), length(trv), A_ttv * 1440, Nsuper)

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
     "K_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 30.0]),
     "rr_k1" => Dict("type" => "UniformPrior", "args" => [0.05, 0.20]),
     "b_k1"  => Dict("type" => "UniformPrior", "args" => [0.0, 1.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
      "n_temps" => 6, "n_walkers" => 60, "n_steps" => 2500, "n_burnin" => 1500,
      "show_progress" => false)),
  "output" => Dict("plots" => ["ttv_oc", "pm_phasefold", "corner"], "save_pdf" => false),
)

ok = true
chk(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
if REPLOT
    @printf("\n[replot] re-dispatching ttv_oc from saved chains...\n")
else
    @printf("\nrun_job TTV (RVPM, linear ephemeris, injected O−C)...\n")
    s = Nereus.run_job(cfg)
    fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
    gv(k) = get(get(fp, k, Dict()), "value", NaN)
    P, Tc, rr = gv("P_k1"), gv("Tc_k1"), gv("rr_k1")
    @printf("\nrecovery: P=%.4f (truth %.4f) | Tc=%.4f (truth %.4f) | rr=%.3f (truth %.2f)\n",
            P, P0, Tc, Tc0, rr, rr0)
    chk("status ok", s["status"] == "ok")
    chk("P within 0.01 d", isfinite(P) && abs(P - P0) < 0.01)
    chk("rr within 0.02",  isfinite(rr) && abs(rr - rr0) < 0.02)
end

# rebuild params/data with the CORRECT instrument names, dispatch ttv_oc directly
ch, _ = Nereus.load_chains(joinpath(out, "chains.nc"))
data, nrv, npm = Nereus._build_data(cfg["data"])
params, _ = Nereus._build_model(cfg, data, Nereus._build_star(cfg["star"]), nrv, npm)
Nereus._dispatch_plot("ttv_oc", ch, params, data, joinpath(out, "plots"), Dict{Symbol, Any}())
Nereus._dispatch_plot("transit_overlay", ch, params, data, joinpath(out, "plots"), Dict{Symbol, Any}())

# measured O−C vs injected Δ_n — does the diagram recover the injection?
theta_med = Nereus.Theta{Float64}(params)
Nereus.set_n_p!(theta_med, params.config.max_kplanet)
for nm in keys(params.layout.name_to_idx)
    sym = Symbol(nm)
    sym in names(ch, :parameters) || continue
    theta_med.values[params.layout.name_to_idx[nm]] = median(vec(Array(ch[:, sym, :])))
end
Pm  = theta_med.values[params.layout.name_to_idx["P_k1"]]
Tcm = theta_med.values[params.layout.name_to_idx["Tc_k1"]]
nlo = ceil(Int, (minimum(data.t_phot) + 0.1 - Tcm) / Pm)
nhi = floor(Int, (maximum(data.t_phot) - 0.1 - Tcm) / Pm)
tcs_pred = [Tcm + n * Pm for n in nlo:nhi]
try
    meas = Nereus.measure_per_transit_tcs(data, theta_med, params; tcs_predicted = tcs_pred,
                                            half_window = 0.5, grid_min = 0.4, ngrid = 801, planet_k = 1)
    inj = [A_ttv * sin(2π * n / Nsuper) for n in nlo:nhi]      # injected Δ at each cycle
    r = cor(meas.mle_dt, inj)
    @printf("\nO−C: measured vs injected correlation = %.2f over %d transits (amp meas≈%.0f min)\n",
            r, length(inj), maximum(abs.(meas.mle_dt)) * 1440)
    chk("measured O−C recovers injected TTV (corr > 0.6)", isfinite(r) && r > 0.6)
catch err
    @printf("\nO−C measurement raised %s\n", err)
    chk("measured O−C recovers injected TTV (corr > 0.6)", false)
end

exists(rel) = (f = joinpath(out, rel); isfile(f) && filesize(f) > 1000)
chk("ttv_oc plot emitted", exists(joinpath("plots", "models", "ttv_oc.png")))
chk("transit_overlay plot emitted", exists(joinpath("plots", "models", "transit_overlay_K1.png")))

println(ok ? "\n✅ RUN_JOB TTV (ttv_oc) PASS" : "\n❌ RUN_JOB TTV — FAILURES ABOVE")
@printf("plot: %s\n", joinpath(out, "plots", "models", "ttv_oc.png"))
