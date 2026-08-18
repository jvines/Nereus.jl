#!/usr/bin/env julia
# Absolute-astrometry plot group (iad_residuals + pm_anomaly) through run_job.
# ε Eri (HIP 16537): real Hipparcos IAD + Gaia GOST + modern RV, single planet
# RVAS. Tight-ish orbit priors near the published ε Eri b orbit + short sampler
# (this validates the PLOT PATH, not a publication orbit). Asserts iad_residuals
# and pm_anomaly emit through the run_job dispatch.
using Nereus, Printf, Statistics, DelimitedFiles

out = joinpath(@__DIR__, "..", "..", "results", "runjob_iadgost_check")
const REPLOT = get(ENV, "IADGOST_REPLOT", "0") == "1"
REPLOT || (rm(out; force = true, recursive = true); mkpath(out))

# --- ε Eri modern RV subset (≥1 m/s era), 5σ MAD clip per instrument ----------
RV_FILE = joinpath(@__DIR__, "..", "data", "eps_eri.csv")
raw = readdlm(RV_FILE, ',', Any, '\n'; header = true)[1]
_f(c) = Float64[v isa AbstractString ? (isempty(strip(v)) ? NaN : parse(Float64, strip(v))) : Float64(v) for v in c]
bjd = _f(raw[:, 1]); rv = _f(raw[:, 2]); rv_err = _f(raw[:, 3]); inst_str = String.(raw[:, 16])
keep_inst = Set(["HARPS_PRE", "HARPS_POST", "HIRES_POST"])     # 3-instrument modern subset
keep = findall(s -> s in keep_inst, inst_str)
bjd = bjd[keep]; rv = rv[keep]; rv_err = rv_err[keep]; inst_str = inst_str[keep]
let names = sort(unique(inst_str)), m = Dict(n => i for (i, n) in enumerate(sort(unique(inst_str))))
    ri = [m[s] for s in inst_str]; k = trues(length(bjd))
    for g in unique(ri)
        idx = findall(==(g), ri); med = median(rv[idx]); mad = median(abs.(rv[idx] .- med))
        mad == 0 && continue
        for j in idx; abs(rv[j] - med) > 5 * 1.4826 * mad && (k[j] = false); end
    end
    global bjd = bjd[k]; global rv = rv[k]; global rv_err = rv_err[k]; global inst_str = inst_str[k]
end
@printf("ε Eri: %d modern RVs (%s) over %.1f yr | + HIP IAD + Gaia GOST\n",
        length(bjd), join(sort(unique(inst_str)), ","), (maximum(bjd) - minimum(bjd)) / 365.25)

cfg = Dict(
  "version" => "1.0", "seed" => 7, "output_dir" => out,
  "star" => Dict("M_s" => 0.82),
  "data" => Dict(
     "rv" => Dict("values" => Dict("bjd" => bjd, "rv" => rv, "rv_err" => rv_err,
                  "instrument" => inst_str)),
     "iad"  => Dict("hip_id" => 16537),
     "gost" => Dict("ra_deg" => 53.232665, "dec_deg" => -9.458250,
                    "from" => "2014-07-26T00:00:00", "to" => "2025-01-15T00:00:00")),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RVAS"],
     "parametrization" => Dict("mass" => "M_sec_driven", "time" => "Tp", "ew" => "sesinw"),
     "stability" => "none"),
  "priors" => Dict(
     "P_k1"     => Dict("type" => "NormalPrior",     "args" => [2700.0, 80.0, 2300.0, 3100.0]),
     "M_sec_k1" => Dict("type" => "LogUniformPrior", "args" => [1.0e-4, 0.05]),
     "inc_k1"   => Dict("type" => "SinePrior",       "args" => Float64[]),
     "Omega_k1" => Dict("type" => "UniformPrior",    "args" => [0.0, 2π]),
     "plx"      => Dict("type" => "NormalPrior",     "args" => [310.94, 0.17, 305.0, 320.0])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
      "n_temps" => 8, "n_walkers" => 60, "n_steps" => 3000, "n_burnin" => 1800,
      "show_progress" => false)),
  "output" => Dict("plots" => ["iad_residuals", "pm_anomaly", "rv_timeseries", "corner"],
                   "save_pdf" => false),
)

ok = true
chk(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
if REPLOT
    @printf("\n[replot] re-dispatching iad/gost plots from saved chains...\n")
    data, nrv, npm = Nereus._build_data(cfg["data"])
    params, _ = Nereus._build_model(cfg, data, Nereus._build_star(cfg["star"]), nrv, npm)
    ch, _ = Nereus.load_chains(joinpath(out, "chains.nc"))
    for p in ["iad_residuals", "pm_anomaly"]
        Nereus._dispatch_plot(p, ch, params, data, joinpath(out, "plots"), Dict{Symbol,Any}())
    end
    # Post-fit pipeline timing on the REAL ε Eri data (968 RV / 19 yr) — confirms
    # run_job's PPC/LOO/detection_limits/fit_health all finish in sane wall-clock.
    summ = Dict{String, Any}()
    @printf("\npost-fit pipeline timing (real ε Eri: %d RV / 19 yr):\n", length(data.t_rv))
    for (nm, fn) in [
            ("ppc",              () -> Nereus._run_ppc!(cfg, ch, params, data, out, summ)),
            ("detection_limits", () -> Nereus._run_detection_limits!(cfg, ch, params, data, out, summ)),
            ("loo",              () -> Nereus._run_loo!(cfg, ch, params, data, nothing, summ)),
            ("fit_health",       () -> Nereus._run_fit_health!(cfg, ch, params, summ))]
        dt = @elapsed (try fn() catch e; @printf("    (%s threw: %s)\n", nm, sprint(showerror, e)) end)
        @printf("  %-18s %7.1f s\n", nm, dt)
    end
else
    @printf("\nrun_job ε Eri RVAS (RV + HIP IAD + Gaia GOST, ptemcee)...\n"); t0 = time()
    s = Nereus.run_job(cfg)
    @printf("done %.1f min  status=%s\n", (time() - t0) / 60, s["status"])
    fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
    dp = get(s, "derived", Dict("parameters" => Dict()))["parameters"]
    gv(k, b) = get(get(b, k, Dict()), "value", NaN)
    P = gv("P_k1", fp); Msec = gv("M_sec_k1", fp); inc = gv("inc_k1", fp)
    @printf("\nrecovery: P=%.0f d  M_sec=%.4f M_sun (%.2f M_J)  inc=%.0f°  (ε Eri b: P≈2700, m sin i≈0.66 M_J)\n",
            P, Msec, Msec * 1047.6, rad2deg(inc))
    chk("status ok", s["status"] == "ok")
    chk("P near ε Eri b (within 300 d)", isfinite(P) && abs(P - 2700) < 300)
end

exists(rel) = (f = joinpath(out, rel); isfile(f) && filesize(f) > 1000)
anyfile(substr) = any(occursin(substr, lowercase(f)) && filesize(joinpath(r, f)) > 1000
                      for (r, _, fs) in walkdir(joinpath(out, "plots")) for f in fs)
println("\n=== absolute-astrometry plot group ===")
chk("iad_residuals emitted (Hipparcos IAD abscissa residuals)", anyfile("iad_residuals"))
chk("pm_anomaly emitted (Gaia GOST proper-motion reflex)",      anyfile("pm_anomaly"))

println(ok ? "\n✅ RUN_JOB IAD+GOST PLOTS PASS" : "\n❌ RUN_JOB IAD+GOST — FAILURES ABOVE")
@printf("plots dir: %s\n", joinpath(out, "plots", "models"))
