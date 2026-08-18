#!/usr/bin/env julia
# Astrometry plot group (RV + relative astrometry) through run_job. Synthesize a
# known orbit, fit RVAS, and assert: the orbit is recovered (incl. INCLINATION —
# astrometry-only, so a correct inc proves astrometry is in the fit), the fit
# CONVERGES (assess_fit on the cube), and the astrometry plot suite emits
# (orbit_skyplane, relastrom residuals/timeseries, rv_astrom_phasefold). inc is
# bounded to <90° to break the i↔180−i mirror so the demo fit is unimodal.
using Nereus, Printf, Random, Statistics

out = joinpath(@__DIR__, "..", "..", "results", "runjob_astrom_check")
rm(out; force = true, recursive = true); mkpath(out)

# --- truth orbit (e≠0; prograde inc<90° so the mirror is broken) -------------
P0, K0 = 900.0, 60.0
e0, w0, Mo0 = 0.30, 0.7, 1.2
inc0, Om0   = deg2rad(63.0), deg2rad(80.0)
Mpri0, plx0 = 1.0, 25.0
sesinw0, secosw0 = sqrt(e0) * sin(w0), sqrt(e0) * cos(w0)
rng = MersenneTwister(20260628)

t_rv = collect(54000.0:12.0:54000.0 + 1800.0)         # MJD, ~2.7 yr (~1 orbit)
t_as = collect(54100.0:120.0:54100.0 + 1680.0)        # ~15 relastrom epochs
σrv, σas = 2.0, 0.5                                    # m/s, mas

ph_rv = (t = t_rv, rv = zeros(length(t_rv)), rv_err = fill(σrv, length(t_rv)))
ph_as = RelAstromData(t = t_as, ra_off = zeros(length(t_as)), dec_off = zeros(length(t_as)),
                      ra_err = fill(σas, length(t_as)), dec_err = fill(σas, length(t_as)))
tgt0 = build_target(M_pri = Mpri0, plx = plx0,
    planets = (b = (P = LogUniformPrior(100.0, 5000.0), K = LogUniformPrior(1.0, 500.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2π), inc = SinePrior(), Omega = UniformPrior(0.0, 2π)),),
    rv = (HARPS = (data = ph_rv, sigma = LogUniformPrior(0.1, 30.0),
                   gamma = UniformPrior(-200.0, 200.0)),),
    relAST = ph_as)
th = Nereus.Theta{Float64}(tgt0.params)
for (nm, v) in (("P_k1",P0),("K_k1",K0),("sesinw_k1",sesinw0),("secosw_k1",secosw0),
                ("Mo_k1",Mo0),("inc_k1",inc0),("Omega_k1",Om0),
                ("gamma_HARPS",0.0),("sigma_HARPS",1.0))
    Nereus.set_param!(th, nm, v)
end
t_ref = tgt0.data.t_ref
rv_true = Nereus.compute_rv_model_on_grid(th, tgt0.data, t_rv)
orb, Msec0 = Nereus._planet_orbit(th, 1, Nereus.astrom_M_pri(th), Nereus.astrom_plx(th), t_ref)
ra_true = Float64[]; dec_true = Float64[]
for t in t_as
    dra, ddec = relastrom_offset(orb, t)
    push!(ra_true, dra); push!(dec_true, ddec)
end
rv_obs  = rv_true  .+ σrv .* randn(rng, length(t_rv))
ra_obs  = ra_true  .+ σas .* randn(rng, length(t_as))
dec_obs = dec_true .+ σas .* randn(rng, length(t_as))
@printf("RVAS synthetic: %d RV + %d relastrom epochs | M_sec=%.4f Msun, astrom amp≈%.1f mas\n",
        length(t_rv), length(t_as), Msec0, maximum(abs.(vcat(ra_true, dec_true))))

requested = ["orbit_skyplane", "relastrom_residuals", "relastrom_timeseries",
             "rv_astrom_phasefold", "corner"]
cfg = Dict(
  "version" => "1.0", "seed" => 1, "output_dir" => out,
  "star" => Dict("M_s" => Mpri0),
  "data" => Dict(
     "rv" => Dict("values" => Dict("bjd" => t_rv, "rv" => rv_obs,
                  "rv_err" => fill(σrv, length(t_rv)), "instrument" => fill("HARPS", length(t_rv)))),
     "relastrom" => Dict("values" => Dict("t" => t_as, "ra_off" => ra_obs, "dec_off" => dec_obs,
                  "ra_err" => fill(σas, length(t_as)), "dec_err" => fill(σas, length(t_as))))),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RVAS"],
     "parametrization" => Dict("mass" => "K_driven", "time" => "Mo", "ew" => "sesinw"),
     "stability" => "none"),
  "priors" => Dict(
     "P_k1"   => Dict("type" => "NormalPrior",  "args" => [P0, 5.0, P0 - 100, P0 + 100]),
     "plx"    => Dict("type" => "NormalPrior",  "args" => [plx0, 0.1, plx0 - 2, plx0 + 2]),
     # bound inc < 90° to break the i↔180−i relastrom mirror (unimodal demo)
     "inc_k1" => Dict("type" => "UniformPrior", "args" => [0.0, π/2])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
      "n_temps" => 8, "n_walkers" => 60, "n_steps" => 3500, "n_burnin" => 2000,
      "show_progress" => false)),
  "output" => Dict("plots" => requested, "save_pdf" => false),
)

@printf("\nrun_job RVAS (RV + relative astrometry, ptemcee)...\n")
s = Nereus.run_job(cfg)

ok = true
chk(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
dp = get(s, "derived", Dict("parameters" => Dict()))["parameters"]
gv(k, b = fp) = get(get(b, k, Dict()), "value", NaN)
P, K, ecc, inc = gv("P_k1"), gv("K_k1"), gv("ecc_k1", dp), gv("inc_k1")
@printf("\nrecovery: P=%.1f (truth %.1f) | K=%.1f (truth %.1f) | ecc=%.3f (truth %.3f) | inc=%.1f° (truth 63°)\n",
        P, P0, K, K0, ecc, e0, rad2deg(inc))
chk("status ok", s["status"] == "ok")
chk("P within 20 d",        isfinite(P) && abs(P - P0) < 20)
chk("K within 8 m/s",       isfinite(K) && abs(K - K0) < 8)
chk("ecc within 0.06",      isfinite(ecc) && abs(ecc - e0) < 0.06)
chk("inc within 6° (astrometry constrains it)", isfinite(inc) && abs(rad2deg(inc) - 63.0) < 6.0)

# convergence on the cube
ch, _ = Nereus.load_chains(joinpath(out, "chains.nc"))
hr = assess_fit(ch)
cstat(name) = (i = findfirst(c -> c.name === name, hr.checks); i === nothing ? :absent : hr.checks[i].status)
@printf("\nconvergence: overall=%s | convergence=%s | multimodality=%s\n",
        hr.overall, cstat(:convergence), cstat(:multimodality))
chk("fit converged (assess_fit :convergence ok)", cstat(:convergence) === :ok)

# astrometry plot suite emitted
exists_nonempty(rel) = (f = joinpath(out, rel); isfile(f) && filesize(f) > 1000)
count_png(d) = (p = joinpath(out, d); isdir(p) ?
    sum(endswith(f, ".png") for (r,_,fs) in walkdir(p) for f in fs; init = 0) : 0)
println("\n=== astrometry plot suite ===")
for (label, rel) in [
        ("orbit skyplane",     "plots/models/orbit_skyplane_K1.png"),
        ("relastrom resid",    "plots/models/relastrom_residuals_K1.png"),
        ("relastrom tseries",  "plots/models/relastrom_timeseries_K1.png"),
        ("rv_astrom fold",     "plots/models/rv_astrom_phasefold_K1.png"),
        ("corner",             "plots/corner.png")]
    @printf("  %-18s %s\n", label, exists_nonempty(rel) ? "✓" : "MISSING")
    chk("$label emitted", exists_nonempty(rel))
end
@printf("  traces=%d posteriors=%d\n", count_png("plots/traces"), count_png("plots/posteriors"))

println(ok ? "\n✅ RUN_JOB ASTROMETRY PLOT-SUITE PASS" : "\n❌ RUN_JOB ASTROMETRY — FAILURES ABOVE")
@printf("plots dir: %s\n", joinpath(out, "plots"))
