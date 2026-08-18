#!/usr/bin/env julia
# RVAS (RV + relative astrometry) recovery through run_job — validates the
# astrometry DISPATCH path of the documented entry point (the PM path had a
# blocker; this checks the AS path). Synthesizes RV + relastrom for a known
# orbit, fits mode RVAS, and asserts recovery — especially INCLINATION, which RV
# alone cannot constrain, so a correct inc proves astrometry is in the fit.
using Nereus, Printf, Random, Statistics

# --- truth orbit (e≠0 to avoid the sesinw/secosw origin degeneracy) -----------
P0, K0 = 900.0, 60.0
e0, w0, Mo0 = 0.30, 0.7, 1.2
inc0, Om0   = deg2rad(63.0), deg2rad(80.0)
Mpri0, plx0 = 1.0, 25.0
sesinw0, secosw0 = sqrt(e0) * sin(w0), sqrt(e0) * cos(w0)
rng = MersenneTwister(20260628)

t_rv = collect(54000.0:12.0:54000.0+1800.0)          # MJD, ~2.7 yr (~1 orbit)
t_as = collect(54100.0:150.0:54100.0+1650.0)         # ~12 relastrom epochs
σrv, σas = 2.0, 0.5                                   # m/s, mas

# placeholder RVAS target (K_driven) to get params + t_ref, then forward-model truth
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

# forward-model RV + relastrom at truth
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
@printf("RVAS synthetic: %d RV + %d relastrom epochs; truth M_sec=%.4f Msun, astrom amp≈%.1f mas\n",
        length(t_rv), length(t_as), Msec0, maximum(abs.(vcat(ra_true, dec_true))))

# --- fit through run_job ------------------------------------------------------
out = mktempdir()
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
     "P_k1"  => Dict("type" => "NormalPrior",  "args" => [P0, 5.0, P0 - 100, P0 + 100]),
     "plx"   => Dict("type" => "NormalPrior",  "args" => [plx0, 0.1, plx0 - 2, plx0 + 2])),
  "noise_models" => [],
  "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict(
      "n_temps" => 6, "n_walkers" => 44, "n_steps" => 2500, "n_burnin" => 1200, "show_progress" => false)),
  "output" => Dict("plots" => String[], "save_pdf" => false),
)

@printf("\nrun_job RVAS (RV + relative astrometry)...\n")
s = Nereus.run_job(cfg)

ok = true
check(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
fp = get(s, "fitted", Dict("parameters" => Dict()))["parameters"]
dp = get(s, "derived", Dict("parameters" => Dict()))["parameters"]
println("=== RVAS dispatch + recovery ===")
check("status ok (run_job astrometry dispatch works)", s["status"] == "ok")
check("fitted has K_k1 (RV)",        haskey(fp, "K_k1"))
check("fitted has inc_k1 (ASTROMETRY-only constraint)", haskey(fp, "inc_k1"))
check("fitted has Omega_k1",         haskey(fp, "Omega_k1"))
check("derived has ecc_k1",          haskey(dp, "ecc_k1"))
check("derived has a_au_k1",         haskey(dp, "a_au_k1"))

gv(k, b = fp) = get(get(b, k, Dict()), "value", NaN)
P, K, ecc = gv("P_k1"), gv("K_k1"), gv("ecc_k1", dp)
inc = gv("inc_k1")                                  # radians
@printf("\n recovery: P=%.1f (truth %.1f)  K=%.1f (truth %.1f)  ecc=%.3f (truth %.3f)  inc=%.1f° (truth 63°/117° mirror)\n",
        P, P0, K, K0, ecc, e0, rad2deg(inc))
check("P within 20 d",   isfinite(P) && abs(P - P0) < 20)
check("K within 8 m/s",  isfinite(K) && abs(K - K0) < 8)
check("ecc within 0.06", isfinite(ecc) && abs(ecc - e0) < 0.06)
# inclination: astrometry-only; allow the i↔180−i mirror (relastrom degeneracy)
check("|sin i| matches truth (astrometry working)",
      isfinite(inc) && abs(abs(sin(inc)) - sin(inc0)) < 0.08)

println(ok ? "\n✅ RVAS DISPATCH + RECOVERY PASS" : "\n❌ RVAS FAIL")
