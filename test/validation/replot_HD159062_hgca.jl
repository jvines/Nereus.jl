#!/usr/bin/env julia
# Re-render hgca_pm_residuals (HGCA proper-motion residuals) for HD 159062 from
# the existing run_job chains — validates the absolute-astrometry HGCA plot path
# through the run_job dispatch, no re-fit. HD159062 = real RV + relAST + HGCA.
using Nereus, Printf

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
out     = joinpath(@__DIR__, "..", "..", "results", "HD159062_runjob")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))

cfg = Dict(
  "version" => "1.0", "seed" => 42, "output_dir" => out,
  "star" => Dict("M_s" => 0.81),
  "data" => Dict(
     "rv" => Dict("values" => Dict(
        "bjd" => rvdat.t, "rv" => rvdat.rv, "rv_err" => rvdat.rv_err,
        "instrument" => fill("HIRES", length(rvdat.t)))),
     "relastrom" => Dict("values" => Dict(
        "t" => relast.t, "ra_off" => relast.ra_off, "dec_off" => relast.dec_off,
        "ra_err" => relast.ra_err, "dec_err" => relast.dec_err)),
     "hgca" => Dict("values" => Dict(
        "t" => collect(hgca.epochs), "pmra" => collect(hgca.pmra),
        "pmdec" => collect(hgca.pmdec),
        "cov_ep" => [hgca.cov_ep[1], hgca.cov_ep[2], hgca.cov_ep[3]],
        "plx" => hgca.plx, "plx_err" => hgca.plx_err, "hip_id" => hgca.hip_id))),
  "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RVAS"],
     "parametrization" => Dict("mass" => "a_driven", "time" => "Mo", "ew" => "sesinw"),
     "stability" => "none"),
  "priors" => Dict(
     "a_k1"     => Dict("type" => "LogUniformPrior", "args" => [5.0, 1500.0]),
     "M_sec_k1" => Dict("type" => "LogUniformPrior", "args" => [0.05, 1.5]),
     "M_pri"    => Dict("type" => "NormalPrior",     "args" => [0.81, 0.04, 0.5, 1.2])),
  "noise_models" => [], "sampler" => Dict("name" => "ptemcee", "kwargs" => Dict()),
)

data, nrv, npm = Nereus._build_data(cfg["data"])
star = Nereus._build_star(cfg["star"])
params, _ = Nereus._build_model(cfg, data, star, nrv, npm)
ch, _ = Nereus.load_chains(joinpath(out, "chains.nc"))
@printf("HD159062: HGCA HIP %d, %d RV, %d relAST | chains %s\n",
        hgca.hip_id, length(rvdat.t), length(relast.t), size(ch))

f = Nereus._dispatch_plot("hgca_pm_residuals", ch, params, data,
                           joinpath(out, "plots"), Dict{Symbol,Any}())
hits = [joinpath(r, fn) for (r, _, fs) in walkdir(joinpath(out, "plots")) for fn in fs
        if occursin("hgca", lowercase(fn))]
println("hgca_pm_residuals emitted:"); foreach(p -> println("  ", relpath(p, out)), hits)
ok = !isempty(hits) && all(p -> filesize(p) > 1000, hits)
println(ok ? "\n✅ HD159062 hgca_pm_residuals PASS" : "\n❌ hgca_pm_residuals MISSING")
