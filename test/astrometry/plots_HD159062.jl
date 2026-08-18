#!/usr/bin/env julia
# HD 159062 — validated RV+HGCA+relAST orbit (orvara/Brandt ground truth:
# a=58, e=0.11, i=62°, M_sec=0.608) rendered with the full astrometry plot
# suite. Sampler = the menu's recovering recipe (prior-init ptemcee, 55 s).
# Outputs: test/astrometry/plots_HD159062/*.png

using Nereus
using MCMCChains
using Statistics: median, quantile
using Printf

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
OUTDIR  = joinpath(@__DIR__, "plots_HD159062")
mkpath(OUTDIR)

hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
# GOST scans over the Gaia eDR3 window → HGCA Mode B (window-averaged Gaia
# PM reflex). RA/Dec from SIMBAD for HD 159062 / HIP 85653.
gost = fetch_gost(262.56845, 47.40219;
                   from = "2014-07-25T11:00:00",   # GOST interval opens 10:31:26 TCB
                   to   = "2017-05-28T00:00:00")
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca, gost=gost)
@printf("Loaded: %d RV, %d relAST, HGCA HIP 85653, GOST %d scans (Mode B)\n",
        length(rvdat.t), n_relast(relast), n_gost(gost))

params = Params(
    max_kplanet = 1, planet_modes = [RVAS],
    instruments = InstrumentConfig(rv = ["HIRES"]),
    data = data, stability = :none, M_s = 0.81,
    parametrization = ParametrizationConfig(mass = :a_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"        => FixedPrior(1.0),
        "a_k1"       => LogUniformPrior(5.0, 1500.0),
        "M_sec_k1"   => LogUniformPrior(0.05, 1.5),
        "sesinw_k1"  => UniformPrior(-1.0, 1.0),
        "secosw_k1"  => UniformPrior(-1.0, 1.0),
        "Mo_k1"      => UniformPrior(0.0, 2π),
        "inc_k1"     => SinePrior(),
        "Omega_k1"   => UniformPrior(0.0, 2π),
        "sigma_HIRES"=> LogUniformPrior(1e-3, 1e3),
        "M_pri"      => NormalPrior(0.81, 0.04, 0.5, 1.2),
    ),
)
target = NereusTarget(params, data)

t0 = time()
res = sample_ptemcee(target, data; n_temps = 16, n_walkers = 150,
                      n_steps = 12000, n_burnin = 5000,
                      init_strategy = :prior, show_progress = false)
chains = res.chains
a_med = median(vec(Array(chains[:, :a_k1, :])))
e_med = median(vec(Array(chains[:, :sesinw_k1, :])).^2 .+ vec(Array(chains[:, :secosw_k1, :])).^2)
i_med = rad2deg(median(vec(Array(chains[:, :inc_k1, :]))))
M_med = median(vec(Array(chains[:, :M_sec_k1, :])))
@printf("ptemcee %.0fs  logZ=%.1f | a=%.1f (58) e=%.3f (0.11) i=%.1f (62) M_sec=%.4f (0.608)\n",
        time()-t0, res.log_evidence, a_med, e_med, i_med, M_med)

for (fn, name) in [(plot_orbit_skyplane,      "orbit_skyplane"),
                   (plot_pm_residuals,        "pm_residuals"),
                   (plot_rv_astrom_phasefold, "rv_astrom_phasefold"),
                   (plot_relastrom_timeseries,"relastrom_timeseries"),
                   (plot_pm_anomaly,          "pm_anomaly")]
    try
        fn(chains, params, data; output = OUTDIR)
        println("saved $name.png")
    catch err
        println("$name FAILED: ", sprint(showerror, err))
    end
end
println("plots in $OUTDIR")
