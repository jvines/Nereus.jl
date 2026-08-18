#!/usr/bin/env julia
# HD 48265 (HIP 31895) — RV + HGCA joint fit, scored against Wu et al. 2026
# AJ 171, 189 (arXiv:2601.11280): two cold Jupiters,
#   b: P=789.6±1.1 d, e=0.35±0.02, Msini=1.67 Mjup, a=1.87 au
#      (i bimodal ~11.7° / ~160° → true mass ~7 Mjup at the face-on mode)
#   c: P=10418 (+2451/-1412) d ≈ 28.5 yr, e=0.42, M_true=4.09±0.2 Mjup,
#      i=89±29°, Ω=196±29°, a=10.4 au
# M* = 1.38 ± 0.06. CAUTION per the paper: HGCA acceleration is only 1.8σ.
# RVs: 129 points (MIKE + HARPS pre/post + PFS), Wu+2026 Table 3 MRT.
# Outputs: plots in test/astrometry/plots_HD48265/.

using Nereus
using MCMCChains
using DelimitedFiles: readdlm
using Statistics: median, quantile
using Printf

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD48265")
HGCAFIT = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062", "HGCA_vEDR3.fits")
OUTDIR  = joinpath(@__DIR__, "plots_HD48265")
mkpath(OUTDIR)

rr = readdlm(joinpath(DATADIR, "HD48265_RVs_Wu2026.csv"), ',', Any, '\n'; header=true)
raw = rr[1]; hdr = vec(rr[2]); cl(n) = findfirst(==(n), hdr)
bjd = Float64.(raw[:, cl("bjd")]); rv = Float64.(raw[:, cl("rv")]); rve = Float64.(raw[:, cl("rv_err")])
istr = String.(strip.(string.(raw[:, cl("instrument")])))
inames = sort(unique(istr)); i2i = Dict(n => i for (i, n) in enumerate(inames))
rinst = [i2i[s] for s in istr]
hgca = load_hgca_row(HGCAFIT, 31895)
# GOST scan forecast over the Gaia eDR3 window → HGCA "Mode B": the Gaia
# PM reflex is window-averaged over the actual scan epochs
# (gost_window_avg_pm) instead of evaluated instantaneously — the proper
# epoch-level treatment for a P ≈ 28 yr companion whose reflex curves
# within the Gaia window. (RA/Dec from SIMBAD for HD 48265 / HIP 31895.)
# GOST's queryable interval opens at 2014-07-25T10:31:26 TCB — an
# earlier `from` is REJECTED by the service, so start at 11:00.
gost = fetch_gost(100.00720, -48.54196;
                   from = "2014-07-25T11:00:00",
                   to   = "2017-05-28T00:00:00")
data = Data(t_rv=bjd, rv=rv, rv_err=rve, rv_inst=rinst, hgca=hgca, gost=gost)
@printf("Loaded: %d RVs (%s), HGCA HIP 31895, GOST %d scans (Mode B)\n",
        length(bjd), join(inames, ", "), n_gost(gost))

priors = Dict{String, PriorSpec}(
    "n_p" => FixedPrior(2.0),
    "M_pri" => NormalPrior(1.38, 0.06, 1.0, 1.8),
)
for k in 1:2
    priors["a_k$k"]      = LogUniformPrior(0.3, 60.0)
    priors["M_sec_k$k"]  = LogUniformPrior(3e-4, 0.1)   # ~0.3-100 Mjup
    priors["sesinw_k$k"] = UniformPrior(-1.0, 1.0)
    priors["secosw_k$k"] = UniformPrior(-1.0, 1.0)
    priors["Mo_k$k"]     = UniformPrior(0.0, 2π)
    priors["inc_k$k"]    = SinePrior()
    priors["Omega_k$k"]  = UniformPrior(0.0, 2π)
end
for n in inames
    priors["gamma_$n"] = UniformPrior(-200.0, 200.0)
    priors["sigma_$n"] = LogUniformPrior(1e-2, 50.0)
end
params = Params(
    max_kplanet = 2, planet_modes = [RVAS, RVAS],
    instruments = InstrumentConfig(rv = inames),
    data = data, stability = :none, M_s = 1.38,
    parametrization = ParametrizationConfig(mass = :a_driven),
    priors = priors,
)
target = NereusTarget(params, data)
@printf("Free params: %d\n", n_unfrozen(params))

N_TEMPS   = parse(Int, get(ENV, "NEREUS_NTEMPS",   "16"))
N_WALKERS = parse(Int, get(ENV, "NEREUS_NWALKERS", "150"))
N_STEPS   = parse(Int, get(ENV, "NEREUS_NSTEPS",   "16000"))
N_BURNIN  = parse(Int, get(ENV, "NEREUS_NBURNIN",  "7000"))
SEED      = parse(Int, get(ENV, "NEREUS_SEED",     "42"))
t0 = time()
res = sample_ptemcee(target, data; n_temps = N_TEMPS, n_walkers = N_WALKERS,
                      n_steps = N_STEPS, n_burnin = N_BURNIN, seed = SEED,
                      init_strategy = :prior, show_progress = false)
chains = res.chains
secs = time() - t0
save_chains(joinpath(OUTDIR, "chains.nc"), chains, params;
             data = data, log_evidence = res.log_evidence)

# score vs Wu+2026 (slot order = period order via the fixed-dim gate)
g(s) = vec(Array(chains[:, Symbol(s), :]))
for k in 1:2
    a_med = median(g("a_k$k"))
    e_med = median(g("sesinw_k$k").^2 .+ g("secosw_k$k").^2)
    i_med = rad2deg(median(g("inc_k$k")))
    M_med = median(g("M_sec_k$k")) * 1047.57   # Msun -> Mjup
    @printf("planet %d: a=%.2f au  e=%.3f  i=%.1f°  M=%.2f Mjup\n",
            k, a_med, e_med, i_med, M_med)
end
@printf("ptemcee %.0fs  logZ=%.1f\n", secs, res.log_evidence)
println("REF b: a=1.87 e=0.35 Msini=1.67Mjup (i~11.7 or ~160)")
println("REF c: a=10.4 e=0.42 M=4.09Mjup i=89 Ω=196")

for (fn, name) in [(plot_orbit_skyplane,      "orbit_skyplane"),
                   (plot_pm_residuals,        "pm_residuals"),
                   (plot_rv_astrom_phasefold, "rv_astrom_phasefold"),
                   (plot_pm_anomaly,          "pm_anomaly")]
    for k in 1:2
        try
            fn(chains, params, data; planet_idx = k, output = OUTDIR)
            println("saved $name K$k")
        catch err
            println("$name K$k FAILED: ", sprint(showerror, err))
        end
    end
end
println("plots in $OUTDIR")
try plot_posteriors_lp(chains, params; output=OUTDIR)
catch err; println("posteriors_lp FAILED: ", sprint(showerror, err)) end
