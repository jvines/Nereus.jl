# HD 159062 with OFTI rejection sampling. Should explore both
# (e=0.1, i=63°) and (e=0.75, i=32°) modes if both are real
# high-evidence basins of the posterior.

using Nereus
using MCMCChains
using Statistics: median, quantile
using Printf

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

const M_PRI = 0.81

params = Params(
    max_kplanet = 1, planet_modes = [RVAS],
    instruments = InstrumentConfig(rv = ["HIRES"]),
    data = data, stability = :none, M_s = M_PRI,
    parametrization = ParametrizationConfig(mass = :a_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"        => FixedPrior(1.0),
        # WIDE prior on a — OFTI scales it heavily
        "a_k1"       => LogUniformPrior(1.0, 5000.0),
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
println("=" ^ 70)
println("HD 159062 OFTI sampling")
println("=" ^ 70)

target = NereusTarget(params, data)
t0 = time()
chains = ofti_sample(target;
                     n_attempts = 10_000_000,
                     planet_idx = 1,
                     epoch_idx  = 1,
                     seed       = 42,
                     show_progress = true)
println(@sprintf("Done in %.1f min, %d accepted samples", (time()-t0)/60, length(chains)))
println()

a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
Mp_v  = vec(Array(chains[:, :M_pri, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
ses_v = vec(Array(chains[:, :sesinw_k1, :]))
sec_v = vec(Array(chains[:, :secosw_k1, :]))
Ω_v   = vec(Array(chains[:, :Omega_k1, :]))
e_arr = ses_v.^2 .+ sec_v.^2
P_yr  = [sqrt(a_v[j]^3 / (Mp_v[j] + M_sec[j])) for j in 1:length(a_v)]

println("OFTI posterior (no thinning, no burnin — pure rejection):")
for (k, v) in pairs((a_au=a_v, P_yr=P_yr, M_pri=Mp_v, M_sec=M_sec,
                     e=e_arr, i_deg=rad2deg.(inc_v), Ω_deg=rad2deg.(Ω_v)))
    q16, q50, q84 = quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
    @printf("  %-10s = %12.4f  [+%.4f, -%.4f]\n",
            string(k), q50, q84-q50, q50-q16)
end
println()

# Mode breakdown
n = length(M_sec)
n_loweccen = sum(e_arr .< 0.4)
n_higheccen = sum(e_arr .>= 0.4)
@printf("Mode breakdown: low-e (e<0.4): %d (%.1f%%);  high-e (e≥0.4): %d (%.1f%%)\n",
        n_loweccen, 100*n_loweccen/n, n_higheccen, 100*n_higheccen/n)
n_highinc = sum(rad2deg.(inc_v) .> 50)
n_lowinc  = sum(rad2deg.(inc_v) .<= 50)
@printf("                low-i (i≤50°): %d (%.1f%%);  high-i (i>50°): %d (%.1f%%)\n",
        n_lowinc, 100*n_lowinc/n, n_highinc, 100*n_highinc/n)
println()

println("Brandt+ 2021 Table 5: M_sec=0.608+0.008/-0.007, a=61.9±7, P=411±70, e=0.10+0.11/-0.07, i=63.0+1.8/-2.4")
M_med = quantile(M_sec, 0.50); M_sig = (quantile(M_sec, 0.84) - quantile(M_sec, 0.16))/2
M_diff = abs(M_med - 0.608) / sqrt(M_sig^2 + 0.008^2)
@printf("M_companion: %.4f ± %.4f M_sun → %.2fσ from Brandt\n", M_med, M_sig, M_diff)
