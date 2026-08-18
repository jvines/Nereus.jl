DATADIR = "/Users/jayvains/github/personal/Nereus/data/HD159062"
using Nereus, MCMCChains, Statistics, Printf

hgca = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err, rv_inst=rvdat.rv_inst,
            relastrom=relast, hgca=hgca)
params = Params(max_kplanet=1, planet_modes=[RVAS], instruments=InstrumentConfig(rv=["HIRES"]),
    data=data, stability=:none, M_s=0.81,
    parametrization=ParametrizationConfig(mass=:a_driven),
    priors=Dict{String,PriorSpec}(
        "n_p"=>FixedPrior(1.0),
        "a_k1"=>LogUniformPrior(5.0, 1500.0),
        "M_sec_k1"=>LogUniformPrior(0.05, 1.5),
        "sesinw_k1"=>UniformPrior(-1.0, 1.0),
        "secosw_k1"=>UniformPrior(-1.0, 1.0),
        "Mo_k1"=>UniformPrior(0.0, 2π),
        "inc_k1"=>SinePrior(),
        "Omega_k1"=>UniformPrior(0.0, 2π),
        "sigma_HIRES"=>LogUniformPrior(1e-3, 1e3),
        "M_pri"=>NormalPrior(0.81, 0.04, 0.5, 1.2)))
target = NereusTarget(params, data)

t0 = time()
res = sample_ptemcee(target, data; n_temps=16, n_walkers=44, n_steps=8000,
    n_burnin=4000, adapt_ladder=true, init_strategy=:prior, seed=42, show_progress=false)
runtime_min = (time() - t0) / 60

chn = res.chains
nms = names(chn)
println("CHAIN PARAM NAMES: ", nms)
println("N_SAMPLES_TOTAL: ", length(chn) * size(chn, 3))  # iters * chains(walkers)

# extract arrays
function getp(sym)
    Array(chn[sym])  # iters x walkers
end

a   = getp(:a_k1)
ses = getp(:sesinw_k1)
sec = getp(:secosw_k1)
inc = getp(:inc_k1)
Om  = getp(:Omega_k1)
Ms  = getp(:M_sec_k1)
e   = ses .^ 2 .+ sec .^ 2

a_med = median(vec(a))
e_med = median(vec(e))
i_deg_med = rad2deg(median(vec(inc)))
M_sec_med = median(vec(Ms))

a_q = quantile(vec(a), [0.16, 0.84])

@printf("a_med=%.4f  a_ci=[%.4f, %.4f]\n", a_med, a_q[1], a_q[2])
@printf("e_med=%.4f\n", e_med)
@printf("i_deg_med=%.4f\n", i_deg_med)
@printf("M_sec_med=%.4f\n", M_sec_med)

# R-hat / ESS over orbital params: a, e(via ses/sec), i, Omega, M_sec
# MCMCChains needs raw chain params. Build a sub-chain with derived e by
# adding ses^2+sec^2 as a new param would require reconstruction; instead
# compute rhat/ess on raw params a_k1, sesinw_k1, secosw_k1, inc_k1, Omega_k1, M_sec_k1.
diag_params = [:a_k1, :sesinw_k1, :secosw_k1, :inc_k1, :Omega_k1, :M_sec_k1]
sub = chn[diag_params]
rh = rhat(sub)
es = ess(sub)
println("RHAT TABLE:")
println(rh)
println("ESS TABLE:")
println(es)

rhat_vals = rh[:, :rhat]
ess_vals = es[:, :ess]
rhat_max = maximum(filter(isfinite, rhat_vals))
ess_min = minimum(filter(isfinite, ess_vals))
@printf("RHAT_MAX=%.4f\n", rhat_max)
@printf("ESS_MIN=%.2f\n", ess_min)
@printf("RUNTIME_MIN=%.3f\n", runtime_min)
@printf("LOGZ=%.3f\n", res.log_evidence)
println("ACC_WITHIN: ", res.acceptance_within)
println("ACC_SWAP: ", res.acceptance_swap)

# multimodality check on a
amin, amax = extrema(vec(a))
@printf("A_RANGE=[%.3f, %.3f]\n", amin, amax)
# fraction of a-samples within reference band [50,66]
frac_ref = count(x -> 50 <= x <= 66, vec(a)) / length(vec(a))
@printf("A_FRAC_IN_REFBAND=%.4f\n", frac_ref)
# crude bimodality: split at median, report mode density
println("N_POST_SAMPLES=", length(vec(a)))
