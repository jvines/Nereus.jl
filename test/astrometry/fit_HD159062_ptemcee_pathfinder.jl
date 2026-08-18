using Nereus, MCMCChains, Statistics, Serialization

DATADIR = "/Users/jayvains/github/personal/Nereus/data/HD159062"

hgca = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err, rv_inst=rvdat.rv_inst,
            relastrom=relast, hgca=hgca)
params = Params(max_kplanet=1, planet_modes=[RVAS],
    instruments=InstrumentConfig(rv=["HIRES"]), data=data, stability=:none, M_s=0.81,
    parametrization=ParametrizationConfig(mass=:a_driven),
    priors=Dict{String,PriorSpec}(
        "n_p" => FixedPrior(1.0),
        "a_k1" => LogUniformPrior(5.0, 1500.0),
        "M_sec_k1" => LogUniformPrior(0.05, 1.5),
        "sesinw_k1" => UniformPrior(-1.0, 1.0),
        "secosw_k1" => UniformPrior(-1.0, 1.0),
        "Mo_k1" => UniformPrior(0.0, 2π),
        "inc_k1" => SinePrior(),
        "Omega_k1" => UniformPrior(0.0, 2π),
        "sigma_HIRES" => LogUniformPrior(1e-3, 1e3),
        "M_pri" => NormalPrior(0.81, 0.04, 0.5, 1.2)))
target = NereusTarget(params, data)

CACHE = "/tmp/hd159062_ptemcee_pf.jls"
if isfile(CACHE)
    res, runtime_min = deserialize(CACHE)
    println("LOADED_CACHE")
else
    t0 = time()
    res = sample_ptemcee(target, data;
        n_temps=16, n_walkers=44, n_steps=6000, n_burnin=3000,
        adapt_ladder=true, init_strategy=:pathfinder, pf_n_runs=4, seed=42,
        show_progress=false)
    runtime_min = (time() - t0) / 60
    serialize(CACHE, (res, runtime_min))
end

ch = res.chains
orb_params = ["a_k1", "M_sec_k1", "sesinw_k1", "secosw_k1", "inc_k1", "Omega_k1", "Mo_k1"]
present = intersect(orb_params, string.(names(ch)))
println("PARAM_NAMES: ", names(ch))
println("PRESENT_ORB: ", present)

sub = ch[:, Symbol.(present), :]

# R-hat and ESS over orbital params (ChainDataFrame -> .nt named tuple)
rh_cdf = MCMCChains.rhat(sub)
es_cdf = MCMCChains.ess(sub)
println("RHAT_NT_KEYS: ", keys(rh_cdf.nt))
println("ESS_NT_KEYS: ", keys(es_cdf.nt))
rhat_vals = rh_cdf.nt.rhat
ess_vals = es_cdf.nt.ess
println("RHAT_PER_PARAM: ", collect(zip(rh_cdf.nt.parameters, rhat_vals)))
println("ESS_PER_PARAM: ", collect(zip(es_cdf.nt.parameters, ess_vals)))

# Build derived posterior arrays (flatten walkers+steps)
function vec_of(sym)
    vec(Array(ch[:, sym, :]))
end
a = vec_of(:a_k1)
Msec = vec_of(:M_sec_k1)
ses = vec_of(:sesinw_k1)
sec = vec_of(:secosw_k1)
inc = vec_of(:inc_k1)
ecc = ses .^ 2 .+ sec .^ 2
inc_deg = rad2deg.(inc)
# fold inc into [0,90] convention? report median of inc_deg directly but also folded
inc_deg_folded = [d > 90 ? 180 - d : d for d in inc_deg]

q(x) = quantile(x, [0.16, 0.5, 0.84])

println("N_POST_SAMPLES: ", length(a))
println("A_QUANT: ", q(a))
println("E_QUANT: ", q(ecc))
println("INC_DEG_QUANT: ", q(inc_deg))
println("INC_DEG_FOLDED_QUANT: ", q(inc_deg_folded))
println("MSEC_QUANT: ", q(Msec))

rhat_max = maximum(skipmissing(rhat_vals))
ess_min = minimum(skipmissing(ess_vals))
println("RHAT_MAX: ", rhat_max)
println("ESS_MIN: ", ess_min)
println("RUNTIME_MIN: ", runtime_min)
println("LOG_EVIDENCE: ", res.log_evidence)
println("ACC_WITHIN: ", res.acceptance_within)
println("ACC_SWAP: ", res.acceptance_swap)
println("BETAS: ", res.betas)

# multimodality check on a: histogram of a in plausible band vs spread
println("A_MIN_MAX: ", (minimum(a), maximum(a)))
# fraction of a within reference band
frac_band = count(x -> 50 <= x <= 66, a) / length(a)
println("A_FRAC_IN_50_66: ", frac_band)
