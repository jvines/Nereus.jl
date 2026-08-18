# Reproducibility run: HD 159062 ptemcee, seed=123.
# Same "long adaptive" config as the seed-42 reference, only seed differs.
# Tests run-to-run reproducibility of the recovered orbit.

using Nereus, MCMCChains, Statistics
using Printf

DATADIR = "/Users/jayvains/github/personal/Nereus/data/HD159062"
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

params = Params(max_kplanet=1, planet_modes=[RVAS],
    instruments=InstrumentConfig(rv=["HIRES"]), data=data, stability=:none, M_s=0.81,
    parametrization=ParametrizationConfig(mass=:a_driven),
    priors=Dict{String,PriorSpec}(
        "n_p"=>FixedPrior(1.0),
        "a_k1"=>LogUniformPrior(5.0,1500.0),
        "M_sec_k1"=>LogUniformPrior(0.05,1.5),
        "sesinw_k1"=>UniformPrior(-1.0,1.0),
        "secosw_k1"=>UniformPrior(-1.0,1.0),
        "Mo_k1"=>UniformPrior(0.0,2π),
        "inc_k1"=>SinePrior(),
        "Omega_k1"=>UniformPrior(0.0,2π),
        "sigma_HIRES"=>LogUniformPrior(1e-3,1e3),
        "M_pri"=>NormalPrior(0.81,0.04,0.5,1.2)))
target = NereusTarget(params, data)

println("Free params: $(n_unfrozen(params)) | threads: $(Threads.nthreads())")
flush(stdout)

t0 = time()
res = sample_ptemcee(target, data;
    n_temps      = 12,
    n_walkers    = 120,
    n_steps      = 8000,
    n_burnin     = 4000,
    adapt_ladder = true,
    init_strategy = :prior,
    pf_n_runs    = 2,
    seed         = 123,
    show_progress = false)
elapsed_min = (time() - t0) / 60
@printf("Done in %.2f min\n", elapsed_min)

ch = res.chains
# Orbital params for diagnostics
orb = [:a_k1, :M_sec_k1, :sesinw_k1, :secosw_k1, :inc_k1, :Omega_k1]
sub = ch[:, orb, :]

rh = rhat(sub)
es = ess(sub)
println("\n--- R-hat ---"); println(rh)
println("\n--- ESS ---"); println(es)

rhat_vals = rh[:, :rhat]
ess_vals  = es[:, :ess]
rhat_max = maximum(skipmissing(rhat_vals))
ess_min  = minimum(skipmissing(ess_vals))

# Flatten cold-chain draws
a_v   = vec(Array(ch[:, :a_k1, :]))
M_v   = vec(Array(ch[:, :M_sec_k1, :]))
ses_v = vec(Array(ch[:, :sesinw_k1, :]))
sec_v = vec(Array(ch[:, :secosw_k1, :]))
inc_v = vec(Array(ch[:, :inc_k1, :]))
e_v   = ses_v.^2 .+ sec_v.^2
i_deg = rad2deg.(inc_v)
n_samp = length(a_v)

q(v,p) = quantile(v, p)
@printf("\nn_post_samples = %d\n", n_samp)
@printf("a_med   = %.4f  CI=[%.4f, %.4f]\n", q(a_v,0.5), q(a_v,0.16), q(a_v,0.84))
@printf("e_med   = %.4f\n", q(e_v,0.5))
@printf("i_med   = %.4f deg\n", q(i_deg,0.5))
@printf("M_sec_med = %.4f\n", q(M_v,0.5))
@printf("rhat_max = %.4f | ess_min = %.2f\n", rhat_max, ess_min)
@printf("log_Z = %.3f\n", res.log_evidence)

# a-mode diagnostic: histogram of a in log space
println("\na-posterior histogram (log10 a bins):")
la = log10.(a_v)
edges = range(log10(5.0), log10(1500.0), length=21)
for k in 1:length(edges)-1
    cnt = count(x -> edges[k] <= x < edges[k+1], la)
    frac = cnt / n_samp
    bar = repeat("#", round(Int, frac*100))
    @printf("  [%.1f, %.1f) AU : %s %.3f\n", 10^edges[k], 10^edges[k+1], bar, frac)
end
@printf("\nelapsed_min = %.3f\n", elapsed_min)
