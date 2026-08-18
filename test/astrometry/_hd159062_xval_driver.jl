using Nereus, MCMCChains, Statistics, Printf

const DATADIR = "/Users/jayvains/github/personal/Nereus/data/HD159062"

function build_target()
    hgca = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
    rvdat = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
    relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
    data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err, rv_inst=rvdat.rv_inst,
                relastrom=relast, hgca=hgca)
    params = Params(max_kplanet=1, planet_modes=[RVAS],
        instruments=InstrumentConfig(rv=["HIRES"]), data=data, stability=:none, M_s=0.81,
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
    return target, data
end

# Pull orbital quantities from a Chains object; returns (a, e, i_deg, M_sec) vectors
function extract_orbit(chains)
    nms = names(chains)
    getp(s) = vec(Array(chains[Symbol(s)]))
    a = getp("a_k1")
    ses = getp("sesinw_k1")
    sec = getp("secosw_k1")
    e = ses .^ 2 .+ sec .^ 2
    inc = getp("inc_k1")
    i_deg = rad2deg.(inc)
    M_sec = getp("M_sec_k1")
    return a, e, i_deg, M_sec
end

q16(x) = quantile(x, 0.16); q50(x) = quantile(x, 0.50); q84(x) = quantile(x, 0.84)

function report_block(tag, chains; can_diag)
    a, e, i_deg, M_sec = extract_orbit(chains)
    println("=== RESULT[$tag] ===")
    @printf("a_med %.6g\n", q50(a))
    @printf("a_lo %.6g\n", q16(a))
    @printf("a_hi %.6g\n", q84(a))
    @printf("e_med %.6g\n", q50(e))
    @printf("i_deg_med %.6g\n", q50(i_deg))
    @printf("M_sec_med %.6g\n", q50(M_sec))
    @printf("n_post_samples %d\n", length(a))
    if can_diag
        orb = [:a_k1, :sesinw_k1, :secosw_k1, :inc_k1, :Omega_k1, :M_sec_k1]
        present = [p for p in orb if p in names(chains)]
        sub = chains[present]
        rh = MCMCChains.rhat(sub)
        es = MCMCChains.ess(sub)
        rhvals = rh.nt.rhat
        esvals = es.nt.ess
        rhvals = filter(isfinite, rhvals)
        esvals = filter(isfinite, esvals)
        @printf("rhat_max %.6g\n", isempty(rhvals) ? NaN : maximum(rhvals))
        @printf("ess_min %.6g\n", isempty(esvals) ? NaN : minimum(esvals))
        # number of modes in a (crude): count via histogram peak separation
        println("a_unique_chainstd ", std(a))
    else
        println("rhat_max NA")
        println("ess_min NA")
    end
    println("=== ENDRESULT[$tag] ===")
end

target, data = build_target()

# ---------------- 1) WARM Pigeons reference ----------------
t0 = time()
println(">>> Running sample_pt_warm (Pigeons backend) ...")
chains_w, logZ_w, pf_w = sample_pt_warm(target; n_pathfinder_runs=16, n_rounds=14,
                                        n_chains=10, seed=42, show_report=false)
rt_w = (time() - t0) / 60
@printf("pigeons_logZ %.6g\n", logZ_w)
@printf("pigeons_runtime_min %.4g\n", rt_w)
# Pigeons collapses to a single output chain -> rhat/ess via MCMCChains not meaningful
report_block("PIGEONS", chains_w; can_diag=false)

# ---------------- 2) ptemcee ----------------
t1 = time()
println(">>> Running sample_ptemcee ...")
res = sample_ptemcee(target, data;
    n_temps=10, n_walkers=60, n_steps=6000, n_burnin=3000,
    adapt_ladder=true, init_strategy=:prior, pf_n_runs=16, seed=42,
    show_progress=false)
rt_p = (time() - t1) / 60
@printf("ptemcee_runtime_min %.4g\n", rt_p)
@printf("ptemcee_logZ %.6g\n", res.log_evidence)
report_block("PTEMCEE", res.chains; can_diag=true)
println(">>> DONE")
