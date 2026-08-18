# Quantify nested's seed-dependence on the multimodal HD 159062 astrometry
# posterior. The tuning run showed dlogz is irrelevant and n_live helps only
# weakly, but seed 1 (0/4) vs seed 7 (3/4) swung wildly. Run a seed sweep at
# the documented default (n_live=1500, dlogz=0.1, :rslice) and report the
# recovery distribution: how often does a single nested run land on the
# orvara mode vs a wrong high-e mode? ptemcee recovers every time in ~55s.

using Nereus
using MCMCChains
using Statistics: quantile
using Printf

const RESULTS = joinpath(@__DIR__, "nested_seedsweep_HD159062_results.txt")
logline(s) = (open(RESULTS, "a") do io; println(io, s); end; println(s); flush(stdout))

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

make_params() = Params(
    max_kplanet = 1, planet_modes = [RVAS],
    instruments = InstrumentConfig(rv = ["HIRES"]),
    data = data, stability = :none, M_s = 0.81,
    parametrization = ParametrizationConfig(mass = :a_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"=>FixedPrior(1.0), "a_k1"=>LogUniformPrior(5.0, 1500.0),
        "M_sec_k1"=>LogUniformPrior(0.05, 1.5), "sesinw_k1"=>UniformPrior(-1.0, 1.0),
        "secosw_k1"=>UniformPrior(-1.0, 1.0), "Mo_k1"=>UniformPrior(0.0, 2π),
        "inc_k1"=>SinePrior(), "Omega_k1"=>UniformPrior(0.0, 2π),
        "sigma_HIRES"=>LogUniformPrior(1e-3, 1e3), "M_pri"=>NormalPrior(0.81, 0.04, 0.5, 1.2),
    ),
)
const REF = (a=58.0, e=0.11, i_deg=62.0)

nrec = 0; nclose = 0; ntot = 0
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
logline("# nested HD159062 seed sweep  (nlive=1500 dlogz=0.1 :rslice; ref a=58 e=0.11 i=62)")
for sd in seeds
    target = NereusTarget(make_params(), data)
    t0 = time()
    try
        chains, logZ = sample_nested(target, data; n_live=1500, dlogz=0.1,
                                      proposal=:rslice, seed=sd)
        a_v = vec(Array(chains[:, :a_k1, :]))
        se = vec(Array(chains[:, :sesinw_k1, :])); co = vec(Array(chains[:, :secosw_k1, :]))
        ic = vec(Array(chains[:, :inc_k1, :])); e = se.^2 .+ co.^2
        am, em, im = quantile(a_v,.5), quantile(e,.5), rad2deg(quantile(ic,.5))
        pa = abs(am-REF.a)<15; pe = abs(em-REF.e)<0.15; pi = abs(im-REF.i_deg)<8
        np = pa + pe + pi
        global ntot += 1
        global nrec += (np == 3); global nclose += (np >= 2)
        logline(@sprintf("seed=%-3d %4.0fs logZ=%-8.1f a=%.1f e=%.3f i=%.1f  %d/3  %s",
                         sd, time()-t0, logZ, am, em, im, np,
                         np==3 ? "✅" : np==2 ? "~" : "✗"))
    catch err
        logline(@sprintf("seed=%-3d %4.0fs FAILED: %s", sd, time()-t0,
                         sprint(showerror, err)))
    end
end
logline(@sprintf("# SUMMARY: %d/%d recovered (3/3), %d/%d within-1 (>=2/3)  [ptemcee: 1/1 in 55s]",
                 nrec, ntot, nclose, ntot))
println("\nDone.")
