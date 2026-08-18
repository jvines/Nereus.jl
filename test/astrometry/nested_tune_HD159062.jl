# Is nested's HD 159062 failure TUNING (premature dlogz=0.5 termination) or
# STRUCTURAL (ellipsoidal bound can't find the dominant mode)?
#
# The recovery menu ran nested at dlogz=0.5, but sample_nested's documented
# default is dlogz=0.1 (looser → premature stop → much lower logZ, per the
# docstring). ptemcee recovered (a=58/e=0.11/i=62, logZ=+178.5); nested at
# dlogz=0.5 finished in 31s with logZ=-340.6 in a wrong high-e mode. If the
# DEFAULT dlogz=0.1 (or tighter / more live points) recovers, the menu cell
# was crippled by a non-default knob, not by ellipsoidal structure.

using Nereus
using MCMCChains
using Statistics: quantile
using Printf

const RESULTS = joinpath(@__DIR__, "nested_tune_HD159062_results.txt")
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
const REF = (a=58.0, e=0.11, i_deg=62.0, M_sec=0.608)

function score(tag, chains, logZ, secs)
    a_v = vec(Array(chains[:, :a_k1, :])); M = vec(Array(chains[:, :M_sec_k1, :]))
    se = vec(Array(chains[:, :sesinw_k1, :])); co = vec(Array(chains[:, :secosw_k1, :]))
    ic = vec(Array(chains[:, :inc_k1, :])); e = se.^2 .+ co.^2
    am, em = quantile(a_v, .5), quantile(e, .5)
    im, Mm = rad2deg(quantile(ic, .5)), quantile(M, .5)
    ok = abs(am-REF.a)<15 && abs(em-REF.e)<0.15 && abs(im-REF.i_deg)<8
    logline(@sprintf("%-26s %5.0fs logZ=%-9.1f a=%.1f e=%.3f i=%.1f M=%.4f  %s",
                     tag, secs, logZ, am, em, im, Mm,
                     ok ? "✅ RECOVERED" : "⚠ MISMATCH"))
end

# (n_live, dlogz, proposal, seed)
configs = [
    (1500, 0.1,  :rslice, 1),   # documented default — the key test
    (1500, 0.05, :rslice, 1),   # tighter stop
    (3000, 0.1,  :rslice, 1),   # more live points
    (1500, 0.1,  :rslice, 7),   # default, different seed (structural check)
]
logline("# nested HD159062 tuning  (ref a=58 e=0.11 i=62 M=0.608; menu used dlogz=0.5 → MISMATCH)")
for (nl, dz, prop, sd) in configs
    target = NereusTarget(make_params(), data)
    tag = @sprintf("nlive=%d dlogz=%.2f s%d", nl, dz, sd)
    println("\n--- $tag ---")
    t0 = time()
    try
        chains, logZ = sample_nested(target, data; n_live=nl, dlogz=dz,
                                      proposal=prop, seed=sd)
        score(tag, chains, logZ, time()-t0)
    catch e
        logline(@sprintf("%-26s %5.0fs FAILED: %s", tag, time()-t0,
                         sprint(showerror, e)))
    end
end
println("\nDone.")
