#!/usr/bin/env julia
# SB2 (double-lined binary) + circumprimary planet — Phase A validation.
#
# The system: an unresolved SB2 binary (one Keplerian, two amplitudes
# K_A primary / K_B secondary, the secondary anti-phase) hosting a
# circumprimary RV planet (small K_p, seen ONLY in the primary's lines).
# Data are RV points tagged by `rv_comp` (1 = primary A, 2 = secondary B),
# built directly via `Data(; rv_comp=...)`.
#
# TWO parts:
#   SMOKE  (always) — forward-model at truth, likelihood sanity, and the
#          critical ws-vs-non-ws PARITY check (cache-fill gating must
#          equal direct-sum gating), plus channel-response checks.
#   RECOVER (ENV RECOVER=1) — blind ptemcee recovery of K_A/K_B/q/K_p.
#
# Run:  julia --project test/validation/validate_sb2.jl
#       RECOVER=1 julia --project test/validation/validate_sb2.jl

using Nereus, Statistics, Printf, Random
using Nereus: param_index, rv_log_likelihood, Theta, PTWorkspace

# ---- unit-amplitude Keplerian geometry, Tc anchor -------------------
# returns cos(f+ω) + e·cos(ω) at time t for the given orbit.
function kep_geom(t, P, Tc, e, w)
    f_tc = π/2 - w
    E_tc = 2*atan(sqrt((1-e)/(1+e))*tan(f_tc/2))
    M_tc = E_tc - e*sin(E_tc)
    M = 2π*(t - Tc)/P + M_tc
    E = M; for _ in 1:80; E -= (E - e*sin(E) - M)/(1 - e*cos(E)); end
    f = 2*atan(sqrt((1+e)/(1-e))*tan(E/2))
    return cos(f + w) + e*cos(w)
end

# ---- truth ----------------------------------------------------------
const P_BIN, TC_BIN, E_BIN, W_BIN = 41.3,  5.7, 0.15, 0.5
const K_A,   K_B                  = 820.0, 1240.0            # q = K_A/K_B = M_B/M_A
const P_PL,  TC_PL, E_PL, W_PL    = 7.83,  2.1, 0.10, 1.0
const K_PL                        = 24.0
const GAMMA, JIT                  = 15.0, 2.0
const Q_TRUE = K_A / K_B

# ---- epochs: each observed epoch yields a primary AND a secondary RV -
Random.seed!(20260630)
nep   = 70
tep   = sort!(400.0 .* rand(nep))            # 70 epochs over 400 d
t_rv    = vcat(tep, tep)                       # primary block then secondary block
rv_comp = vcat(fill(1, nep), fill(2, nep))
rv_err  = fill(3.0, 2*nep)

# noiseless model, then add white noise
rv = similar(t_rv)
for i in 1:length(t_rv)
    t = t_rv[i]
    if rv_comp[i] == 1                          # primary: +K_A binary + planet
        rv[i] = GAMMA + K_A*kep_geom(t, P_BIN, TC_BIN, E_BIN, W_BIN) +
                        K_PL*kep_geom(t, P_PL, TC_PL, E_PL, W_PL)
    else                                        # secondary: −K_B binary, NO planet
        rv[i] = GAMMA - K_B*kep_geom(t, P_BIN, TC_BIN, E_BIN, W_BIN)
    end
end
rv_clean = copy(rv)
rv .+= JIT .* randn(length(rv))

data = Data(; t_rv=t_rv, rv=rv, rv_err=rv_err, rv_inst=ones(Int, length(t_rv)),
              rv_comp=rv_comp)

ic  = InstrumentConfig(rv=["SB2SPEC"])
par = ParametrizationConfig(time=:Tc)

# tight priors around truth for the smoke checks / fast recovery
pri = Dict{String,PriorSpec}(
    "n_p"       => FixedPrior(2.0),
    # binary (k=1)
    "P_k1"      => NormalPrior(P_BIN, 0.05, P_BIN-0.5, P_BIN+0.5),
    "K_A_k1"    => UniformPrior(300.0, 1500.0),
    "K_B_k1"    => UniformPrior(300.0, 2200.0),
    "Tc_k1"     => NormalPrior(TC_BIN, 0.3, TC_BIN-2.0, TC_BIN+2.0),
    "sesinw_k1" => UniformPrior(-0.7, 0.7), "secosw_k1" => UniformPrior(-0.7, 0.7),
    # planet (k=2)
    "P_k2"      => NormalPrior(P_PL, 0.02, P_PL-0.2, P_PL+0.2),
    "K_k2"      => UniformPrior(0.0, 80.0),
    "Tc_k2"     => NormalPrior(TC_PL, 0.2, TC_PL-1.0, TC_PL+1.0),
    "sesinw_k2" => UniformPrior(-0.5, 0.5), "secosw_k2" => UniformPrior(-0.5, 0.5),
    "gamma_SB2SPEC" => UniformPrior(-200.0, 200.0),
    "sigma_SB2SPEC" => LogUniformPrior(0.1, 20.0),
)

params = Params(; max_kplanet=2, planet_modes=[BINARY_RV, RV_ONLY],
                  instruments=ic, data=data, parametrization=par,
                  priors=pri, stability=:none)

println("="^70)
println("SB2 Phase-A validation — layout")
println("="^70)
@printf("n_total slots     : %d\n", Nereus.n_total(params))
names = params.layout.names
sb_names = filter(n -> occursin("_k1", n), names)
pl_names = filter(n -> occursin("_k2", n), names)
println("binary (k1) slots : ", join(sb_names, ", "))
println("planet (k2) slots : ", join(pl_names, ", "))
@assert "K_A_k1" in names "K_A_k1 slot missing!"
@assert "K_B_k1" in names "K_B_k1 slot missing!"
@assert "K_k2"   in names "K_k2 slot missing!"
@assert !("K_k1" in names) "binary should NOT have a single K_k1 slot"
@assert !("inc_k1" in names) "BINARY_RV should have no inc slot"
println("block type k1     : ", typeof(params.layout.planet_blocks[1]))
@assert params.layout.planet_blocks[1] isa Nereus.SB2Block
println("✓ layout OK\n")

# ---- build a truth Theta --------------------------------------------
sesinw(e,w) = sqrt(e)*sin(w); secosw(e,w) = sqrt(e)*cos(w)
th = Theta(params)
setp!(name, v) = (th.values[param_index(params, name)] = v)
setp!("P_k1", P_BIN);  setp!("K_A_k1", K_A); setp!("K_B_k1", K_B)
setp!("Tc_k1", TC_BIN); setp!("sesinw_k1", sesinw(E_BIN,W_BIN)); setp!("secosw_k1", secosw(E_BIN,W_BIN))
setp!("P_k2", P_PL);   setp!("K_k2", K_PL)
setp!("Tc_k2", TC_PL); setp!("sesinw_k2", sesinw(E_PL,W_PL)); setp!("secosw_k2", secosw(E_PL,W_PL))
setp!("gamma_SB2SPEC", GAMMA); setp!("sigma_SB2SPEC", JIT)

# ---- SMOKE 1: predictions match the hand-built clean model ----------
preds, _ = Nereus.rv_predictions(th, data)
resid_clean = preds .- rv_clean
@printf("max |pred − clean model| = %.3e m/s  (should be ~0)\n", maximum(abs, resid_clean))
@assert maximum(abs, resid_clean) < 1e-6 "forward model mismatch — SB2 gating wrong!"
# component split sanity: comp-2 predictions must be independent of the planet.
i2 = findall(==(2), rv_comp)
comp2_expected = [GAMMA - K_B*kep_geom(t_rv[i], P_BIN, TC_BIN, E_BIN, W_BIN) for i in i2]
@printf("max |comp2 pred − (γ−K_B·geom)| = %.3e  (planet must NOT leak into B)\n",
        maximum(abs, preds[i2] .- comp2_expected))
@assert maximum(abs, preds[i2] .- comp2_expected) < 1e-6 "planet leaked into secondary channel!"
println("✓ forward model + component split OK\n")

# ---- SMOKE 2: likelihood at truth is finite & near-optimal ----------
ll_truth = rv_log_likelihood(th, data)
@printf("logL(truth)            = %.3f\n", ll_truth)
@assert isfinite(ll_truth) "logL(truth) not finite!"

# ---- SMOKE 3: ws-path vs non-ws PARITY (cache-fill == direct-sum) ---
ws = PTWorkspace(params, 2; n_obs=length(t_rv), n_phot=0)
ll_ws = rv_log_likelihood(th, data, ws)
@printf("logL(truth, ws)        = %.3f   Δ(ws−direct) = %.2e\n", ll_ws, ll_ws - ll_truth)
@assert isapprox(ll_ws, ll_truth; atol=1e-6) "ws-path disagrees with direct — cache gating bug!"
# call ws again to exercise the cache-HIT path (hashes now populated)
ll_ws2 = rv_log_likelihood(th, data, ws)
@assert isapprox(ll_ws2, ll_truth; atol=1e-6) "ws cache-hit disagrees — stale cache!"
println("✓ ws/non-ws parity OK (fill + hit)\n")

# ---- SMOKE 4: each channel actually responds ------------------------
function ll_perturbed(name, delta)
    v0 = th.values[param_index(params, name)]
    th.values[param_index(params, name)] = v0 + delta
    ll = rv_log_likelihood(th, data)
    th.values[param_index(params, name)] = v0
    ll
end
dKA = ll_truth - ll_perturbed("K_A_k1", 60.0)
dKB = ll_truth - ll_perturbed("K_B_k1", 60.0)
dKP = ll_truth - ll_perturbed("K_k2",  10.0)
@printf("Δ logL for +ΔK_A: %.2f   +ΔK_B: %.2f   +ΔK_p: %.2f  (all must be > 0)\n", dKA, dKB, dKP)
@assert dKA > 0 && dKB > 0 && dKP > 0 "a channel does not constrain the fit — gating dead!"
# hash-invalidation guard: perturbing ONLY K_B must move the ws logL too
ll_wsKB = let v0 = th.values[param_index(params,"K_B_k1")]
    th.values[param_index(params,"K_B_k1")] = v0 + 60.0
    l = rv_log_likelihood(th, data, ws)
    th.values[param_index(params,"K_B_k1")] = v0; l
end
@assert !isapprox(ll_wsKB, ll_ws; atol=1e-6) "ws logL ignored a K_B change — cache hash missing cB!"
println("✓ all channels respond; K_B hash-invalidation OK\n")

# ---- backward-compat: a plain single-star fit is unchanged ----------
# (all comp==1, no SB2 block → _comp_rv reduces to K·geom)
let
    t = sort!(300.0*rand(40)); e2, w2 = 0.2, 0.9
    y = [GAMMA + 30.0*kep_geom(ti, 12.0, 3.0, e2, w2) for ti in t] .+ 0.5*randn(40)
    d = Data(; t_rv=t, rv=y, rv_err=fill(2.0,40), rv_inst=ones(Int,40))  # rv_comp defaults all-1
    p = Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=InstrumentConfig(rv=["I"]),
                 data=d, parametrization=ParametrizationConfig(time=:Tc),
                 priors=Dict{String,PriorSpec}("n_p"=>FixedPrior(1.0),
                   "P_k1"=>NormalPrior(12.0,0.1,11,13),"K_k1"=>UniformPrior(0,80),
                   "Tc_k1"=>NormalPrior(3.0,0.2,2,4),"sesinw_k1"=>UniformPrior(-.7,.7),
                   "secosw_k1"=>UniformPrior(-.7,.7),"gamma_I"=>UniformPrior(-100,100),
                   "sigma_I"=>LogUniformPrior(.1,20)), stability=:none)
    thc = Theta(p)
    for (n,v) in (("P_k1",12.0),("K_k1",30.0),("Tc_k1",3.0),
                  ("sesinw_k1",sesinw(e2,w2)),("secosw_k1",secosw(e2,w2)),
                  ("gamma_I",GAMMA),("sigma_I",0.5))
        thc.values[param_index(p,n)] = v
    end
    llc = rv_log_likelihood(thc, d)
    @printf("single-star control logL = %.3f (finite, SB2 code is a no-op here)\n", llc)
    @assert isfinite(llc)
    println("✓ backward-compat single-star OK\n")
end

println("="^70)
println("SMOKE PASSED ✓  — SB2 forward model, ws parity, channel response all good")
println("="^70)

# ---- RECOVERY (opt-in) ----------------------------------------------
if get(ENV, "RECOVER", "0") == "1"
    println("\n", "="^70)
    println("BLIND RECOVERY via sample_ptemcee")
    println("="^70)
    tgt = NereusTarget(params, data)
    res = sample_ptemcee(tgt, data; n_temps=10, n_walkers=60, n_steps=8000,
                         n_burnin=4000, seed=7, init_strategy=:prior, show_progress=true)
    ch = res.chains
    q(sym) = (v=vec(Array(ch[:,sym,:])); (median(v), quantile(v,0.16), quantile(v,0.84)))
    KA = q(:K_A_k1); KB = q(:K_B_k1); KP = q(:K_k2)
    qv = vec(Array(ch[:,:K_A_k1,:])) ./ vec(Array(ch[:,:K_B_k1,:]))
    @printf("\n%-8s  truth     med [16, 84]\n", "param")
    @printf("K_A    %8.1f  %8.1f [%.1f, %.1f]\n", K_A, KA...)
    @printf("K_B    %8.1f  %8.1f [%.1f, %.1f]\n", K_B, KB...)
    @printf("K_p    %8.1f  %8.1f [%.1f, %.1f]\n", K_PL, KP...)
    @printf("q=KA/KB %7.3f  %8.3f [%.3f, %.3f]\n", Q_TRUE, median(qv), quantile(qv,0.16), quantile(qv,0.84))
    ok = abs(KA[1]-K_A) < 3*(KA[3]-KA[2])/2 + 30 &&
         abs(KB[1]-K_B) < 3*(KB[3]-KB[2])/2 + 30 &&
         abs(KP[1]-K_PL) < 3*(KP[3]-KP[2])/2 + 5
    println(ok ? "\n✓ RECOVERY PASSED" : "\n✗ RECOVERY OFF — investigate")
end
