#!/usr/bin/env julia
# SB2 binary + circumprimary TRANSITING planet — Phase B (transit dilution).
#
# The binary (BINARY_RV, invisible to photometry: has_geometry=false) hosts a
# transiting planet on star A (RVPM). Star B's constant light DILUTES the
# transit: observed = (1−D)·transit + D. The dilution D is the transit-band
# luminosity fraction L_B/(L_A+L_B) — a per-PM-instrument param, auto-freed to
# Uniform when an SB2 binary is present (else transit depth biases low).
#
# SMOKE (always): dilution prior auto-freed, joint logL at truth finite,
# dilution is LIVE (perturbing D moves the transit logL), depth actually
# diluted, ws/non-ws parity. RECOVER=1: recover K_A/K_B/rr/D.
#
# Run:  julia --project test/validation/validate_sb2_transit.jl

using Nereus, Statistics, Printf, Random
using Nereus: param_index, rv_log_likelihood, transit_log_likelihood, Theta,
               PTWorkspace, compute_transit_model_on_grid

function kep_geom(t, P, Tc, e, w)
    f_tc = π/2 - w
    E_tc = 2*atan(sqrt((1-e)/(1+e))*tan(f_tc/2)); M_tc = E_tc - e*sin(E_tc)
    M = 2π*(t - Tc)/P + M_tc
    E = M; for _ in 1:80; E -= (E - e*sin(E) - M)/(1 - e*cos(E)); end
    f = 2*atan(sqrt((1+e)/(1-e))*tan(E/2))
    return cos(f + w) + e*cos(w)
end

# ---- truth ----------------------------------------------------------
const P_BIN, TC_BIN, E_BIN, W_BIN = 41.3, 5.7, 0.15, 0.5
const K_A, K_B                    = 820.0, 1240.0
const P_PL, TC_PL, E_PL, W_PL     = 7.83, 2.10, 0.0, 0.0
const RR_PL, B_PL                 = 0.11, 0.30            # Rp/Rs, impact param
const Q1, Q2                      = 0.40, 0.30            # Kipping LD
const DIL                         = 0.35                  # third light L_B/(L_A+L_B), TESS band
const GAMMA, JIT                  = 15.0, 2.0
const M_S, R_S                    = 1.00, 1.00

Random.seed!(20260630)

# ---- RV epochs (primary + secondary), as in Phase A -----------------
nep = 60; tep = sort!(300.0 .* rand(nep))
t_rv = vcat(tep, tep); rv_comp = vcat(fill(1,nep), fill(2,nep)); rv_err = fill(3.0, 2nep)

# ---- photometry: fine cadence around 4 transits + sparse baseline ---
cad = 2.0/(60*24)                       # 2-min cadence
t_phot = Float64[]
for n in 0:3
    tc = TC_PL + n*P_PL
    append!(t_phot, tc-0.28 : cad : tc+0.28)
end
append!(t_phot, 0.0:0.5:32.0)           # sparse baseline
sort!(unique!(t_phot))
nph = length(t_phot); flux_err = fill(3.0e-4, nph)

# provisional data (flux filled after truth theta is built)
data0 = Data(; t_rv=t_rv, rv=zeros(2nep), rv_err=rv_err, rv_inst=ones(Int,2nep),
               rv_comp=rv_comp, t_phot=t_phot, flux=ones(nph), flux_err=flux_err,
               phot_inst=ones(Int,nph))

ic  = InstrumentConfig(rv=["SB2SPEC"], pm=["TESS"])
par = ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=false)
pri = Dict{String,PriorSpec}(
    "n_p"=>FixedPrior(2.0),
    "P_k1"=>NormalPrior(P_BIN,0.05,P_BIN-0.5,P_BIN+0.5),
    "K_A_k1"=>UniformPrior(300.0,1500.0), "K_B_k1"=>UniformPrior(300.0,2200.0),
    "Tc_k1"=>NormalPrior(TC_BIN,0.3,TC_BIN-2,TC_BIN+2),
    "sesinw_k1"=>UniformPrior(-0.7,0.7), "secosw_k1"=>UniformPrior(-0.7,0.7),
    "P_k2"=>NormalPrior(P_PL,0.005,P_PL-0.05,P_PL+0.05),
    "K_k2"=>UniformPrior(0.0,80.0),
    "Tc_k2"=>NormalPrior(TC_PL,0.05,TC_PL-0.3,TC_PL+0.3),
    "sesinw_k2"=>UniformPrior(-0.3,0.3), "secosw_k2"=>UniformPrior(-0.3,0.3),
    "rr_k2"=>UniformPrior(0.02,0.25), "b_k2"=>UniformPrior(0.0,0.95),
    "q1_TESS"=>UniformPrior(0.0,1.0), "q2_TESS"=>UniformPrior(0.0,1.0),
    "gamma_SB2SPEC"=>UniformPrior(-200.0,200.0), "sigma_SB2SPEC"=>LogUniformPrior(0.1,20.0),
    "offset_TESS"=>NormalPrior(0.0,0.005,-1.0,1.0), "jitter_TESS"=>ModJeffreysPrior(1e-5,1e-2),
    # dilution_TESS intentionally omitted → exercises the auto-default
)
params = Params(; max_kplanet=2, planet_modes=[BINARY_RV, RVPM], instruments=ic,
                  data=data0, parametrization=par, priors=pri, stability=:none,
                  M_s=M_S, R_s=R_S)

println("="^70); println("SB2 + transiting planet — Phase B (dilution)"); println("="^70)
names = params.layout.names
@printf("block k1=%s  k2=%s\n", typeof(params.layout.planet_blocks[1]),
        typeof(params.layout.planet_blocks[2]))
@assert params.layout.planet_blocks[1] isa Nereus.SB2Block
@assert !(params.layout.planet_blocks[2] isa Nereus.SB2Block)
@assert "rr_k2" in names && "b_k2" in names "transiting planet geometry missing"
# dilution auto-freed because has_sb
dil_prior = params.config.priors["dilution_TESS"]
@printf("dilution_TESS default prior : %s\n", typeof(dil_prior))
@assert occursin("Uniform", string(typeof(dil_prior))) "dilution NOT auto-freed for SB2 — depth will bias!"
println("✓ layout OK; dilution auto-freed\n")

# ---- truth theta ----------------------------------------------------
sesinw(e,w)=sqrt(e)*sin(w); secosw(e,w)=sqrt(e)*cos(w)
th = Theta(params); setp!(n,v)=(th.values[param_index(params,n)]=v)
setp!("P_k1",P_BIN); setp!("K_A_k1",K_A); setp!("K_B_k1",K_B); setp!("Tc_k1",TC_BIN)
setp!("sesinw_k1",sesinw(E_BIN,W_BIN)); setp!("secosw_k1",secosw(E_BIN,W_BIN))
setp!("P_k2",P_PL); setp!("K_k2",24.0); setp!("Tc_k2",TC_PL)
setp!("sesinw_k2",sesinw(E_PL,W_PL)); setp!("secosw_k2",secosw(E_PL,W_PL))
setp!("rr_k2",RR_PL); setp!("b_k2",B_PL); setp!("q1_TESS",Q1); setp!("q2_TESS",Q2)
setp!("gamma_SB2SPEC",GAMMA); setp!("sigma_SB2SPEC",JIT)
setp!("offset_TESS",0.0); setp!("jitter_TESS",1e-4); setp!("dilution_TESS",DIL)

# ---- build synthetic diluted flux from truth ------------------------
flux_clean = compute_transit_model_on_grid(th, data0, t_phot, 1)
depth_diluted = 1 - minimum(flux_clean)
# undiluted model for comparison (same theta, D=0)
setp!("dilution_TESS", 0.0)
flux_undil = compute_transit_model_on_grid(th, data0, t_phot, 1)
depth_undiluted = 1 - minimum(flux_undil)
setp!("dilution_TESS", DIL)   # restore truth
@printf("transit depth: undiluted=%.4f  diluted=%.4f  ratio=%.3f (expect 1−D=%.3f)\n",
        depth_undiluted, depth_diluted, depth_diluted/depth_undiluted, 1-DIL)
@assert isapprox(depth_diluted/depth_undiluted, 1-DIL; atol=2e-3) "dilution not applied to depth!"
println("✓ depth diluted by (1−D)\n")

# also build primary RV (planet + K_A binary) so the joint fit is realistic
rv = similar(t_rv)
for i in eachindex(t_rv)
    t = t_rv[i]
    rv[i] = rv_comp[i]==1 ? GAMMA + K_A*kep_geom(t,P_BIN,TC_BIN,E_BIN,W_BIN) + 24.0*kep_geom(t,P_PL,TC_PL,E_PL,W_PL) :
                            GAMMA - K_B*kep_geom(t,P_BIN,TC_BIN,E_BIN,W_BIN)
end
rv .+= JIT .* randn(2nep)
flux = flux_clean .+ flux_err .* randn(nph)

data = Data(; t_rv=t_rv, rv=rv, rv_err=rv_err, rv_inst=ones(Int,2nep), rv_comp=rv_comp,
              t_phot=t_phot, flux=flux, flux_err=flux_err, phot_inst=ones(Int,nph))

# ---- joint logL at truth (finite) + dilution is LIVE ----------------
lljoint(t,d) = rv_log_likelihood(t,d) + transit_log_likelihood(t,d)
ll_truth = lljoint(th, data)
@printf("joint logL(truth) = %.2f\n", ll_truth)
@assert isfinite(ll_truth)
# perturb dilution → transit logL must move (dilution live)
llt0 = transit_log_likelihood(th, data)
d0 = th.values[param_index(params,"dilution_TESS")]
th.values[param_index(params,"dilution_TESS")] = d0 + 0.15
llt1 = transit_log_likelihood(th, data)
th.values[param_index(params,"dilution_TESS")] = d0
@printf("transit logL Δ for +ΔD = %.2f (must be < 0, truth is better)\n", llt1 - llt0)
@assert llt1 < llt0 "dilution does not constrain the transit fit!"
println("✓ dilution live; joint logL finite\n")

# ---- ws / non-ws parity (joint) -------------------------------------
ws = PTWorkspace(params, 2; n_obs=length(t_rv), n_phot=nph)
ll_ws = rv_log_likelihood(th,data,ws) + transit_log_likelihood(th,data,ws)
@printf("joint logL(ws) = %.2f   Δ(ws−direct)=%.2e\n", ll_ws, ll_ws-ll_truth)
@assert isapprox(ll_ws, ll_truth; atol=1e-5) "ws parity broken (SB2+transit)!"
println("✓ ws/non-ws parity OK\n")

println("="^70); println("PHASE B SMOKE PASSED ✓"); println("="^70)

if get(ENV,"RECOVER","0") == "1"
    println("\nBLIND RECOVERY (K_A/K_B/rr/D) via sample_ptemcee ...")
    tgt = NereusTarget(params, data)
    res = sample_ptemcee(tgt, data; n_temps=10, n_walkers=64, n_steps=9000,
                         n_burnin=4500, seed=11, init_strategy=:prior, show_progress=true)
    ch = res.chains
    q(s)=(v=vec(Array(ch[:,s,:])); (median(v),quantile(v,0.16),quantile(v,0.84)))
    for (nm,tv,sym) in (("K_A",K_A,:K_A_k1),("K_B",K_B,:K_B_k1),("rr",RR_PL,:rr_k2),
                        ("dilution",DIL,:dilution_TESS),("b",B_PL,:b_k2))
        m,lo,hi=q(sym); @printf("%-9s truth=%7.3f  med=%7.3f [%.3f, %.3f]\n",nm,tv,m,lo,hi)
    end
end
