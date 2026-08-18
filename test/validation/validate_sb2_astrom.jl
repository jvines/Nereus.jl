#!/usr/bin/env julia
# SB2 binary + astrometry — Phase C (luminous-companion photocenter).
#
# BINARY = (:RV, :AS, :SB): the SB2 binary now carries an inclination and Ω
# (absolute astrometry), so the reflex breaks the sin-i degeneracy → absolute
# masses. A LUMINOUS secondary shifts the observed PHOTOCENTER toward the
# barycentre: the star-A reflex is scaled by (1 − f_light/f_mass), with
# f_mass = M_B/M_total = K_A/(K_A+K_B), f_light = L_B/(L_A+L_B) (astrometric
# band). Applied at the single `_planet_orbit` chokepoint (IAD/GOST/HGCA/G23H
# inherit it; relAST is excluded).
#
# SMOKE (always): BINARY layout has inc/Omega/f_light; the effective reflex
# mass returned by _planet_orbit == M_B·(1−f_light/f_mass); star_reflex scales
# LINEARLY in mass (validates the effective-mass approach); astrom logL is
# finite and f_light-responsive.
#
# Run:  julia --project test/validation/validate_sb2_astrom.jl

using Nereus, Statistics, Printf, Random
using Nereus: param_index, Theta, system_f_light, planet_M_sec, msec_from_K,
               astrom_M_pri, astrom_plx, star_reflex_pm, star_reflex_offset,
               astrom_log_likelihood, compute_derived
const P = Nereus

# ---- truth ----------------------------------------------------------
const P_BIN, TC_BIN, E_BIN, W_BIN = 2100.0, 55000.0, 0.30, 0.9   # long-period binary (MJD anchor)
const K_A, K_B                    = 4200.0, 6100.0               # m/s
const INC_BIN, OM_BIN             = 1.05, 2.1                    # rad
const F_LIGHT                     = 0.12                          # L_B/(L_A+L_B), Gaia band
const M_A, PLX                    = 0.95, 25.0                    # M_sun, mas
const GAMMA, JIT                  = 120.0, 5.0

Random.seed!(1234)

# ---- RV epochs (primary + secondary) --------------------------------
nep = 40; tep = sort!(TC_BIN .+ 1600.0 .* rand(nep) .- 800.0)
t_rv = vcat(tep, tep); rv_comp = vcat(fill(1,nep), fill(2,nep)); rv_err = fill(8.0, 2nep)

# ---- synthetic HGCA (3-epoch absolute PM), built from the truth reflex ----
ep = P.mjd_epochs((1991.25, 2004.5, 2016.0))
hgca0 = HGCAData(; epochs=ep, pmra=(0.0,0.0,0.0), pmdec=(0.0,0.0,0.0),
                   sigma_pmra=(0.30,0.10,0.05), sigma_pmdec=(0.30,0.10,0.05),
                   plx=PLX, plx_err=0.05, hip_id=999999)

data0 = Data(; t_rv=t_rv, rv=zeros(2nep), rv_err=rv_err, rv_inst=ones(Int,2nep),
               rv_comp=rv_comp, hgca=hgca0)

ic  = InstrumentConfig(rv=["SB2SPEC"])
par = ParametrizationConfig(time=:Tc)
pri = Dict{String,PriorSpec}(
    "n_p"=>FixedPrior(1.0),
    "P_k1"=>NormalPrior(P_BIN,20.0,P_BIN-200,P_BIN+200),
    "K_A_k1"=>UniformPrior(1000.0,8000.0), "K_B_k1"=>UniformPrior(1000.0,10000.0),
    "Tc_k1"=>NormalPrior(TC_BIN,50.0,TC_BIN-400,TC_BIN+400),
    "sesinw_k1"=>UniformPrior(-0.9,0.9), "secosw_k1"=>UniformPrior(-0.9,0.9),
    "inc_k1"=>UniformPrior(0.0,π), "Omega_k1"=>UniformPrior(0.0,2π),
    "gamma_SB2SPEC"=>UniformPrior(-5000.0,5000.0), "sigma_SB2SPEC"=>LogUniformPrior(0.1,50.0),
    "M_pri"=>FixedPrior(M_A), "plx"=>NormalPrior(PLX,0.05,PLX-1,PLX+1),
    "f_light"=>UniformPrior(0.0,0.5),
)
params = Params(; max_kplanet=1, planet_modes=[BINARY], instruments=ic,
                  data=data0, parametrization=par, priors=pri, stability=:none,
                  M_s=M_A, R_s=1.0)

println("="^70); println("SB2 + astrometry — Phase C (f_light photocenter)"); println("="^70)
names = params.layout.names
@printf("block k1 : %s   has_AS=%s\n", typeof(params.layout.planet_blocks[1]),
        Nereus.has_AS(params.layout.planet_blocks[1]))
@assert params.layout.planet_blocks[1] isa Nereus.SB2Block
@assert Nereus.has_AS(params.layout.planet_blocks[1]) "BINARY must have astrometry (Omega≠0)"
for s in ("inc_k1","Omega_k1","f_light","K_A_k1","K_B_k1")
    @assert s in names "missing slot $s"
end
println("slots: ", join(filter(n->occursin("_k1",n)||n=="f_light"||n=="M_pri"||n=="plx", names), ", "))
println("✓ BINARY layout has inc/Omega/f_light\n")

# ---- truth theta ----------------------------------------------------
sesinw(e,w)=sqrt(e)*sin(w); secosw(e,w)=sqrt(e)*cos(w)
th = Theta(params); setp!(n,v)=(th.values[param_index(params,n)]=v)
setp!("P_k1",P_BIN); setp!("K_A_k1",K_A); setp!("K_B_k1",K_B); setp!("Tc_k1",TC_BIN)
setp!("sesinw_k1",sesinw(E_BIN,W_BIN)); setp!("secosw_k1",secosw(E_BIN,W_BIN))
setp!("inc_k1",INC_BIN); setp!("Omega_k1",OM_BIN)
setp!("gamma_SB2SPEC",GAMMA); setp!("sigma_SB2SPEC",JIT)
setp!("M_pri",M_A); setp!("plx",PLX); setp!("f_light",F_LIGHT)

# ---- CORE unit test: photocenter scaling of the effective reflex mass ----
M_pri = astrom_M_pri(th); plx = astrom_plx(th); t_ref = data0.t_ref
f_mass = K_A / (K_A + K_B)
# TRUE absolute masses from the SB2 two-amplitude relations (M_A is DERIVED,
# not an independent input — the earlier M_A=0.95 label was over-specified).
MA_true, MB_true = P._sb2_masses(th, 1)
M_B_true = MB_true                  # planet_M_sec(::SB2Block) == this
@printf("SB2 truth masses: M_A=%.4f  M_B=%.4f M_sun ; f_mass=K_A/(K_A+K_B)=%.4f\n",
        MA_true, MB_true, f_mass)

# _planet_orbit returns the EFFECTIVE reflex mass (photocenter-corrected)
_, Msec_eff = P._planet_orbit(th, 1, M_pri, plx, t_ref)
expected = M_B_true * (1 - F_LIGHT/f_mass)
@printf("effective reflex mass: got=%.5f  expected M_B·(1−f_light/f_mass)=%.5f\n", Msec_eff, expected)
@assert isapprox(Msec_eff, expected; rtol=1e-6) "photocenter scaling wrong!"

# f_light = 0 ⇒ effective mass == M_B (dark-companion, unchanged)
setp!("f_light", 0.0)
_, Msec_dark = P._planet_orbit(th, 1, M_pri, plx, t_ref)
@assert isapprox(Msec_dark, M_B_true; rtol=1e-6) "f_light=0 must leave reflex unchanged (backward-compat)!"
setp!("f_light", F_LIGHT)
println("✓ effective reflex mass = M_B·(1−f_light/f_mass); dark case unchanged\n")

# ---- validate the linearity assumption (raoff ∝ mass at fixed orbit) ----
orb, _ = P._planet_orbit(th, 1, M_pri, plx, t_ref)
μ1 = star_reflex_pm(orb, ep[3], 1.0)
μ2 = star_reflex_pm(orb, ep[3], 0.5)
@printf("star_reflex_pm linearity: μ(M=1)/μ(M=0.5) = (%.4f, %.4f) (expect 2,2)\n",
        μ1[1]/μ2[1], μ1[2]/μ2[2])
@assert isapprox(μ1[1]/μ2[1], 2.0; rtol=1e-6) && isapprox(μ1[2]/μ2[2], 2.0; rtol=1e-6) "reflex NOT linear in mass — effective-mass trick invalid!"
println("✓ star_reflex scales linearly in mass — effective-mass chokepoint valid\n")

# ---- astrometry likelihood finite + f_light-responsive --------------
# Rebuild HGCA SELF-CONSISTENTLY: replicate the Brandt model exactly so truth
# is the optimum. Hip (ie1) & Gaia (ie3) = instantaneous reflex PM; HG (ie2) =
# scaled POSITION DIFFERENCE (pos_Gaia − pos_Hip)/baseline (NOT instantaneous).
# Uses the photocenter-effective reflex mass Msec_eff.
function hgca_model_pm(orb, ep, Msec)
    μ_hip  = star_reflex_pm(orb, ep[1], Msec)
    μ_gaia = star_reflex_pm(orb, ep[3], Msec)
    oH = star_reflex_offset(orb, ep[1], Msec); oG = star_reflex_offset(orb, ep[3], Msec)
    dt_yr = (ep[3]-ep[1])/365.25
    μ_hg = ((oG[1]-oH[1])/dt_yr, (oG[2]-oH[2])/dt_yr)
    return (μ_hip, μ_hg, μ_gaia)
end
mh, mhg, mg = hgca_model_pm(orb, ep, Msec_eff)
hgca = HGCAData(; epochs=ep, pmra=(mh[1],mhg[1],mg[1]), pmdec=(mh[2],mhg[2],mg[2]),
                  sigma_pmra=(0.30,0.10,0.05), sigma_pmdec=(0.30,0.10,0.05),
                  plx=PLX, plx_err=0.05, hip_id=999999)
rv = similar(t_rv)
for i in eachindex(t_rv)
    t=t_rv[i]
    kg(A)= A*(let e=E_BIN,w=W_BIN,Pp=P_BIN,Tc=TC_BIN
                f_tc=π/2-w; E_tc=2*atan(sqrt((1-e)/(1+e))*tan(f_tc/2)); M_tc=E_tc-e*sin(E_tc)
                M=2π*(t-Tc)/Pp+M_tc; E=M; for _ in 1:80; E-=(E-e*sin(E)-M)/(1-e*cos(E)); end
                ff=2*atan(sqrt((1+e)/(1-e))*tan(E/2)); cos(ff+w)+e*cos(w) end)
    rv[i] = rv_comp[i]==1 ? GAMMA + kg(K_A) : GAMMA - kg(K_B)
end
rv .+= JIT .* randn(2nep)
data = Data(; t_rv=t_rv, rv=rv, rv_err=rv_err, rv_inst=ones(Int,2nep), rv_comp=rv_comp, hgca=hgca)

ll_astro = astrom_log_likelihood(th, data)
@printf("astrom logL(truth) = %.3f\n", ll_astro)
@assert isfinite(ll_astro) "astrom logL not finite for SB2!"
# Catalog is now SELF-CONSISTENT (built from the exact Brandt model at truth),
# so truth IS the astrometry optimum: perturbing f_light must REDUCE the logL.
f0 = th.values[param_index(params,"f_light")]
th.values[param_index(params,"f_light")] = f0 + 0.15
ll_astro2 = astrom_log_likelihood(th, data)
th.values[param_index(params,"f_light")] = f0
@printf("astrom logL Δ for +Δf_light = %.3f (must be < 0; truth is optimum)\n", ll_astro2 - ll_astro)
@assert ll_astro2 < ll_astro "f_light does NOT constrain astrometry — photocenter not wired!"
# dark-companion baseline (f_light=0) is a WORSE fit than the luminous truth
th.values[param_index(params,"f_light")] = 0.0
ll_dark = astrom_log_likelihood(th, data)
th.values[param_index(params,"f_light")] = f0
@assert ll_dark < ll_astro "dark baseline should fit worse than luminous truth"
@printf("astrom logL(f_light=0, dark) = %.3f  vs truth %.3f\n", ll_dark, ll_astro)
@assert isfinite(ll_dark)
println("✓ astrom logL finite; f_light is live in the astrometry path\n")

println("="^70); println("PHASE C SMOKE PASSED ✓  — f_light photocenter validated"); println("="^70)

# ---- C2: full BINARY recovery of i_bin + absolute masses ------------
if get(ENV,"RECOVER","0") == "1"
    println("\n", "="^70)
    println("C2: BINARY (RV+AS+SB) blind recovery of i_bin + absolute masses")
    println("="^70)
    # noisy, self-consistent HGCA catalog (reflex model at truth + catalog noise)
    σμ = (0.30,0.10,0.05)
    nz(m,s)= (m[1]+s*randn(), m[2]+s*randn())
    mh_n = nz(mh,σμ[1]); mhg_n = nz(mhg,σμ[2]); mg_n = nz(mg,σμ[3])
    hgca_r = HGCAData(; epochs=ep, pmra=(mh_n[1],mhg_n[1],mg_n[1]),
                        pmdec=(mh_n[2],mhg_n[2],mg_n[2]),
                        sigma_pmra=σμ, sigma_pmdec=σμ, plx=PLX, plx_err=0.05, hip_id=999999)
    data_r = Data(; t_rv=t_rv, rv=rv, rv_err=rv_err, rv_inst=ones(Int,2nep),
                    rv_comp=rv_comp, hgca=hgca_r)
    # f_light constrained by the CCF flux ratio (tight prior) — the realistic
    # science case; the RV pins K_A/K_B, the astrometry pins inc → absolute masses.
    pri_r = copy(pri); pri_r["f_light"] = NormalPrior(F_LIGHT, 0.02, 0.0, 0.5)
    params_r = Params(; max_kplanet=1, planet_modes=[BINARY], instruments=ic,
                        data=data_r, parametrization=par, priors=pri_r, stability=:none,
                        M_s=M_A, R_s=1.0)
    tgt = NereusTarget(params_r, data_r)
    res = sample_ptemcee(tgt, data_r; n_temps=10, n_walkers=64, n_steps=9000,
                         n_burnin=4500, seed=3, init_strategy=:prior, show_progress=true)
    ch = res.chains
    q(s)=(v=vec(Array(ch[:,s,:])); (median(v),quantile(v,0.16),quantile(v,0.84)))
    incd = q(:inc_k1)
    @printf("\n%-9s truth      med [16, 84]\n","param")
    @printf("K_A     %8.1f  %8.1f [%.1f, %.1f]\n", K_A, q(:K_A_k1)...)
    @printf("K_B     %8.1f  %8.1f [%.1f, %.1f]\n", K_B, q(:K_B_k1)...)
    @printf("inc(°)  %8.2f  %8.2f [%.2f, %.2f]\n", rad2deg(INC_BIN),
            rad2deg(incd[1]), rad2deg(incd[2]), rad2deg(incd[3]))
    # absolute masses via compute_derived (D2) — the science payoff
    dv = compute_derived(ch, params_r)
    bind = findfirst(d -> d.name == "binary_1", dv)
    if bind !== nothing
        D = dv[bind].values
        qs = D["mass_ratio_q"]; MA = get(D,"M_A",Float64[]); MB = get(D,"M_B",Float64[])
        @printf("q=M_B/M_A truth=%.3f  med=%.3f [%.3f, %.3f]\n", K_A/K_B,
                median(qs), quantile(qs,0.16), quantile(qs,0.84))
        if !isempty(MA)
            @printf("M_A(M_sun)  truth=%.3f  med=%.3f [%.3f, %.3f]\n", MA_true,
                    median(MA), quantile(MA,0.16), quantile(MA,0.84))
            @printf("M_B(M_sun)  truth=%.3f  med=%.3f [%.3f, %.3f]\n", MB_true,
                    median(MB), quantile(MB,0.16), quantile(MB,0.84))
            okM = abs(median(MA)-MA_true)/MA_true < 0.15 &&
                  abs(median(MB)-MB_true)/MB_true < 0.15
            oki = abs(rad2deg(incd[1]) - rad2deg(INC_BIN)) < 8
            println((okM && oki) ? "\n✓ C2 RECOVERY PASSED — i_bin + absolute masses recovered" :
                                   "\n✗ C2 recovery off — investigate")
        end
    else
        println("!! no binary_1 DerivedParams — compute_derived SB2 branch missing")
    end
end
