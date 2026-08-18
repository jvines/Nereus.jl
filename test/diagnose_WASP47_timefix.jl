#!/usr/bin/env julia
# WASP-47 time-system bug confirmation. The 3 LCs are on mixed time
# systems (K2 = full BJD ~2,456,977; TESS s42/s92 = BTJD ~2,000-4,000,
# mislabeled bjd_tdb). Concatenated, no single ephemeris can align all
# three -> joint transit fit collapses. This converts TESS BTJD ->
# full BJD (+2,457,000) and shows the photometry logL at b's geometry
# jumps from flat/shallow (mixed) to a sharp deep peak (corrected),
# scanning Tc at fixed P=Bryant geometry. No sampling — just logL evals.

using Nereus
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const P_B = 4.1591287
const BTJD_OFFSET = 2_457_000.0

pm_names=["K2","TESS42","TESS92"]
files=["WASP-47_k2_c03_everest_lc.csv","WASP-47_tess_s42_lc.csv","WASP-47_tess_s92_lc.csv"]
is_btjd=[false,true,true]   # K2 already full BJD; TESS files are BTJD

function build(corrected::Bool)
    t=Float64[]; f=Float64[]; e=Float64[]; ii=Int[]
    for (ix,fn) in enumerate(files)
        lc=load_tess_lc(joinpath(DATADIR,fn))
        tt=copy(lc.t)
        corrected && is_btjd[ix] && (tt .+= BTJD_OFFSET)   # BTJD -> full BJD
        append!(t,tt); append!(f,lc.flux); append!(e,lc.flux_err); append!(ii,fill(ix,length(lc.t)))
    end
    p=sortperm(t); (t[p],f[p],e[p],ii[p])
end

function peak_phot_logL(corrected::Bool)
    t,f,e,ii = build(corrected)
    @printf("  [%s] baseline = %.1f d  (%.0f periods)\n",
            corrected ? "CORRECTED" : "MIXED    ", t[end]-t[1], (t[end]-t[1])/P_B)
    # minimal RV so RVPM is valid; transit_log_likelihood ignores it
    data=Data(;t_rv=[0.0,1.0,2.0],rv=[0.0,1.0,0.0],rv_err=[1.0,1.0,1.0],rv_inst=[1,1,1],
              t_phot=t,flux=f,flux_err=e,phot_inst=ii)
    ic=InstrumentConfig(rv=["X"],pm=pm_names)
    pri=Dict{String,PriorSpec}()
    pri["n_p"]=FixedPrior(1.0)
    pri["P_k1"]=UniformPrior(4.0,4.3); pri["K_k1"]=UniformPrior(0.0,250.0)
    pri["sesinw_k1"]=UniformPrior(-0.7,0.7); pri["secosw_k1"]=UniformPrior(-0.7,0.7)
    pri["Tc_k1"]=UniformPrior(t[1],t[1]+P_B)
    pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
    pri["gamma_X"]=UniformPrior(-10.0,10.0); pri["sigma_X"]=ModJeffreysPrior(0.1,50.0)
    for n in pm_names
        pri["offset_$n"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_$n"]=LogUniformPrior(1e-5,5e-3)
        pri["q1_$n"]=UniformPrior(0.0,1.0); pri["q2_$n"]=UniformPrior(0.0,1.0); end
    pri["rho_s"]=NormalPrior(0.707,0.2,0.1,5.0)
    params=Params(;max_kplanet=1,planet_modes=[RVPM],instruments=ic,data=data,M_s=1.04,R_s=1.137,
        parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,
        trend_order=0,stability=:none)
    θ=Theta(params)
    set_param!(θ,"P_k1",P_B); set_param!(θ,"K_k1",140.0)
    set_param!(θ,"sesinw_k1",0.0); set_param!(θ,"secosw_k1",0.0)
    set_param!(θ,"b_k1",0.3); set_param!(θ,"rr_k1",0.094); set_param!(θ,"rho_s",0.707)
    for n in pm_names
        set_param!(θ,"offset_$n",0.0); set_param!(θ,"jitter_$n",3e-4)
        set_param!(θ,"q1_$n",0.3); set_param!(θ,"q2_$n",0.3); end
    set_param!(θ,"gamma_X",0.0); set_param!(θ,"sigma_X",5.0)
    # scan Tc over one period; report the deepest-transit (max logL) phase
    best=-Inf; bestTc=t[1]
    for Tc in range(t[1], t[1]+P_B; length=600)
        set_param!(θ,"Tc_k1",Tc); L=transit_log_likelihood(θ,data)
        L>best && (best=L; bestTc=Tc)
    end
    # baseline (no transit): rr -> ~0
    set_param!(θ,"Tc_k1",bestTc); set_param!(θ,"rr_k1",1e-4); set_param!(θ,"K_k1",0.0)
    Lflat=transit_log_likelihood(θ,data)
    @printf("           best Tc=%.4f   logL_phot(transit)=%.1f   logL_phot(no-transit)=%.1f   Δ=%+.1f\n",
            bestTc, best, Lflat, best-Lflat)
    return best-Lflat
end

println("== WASP-47 photometry logL at b's geometry: MIXED vs CORRECTED times ==")
dmix = peak_phot_logL(false)
dcor = peak_phot_logL(true)
println()
@printf("Δ(transit − no-transit):  MIXED = %+.1f      CORRECTED = %+.1f\n", dmix, dcor)
println(dcor > 10*max(dmix,1) ?
        ">> CONFIRMED: mixed time systems destroy transit alignment; +2,457,000 on TESS fixes it." :
        ">> inconclusive — inspect.")
