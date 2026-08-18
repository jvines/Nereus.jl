#!/usr/bin/env julia
# Fix the GP-injected bumps at the source: don't MASK b (which punches
# regularly-spaced gaps the GP rings across), instead FIT and SUBTRACT b's
# transit so the LC is continuous, then GP-detrend the continuous flux.
# No gaps -> no interpolation artifacts -> no injected upward bumps. The
# long-timescale GP ignores the shallow narrow d/e (preserved).

using Nereus
using CairoMakie
using Statistics: std, median
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pb, Pd, Pe = 4.1591287, 9.030672, 0.789593
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err; t1=t[1]; w0=1.0 ./ e.^2

function binfold(flux, ph, ww; nb=140)
    edges=range(-0.5,0.5;length=nb+1); bx=Float64[]; by=Float64[]
    for k in 1:nb
        sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<4 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by,sum(ww[sel].*flux[sel])/sum(ww[sel]))
    end
    bx,by
end
function tcscan(P)
    ph=@. mod((t-t1)/P+0.5,1.0)-0.5; bx,by=binfold(f,ph,w0); t1+bx[argmin(by)]*P
end

# --- fit b's transit (PM-only, tight P), subtract its model ---
T0b = tcscan(Pb)
data=Data(; t_phot=t, flux=f, flux_err=e, phot_inst=ones(Int,length(t)))
pri=Dict{String,PriorSpec}()
pri["P_k1"]=NormalPrior(Pb,1e-4,Pb-3e-3,Pb+3e-3); pri["Tc_k1"]=NormalPrior(T0b,0.02,T0b-0.12,T0b+0.12)
pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0); pri["rho_s"]=NormalPrior(0.71,0.2,0.1,5.0)
params=Params(;max_kplanet=1,planet_modes=[PM_ONLY],instruments=InstrumentConfig(pm=["K2"]),data=data,M_s=1.04,R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,stability=:none)
res=sample_map(NereusTarget(params,data;unconstrained=true); n_starts=6, maxiter=3000)
θ=Theta(params); for (nm,v) in zip(res.param_names,res.x_map); set_param!(θ,nm,v); end
bmodel,_=phot_predictions(θ,data)
blk=params.layout.planet_blocks[1]
@printf("b fit: P=%.6f rr=%.4f b=%.3f conv=%s railed=%s\n", planet_P(θ,1), θ.values[blk.r], θ.values[blk.b], res.converged, res.railed)
resid_b = f .- bmodel .+ 1.0   # b removed, LC CONTINUOUS (no gaps)

# --- GP-detrend the CONTINUOUS b-subtracted flux, NO masking ---
function runmed(y,w); n=length(y); m=similar(y); h=w÷2; for i in 1:n; lo=max(1,i-h); hi=min(n,i+h); m[i]=median(@view y[lo:hi]); end; m; end
jit = 1.4826*median(abs.((resid_b.-runmed(resid_b,49)) .- median(resid_b.-runmed(resid_b,49))))
e_eff=sqrt.(e.^2 .+ jit^2)
gpp=Dict{String,PriorSpec}("gp_log_omega0_phot"=>UniformPrior(-3.0,0.3),"gp_log_Q_phot"=>UniformPrior(-1.0,0.5))
fc = detrend_gp(t,resid_b,e_eff,CeleriteSHO(); sector_id=ones(Int,length(t)), joint_segments=false, gp_priors=gpp).flux_detrended
fc = fc .+ (bmodel .- 1.0)      # add b's transit back so d/e search has the full LC (b dips present, bowl gone)
@printf("jitter=%.0fppm  detrended scatter (no mask, no clip)=%.0fppm\n", jit*1e6, std(fc)*1e6)

t0=minimum(t)
fig=Figure(size=(1300,800))
ax1=Axis(fig[1,1]; ylabel="detrended flux (full)", xgridvisible=false, ygridvisible=false)
scatter!(ax1, t.-t0, fc; markersize=3, color=(:black,0.55))
text!(ax1,0.005,0.98; text="K2 detrend: SUBTRACT b (no mask gaps) -> GP -> add b back. No clipping.", space=:relative, align=(:left,:top), fontsize=14)
ax2=Axis(fig[2,1]; xlabel="time − t₀ [d]", ylabel="locus zoom", xgridvisible=false, ygridvisible=false)
scatter!(ax2, t.-t0, fc; markersize=3, color=(:black,0.55)); ylims!(ax2,0.9990,1.0010)
text!(ax2,0.005,0.98; text="zoom ±1000ppm — injected bumps gone WITHOUT clipping?", space=:relative, align=(:left,:top), fontsize=14)
out=joinpath(@__DIR__,"WASP47_k2_nogapgp.png"); save(out,fig; px_per_unit=2); println("saved $out")
