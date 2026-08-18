#!/usr/bin/env julia
# Gate before the full 4-planet run: are d (9.03 d) and e (0.79 d)
# detectable in the BLS once b's transits are removed? The informed
# births consume exactly this BLS, so if d/e surface here they can be
# proposed transit-precise. Mask b's in-transit cadences (ephemeris from
# the time-fixed recovery fit) and BLS the rest over [0.5,30].

using Nereus
using Printf
using Statistics: median

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pb=4.159156; const Tcb=2456978.8126; const DURb=0.10
const Pd=9.030672; const Pe=0.789593

files = get(ENV,"USE_NOTCH","0")=="1" ?
    ["WASP-47_k2_c03_notch.csv","WASP-47_tess_s42_notch.csv","WASP-47_tess_s92_notch.csv"] :
    ["WASP-47_k2_c03_everest_lc.csv","WASP-47_tess_s42_lc.csv","WASP-47_tess_s92_lc.csv"]
println("LC set: ", get(ENV,"USE_NOTCH","0")=="1" ? "NOTCH-detrended" : "raw")
t=Float64[]; f=Float64[]; e=Float64[]
for fn in files
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(t,lc.t); append!(f,lc.flux); append!(e,lc.flux_err); end
p=sortperm(t); t=t[p]; f=f[p]; e=e[p]
# mask b's transits (±0.8·dur around each b transit)
keep = [abs(let d=mod(x-Tcb,Pb); d>Pb/2 ? d-Pb : d; end) > 0.8*DURb for x in t]
tb=t[keep]; fb=f[keep]; eb=e[keep]
@printf("masked %d/%d b-transit cadences; %d remain\n", count(.!keep), length(t), length(tb))

periods = exp.(range(log(0.5), log(30.0); length=4000))
res = Nereus._box_least_squares(tb, fb, eb, periods; n_phase_bins=200, n_peaks=15)
P=res[1]; depth=res[2]; snr=res[4]
ord=sortperm(snr;rev=true)
println("\ntop BLS peaks on b-masked LC (P / SNR / depth):")
shown=0
for j in ord
    shown>=12 && break
    tag = abs(P[j]-Pd)/Pd<0.02 ? "  <== d" : abs(P[j]-Pe)/Pe<0.02 ? "  <== e" :
          abs(P[j]-Pb)/Pb<0.03 ? "  (b resid)" :
          any(abs(P[j]-m*Pb)/(m*Pb)<0.02 for m in 2:6) ? "  (b harm)" :
          any(abs(P[j]-Pb/m)/(Pb/m)<0.02 for m in 2:6) ? "  (b/n)" : ""
    @printf("  P=%8.4f  SNR=%7.2f  depth=%.5f%s\n", P[j], snr[j], depth[j], tag); global shown+=1
end
for (nm,P0) in (("d",Pd),("e",Pe))
    hits=[j for j in eachindex(P) if abs(P[j]-P0)/P0<0.02]
    if isempty(hits); @printf(">> %s (%.3f d) NOT in top-15 peaks\n", nm, P0)
    else r=findfirst(==(hits[1]),ord); @printf(">> %s found: P=%.4f rank %d/%d SNR=%.1f\n", nm, P[hits[1]], r, length(ord), snr[hits[1]]); end
end
