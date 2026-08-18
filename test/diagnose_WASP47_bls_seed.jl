#!/usr/bin/env julia
# WASP-47 BLS-seed diagnostic. The γ + blind-RV tests proved the failure
# is in the JOINT/trans-dim machinery, and the prime suspect is the
# informed-birth BLS seed — specifically whether the MAX_BLS_OBS=20_000
# strided decimation I added for speed undersamples b's ~2.4h transit and
# kills the seed. This calls the SAME `_box_least_squares` the informed
# births use, on the SAME K2+s42+s92 photometry, FULL vs decimated, over
# the SAME grid — and checks whether b (4.159d) survives as a top peak.

using Nereus
using Printf
using Statistics: median

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const P_B = 4.1591287

files = ["WASP-47_k2_c03_everest_lc.csv","WASP-47_tess_s42_lc.csv","WASP-47_tess_s92_lc.csv"]
t=Float64[]; f=Float64[]; e=Float64[]; src=Int[]
for (ix,fn) in enumerate(files)
    lc=load_tess_lc(joinpath(DATADIR,fn))
    append!(t,lc.t); append!(f,lc.flux); append!(e,lc.flux_err); append!(src,fill(ix,length(lc.t)))
end
perm=sortperm(t); t=t[perm]; f=f[perm]; e=e[perm]; src=src[perm]
N=length(t)
@printf("loaded %d photometry points (K2=%d, s42=%d, s92=%d)\n",
        N, count(==(1),src), count(==(2),src), count(==(3),src))

# typical cadence per source (median Δt within a source)
for (ix,nm) in enumerate(("K2","s42","s92"))
    ti=sort(t[src.==ix]); length(ti)<3 && continue
    dt=median(diff(ti))*24*60   # minutes
    @printf("  %-4s median cadence = %5.1f min  (n=%d)\n", nm, dt, length(ti))
end

# in-transit point count for b, full vs decimated (transit dur ~2.4h)
dur_d = 2.4/24
intransit(tt) = count(x->(let ph=mod(x-t[1],P_B)/P_B; ph<dur_d/(2P_B) || ph>1-dur_d/(2P_B); end), tt)

periods = exp.(range(log(0.5), log(30.0); length=2000))   # informed-birth grid

function topbls(tt,ff,ee; label="")
    res = Nereus._box_least_squares(tt, ff, ee, periods; n_phase_bins=100, n_peaks=10)
    P=res[1]; depth=res[2]; snr=res[4]
    @printf("\n[%s]  n=%d  in-transit(b)=%d\n", label, length(tt), intransit(tt))
    if isempty(P); println("  (no peaks)"); return end
    ord = sortperm(snr; rev=true)
    println("  top BLS peaks (P / SNR / depth):")
    for j in ord[1:min(6,length(ord))]
        isb = abs(P[j]-P_B)/P_B<0.02 ? "  <== b" :
              (abs(P[j]-2P_B)/(2P_B)<0.02 ? "  (2×b)" :
               abs(P[j]-P_B/2)/(P_B/2)<0.02 ? "  (b/2)" : "")
        @printf("    P=%8.4f d   SNR=%7.2f   depth=%.5f%s\n", P[j], snr[j], depth[j], isb)
    end
    bidx = findall(j->abs(P[j]-P_B)/P_B<0.02, eachindex(P))
    if isempty(bidx)
        println("  >> b (4.159d) NOT among returned peaks")
    else
        rank = findfirst(==(bidx[1]), ord)
        @printf("  >> b found: rank %d/%d by SNR\n", rank, length(ord))
    end
end

# FULL light curve
topbls(t,f,e; label="FULL")

# DECIMATED exactly as the informed-birth BLS does it
s = cld(N, 20_000)
@printf("\nMAX_BLS_OBS decimation: stride=%d  ->  %d points\n", s, length(1:s:N))
topbls(t[1:s:end], f[1:s:end], e[1:s:end]; label="DECIMATED (stride $s)")
