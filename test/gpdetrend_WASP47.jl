#!/usr/bin/env julia
# Proper GP detrend of WASP-47 photometry with ALL known transits masked
# (b, d, e — ephemerides known). celerite CeleriteSHO per segment
# (joint_segments=false; K2 bowl vs TESS per-orbit need different
# timescales). Masked (in-transit) cadences are downweighted ×1e8 so the
# GP can't track transit shape into the trend, then predicted everywhere
# and removed → transits preserved, systematics-around-transits gone.

using Nereus
using CairoMakie
using DelimitedFiles: writedlm
using Statistics: quantile, std, median
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
# ephemerides: literature periods + empirical Tc anchors (BTJD) + 2,457,000
const Pb, Pd, Pe = 4.1591287, 9.030672, 0.789593
const T0b, T0d, T0e = 2461.834 + 2_457_000, 2458.373 + 2_457_000, 2461.201 + 2_457_000

files  = ["WASP-47_k2_c03_everest_lc.csv","WASP-47_tess_s42_lc.csv","WASP-47_tess_s92_lc.csv"]
labels = ["K2 C3", "TESS S42", "TESS S92"]
cols   = [:dodgerblue3, :darkorange2, :purple3]
pt_t=Float64[]; pt_f=Float64[]; pt_e=Float64[]; pt_i=Int[]
for (ix,fn) in enumerate(files)
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(pt_t,lc.t); append!(pt_f,lc.flux)
    append!(pt_e,lc.flux_err); append!(pt_i,fill(ix,length(lc.t))); end
perm=sortperm(pt_t); pt_t=pt_t[perm]; pt_f=pt_f[perm]; pt_e=pt_e[perm]; pt_i=pt_i[perm]

# mask all three planets (per-planet windows; e is ultra-short → wide phase window)
mb = mask_transits(pt_t, [Pb], [T0b]; window = 0.030)
md = mask_transits(pt_t, [Pd], [T0d]; window = 0.020)
me = mask_transits(pt_t, [Pe], [T0e]; window = 0.060)
mask = mb .| md .| me
@printf("masked: b=%d d=%d e=%d  union=%d / %d (%.1f%%)\n",
        count(mb), count(md), count(me), count(mask), length(mask), 100*count(mask)/length(mask))

println("GP detrend (CeleriteSHO, per-segment, transits masked) ...")
t0 = time()
res = detrend_gp(pt_t, pt_f, pt_e, CeleriteSHO();
                 transit_mask = mask, sector_id = pt_i, joint_segments = false)
fd = res.flux_detrended
@printf("done %.0fs   resid σ (OOT): raw→ %.5f  gp→ %.5f\n",
        time()-t0, std(pt_f[.!mask]), std(fd[.!mask]))

# save detrended CSVs per segment
for (ix,fn) in enumerate(files)
    sel = pt_i .== ix
    ofn = replace(fn, ".csv" => "_gp.csv")
    open(joinpath(DATADIR, ofn), "w") do io
        println(io, "bjd_tdb,pdcsap_flux_norm,pdcsap_flux_err_norm")
        writedlm(io, [pt_t[sel] fd[sel] pt_e[sel]], ',')
    end
end

# ---- figure: 3 segment overviews + phase-folded b (near-transit) ----
fig = Figure(size = (1150, 1150))
for (i,(lab,col)) in enumerate(zip(labels,cols))
    sel = pt_i .== i
    ax = Axis(fig[i,1]; ylabel = "GP-detrended flux", xgridvisible=false, ygridvisible=false,
              xlabel = i==3 ? "time − t₀  [d]" : "")
    t0s = minimum(pt_t[sel])
    scatter!(ax, pt_t[sel] .- t0s, fd[sel]; markersize=2.5, color=(col,0.35), rasterize=2)
    ql,qh = quantile(fd[sel],[0.003,0.997]); pad=0.05*(qh-ql); ylims!(ax, ql-pad, qh+pad)
    text!(ax, 0.005, 0.98; text="$lab  resid σ=$(round(Int,std(fd[sel])*1e6))ppm",
          space=:relative, align=(:left,:top), fontsize=15)
end
# phase-folded b transit on the GP-detrended LC (the near-transit region)
axb = Axis(fig[4,1]; xlabel="phase (P_b)", ylabel="GP-detrended flux",
           xgridvisible=false, ygridvisible=false)
ph = @. mod((pt_t - T0b)/Pb + 0.5, 1.0) - 0.5
near = abs.(ph) .< 0.06
scatter!(axb, ph[near], fd[near]; markersize=3, color=(:black,0.25), rasterize=2)
ql,qh = quantile(fd[near],[0.002,0.999]); ylims!(axb, ql-0.0005, qh+0.0005)
text!(axb, 0.005, 0.98; text="b phase-fold (GP-detrended) — eyeball near-transit systematics",
      space=:relative, align=(:left,:top), fontsize=15)
out = joinpath(@__DIR__, "WASP47_sectors_gp.png")
save(out, fig; px_per_unit=2)
println("saved $out")
