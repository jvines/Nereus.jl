#!/usr/bin/env julia
# 3x3 grid: rows = b/d/e, cols = K2/S42/S92. Each cell = full phase-fold
# of that epoch's RAW flux on the published Vanderburg+2017 ephemeris,
# inverse-variance binned (red). Shows where each transit actually lands
# per epoch — i.e. the ephemeris drift across the K2->TESS baseline.

using Nereus
using CairoMakie
using Statistics: quantile

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
pl = [("b", 4.1591287, 2456979.7641),
      ("d", 9.030672,  2456982.349),
      ("e", 0.789593,  2456983.178)]
segs = [("K2","WASP-47_k2_c03_everest_lc.csv"),
        ("S42","WASP-47_tess_s42_lc.csv"),
        ("S92","WASP-47_tess_s92_lc.csv")]
lcs = Dict(sl => load_tess_lc(joinpath(DATADIR,fn)) for (sl,fn) in segs)

function binned(ph, fl, fe; nb=90)
    edges=range(-0.5,0.5;length=nb+1); w=1.0 ./ fe.^2; bx=Float64[]; by=Float64[]
    for k in 1:nb
        sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<4 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by, sum(w[sel].*fl[sel])/sum(w[sel])); end
    bx,by
end

fig=Figure(size=(1400,1000))
for (r,(nm,P,T0)) in enumerate(pl), (c2,(sl,_)) in enumerate(segs)
    lc=lcs[sl]; ph=@. mod((lc.t-T0)/P+0.5,1.0)-0.5
    ax=Axis(fig[r,c2]; xlabel = r==3 ? "phase" : "", ylabel = c2==1 ? "flux" : "",
            xgridvisible=false, ygridvisible=false)
    scatter!(ax, ph, lc.flux; markersize=2, color=(:gray70,0.12), rasterize=2)
    bx,by=binned(ph, lc.flux, lc.flux_err)
    scatter!(ax, bx, by; markersize=5, color=:red)
    ql,qh=quantile(by,[0.02,0.98]); pad=0.3*(qh-ql)+1e-4; ylims!(ax, ql-3pad, qh+pad)
    text!(ax, 0.02, 0.97; text="$nm — $sl", space=:relative, align=(:left,:top), fontsize=14)
end
out=joinpath(@__DIR__,"WASP47_ephem_drift.png"); save(out,fig; px_per_unit=2); println("saved $out")
