#!/usr/bin/env julia
# Is the literature period precise enough to phase-align b/d/e across the
# K2(2014)→TESS(2021/25) gap? Fold RAW flux PER EPOCH on the PUBLISHED
# Vanderburg+2017 ephemerides and report the phase of the binned-minimum
# (transit) in each. If a planet's dip sits at phase ~0 in K2 (the
# reference epoch) but drifts away in TESS, the period is too imprecise
# for a single (P,T0) mask/fold across the baseline.

using Nereus
using Printf
using Statistics: mean

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
# PUBLISHED (Vanderburg+2017), full BJD
pl = [("b", 4.1591287, 2456979.7641),
      ("d", 9.030672,  2456982.349),
      ("e", 0.789593,  2456983.178)]
segs = [("K2",  "WASP-47_k2_c03_everest_lc.csv"),
        ("S42", "WASP-47_tess_s42_lc.csv"),
        ("S92", "WASP-47_tess_s92_lc.csv")]

function dip_phase(t, f, e, P, T0; nb=80)
    ph = @. mod((t - T0)/P + 0.5, 1.0) - 0.5
    edges = range(-0.5, 0.5; length=nb+1); w = 1.0 ./ e.^2
    bx=Float64[]; by=Float64[]
    for k in 1:nb
        sel = (ph .>= edges[k]) .& (ph .< edges[k+1]); count(sel)<5 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by, sum(w[sel].*f[sel])/sum(w[sel]))
    end
    isempty(by) && return (NaN, NaN)
    j = argmin(by)
    return (bx[j], 1.0 - by[j])   # (phase of deepest bin, depth)
end

for (nm, P, T0) in pl
    @printf("\n%s  (published P=%.6f, T0=%.4f):\n", nm, P, T0)
    for (sl, fn) in segs
        lc = load_tess_lc(joinpath(DATADIR, fn))
        php, dep = dip_phase(lc.t, lc.flux, lc.flux_err, P, T0)
        ncyc = round(Int, (mean(lc.t) - T0)/P)
        flag = abs(php) < 0.02 ? "ALIGNED" : @sprintf("DRIFTED %+.0f min", php*P*24*60)
        @printf("   %-3s: dip@phase=%+.3f  depth=%.4f   (~%d cycles from T0)   %s\n",
                sl, php, dep, ncyc, flag)
    end
end
