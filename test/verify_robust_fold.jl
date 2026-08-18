#!/usr/bin/env julia
# Verify the new robust occupancy-scaled binning in plot_transit_phasefold
# (module code path, not the test-script binfold). b = dense case, d = sparse
# case (b masked out). Uses the data-driven WASP-47 ephemerides.

using Nereus
using Statistics: median
const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux
runmed(y,w)=(n=length(y);m=similar(y);h=w÷2;for i in 1:n;lo=max(1,i-h);hi=min(n,i+h);m[i]=median(@view y[lo:hi]);end;m)
flat=f./runmed(f,151)

# data-driven ephemerides (from diagnose_WASP47_k2_databls.jl)
Pb,T0b,durb = 4.158517, 2456978.8256, 0.166
Pd,T0d,durd = 9.029180, 2456979.2959, 0.181

# b: dense, full LC
resb = (flux_detrended=flat, periods=[Pb], t0s=[T0b], durations=[durb], snr=[1393.1])
plot_transit_phasefold(t, resb; xwin=0.06, output=joinpath(@__DIR__,"verify_fold_b.png"), save_pdf=false)

# d: sparse, b masked out -> NaN so it's excluded from the fold
flat_d = copy(flat); flat_d[mask_transits(t,[Pb],[T0b];window=0.12)] .= NaN
resd = (flux_detrended=flat_d, periods=[Pd], t0s=[T0d], durations=[durd], snr=[94.7])
plot_transit_phasefold(t, resd; xwin=0.06, output=joinpath(@__DIR__,"verify_fold_d.png"), save_pdf=false)
println("saved verify_fold_b.png and verify_fold_d.png")
