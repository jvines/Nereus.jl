#!/usr/bin/env julia
# Fixed-configuration evidences for HD 18599, one run per noise model.
# Pairs with the trans-dim occupancy: occupancy answers P(M | D) from ONE
# chain; these answer it again from independent per-model log Z values, so
# the two are a cross-check on each other rather than one number to trust.
#
# Every configuration carries the always-on IndicatorFloor, so all of them
# score the SAME data. Without it a mean-model (AD) and a joint RV+indicator
# GP (ActivityGP) are fit to different likelihoods and their log Z values are
# not comparable -- the comparison that motivated the floor in the first place.
using Nereus, JSON3, Printf

const TAGS = ["white", "AD", "AGP", "GPRot", "ErrScale"]
const HERE = @__DIR__

for tag in TAGS
    cfg = joinpath(HERE, "job_fixed_$(tag).json")
    @printf("\n===== %s =====\n", tag); flush(stdout)
    t0 = time()
    try
        s = run_job(cfg)
        @printf("%s: status=%s  log_z=%.3f  (%.1f min)\n", tag,
                get(s, "status", "?"), get(s, "log_z", NaN), (time() - t0) / 60)
    catch err
        @printf("%s: FAILED after %.1f min -- %s\n", tag,
                (time() - t0) / 60, sprint(showerror, err))
    end
    flush(stdout)
end
println("\nall fixed configurations attempted")
