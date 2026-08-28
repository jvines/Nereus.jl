#!/usr/bin/env julia
# HD 18599 (TOI-179) — build the run_job input from the raw RV table.
#
# Reproduces the selection used in the paper's analysis, in one place, so the
# artifact is (raw CSV -> this script -> config -> run_job) end to end:
#
#   1. Drop HARPS_POST rows with provenance ESO_PHASE3. That pipeline reports
#      rv_error in different units (a known catalogue defect, off by 1e3);
#      the HARPS_POST epochs are kept from the other provenance.
#   2. Keep the three instruments the published analysis used.
#   3. 5-sigma MAD clip per instrument (robust scatter, not std).
#
# Indicator values are written RAW (not normalised) -- Nereus normalises them
# per instrument internally, rescaling values and their errors together.
using DelimitedFiles, Statistics, Printf

# Raw table ships WITH the artifact: the copy under studies/ is not tracked
# in git, so an external reader could not reach it.
const SRC  = joinpath(@__DIR__, "input", "hd18599_raw.csv")
const DEST = joinpath(@__DIR__, "hd18599_clean.csv")
const PAPER_INST = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

raw, hdr = readdlm(SRC, ',', Any, '\n'; header = true)
h = vec(String.(hdr)); col(n) = findfirst(==(n), h)
num(x) = x isa Real ? Float64(x) :
         (s = strip(string(x)); isempty(s) ? NaN : something(tryparse(Float64, s), NaN))

ins  = [strip(String(raw[i, col("instrument")])) for i in 1:size(raw, 1)]
prov = [strip(String(raw[i, col("provenance")])) for i in 1:size(raw, 1)]
keep = [i for i in 1:size(raw, 1)
        if ins[i] in PAPER_INST && !(ins[i] == "HARPS_POST" && prov[i] == "ESO_PHASE3")]
@printf("raw rows %d -> after provenance/instrument cut %d\n", size(raw, 1), length(keep))

bjd = num.(raw[keep, col("bjd")]); rv = num.(raw[keep, col("rv")])
err = num.(raw[keep, col("rv_error")]); insk = ins[keep]

# 5-sigma MAD clip, per instrument
ok = trues(length(bjd))
for nm in unique(insk)
    idx = findall(==(nm), insk)
    m   = median(rv[idx])
    thr = 5 * max(1.4826 * median(abs.(rv[idx] .- m)), 1e-6)
    for k in idx; abs(rv[k] - m) > thr && (ok[k] = false); end
end
@printf("after 5-sigma MAD clip %d (dropped %d)\n", count(ok), count(.!ok))

IND = ["bis" => "bisector_span", "fwhm" => "fwhm",
       "halpha" => "halpha", "logrhk" => "log_rhk"]
outcols = ["bjd", "rv", "rv_err", "instrument"]
data = Any[bjd[ok], rv[ok], err[ok], insk[ok]]
insel = insk[ok]

# Indicator 1-sigmas must be finite and strictly positive -- Nereus rejects the
# channel otherwise, and an indicator GP needs them. The source table reports
# no log R'HK uncertainty at all (0 of 114 rows), and other channels can have
# isolated gaps. Fill per instrument, in this order:
#   * median of that instrument's reported errors, when it has any;
#   * else 0.1 x MAD of that instrument's values -- a 10% floor, which is what
#     the original analysis used after normalising by MAD.
function fill_errs(v, e, insts)
    out = copy(e)
    for nm in unique(insts)
        idx = findall(==(nm), insts)
        good = [k for k in idx if isfinite(e[k]) && e[k] > 0]
        fillv = if !isempty(good)
            median(e[good])
        else
            fin = [k for k in idx if isfinite(v[k])]
            mad = isempty(fin) ? NaN :
                  1.4826 * median(abs.(v[fin] .- median(v[fin])))
            (isfinite(mad) && mad > 0) ? 0.1 * mad : NaN
        end
        for k in idx
            (isfinite(out[k]) && out[k] > 0) || (out[k] = fillv)
        end
    end
    return out
end

for (short, src) in IND
    v = num.(raw[keep, col(src)])[ok]
    e = fill_errs(v, num.(raw[keep, col(src * "_error")])[ok], insel)
    nbad = count(k -> isfinite(v[k]) && !(isfinite(e[k]) && e[k] > 0), eachindex(v))
    nbad == 0 || @printf("  WARNING %s: %d rows still lack a usable error\n", short, nbad)
    @printf("  %-7s values %3d finite, errors filled -> median %.4g\n",
            short, count(isfinite, v), median(filter(isfinite, e)))
    push!(outcols, short); push!(data, v)
    push!(outcols, short * "_err"); push!(data, e)
end

open(DEST, "w") do io
    println(io, join(outcols, ","))
    for i in 1:count(ok)
        println(io, join((x isa AbstractString ? x :
                          (isfinite(x) ? string(x) : "") for x in (c[i] for c in data)), ","))
    end
end
@printf("wrote %s: %d rows, instruments %s\n", DEST, count(ok),
        join(sort(unique(insk[ok])), " "))
for nm in sort(unique(insk[ok]))
    @printf("   %-12s n=%3d\n", nm, count(==(nm), insk[ok]))
end
