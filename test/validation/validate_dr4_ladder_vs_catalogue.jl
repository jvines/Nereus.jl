#!/usr/bin/env julia
# END-TO-END VALIDATION of the astrometric solution ladder against Gaia's own
# published solution.
#
# The closed-form solve returns parallax and proper motion from the DR4
# pre-release epoch abscissae. Those numbers LOOK right, which is worth nothing:
# a swapped sin/cos, a sign flip on the parallax factor, or a wrong reference
# epoch all produce plausible output. The tomography kernel had exactly that bug
# (atan2 arguments transposed) and it passed every test that did not involve a
# real orbit.
#
# So: fit the DR4 epoch data ourselves, pull Gaia's catalogue solution for the
# same source_id, and require agreement. This checks units, sign conventions,
# the parallax-factor convention, the J2017.5 reference epoch, and the
# barycentric time correction in one shot.
#
# EXPECT DISAGREEMENT AT THE 1-2 SIGMA LEVEL, and do not "fix" it:
#   * the catalogue is DR3 (~2.8 yr baseline); the epoch data here is DR4-format
#     (~5.5 yr). Genuinely different data, so genuinely different estimates.
#   * our proper priors shrink slightly toward zero; the catalogue solution is
#     effectively unpenalised.
# What must NOT happen is a sign error, a factor of 2 or 1000, or a scatter far
# wider than the quoted errors.
using Nereus, Printf, Statistics, Downloads
using Nereus: astrom_solution, ladder_probabilities, n_iad, nereus_cache_dir

const TAP = "https://gea.esac.esa.int/tap-server/tap/sync"

"Cached Gaia DR3 catalogue solution for the given source_ids."
function gaia_dr3_solutions(ids::Vector{Int64})
    dir = nereus_cache_dir("GaiaDR4"); mkpath(dir)
    cache = joinpath(dir, "dr3_solutions.csv")
    if !isfile(cache)
        q = "SELECT source_id,parallax,parallax_error,pmra,pmra_error," *
            "pmdec,pmdec_error,ruwe,astrometric_params_solved " *
            "FROM gaiadr3.gaia_source WHERE source_id IN (" *
            join(string.(ids), ",") * ")"
        # Hand-encode: only spaces, commas and parens need it for ADQL, and
        # Downloads has no escapeuri.
        enc = replace(q, " " => "%20", "," => "%2C", "(" => "%28", ")" => "%29",
                         "*" => "%2A")
        url = TAP * "?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=" * enc
        @info "querying Gaia archive (cached to $cache)"
        Downloads.download(url, cache)
    end
    out = Dict{Int64,NamedTuple}()
    for (i, line) in enumerate(eachline(cache))
        i == 1 && continue
        f = split(strip(line), ',')
        length(f) >= 9 || continue
        g(k) = isempty(f[k]) ? NaN : parse(Float64, f[k])
        out[parse(Int64, f[1])] = (plx = g(2), plx_e = g(3), pmra = g(4),
                                    pmra_e = g(5), pmdec = g(6), pmdec_e = g(7),
                                    ruwe = g(8), solved = g(9))
    end
    return out
end

srcs = read_gaia_epoch_votable(fetch_gaia_dr4_prerelease())
ids  = sort(collect(keys(srcs)))
cat  = gaia_dr3_solutions(ids)

# Wrapped in a function: assigning to an outer name from inside a top-level
# `for` hits Julia's soft-scope rule and throws at the first accumulation.
function compare(srcs, cat, ids)
    println("\nDR4 epoch fit (ours) vs Gaia DR3 catalogue\n")
    @printf("%-20s %8s %8s %6s   %8s %8s %6s   %8s %8s %6s  %5s %5s\n",
            "source_id","plx_us","plx_cat","n_sig","pmra_us","pmra_cat","n_sig",
            "pmdec_us","pmdec_cat","n_sig","RUWE","solv")
    nsigs = Float64[]; checked = 0; type_agree = 0; type_total = 0
    for sid in ids
        haskey(cat, sid) || continue
        c = cat[sid]; iad = srcs[sid].iad
        n_iad(iad) > 20 || continue
        sol = astrom_solution(iad, 5)
        lad = ladder_probabilities(iad)
        pull(a, ea, b, eb) = (a - b) / sqrt(ea^2 + eb^2)
        p1 = pull(sol.q[3], sol.sigma[3], c.plx,   c.plx_e)
        p2 = pull(sol.q[4], sol.sigma[4], c.pmra,  c.pmra_e)
        p3 = pull(sol.q[5], sol.sigma[5], c.pmdec, c.pmdec_e)
        @printf("%-20d %8.4f %8.4f %6.1f   %8.4f %8.4f %6.1f   %8.4f %8.4f %6.1f  %5.2f %5d\n",
                sid, sol.q[3], c.plx, p1, sol.q[4], c.pmra, p2,
                sol.q[5], c.pmdec, p3, c.ruwe, Int(c.solved))
        # Only sources the ladder calls adequate can be compared: where no rung
        # fits (BH3 and friends) the 5p numbers are meaningless for both.
        if lad.adequate
            append!(nsigs, abs.([p1, p2, p3])); checked += 1; type_total += 1
            # astrometric_params_solved: 31 = 5p, 95 = 6p (pseudocolour — still
            # "5p" in ladder terms, the extra parameter is colour not motion)
            (Int(c.solved) in (31, 95)) && lad.best == 5 && (type_agree += 1)
        end
    end
    return nsigs, checked, type_agree, type_total
end

nsigs, checked, type_agree, type_total = compare(srcs, cat, ids)

@printf("\nadequate sources compared: %d   (%d parameter comparisons)\n",
        checked, length(nsigs))
@printf("|n_sigma|: median %.2f  max %.2f\n", median(nsigs), maximum(nsigs))
@printf("solution-type agreement (ours 5p vs catalogue 5p/6p): %d/%d\n",
        type_agree, type_total)

if maximum(nsigs) < 5.0 && median(nsigs) < 2.0
    println("\n✅ VALIDATED — units, signs, parallax-factor and reference epoch")
    println("   all consistent with the published solution.")
    exit(0)
else
    println("\n❌ MISMATCH — not a DR3-vs-DR4 baseline difference at this size.")
    println("   Suspect sign/units/reference epoch before anything else.")
    exit(1)
end
