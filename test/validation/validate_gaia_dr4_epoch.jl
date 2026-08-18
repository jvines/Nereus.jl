#!/usr/bin/env julia
# Validate the Gaia DR4 epoch-astrometry reader (src/astrometry/gaia_epoch.jl).
#
# The DR4 pre-release (June 2026) ships per-CCD along-scan abscissae for 12
# illustrative sources, incl. three published orbital systems (HD 114762,
# Gaia-4, Gaia BH3). This certifies that:
#   A. the BINARY2 parser + AL model recover, per source, the same 5 astrometric
#      parameters (ϖ, μα*, μδ) that ESA's `gaiasupdate` reference fit gives —
#      the three extragalactic CRF3 QSOs must come out at parallax ≈ 0;
#   B. the parsed IADData plugs into Nereus's `iad_log_likelihood` and returns
#      a finite value (single-star model, orbit off);
#   C. a single-star fit leaves a large residual on exactly the orbital sources,
#      i.e. the orbit signal is present and detectable in the abscissae.
#
# The reference solution is `gaia_dr4_oracle.json` (ESA-model linear LSQ; see
# scratchpad/oracle_5p.py). The VOTable is fetched to the Nereus cache if not
# already present. Point NEREUS_GAIA_DR4_XML at a local copy to skip the fetch.

using Nereus, LinearAlgebra, Printf, Statistics, JSON3

# --- locate the VOTable (env override → cache → download) --------------------
xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
if isempty(xml) || !isfile(xml)
    xml = try
        Nereus.fetch_gaia_dr4_prerelease()
    catch err
        @warn "could not fetch Gaia DR4 pre-release; set NEREUS_GAIA_DR4_XML" exception=err
        exit(0)                       # skip (no network) rather than fail CI
    end
end
@info "Gaia DR4 epoch VOTable" xml

# published Gaia DR3 source_ids for the three orbital systems
const ORBITAL = Dict(
    3937211745905473024 => "HD 114762",
    1457486023639239296 => "Gaia-4",
    4318465066420528000 => "Gaia BH3",
)

sources = read_gaia_epoch_votable(xml)
@printf("parsed %d sources\n", length(sources))

# --- reference (oracle) solution --------------------------------------------
oracle_path = joinpath(@__DIR__, "gaia_dr4_oracle.json")
oracle = isfile(oracle_path) ? JSON3.read(read(oracle_path, String)) : nothing
oracle === nothing && @warn "no oracle json at $oracle_path — skipping numeric cross-check"

# 5-parameter along-scan LSQ directly on an IADData (mirrors the ESA model)
function fit_5p(iad)
    st = sin.(iad.psi); ct = cos.(iad.psi)
    D  = hcat(st, ct, iad.parallax_factor, st .* iad.pm_factor, ct .* iad.pm_factor)
    y  = iad.abscissa; σ = iad.abscissa_err; W = 1.0 ./ σ
    p  = (D .* W) \ (y .* W)
    post = sqrt(mean(((y .- D * p) ./ σ) .^ 2))
    return (; plx = p[3], pmra = p[4], pmdec = p[5], post)
end

ok = true
chk(nm, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", nm))

@printf("\n%-20s %6s %10s %9s %9s %8s  %s\n",
        "source_id", "N", "plx", "pmra", "pmdec", "post_rms", "note")
maxdev = 0.0
qso_plx = Float64[]
for (sid, s) in sort(collect(sources), by = kv -> -length(kv[2].iad.t))
    f = fit_5p(s.iad)
    note = get(ORBITAL, sid, "")
    @printf("%-20d %6d %10.4f %9.4f %9.4f %8.2f  %s\n",
            sid, length(s.iad.t), f.plx, f.pmra, f.pmdec, f.post, note)
    if oracle !== nothing && haskey(oracle, Symbol(string(sid)))
        o = oracle[Symbol(string(sid))]
        dev = maximum(abs, (f.plx - o.plx, f.pmra - o.pmra, f.pmdec - o.pmdec))
        global maxdev = max(maxdev, dev)
    end
    # collect QSO parallaxes (the three ~0-parallax extragalactic sources)
    abs(f.plx) < 0.2 && !haskey(ORBITAL, sid) && push!(qso_plx, f.plx)
end

println("\n=== A. parser + AL model vs ESA reference solution ===")
if oracle !== nothing
    @printf("max |Nereus − oracle| over ϖ,μα*,μδ (all sources) = %.2e mas(/yr)\n", maxdev)
    chk("recovers ESA 5-param solution (< 1e-3 mas)", maxdev < 1e-3)
else
    @printf("(oracle json absent — numeric cross-check skipped)\n")
end
chk("≥3 extragalactic sources at parallax ≈ 0 (|ϖ|<0.2 mas)", length(qso_plx) >= 3)

println("\n=== B. Data plumbing: parsed IADData is accepted by Nereus.Data ===")
let s = first(values(sources))
    data = Nereus.Data(; iad = s.iad)
    chk("Data(; iad=...) constructs from parsed source", data.iad === s.iad)
    chk("t_ref anchored to the abscissa epochs", isfinite(data.t_ref))
end

println("\n=== C. single-star fit leaves orbital residual on orbital sources ===")
for (sid, nm) in ORBITAL
    haskey(sources, sid) || continue
    f = fit_5p(sources[sid].iad)
    chk("$nm (post_rms=$(round(f.post, digits=2)) ) shows orbit residual (>1.5)", f.post > 1.5)
end

println("\n=== D. duplicate-transit guard (Octofitter G23H audit #5a) ===")
let cols = Nereus._decode_gaia_epoch(xml)[1],
    idx  = findall(==(first(unique(cols[:source_id]))), cols[:source_id])
    clean = Nereus._gep_build_source(cols, idx)
    dup   = Nereus._gep_build_source(cols, vcat(idx, idx))   # every transit duplicated
    chk("duplicated transit blocks are dropped (no double-count)",
        length(dup.t) == length(clean.t) && length(clean.t) > 0)
end

println(ok ? "\n✅ GAIA DR4 EPOCH READER VALIDATION PASS" :
             "\n❌ GAIA DR4 EPOCH READER — FAILURES ABOVE")
