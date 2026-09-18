# Label-switching diagnostic for exchangeable-slot (trans-dim) runs.
#
# A trans-dimensional run over `g` candidate slots carries a permutation
# symmetry: the slot labels are exchangeable, so one physical configuration
# appears in up to N! relabelled copies. That symmetry is harmless for the
# marginal science (P, K, e are read per-component, not per-slot) but it is
# NOT harmless for the sampler: a component that trades between slots is a
# set of equivalent modes the kernel has to tunnel between, and a *global*
# validity gate (nested-sampling insertion index, split-R-hat on slot-indexed
# params) fails the whole run because of it.
#
# The failure is measurable and it is usually LOCAL. On the HD 10180 blind
# run (g=8, n_live=1500) six of seven known planets pinned to a single slot
# and the entire pathology lived on ONE component. So the useful number is
# not "does this run label-switch" but "WHICH component switches, and how
# badly" — hence a per-component report, not a scalar.
#
# The measure is the Shannon entropy, in bits, of the distribution over
# *which slot holds a given component*, computed only over the samples where
# that component is occupied:
#
#     H_c = -Σ_k p(slot k holds c | c occupied) log2 p(...)
#
#   H = 0                 → the component is self-labelled (one slot always
#                           holds it); no permutation degeneracy to tunnel.
#   H = log2(m)           → m slots trade it uniformly; m equivalent modes.
#   H ≤ log2(n_slots)     → the ceiling.
#
# Empirical reading from the HD 10180 dissection: H < 0.5 bit = pinned,
# H > 1 bit = a genuine multi-slot interchange. Switching needs a component
# PRESENT (so several slots can claim it) but not DOMINANT (or one slot
# monopolises it), so expect it on the MIDDLE-confidence components — not on
# the strongest and not on the marginal ones.
#
# Reported alongside, because it is a different pathology that looks similar
# in a slot-indexed trace: `duplicate_frac`, the fraction of occupied samples
# in which TWO OR MORE active slots sit inside one component's band. That is
# near-duplicate splitting (a within-band degeneracy), not a permutation one,
# and a label-free/occupation-number reformulation does NOT remove it.
#
# Works on trans-dim chains (per-slot `planet_active_<k>`) and on fixed-dim
# multi-planet chains (all slots always active), since fixed-dim sampling
# over k planets carries the same k! symmetry.

using MCMCChains
using Statistics: median, mean
using Printf: @printf, @sprintf

export SlotSwitchStat, LabelSwitchingReport, label_switching,
       print_label_switching

# =====================================================================
# Result types
# =====================================================================

"""
    SlotSwitchStat

Per-component label-switching statistics.

- `label`          : component name (reference label, or `"P=<median> d"`)
- `period`         : reference / cluster-centre period [d]
- `p_occupied`     : P(component occupied | D)
- `entropy_bits`   : Shannon entropy of the holding-slot distribution [bit]
- `entropy_mm`     : Miller-Madow small-sample correction of `entropy_bits`
- `entropy_fixed_n`: switch entropy at FIXED active-slot count [bit]
- `mechanism`      : `:pinned` · `:relabelling` · `:permutation` (see below)
- `slot_probs`     : `slot => P(slot holds it | occupied)`, descending
- `duplicate_frac` : P(≥2 active slots inside the band | occupied)
- `n_occupied`     : number of samples in which the component is occupied

`entropy_bits` is the plug-in (maximum-likelihood) estimator — the quantity
reported in the literature and in the HD 10180 table, and what `status` keys
off. It is biased LOW by about `(m-1)/(2·n_occupied·ln2)` bits (`m` = number
of slots seen holding the component), and that bias grows as occupancy falls
— precisely the marginal-component regime, where it would make a switching
component look spuriously pinned. `entropy_mm` adds that correction back.
Compare the two whenever `n_occupied` is small; they agree to <0.01 bit once
`n_occupied` is in the thousands.

`entropy_fixed_n` separates the TWO mechanisms that both raise
`entropy_bits`:

  1. **permutation switching** — at a fixed model dimension, several slots
     trade the same component. This is the degeneracy a label-free /
     occupation-number reformulation dissolves.
  2. **dimension-induced relabelling** — births fill the lowest free slot, so
     the same physical component sits in slot 2 when Nₚ=2 and slot 3 when
     Nₚ=3. The raw entropy sees slot churn; there is no within-dimension
     permutation degeneracy to remove.

It is the occupancy-weighted mean of the per-`n_active` entropies, i.e. the
entropy that survives after conditioning on the active-slot count.
`mechanism` encodes the comparison:

| `entropy_bits` | `entropy_fixed_n` | `mechanism` | meaning |
|---------------:|------------------:|-------------|---------|
| high | high | `:permutation` | genuine permutation switching — a label-free / occupation-number reformulation dissolves it |
| high | ≪ raw | `:relabelling` | dimension-induced slot churn — label-free does NOT remove it |
| < 0.1 | — | `:pinned` | self-labelled |

Measured examples of both: HD 10180 g is `:permutation` (1.20 bit raw AND at
fixed N); GJ 876 b at 20 chains is `:relabelling` (0.60 raw → 0.05 fixed).

**Caveat for Nereus's own trans-dim chains.** The in-house engine enforces
canonical period-sorted labeling (`_sort_group_periods!` + the ordering
gate), so slot index = period RANK within the active set and true label
degeneracy is broken by construction. On such chains, `:permutation` means
the component's rank churns at fixed count — i.e. the active SET varies
(composition uncertainty: which planets are in the model), not slot labels.
That is genuine model uncertainty; a label-free reformulation would not (and
should not) remove it. The `:permutation` reading applies literally to
samplers WITHOUT canonical labeling (labelled MoMS/nested runs, external
chains with an explicit occupancy matrix, e.g. the HD 10180 npz). Note also
that PT trans-dim chains carry no `planet_active_*` columns, so the
order-assuming `n_planets` fallback below is what is actually exercised.
"""
struct SlotSwitchStat
    label::String
    period::Float64
    p_occupied::Float64
    entropy_bits::Float64
    entropy_mm::Float64
    entropy_fixed_n::Float64
    mechanism::Symbol
    slot_probs::Vector{Pair{Int, Float64}}
    duplicate_frac::Float64
    n_occupied::Int
end

"""
    LabelSwitchingReport

Aggregate returned by [`label_switching`](@ref).

- `components`   : per-component [`SlotSwitchStat`](@ref), ascending in period
- `n_samples`    : number of posterior draws examined
- `n_slots`      : number of candidate slots in the chain
- `mean_active`  : mean number of active slots per draw
- `max_entropy`  : worst per-component RAW entropy [bit] (0.0 if no components)
- `max_entropy_fixed` : worst per-component fixed-n entropy [bit]
- `ceiling`      : log2(n_slots) — the entropy a uniform full interchange gives
- `status`       : `:ok` (< 0.5 bit), `:warn` (< 1.0), `:fail` (≥ 1.0), on RAW
  entropy. This is the VALIDITY verdict: any slot churn — permutation or
  relabelling — breaks slot-indexed statistics (insertion index, per-slot
  split-R̂), so raw is what those gates feel. Comparable to published plug-in
  numbers.
- `fixed_status` : same thresholds on the FIXED-N entropy. This is the
  MECHANISM verdict: whether a genuine permutation degeneracy exists for the
  sampler to tunnel (and for a label-free construction to dissolve). `status`
  = fail with `fixed_status` = ok reads "slot-indexed gates are broken, but
  by dimension-relabelling — don't reach for second quantization".
- `mode`         : `:reference` (periods supplied) or `:blind` (KDE modes)
- `tol`          : fractional period tolerance — reference-mode band half-width
  and blind-mode KDE bandwidth (in log-period)
"""
struct LabelSwitchingReport
    components::Vector{SlotSwitchStat}
    n_samples::Int
    n_slots::Int
    mean_active::Float64
    max_entropy::Float64
    max_entropy_fixed::Float64
    ceiling::Float64
    status::Symbol
    fixed_status::Symbol
    mode::Symbol
    tol::Float64
end

# =====================================================================
# Chain extraction
# =====================================================================

# Slot indices present as `<prefix><k>` columns (default `P_k1`, `P_k2`, …).
function _slot_indices(chains, prefix::AbstractString)
    ks = Int[]
    for s in names(chains, :parameters)
        str = String(s)
        startswith(str, prefix) || continue
        k = tryparse(Int, str[nextind(str, lastindex(prefix)):end])
        k === nothing && continue
        push!(ks, k)
    end
    return sort!(unique!(ks))
end

# (periods, active) as n_samples × n_slots matrices.
#
# Activity precedence:
#   1. `planet_active_<k>`  — the truth for a trans-dim run.
#   2. `n_planets`          — ORDER-ASSUMING fallback (slot k active iff
#      k ≤ n_planets). Warns: if the sampler does not pack active slots into
#      the low indices this fabricates exactly the structure being measured.
#   3. neither              — fixed-dim chain, every slot always active.
function _slot_matrices(chains, prefix::AbstractString)
    ks = _slot_indices(chains, prefix)
    isempty(ks) && throw(ArgumentError(
        "no `$(prefix)<k>` columns in the chain — nothing to measure " *
        "(pass `period_prefix=` if this run names periods differently)"))
    allnames = names(chains, :parameters)
    P = hcat((vec(Array(chains[Symbol(prefix, k)])) for k in ks)...)
    n = size(P, 1)

    have_active = all(Symbol("planet_active_$k") in allnames for k in ks)
    if have_active
        A = hcat((vec(Array(chains[Symbol("planet_active_$k")])) .> 0.5
                  for k in ks)...)
    elseif :n_planets in allnames
        np = vec(Array(chains[:n_planets]))
        @warn "no `planet_active_*` columns; falling back to `n_planets ≥ k`. " *
              "This ASSUMES active slots occupy the low indices — if the " *
              "sampler does not pack them, the switch entropy below is an " *
              "artifact of the fallback, not a measurement."
        A = hcat((np .>= k for k in ks)...)
    else
        A = trues(n, length(ks))
    end
    return P, A, ks
end

# =====================================================================
# Blind component discovery
# =====================================================================

# KDE + watershed over the pooled ACTIVE log-periods. Bandwidth = log1p(tol)
# (same fractional units as the reference-mode band half-width), density on a
# binned grid, modes from `_local_maxima` (periodogram/peaks.jl), weak modes
# merged by prominence, and component BANDS = watershed basins (the density
# valleys between retained modes) rather than fixed ±tol windows.
#
# Why not single linkage (the previous implementation): it has two opposite
# failure modes on wide-dynamic-range systems, both measured on HD 10180
# (1.18–2222 d). A component with a BROAD posterior (g, ~601 d) fragments
# into sub-`tol` shards that each fall below `min_occupancy` — at tol=5% the
# pooled occupancy summed to 4.43 of 6.18 active slots/draw and g was missed
# entirely; loosening to tol=20% instead chained the whole period axis into
# ONE cluster. There is no tol that works. A density valley is the right cut:
# a broad unimodal posterior stays one basin no matter how wide, and two
# resolved components stay separate no matter how close their tails.
#
# Prominence rule: adjacent modes are merged (lower one dropped) while the
# valley between them stays above 0.8× the lower peak. Calibration, from the
# valley/peak ratio of two equal Gaussians at separation d: r(2σ)≈1 (barely
# bimodal), r(2.7σ)≈0.80, r(3σ)≈0.66, r(4σ)≈0.27 — so 0.8 splits modes
# separated by ≳2.7 bandwidths. Finite-sample wiggle on ONE broad mode sits
# near r≈0.95 (relative KDE noise ~1/√n_eff per bandwidth), safely above the
# cut, so a plateau does not fragment.
function _blind_components(P, A, tol)
    vals = Float64[]
    @inbounds for i in eachindex(A)
        (A[i] && isfinite(P[i]) && P[i] > 0) && push!(vals, log(P[i]))
    end
    isempty(vals) && return NTuple{3, Float64}[]
    σ = log1p(tol)
    lo, hi = extrema(vals)
    lo -= 4σ; hi += 4σ
    ngrid = 2048
    step = (hi - lo) / (ngrid - 1)

    # Binned KDE: histogram, then convolve with a Gaussian kernel.
    hist = zeros(Float64, ngrid)
    @inbounds for v in vals
        hist[clamp(round(Int, (v - lo) / step) + 1, 1, ngrid)] += 1.0
    end
    hw = ceil(Int, 4σ / step)
    kern = [exp(-0.5 * (j * step / σ)^2) for j in -hw:hw]
    dens = zeros(Float64, ngrid)
    @inbounds for i in 1:ngrid
        hist[i] == 0.0 && continue
        j0, j1 = max(1, i - hw), min(ngrid, i + hw)
        for j in j0:j1
            dens[j] += hist[i] * kern[j - i + hw + 1]
        end
    end

    # Modes, then prominence merging: repeatedly drop the peak whose valley
    # to its neighbour is least prominent, until every adjacent pair is
    # separated by a valley below 0.8× the lower peak.
    peaks = _local_maxima(dens)
    isempty(peaks) && (peaks = [argmax(dens)])
    valley(a, b) = minimum(@view dens[a:b])
    merged = true
    while merged && length(peaks) > 1
        merged = false
        for i in 1:(length(peaks) - 1)
            pl, pr = peaks[i], peaks[i + 1]
            if valley(pl, pr) > 0.8 * min(dens[pl], dens[pr])
                deleteat!(peaks, dens[pl] <= dens[pr] ? i : i + 1)
                merged = true
                break
            end
        end
    end

    # Watershed: basin boundaries at the deepest valley between retained
    # modes; outer basins extend to ±∞ so every active draw is assignable.
    bounds = Float64[-Inf]
    for i in 1:(length(peaks) - 1)
        seg = peaks[i]:peaks[i + 1]
        push!(bounds, lo + (seg[argmin(@view dens[seg])] - 1) * step)
    end
    push!(bounds, Inf)

    # Centre = median of the pooled values inside each basin.
    comps = NTuple{3, Float64}[]
    for i in eachindex(peaks)
        inb = filter(v -> bounds[i] < v <= bounds[i + 1], vals)
        isempty(inb) && continue
        push!(comps, (exp(median(inb)), bounds[i], bounds[i + 1]))
    end
    return comps
end

# =====================================================================
# Core
# =====================================================================

"""
    label_switching(chains; reference_periods=nothing, labels=nothing,
                    tol=0.05, min_occupancy=0.02,
                    period_prefix="P_k") -> LabelSwitchingReport

Measure the per-component slot-permutation degeneracy of an
exchangeable-slot run.

For each component, over the draws in which it is occupied, build the
distribution over *which slot holds it* and report that distribution's
Shannon entropy in bits. Zero means the component is self-labelled; `log2(m)`
means `m` slots trade it. See the file header for the reading.

**Component definition.** A slot holds component `c` in a draw when the slot
is active and its period sits inside `c`'s band. If several qualify, the
closest in log-period to the band centre is credited as the holder and the
draw is counted in `duplicate_frac`.

**Modes.**
- `reference_periods` supplied → measure exactly those components (the
  published-orbit table; use `labels` for names); each band is ±`tol`
  (fractional) around the given period. Components the run never finds come
  back with `p_occupied ≈ 0`.
- omitted → blind: discover components as the modes of a KDE over the pooled
  active log-periods (bandwidth `log1p(tol)`), with bands = the watershed
  basins between modes — so a component with a posterior much broader than
  `tol` is still ONE component (single-linkage clustering, the previous
  implementation, fragmented broad components and missed them entirely on
  HD 10180). Components below `min_occupancy` are dropped.

Accepts trans-dim chains (`planet_active_<k>`) and fixed-dim multi-planet
chains (every slot always active — the k! symmetry is there too).

```julia
rep = label_switching(chains)                                   # blind
rep = label_switching(chains; reference_periods=[5.76, 16.36],  # known orbits
                              labels=["c", "d"])
print_label_switching(rep)
```
"""
function label_switching(chains;
                          reference_periods::Union{Nothing, AbstractVector{<:Real}}=nothing,
                          labels::Union{Nothing, AbstractVector{<:AbstractString}}=nothing,
                          tol::Real=0.05,
                          min_occupancy::Real=0.02,
                          period_prefix::AbstractString="P_k")
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))
    P, A, ks = _slot_matrices(chains, period_prefix)
    n, g = size(P)

    cut = log1p(tol)
    mode = reference_periods === nothing ? :blind : :reference
    # Bands as (centre, lo, hi) in log-period: watershed basins in blind
    # mode, symmetric ±tol windows around the published periods in reference
    # mode.
    bands = if mode === :blind
        _blind_components(P, A, tol)
    else
        [(p = Float64(rp); isfinite(p) && p > 0 ?
              (p, log(p) - cut, log(p) + cut) : (NaN, NaN, NaN))
         for rp in reference_periods]
    end
    if labels !== nothing && mode === :reference
        length(labels) == length(bands) || throw(ArgumentError(
            "labels has $(length(labels)) entries but reference_periods has " *
            "$(length(bands))"))
    end

    nact = vec(sum(A; dims = 2))          # active-slot count per draw
    comps = SlotSwitchStat[]
    for (ci, (pref, blo, bhi)) in enumerate(bands)
        (isfinite(pref) && pref > 0) || continue
        lref = log(pref)
        counts = zeros(Int, g)
        n_occ = 0
        n_dup = 0
        # (n_active, slot) tallies, for the fixed-dimension entropy.
        by_n = Dict{Int, Vector{Int}}()
        @inbounds for s in 1:n
            best_k, best_d, n_in = 0, Inf, 0
            for k in 1:g
                (A[s, k] && isfinite(P[s, k]) && P[s, k] > 0) || continue
                lp = log(P[s, k])
                blo <= lp <= bhi || continue
                d = abs(lp - lref)
                n_in += 1
                if d < best_d
                    best_d, best_k = d, k
                end
            end
            n_in == 0 && continue
            n_occ += 1
            n_in > 1 && (n_dup += 1)
            counts[best_k] += 1
            v = get!(by_n, nact[s]) do; zeros(Int, g) end
            v[best_k] += 1
        end

        p_occ = n_occ / n
        mode === :blind && p_occ < min_occupancy && continue

        # Entropy + holding-slot distribution over the OCCUPIED draws only.
        # Plug-in estimator, plus the Miller-Madow bias correction: the
        # plug-in value is biased LOW by ≈(m-1)/(2n ln2), which matters
        # exactly where occupancy is thin.
        H = 0.0
        sp = Pair{Int, Float64}[]
        if n_occ > 0
            for k in 1:g
                counts[k] == 0 && continue
                p = counts[k] / n_occ
                H -= p * log2(p)
                push!(sp, ks[k] => p)
            end
            sort!(sp; by = x -> -x.second)
        end
        H_mm = n_occ > 0 && !isempty(sp) ?
               H + (length(sp) - 1) / (2 * n_occ * log(2)) : 0.0

        # Entropy that survives conditioning on the active-slot count:
        # occupancy-weighted mean of the within-n_active entropies. Strips
        # the "births fill the lowest free slot" relabelling and leaves only
        # genuine same-dimension permutation switching.
        H_fixed = 0.0
        if n_occ > 0
            for (_, v) in by_n
                sub = sum(v)
                sub == 0 && continue
                h = 0.0
                for c in v
                    c == 0 && continue
                    p = c / sub
                    h -= p * log2(p)
                end
                H_fixed += (sub / n_occ) * h
            end
        end

        lbl = if labels !== nothing && mode === :reference
            String(labels[ci])
        else
            @sprintf("P=%.4g d", pref)
        end
        mech = H < 0.1            ? :pinned :
               H_fixed < 0.5 * H  ? :relabelling : :permutation
        push!(comps, SlotSwitchStat(lbl, Float64(pref), p_occ, H, H_mm, H_fixed,
                                    mech, sp, n_occ > 0 ? n_dup / n_occ : 0.0,
                                    n_occ))
    end
    sort!(comps; by = c -> c.period)

    mean_active = n > 0 ? sum(A) / n : 0.0
    maxH  = isempty(comps) ? 0.0 : maximum(c.entropy_bits for c in comps)
    maxHf = isempty(comps) ? 0.0 : maximum(c.entropy_fixed_n for c in comps)
    verdict(h) = h < 0.5 ? :ok : (h < 1.0 ? :warn : :fail)
    return LabelSwitchingReport(comps, n, g, mean_active, maxH, maxHf,
                                g > 0 ? log2(g) : 0.0, verdict(maxH),
                                verdict(maxHf), mode, Float64(tol))
end

# =====================================================================
# Reporting
# =====================================================================

"""
    print_label_switching(rep::LabelSwitchingReport; io=stdout)

Print the per-component table: occupancy, switch entropy, holding slots, and
near-duplicate fraction, ordered by ascending period.
"""
function print_label_switching(rep::LabelSwitchingReport; io::IO=stdout)
    lbl(s) = s === :ok ? "OK" : (s === :warn ? "WARN" : "FAIL")
    mark = rep.status === rep.fixed_status ? lbl(rep.status) :
           string(lbl(rep.status), " raw · ", lbl(rep.fixed_status), " fixed-n")
    println(io, "Label-switching diagnostic  [", mark, "]")
    @printf(io, "  %d draws · %d slots · %.2f active slots/draw · mode=%s · tol=%.1f%%\n",
            rep.n_samples, rep.n_slots, rep.mean_active, rep.mode, 100 * rep.tol)
    @printf(io, "  max switch entropy %.2f bit raw, %.2f at fixed n (ceiling %.2f bit)\n",
            rep.max_entropy, rep.max_entropy_fixed, rep.ceiling)
    if isempty(rep.components)
        println(io, "  (no components found)")
        return nothing
    end
    println(io, "")
    @printf(io, "  %-14s %8s %10s %10s %7s  %s\n",
            "component", "P(occ)", "H [bit]", "H|fixed n", "dup", "holding slots")
    for c in rep.components
        slots = isempty(c.slot_probs) ? "—" :
                join((@sprintf("s%d:%.2f", p.first, p.second)
                      for p in Iterators.take(c.slot_probs, 4)), ", ")
        length(c.slot_probs) > 4 && (slots *= ", …")
        @printf(io, "  %-14s %8.2f %10.2f %10.2f %7.2f  %s\n",
                c.label, c.p_occupied, c.entropy_bits, c.entropy_fixed_n,
                c.duplicate_frac, slots)
    end
    # Where the raw entropy is mostly slot churn induced by dimension jumps,
    # say so — a label-free reformulation would NOT remove that part.
    relab = filter(c -> c.mechanism === :relabelling, rep.components)
    if !isempty(relab)
        println(io, "")
        println(io, "  mostly DIMENSION-RELABELLING, not permutation switching")
        println(io, "  (slot index tracks the active-slot count; conditioning on it removes it):")
        for c in relab
            @printf(io, "    %-14s H = %.2f → %.2f bit at fixed n\n",
                    c.label, c.entropy_bits, c.entropy_fixed_n)
        end
    end
    # Flag thin-occupancy components whose plug-in entropy is materially
    # bias-suppressed — otherwise they read as pinned when they are not.
    thin = filter(c -> c.entropy_mm - c.entropy_bits > 0.05, rep.components)
    if !isempty(thin)
        println(io, "")
        println(io, "  small-sample bias (plug-in H is low; Miller-Madow in brackets):")
        for c in thin
            @printf(io, "    %-14s H = %.2f [%.2f] bit  from %d occupied draws\n",
                    c.label, c.entropy_bits, c.entropy_mm, c.n_occupied)
        end
    end
    if rep.fixed_status !== :ok
        worst = argmax(c -> c.entropy_fixed_n, rep.components)
        println(io, "")
        @printf(io, "  ⇒ %s carries a GENUINE permutation degeneracy (%.2f bit at fixed n,\n",
                worst.label, worst.entropy_fixed_n)
        @printf(io, "    %d slots trading it). Slot-indexed convergence gates (insertion\n",
                length(worst.slot_probs))
        println(io, "    index, per-slot split-R̂) will fail on this component alone;")
        println(io, "    a label-free (occupation-number) formulation dissolves it.")
    elseif rep.status !== :ok
        worst = argmax(c -> c.entropy_bits, rep.components)
        println(io, "")
        @printf(io, "  ⇒ %s churns slots (%.2f bit raw) but by DIMENSION-RELABELLING,\n",
                worst.label, worst.entropy_bits)
        println(io, "    not permutation switching. Slot-indexed statistics are still")
        println(io, "    unreliable, but a label-free formulation would NOT remove this —")
        println(io, "    condition per-slot summaries on the active-slot count instead.")
    end
    return nothing
end

function Base.show(io::IO, rep::LabelSwitchingReport)
    print(io, "LabelSwitchingReport(", rep.status, " raw/", rep.fixed_status,
          " fixed, ", length(rep.components), " components, max H = ",
          @sprintf("%.2f", rep.max_entropy), "/",
          @sprintf("%.2f", rep.max_entropy_fixed), " bit)")
end
