# label_switching — per-component slot-permutation entropy.
#
# Every case below plants a KNOWN switching structure in a synthetic chain and
# checks the measured entropy against the analytic value, so a regression shows
# up as a number that moved, not as a vibe. Assignments are deterministic (not
# sampled) so the expected entropies are exact.
#
# Covered: pinned slots (0 bit), 2-way (1 bit), 3-way uniform (log2 3), the
# HD 10180 "planet g" non-uniform split (1.21 bit), stale junk in INACTIVE
# slots (must not register as occupancy), near-duplicate splitting, absent
# reference components, blind discovery, fixed-dim chains, and the
# order-assuming n_planets fallback.

using Nereus, Test, MCMCChains
using Random: MersenneTwister
using Statistics: mean

# Build a chain from a per-draw assignment.
#
# `assignments[s]` is a `slot => period` collection for the ACTIVE slots of
# draw `s`. Inactive slots get `junk` (stale-junk emulation: trans-dim inactive
# slots retain whatever the last death left behind, and that value can sit
# right on top of a real component).
function _mk_chain(assignments, g; junk = 137.0, jitter = 0.005,
                    with_active = true, with_np = false, seed = 4)
    rng = MersenneTwister(seed)
    n = length(assignments)
    P = fill(junk, n, g)
    A = falses(n, g)
    for (s, asg) in enumerate(assignments)
        for (k, p) in asg
            P[s, k] = p * (1 + jitter * (2 * rand(rng) - 1))
            A[s, k] = true
        end
    end
    cols = Symbol[]
    blocks = Matrix{Float64}[]
    push!(blocks, P); append!(cols, [Symbol("P_k$k") for k in 1:g])
    if with_active
        push!(blocks, Float64.(A))
        append!(cols, [Symbol("planet_active_$k") for k in 1:g])
    end
    if with_np
        push!(blocks, reshape(Float64.(vec(sum(A; dims = 2))), n, 1))
        push!(cols, :n_planets)
    end
    return Chains(reshape(hcat(blocks...), n, length(cols), 1), cols)
end

_comp(rep, label) = rep.components[findfirst(c -> c.label == label, rep.components)]
_above(rep, P) = first(filter(c -> c.period > P, rep.components))

@testset "label_switching — slot-permutation entropy" begin

    # -----------------------------------------------------------------
    @testset "pinned slots => 0 bit, :ok" begin
        # Three components, each permanently in its own slot. No permutation
        # degeneracy exists, so every entropy must be exactly zero.
        n, g = 600, 5
        asg = [[1 => 5.0, 2 => 20.0, 3 => 100.0] for _ in 1:n]
        rep = label_switching(_mk_chain(asg, g))

        @test rep isa LabelSwitchingReport
        @test rep.status === :ok
        @test length(rep.components) == 3
        @test all(c -> c.entropy_bits == 0.0, rep.components)
        @test all(c -> c.p_occupied ≈ 1.0, rep.components)
        # ascending in period, and the cluster centres land on the truth
        @test [c.period for c in rep.components] ≈ [5.0, 20.0, 100.0] rtol = 0.01
        @test rep.n_slots == 5
        @test rep.ceiling ≈ log2(5)
        @test rep.mean_active ≈ 3.0
        @test rep.mode === :blind
    end

    # -----------------------------------------------------------------
    @testset "2-way interchange => exactly 1 bit" begin
        # The 100 d component alternates between slots 3 and 4, 50/50.
        n, g = 600, 5
        asg = [[1 => 5.0, 2 => 20.0, (iseven(s) ? 3 : 4) => 100.0] for s in 1:n]
        rep = label_switching(_mk_chain(asg, g))

        @test rep.status === :fail                     # ≥ 1.0 bit
        pinned = filter(c -> c.period < 50, rep.components)
        @test all(c -> c.entropy_bits == 0.0, pinned)  # the others stay pinned
        sw = _above(rep, 50)
        @test sw.entropy_bits ≈ 1.0 atol = 1e-12
        @test sw.p_occupied ≈ 1.0
        @test length(sw.slot_probs) == 2
        @test all(p -> p.second ≈ 0.5, sw.slot_probs)
        @test Set(p.first for p in sw.slot_probs) == Set([3, 4])
        @test sw.duplicate_frac == 0.0
        @test rep.max_entropy ≈ 1.0 atol = 1e-12
    end

    # -----------------------------------------------------------------
    @testset "3-way uniform => log2(3)" begin
        n, g = 600, 6
        asg = [[1 => 5.0, (2 + s % 3) => 100.0] for s in 1:n]
        rep = label_switching(_mk_chain(asg, g))
        sw = _above(rep, 50)

        @test sw.entropy_bits ≈ log2(3) atol = 1e-12
        @test length(sw.slot_probs) == 3
        @test rep.status === :fail
    end

    # -----------------------------------------------------------------
    @testset "HD 10180 planet-g split (0.57/0.38/0.05) => 1.21 bit" begin
        # Reproduces the non-uniform 3-slot interchange dissected in
        # paper/second_quantization_exchangeable_slots.md (reported 1.22 bit).
        n, g = 1000, 8
        counts = [(6, 570), (2, 380), (5, 50)]
        slots = reduce(vcat, [fill(k, c) for (k, c) in counts])
        asg = [[1 => 5.0, slots[s] => 600.0] for s in 1:n]
        rep = label_switching(_mk_chain(asg, g))
        sw = _above(rep, 100)

        H_exact = -(0.57log2(0.57) + 0.38log2(0.38) + 0.05log2(0.05))
        @test sw.entropy_bits ≈ H_exact atol = 1e-12
        @test round(sw.entropy_bits; digits = 2) == 1.21
        @test [p.first for p in sw.slot_probs] == [6, 2, 5]      # descending
        @test [p.second for p in sw.slot_probs] ≈ [0.57, 0.38, 0.05]
        @test rep.status === :fail
    end

    # -----------------------------------------------------------------
    @testset "stale junk in INACTIVE slots does not register" begin
        # Slot 5 is never active but permanently parked on the 100 d component
        # — the documented trans-dim inactive-slot behaviour. Occupancy and
        # entropy must both ignore it entirely.
        n, g = 600, 5
        asg = [[1 => 5.0, 3 => 100.0] for _ in 1:n]
        ch = _mk_chain(asg, g; junk = 100.0)          # ALL inactive slots at 100 d
        rep = label_switching(ch)

        sw = _above(rep, 50)
        @test sw.entropy_bits == 0.0                  # only slot 3 counted
        @test sw.slot_probs == [3 => 1.0]
        @test sw.duplicate_frac == 0.0
        @test rep.mean_active ≈ 2.0
        @test rep.status === :ok
    end

    # -----------------------------------------------------------------
    @testset "near-duplicate splitting is counted separately from switching" begin
        # Two ACTIVE slots inside one band on 25% of draws. That is a
        # within-band degeneracy, not a permutation one: the holder is the
        # closest slot, so entropy stays low while duplicate_frac fires.
        n, g = 800, 5
        asg = [s % 4 == 0 ? [1 => 5.0, 3 => 100.0, 4 => 101.5] :
                            [1 => 5.0, 3 => 100.0] for s in 1:n]
        rep = label_switching(_mk_chain(asg, g; jitter = 0.0))
        sw = _above(rep, 50)

        @test sw.duplicate_frac ≈ 0.25 atol = 1e-12
        @test sw.entropy_bits == 0.0                  # slot 3 is always closest
        @test sw.slot_probs == [3 => 1.0]
    end

    # -----------------------------------------------------------------
    @testset "reference mode: absent component, labels, occupancy" begin
        # b-like marginal component present in only 5% of draws, plus a
        # reference period the run never occupies at all.
        n, g = 1000, 6
        asg = [s <= 50 ? [1 => 5.0, 2 => 50.0] : [1 => 5.0] for s in 1:n]
        ch = _mk_chain(asg, g)
        rep = label_switching(ch; reference_periods = [5.0, 50.0, 999.0],
                                   labels = ["c", "b", "ghost"])

        @test rep.mode === :reference
        @test length(rep.components) == 3             # nothing dropped in ref mode
        @test _comp(rep, "c").p_occupied ≈ 1.0
        @test _comp(rep, "b").p_occupied ≈ 0.05
        @test _comp(rep, "b").n_occupied == 50
        @test _comp(rep, "ghost").p_occupied == 0.0
        @test _comp(rep, "ghost").entropy_bits == 0.0
        @test isempty(_comp(rep, "ghost").slot_probs)
        @test rep.status === :ok

        @test_throws ArgumentError label_switching(ch;
            reference_periods = [5.0, 50.0], labels = ["only-one"])
    end

    # -----------------------------------------------------------------
    @testset "blind mode drops sub-threshold components" begin
        n, g = 1000, 6
        asg = [s <= 10 ? [1 => 5.0, 2 => 50.0] : [1 => 5.0] for s in 1:n]
        ch = _mk_chain(asg, g)

        @test length(label_switching(ch; min_occupancy = 0.05).components) == 1
        @test length(label_switching(ch; min_occupancy = 0.005).components) == 2
    end

    # -----------------------------------------------------------------
    @testset "tol controls band membership and clustering" begin
        # Two components 3% apart: merged at the 5% default, resolved at 1%.
        n, g = 600, 4
        asg = [[1 => 100.0, 2 => 103.0] for _ in 1:n]
        ch = _mk_chain(asg, g; jitter = 0.0)

        @test length(label_switching(ch; tol = 0.05).components) == 1
        rep_tight = label_switching(ch; tol = 0.01)
        @test length(rep_tight.components) == 2
        @test all(c -> c.entropy_bits == 0.0, rep_tight.components)
        # Merged at 5%: both slots land in one band => reads as duplicates,
        # NOT as switching (the closest slot is credited each draw).
        rep_wide = label_switching(ch; tol = 0.05)
        @test rep_wide.components[1].duplicate_frac ≈ 1.0
        @test_throws ArgumentError label_switching(ch; tol = 0.0)
    end

    # -----------------------------------------------------------------
    @testset "fixed-dim chain: every slot active, k! symmetry still measured" begin
        # No planet_active_*, no n_planets — a plain fixed-dim 2-planet chain
        # whose two slots trade the same pair of periods.
        n, g = 600, 2
        asg = [iseven(s) ? [1 => 5.0, 2 => 100.0] : [1 => 100.0, 2 => 5.0]
               for s in 1:n]
        rep = label_switching(_mk_chain(asg, g; with_active = false))

        @test rep.n_slots == 2
        @test rep.mean_active ≈ 2.0
        @test length(rep.components) == 2
        @test all(c -> c.entropy_bits ≈ 1.0, rep.components)   # full 2! swap
        @test rep.max_entropy ≈ 1.0
        @test rep.ceiling ≈ 1.0                                 # log2(2)
        @test rep.status === :fail
    end

    # -----------------------------------------------------------------
    @testset "n_planets fallback warns (it can fabricate the signal)" begin
        n, g = 400, 4
        asg = [[1 => 5.0, 2 => 20.0] for _ in 1:n]
        ch = _mk_chain(asg, g; with_active = false, with_np = true)
        rep = @test_logs (:warn,) match_mode = :any label_switching(ch)
        @test rep.mean_active ≈ 2.0
        @test all(c -> c.entropy_bits == 0.0, rep.components)
    end

    # -----------------------------------------------------------------
    @testset "entropy_fixed_n separates relabelling from switching" begin
        g = 5
        # (a) PURE DIMENSION-RELABELLING. The 100 d component sits in slot 2
        # when it is the only other planet and in slot 3 when a 20 d planet
        # is also active — "births fill the lowest free slot". Raw entropy
        # sees 50/50 slot churn (1 bit); at FIXED active-slot count each
        # group is deterministic, so the conditioned entropy is 0.
        asg_rel = [iseven(s) ? [1 => 5.0, 2 => 100.0] :
                               [1 => 5.0, 2 => 20.0, 3 => 100.0] for s in 1:800]
        rep_rel = label_switching(_mk_chain(asg_rel, g))
        rel = _above(rep_rel, 50)
        @test rel.entropy_bits ≈ 1.0 atol = 1e-12
        @test rel.entropy_fixed_n ≈ 0.0 atol = 1e-12
        @test rel.mechanism === :relabelling
        # Report level: raw says the slot statistics are broken (validity),
        # fixed-n says there is no permutation degeneracy behind it.
        @test rep_rel.status === :fail
        @test rep_rel.fixed_status === :ok

        # (b) GENUINE PERMUTATION SWITCHING at constant dimension: always
        # exactly two active slots, and the 100 d component trades between
        # slots 2 and 3 while 5 d takes whichever is left. Conditioning on
        # the count removes nothing.
        asg_sw = [iseven(s) ? [2 => 100.0, 3 => 5.0] :
                              [3 => 100.0, 2 => 5.0] for s in 1:800]
        rep_sw = label_switching(_mk_chain(asg_sw, g))
        sw = _above(rep_sw, 50)
        @test sw.entropy_bits ≈ 1.0 atol = 1e-12
        @test sw.entropy_fixed_n ≈ 1.0 atol = 1e-12
        @test sw.mechanism === :permutation
        @test rep_sw.status === :fail
        @test rep_sw.fixed_status === :fail

        # The printed report distinguishes them.
        io = IOBuffer(); print_label_switching(label_switching(_mk_chain(asg_rel, g)); io = io)
        @test occursin("DIMENSION-RELABELLING", String(take!(io)))
        io2 = IOBuffer(); print_label_switching(label_switching(_mk_chain(asg_sw, g)); io = io2)
        @test !occursin("DIMENSION-RELABELLING", String(take!(io2)))

        # Pinned components have nothing to separate.
        pin = label_switching(_mk_chain([[1 => 5.0, 2 => 100.0] for _ in 1:300], g))
        @test all(c -> c.entropy_fixed_n == 0.0, pin.components)
        @test all(c -> c.mechanism === :pinned, pin.components)
        @test pin.fixed_status === :ok
    end

    # -----------------------------------------------------------------
    @testset "blind KDE: broad component is ONE basin (single-linkage killer)" begin
        # The HD 10180 g failure in miniature: a component whose posterior is
        # much BROADER than tol (σ = 12% in log-period vs tol = 5%). Gap-based
        # single linkage fragmented it into sub-tol shards that each fell
        # below min_occupancy — at tol=5% it missed g entirely, and no tol
        # fixed it (20% chained the whole axis into one cluster). A density
        # basin has no width limit, so the broad component must come back as
        # exactly one component at the DEFAULT tol, with its full occupancy.
        rng = MersenneTwister(11)
        n, g = 4000, 6
        asg = [[1 => 5.0, 2 => 16.36, 3 => 600.0 * exp(0.12 * randn(rng))]
               for _ in 1:n]
        rep = label_switching(_mk_chain(asg, g))

        @test length(rep.components) == 3
        broad = _above(rep, 300)
        @test broad.p_occupied ≈ 1.0             # nothing fell out of the basin
        @test broad.entropy_bits == 0.0          # pinned to slot 3 throughout
        @test 500 < broad.period < 720           # centre near the truth
        # narrow companions unharmed
        @test _above(rep, 10).period ≈ 16.36 rtol = 0.02
        @test rep.status === :ok

        # Same broad component now TRADED between slots 3 and 4 — the
        # switching must be measured across the whole basin, not a ±tol core.
        asg2 = [[1 => 5.0, (iseven(s) ? 3 : 4) => 600.0 * exp(0.12 * randn(rng))]
                for s in 1:n]
        sw = _above(label_switching(_mk_chain(asg2, g)), 300)
        @test sw.p_occupied ≈ 1.0
        @test sw.entropy_bits ≈ 1.0 atol = 0.01
        @test sw.mechanism === :permutation
    end

    # -----------------------------------------------------------------
    @testset "Miller-Madow correction tracks occupancy" begin
        # Same 3-way split, thin vs thick occupancy. The plug-in entropy is
        # identical (same proportions); the bias correction is not, and it
        # must shrink as n_occupied grows. Analytic: (m-1)/(2n ln2).
        g = 6
        mk(reps) = [[1 => 5.0, (2 + s % 3) => 100.0] for s in 1:(3 * reps)]

        thin = _above(label_switching(_mk_chain(mk(10), g);
                                       min_occupancy = 0.0), 50)   # 30 occupied
        thick = _above(label_switching(_mk_chain(mk(2000), g);
                                        min_occupancy = 0.0), 50)  # 6000 occupied

        @test thin.n_occupied == 30
        @test thick.n_occupied == 6000
        @test thin.entropy_bits ≈ thick.entropy_bits ≈ log2(3)      # plug-in equal
        @test thin.entropy_mm ≈ log2(3) + 2 / (2 * 30 * log(2))
        @test thick.entropy_mm ≈ log2(3) + 2 / (2 * 6000 * log(2))
        @test thin.entropy_mm - thin.entropy_bits >
              thick.entropy_mm - thick.entropy_bits                 # shrinks with n
        @test thick.entropy_mm - thick.entropy_bits < 0.01          # negligible when thick

        # A component nobody holds has no entropy to correct.
        ch = _mk_chain([[1 => 5.0] for _ in 1:200], g)
        ghost = label_switching(ch; reference_periods = [900.0], labels = ["x"])
        @test ghost.components[1].entropy_mm == 0.0

        # The printed report flags the correction only when it is material
        # relative to the 0.5/1.0-bit decision thresholds (cut: 0.05 bit).
        # 15 occupied draws over 3 slots => 0.096 bit, flagged;
        # 30 occupied draws over 3 slots => 0.048 bit, not worth the noise.
        _printed(reps) = (io = IOBuffer();
            print_label_switching(label_switching(_mk_chain(mk(reps), g);
                                                   min_occupancy = 0.0); io = io);
            String(take!(io)))
        @test 2 / (2 * 15 * log(2)) > 0.05 > 2 / (2 * 30 * log(2))   # the straddle
        @test occursin("small-sample bias", _printed(5))
        @test !occursin("small-sample bias", _printed(10))
    end

    # -----------------------------------------------------------------
    @testset "errors and printing" begin
        n = 100
        bare = Chains(reshape(randn(MersenneTwister(1), n, 1), n, 1, 1), [:K])
        @test_throws ArgumentError label_switching(bare)

        asg = [[1 => 5.0, (iseven(s) ? 2 : 3) => 100.0] for s in 1:200]
        rep = label_switching(_mk_chain(asg, 4))
        io = IOBuffer()
        print_label_switching(rep; io = io)
        out = String(take!(io))
        @test occursin("FAIL", out)
        @test occursin("permutation degeneracy", out)
        @test occursin("s2:", out) && occursin("s3:", out)

        io2 = IOBuffer(); show(io2, rep)
        @test occursin("LabelSwitchingReport", String(take!(io2)))
    end
end
