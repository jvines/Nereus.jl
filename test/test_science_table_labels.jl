# Units and LaTeX symbols in the science tables.
#
# Every failure below shipped in a real table. A gaia4 astrometric fit wrote
#
#     $\mathrm{inc}\,\mathrm{k1}$ & $2.114$          & deg    <- radians, labelled deg
#     $\omega\,\mathrm{k1}$       & $176.5^{+178.5}$ & deg    <- that is Omega, not omega
#     $\mathrm{M}\,\mathrm{sec}\,\mathrm{k1}$ & ...  &        <- no unit at all
#     $\mathrm{plx}$              & $13.624$         &        <- no unit at all
#
# for an inclination of 121.11 deg and a longitude of the ascending node. None
# of it is catchable by a reader of the table, which is what makes it worth a
# test rather than a comment.

using Test, Nereus

@testset "science table units" begin
    # inc and i are SAMPLED IN RADIANS and reported in degrees, so the
    # convert-from-radians flag must be set. `("deg", false)` means "call it
    # degrees and never convert".
    for nm in ("inc", "i")
        unit, is_rad = Nereus._SCI_UNITS[nm]
        @test unit == "deg"
        @test is_rad
    end
    # inc_deg is a DERIVED key, already in degrees: it must NOT be converted.
    @test Nereus._SCI_UNITS["inc_deg"] == ("deg", false)

    # Angles that were already right stay right.
    for nm in ("w", "omega", "Mo", "M0", "Omega", "lambda")
        @test Nereus._SCI_UNITS[nm] == ("deg", true)
    end

    # Sampled in astrometry, absent from the registry, so both published as
    # dimensionless.
    @test Nereus._SCI_UNITS["plx"]   == ("mas", false)
    @test Nereus._SCI_UNITS["M_sec"] == ("M_sun", false)
end

@testset "science table LaTeX symbols" begin
    # The ascending node is a CAPITAL omega. The greek lookup lowercased every
    # token first, so Omega_k1 was published as \omega -- the argument of
    # periastron, a different angle, in the same table.
    @test Nereus._texsym("Omega_k1") == "\${\\Omega}_{1}\$"
    @test occursin("\\Omega", Nereus._texsub("Omega_k1"))
    # ... without breaking the lowercase one.
    @test occursin("\\omega", Nereus._texsub("omega_k1"))
    @test !occursin("\\Omega", Nereus._texsub("omega_k1"))

    # The symbol table keyed only the DERIVED spellings (a_au, inc_deg), so
    # the FITTED table fell through to \mathrm{} for half its rows.
    @test Nereus._texsym("a_k1")     == "\${a}_{1}\$"
    @test Nereus._texsym("inc_k1")   == "\${i}_{1}\$"
    @test Nereus._texsym("M_sec_k1") == "\${M_{\\rm sec}}_{1}\$"
    @test Nereus._texsym("plx")      == "\$\\varpi\$"

    # Untouched: the ones that were already correct.
    @test Nereus._texsym("sesinw_k1") == "\${\\sqrt{e}\\sin\\omega}_{1}\$"
    @test Nereus._texsym("Mo_k1")     == "\${M_0}_{1}\$"
    @test Nereus._texsym("gamma_HARPS") == "\$\\gamma_{\\mathrm{HARPS}}\$"

    # The tell-tale of a missing symbol: the planet suffix leaking into roman
    # text as \mathrm{k1} instead of becoming the subscript _{1}.
    for nm in ("a_k1", "M_sec_k1", "inc_k1", "Omega_k1", "sesinw_k1", "Mo_k1")
        @test !occursin("\\mathrm{k", Nereus._texsym(nm))
    end
    # And no row may carry a raw underscore, which breaks math mode.
    for nm in ("a_k1", "M_sec_k1", "inc_k1", "Omega_k1", "plx", "rho_s",
               "sigma_HIRES", "gamma_Lick", "sesinw_k1")
        body = replace(Nereus._texsym(nm), "\\_" => "")   # escaped ones are fine
        @test !occursin(r"_[A-Za-z0-9]{2,}", body)
    end
end

@testset "science table rounding follows the smaller error" begin
    # Rounding to the larger error destroys the precise side of an asymmetric
    # interval: 0.484 +0.082 -1.027 became 0.5 +0.1 -1.0.
    v, lo, hi = Nereus._fmt3(0.4836, -1.027, 0.082)
    @test v == "0.484" && lo == "1.027" && hi == "0.082"
end
