using Test, Logging
using Nereus

# Makie logs "Could not find font ..." for a missing family. `_serif_family` probes
# families one by one, so every Linux machine without Times New Roman used to get
# that warning, although the theme then picked a serif fallback correctly. The probe
# is quiet now; the theme's own warning (no serif family at all) is still allowed.
@testset "font probe is quiet" begin
    logs, fam = Test.collect_test_logs(min_level = Logging.Warn) do
        Nereus._serif_family()
    end
    @test !any(l -> occursin("Could not find font", string(l.message)), logs)
    @test fam.regular isa AbstractString
    @test !Nereus._font_available("No Such Font Family Nereus")
end

# The theme is built for every figure; the probe (and its warning, when there is
# no serif family) runs once per session.
@testset "font probe runs once" begin
    Nereus._SERIF_FAMILY[] = nothing
    first_logs, fam = Test.collect_test_logs(min_level = Logging.Warn) do
        Nereus._serif_family()
    end
    logs, again = Test.collect_test_logs(min_level = Logging.Warn) do
        Nereus.nereus_theme()
        Nereus._serif_family()
    end
    @test isempty(logs)
    @test again === fam
end
