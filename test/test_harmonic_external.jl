# HarmonicBlock driven by an EXTERNAL frequency comb.
#
# The use case: a transit host that is also a coherent pulsator. The mode
# frequencies are measured elsewhere, over a photometric baseline far longer
# than any single fit, so they are fixed here; the amplitudes and phases stay
# marginalised inside the transit fit. That is the difference that matters --
# pre-subtracting a fitted mode model treats it as exact and understates the
# depth uncertainty, whereas marginalising propagates it.
using Test
using Nereus
using Nereus: HarmonicBlock, noise_param_names, _harmonic_factor

@testset "HarmonicBlock with an external frequency comb" begin
    FREQ = [1.504116, 2.892259, 10.290688]      # /d

    ext = HarmonicBlock(channel = :phot, nharm = 3, freqs = FREQ)
    rot = HarmonicBlock(channel = :phot, nharm = 3)

    @test ext.freqs == FREQ
    @test ext.nharm == length(FREQ)          # nharm follows the list
    @test isempty(rot.freqs)                 # default is unchanged

    inst = Nereus.InstrumentConfig(String[], ["TESS"])

    # The rotation form fits a period; the external form has nothing to fit but
    # the per-instrument amplitude, because the phases are marginalised.
    nm_rot = noise_param_names(rot, inst)
    nm_ext = noise_param_names(ext, inst)
    @test "harm_period_phot" in nm_rot
    @test !("harm_period_phot" in nm_ext)
    @test "harm_amp_TESS_phot" in nm_ext
    @test length(nm_ext) == 1

    # A block with no free period must not acquire a period prior either,
    # or the layout allocates a parameter nothing can constrain.
    dic = Dict{String, Any}()
    dat = Nereus.Data(; t_phot = [0.0, 1.0], flux = [1.0, 1.0],
                       flux_err = [1e-3, 1e-3], phot_inst = [1, 1])
    Nereus._default_noise_priors!(dic, ext, inst, dat, 1.0, 0.01)
    @test !haskey(dic, "harm_period_phot")
    @test haskey(dic, "harm_amp_TESS_phot")
end
