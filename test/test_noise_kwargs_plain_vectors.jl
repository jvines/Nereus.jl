# Noise-model kwargs that reach `_build_noise_models` as plain `Vector{Any}` — the
# shape `_replay_config` produces when it re-reads a finished job's
# job_config.json through `_plain` (replot, detection limits) — must narrow to
# the constructor's field type. `ActivityDecorrelation(indicators::Vector{String})`
# refused `Vector{Any}` with a TypeError, so no run with a decorrelation member
# could be re-plotted.
using Test, Nereus

@testset "_build_noise_models: Vector{Any} kwargs narrow to the field type" begin
    nm = [Dict{String,Any}("kind" => "ActivityDecorrelation", "channel" => "rv",
                           "kwargs" => Dict{String,Any}("indicators" => Any["bis", "fwhm"],
                                                        "derivative" => false, "label" => "lin"))]
    models = Nereus._build_noise_models(nm, nothing)
    m = only(models)
    @test m isa Nereus.ActivityDecorrelation
    @test m.indicators == ["bis", "fwhm"]
    @test m.indicators isa Vector{String}
end
