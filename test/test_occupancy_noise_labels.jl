# The occupancy plot named the noise models "model 1", "model 2", ... in every
# run_job fit: the dispatcher passed neither the trans-dim config nor labels.
# `_occupancy_noise_labels` resolves them from the job config, in the
# `noise_active_<i>` order the sampler uses.
using Test, Nereus, Random

@testset "occupancy plot: noise-model labels from the job config" begin
    nm(kind, label) = Dict{String,Any}("kind" => kind, "channel" => "rv", "label" => label)
    nms = Any[nm("ErrorScale", "ErrScale"), nm("CeleriteRotation", "GP-Rot"),
              Dict{String,Any}("kind" => "ActivityDecorrelation", "channel" => "rv",
                               "kwargs" => Dict{String,Any}("indicators" => ["bis"]))]
    cfg(tog) = Dict{String,Any}("noise_models" => nms,
        "transdim" => Dict{String,Any}("max_kplanet" => 1, "toggleable" => tog))

    # labels given in the config, by index and by label, in toggleable order
    @test Nereus._occupancy_noise_labels(cfg(Any[1, 2]), nothing, nothing) == ["ErrScale", "GP-Rot"]
    @test Nereus._occupancy_noise_labels(cfg(Any["GP-Rot", 1]), nothing, nothing) == ["GP-Rot", "ErrScale"]
    # an entry with no label falls back to its kind
    @test Nereus._occupancy_noise_labels(cfg(Any[3]), nothing, nothing) == ["ActivityDecorrelation"]
    # no trans-dim block, or none toggleable: the plot numbers them
    @test Nereus._occupancy_noise_labels(Dict{String,Any}("noise_models" => nms), nothing, nothing) === nothing
    @test Nereus._occupancy_noise_labels(cfg(Any[]), nothing, nothing) === nothing

    # a noise_menu: the menu's toggleable members, in the menu's order
    rng = MersenneTwister(5); N = 30
    t = sort(rand(rng, N)) .* 60.0
    data = Data(t_rv = t, rv = randn(rng, N), rv_err = ones(N), rv_inst = ones(Int, N),
                indicators = Dict("bis" => randn(rng, N), "fwhm" => randn(rng, N)),
                indicator_errs = Dict("bis" => ones(N), "fwhm" => ones(N)))
    mc = Dict{String,Any}("noise_menu" => Dict{String,Any}("indicators" => ["bis", "fwhm"]),
                          "transdim" => Dict{String,Any}("max_kplanet" => 1))
    want = [Nereus._noise_model_label_scoped(m) for m in default_noise_menu(data; indicators = ["bis", "fwhm"]).toggleable]
    got = Nereus._occupancy_noise_labels(mc, nothing, data)
    @test got == want
    @test !isempty(got) && !any(startswith("model "), got)
end
