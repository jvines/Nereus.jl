# Every Nereus sampler runs an obliquity target through run_job (briefly --
# this is about the plumbing, not convergence): group 3 of RJ_SAMPLER_GROUPS
# (fixtures/obliquity_runjob.jl).
using Test
@isdefined(rj_config) || include(joinpath(@__DIR__, "fixtures", "obliquity_runjob.jl"))

@testset "obliquity run_job samplers, group 3" begin
    rj_sampler_testsets(RJ_SAMPLER_GROUPS[3])
end
