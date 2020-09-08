using BAT
using Test
using Plots
using Random, Distributions, StatsBase, IntervalSets

μ_1, σ_1, μ_2, σ_2 = 0.0, 1.0, 2.0, 2.0
likelihood = let μ_1 = μ_1, σ_1 = σ_1, μ_2 = μ_2, σ_2 = σ_2
    params -> begin
        l = logpdf.(Normal(μ_1, σ_1), params.a)
        k = logpdf.(Normal(μ_2, σ_2), params.b)
    return LogDVal(l + k)
    end
end

μ_prior, σ_prior = 1.5, 1.0
prior = BAT.NamedTupleDist(
    a = -4.0..4.0,
    b = Normal(μ_prior, σ_prior)
)

posterior = PosteriorDensity(likelihood, prior)
truncated_posterior = BAT.TruncatedDensity(posterior, BAT.estimate_finite_bounds(posterior), 0)

n_samples = 10^5

@testset "grid_binning" begin
    @testset "mod_binning" begin

        samples = bat_sample(posterior, n_samples, GridSampler())
        original_binning_a, original_binning_b = trunc(Int, sqrt(length(samples.result.v.a))), trunc(Int, sqrt(length(samples.result.v.b)))

        # test original binning with ppa = n^(1/dim)
        @test original_binning_a == length(BAT._gen_samples(GridSampler(), n_samples, truncated_posterior).rngs[1])
        @test original_binning_b == length(BAT._gen_samples(GridSampler(), n_samples, truncated_posterior).rngs[2])

        # test mod function to reduce number of bins so it matches multiplicity of ppa
        global n_new = 100
        while mod(sqrt(length(samples.result.v)), n_new) != 0
            n_new -= 1
        end
        step = original_binning_a / n_new
        @test mod(original_binning_a, n_new) == 0
        # @test (length(samples.bins[1]) - 1) * (length(samples.bins[2]) - 1) * step^2 == original_binning_a * original_binning_b
    end

    @testset "mod_binning_n_tuple" begin

        samples = bat_sample(posterior, (50, 100), GridSampler())
        chosen_bins = (50, 100)
        Δ_1, Δ_2 = 8.0, BAT.var_bounds(truncated_posterior).vol.hi[2] - BAT.var_bounds(truncated_posterior).vol.lo[2]
        println(length(samples.bins[2]))
        @test length(samples.bins[1]) == chosen_bins[1]
        @test length(samples.bins[2]) == chosen_bins[2]
        @test isapprox(sqrt(length(samples.result.v.a))^2, 50*100; rtol = 0.05)
        @test isapprox(Δ_1 / chosen_bins[1], samples.bins[1][2] - samples.bins[1][1]; rtol = 0.05)
        @test isapprox(Δ_2 / chosen_bins[2], samples.bins[2][2] - samples.bins[2][1]; rtol = 0.05)
    end
end
