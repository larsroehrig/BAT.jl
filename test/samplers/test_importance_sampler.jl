using BAT
using Test
using Plots
using Sobol
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

# Test will not pass with n_samples < 10^5 due to statistical uncertainties in
# the standard deviation calculation from the test samples
n_samples = 10^5

# Product of two Normal distributions results in combined calculations of mean μ
# and standard deviation σ
# Also see [https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html]
mean_truth = [μ_1, (μ_2 * σ_prior^2 + μ_prior * σ_2^2) / (σ_prior^2 + σ_2^2)]
std_truth = [σ_1, sqrt((σ_2^2 * σ_prior^2) / (σ_prior^2 + σ_2^2))]

@testset "importance_sampler" begin
    @testset "sobol_sampler" begin

        sobol = SobolSeq([-4.0, -2.76],[4.0, 5.76])
        p = vcat([[Sobol.next!(sobol)] for i in 1:n_samples]...)
        samples_Sobol = bat_sample(posterior, n_samples, SobolSampler()).result
        logvals = BAT.logvalof(posterior, [samples_Sobol.v.a[1], samples_Sobol.v.b[1]])
        @test length(samples_Sobol) == n_samples
        @test isapprox([samples_Sobol.v.a[1], samples_Sobol.v.b[1]], p[1]; rtol = 0.05)
        @test isapprox(mean(unshaped.(samples_Sobol.v), FrequencyWeights(samples_Sobol.weight)), mean_truth; rtol = 0.05)
        @test isapprox(std(unshaped.(samples_Sobol.v), FrequencyWeights(samples_Sobol.weight)), std_truth; rtol = 0.05)
        @test isapprox(BAT.estimate_finite_bounds(prior).vol.lo, [-4.0, -2.76]; rtol = 0.05)
        @test isapprox(BAT.estimate_finite_bounds(prior).vol.hi, [4.0, 5.76]; rtol = 0.05)
        @test exp(logvals) == samples_Sobol.weight[1]
    end
    @testset "grid_sampler" begin

        samples_Grid = bat_sample(posterior, n_samples, GridSampler()).result
        samples_Grid_Binning = bat_sample(posterior, n_samples, GridSampler((a = 100, b = 200)))
        logvals = BAT.logvalof(posterior, [samples_Grid.v.a[1], samples_Grid.v.b[1]])
        @test isapprox([samples_Grid.v.a[1], samples_Grid.v.b[1]], [-4.0, -2.76]; rtol = 0.05)
        @test isapprox(mean(unshaped.(samples_Grid.v), FrequencyWeights(samples_Grid.weight)), mean_truth; rtol = 0.05)
        @test isapprox(std(unshaped.(samples_Grid.v), FrequencyWeights(samples_Grid.weight)), std_truth; rtol = 0.05)
        @test exp(logvals) == samples_Grid.weight[1]
        @test (length(samples_Grid_Binning.bins[1]), length(samples_Grid_Binning.bins[2])) == (100, 200)
    end
    @testset "prior_importance_sampler" begin

        samples_PriorImportance = bat_sample(posterior, n_samples, PriorImportanceSampler()).result
        logvals = BAT.logvalof(posterior, [samples_PriorImportance.v.a[1], samples_PriorImportance.v.b[1]])
        logpriorvals = BAT.logvalof(BAT.getprior(posterior), [samples_PriorImportance.v.a[1], samples_PriorImportance.v.b[1]])
        @test length(samples_PriorImportance) == n_samples
        @test isapprox(mean(unshaped.(samples_PriorImportance.v), FrequencyWeights(samples_PriorImportance.weight)), mean_truth; rtol = 0.05)
        @test isapprox(std(unshaped.(samples_PriorImportance.v), FrequencyWeights(samples_PriorImportance.weight)), std_truth; rtol = 0.05)
        @test isapprox([minimum(samples_PriorImportance.v.a), maximum(samples_PriorImportance.v.a)], [-4.0, 4.0]; rtol = 0.05)
        @test exp(logvals - logpriorvals) == samples_PriorImportance.weight[1]
    end
end
