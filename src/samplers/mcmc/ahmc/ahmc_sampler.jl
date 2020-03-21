export AHMC

struct AHMC <: AbstractSamplingAlgorithm end


"""
    function bat_sample(
        posterior::AnyPosterior,
        n::AnyNSamples,
        algorithm::AHMC;
        n_adapts::Int = min(div(n[1], 10), 1_000),
        metric = DiagEuclideanMetric(),
        gradient = ForwardDiff,
        integrator = Leapfrog(),
        proposal = NUTS(),
        adaptor =  StanHMCAdaptor(),
        verbose::Bool = true,
        drop_warmup::Bool = true
    )
Sample `posterior` via Hamiltonian Monte Carlo using AdvancedHMC.jl.
# AHMC related keyword arguments
Also see https://github.com/TuringLang/AdvancedHMC.jl for more information.
## Metric
default: `metric = DiagEuclideanMetric()`
options:
- `UnitEuclideanMetric()`
- `DenseEuclideanMetric()`
## Integrator
default: `integrator = Leapfrog(ϵ::Real = 0)`, with stepsize `ϵ. `
When ϵ = 0, the initial stepsize is determined using `AdvancedHMC.find_good_eps()`
options:
- `JitteredLeapfrog(ϵ::Real = 0, n::Real = 1.0)` with the jitter rate `n`,
- `TemperedLEapfrog(ϵ::Real = 0, a::Real = 1.05)` with tempering rate `a`
## Proposal
default: `proposal = NUTS(sampling::Symbol = :MultinomialTS, nuts::Symbol = :ClassicNoUTurn)`
options:
- `StaticTrajectory(n::Real = 10)`
- `HMCDA(len_traj::Real = 2)`
- `NUTS(sampling::Symbol, nuts::Symbol)`
with
    - `sampling =` `:SliceTS` or `:MultinomialTS`
    - `nuts = ` `:ClassicNoUTurn` or  `:GeneralisedNoUTUrn`

## Adaptor
default: `adaptor =  StanHMCAdaptor(δ::Real = 0.8)`
options:
- `Preconditioner()`
- `NesterovDualAveraging(δ::Real = 0.8)`
- `NaiveHMCAdaptor(δ::Real = 0.8)`
"""
function bat_sample(
    posterior::AnyPosterior,
    n::AnyNSamples,
    algorithm::AHMC;
    initial_v::Array{Array{Float64,1},1} = [rand(getprior(posterior)) for i in 1:n[2]],
    n_adapts::Int = min(div(n[1], 10), 1_000),
    metric = DiagEuclideanMetric(),
    gradient = ForwardDiff,
    integrator = Leapfrog(),
    proposal = NUTS(),
    adaptor =  StanHMCAdaptor(),
    verbose::Bool = false,
    drop_warmup::Bool = true
)

    dim = length(initial_v[1])
    metric = get_AHMCmetric(metric, dim)
    n_samples = n[1]; n_chains = n[2]

    sample_arr = Vector{Array{Array{Float64, 1},1}}(undef, n_chains)
    stats_arr =  Vector{Array{NamedTuple, 1}}(undef, n_chains)

    logval_posterior(v) = density_logval(posterior, v)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, logval_posterior, gradient)


    Threads.@threads for i in 1:n_chains
        integrator.ϵ == 0 ? integrator.ϵ = AdvancedHMC.find_good_eps(hamiltonian, initial_v[i]) : nothing
        bat_integrator = get_AHMCintegrator(integrator)

        bat_proposal = get_AHMCproposal(proposal, bat_integrator)
        bat_adaptor = get_AHMCAdaptor(adaptor, metric, bat_integrator)

        # sample using AdvancedHMC
        samples, stats = AdvancedHMC.sample(
            hamiltonian,
            bat_proposal,
            initial_v[i],
            n_samples,
            bat_adaptor,
            n_adapts;
            progress=false,
            verbose=verbose,
            drop_warmup = drop_warmup
        )

        sample_arr[i] = samples
        stats_arr[i] = stats
    end

    samples = vcat(sample_arr...)
    stats = vcat(stats_arr...)

    bat_samples = convert_to_bat_samples(samples, posterior)

    return (result = bat_samples, chains = stats)
end
