abstract type ImportanceSampler <: AbstractSamplingAlgorithm end

"""
    SobolSampler

Constructors:

    SobolSampler()

Sample from Sobol sequence. Also see [Sobol.jl](https://github.com/stevengj/Sobol.jl).
"""
struct SobolSampler <: ImportanceSampler end
export SobolSampler


function bat_sample_impl(
    rng::AbstractRNG,
    density::AnyDensityLike,
    n::Integer,
    algorithm::SobolSampler
)
    shape = varshape(density)

    bounds = var_bounds(density)
    truncated_density = if isinf(bounds)
        TruncatedDensity(density, estimate_finite_bounds(density), 0)
    else
        TruncatedDensity(density, bounds, 0)
    end

    samples = _gen_samples(algorithm, n, truncated_density)
    logvals = logvalof.(Ref(truncated_density), samples)
    weights = exp.(logvals)

    bat_samples = shape.(DensitySampleVector(samples, logvals, weight = weights))
    return (result = bat_samples,)
end


function _gen_samples(algorithm::SobolSampler, n::Integer, density::TruncatedDensity)
    bounds = var_bounds(density)
    sobol = Sobol.SobolSeq(bounds.vol.lo, bounds.vol.hi)
    p = vcat([[Sobol.next!(sobol)] for i in 1:n]...)
    return p
end


"""
    GridSampler

Constructors:

    GridSampler()

Sample from equidistantly distributed points in each dimension.
"""
@with_kw struct GridSampler <: ImportanceSampler
    bins::Union{Integer, Union{NamedTuple, Tuple{Vararg{Integer}}}, Nothing} = nothing
end
export GridSampler
export modify_binning


function bat_sample_impl(
    rng::AbstractRNG,
    density::AnyDensityLike,
    n::AnyNSamples,
    algorithm::GridSampler
)
    shape = varshape(density)

    bounds = var_bounds(density)
    truncated_density = if isinf(bounds)
        TruncatedDensity(density, estimate_finite_bounds(density), 0)
    else
        TruncatedDensity(density, bounds, 0)
    end

    n = if typeof(algorithm.bins) == Int && length(bounds.bt) > 1
        algorithm.bins^(length(bounds.bt))
    else
        algorithm.bins == nothing ? n = n : n = values(algorithm.bins)
    end

    samples = _gen_samples(algorithm, n, truncated_density).sampl

    # keys = all_active_names(shape)
    # println(Symbol.(keys))
    # bin_edges = NamedTuple{all_active_names(shape)}(_gen_samples(algorithm, n, truncated_density).rngs)

    logvals = logvalof.(Ref(truncated_density), samples)
    weights = exp.(logvals)

    bat_samples = shape.(DensitySampleVector(samples, logvals, weight = weights))

    return (result = bat_samples, bins = nothing)
end


function _gen_samples(algorithm::GridSampler, n::Integer, density::TruncatedDensity)
    bounds = var_bounds(density)
    dim = length(bounds.bt)
    ppa = BAT._points_per_dimension(n, dim)
    ranges = [range(bounds.vol.lo[i], bounds.vol.hi[i], length = trunc(Int, ppa)) for i in 1:dim]
    p = vec(collect(Iterators.product(ranges...)))
    return (sampl = [collect(p[i]) for i in 1:length(p)], rngs = ranges)
end


function _points_per_dimension(n::Integer, dim::Integer)
    return n^(1/dim)
end


function _gen_samples(algorithm::GridSampler, n::AnyNSamples, density::TruncatedDensity)
    bounds = var_bounds(density)
    ppa = BAT._points_per_dimension(n)
    ranges = [range(bounds.vol.lo[i], bounds.vol.hi[i], length = ppa[i]) for i in 1:length(ppa)]
    p = vec(collect(Iterators.product(ranges...)))
    return (sampl = [collect(p[i]) for i in 1:length(p)], rngs = ranges)
end


function _points_per_dimension(n::AnyNSamples)
    return n
end


function modify_binning(original::NamedTuple, n_new::Union{NamedTuple, Union{Integer, Tuple{Vararg{Integer}}}})
    dim = length(original)
    bins = [length(collect(values(original[i]))) for i in 1:dim]
    divisors = [BAT._find_divisors(bins[i]) for i in 1:dim]
    binning, step = if isa(n_new, Integer) && mod(bins[1], n_new) == 0
        bins, 1
    elseif isa(n_new, Integer) && mod(bins[1], n_new) != 0
        min_divisors_index = [argmin(abs.(BAT._find_divisors(bins[i]) .- n_new)) for i in 1:dim]
        b = [divisors[i][j] for (i,j) = zip(1:length(divisors), min_divisors_index)]
        b, bins ./ b
    elseif [mod(bins[1], n_new[i]) for i in 1:dim] == zeros(dim)
        bins, ones(dim)
    elseif [mod(bins[1], n_new[i]) for i in 1:dim] != zeros(dim)
        min_divisors_index = [argmin([abs.(BAT._find_divisors(bins[i]) .- n_new[i]) for i in 1:dim][j]) for j in 1:dim]
        b = [divisors[i][j] for (i,j) = zip(1:length(divisors), min_divisors_index)]
        b, bins ./ b
    end
    Δ = [maximum(original[i]) - minimum(original[i]) for i in 1:dim]
    return NamedTuple{keys(original)}([range(minimum(original[i]), maximum(original[i]), step = Δ[i] / binning[i]) for i in 1:dim])
end


function _find_divisors(n::Integer)
    divisors::Array{Float64, 1} = []
    i::Int = 1
    while i < n
        if mod(n, i) == 0
            append!(divisors, i)
        end
        i += 1
    end
    return divisors
end



"""
    PriorImportanceSampler

Constructors:

    PriorImportanceSampler()

Sample randomly from prior distribution.
"""
struct PriorImportanceSampler <: AbstractSamplingAlgorithm end
export PriorImportanceSampler

function bat_sample_impl(
    rng::AbstractRNG,
    posterior::AbstractPosteriorDensity,
    n::AnyNSamples,
    algorithm::PriorImportanceSampler
)
    shape = varshape(posterior)

    prior = getprior(posterior)
    priorsmpl = bat_sample(prior, n)
    unshaped_prior_samples = unshaped.(priorsmpl.result)

    v = unshaped_prior_samples.v
    prior_weight = unshaped_prior_samples.weight
    posterior_logd = logvalof.(Ref(posterior), v)
    weight = exp.(posterior_logd - unshaped_prior_samples.logd) .* prior_weight

    posterior_samples = shape.(DensitySampleVector(v, posterior_logd, weight = weight))
    priorsmpl = bat_sample(prior, n)

    return (result = posterior_samples, priorsmpl = priorsmpl)
end
