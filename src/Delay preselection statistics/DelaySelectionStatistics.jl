# All function related to the different Delay pre-selection statistics. Currently 
# there is only the continuity statistic [`pecora`](@ref) and the range-function
# implemented (range function means that there is actually no preselection performed)


# Constructors:

## Constructors for DelayPreSelection Functions
"""
    Continuity_function <: AbstractDelayPreselection

    Constructor for the continuity function `⟨ε★⟩` by Pecora et al.[^Pecora2007],
    see [`pecora`](@ref).

    ## Fieldnames
    * `K::Int`: the amount of nearest neighbors in the δ-ball. Must be at
      least 8 (in order to gurantee a valid statistic). `⟨ε★⟩` is computed taking
      the minimum result over all `k ∈ K` (read algorithm description in [`pecora`](@ref)).
    * `samplesize::Real`: determine the fraction of all phase space points
      to be considered (fiducial points v) to average ε★ to produce `⟨ε★⟩`.
    * `α::Real = 0.05`: The significance level for obtaining the continuity statistic
    * `p::Real = 0.5`: The p-parameter for the binomial distribution used for the
      computation of the continuity statistic ⟨ε★⟩.

    ## Defaults
    * When calling `Continuity_function()` a Continuity_function-object is constructed
      with `K=13`, `samplesize=1.`, `α=0.05` and `p=0.5`.

"""
struct Continuity_function <: AbstractDelayPreselection
    K::Int
    samplesize::Real
    α::Real
    p::Real
    # Constraints and Defaults
    Continuity_function(k=13,x=1.,y=0.05,z=0.5) = begin
        @assert k > 7 "At least 8 nearest neighbors must be in the δ-ball, in order to gurantee a valid statistic."
        @assert 0. < x ≤ 1. "Please select a valid `samplesize`, which denotes a fraction of considered fiducial points, i.e. `samplesize` ∈ (0 1]"
        @assert 1. > y > 0.
        @assert 1. > z > 0.
        new(k,x,y,z)
    end
end

"""
    Range_function <: AbstractDelayPreselection

    Constructor for a range of possible delay values. In this case there is
    actually no "pre-selection" of delay, but rather all possible delays, given
    in the input `τs` (see, [`mcdts_embedding`](@ref)) are considered. This can significantly affect the
    computation time. There are no fieldnames, simply construct by typing
    `RangeFunction()`.
"""
struct Range_function <: AbstractDelayPreselection end


# Functions:

"""
    get_max_idx(Λ::AbstractDelayPreselection, dps::Vector, τ_vals, ts_vals) → max_idx

    Compute the candidate delay values from the given delay pre-selection statistic
    `dps` with respect to `Λ`, which determined how `dps` was obtained and how
    to select the candidates (e.g. pick the maxima of `dps` in case of the
    `Λ` being the Continuity function). See [`Continuity_function`](@ref) and
    [`Range_function`](@ref).
"""
function get_max_idx(Λ::Range_function, dps::Vector{T}, τ_vals, ts_vals, ts) where {T}
    max_idx = Vector(dps[2:end].+1)
    ts_idx = findall(e->e==ts, ts_vals) # do not consider already taken delays
    filter!(e->e∉(τ_vals[ts_idx] .+ 2), max_idx) # do not consider already taken delays
    return max_idx
end
function get_max_idx(Λ::Continuity_function, dps::Vector{T}, τ_vals, ts_vals, ts) where {T}
    _, max_idx = get_maxima(dps) # determine local maxima in delay_pre_selection_statistic
    ts_idx = findall(e->e==ts, ts_vals) # do not consider already taken delays
    filter!(e->e∉(τ_vals[ts_idx] .+ 2), max_idx) # do not consider already taken delays
    return max_idx
end


## Methods for delay preselection stats

"""
    get_delay_statistic(optimalg.Λ<: AbstractDelayPreselection, Ys, τs, w, τ_vals, ts_vals; kwargs... )

    Compute the delay statistic according to the chosen method in `optimalg.Λ` (see [`MCDTSOptimGoal`](@ref))
"""
function get_delay_statistic(Λ::Continuity_function, Ys, τs, w, τ_vals, ts_vals; metric = Euclidean(), kwargs... )

    ε★, _ = DelayEmbeddings.pecora(Ys, Tuple(τ_vals), Tuple(ts_vals); delays = τs, w = w,
            samplesize = Λ.samplesize, K = Λ.K, metric = metric, α = Λ.α,
            p = Λ.p)
    return ε★
end
function get_delay_statistic(Λ::Range_function, Ys, τs, w, τ_vals, ts_vals; kwargs... )
    return repeat(Vector(1:length(τs)), outer = [1,size(Ys,2)])
end