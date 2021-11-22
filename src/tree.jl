using DynamicalSystemsBase
using DelayEmbeddings
using Revise
import Base.show

export mcdts_embedding

"""
    The MCDTS algorithm is implemented as a tree with different kind types encoding
    the leafs and the root of the tree. AbstractTreeElement is the abstract type of
    these types.
"""
abstract type AbstractTreeElement end

"""
    mutable struct Root <: AbstractTreeElement

    The 'start'/root of Tree. Each node contains its children. The root contains the starting branches/nodes.
    For initialization type `r = Root()`.

    ## Fieldnames:
    * `children::Union{Array{Node,1},Nothing}`: The first nodes of the tree.
    * `Lmin`; Is the global minimum of the cumulative ΔL statistic found so far.
"""
mutable struct Root <: AbstractTreeElement
    children
    Lmin
end
Root()=Root(nothing,0)
get_τs(n::Root) = Int[]
get_ts(n::Root) = Int[]

function Base.show(io::IO,n::Root)

    if n.children == nothing
        return print(io,string("Embedding tree, no tree search yet performed"))
    else
        best_node = best_embedding(n)

        return print(io,string("Embedding tree with current best embedding: L=",best_node.L," - full embd. τ=",best_node.τs," ,i_ts=",best_node.ts))
    end
end

"""
    mutable struct Node{T}

    A node of the tree. Each node contains its children and information about the current embedding.

    ## Fieldnames:
    * `τ::Int`: The delay value of this node
    * `L::T`: The value of the cumulative ΔL statistic at this node
    * `τs::Array{Int,1}`: The complete vector with all τs chosen along this path up until this node
    * `ts::Array{Int,1}`: The complex vector which of the possibly multivariate time series is used at each embedding step i
    * `children::Union{Array{Node,1},Nothing}`: The children of this node
    * `temp::S`: additional "work"/"temporary" array/field that can be manipulated and saved by the optimization e.g. to reuse prior computations
"""
mutable struct Node{T,S} <: AbstractTreeElement
    τ::Int
    t::Int
    L::T
    τs::Array{Int,1}
    ts::Array{Int,1}
    children::Union{Array{Node,1},Nothing}
    temp::S
end

N_children(n::AbstractTreeElement) = n.children == nothing ? 0 : length(n.children)
get_τs(n::Node) = n.τs
get_ts(n::Node) = n.ts
get_children_Ls(n::AbstractTreeElement) = [n.children[i].L for i in 1:N_children(n)]
get_children_τs(n::AbstractTreeElement) = [n.children[i].τ for i in 1:N_children(n)]
get_children_ts(n::AbstractTreeElement) = [n.children[i].t for i in 1:N_children(n)]
Base.show(io::IO,n::Node) = print(io,string("Node with τ=",n.τ,", i_t=",n.t," ,L=",n.L," - full embd. τ=",n.τs," ,i_ts=",n.ts))

"""
    choose_children(n::AbstractTreeElement, τ::Int, t:Int)

Pick one of the children of the tree node `n` with values `τ` and `t`. If there
is none, return `nothing`.
"""
function choose_children(n::AbstractTreeElement, τ::Int, t::Int)
    τs=get_children_τs(n)
    ts=get_children_ts(n)

    res = intersect(findall(τs .== τ), findall(ts .== t))
    if length(res)==0
        return nothing
    elseif length(res)==1
        return n.children[res[1]]
    else
        error("There's something wrong, there shouldn't be multiple children with the same values.")
    end
end

"""
    next_embedding(n::Node, optimalg::AbstractMCDTSOptimGoal, Ys::Dataset{D, T},
                            w::Int, τs; kwargs...) → τ_pot, ts_pot, L_pot, flag

    Performs the next embedding step. For the actual embedding contained in tree
    leaf `n` compute as many statistics determined by `optimalg.Λ`
    (see [`MCDTSOptimGoal`](@ref)) as there are time series in the Dataset
    `Ys` for a range of possible delays `τs`. Return the values for the best delay
    `τ_pot`, its corresponding time series index `ts_pot` the according Loss-value
    `L_pot` and `flag`, following the minimization of the Loss determined by
    `optimalg.Γ`.

    ## Keyword arguments
    * See [`mcdts_embedding`](@ref) for a list of all keywords.

    ## Returns
    * `τ_pot`: Next delay
    * `ts_pot`: Index of the time series used (in case of multivariate time series)
    * `L_pot`: L statistic of next embedding step with delay `τ_pot` from `ts_pot`.
    * `flag`: Did the embedding converge? i.e. L can not be further minimized anymore

"""
function next_embedding(n::Node, optimalg::AbstractMCDTSOptimGoal, Ys::Dataset{D, T},
                                        w::Int, τs; kwargs...) where {D, T<:Real}

    τs_old = get_τs(n)
    ts_old = get_ts(n)
    L_old = n.L
    # do the next embedding step
    τ_pot, ts_pot, L_pot, flag, temp = get_potential_delays(optimalg::AbstractMCDTSOptimGoal,
                            Ys, τs, w, Tuple(τs_old), Tuple(ts_old), L_old; temp=n.temp, kwargs...)

    return τ_pot, ts_pot, L_pot, flag, temp
end

function next_embedding(n::Root, optimalg::AbstractMCDTSOptimGoal, Ys::Dataset{D, T},
                                        w::Int, τs; kwargs...) where {D, T<:Real}
    # initialize first embedding step
    τ_pot, ts_pot, L_pot, temp = init_embedding_params(optimalg.Γ, size(Ys,2))

    return τ_pot, ts_pot, L_pot, false, temp
end


"""
    choose_next_node(n::Union{Node,Root}, func, Lmin_global, i_trial::Int=1)

    Returns one of the children of based on the function `func(Ls)->i_node`,
    Lmin_global is the best L value so far in the optimization process, if any of
    the input Ls to choose from is smaller than it, it is always chosen.
    `choose_mode` is only relevant for the first embedding step right now: it
    determines if the first step is chosen uniform (`choose_mode==0`) or with the
    `func` (`choose_mode==1`).
"""
function choose_next_node(n::Node, func, Lmin_global=-Inf, choose_mode=1)
    N = N_children(n)
    if N == 0
        return nothing
    elseif N == 1
        return n.children[1]
    else
        Ls = get_children_Ls(n)
        # check if any is smaller than the global min
        Lmins = findall(Ls .< Lmin_global)

        if isempty(Lmins)
            return n.children[func(Ls)]
        else
            return n.children[argmin(Ls)]
        end
    end
end

function choose_next_node(n::Root, func, Lmin_global=-Inf, i_trial::Int=1, choose_mode=0)
    N = N_children(n)
    if N == 0
        return nothing
    elseif N == 1
        return n.children[1]
    else
        if choose_mode==0
            return n.children[rand(1:N)]
        elseif choose_mode==1
            Ls = get_children_Ls(n)
            return n.children[func(Ls)]
        end
    end
end

minL(Ls) = argmin(Ls)

"""
    softmaxL(Ls; β=1.5)

    Returns an index with prob computed by a softmax of all Ls.
"""
function softmaxL(Ls; β=1.5)
    softmaxnorm = sum(exp.(-β*Ls))

    p_L = exp.(-β.*Ls) ./ softmaxnorm
    p_L = cumsum(p_L)
    rand_number = rand()

    for i=1:length(p_L)
        if p_L[i] > rand_number
            return i
        end
    end
    return length(p_L)
end

"""
    expand!(n::Union{Node,Root}, optimalg::AbstractMCDTSOptimGoal, data::Dataset,
                                            w::Int, delays, choose_mode; kwargs...)

    This is one single rollout and backprop of the tree. For details please see
    the accompanying paper [^Kraemer2021b].

    * `n`: Starting node
    * `optimalg::AbstractMCDTSOptimGoal`: Determines the delay preselection and
      cost function (see [`MCDTSOptimGoal`](@ref)).
    * `data`: data
    * `w`: Theiler Window
    * `delays = 0:100`: The possible time lags
    * `choose_mode::Int=0`: Possibility for different modes of choosing the next
      node based on which trial this is.

    ## Keyword arguments
    * See [`mcdts_embedding`](@ref) for a list of all keywords.
"""
function expand!(n::Root, optimalg::AbstractMCDTSOptimGoal, data::Dataset{D, T}, w::Int,
            delays::AbstractRange{DT} = 0:100, choose_mode::Int=0; max_depth::Int=20,
            verbose=false, choose_func = (L)->(MCDTS.softmaxL(L,β=2.)), kwargs...) where {D, DT, T<:Real}

    @assert w > 0 && w < length(data) "Provide a valid Theiler window, i.e. a positive integer (meaning sampling units)."
    current_node = n

    for i=1:max_depth # loops until converged or max_depth is reached
        # next embedding step
        # only if it was not already computed
        if current_node.children == nothing
            τs, ts, Ls, converged, temps = next_embedding(current_node, optimalg, data, w, delays; kwargs...)

            if converged
                break
            else
                # spawn children
                children = Node[]
                for j = 1:length(τs)
                    push!(children, (τs[j],ts[j],Ls[j], temps[j]), optimalg.Γ, current_node)
                end
                current_node.children = children
            end
        end
        # choose next node
        current_node = choose_next_node(current_node, choose_func, n.Lmin, choose_mode)
        if verbose
            println(current_node)
        end
    end
    # now backprop the values (actually we go to top to bottom, but we know were to end because we got the correct τs and ts)
    backprop!(n, current_node.τs, current_node.ts, current_node.L)
end

"""
    backprop!(n::Root,τs,ts,L_min)

    Backpropagation of the tree spanned by all children in `n` (for this run).
    All children-nodes L-values get set to the final value achieved in this run.
    This function is ususally called be [`expand!`](@ref).
"""
function backprop!(n::Root,τs,ts,L_min)

    if n.Lmin > L_min
        n.Lmin = L_min
    end

    current_node = n
    for i=1:length(τs)
        # the initial embedding step is left out of the backprop
        current_node = choose_children(current_node,τs[i],ts[i])
        if current_node.L > L_min
            current_node.L = L_min
        end
    end
end


"""
    Legacy name, please use [`mcdts_embedding`](@ref)
"""
mc_delay(varargs...; kwargs...) = mcdts_embedding(varargs...; kwargs...)


"""
    best_embedding(r::Root)

    Given the root `r` of a tree, return the best embedding in the form of the
    final node at the end of the best embedding.
"""
function best_embedding(r::Root)

    not_finished = true
    current_node = r
    while not_finished

        if current_node.children == nothing
            not_finished = false
            return current_node
        else
            Ls = get_children_Ls(current_node)

            # traverse down the tree always with minimal L
            current_node = current_node.children[argmin(Ls)]
        end
    end
    return current_node
end

## Export function

"""
    mcdts_embedding

    ## Convenience / default option call

    mcdts_embedding(data::Dataset, N::Int=100; kwargs...)

    Do the MCDTS embedding of the `data` with `N` trials according to the
    PECUZAL algorithm [^Kraemer2021], returns the tree. Embedding parameters
    like the Theiler window, considered delays and the function that chooses the
    next embedding step are all estimated automatically or the default option is
    used. `data` is a `DynamicalSystems.Dataset`.

    ## All options

    mcdts_embedding(data::DynamicalSystems.Dataset, optimalg::AbstractMCDTSOptimGoal,
                            w::Int, delays::AbstractRange{D}, N::Int=40;  kwargs...)

    * `optimalg::AbstractMCDTSOptimGoal` determines how the embedding is performed in
      each cycle. Specifically it sets the delay pre-selection statistic Λ, which
      pre-selects the potential delays in each embedding cycle as well as the the
      Loss-Statistic Γ, which determines the Loss to be minimized by MCDTS
      [^Kraemer2021b] (see [`MCDTSOptimGoal`](@ref)).
    * `w` is the Theiler window (neighbors in time with index `w` close to the point,
      that are excluded from being true neighbors. `w=0` means to exclude only the
      point itself, and no temporal neighbors. In case of multivariate time series
      input choose `w` as the maximum of all `wᵢ's`. As a default in the convience
      call this is estimated with a mutual information minimum method of DelayEmbeddings.jl
    * `delays = 0:100`: The possible time lags
    * `N::Int`: The number of tree expansions

    ## Keyword Arguments
    * `choose_func`: Function to choose next node in the tree with, default
      choice: `(L)->(MCDTS.softmaxL(L,β=2.))`
    * `max_depth = 20`: Threshold, which determines the algorithm. It either breaks,
      when it converges, i.e. when there is no way to reduce the cost-function any
      further, or when this threshold is reached.
    * `verbose`: Either `true` or `false` (default); prints status of embedding optimization.
    * `metric`: norm for distance computation (default is `Euclidean()`)






    * `KNN = 3`: The number of nearest neighbors considered in the computation of
      the L-statistic.
    * `FNN:Bool = false`: Determines whether the algorithm should minimize the
      L-statistic or the FNN-statistic.
    * `PRED::Bool = false`: Determines whether the algorithm should minimize the
      L-statistic or a cost function based on minimizing the `Tw`-step-prediction error
    * `Tw::Int = 1`: If `PRED = true`, this is the considered prediction horizon
    * `linear::Bool=false`: If `PRED = true`, this determines whether the prediction shall
      be made on the zeroth or a linear predictor.
    * `PRED_mean::Bool=false`: If `PRED = true`, this determines whether the prediction shall
      be optimized on the mean MSE of all components or only on the 1st-component (Default)
    * `PRED_L::Bool=false`: If `PRED = true`, this determines whether the prediction shall
      be optimized on possible delay values gained from the continuity statistic or on
      delays = 0:25 (Default)
    * `PRED_KL::Bool=false`: If `PRED = true`, this determines whether the prediction shall
      be optimized on the Kullback-Leibler-divergence of the in-sample prediction and
      the true in-sample values, or if the optimization shall be made on the MSE of them (Default)
    * `CCM:Bool=false`: Determines whether the algorithm should maximize the CCM-correlation
      coefficient of the embedded vector from `Ys` and the given time series `Y_CCM`.
    * `Y_CCM`: The time series CCM should cross map to.
    * `threshold::Real = 0`: The algorithm does not pick a peak from the continuity
      statistic, when its corresponding `ΔL`/FNN-value exceeds this threshold. Please
      provide a positive number for both, `L` and `FNN`-statistic option (since the
      `ΔL`-values are negative numbers for meaningful embedding cycles, this threshold
      gets internally sign-switched).
    * `tws::Range = 2:delays[end]`: Customization of the sampling of the different T's,
      when computing Uzal's L-statistics. Here any kind of integer ranges (starting
      at 2) are allowed, up to `delays[end]`.

      [^Kraemer2021]: Kraemer, K.H., Datseris, G., Kurths, J., Kiss, I.Z., Ocampo-Espindola, Marwan, N. (2021). [A unified and automated approach to attractor reconstruction. New Journal of Physics 23(3), 033017](https://iopscience.iop.org/article/10.1088/1367-2630/abe336).
      [^Kraemer2021b]: Kraemer, K.H., Gelbrecht, M., Pavithran, I., Sujith, R. I. and Marwan, N. (2021). [Optimal state space reconstruction via Monte Carlo Decision Tree Search. Submitted to Nonlinear Dynamics](https://doi.org/10.21203/rs.3.rs-899760/v1)
"""
function mcdts_embedding(data::Dataset, optimalg::AbstractMCDTSOptimGoal, w::Int,
                            delays::AbstractRange{D}, N::Int=40; verbose::Bool = false, kwargs...) where {D}
    @assert N > 0
    @assert w >= 0

    # initialize tree and expand N times
    tree = Root()
    for i=1:N
        choose_mode=i<(N/2) ? 0 : 1 # in the first half of the trials uniform branching in the first embedding cycle,
        # in the second half use softmax function for branching even in the first cycle.
        expand!(tree, optimalg, data, w, delays, choose_mode; kwargs...)
        if verbose
            if (i%1)==0
                println(i,"/",N)
                println(best_embedding(tree))
                println("---")
            end
        end
    end
    return tree
end

function mcdts_embedding(data::Dataset, N::Int=40; kwargs...)

    # estimate Theiler window
    w = []
    for i=1:size(data,2)
        push!(w,DelayEmbeddings.estimate_delay(data[:,i],"mi_min"))
    end
    w=maximum(w)

    # consider delays up to 100 (if the time series is that long)
    delays = (size(data,1) > 101) ? (0:100) : (0:(size(data,1)-1))
    pecuzal = PecuzalOptim()

    return mcdts_embedding(data, pecuzal, w, delays, N; kwargs...)
end
