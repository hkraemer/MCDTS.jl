abstract type AbstractMCDTSOptimGoal end

abstract type AbstractMCDTSpredictionType end

abstract type AbstractPredictionLoss end

abstract type AbstractPredictionMethod end

abstract type AbstractDelayPreselection end

abstract type AbstractLoss end

"""
    MCDTSOptimGoal <: AbstractMCDTSOptimGoal

    Constructor, which handles the loss-/objective function `Γ` and the delay
    pre-selection statistic `Λ` MCDTS uses.

    ## Fieldnames
    * `Γ::AbstractLoss`: Chosen loss-function, see the so far available
      [`L_statistic`](@ref), [`FNN_statistic`](@ref), [`CCM_ρ`](@ref) and
      [`Prediction_error`](@ref).
    * `Λ::AbstractDelayPreselection`: Chosen delay Pre-selection method, see the so
      far available [`Continuity_function`](@ref) and [`Range_function`](@ref).

    ## Defaults
    * TBD
"""
struct MCDTSOptimGoal <: AbstractMCDTSOptimGoal
    Γ::AbstractLoss
    Λ::AbstractDelayPreselection
    # Constraints and Defaults
    MCDTSOptimGoal() = new(L_statistic(), Continuity_function())
end


## Constructors for Loss Functions

"""
    L_statistic <: AbstractLoss

    Constructor for the L-statistic loss function based on Uzal et al.[^Uzal2011]. Here
    we consider the decrease of the L-statistic `ΔL` in between embedding cycles,
    according to Kraemer et al.[^Kraemer2021] ([`pecuzal_embedding`](@ref)).

    ## Fieldnames
    * `threshold::Float`: A threshold for the tolerable `ΔL` decrease for the current
      embedding. When `ΔL` exceeds this threshold in an embedding cycle the embedding
      stops. Note that `ΔL` is a negative value therefore `threshold` must be a small
      negative number.
    * `KNN::Int`: the amount of nearest neighbors considered, in order to compute the
     L-statistic, in particular `σ_k^2` (read algorithm description [`uzal_cost`]@ref).
    * `tws::AbstractRange{Int}`: Customization of the sampling of the different time horizons
      (T's), when computing Uzal's L-statistics. Here any kind of integer ranges (starting at 2)
      are allowed.

    ## Defaults
    * When calling `L_statistic()`, a L_statistic-object is created, which uses no
      threshold and consideres 3 nearest neighbors for time horizons `tws=2:100`.
    * When calling `L_statistic(threshold)`, a L_statistic-object is created, which uses
      the given `threshold` and consideres 3 nearest neighbors for time horizons `tws=2:100`.
    * When calling `L_statistic(threshold,KNN)`, a L_statistic-object is created, which uses
      the given `threshold`, consideres `KNN` nearest neighbors for time horizons `tws=2:100`.

    [^Kraemer2021]: Kraemer, K.H., Datseris, G., Kurths, J., Kiss, I.Z., Ocampo-Espindola, Marwan, N. (2021). [A unified and automated approach to attractor reconstruction. New Journal of Physics 23(3), 033017](https://iopscience.iop.org/article/10.1088/1367-2630/abe336).

    [^Uzal2011]: Uzal, L. C., Grinblat, G. L., Verdes, P. F. (2011). [Optimal reconstruction of dynamical systems: A noise amplification approach. Physical Review E 84, 016223](https://doi.org/10.1103/PhysRevE.84.016223).
"""
struct L_statistic <: AbstractLoss
    threshold::AbstractFloat
    KNN::Int
    tws::AbstractRange{Int}
    # Constraints and Defaults
    L_statistic(x,y,z) = begin
        @assert x <= 0 "Please provide a (small) negative number for the threshold of ΔL."
        @assert y > 0
        @assert z[1] == 2 "The considered range for the time horizon of the L-function must start at 2."
    end
    L_statistic(x,y,z) = typeof(x) <: Int ? new(convert(AbstractFloat, x),y,z) : error("Fieldname threshold must be an integer or float < 0.")
    L_statistic(x,y) = new(x,y,2:100)
    L_statistic(x) = new(x,3,2:100)
    L_statistic() = new(0,3,2:100)
end

"""
    FNN_statistic <: AbstractLoss

    Constructor for the FNN-statistic loss function (false nearest neighbor) based
    on Hegger & Kantz [^Hegger1999].

    ## Fieldnames
    * `threshold::Float`: A threshold for the tolerable cumulative FNN decrease
      for the current embedding. When the fraction of FNNs fall below this threshold
      in an embedding cycle the embedding stops.
    * `r::Float`: The FNN-distance-expansion threshold (typically set to 2).

    ## Defaults
    * When calling `FNN_statistic()`, a FNN_statistic-object is created, which uses no
      threshold and uses the FNN-inter threshold `r=2`.
    * When calling `FNN_statistic(threshold)`, a FNN_statistic-object is created, which uses
      the given `threshold` and uses the FNN-inter threshold `r=2`.

    [^Hegger1999]: Hegger & Kantz, [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
struct FNN_statistic <: AbstractLoss
    threshold::AbstractFloat
    r::AbstractFloat
    # Constraints and Defaults
    FNN_statistic(x,y) = begin
        @assert x >= 0
        @assert y > 0
    end
    FNN_statistic(x,y) = typeof(x) <: Int ? new(convert(AbstractFloat, x),y)
    FNN_statistic(x,y) = typeof(y) <: Int ? new(x,convert(AbstractFloat, y))
    FNN_statistic(x) = new(x,2.)
    FNN_statistic() = new(0.,2.)
end

"""
    CCM_ρ <: AbstractLoss

    Constructor for the CCM_ρ loss function (correlation coefficient of the
    convergent cross mapping) based on Sugihara et al.[^Sugihara2012], see also
    [`ccm`](@ref). In this case MCDTS tries to maximize the correlation coefficient
    of the convergent cross mapping from the input `data` and `Y_CCM`, the time
    series CCM should cross map to from `data` (see [`mcdts_embedding`](@ref)).

    ## Fieldnames
    * `threshold::Float`: A threshold for the sufficient correlation of the
      cross-mapped values and the true values from `Y_CMM` for the current embedding.
      When the correlation coefficient exeeds this threshold in an embedding cycle
      the embedding stops.

    ## Defaults
    * When calling `CCM_ρ()`, a CCM_ρ-object is created, which uses the threshold 1,
      i.e. no threshold, since the correlation coefficient can not exceed 1.

    [^Sugihara2012]: Sugihara et al., [Detecting Causality in Complex Ecosystems. Science 338, 6106, 496-500](https://doi.org/10.1126/science.1227079)
"""
struct CCM_ρ <: AbstractLoss
    threshold::AbstractFloat
    # Constraints and Defaults
    CCM_ρ(x) = begin
        @assert x > 0 && x <= 1
    end
    CCM_ρ(x) = typeof(x) <: Int ? new(convert(AbstractFloat, x))
    CCM_ρ() = new(1.)
end

"""
    Prediction_error <: AbstractLoss

    Constructor for the Prediction_error loss function.

    ## Fieldnames
    * `PredictionType::MCDTSpredictionType`: Determines the prediction type by
      setting a prediction-method and the way the prediction error is measured,
      see [`MCDTSpredictionType`](@ref).
    * `threshold::Float`: A threshold for the sufficient minimum prediction error
      for the current embedding. When the prediction error, specified in
      `PredictionType`, falls below this threshold in an embedding cycle the
      embedding stops.

    ## Defaults
    * When calling `Prediction_error()`, a Prediction_error-object is created,
      which uses the threshold 0, i.e. no threshold and a zeroth-order predictor
      (see [`MCDTSpredictionType`](@ref), [`PredictionLoss`](@ref) &
      [`local_model`](@ref))
"""
struct Prediction_error <: AbstractLoss
    PredictionType::AbstractMCDTSpredictionType
    threshold::AbstractFloat
    # Constraints and Defaults
    Prediction_error(x,y) = begin
        @assert y >= 0
    end
    Prediction_error(x) = new(x,0.)
    Prediction_error() = new(MCDTSpredictionType(),0.)
end


"""
    MCDTSpredictionType <: AbstractMCDTSpredictionType

    Constructor, which determines the way how predictions are made technically.

    ## Fieldnames
    * `loss::AbstractPredictionLoss`: Indicates the way of computing the prediction error.
       See [`PredictionLoss`](@ref) for information on how to construct this object.
    * `method::AbstractPredictionMethod`: The method based on the state space reconstruction,
       which makes the actual prediction. See [`local_model`](@ref)

    ## Default settings
    * When calling `MCDTSpredictionType()` a MCDTSpredictionType-object is constructed
      with a `local_zeroth`-predictor [`local_model`](@ref), using 2 nearest neighbors
      and a 1-step-ahead-prediction. The loss-function is the root mean squared prediction
      error over all components [`PredictionLoss`](@ref).
"""
struct MCDTSpredictionType <: AbstractMCDTSpredictionType
    loss::AbstractPredictionLoss
    method::AbstractPredictionMethod
    # Constraints and Defaults
    MCDTSpredictionType(x) = new(x,local_model("zeroth",2,1))
    MCDTSpredictionType() = new(1,local_model("zeroth",2,1))
end


"""
    PredictionLoss <: AbstractPredictionLoss

    Constructor, which indicates the way of computing the prediction error. This
    object is used for the constructor, which determines the way how predictions are
    made methodologically [`MCDTSpredictionType`](@ref).

    ## Fieldnames
    * `type::Int` is an integer, which encodes the type of prediction error:
    * For `type = 1` the root mean squared prediction error over all components
      (dimensionality of the state space) is used.
    * For `type = 2` the mean Kullback-Leibler Distance of the predicted and the true
      values over all components (dimensionality of the state space) is used.

    ## Default settings
    * When calling `PredictionLoss()` a PredictionLoss-object is constructed with
      fieldname `type = 1` (≡root mean squared prediction error over all components)
"""
struct PredictionLoss <: AbstractPredictionLoss
    type::Int
    # Constraints and Defaults
    PredictionLoss(x) = begin
        @assert x == 1 || x == 2
    end
    PredictionLoss() = PredictionLoss(1)
end



"""
    local_model <: AbstractPredictionMethod

    Constructor, which indicates the local state space prediction model.

    ## Fieldnames
    * `method::String`: Could be `"zeroth"` (averaged `Tw`-step-ahead image of the
     `KNN`-nearest neighbors) or `"linear"` (local linear regression on the
     `KNN`-nearest neighbors).
    * `KNN::Int`: The number of considered nearest neighbors.
    * `Tw::Int` : The prediction horizon in sampling units.

    ## Default settings
    * When calling `local_model()` a local_model-object is constructed with a zeroth
      order prediction scheme, 2 nearest neighbors and a 1-step-ahead prediction.
    * When calling `local_model(method)` a local_model-object is constructed with a
      `method`-prediction scheme, 2 nearest neighbors and a 1-step-ahead prediction.
    * When calling `local_model(method,KNN)` a local_model-object is constructed with a
     `method`-prediction scheme, `KNN` nearest neighbors and a 1-step-ahead prediction.
"""
struct local_model <: AbstractPredictionMethod
    method::String
    KNN::Int
    Tw::Int
    # Constraints and Defaults
    local_model(x,y,z) = begin
        @assert x == "zeroth" || "linear"
        @assert KNN > 0
        @assert Tw > 1
    end
    local_model(x,y) = new(x,y,1)
    local_model(x) = new(x,2)
    local_model() = local_model("zeroth")
end



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
struct Continuity_function{P::Int,T<:Real} <: AbstractDelayPreselection
    K::P
    samplesize::T
    α::T
    p::T
    # Constraints and Defaults
    Continuity_function(k,x,y,z) = begin
        @assert k > 7 "At least 8 nearest neighbors must be in the δ-ball, in order to gurantee a valid statistic."
        @assert 0. < x ≤ 1. "Please select a valid `samplesize`, which denotes a fraction of considered fiducial points, i.e. `samplesize` ∈ (0 1]"
        @assert 1. > y > 0.
        @assert 1. > z > 0.
    end
    Continuity_function() = new(13,1.,0.05,0.5)
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

## Some Defaults for the MCDTSOptimGoal-struct:

# PECUZAL
Pecuzal() = MCDTSOptimGoal(L_statistic(), Continuity_function())
# Continuity & FNN-statistic
FNN() = MCDTSOptimGoal(FNN_statistic(), Continuity_function())
# For CCM-causality analysis
CCM() = MCDTSOptimGoal(CCM_ρ(), Range_function())
# For prediction with zeroth order predictor
Predict() = MCDTSOptimGoal(CCM_ρ(), Range_function())
