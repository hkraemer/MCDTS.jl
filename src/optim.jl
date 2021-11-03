abstract type AbstractMCDTSOptimGoal end

struct MCDTSOptimGoal <: AbstractMCDTSOptimGoal
    Γ::AbstractLoss
    Λ::AbstractDelayPreselection  
end

Pecuzal() = MCDTSOptimGoal(L_function(), Continuity_function())

abstract type AbstractDelayPreselection end

abstract type AbstractLoss end


struct L_function <: AbstractLoss

end

struct Continuity_function <: AbstractDelayPreselection

end



function give_potential_delays(OptimType::AbstractMCDTSOptimGoal, Yss::Dataset{D, T}, τs, w::Int, τ_vals, ts_vals, L_old;
                samplesize::Real=1, K::Int = 13, α::Real = 0.05, p::Real = 0.5,
                KNN::Int = 3, FNN::Bool = false, PRED::Bool = false, Tw::Int = 1,
                threshold::Real = 0, tws::AbstractRange{Int} = 2:τs[end],
                linear::Bool=false, PRED_mean::Bool=false, PRED_L::Bool=false,
                PRED_KL::Bool=false, CCM::Bool=false, Y_CCM = Dataset(zeros(size(Yss)))) where {D, T}


abstract type AbstractPredictionError <: AbstractMCDTSOptimGoal end
