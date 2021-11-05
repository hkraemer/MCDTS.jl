using DelayEmbeddings
using DynamicalSystemsBase
using Random
using Neighborhood
using Distances
using Statistics
using LinearAlgebra
using DataFrames
using StatsBase
using GLM
using Revise


## Methods for delay preselection stat

# DelayPreselect(c::Continuity_function, ...) != ϵ
# DelayPreselect(c::Range_function, ...) != ϵ
# ....


## Methods for Loss stats
# Loss(c::L_function, ...) != L_pot
# Loss(c::FNN, ...) != L_pot
# Loss(c::CCM, ...) != L_pot
# ...



function give_potential_delays(OptimType::AbstractMCDTSOptimGoal, Yss::Dataset{D, T}, τs, w::Int, τ_vals, ts_vals, L_old;
                samplesize::Real=1, K::Int = 13, α::Real = 0.05, p::Real = 0.5,
                KNN::Int = 3, FNN::Bool = false, PRED::Bool = false, Tw::Int = 1,
                threshold::Real = 0, tws::AbstractRange{Int} = 2:τs[end],
                linear::Bool=false, PRED_mean::Bool=false, PRED_L::Bool=false,
                PRED_KL::Bool=false, CCM::Bool=false, Y_CCM = Dataset(zeros(size(Yss)))) where {D, T}
