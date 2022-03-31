# provides a simple struct that saves one tuple of embedding parameters 

abstract type AbstractEmbeddingPars end

"""
    EmbeddingPars{S,T}

`EmbeddingPars` save information of a single embedding cycle. 

# Fields

    * `τ`
    * `t`
    * `L` value of loss function 
    * `temp` (optional) additional information

"""
Base.@kwdef mutable struct EmbeddingPars{S,T} <: AbstractEmbeddingPars 
    τ::Integer 
    t::Integer 
    L::S  
    temp::T=nothing
end 

τ(e::EmbeddingPars) = e.τ
t(e::EmbeddingPars) = e.t
L(e::EmbeddingPars) = e.L 
temp(e::EmbeddingPars) = e.temp
τ(n::Nothing) = nothing
t(n::Nothing) = nothing
L(n::Nothing) = nothing 
temp(n::Nothing) = nothing
Base.show(io::IO, e::EmbeddingPars) = string("τ=",τ(e),", t=",t(e),", L=",L(e))