using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters

const STATE_SIZE = 2
const I₂ = Matrix{Float64}(I(STATE_SIZE))

@with_kw mutable struct POMDPscenario
    rng :: MersenneTwister
    F :: Matrix{Float64}  
    H :: Matrix{Float64}
    Σw :: Matrix{Float64}
    Σv₀ :: Matrix{Float64} = I₂
    beacons :: Matrix{Float64} = I₂
    d :: Float64 = 0.0
    rmin :: Float64 = 0.0
    cost :: Function
    costₜ :: Function
    𝒜 :: Vector{Vector{Float64}}
end

function OrderBeacons(x,y)::Matrix{Float64}
    X = x' .* ones(3)
    Y  = ones(3)' .* y
    beacons = hcat(X[:],Y[:])
    return beacons
end