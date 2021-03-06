using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters

includet("./00_misc.jl")
includet("./01_models.jl")

function Plan(𝒫 :: POMDPscenario, b :: FullNormal, L :: Int)
    #returns action and cost
    if L <= 0
        return (a = nothing, J = 𝒫.costₜ(b))
    end
    
    best  = (a = nothing, J = Inf)
    for a in 𝒫.𝒜
        J = 𝒫.cost(b,a)
        b⁻ = PropagateBelief(𝒫, b, a)
        
        obs = ObservationModel(𝒫, b⁻) #Z::FullNormal
        if obs !== nothing
            zᵢ, wᵢ = generateSigmaPoints(obs.Z) 
            for i in 1:length(zᵢ)
                b⁺ = TranistBeliefMDP(𝒫, b, a, zᵢ[i], obs.r)
                a⁺, J⁺ = Plan(𝒫, b⁺, L-1)
                J += wᵢ[i] * J⁺
            end
        else
            a⁺, J⁺ = Plan(𝒫, b⁻, L-1)
            J += J⁺
        end
        
        if J < best.J
            best = (a = a, J = J)
        end
    end
    return best
end

function generateSigmaPoints(p::FullNormal; β = 2, α = 1, n = 2)
    #https://en.wikipedia.org/wiki/Unscented_transform
    #https://drive.google.com/file/d/0By_SW19c1BfhSVFzNHc0SjduNzg/view?resourcekey=0-41olC9ht9xE3wQe2zHZ45A
    κ = 3 - n
    λ = α^2 * (n+κ) - n
    M = sqrt(n+λ)*p.Σ

    points = []
    for i=1:n
        push!(points,p.μ + M[:,i])
    end
    for i=n+1:2*n
        push!(points,p.μ - M[:,i-n])
    end
    push!(points,p.μ)

    weights = 0.5/(n+λ) * ones(2*n+1)
    weights[2*n+1] = λ/(n+λ) #overwrite

    return points, weights
end