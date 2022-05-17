using Revise
using Distributions
using Random
using Plots
using StatsPlots
using Colors, ColorSchemes
includet("./00_misc.jl")
includet("./01_models.jl")
includet("./02_plan.jl")

function main()
    x_goal = [4,-9]
    x_gt = [-0.5, -0.2] #initial

    λ = 0
    cost(b,a) = norm(b.μ-x_goal) - λ*det(b.Σ)
    costₜ(b) = norm(b.μ-x_goal) - λ*det(b.Σ)
    beacons = OrderBeacons(LinRange(0,9,3), LinRange(0,9,3))
    rng = MersenneTwister(1)
    𝒫 = POMDPscenario(
                        F= I₂,
                        H = I₂,
                        Σw = 0.1^2*I₂,
                        Σv₀ = 0.01^2*I₂, 
                        rng = rng, 
                        beacons = beacons, 
                        d=1.0, 
                        rmin=0.1,
                        𝒜 = [[1,0],[-1,0],[0,1],[0,-1],[1/√(2),1/√(2)],[-1/√(2),1/√(2)],[1/√(2),-1/√(2)],[-1/√(2),-1/√(2)],[0,0]],
                        cost = cost,
                        costₜ = costₜ
                        ) 

    T = 15 #steps 
    L = 2 #horrizon

    #Simulation!
    μ0 = [0.0,0.0]
    Σ0 = I₂
    b = MvNormal(μ0, Σ0)
    for t in 1:T
        #plan
        a, J = Plan(𝒫, b, L)
        
        #act
        x_gt = SampleMotionModel(𝒫, a, x_gt)

        #obeserve
        obs = GenerateObservation(𝒫, x_gt)
        
        #update belief via kalman
        b⁻ = PropagateBelief(𝒫, b, a) #first step
        if obs !== nothing
            b = UpdateBelief(𝒫, b⁻, obs.z, obs.r)
        else
            b = b⁻
        end

        println(a, b.μ)

    end
end

main()

