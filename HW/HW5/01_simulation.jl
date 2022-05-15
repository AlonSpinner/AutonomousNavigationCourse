using Revise
using Distributions
using Random
using Plots
using StatsPlots
using Colors, ColorSchemes
includet("./00_functions.jl") #include and track changes

function main()
    xgoal = [9,9]
    xgt = [-0.5, -0.2] #initial

    𝒫 = POMDPscenario(F= I₂,
                        H = I₂,
                        Σw = 0.1^2*I₂,
                        Σv₀ = 0.01^2*I₂, 
                        rng = MersenneTwister(1) , 
                        beacons=OrderBeacons(LinRange(0,9,3), LinRange(0,9,3)), 
                        d=1.0, rmin=0.1) 

    T = 15 #steps 
    𝒜 = [[1,0],[-1,0],[0,1],[0,-1],[1/√(2),1/√(2)],[-1/√(2),1/√(2)],[1/√(2),-1/√(2)],[-1/√(2),-1/√(2)],[0,0]] #action space
   
    #cost functions
    λ = 0.5
    cost(b,a) = norm(b.μ-xgoal) - λ*det(b.Σ) 
    costₜ(b) = cost(b,0) #'partial' for noobs

    #Simulation!
    μ0 = [0.0,0.0]
    Σ0 = I₂
    b = MvNormal(μ0, Σ0)
    for t in 1:T
        #plan
        a, cost = Plan(𝒫, b, 𝒜, cost, costₜ)
        
        #act
        xgt = SampleMotionModel(𝒫, a, x_gt)     

        #obeserve
        z = GenerateObservation(𝒫, xgt)
        
        #update belief
        b = TranistBeliefMDP(𝒫, b, a, z)
    end
end

main()

