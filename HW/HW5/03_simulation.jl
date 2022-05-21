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
    x_goal = [5,-5]
    x_gt = [-0.5, -0.2] #initial

    λ = 10000
    cost(b,a) = norm(b.μ-x_goal) + λ*det(b.Σ)
    costₜ(b) = norm(b.μ-x_goal) + λ*det(b.Σ)
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

    T = 10 #steps 
    L = 3 #horrizon

    #Simulation!
    μ0 = [0.0,0.0]
    Σ0 = I₂
    b = MvNormal(μ0, Σ0)
    collect_b = [b]
    collect_x_gt = []
    collect_a = []
    collect_z = []
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

        push!(collect_b, b)
        push!(collect_x_gt, x_gt)
        push!(collect_a, a)
        if obs !== nothing
            push!(collect_z, obs.z)
        end

        println("finished time step $t")
    end
    
    plt = plot(; xlabel="x", ylabel="y", aspect_ratio = 1.0,  grid=:true, legend=:outertopright, legendfont=font(5),
    title = "x_goal = $x_goal, L = $L, λ = $λ", titlefont = font(10))
    scatter!(beacons[:,1], beacons[:,2], label="beacons", markershape = :hexagon, color = "yellow")
    scatter!([x[1] for x in collect_x_gt], [x[2] for x in collect_x_gt], label="ground truth", color = "red")
    scatter!([x.μ[1] for x in collect_b], [x.μ[2] for x in collect_b], label="belief", color = "blue", markersize = 2)
    for i in 1:length(collect_b)
        covellipse!(collect_b[i].μ, collect_b[i].Σ, n_std=3, label = "", color = "blue")
    end
    scatter!([x[1] for x in collect_z], [x[2] for x in collect_z], label="measurements", color = "cyan", markersize = 2)
    scatter!([x_goal[1]], [x_goal[2]], label="goal",  markersize = 5, markershape = :star)

    #QUIVER DESTROYS LEGEND
    s = 0.5 #scale for quiver size
    quiver!(plt, [x.μ[1] for x in collect_b],[x.μ[2] for x in collect_b], 
    quiver = ([s*x[1] for x in collect_a],[s*x[2] for x in collect_a]),color = "black", label = "chosen actions")
    
    display(plt)
    savefig(plt,"./out/Q1_2.pdf")

    println("finished")
end

main()



