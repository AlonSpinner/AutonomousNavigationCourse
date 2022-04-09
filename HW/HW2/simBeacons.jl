using Revise
using Distributions
using Random
using Plots
using StatsPlots
includet("./functions.jl") #include and track changes

function main()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = I₂
    b0 = MvNormal(μ0, Σ0)
    d =1.0 
    rmin = 0.1

    beacons = OrderBeacons(LinRange(0,10,3), LinRange(0,10,3))
    𝒫 = POMDPscenario(F= I₂,
                        H = I₂,
                        Σw = 0.1^2*I₂,
                        Σv = 0.01^2*I₂*1000, 
                        rng = rng , 
                        beacons=beacons, 
                        d=d, rmin=rmin) 
    ak = [0.1, 0.1]
    xgt0 = [-0.5, -0.2]
    T = 100

    #Initalization
    x_gt, x_kalman, x_deadreckoning  = xgt0, b0, b0
    Hist_gt, Hist_obs = [x_gt], []
    Hist_deadreckoning, Hist_kalman  = [b0], [b0]
    for _ in 1:T-1
        #move robot
        x_gt = SampleMotionModel(𝒫, ak, x_gt)

        #generate observation
        z = GenerateObservationFromBeacons(𝒫, x_gt)

        #generate beliefs
        x_deadreckoning = PropagateBelief(x_deadreckoning, 𝒫, ak)

        if isnothing(z)
            x_kalman = PropagateBelief(x_kalman, 𝒫, ak)
        else
            x_kalman = PropagateUpdateBelief(x_kalman, 𝒫, ak, z)
        end

        #record to history
        push!(Hist_gt,x_gt)
        push!(Hist_obs,z)
        push!(Hist_deadreckoning,x_deadreckoning)
        push!(Hist_kalman,x_kalman)
    end

    ##----- plot dead_reckoning
    p = plot(; xlabel="x", ylabel="y", aspect_ratio = 1.0,  grid=:true, legend=:outertopright, legendfont=font(5))
    covellipse!(Hist_deadreckoning[1].μ, Hist_deadreckoning[1].Σ, n_std=1, label = "est", color = "blue")
    for i in 2:T
        covellipse!(Hist_deadreckoning[i].μ, Hist_deadreckoning[i].Σ, n_std=1, label = "", color = "blue")
    end
    scatter!([x[1] for x in Hist_gt], [x[2] for x in Hist_gt], label="gt")
    scatter!([b[1] for b in beacons], [b[2] for b in beacons], label="beacons", markershape = :hexagon)
    savefig(p,"simBeacons_dead_reckoning.pdf")

    ##----- plot kalman_filter
    p = plot(; xlabel="x", ylabel="y", aspect_ratio = 1.0,  grid=:true, legend=:outertopright, legendfont=font(5))
    covellipse!(Hist_kalman[1].μ, Hist_kalman[1].Σ, n_std=1, label = "est", color = "blue")
    for i in 2:T
        covellipse!(Hist_kalman[i].μ, Hist_kalman[i].Σ, n_std=1, label = "", color = "blue")
    end
    scatter!([x[1] for x in Hist_gt], [x[2] for x in Hist_gt], label="gt")
    scatter!([x[1] for x in Hist_obs], [x[2] for x in Hist_obs], label="measurements", markersize=3)
    scatter!([b[1] for b in beacons], [b[2] for b in beacons], label="beacons", markershape = :hexagon)
    savefig(p,"simBeacons_kalman.pdf")

    print("finished\n")
end

main()

