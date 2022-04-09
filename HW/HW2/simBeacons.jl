using Revise
using Distributions
using Random
using Plots
using StatsPlots
includet("./functions.jl") #include and track changes
using .ModelsAndScenario

function main()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    b0 = MvNormal(μ0, Σ0)
    d =1.0 
    rmin = 0.1
    # set beacons locations 
    beacons = [0.0 1.0;
                0.0 2.0]
    𝒫 = POMDPscenario(F= [1.0 0.0; 0.0 1.0],
                        H = [1.0 0.0; 0.0 1.0],
                        Σw = 0.1^2*[1.0 0.0; 0.0 1.0],
                        Σv = 0.01^2*[1.0 0.0; 0.0 1.0], 
                        rng = rng , beacons=beacons, d=d, rmin=rmin) 
    ak = [1.0, 0.0]
    xgt0 = [0.5, 0.2]
    T = 10

    #Initalization
    x_gt, x_kalman, x_deadreckoning  = xgt0, b0, b0
    Hist_gt, Hist_obs_gps = [x_gt], []
    Hist_deadreckoning, Hist_kalman_gps  = [b0], [b0]
    for _ in 1:T-1
        #move robot
        x_gt = SampleMotionModel(𝒫, ak, x_gt)

        #generate GPS observation
        z_gps = GenerateObservation(𝒫, x_gt)

        #generate beliefs
        x_deadreckoning = PropagateBelief(x_deadreckoning, 𝒫, ak)
        x_kalman_gps = PropagateUpdateBelief(x_kalman, 𝒫, ak, z_gps)

        #record to history
        push!(Hist_gt,x_gt)
        push!(Hist_obs_gps,z_gps)
        push!(Hist_deadreckoning,x_deadreckoning)
        push!(Hist_kalman_gps,x_kalman_gps)
    end

    ##----- plot dead_reckoning
    p = plot(; xlabel="x", ylabel="y", aspect_ratio = 1.0,  grid=:true, legend=:outertopright, legendfont=font(5))
    scatter!([x[1] for x in Hist_gt], [x[2] for x in Hist_gt], label="gt")
    for i in 1:T
        covellipse!(Hist_deadreckoning[i].μ, Hist_deadreckoning[i].Σ, n_std=1, label="step $i")
    end
    savefig(p,"dead_reckoning.pdf")

    ##----- plot kalman_filter
    p = plot(; xlabel="x", ylabel="y", aspect_ratio = 1.0,  grid=:true, legend=:outertopright, legendfont=font(5))
    scatter!([x[1] for x in Hist_obs_gps], [x[2] for x in Hist_obs_gps], label="gps measurements")
    for i in 1:T
        covellipse!(Hist_kalman_gps[i].μ, Hist_kalman_gps[i].Σ, n_std=1, label="step $i")
    end
    savefig(p,"Hist_kalman_gps.pdf")
end

main()

