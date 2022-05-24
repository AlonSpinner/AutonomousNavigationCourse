import numpy as np
import matplotlib.pyplot as plt
import gtsam


import cbMDP.utils.plotting as plotting
from cbMDP.solver import solver
from cbMDP.robot import robot
from cbMDP.map import map


def scenario():
    np.random.seed(seed=2)

    #------Build worldmap
    xrange = (-10,20); yrange = (-10,20)
    fig , ax = plotting.spawnWorld(xrange, yrange)
    
    worldMap = map()
    worldMap.fillMapRandomly(10,["rose"],(2,6),(5,7))
    worldMap.fillMapRandomly(10,["lily"],(2,8),(15,18))
    worldMap.fillMapRandomly(10,["hydrangea"],(14,17),(17,19))
    worldMap.fillMapRandomly(10,["tulip"],(17,19),(-2,2))
    worldMap.fillMapRandomly(10,["orchid"],(10,12),(-7,-9))
    worldMap.fillMapRandomly(10,["peony"],(-8,-6),(-2,-4))

    #------Spawn Robot
    pose0 = gtsam.Pose2(1.0,0.0,np.pi/2)
    car = robot(ax = ax, pose = pose0, FOV = np.radians(90), range = 2)
    vel = 2.0
    
    #----- Goals to visit
    goals = np.array([[3,6],
                        [4,17],
                        [16,18],
                        [18,0],
                        [11,-8],
                        [-7,-3]])

    return car, worldMap, ax, fig, goals, vel

car, worldMap, ax, fig, goals, vel = scenario()
worldMap.plot(ax = ax, plotIndex = False, plotCov = False)
plotting.plot_goals(ax, goals)


# backend = solver(ax = ax,X0 = car.pose ,X0cov = car.odometry_noise/1000, semantics = worldMap.exportSemantics())

# #init history loggers
# hist_GT, hist_DR = car.pose.translation(), car.pose.translation()

# #set graphics
# graphic_GT_traj, = plt.plot([], [],'ko-',markersize = 1)
# graphic_DR_traj, = plt.plot([], [],'ro-',markersize = 1)

# #run and plot simulation
# xcurrent_DR = car.pose
# with plt.ion():
#     for odom in gt_odom:
#         meas_odom = car.moveAndMeasureOdometrey(odom)
#         meas_lms = car.measureLandmarks(worldMap.landmarks)

#         backend.i += 1 #increase time index. Must be done before adding measurements as factors
#         backend.addOdomMeasurement(meas_odom)
#         for meas_lm in meas_lms:
#             backend.addlandmarkMeasurement(meas_lm)
#         backend.update(N=0)
        
#         #log history
#         hist_GT = np.vstack([hist_GT,car.pose.translation()])

#         #plot
#         car.plot()
#         backend.plot()

#         graphic_GT_traj.set_data(hist_GT[:,0],hist_GT[:,1])
        
#         plt.pause(0.5)

plt.show()