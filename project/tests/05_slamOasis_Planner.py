import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.symbol_shorthand import L, X

import cbMDP.utils.plotting as plotting
from cbMDP.solver import solver
from cbMDP.robot import robot
from cbMDP.map import map
from cbMDP.planner import planner

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
    dx = 1.0 #how much the robot goes forward in each timestep
    
    #----- Goals to visit
    goals = np.array([[3,6],
                        [4,17],
                        [16,18],
                        [18,0],
                        [11,-8],
                        [-7,-3]])
    targetRangeSwitch = 1.0

    return car, worldMap, ax, fig, goals, targetRangeSwitch, dx

car, worldMap, ax, fig, goals, targetRangeSwitch, dx = scenario()

#init estimator and controller
backend = solver(ax = ax,X0 = car.pose ,X0cov = car.odometry_noise/1000, semantics = worldMap.exportSemantics())
controller = planner(car_dx = dx, cov_w = car.odometry_noise, cov_v = car.rgbd_noise)
u = np.zeros(5) #initial guess for action. Determines horizon aswell

#init history loggers
hist_GT, hist_DR = car.pose.translation(), car.pose.translation()

# set graphics
worldMap.plot(ax = ax, plotIndex = False, plotCov = False)
plotting.plot_goals(ax, goals)
graphic_GT_traj, = plt.plot([], [],'ko-',markersize = 1)
graphic_DR_traj, = plt.plot([], [],'ro-',markersize = 1)

# run and plot simulation
xcurrent_DR = car.pose
targetIndex = 0
with plt.ion():
    for k in range(0,1000):

        #switch target if reached previous
        if np.linalg.norm(car.pose.translation() - goals[targetIndex]) < targetRangeSwitch:
                targetIndex += 1
        if targetIndex == len(goals):
                break #reached last goal

        #Controller
        u = controller.outerLayer(backend.copyObject(), u ,goals[targetIndex]) #use previous u as initial condition
        odom_cmd = gtsam.Pose2(dx,0,u)        

        meas_odom = car.moveAndMeasureOdometrey(odom_cmd)
        meas_lms = car.measureLandmarks(worldMap.landmarks)

        backend.i += 1 #increase time index. Must be done before adding measurements as factors
        backend.addOdomMeasurement(meas_odom)
        for meas_lm in meas_lms:
            backend.addlandmarkMeasurement(meas_lm)
        backend.update(N=0)
        
        #dead reckoning integration
        xcurrent_DR = xcurrent_DR.compose(meas_odom.dpose)

        #log history
        hist_GT = np.vstack([hist_GT,car.pose.translation()])
        hist_DR = np.vstack([hist_DR,xcurrent_DR.translation()])

        #plot
        car.plot(markerSize = 10)
        backend.plot()
        # graphic_GT_traj.set_data(hist_GT[:,0],hist_GT[:,1]) #plot ground truth trajectory
        # graphic_DR_traj.set_data(hist_DR[:,0],hist_DR[:,1])
        
        plt.pause(0.3)

plt.show()