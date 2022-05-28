import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.symbol_shorthand import L, X

import cbMDP.utils.plotting as plotting
from cbMDP.solver import solver
from cbMDP.robot import robot
from cbMDP.map import map
from cbMDP.planner import planner

from copy import deepcopy

def scenario():
    np.random.seed(seed=2)

    #------Build worldmap
    xrange = (-10,30); yrange = (-10,30)
    fig , ax = plotting.spawnWorld(xrange, yrange)
    
    N = 4
    worldMap = map()
    worldMap.fillMapRandomly(N,["rose"],(2,6),(5,7))
    worldMap.fillMapRandomly(N,["lily"],(2,8),(15,18))
    worldMap.fillMapRandomly(N,["hydrangea"],(12,15),(17,19))
    worldMap.fillMapRandomly(N,["tulip"],(17,19),(-2,2))
    worldMap.fillMapRandomly(N,["orchid"],(10,12),(-7,-9))
    worldMap.fillMapRandomly(N,["peony"],(-8,-6),(-2,-4))

    #------Spawn Robot
    pose0 = gtsam.Pose2(1.0,0.0,np.pi/2)
    car = robot(ax = ax, pose = pose0, FOV = np.radians(360), range = 2)
    dx = 1 #how much the robot goes forward in each timestep
    
    #----- Goals to visit
    goals = np.array([[3,6],
                        [4,17],
                        [14,18],
                        [18,0],
                        [11,-8],
                        [-7,-3]])
    targetRangeSwitch = 2

    return car, worldMap, ax, fig, goals, targetRangeSwitch, dx

car, worldMap, ax, fig, goals, targetRangeSwitch, dx = scenario()

#init estimator and controller
backend = solver(ax = ax, 
                X0 = car.pose ,X0cov = car.odometry_noise/1000, 
                semantics = worldMap.exportSemantics())
controller = planner(r_dx = dx, 
                    r_cov_w = car.odometry_noise, r_cov_v = car.rgbd_noise,
                    r_range = car.range, r_FOV = car.FOV, ax = ax)
u = np.zeros(5) #initial guess for action. Determines horizon aswell

#init loggers
hist_GT, hist_DR = car.pose.translation(), car.pose.translation()
targetIndex = 0

# set graphics
worldMap.plot(ax = ax, plotIndex = False, plotCov = False)
plotting.plot_goals(ax, goals)
graphic_GT_traj, = plt.plot([], [],'ko-',markersize = 1)
graphic_DR_traj, = plt.plot([], [],'ro-',markersize = 1)
graphic_Plan_traj, = plt.plot([], [],'go-',markersize = 1)
ax.set_title(f'target: {targetIndex}')

# run and plot simulation
xcurrent_DR = car.pose
with plt.ion():
    for k in range(0,1000):

        #switch target if reached previous
        if np.linalg.norm(car.pose.translation() - goals[targetIndex]) < targetRangeSwitch:
                targetIndex += 1
                ax.set_title(f'target: {targetIndex}')
        if targetIndex == len(goals):
                break #reached last goal

        #Controller
        # u, J, plannedBackend = controller.outerLayer(backend.copyObject(), u0 ,goals[targetIndex]) #use previous u as initial condition
        u = controller.outerLayer(backend.copyObject(), u ,goals[targetIndex]) #use previous u as initial condition
        odom_cmd = gtsam.Pose2(dx,0,u[0])        

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
        # controller.plotPlan(u, plannedBackend, ax)
        # graphic_GT_traj.set_data(hist_GT[:,0],hist_GT[:,1]) #plot ground truth trajectory
        # graphic_DR_traj.set_data(hist_DR[:,0],hist_DR[:,1])
        
        plt.pause(0.01)

plt.show()