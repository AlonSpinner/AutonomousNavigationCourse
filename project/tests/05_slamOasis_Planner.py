import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import gtsam

import cbMDP.utils.plotting as plotting
from cbMDP.solver import solver
from cbMDP.robot import robot
from cbMDP.map import map
from cbMDP.planner import planner, stupidController

def scenario():
    np.random.seed(seed=1)

    #------Build worldmap
    xrange = (-10,25); yrange = (-10,25)
    fig , ax = plotting.spawnWorld(xrange, yrange)
    
    N = 4
    worldMap = map()
    worldMap.fillMapRandomly(2*N,["rose"],(1,7),(4,8))
    worldMap.fillMapRandomly(N,["lily"],(2,8),(15,18))
    worldMap.fillMapRandomly(N,["hydrangea"],(14,17),(17,19))
    worldMap.fillMapRandomly(N,["tulip"],(12,14),(-2,2))
    worldMap.fillMapRandomly(N,["orchid"],(5,7),(-7,-9))
    worldMap.fillMapRandomly(N,["peony"],(-8,-6),(6,8))

    #------Spawn Robot
    pose0 = gtsam.Pose2(1.0,0.0,np.pi/2)
    car = robot(ax = ax, pose = pose0, FOV = np.radians(120), range = 3)
    dx = 1 #how much the robot goes forward in each timestep
    
    #----- Goals to visit
    goals = np.array([[3,6],
                        [4,17],
                        [16,18],
                        [13,0],
                        [6,-8],
                        [-7,7]])
    targetRangeSwitch = 2

    return car, worldMap, ax, fig, goals, targetRangeSwitch, dx

car, worldMap, ax, fig, goals, targetRangeSwitch, dx = scenario()

moviewriter = PillowWriter(fps = 18)
moviewriter.setup(fig,'05_movie.gif',dpi = 100)

#init estimator and controller
backend = solver(ax = ax, 
                X0 = car.pose ,X0cov = car.odometry_noise/1000, 
                semantics = worldMap.exportSemantics())
controller = planner(r_dx = dx, 
                    r_cov_w = car.odometry_noise, r_cov_v = car.rgbd_noise,
                    r_range = car.range, r_FOV = car.FOV, ax = ax)
controller.moviewriter = moviewriter
u0 = np.zeros(5) #initial guess for action. Determines horizon aswell

#init loggers
hist_GT, hist_DR = car.pose.translation(), car.pose.translation()
targetIndex = 0

# set graphics
worldMap.plot(ax = ax, plotIndex = False, plotCov = False)
plotting.plot_goals(ax, goals)
graphic_GT_traj, = plt.plot([], [],'ko-',markersize = 1)
graphic_DR_traj, = plt.plot([], [],'ro-',markersize = 1)
ax.set_title(f'target: {targetIndex}')
car.plot(markerSize = 10)

# run and plot simulation
xcurrent_DR = car.pose
realTime = False
k = 0
with plt.ion():
    while True:

        #switch target if reached previous
        if np.linalg.norm(car.pose.translation() - goals[targetIndex]) < targetRangeSwitch:
                targetIndex += 1
                ax.set_title(f'target: {targetIndex}')
        if targetIndex == len(goals):
                break #reached last goal

        #Controller
        if k < 0:
            u_stupid = stupidController(k, backend.copyObject(), goals[targetIndex])
            odom_cmd = gtsam.Pose2(dx,0,u_stupid)
        else:
            u, J, plannedBackend = controller.outerLayer(k, backend.copyObject(), u0 ,goals[targetIndex]) #use previous u as initial condition
            odom_cmd = gtsam.Pose2(dx,0,u[0])        
            u0[:-1] = u[1:]; u0[-1] = 0.0
            realTime = True
            #plannedBackend.plot()
        
        meas_odom = car.moveAndMeasureOdometrey(odom_cmd)
        meas_lms = car.measureLandmarks(worldMap.landmarks)

        backend.i += 1 #increase time index. Must be done before adding measurements as factors
        backend.addOdomMeasurement(meas_odom)
        for meas_lm in meas_lms:
            backend.addlandmarkMeasurement(meas_lm)
        backend.update(N=0)
        
        #dead reckoning integration
        xcurrent_DR = xcurrent_DR.compose(meas_odom.dpose)

        #update time index
        k += 1

        #log history
        hist_GT = np.vstack([hist_GT,car.pose.translation()])
        hist_DR = np.vstack([hist_DR,xcurrent_DR.translation()])

        #plot
        car.plot(markerSize = 10)
        backend.plot()
        # controller.plotPlan(u, plannedBackend, ax)
        # graphic_GT_traj.set_data(hist_GT[:,0],hist_GT[:,1]) #plot ground truth trajectory
        # graphic_DR_traj.set_data(hist_DR[:,0],hist_DR[:,1])
        
        moviewriter.grab_frame()
        plt.pause(0.01)

moviewriter.finish()
plt.show()