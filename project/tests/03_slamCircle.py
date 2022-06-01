import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import gtsam


import cbMDP.utils.plotting as plotting
from cbMDP.solver import solver
from cbMDP.robot import robot
from cbMDP.map import map


def scenario():
    np.random.seed(seed=2)

    #------Build worldmap
    xrange = (-2,2); yrange = (-2,2)
    fig , ax = plotting.spawnWorld(xrange, yrange)
    
    #------landmarks
    N = 40 
    semantics = ("table","MEP","chair","pillar")
    worldMap = map()
    worldMap.fillMapRandomly(N,semantics,xrange,yrange)
    worldMap.plot(ax = ax, plotIndex = True, plotCov = False)

    #------Spawn Robot
    pose0 = gtsam.Pose2(1.0,0.0,np.pi/2)
    car = robot(ax = ax, pose = pose0, FOV = np.radians(90), range = 2)
    
    #------ ground truth odometrey
    r = 1; m = 20
    dx = 0.309017 ;dy = 0.0489435; dtheta = 0.314159
    gt_odom = [gtsam.Pose2(dx,dy,dtheta)] * 2*m

    return car, worldMap, ax, fig, gt_odom

car, worldMap, ax, fig, gt_odom = scenario()
backend = solver(ax = ax,X0 = car.pose ,X0cov = car.odometry_noise/1000, semantics = worldMap.exportSemantics())

#init history loggers
hist_GT, hist_DR = car.pose.translation(), car.pose.translation()

#set graphics
moviewriter = PillowWriter(fps = 5)
moviewriter.setup(fig,'03_movie.gif',dpi = 100)
graphic_GT_traj, = plt.plot([], [],'ko-',markersize = 1)
graphic_DR_traj, = plt.plot([], [],'ro-',markersize = 1)

#run and plot simulation
xcurrent_DR = car.pose
with plt.ion():
    for odom in gt_odom:
        meas_odom = car.moveAndMeasureOdometrey(odom)
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
        car.plot()
        ax.set_title([lm.id for lm in meas_lms])
        backend.plot()

        graphic_GT_traj.set_data(hist_GT[:,0],hist_GT[:,1])
        graphic_DR_traj.set_data(hist_DR[:,0],hist_DR[:,1])
        
        moviewriter.grab_frame()
        plt.pause(0.5)

moviewriter.finish()
plt.show()