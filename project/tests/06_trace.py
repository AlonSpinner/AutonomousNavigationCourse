import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.symbol_shorthand import L, X

import cbMDP.utils.plotting as plotting
from cbMDP.solver import solver
from cbMDP.robot import robot
from cbMDP.map import map

def scenario():
    np.random.seed(seed=2)

    #------Build worldmap
    xrange = (-2,10); yrange = (-2,10)
    fig , ax = plotting.spawnWorld(xrange, yrange)
    
    #------landmarks
    worldMap = map()
    worldMap.plot(ax = ax, plotIndex = True, plotCov = False)

    #------Spawn Robot
    pose0 = gtsam.Pose2(1.0,0.0,np.pi/2)
    car = robot(ax = ax, pose = pose0, FOV = np.radians(90), range = 2)

    return car, worldMap, ax, fig

car, worldMap, ax, fig = scenario()
backend = solver(ax = ax,X0 = car.pose ,X0cov = car.odometry_noise/1000, semantics = worldMap.exportSemantics())

#init history loggers
hist_GT = car.pose.translation()

gt_odom = [gtsam.Pose2(2,0,0)] * 3 + [gtsam.Pose2(0,0,-np.pi/2)] + [gtsam.Pose2(2,0,0)] * 3
with plt.ion():
    for k,odom in enumerate(gt_odom):
        meas_odom = car.moveAndMeasureOdometrey(odom)
        meas_lms = car.measureLandmarks(worldMap.landmarks)

        backend.i += 1 #increase time index. Must be done before adding measurements as factors
        backend.addOdomMeasurement(meas_odom)
        backend.update(N=0)

        #plot
        car.plot(markerSize = 10)
        ax.set_title(f"trace of cov {np.trace(backend.isam2.marginalCovariance(X(k)))}")
        backend.plot()
        
        plt.pause(1)

plt.show()