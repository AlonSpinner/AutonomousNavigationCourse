import numpy as np
import gtsam
import matplotlib.pyplot as plt

from cbMDP.robot import robot
import cbMDP.utils.plotting as plotting

pose0 = gtsam.Pose2(1.0,0.0,np.pi/2)
car = robot(pose = pose0)

#control inputs
r = 1
m = 20
dx = 0.309017 #taken from test 00: relative odometrey
dy = 0.0489435
dtheta = 0.314159
gt_odom = [gtsam.Pose2(dx,dy,dtheta)] * 2*m

#init history loggers
hist_GT, hist_DR = car.pose.translation(), car.pose.translation()

#set graphics
_ , ax = plotting.spawnWorld(xrange = (-2*r,2*r), yrange = (-2*r,2*r))
graphic_GT_traj, = plt.plot([], [],'ko-',markersize = 5)
graphic_DR_traj, = plt.plot([], [],'ro-',markersize = 5)
ax.legend(["ground truth","dead reckoning"])

#run and plot simulation
xcurrent_DR = car.pose
with plt.ion():
    for odom in gt_odom:
        meas_odom = car.moveAndMeasureOdometrey(odom)
        
        #dead reckoning integration
        xcurrent_DR = xcurrent_DR.compose(meas_odom.dpose)
        
        #log history
        hist_DR = np.vstack([hist_DR,xcurrent_DR.translation()])
        hist_GT = np.vstack([hist_GT,car.pose.translation()])

        #plot
        graphic_GT_traj.set_data(hist_GT[:,0],hist_GT[:,1])
        graphic_DR_traj.set_data(hist_DR[:,0],hist_DR[:,1])
        plt.pause(0.1)