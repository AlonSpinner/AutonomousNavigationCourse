import numpy as np
import gtsam
import matplotlib.pyplot as plt

from cbMDP.utils import plotting

r = 1
m = 20 #steps
dtheta = np.pi * 2 / m
x0, y0, theta0 = 0, 0, 0 #circle trajectory parameters
pose_ii = []
for ii in range(m):
    theta = theta0 + ii*dtheta #circle parameter
    alpha = np.pi/2 +theta #ego tangent vector angle relative to x vector
    t_w_w2ii = np.array([[r * np.cos(theta)- x0],
                [r * np.sin(theta)- y0]])
    
    pose = gtsam.Pose2(t_w_w2ii[0],t_w_w2ii[1],alpha)
    pose_ii.append(pose)


_ , ax = plotting.spawnWorld(xrange = (-2*r,2*r), yrange = (-2*r,2*r))
for pose in pose_ii:
    plotting.plot_pose(ax ,pose)
    plt.pause(0.1)

dpose = (gtsam.Pose2.between(pose_ii[0],pose_ii[1]))
print(f"dpose type: {type(dpose)}")
print(f"dpose value {dpose}")
print(f"pose0 + dpose == pose1? {pose_ii[0].compose(dpose).equals(pose_ii[1],tol = 0.001)}")