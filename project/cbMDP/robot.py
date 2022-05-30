import numpy as np
import matplotlib.pyplot as plt

import gtsam

from .utils.datatypes import landmark
from .utils.datatypes import meas_landmark
from .utils.datatypes import meas_odom
from .utils.datatypes import pose2ToNumpy

class robot:
    def __init__(self,odometry_noise = None, rgbd_noise = None,
                 FOV = 60.0*np.pi/180.0, range = 1.0, pose: gtsam.Pose2 = None, 
                 ax = None):

        #input assumptions
        if odometry_noise is None:
            odometry_noise = np.zeros((3,3))
            odometry_noise[0,0] = 0.1**2 #dx noise
            odometry_noise[1,1] = 0.1**2 #dy noise
            odometry_noise[2,2] =np.radians(1.1)**2 #angular noise

        if rgbd_noise is None:
            rgbd_noise = np.zeros((2,2))
            rgbd_noise[0,0] = 0.01**2 #depth noise
            rgbd_noise[1,1] = np.radians(1)**2 #angular noise

        #starting pose
        if pose is None: #not set in defaults as numpy arrays are mutable! it causes the class to rememmber inital value
            pose = gtsam.Pose2(0,0,0)
        self.pose = pose
        
        #sensor physics
        self.FOV = FOV #radians
        self.range = range #[m]
        self.odometry_noise = odometry_noise
        self.rgbd_noise = rgbd_noise
        
        #place holder for graphic handles
        self.graphic_rgbd = []
        self.graphic_car = []

        self.axes = ax

    def moveAndMeasureOdometrey(self,odom: gtsam.Pose2): 
        self.pose = self.pose.compose(odom)
        return self.noiseyOdomModel(odom)
    
    def measureLandmarks(self,landmarks):
            measurements = []
            for lm in landmarks:
                    meas = self.noiseyRgbdModel(lm)
                    if meas: #if measurement took place
                        measurements.append(meas)
            return measurements

    def noiseyRgbdModel(self,lm: landmark):
        gt_angle = self.pose.bearing(lm.xy).theta()
        gt_r = self.pose.range(lm.xy)

        if (gt_angle > -self.FOV/2 and gt_angle < +self.FOV/2) \
             and (gt_r < self.range): #if viewed, compute noisy measurement
            mu = np.array([gt_r,gt_angle])
            meas_dr, meas_dangle = np.random.multivariate_normal(mu, self.rgbd_noise) 
            return meas_landmark(lm.id, meas_dr, meas_dangle, self.rgbd_noise, lm.classLabel)

    def noiseyOdomModel(self,odom):
        meas_dx, meas_dy, meas_dtheta = np.random.multivariate_normal(pose2ToNumpy(odom), self.odometry_noise)
        dpose = gtsam.Pose2(meas_dx,meas_dy,meas_dtheta)
        return meas_odom(dpose,self.odometry_noise)

    def plot(self,ax: plt.Axes = None, markerSize = 200):
        if ax is None:
            try: 
                ax = self.axes
            except:
                print('no axes provided to robot previously via plot() or init()')


        #rgbd - cone graphics values
        phi = np.linspace(-self.FOV/2,self.FOV/2,10)
        xy_ego = self.range * np.array([np.cos(phi),np.sin(phi)])
        xy_world = np.vstack((self.pose.translation(),self.ego2World(xy_ego).T))

        if self.graphic_car: #car was plotted before
            self.graphic_car.set_offsets(self.pose.translation()) #https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
            self.graphic_rgbd.set_xy(xy_world)  #https://stackoverflow.com/questions/38341722/animation-to-translate-polygon-using-matplotlib
        else:
            self.graphic_car = ax.scatter(self.pose.x(),self.pose.y(),s = markerSize, c = 'k', marker = 'o')
            self.graphic_rgbd, = ax.fill(xy_world[:,0],xy_world[:,1], facecolor = "b" , alpha=0.1, animated = False)

    def world2Ego(self,xyWorld): #currently not used
        #accepts 2XM and return 2XM
        #pose rotation is Re2w, and translation is t_w_w2e
        Rw2e = self.pose.rotation().matrix().T
        t_e_e2w = Rw2e @ (-self.pose.translation().reshape(2,1)) #the reshape turns  the 1D array into a column vector
        xyEgo = Rw2e @ xyWorld + t_e_e2w
        return xyEgo

    def ego2World(self,xyEgo): #curently only used for rgbd graphics
        #accepts 2XM and return 2XM
        #pose rotation is Re2w, and translation is t_w_w2e
        xyWorld = self.pose.rotation().matrix() @ xyEgo + self.pose.translation().reshape(2,1) #the reshape turns the 1D array into a column vector
        return xyWorld
    

