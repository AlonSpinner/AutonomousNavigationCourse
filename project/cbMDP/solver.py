import gtsam #https://github.com/borglab/gtsam
from gtsam.symbol_shorthand import L, X
import numpy as np
from .utils.datatypes import meas_landmark
from .utils.datatypes import meas_odom
from .utils.datatypes import landmark
from .utils import plotting

from copy import deepcopy, copy

class solver:
    
    def __init__(self,X0 = None,X0cov = None, semantics = None ,ax = None):
        if X0 is None:
            X0 = gtsam.Pose2((0,0,0))
        if X0cov is None:
            X0cov = 0.001* np.eye(3)

        #initalize solver
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values() #only holds new variables in time step

        #insert X0 to initial_estiamte and graph as prior
        X0_prior_noise = gtsam.noiseModel.Gaussian.Covariance(X0cov)
        self.initial_estimate.insert(X(0), X0)
        self.graph.push_back(gtsam.PriorFactorPose2(X(0), X0, X0_prior_noise))

        #initalize solver isam2
        self.isam2Initalize() #sets self.isam2
        self.update()

        #time index
        self.i = 0 

        #seen_landmarks to avoid initalizing landmarks twice (breaks solver)
        self.seen_landmarks = {
                             "id": [],
                             "classLabel": []
                              }

        #for extensive plotting or data assosication
        #exported from map object. dictionary with following keys: "classLabel","color","marker", each holding a list, representing tabular data
        self.semantics = semantics 

        #graphics
        self.ax = ax #might be None, depending on input
        self.graphics_landmarks = []
        self.graphics_poses = []

    def update(self, N=0):
        self.isam2.update(self.graph, self.initial_estimate)
        for kk in range(N):
            self.isam2.update() #can be called additional times to perform multiple optimizer iterations
        
        self.initial_estimate.clear() #inital_estimates only holds guesses for new variables.learnt from Pose2ISAM2Example.

    def calculateEstimate(self, bestEstimate = False):
        if bestEstimate:
            # https://gtsam.org/doxygen/4.0.0/a03643.html#a9d6f2b0d018f817f64fee6abdaa413ff
            #compute estimate using full back-substituion
            return self.isam2.calculateBestEstimate()
        else:
            return self.isam2.calculateEstimate()

    def optimizeFullState(self):
            self.update() #update isam2 with newly added priors and clear inital_estimate
            isam2_estimate = self.calculateEstimate(bestEstimate = True)

            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, 
                                                          isam2_estimate,
                                                          params)
            global_estimate = optimizer.optimize()
            self.isam2Initalize() #reinitalize Isam2
            self.isam2.update(self.graph, global_estimate) #first update with global solver

    def isam2Initalize(self):
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.setRelinearizeSkip(1)
        self.isam2 = gtsam.ISAM2(parameters)
           
    def addOdomMeasurement(self,meas: meas_odom):
        odom_noise = gtsam.noiseModel.Gaussian.Covariance(meas.cov) #https://gtsam.org/doxygen/a03876.html
        
        factor = gtsam.BetweenFactorPose2(
                        X(self.i-1), X(self.i), 
                        meas.dpose, odom_noise)
        self.graph.push_back(factor)

        pose = self.isam2.calculateEstimatePose2(X(self.i-1))
        initial_Xi = pose.compose(meas.dpose)
        self.initial_estimate.insert(X(self.i), initial_Xi)
    
    def addlandmarkMeasurement(self,meas: meas_landmark):
        rgbd_noise = gtsam.noiseModel.Gaussian.Covariance(meas.cov) #https://gtsam.org/doxygen/a03876.html
        
        factor = gtsam.BearingRangeFactor2D(
                    X(self.i), L(meas.id),
                    gtsam.Rot2.fromAngle(meas.angle), meas.range, rgbd_noise)
        self.graph.push_back(factor)

        if meas.id not in self.seen_landmarks["id"]: #then mark it as seen and it add landmark to inital_estimate
            
            self.seen_landmarks["id"].append(meas.id)
            self.seen_landmarks["classLabel"].append(meas.classLabel)

            pose = self.initial_estimate.atPose2(X(self.i))
            xy_ego = meas.range * np.array([np.cos(meas.angle),np.sin(meas.angle)])
            xy_world = pose.transformFrom((xy_ego))
            initial_L = gtsam.Point2(xy_world)

            self.initial_estimate.insert(L(meas.id), initial_L)

    def addlandmarkPrior(self, lm: landmark):
        L_prior_noise = gtsam.noiseModel.Gaussian.Covariance(lm.cov) #https://gtsam.org/doxygen/a03876.html
        Li = gtsam.Point2(lm.xy)
        self.graph.push_back(gtsam.PriorFactorPoint2(L(lm.id), 
                                                     Li, 
                                                     L_prior_noise))
 
        if lm.id not in self.seen_landmarks["id"]:
            self.seen_landmarks["id"].append(lm.id)
            self.seen_landmarks["classLabel"].append(lm.classLabel)
            self.initial_estimate.insert(L(lm.id), Li)

    def plot(self,poses = True ,poses_axis_length = 0.1, poses_Cov = True, landmark_Cov = True,
                  landmarks = True, landmarks_Index = False, landmarks_Semantics = False,
                  bestEstimate = False):
        
        current_estimate = self.calculateEstimate(bestEstimate = bestEstimate)

        if landmark_Cov or poses_Cov:
            marginals = gtsam.Marginals(self.graph, current_estimate)
        else:
            marginals = False

        #wrapper function for plot_poses and plot_landmarks so we wont compute current_estimate and marginals twice

        if poses:
            self.plot_poses(current_estimate, landmark_Cov and marginals, #and returns second argument if both are true
                            axis_length = poses_axis_length)
        if landmarks:
            self.plot_landmarks(current_estimate, poses_Cov and marginals, #and returns second argument if both are true
                                    plotIndex = landmarks_Index, 
                                    plotSemantics = landmarks_Semantics)

    def plot_landmarks(self, current_estimate, marginals, plotIndex = False, plotSemantics = False):
        if self.ax is None:
            raise Exception("you must provide an axes handle to solver if you want to plot")
        if plotSemantics is True and self.semantics is None:
            raise Exception("You must provide semantics if you want to plot them. Currently empty.")

        #remove old drawings if exist
        for graphics_lm in self.graphics_landmarks:
            for graphic in graphics_lm:
                    graphic.remove()

        self.graphics_landmarks = []
        for seen_lm_index, seen_lm_classLabel in zip(self.seen_landmarks["id"],self.seen_landmarks["classLabel"]):

            #prep for plots
            index4plot = None if plotIndex is False else seen_lm_index
            if self.semantics is not None and plotSemantics is True:
                ii = self.semantics["classLabel"].index(seen_lm_classLabel) #find index of class label in list (this needs to be a hashmap)
                markerShape = self.semantics['marker'][ii]
            else:
                markerShape = '.' #default

            #plot
            loc = current_estimate.atPoint2(L(seen_lm_index))
            if marginals:
                cov = marginals.marginalCovariance(L(seen_lm_index))     
                self.graphics_landmarks.append(plotting.plot_landmark(self.ax, loc = loc, cov = cov,
                    markerColor = 'b', markerShape = markerShape, markerSize = 3,
                    index = index4plot, textColor = 'b'))
            else:
                self.graphics_landmarks.append(plotting.plot_landmark(self.ax, loc = loc,
                    markerColor = 'b', markerShape = markerShape, markerSize = 3,
                    index = index4plot, textColor = 'b'))

    def plot_poses(self, current_estimate, marginals, axis_length = 0.1):
        if self.ax is None:
            raise Exception("you must provide an axes handle to solver if you want to plot")

        #remove old drawings if exist
        for graphics_pose in self.graphics_poses:
                for graphic in graphics_pose:
                    graphic.remove()

        self.graphics_poses = []
        ii = 0
        while current_estimate.exists(X(ii)):
            pose = current_estimate.atPose2(X(ii))
            if marginals:
                cov = marginals.marginalCovariance(X(ii))
                self.graphics_poses.append(plotting.plot_pose(self.ax, pose, axis_length = axis_length, covariance = cov))
            else:
                self.graphics_poses.append(plotting.plot_pose(self.ax, pose, axis_length = axis_length))
            
            ii +=1

    def copyObject(self):
        #cant deepcopy regullarly as isam2 is not pickelable (dafaq) 
        copySolver = solver()

        copySolver.graph = deepcopy(self.graph)
        copySolver.initial_estimate = deepcopy(self.initial_estimate)
        copySolver.isam2 = self.isam2 #is mutable !
        copySolver.i = deepcopy(self.i)
        copySolver.seen_landmarks = deepcopy(self.seen_landmarks)

        # -------------------dont require these for computations
        # copySolver.semantics = deepcopy(self.semantics)
        # copySolver.ax = deepcopy(self.ax)
        # copySolver.graphics_landmarks = deepcopy(self.graphics_landmarks)
        # copySolver.graphics_poses = deepcopy(self.graphics_poses)
        gtsam.isam2
        return copySolver
