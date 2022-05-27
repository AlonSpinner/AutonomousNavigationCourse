from cbMDP.robot import robot
from cbMDP.solver import solver
from cbMDP.utils.datatypes import meas_odom, meas_landmark

import gtsam
from gtsam.symbol_shorthand import L, X

import numpy as np
from numpy import trace
import matplotlib.pyplot as plt

class planner():

    def __init__(self, r_dx : float, r_cov_w : np.ndarray, \
                    r_cov_v : np.ndarray, r_range : float, r_FOV : float, axes : plt.Axes = None):
        self.epsConv : float = 1e-7
        self.epsGrad : float = 1e-10
        self.beta : float = 0.5 #[m^2]
        self.alpha_LB  : float = 0.2 #Not stated in article
        self.M_u = 0.1 #weight matrix for u, page 21
        self.lambDa : float = 0.1 #larger number allows for bigger turns
        self.i_max : int = 10 #maximum number of iterations for graident decent
        self.dx = r_dx #for u -> Pose2(robot_dx,0,u) in innerLayer
        self.cov_w : np.ndarray = r_cov_w
        self.cov_v : np.ndarray = r_cov_v
        self.range : float = r_range
        self.FOV : float = r_FOV
        self.axes = axes
        self.graphic_plan = [] #placeholder
    def outerLayer(self,backend : solver ,u : np.ndarray ,goal : np.ndarray): #plan
        J_prev = 1e10 #absurdly big number as initial value
        i = 0

        #set weight matrices
        cov_kpL_bar = self.innerLayer4alpha(backend.copyObject(),u)
        alpha_k = max(min(trace(cov_kpL_bar)/self.beta, 1),self.alpha_LB)
        print(alpha_k)
        M_x = 1-alpha_k
        M_sigma = np.sqrt(alpha_k)

        while True:
            #update u
            dJ = self.computeGradient(backend.copyObject(), u, M_x, M_sigma, goal)
            u = u - self.lambDa * dJ
            # u[u > np.pi/4] = np.pi/4

            #check convergence
            J = self.evaluateObjective(backend.copyObject(), u, M_x, M_sigma, goal)
            if np.linalg.norm(dJ) < self.epsConv:
                print('small graident')
                return u, J
            if np.linalg.norm((J-J_prev)/(J_prev + self.epsConv)) < self.epsConv:
                print('small change in J')
                return u, J
            if i > self.i_max:
                print('max iterations for gradient decent')
                return u, J
            
            i += 1
            J_prev = J

    def computeGradient(self, backend : solver, u : np.ndarray, M_x : float, M_sigma: float, goal : np.ndarray) -> np.ndarray:
        #M_u and L provided from self
        dJ = np.zeros(u.size)
        for i in range(u.size):
            du = np.zeros_like(u)
            du[i] = self.epsGrad
            Jpi = self.evaluateObjective(backend.copyObject(),u + du ,M_x, M_sigma, goal)
            Jmi = self.evaluateObjective(backend.copyObject(),u - du ,M_x, M_sigma, goal)
            dJ[i] = (Jpi-Jmi)/(2 * self.epsGrad)
        return dJ

    def evaluateObjective(self, backend : solver, u : np.ndarray, M_x : float, M_sigma: float, goal : np.ndarray) -> float:
        #M_u and L provided from self
        a = 0
        for u_kpl in u:
            a += zeta(u_kpl)**2 * self.M_u # mahalanobisISqrd doesnt work for scalars.. so...

        b = 0
        ests = []
        for l, u_kpl in enumerate(u):
            est,cov = self.innerLayer(backend,np.array([u_kpl]))
            b += M_sigma**2 * trace(cov)

            ests.append(est)
            # print(M_sigma)

        #use this formulation as we have no control on "gas" only on "wheel"
        dist = np.linalg.norm(np.array([est.translation() for est in ests]) - goal, axis = 1)
        c = M_x * min(dist)

        #currently skipping third term from equation 41, even though it rewards loop closure
        
        J = a + b + c
        return J

    def innerLayer4alpha(self, backend : solver ,u : np.ndarray):
        #returns covariance of X_kpL given no landmark measurements
            for u_kpl in u:
                backend.i += 1
                backend.addOdomMeasurement(meas_odom(gtsam.Pose2(self.dx,0,u_kpl),self.cov_w))
                backend.update()
            return backend.isam2.marginalCovariance(X(backend.i))

    def innerLayer(self, backend : solver, u : np.ndarray):
        covs = []
        ests = []
        
        for u_kpl in u:
            backend.i += 1

            backend.addOdomMeasurement(meas_odom(gtsam.Pose2(self.dx,0,u_kpl),self.cov_w))
            backend.update() #need to update to pull X_kpl for landmark measurements

            X_kpl = backend.isam2.calculateEstimatePose2(X(backend.i))
            lms =  self.simulateMeasuringLandmarks(backend, X_kpl) #need to write this down
            for lm in lms:
                backend.addlandmarkMeasurement(lm)
            backend.update()

            covs.append(backend.isam2.marginalCovariance(X(backend.i))) #i == k
            ests.append(backend.isam2.calculateEstimatePose2(X(backend.i)))
        
        if u.size == 1:
            ests = ests[0]
            covs = covs[0]

        return ests,covs

    def simulateMeasuringLandmarks(self, backend : solver, pose : gtsam.Pose2):
            meas = []
            for lm_index, lm_label in zip(backend.seen_landmarks["id"], backend.seen_landmarks["classLabel"]):
                lmML = backend.isam2.calculateEstimatePoint2(L(lm_index))    
                angle = pose.bearing(lmML).theta()
                r = pose.range(lmML)
                # if abs(angle) < self.FOV/2 and (r < self.range): #if viewed, compute noisy measurement
                cov_v_bar = self.cov_v * max(1,r/self.range)
                meas.append(meas_landmark(lm_index, r, angle, cov_v_bar , lm_label))
            return meas

        #create list of landmarks

    def plotPlan(self,u, backend, axes : plt.Axes):
        if ax is None:
            try: 
                ax = self.axes
            except:
                print('no axes provided to object')

        if self.graphic_plan:
            plan = np.zeros((1+u.size,2))
            belief = backend.isam2.calculateEstimatePose2(X(backend.i))
            plan[0,:] = np.array(belief.translation())
            for i,ui in enumerate(u):
                belief = belief.compose(gtsam.Pose2((self.dx,0,ui)))
                plan[1+i,:] = np.array(belief.translation())
            self.graphic_plan.set_data(plan[:,0],plan[:,1])
        else:
            self.graphic_plan = ax.plot([], [],'go-',markersize = 1)

def mahalanobisIsqrd(a : np.ndarray ,S : np.ndarray):
    return a.T @ S @ a

def zeta(u : float) -> float:
    #bottom of page 18 - some known function that quantifies the usage of control u
    return u # penalizes changes in direciton, page 45, will be squared later in cost