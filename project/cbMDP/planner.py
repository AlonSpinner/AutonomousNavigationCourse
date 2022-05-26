from cbMDP.robot import robot
from cbMDP.solver import solver
from cbMDP.utils.datatypes import meas_odom, meas_landmark

import gtsam
from gtsam.symbol_shorthand import L, X

import numpy as np
from numpy import trace

class planner():

    def __init__(self, car_dx, cov_w, cov_v):
        self.car_dx = car_dx #for u -> Pose2(car_dx,0,u) in innerLayer
        self.eps = 0.001
        self.beta = 5.0 #[m^2]
        self.alpha_LB = 0.2 #Not stated in article
        self.alpha_km1 = self.alpha_LB #initalizaton. Just go towards goal <-> low alpha
        self.M_u = 0.1 #weight matrix for u, page 21
        self.lambDa = 0.1 #stepsize for gradient decent. Not stated in article
        self.i_max = 100.0 #maximum number of iterations for graident decent
        self.cov_w = cov_w
        self.cov_v = cov_v
    
    def outerLayer(self,backend : solver ,u : np.ndarray ,goal : np.ndarray): #plan
        J, J_prev = 0, 0
        i = 0

        #set weight matrices
        cov_kpL_bar = self.innerLayer4alpha(backend.copyObject(),u)
        alpha_k = min(trace(cov_kpL_bar)/self.beta, 1)
        if self.alpha_km1 == 1 and alpha_k > self.alpha_LB: #alpha_k ~ 1 when uncertinity is high
            alpha_k = 1
        M_x = (1-alpha_k) * np.eye(2)
        M_sigma = np.sqrt(alpha_k)
        self.alpha_km1 = alpha_k #update alpha_km1 for next planning session

        while True:
            #update u
            u = u - self.lambDa * self.computeGradient(backend.copyObject(), u, M_x, M_sigma, goal)

            #check convergence
            if np.linalg.norm(J) < self.eps or np.linalg.norm((J-J_prev)/J_prev) < self.eps or i > self.i_max:
                return u
            
            i += 1

    def computeGradient(self, backend : solver, u : np.ndarray, M_x : float, M_sigma: float, goal : np.ndarray) -> np.ndarray:
        #M_u and L provided from self
        J = self.evaluateObjective(backend.copyObject(), u, M_x, M_sigma, goal)
        dJ = np.zeros(u.size)
        for i in range(u.size):
            du = np.zeros_like(u)
            du[i] = self.eps
            Ji = self.evaluateObjective(backend.copyObject(),u + du ,M_x, M_sigma, goal)
            dJ[i] = (Ji-J)/self.eps
        return dJ

    def evaluateObjective(self, backend : solver, u : np.ndarray, M_x : float, M_sigma: float, goal : np.ndarray) -> float:
        #M_u and L provided from self
        J = 0
        for u_kpl in u:
            J += self.M_u * zeta(u_kpl)**2 # mahalanobisISqrd doesnt work for scalars.. so...

        for l, u_kpl in enumerate(u):
            est,cov = self.innerLayer(backend,[u_kpl])
            J += M_sigma**2 * trace(cov)
            #currently skipping third term from equation 41, even though it rewards loop closure

        J += mahalanobisIsqrd(est-goal,M_x)

        return J

    def innerLayer4alpha(self, backend : solver ,u : np.ndarray):
        #returns covariance of X_kpL given no landmark measurements
            for u_kpl in u:
                backend.i += 1
                backend.addOdomMeasurement(meas_odom(gtsam.Pose2(self.car_dx,0,u_kpl),self.cov_w))
                backend.update()
            return backend.isam2.marginalCovariance(X(backend.i))

    def innerLayer(self, backend : solver, u : np.ndarray):
        covs = []
        ests = []
        #returns covariances of X_kpl
        for u_kpl in u:
            backend.i += 1

            backend.addOdomMeasurement(meas_odom(gtsam.Pose2(self.car_dx,0,u_kpl),self.cov_w))
            
            X_kpl = backend.isam2.calculateEstimatePose2(X(backend.i))
            lms =  self.simulateMeasuringLandmarks(backend, X_kpl) #need to write this down
            backend.addlandmarkMeasurement(lms)
            backend.update()

            covs.append(backend.isam2.marginalCovariance(X(backend.i))) #i == k
            ests.append(backend.isam2.calculateEstimatePose2(X(backend.i)))
        
        return ests,covs

    def simulateMeasuringLandmarks(self, backend : solver, pose : gtsam.Pose2):
            meas = []
            for lm_index, lm_label in zip(backend.seen_landmarks["id"], backend.seen_landmarks["classLabel"]):
                lmML = backend.isam2.calculateEstimatePoint2(L(lm_index))         
                angle = self.pose.bearing(lmML.xy).theta()
                r = self.pose.range(lmML.xy)
                meas.append(meas_landmark(lm_index, r, angle, self.cov_v, lm_label))
            return meas

        #create list of landmarks


def mahalanobisIsqrd(a : np.ndarray ,S : np.ndarray):
    return a.T @ S @ a

def zeta(u : float) -> float:
    #bottom of page 18 - some known function that quantifies the usage of control u
    return u # penalizes changes in direciton, page 45, will be squared later in cost