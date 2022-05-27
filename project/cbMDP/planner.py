from cbMDP.robot import robot
from cbMDP.solver import solver
from cbMDP.utils.datatypes import meas_odom, meas_landmark

import gtsam
from gtsam.symbol_shorthand import L, X

import numpy as np
from numpy import trace

class planner():

    def __init__(self, r_dx : float, r_cov_w : np.ndarray, r_cov_v : np.ndarray, r_range : float, r_FOV : float):
        self.epsConv : float = 0.001
        self.epsGrad : float = 1e-3
        self.beta : float = 0.5 #[m^2]
        self.alpha_LB  : float = 0.2 #Not stated in article
        self.alpha_km1  : float = self.alpha_LB #initalizaton. Just go towards goal <-> low alpha
        self.M_u = 0.1 #weight matrix for u, page 21
        self.lambDa : float = 0.01 #larger number allows for bigger turns
        self.i_max : int = 10 #maximum number of iterations for graident decent
        self.dx = r_dx #for u -> Pose2(robot_dx,0,u) in innerLayer
        self.cov_w : np.ndarray = r_cov_w
        self.cov_v : np.ndarray = r_cov_v
        self.range : float = r_range
        self.FOV : float = r_FOV
    
    def outerLayer(self,backend : solver ,u : np.ndarray ,goal : np.ndarray): #plan
        J_prev = 1e10 #absurdly big number as initial value
        i = 0

        #set weight matrices
        cov_kpL_bar = self.innerLayer4alpha(backend.copyObject(),u)
        alpha_k = max(min(trace(cov_kpL_bar)/self.beta, 1),self.alpha_LB)
        if self.alpha_km1 == 1 and alpha_k > self.alpha_LB: #alpha_k ~ 1 when uncertinity is high
            alpha_k = 1
        M_x = (1-alpha_k) * np.eye(2)
        M_sigma = np.sqrt(alpha_k)
        self.alpha_km1 = alpha_k #update alpha_km1 for next planning session

        while True:
            #update u
            dJ = self.computeGradient(backend.copyObject(), u, M_x, M_sigma, goal)
            u = u - self.lambDa * dJ

            #check convergence
            J = self.evaluateObjective(backend.copyObject(), u, M_x, M_sigma, goal)
            if np.linalg.norm(dJ) < self.epsConv:
                print('small graident')
                return u
            if np.linalg.norm((J-J_prev)/(J_prev + self.epsConv)) < self.epsConv:
                print('small change in J')
                return u
            if i > self.i_max:
                print('max iterations for gradient decent')
                return u
            
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
            a += self.M_u * zeta(u_kpl)**2 # mahalanobisISqrd doesnt work for scalars.. so...

        b = 0
        for l, u_kpl in enumerate(u):
            est,cov = self.innerLayer(backend,np.array([u_kpl]))
            b += M_sigma**2 * trace(cov)
            # print(M_sigma)

        c = mahalanobisIsqrd(est.translation()-goal,M_x)

        #currently skipping third term from equation 41, even though it rewards loop closure

        J = a + b + c
        print(b/J)
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
                cov_v_bar = self.cov_v * max(1,r/self.range)**4
                meas.append(meas_landmark(lm_index, r, angle, cov_v_bar , lm_label))
            return meas

        #create list of landmarks


def mahalanobisIsqrd(a : np.ndarray ,S : np.ndarray):
    return a.T @ S @ a

def zeta(u : float) -> float:
    #bottom of page 18 - some known function that quantifies the usage of control u
    return u # penalizes changes in direciton, page 45, will be squared later in cost