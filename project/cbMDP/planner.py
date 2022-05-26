from cbMDP.robot import robot
from cbMDP.solver import solver
from copy import deepcopy

import gtsam
from gtsam.symbol_shorthand import L, X

import numpy as np
from numpy.linalg import inv
from numpy import trace

class planner():

    def __init__(self):
        self.simCar : robot = robot()
        self.car_dx = 0 #for u -> Pose2(car_dx,0,u) in innerLayer
        self.eps = 0.001
        self.beta = 0
        self.alpha_LB = 0
        self.alpha_km1 = 0
        self.M_u = 0.1 #weight matrix for u, page 21
        self.lambDa = 0 #stepsize for gradient decent
        self.alpha_km1 = 0 #required to compute 
        self.i_max = 10 #maximum number of iterations for graident decent
    
    def outerLayer(self,backend,u0,goal): #plan
        J, J_prev = 0, 0
        i = 0

        #set weight matrices
        cov_kpL_bar = self.innerLayer4alpha(deepcopy(backend),u0)
        alpha_k = min(trace(cov_kpL_bar)/self.beta, 1)
        if self.alpha_km1 == 1 and alpha_k > self.alpha_LB: #alpha_k ~ 1 when uncertinity is high
            alpha_k = 1
        M_x = (1-alpha_k) * np.eye(2)
        M_sigma = np.sqrt(alpha_k)
        self.alpha_km1 = alpha_k #update alpha_km1 for next planning session

        while True:
            #update u
            u = u - self.lambDa * self.computeGradient(deepcopy(backend), u, M_x, M_sigma, goal)

            #check convergence
            if np.linalg.norm(J) < self.eps or np.linalg.norm((J-J_prev)/J_prev) < self.eps or i > self.i_max:
                return u
            
            i += 1

    def computeGradient(self, backend : solver, u : np.ndarray, M_x : float, M_sigma: float, goal : np.ndarray) -> np.ndarray:
        #M_u and L provided from self
        J = self.evaluateObjective(deepcopy(backend), u, M_x, M_sigma, goal)
        dJ = np.zeros(u.size)
        for i in range(u.size):
            du = np.zeros_like(u)
            du[i] = self.eps
            Ji = self.evaluateObjective(deepcopy(backend),u + du ,M_x, M_sigma, goal)
            dJ[i] = (Ji-J)/self.eps
        return dJ

    def evaluateObjective(self, backend, u : np.ndarray, M_x : float, M_sigma: float, goal : np.ndarray) -> float:
        #M_u and L provided from self
        J = 0
        for u_kpl in u:
            J += mahalanobisISqrd(zeta(u_kpl),self.M_u)

        for l, u_kpl in enumerate(u):
            est,cov = self.innerLayer(backend,u_kpl)
            J += M_sigma**2 * trace(cov)
            #currently skipping third term from equation 41, even though it rewards loop closure

        J += mahalanobisISqrd(est-goal,M_x)

        return J

    def innerLayer4alpha(self, backend : solver ,u : np.ndarray):
        #returns covariance of X_kpL given no landmark measurements
            for u_kpl in u:
                backend.addOdomMeasurement(gtsam.Pose2(self.car_dx,0,u_kpl))
            return backend.isam2.marginalCovariance(X(backend.i + len(u))) #i == k

    def innerLayer(self, backend : solver, u : np.ndarray):
        covs = []
        ests = []
        #returns covariances of X_kpl
        for l, u_kpl in enumerate(u):
            a_kpl = gtsam.Pose2(self.car_dx,0,u_kpl)
            backend.addOdomMeasurement(a_kpl)
            
            X_kplp1 = backend.isam2.calculateEstimate(X(backend.i + l + 1))
            lms =  self.measureLandmarks(backend, X_kplp1) #need to write this down
            backend.addlandmarkMeasurement(lms)

            covs.append(backend.isam2.marginalCovariance(X(backend.i + l + 1))) #i == k
            ests.append(backend.isam2.calculateEstimate(X(backend.i + l + 1)))
        
        return ests,covs


def mahalanobisISqrd(a : np.ndarray ,S : np.ndarray):
    return a.T @ S @ a

def zeta(u) -> float:
    #bottom of page 18 - some known function that quantifies the usage of control u
    return u**2