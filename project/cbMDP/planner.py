from cbMDP.robot import robot
from cbMDP.solver import solver
from copy import deepcopy

import gtsam
from gtsam.symbol_shorthand import L, X

import numpy as np
from numpy.linalg import inv
from numpy import trace

EPS = 0.001

class planner():

    def __init__(self):
        self.simCar : robot = robot()
        self.car_dx = 0 #for u -> Pose2(car_dx,0,u) in innerLayer
        self.eps = 0
        self.beta = 0
        self.alpha_LB = 0
        self.alpha_km1 = 0
        self.M_sigmaBar = 0 #selection matrix for I, bottom page 21, for us its just an integer of last robot state
        self.M_xBar = 0 #selection matrix for X, bottom page 21
        self.M_u = 0.1 #weight matrix for u, page 21
        self.lambDa = 0 #stepsize for gradient decent
        self.alpha_km1 = 0 #required to compute 
    
    def outerLayer(self,backend,u0,goal): #plan
        J, J_prev = 0, 0
        i = 0

        #set weight matrices
        C_kpL_bar = self.innerLayer4alpha(deepcopy(backend),u0)

        alpha_k = min(trace(C_kpL_bar)/self.beta, 1)
        if self.alpha_km1 == 1 and alpha_k > self.alpha_LB: #alpha_k ~ 1 when uncertinity is high
            alpha_k = 1
        M_x = (1-alpha_k) * self.M_xBar
        M_sigma = np.sqrt(alpha_k) * self.M_sigmaBar
        self.alpha_km1 = alpha_k #update alpha_km1 for next planning session

        while True:
            #update u
            u = u - self.lambDa * self.computeGradient(X_k, I_k, u, self.M_u, M_x, M_sigma)

            #check convergence
            if np.linalg.norm(J) < EPS or np.linalg.norm((J-J_prev)/J_prev) < EPS or i > self.L:
                return u
            
            i += 1

    def computeGradient(self, X_k : np.ndarray, I_k : np.ndarray, u : np.ndarray, M_x : np.ndarray, M_sigma: np.ndarray) -> np.ndarray:
        #M_u and L provided from self
        J = self.evaluateObjective(X_k, I_k, u, M_x, M_sigma)
        dJ = np.zeros(u.size)
        for i in range(u.size):
            du = np.zeros_like(u)
            du[i] = EPS
            Ji = self.evaluateObjective(X_k,I_k,u + du ,M_x, M_sigma)
            dJ[i] = (Ji-J)/EPS
        return dJ

    def evaluateObjective(self, Xk : np.ndarray, Ik : np.ndarray, u : np.ndarray, M_x : np.ndarray, M_sigma: np.ndarray) -> float:
        #M_u and L provided from self
        J = 0
        for i in range(self.L-1):
            J += mahalanobis2(u[i],self.M_u)

        for i in range(self.L):
            X_kplBar, I_kplBar, I_kpl = self.innerLayer(Xk,Ik,u)

            J += trace(M_sigma @ inv(I_kpl) @ M_sigma)
            #currently skipping third term from equation 41, even though it rewards loop closure     
        return J

    def innerLayer4alpha(self, backend : solver ,u : np.ndarray):
        #returns covariance of X_kpL given no landmark measurements
            for u_kpl in u:
                backend.addOdomMeasurement(gtsam.Pose2(self.car_dx,0,u_kpl))
            return backend.isam2.marginalCovariance(X(backend.i + len(u))) #backend.i is the "time index" of the solver

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

            covs.append(backend.marginalCovariance(X(backend.i + l + 1))) #i == k
            ests.append(backend.X_kplp1)


def mahalanobis2(a : np.ndarray ,S : np.ndarray):
    return a.T @ inv(S) @ a

def zeta(u) -> float:
    #bottom of page 18 - some known function that quantifies the usage of control u
    return u**2