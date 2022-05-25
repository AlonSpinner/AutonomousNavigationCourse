import numpy as np
from numpy.linalg import inv
from numpy import trace

EPS = 0.001

class planner():

    def __init__(self):
        self.eps = 0
        self.beta = 0
        self.alpha_LB = 0
        self.alpha_km1 = 0
        self.M_sigmaBar = 0 #selection matrix for I
        self.M_xBar = 0 #selection matrix for X
        self.M_u = 0 #weight matrix for u
        self.lambDa = 0 #stepsize for gradient decent
        self.alpha_km1 = 0 #required to compute 
        self.L = 0 #Horrizon
    
    def outerLayer(self,X_k,I_k,u): #plan
        J, J_prev = 0, 0
        i = 0

        #set weight matrices
        I_kpL_bar = self.innerLayer(X_k,u)
        alpha_k = min(np.trace(self.M_sigmaBar @ inv(I_kpL_bar) @ self.M_sigmaBar.T)/self.beta, 1)
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
        for i in range(self.L):
            X_kplBar, I_kplBar, I_kpl = self.innerLayer(Xk,Ik,u)
            J += mahalanobis2(u[i],self.M_u)
            J += trace(M_sigma @ inv(I_kpl) @ M_sigma)
            #currently skipping third term from equation 41       
        return J

    def innerLayer():
        pass

def mahalanobis2(a : np.ndarray ,S : np.ndarray):
    return a.T @ inv(S) @ a

def zeta(u) -> float:
    #bottom of page 18 - some known function that quantifies the usage of control u
    return u**2