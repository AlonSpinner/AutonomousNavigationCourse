from cbMDP.robot import robot
from cbMDP.solver import solver
from cbMDP.utils.datatypes import meas_odom, meas_landmark

import gtsam
from gtsam.symbol_shorthand import L, X

import numpy as np
from numpy import trace
from numpy.linalg import norm
import matplotlib.pyplot as plt

class planner():

    def __init__(self, r_dx : float, r_cov_w : np.ndarray, \
                    r_cov_v : np.ndarray, r_range : float, r_FOV : float, ax : plt.Axes = None):
        self.k : int = 0 #time step, #46
        #"converger"
        self.epsConvGrad : float = 1e-5
        self.epsConvVal : float = 1e-4
        self.epsGrad : float = 1e-10
        self.lambDa : float = 0.005#0.005 #larger number allows for bigger turns
        self.i_max : int = 10 #maximum number of iterations for graident decent
        #weighting
        self.beta : float = 2.4 #[m^2]
        self.alpha_LB  : float = 0.6 #Not stated in article
        self.alpha_km1 : float = 0.0 #keep previous alpha_k
        self.M_u = 0.1 #0.1 #weight matrix for u, page 21
        #robot simulation
        self.dx : float = r_dx #for u -> Pose2(robot_dx,0,u) in innerLayer
        self.cov_w : np.ndarray = r_cov_w
        self.cov_v : np.ndarray = r_cov_v
        self.range : float = r_range
        self.FOV : float = r_FOV
        #graphics
        self.ax = ax
        self.graphic_plan = [] #placeholder
        self.moviewriter = []
    def outerLayer(self, k : int ,backend : solver ,u : np.ndarray ,goal : np.ndarray): #plan
        self.k = k
        J_prev = 1e10 #absurdly big number as initial value

        # set weight matrices
        ests_bar, covs_bar  = self.innerLayerBAR(backend.copyObject(),u)
        cov_kpL_bar = covs_bar[-1]
        cov_k = backend.isam2.marginalCovariance(X(self.k))
        alpha_k = min(max(trace(cov_kpL_bar),trace(cov_k))/self.beta, 1)
        #turning may reduce covariance trace due to seeing new landmarks. Sometimes in unexpected ways
        #thus, we keep alpha_k high as long as loop closure has not happend as follows:
        if self.alpha_km1 > self.alpha_LB and alpha_k > self.alpha_LB: #self.alpha_km1 > alpha_k instead of alpha_km1 == 1
            alpha_k = max(self.alpha_km1,alpha_k)
        print(f"-------------------------------------->calculated alpha_k = {alpha_k}")
        
        alpha_k = float(alpha_k > 0.5) #use if/else instead of logisticCurve
        if alpha_k == 0: #and self.alpha_km1 != 0: #if loop colsure occured, u[0] should point more towards goal
            pose = backend.isam2.calculateEstimatePose2(X(k))
            u[0] = pose.bearing(goal).theta()/2
        if alpha_k > self.alpha_LB and self.alpha_km1 <= self.alpha_LB:
            pose = backend.isam2.calculateEstimatePose2(X(k))
            u[0] = pose.bearing(np.array([0,0])).theta()/4

        M_x = 1-alpha_k
        M_sigma = alpha_k

        i = 0 #iterations for gradient decent
        while True:
            #update u
            dJ = self.computeGradient(backend.copyObject(), u, M_x, M_sigma, goal)
            u = u - self.lambDa * dJ

            #check convergence
            plannedBackend = backend.copyObject()
            J,_ ,_ = self.evaluateObjective(plannedBackend, u, M_x, M_sigma, goal)
            print(f"J = {J};    norm(dJ) = {norm(dJ)};    (J-J_prev)/J_prev = {norm((J-J_prev)/(J_prev + 1e-10))}")
            
            if norm(dJ) < self.epsConvGrad:
                print('small dJ')
                return u, J, plannedBackend
            if norm((J-J_prev)/(J_prev + 1e-10)) < self.epsConvVal:
                print('small change in J')
                return u, J, plannedBackend
            if i > self.i_max:
                print('max iterations for gradient decent')
                return u, J, plannedBackend
            
            i += 1
            self.alpha_km1 = alpha_k
            J_prev = J

            if self.moviewriter:
                self.plotPlan(u, plannedBackend)
                plt.pause(0.00001)
                self.moviewriter.grab_frame()


    def computeGradient(self, backend : solver, u : np.ndarray, M_x : float, M_sigma: float, goal : np.ndarray) -> np.ndarray:
        #M_u and L provided from self
        dJ = np.zeros(u.size)
        for i in range(u.size):
            du = np.zeros_like(u)
            du[i] = self.epsGrad
            Jpi, backend_left, partialCost_left = self.evaluateObjective(backend.copyObject(),u + du ,M_x, M_sigma, goal) #turn left
            Jmi, backend_right, partialCost_right = self.evaluateObjective(backend.copyObject(),u - du ,M_x, M_sigma, goal) #turn right
            dJ[i] = (Jpi-Jmi)/(2 * self.epsGrad)
        return dJ

    def evaluateObjective(self, backend : solver, u : np.ndarray, M_x : float, M_sigma: float, goal : np.ndarray) -> float:
        
        #term c: distance from goal
        ests_bar,covs_bar = self.innerLayerBAR(backend.copyObject(),u) #copy backend as we need it laterz
        #use this formulation as we have no control on "gas" only on "wheel"
        dist = norm(np.array([est.translation() for est in ests_bar]) - goal, axis = 1)
        c = min(dist/self.range)**2
        c *= M_x
        
        #term a: control effort
        a = 0
        for u_kpl in u:
            a += zeta(u_kpl)**2 # mahalanobisISqrd doesnt work for scalars.. so...
        a *= self.M_u

        #term b: poses covariance
        b = 0
        for l, u_kpl in enumerate(u):
            est_kpl, cov_kpl, lms_kpl = self.innerLayer(backend,np.array([u_kpl]))
            b += trace(cov_kpl)/self.beta
        b *= M_sigma

        #term d: loop closurer. kpl -> kpL after all iterations in term b
        ds = []
        n = len(backend.seen_landmarks['id'])
        last_lm_id = backend.seen_landmarks['id'][-1]
        if n > 0:
            tr_cov_L0 = trace(backend.isam2.marginalCovariance(L(0)))
            tr_cov_Ln = trace(backend.isam2.marginalCovariance(L(last_lm_id)))
            tr = (tr_cov_L0+tr_cov_Ln)/2
            for lm_index in backend.seen_landmarks['id']:#[lm.id for lm in lms_kpl]:
                lm_mu = backend.isam2.calculateEstimatePoint2(L(lm_index))
                tr_lm_cov = trace(backend.isam2.marginalCovariance(L(lm_index)))
                lm_r = est_kpl.range(lm_mu)
                if tr_lm_cov > tr:
                    ds.append((tr_lm_cov/tr) / (lm_r/self.range)) #get away from large covs
                else:
                    ds.append((tr_lm_cov/tr) * (lm_r/self.range)) #get closer to small covs

            d = np.sum(np.array(ds))
            d /= (5) ##normalize
            d **= 2
            d *= M_sigma

        J = a + b + c + d
        return J, backend, np.array([a,b,c,d])

    def innerLayerBAR(self, backend : solver ,u : np.ndarray):
        covs = []
        ests = []

        for u_kpl in u:
            backend.i += 1
            backend.addOdomMeasurement(meas_odom(gtsam.Pose2(self.dx,0,u_kpl),self.cov_w))
            backend.update()

        if u.size == 1:
            ests = ests[0]
            covs = covs[0]

        covs.append(backend.isam2.marginalCovariance(X(backend.i))) #i == k
        ests.append(backend.isam2.calculateEstimatePose2(X(backend.i)))
        return ests,covs

    def innerLayer(self, backend : solver, u : np.ndarray):
        covs = []
        ests = []
        
        for u_kpl in u:
            backend.i += 1

            backend.addOdomMeasurement(meas_odom(gtsam.Pose2(self.dx,0,u_kpl),self.cov_w))
            backend.update(0) #need to update to pull X_kpl for landmark measurements

            X_kpl = backend.isam2.calculateEstimatePose2(X(backend.i))
            lms =  self.simulateMeasuringLandmarks(backend, X_kpl) #need to write this down
            for lm in lms:
                backend.addlandmarkMeasurement(lm)
            backend.update(0)

            covs.append(backend.isam2.marginalCovariance(X(backend.i))) #i == k
            ests.append(backend.isam2.calculateEstimatePose2(X(backend.i)))
        
        if u.size == 1:
            ests = ests[0]
            covs = covs[0]

        return ests, covs, lms

    def simulateMeasuringLandmarks(self, backend : solver, pose : gtsam.Pose2):
            meas = []
            for lm_index, lm_label in zip(backend.seen_landmarks["id"], backend.seen_landmarks["classLabel"]):
                lmML = backend.isam2.calculateEstimatePoint2(L(lm_index))    
                angle = pose.bearing(lmML).theta()
                r = pose.range(lmML)

                if abs(angle) < self.FOV/2 and r < self.range*5: #if simu-viewed, compute noisy measurement
                    cov_v_bar = self.cov_v * max((r/self.range)**2,1)
                    meas.append(meas_landmark(lm_index, r, angle, cov_v_bar, lm_label))
            return meas

        #create list of landmarks

    def plotPlan(self,u, backend, ax : plt.Axes = None):
        if ax is None:
            try: 
                ax = self.ax
            except:
                print('no axes provided to object')

        if  not self.graphic_plan:
            self.graphic_plan, = ax.plot([], [],'go-',markersize = 2)
        
        plan = np.zeros((1+u.size,2))
        belief = backend.isam2.calculateEstimatePose2(X(self.k))
        plan[0,:] = np.array(belief.translation())
        for i,ui in enumerate(u):
            belief = belief.compose(gtsam.Pose2((self.dx,0,ui)))
            plan[1+i,:] = np.array(belief.translation())
        self.graphic_plan.set_data(plan[:,0],plan[:,1])

def mahalanobisIsqrd(a : np.ndarray ,S : np.ndarray):
    return a.T @ S @ a

def zeta(u : float) -> float:
    #bottom of page 18 - some known function that quantifies the usage of control u
    return u # penalizes changes in direciton, page 45, will be squared later in cost

def stupidController(k, backend, goal):
    belief = backend.calculateEstimate().atPose2(X(k))
    err = belief.bearing(goal).theta()
    u = np.sign(err)  * min(abs(err), np.pi/4)
    return u

def logisticCurve(x, x0, k = 0.3, L = 1):
    # https://en.wikipedia.org/wiki/Logistic_function
    y = L/(1 + np.exp(-k*(x-x0)))
    return y