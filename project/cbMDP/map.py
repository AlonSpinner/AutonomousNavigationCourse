import numpy as np
from .utils import plotting
from .utils.datatypes import landmark
import matplotlib.pyplot as plt

class map:

    #class attributes
    markersList = ["o","v","s","D","4","*",">","<","2","P"] #10 classes max
    colors = np.array([[0.244972, 0.287675, 0.53726 , 1.      ],
       [0.709898, 0.868751, 0.169257, 1.      ],
       [0.147607, 0.511733, 0.557049, 1.      ],
       [0.993248, 0.906157, 0.143936, 1.      ],
       [0.281412, 0.155834, 0.469201, 1.      ],
       [0.20803 , 0.718701, 0.472873, 1.      ],
       [0.430983, 0.808473, 0.346476, 1.      ],
       [0.190631, 0.407061, 0.556089, 1.      ],
       [0.267004, 0.004874, 0.329415, 1.      ],
       [0.119699, 0.61849 , 0.536347, 1.      ]])
    
    def __init__(self,landmarks = []):
        #instance attributes
        self.landmarks : list[landmark] = []
        self.classLabels : list[str] = [] #will be filled laters
        self.addLandmarks(landmarks)

    def fillMapRandomly(self, N, classLabels, xrange, yrange, sigmarange = None):
        # N - amount of landmarks
        # classLabels - list of strings
        # xrange, yrange - tuples
        landmarks = [None] * N
        M = len(self.landmarks)

        for ii in range(N):
            xy = np.array([np.random.uniform(xrange[0],xrange[1]),
                           np.random.uniform(yrange[0],yrange[1])])  

            if sigmarange is not None:
                rootcov = np.random.uniform(low=sigmarange[0], high=sigmarange[1], size=(2,2))
                cov = rootcov @ rootcov.T #enforce symmetric and positive definite: https://mathworld.wolfram.com/PositiveDefiniteMatrix.html
            else:
                cov = None

            landmarks[ii] = landmark(M + ii,
                                     xy, 
                                     classLabel = np.random.choice(classLabels),
                                     cov = cov)
        self.addLandmarks(landmarks)

    def addLandmarks(self,landmarks):
        self.landmarks.extend(landmarks)
        self.reIndexifyLandmarks()
        self.defineClassLabelsFromLandmarks()

    def reIndexifyLandmarks(self):
        for i, lm in enumerate(self.landmarks):
            self.landmarks[i] = landmark(i, lm.xy, lm.cov, lm.classLabel)

    #goes over all landmarks to find classLabels.
    def defineClassLabelsFromLandmarks(self):
        if self.landmarks: #not empty
            classLabels = [lm.classLabel for lm in self.landmarks]
            classLabels = list(set(classLabels))
            if len(classLabels) > 10:
                raise Exception("no more than 10 classes are premited. Not enough distinguishable markers in matplotlib")
            else:
                self.classLabels = classLabels

    def exportSemantics(self):
        semantics = {
                    "classLabel": self.classLabels,
                    "color": self.colors[:len(self.classLabels)],
                    "marker": self.markersList[:len(self.classLabels)]
                    }
        return semantics

    def plot(self,ax : plt.Axes = None, plotIndex = False, plotMeshIndex = False, plotCov = False, plotLegend = False ,markerSize = 10):
        if ax == None:
            fig , ax = plotting.spawnWorld()

        for lm in self.landmarks:
            ii = self.classLabels.index(lm.classLabel) #index of classLabel. used for shape and color
            
            cov = None if plotCov is False else lm.cov
            index = None if plotIndex is False else lm.id
            plotting.plot_landmark(ax, loc = lm.xy, cov = cov, 
                                index = index,
                                markerColor = self.colors[ii].reshape(1,-1), 
                                markerShape = self.markersList[ii], 
                                markerSize = markerSize,
                                textColor = 'k')

        if plotLegend:
            legendHandles = []
            for ii in range(len(self.classLabels)):
                legendHandles.append(ax.scatter(np.nan,np.nan,
                                            marker = self.markersList[ii],
                                            c = self.colors[ii].reshape(1,-1)))
            fig.legend(handles = legendHandles, labels = self.classLabels)

        return ax


    

    
