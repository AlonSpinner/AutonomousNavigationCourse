from dataclasses import dataclass
import numpy as np
import gtsam

@dataclass(frozen = True, order = True)
class landmark:
    id : int
    xy : np.ndarray((2))
    cov : np.ndarray((2,2))
    classLabel : str


@dataclass(frozen = True)
class meas_odom:
    dpose : gtsam.Pose2
    cov : np.ndarray((2,2))

@dataclass(frozen = True, order = True)
class meas_landmark:
        id : int
        range : float
        angle : float
        cov : np.ndarray((2,2))
        classLabel : str

def pose2ToNumpy(pose2: gtsam.Pose2):
    return np.array([pose2.x(),pose2.y(),pose2.theta()])
