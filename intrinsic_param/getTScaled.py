#!python
import numpy as np
from numpy.linalg import svd

def getTScaled(F, R):
    """Compute a scaled translation

    """

    Phi = np.array([[-R[2, 0], 0, R[0,0]], [0, R[2,0], -R[1,0]], [0, R[2,1], -R[1,1]], [-R[2,1], 0, R[0,1]]])
    y = np.array([[F[1,0]], [F[0,0]], [F[0,1]], [F[1,1]]])
    return np.dot(np.linalg.pinv(Phi), y)
