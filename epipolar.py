#!python
import numpy as np
from numpy.linalg import svd

def epipolar(F):
    """Compute an epipole from fundamental matrix.

    The algorithm used by this function is based on the singular value
        decomposition of `F`.

    """
    u, s, v = svd(F)
    v = v.T
    length = v.shape[1]
    e = v[:,length-1]
    e = np.ravel(e)
    return e
