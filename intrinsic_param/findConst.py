#!python
import numpy as np
from numpy.linalg import svd

def findConst(F, E):
    """Find a constant that scales F to E.
    """
    for k in range(2):
        for l in range(2):
            if E[k, l]!=0:
                return F[k, l]/E[k, l]
