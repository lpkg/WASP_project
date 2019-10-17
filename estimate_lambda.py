#!python
import numpy as np
from numpy.linalg import svd
from epipolar import epipolar
from numpy import linalg as LA

def estimate_lambda(Ws, list_F, list_i):
    """Estimate some initial projective depth given the measurement matrix Ws
       and Fundamental matrices.
    """

    n = int(Ws.shape[0]//3)
    m = Ws.shape[1]
    Lambda = np.ones((n, m))
    for i in range(n-1):
        j = i+1
        F = list_F[i]
        e = epipolar(F)
        for p in list_i[i]:
            p = p - 1
            q_ip = Ws[3*i:3*i+3, p]
            q_jp = Ws[3*j:3*j+3, p]
            np.cross(e, q_ip)
            tmp1 = LA.norm(np.cross(e, q_ip), 2)**2
            tmp2 = np.dot(np.matmul(F.T, q_jp), np.cross(e, q_ip))
            Lambda[j, p] = Lambda[i, p]/tmp2.sum()*tmp1
    return Lambda

