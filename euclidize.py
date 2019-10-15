#!python
import numpy as np
from numpy.linalg import svd
from numpy import linalg as LA
import pandas as pd
import math

def euclidize(Ws, Lambda, P, X):
    """Estimate some initial projective depth given the measurement matrix Ws
       and Fundamental matrices.
    """

    n = Ws.shape[0]/3
    m = Ws.shape[1]

    #Compute B
    a = np.array([])
    b = np.array([])
    c = np.array([])
    for i in range(n):
        tmp = np.dot(Ws[3*i, :], Lambda[i, :])
        a = np.append(a, np.array([tmp.sum()]), axis = 0)
        tmp = np.dot(Ws[3*i+1, :], Lambda[i, :])
        b = np.append(b, np.array([tmp.sum()]), axis = 0)
        c = np.append(c, np.array([Lambda[i, :].sum()]), axis = 0)

    TempA = -P[2:3*n:3, :]
    TempB = -P[2:3*n:3, :]


    for i in range(n):
        TempA[i, :] = TempA[i, :]*a[i]/c[i]
        TempB[i, :] = TempB[i, :]*b[i]/c[i]

    TempA = TempA + P[0:3*n:3, :]
    TempB = TempB + P[1:3*n:3, :]
    Temp = np.append(TempA, TempB, axis = 0)

    u, s, v = svd(Temp)
    V = v.T
    B = V[:, 3]
    print B

    #Compute A
    Temp = np.array([])
    for i in range(n):
        P1 = P[3*i, :]
        P2 = P[3*i+1, :]
        P3 = P[3*i+2, :]
        u = P1
        v = P2
        tmp = np.array([u[0]*v[0], u[0]*v[1]+u[1]*v[0], u[2]*v[0]+v[2]*u[0], u[0]*v[3]+u[3]*v[0], u[1]*v[1], u[2]*v[1]+v[2]*u[1], u[1]*v[3]+v[1]*u[3], u[2]*v[2], u[3]*v[2]+u[2]*v[3], u[3]*v[3]])
        Temp = np.append(Temp, tmp, axis = 0);
        u = P1
        v = P3
        tmp = np.array([u[0]*v[0], u[0]*v[1]+u[1]*v[0], u[2]*v[0]+v[2]*u[0], u[0]*v[3]+u[3]*v[0], u[1]*v[1], u[2]*v[1]+v[2]*u[1], u[1]*v[3]+v[1]*u[3], u[2]*v[2], u[3]*v[2]+u[2]*v[3], u[3]*v[3]])
        Temp = np.append(Temp, tmp, axis = 0);
        u = P2
        v = P3
        tmp = np.array([u[0]*v[0], u[0]*v[1]+u[1]*v[0], u[2]*v[0]+v[2]*u[0], u[0]*v[3]+u[3]*v[0], u[1]*v[1], u[2]*v[1]+v[2]*u[1], u[1]*v[3]+v[1]*u[3], u[2]*v[2], u[3]*v[2]+u[2]*v[3], u[3]*v[3]])
        Temp = np.append(Temp, tmp, axis = 0);
    Temp = Temp.reshape((3*n, 10))


    if n<4:
        u = P[2, :]
        tmp = np.array([u[0]**2, 2*u[0]*u[1], 2*u[0]*u[2], 2*u[0]*u[3], u[1]**2, 2*u[1]*u[2], 2*u[1]*u[3], u[2]**2, 2*u[2]*u[3], u[3]**2])
        tmp = tmp.reshape((1, 10))
        Temp = np.append(Temp, tmp, axis = 0);
        length = Temp.shape[0]
        b = np.zeros((length, 1))
        b[length-1, 0] = 1
        svd_tmp = np.append(Temp, b, axis = 1)
        np.savetxt("foo.csv", svd_tmp, delimiter=",")
        u, s, v = svd(svd_tmp)
        V = v.T
        np.savetxt("V.csv", V, delimiter=",")
        len = V.shape[1]-1
        q = -1/V[10, len]*V[0:10, len]
    else:
        u, s, v = svd(np.append(Temp, b))
        V = v.T
        q = -V[:, V.shape[1]-1]
    print q
    Q = np.matrix([[q[0], q[1], q[2], q[3]], [q[2], q[4], q[5], q[6]], [q[2], q[5], q[7], q[8]], [q[3], q[6], q[8], q[9]]])
    M = np.matmul(P[0:3, :], Q)
    M = np.matmul(M, P[0:3, :].T)
    if M[0,0]<=0:
        q = -q
        Q = np.matrix([[q[0], q[1], q[2], q[3]], [q[2], q[4], q[5], q[6]], [q[2], q[5], q[7], q[8]], [q[3], q[6], q[8], q[9]]])
    u, s, v = svd(Q)
    S = np.zeros((3, 3))
    for i in range(3):
        S[i, i] = math.sqrt(s[i])


    A = np.matmul(u[:, 0:3], S)
    B = np.reshape(B, (4, 1))
    H = np.append(A, B, axis = 1)



    return H
