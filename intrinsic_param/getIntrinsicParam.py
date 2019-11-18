#!python
import numpy as np
from estimateF import find_F
from numpy.linalg import svd

def getTmpMatrix(F1, E1):
    Phi = np.array([[-F1[0, 0], -F1[1, 0], E1[2,0]], [-F1[0, 1], -F1[1, 1], E1[2,1]]])
    y = np.array([[F1[2,0]], [F1[2,1]]])
    return Phi, y
def getS(T):
    return np.array([[0, -T[2,0], T[1,0]], [T[2,0], 0, -T[0,0]], [-T[1,0], T[0,0], 0]])
def findConst(F, E):
    """Find a constant that scales F to E.
    """
    for k in range(2):
        for l in range(2):
            if E[k, l]!=0:
                return F[k, l]/E[k, l]

def getTScaled(F, R):
    """Compute a scaled translation
    """
    Phi = np.array([[-R[2, 0], 0, R[0,0]], [0, R[2,0], -R[1,0]], [0, R[2,1], -R[1,1]], [-R[2,1], 0, R[0,1]]])
    y = np.array([[F[1,0]], [F[0,0]], [F[0,1]], [F[1,1]]])
    return np.dot(np.linalg.pinv(Phi), y)

def getIntrinsicParam(Ws, R_12, R_13):
    """Compute intrinsic parameters

    """
    R_23 = np.matmul(R_13, np.linalg.pinv(R_12))
    F_12 = find_F(Ws[0:6,:])
    F_23 = find_F(Ws[3:9,:])
    tmp = np.append(Ws[:3,:], Ws[6:9,:], axis = 0)
    F_13 = find_F(tmp)

    T_12 = getTScaled(F_12.T, R_12)
    T_23 = getTScaled(F_23.T, R_23)
    T_13 = getTScaled(F_13.T, R_13)

    S_12 = getS(T_12)
    S_23 = getS(T_23)
    S_13 = getS(T_13)

    E_12 = np.matmul(S_12, R_12).T
    E_23 = np.matmul(S_23, R_23).T
    E_13 = np.matmul(S_13, R_13).T


    const_1 = findConst(F_12, E_12)
    E_12 = E_12*const_1
    const_2 = findConst(F_13, E_13)
    E_13 = E_13*const_2
    const_3 = findConst(F_23, E_23)
    E_23 = E_23*const_3


    # Camera 1 parameters
    Phi_1, y_1 = getTmpMatrix(F_12, E_12)
    Phi_2, y_2 = getTmpMatrix(F_13, E_13)
    Phi = np.append(Phi_1, Phi_2, axis = 0)
    y = np.append(y_1, y_2, axis = 0)
    param_1 = np.dot(np.linalg.pinv(Phi), y)

    # Camera 2 parameters
    Phi_1, y_1 = getTmpMatrix(F_23, E_23)
    Phi_2, y_2 = getTmpMatrix(F_12.T, E_12.T)
    Phi = np.append(Phi_1, Phi_2, axis = 0)
    y = np.append(y_1, y_2, axis = 0)
    param_2 = np.dot(np.linalg.pinv(Phi), y)

    # Camera 3 parameters
    Phi_1, y_1 = getTmpMatrix(F_23.T, E_23.T)
    Phi_2, y_2 = getTmpMatrix(F_13.T, E_13.T)
    Phi = np.append(Phi_1, Phi_2, axis = 0)
    y = np.append(y_1, y_2, axis = 0)
    param_3 = np.dot(np.linalg.pinv(Phi), y)

    return param_1, param_2, param_3
