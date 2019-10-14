import sys
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def p2e(u):
    pd = np.size(u,0)
    if pd==3:
        u_ = u[:2,:]/u[2,:]
        return u_
    elif pd==2:
        return u
    else:
        print('wrong point format!')
        sys.quit(1)
def normalize_u(u):
    u = p2e(u)
    mean_u = np.mean(u,axis=1)
    u = u - mean_u.reshape(-1,1)
    distu = np.sqrt(np.sum(np.power(u,2),axis=0))
    r = np.mean(distu)/np.sqrt(2)
    A = np.diag(np.array([1/r,1/r,1]))
    A[:2,-1] = -mean_u/r
    return A
def find_F(u,norm_op=True):
    # u - [6,n] point pairs from two cameras
    pnum = np.size(u,1)
    Z = np.zeros((pnum,9))
    if norm_op:
        A1 = normalize_u(u[:3,:])
        A2 = normalize_u(u[3:6,:])
        u1 = np.matmul(A1,u[:3,:])
        u2 = np.matmul(A2,u[3:6,:])
    for i in range(pnum):
        Z[i,:] = np.matmul(u1[:,i].reshape(-1,1),u2[:,i].reshape(1,-1)).reshape(1,9,order='F')
    M = np.matmul(Z.T,Z)
    ev,vec = np.linalg.eigh(M)
    import pdb; pdb.set_trace()
    F = vec[:,np.argsort(ev)[0]].reshape(3,3,order='F')
    uu,ss,vv = np.linalg.svd(F)
    min_i = np.argmin(np.abs(ss))
    ss[min_i] = 0
    F = np.matmul(uu,np.matmul(np.diag(ss),vv.T))

    if norm_op:
        F = np.matmul(A1.T,np.matmul(F,A2))
    F = F / np.linalg.norm(F)
    return F
if __name__=='__main__':
    u = np.array([10,14,6,4,14,12,67,5,3,3,5,1,56,23,35,177,46,43,13,566,13,1,5,1])
    u = u.reshape(6,-1)
    f = find_F(u)
    print(f)
