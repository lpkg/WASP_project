import sys
import os
import numpy as np
from estimateF import *
from estimate_lambda import *
from euclidize import *
#import nimfa
np.set_printoptions(threshold=sys.maxsize)

########################################
M = 2
N = 8
########################################
'''
for i in range(M):
    fn = 'camera_'+str(i+1)
    if i==0:
        data=np.loadtxt(fn,delimiter=',').T
    else:
        data = np.vstack((data,np.loadtxt(fn,delimiter=',').T))
'''
data = np.array([[4.0783, 4.8500, 4.7183 , 4.2684, 6.8735,3.1827, 4.8771, 5.8323],[5.7993,    4.1347 ,   8.2692  ,  3.3569 ,   8.7757 ,   3.8092 ,   5.6149    ,5.4351],[1,1,1,1,1,1,1,1],[7.6909 ,  15.0755  ,  7.7271 ,  10.2418 ,   9.5575,    7.2920 ,   8.2885 ,  21.3312],[6.9025,   10.6659 ,   7.5748 ,   6.8240 ,   9.7366  ,  5.8470  ,  7.3433 ,  18.4971],[1,1,1,1,1,1,1,1]])
list_F = list()
list_i = list()
for i in range(M-1):
    for j in range(i+1,M):
        data_pair = np.vstack((data[3*i:3*(i+1),:],data[3*j:3*(j+1),:]))
        list_F.append(find_F(data_pair))
        list_i.append(np.arange(N))
import pdb;pdb.set_trace()
lambdas = estimate_lambda(data,list_F,list_i)
lsnmf
