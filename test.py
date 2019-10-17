import numpy as np
from estimate_lambda import estimate_lambda
from euclidize import euclidize

W = np.array([[0.5377,  -0.4336,    0.7254], [1.8339,    0.3426,   -0.0631], [ -2.2588,    3.5784,   0.7147], [0.8622,    2.7694,   -0.2050], [0.3188,  -1.3499,   -0.1241], [-1.3077,    3.0349,    1.4897]])
F2 = np.array([[0.0774,   -0.0068,    0.3714], [-1.2141,    1.5326,   -0.2256], [-1.1135,  -0.7697,    1.1174]])
F1 = np.array([[0.3192,   -0.0301,   1.0933], [0.3129,  -0.1649,    1.1093], [-0.8649,    0.6277,   -0.8637]])
q = np.array([[0.1, 0.2, 0.1, 0.2, 0.1, 0.3]])
q = q.T
P = np.append(0.5*W, q, axis = 1)
X = P[0:4, 0:3] + 0.25*W[0:4, 0:3]-0.3
pairs = [F1, F2]
ind = [np.array([1]), np.array([2])]
const = 500
lam = estimate_lambda(const*W, pairs, ind)
#print(lam)
p, r, c = euclidize(const*W, lam, P, X)
print(p)

