import pandas as pd
import numpy as np
from rank_nullspace import rank, nullspace
df = pd.read_csv('data.csv')
world = np.array(df[::2], dtype=np.float)
projection = np.array(df[1::2], dtype=np.float)
# print np.array(world[0]).T

print world.shape
for x in range(0, world.shape[0]):
	if x == 0:
		Tmp = np.stack((world[0],  np.zeros(shape = (1, 4)), -projection[0, 0]*world[0]), axis = 1)
		print Tmp
		Tmp = np.array([Tmp])
		tmp = np.append(np.zeros(shape = (1, 4)), tmp0)
		tmp = np.append(tmp, -projection[0, 1]*world[0])
		tmp = np.array([tmp])
		Tmp = np.append(Tmp, tmp, axis = 0)
	else:
		tmp0 = np.append(world[x], 1)
		tmp = np.append(tmp0, np.zeros(shape = (1, 4)))
		tmp = np.array([np.append(tmp, -projection[x, 0]*world[x])])
		Tmp = np.append(Tmp, tmp, axis = 0)
		tmp = np.append(np.zeros(shape = (1, 4)), tmp0)
		tmp = np.array([np.append(tmp, -projection[x, 1]*world[x])])
		Tmp = np.append(Tmp, tmp, axis = 0)

# what I should get?
f = np.array(
[[ 6.71472000e+02,  8.95296000e+02, -6.92160000e+02,  6.84360103e+03],
 [-7.95840000e+02,  9.80880000e+02,  2.30400000e+02 , 1.10172666e+04],
 [ 4.80000000e-01,  6.40000000e-01 , 6.00000000e-01,  1.77418216e+01]])
f = np.reshape(f, (12, 1))
f = f[:11]/f[11]
print f
# b component
projection_new = projection[:,[0, 1]]
projection_new = np.reshape(projection_new, (2*world.shape[0],1))

f = np.dot(np.linalg.pinv(Tmp), projection_new)
print f
