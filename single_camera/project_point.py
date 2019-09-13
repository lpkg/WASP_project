import camera
import numpy as np
import math
c = camera.Camera()



R = np.array(
    [[0.36, 0.48, -0.8],
     [-0.8, 0.6, 0.],
     [0.48, 0.64, 0.6]])
t = np.array([[-1.365061486465], [3.431608806127], [17.74182159488]])
c.set_R(R)
c.set_t(t)

K = np.array(
    [[1225.2, 0, 480.0],
     [0.0, 1225.2, 384.0],
     [0.0, 0.0, 1.0]])

c.set_K(K)

print c.world_to_image(np.array([[4, 3, 0]]).T)
tmp = np.append(R, t, axis = 1)
print np.dot(K, tmp)
