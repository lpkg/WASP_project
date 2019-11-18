import sys
sys.path.append('intrinsic_param')
import pandas as pd
import numpy as np
from estimateF import find_F
from data_generation import total_rotation, degree_to_radiant
from getIntrinsicParam import getIntrinsicParam

df = pd.read_csv('data/camera_1')
camera_1 = np.array(df, dtype=np.float)
df = pd.read_csv('data/camera_2')
camera_2 = np.array(df, dtype=np.float)
df = pd.read_csv('data/camera_3')
camera_3 = np.array(df, dtype=np.float)
Ws = np.append(camera_1.T, camera_2.T, axis = 0)
Ws = np.append(Ws, camera_3.T, axis = 0)
angle_x_2 = degree_to_radiant(-45)
angle_y_2 = degree_to_radiant(0)
angle_z_2 = degree_to_radiant(0)

angle_x_3 = degree_to_radiant(45)
angle_y_3 = degree_to_radiant(0)
angle_z_3 = degree_to_radiant(0)

R_12 = total_rotation(angle_x_2,angle_y_2,angle_z_2)

R_13 = total_rotation(angle_x_3,angle_y_3,angle_z_3)

par1, par2, par3 = getIntrinsicParam(Ws, R_12, R_13)

print(par1)
print(par2)
print(par3)
