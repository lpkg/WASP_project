import numpy as np

# Return a matrix with a rotation along the z-axis
def rotate_z(theta):
    a= [[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0,0,1]]
    return a
# Return a matrix with a rotation along the x-axis  
def rotate_x(theta):
    a= [[1,0, 0],[0, np.cos(theta), -np.sin(theta)],[0,np.sin(theta), np.cos(theta)]]
    return a
# Return a matrix with a rotation along the y-axis 
def rotate_y(theta):
    a= [[np.cos(theta), 0, np.sin(theta)],[0,1, 0],[-np.sin(theta),0, np.cos(theta)]]
    return a

# Return a matrix with a rotation along the three axis  
def total_rotation(theta_x,theta_y,theta_z):
  return np.matmul(np.matmul(rotate_z(theta_z),rotate_y(theta_y)),rotate_x(theta_x))
  
# Converts the angles in degree to radiants 
def degree_to_radiant(theta):
  return theta*np.pi/180

# Return the point X mapped in the image plane of a camera characterized by matrix P
def generate_a_point_from_camera(X,P):
  temp = np.matmul(P,X)
  return temp/temp[-1]

angle_x_1 = degree_to_radiant(0)
angle_y_1 = degree_to_radiant(0)
angle_z_1 = degree_to_radiant(0)

angle_x_2 = degree_to_radiant(-45)
angle_y_2 = degree_to_radiant(0)
angle_z_2 = degree_to_radiant(0)

angle_x_3 = degree_to_radiant(45)
angle_y_3 = degree_to_radiant(0)
angle_z_3 = degree_to_radiant(0)

#Focal lengths and principal components of camera 1 and camera 2

f_1 = 5
x_0_1 = 100
y_0_1 = 150

f_2 = 3
x_0_2 = 50
y_0_2 = 130

f_3 = 10
x_0_3 = 50
y_0_3 = 100


t_1 = [0, 0, 0] 
t_2 = [0, np.sqrt(2), np.sqrt(2)] 
t_3 = [1, -np.sqrt(2), np.sqrt(2)] 


R_1 = total_rotation(angle_x_1,angle_y_1,angle_z_1)

R_2 = total_rotation(angle_x_2,angle_y_2,angle_z_2)

R_3 = total_rotation(angle_x_3,angle_y_3,angle_z_3)

K_1 = [[f_1,0,x_0_1],[0, f_1, y_0_1],[0,0,1]]

K_2 = [[f_2,0,x_0_2],[0, f_2, y_0_2],[0,0,1]]

K_3 = [[f_3,0,x_0_3],[0, f_3, y_0_3],[0,0,1]]


P_1 = np.matmul(K_1,np.column_stack((R_1,t_1)))

P_2 = np.matmul(K_2,np.column_stack((R_2,t_2)))

P_3 = np.matmul(K_3,np.column_stack((R_3,t_3)))

# 3-D world point

number_of_points = 10

camera_1_points=[]

camera_2_points=[]

camera_3_points=[]

X = (np.random.rand(2*number_of_points).reshape(number_of_points,2,1))*np.sqrt(2)*2.0 - np.sqrt(2)

for x in X:
  
  x = np.append(x,np.random.rand(1)*10 + np.sqrt(2))# maximum point on z-axis is 10 + sqrt(2)
  x = np.append(x,[1])
  camera_1_points.append(generate_a_point_from_camera(x,P_1))
  camera_2_points.append(generate_a_point_from_camera(x,P_2))
  camera_3_points.append(generate_a_point_from_camera(x,P_3))
	  
np.savetxt('camera_1',camera_1_points,delimiter=',')
np.savetxt('camera_2',camera_2_points,delimiter=',')
np.savetxt('camera_3',camera_3_points,delimiter=',')


