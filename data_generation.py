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

angle_x_1 = degree_to_radiant(30)
angle_y_1 = degree_to_radiant(0)
angle_z_1 = degree_to_radiant(45)

angle_x_2 = degree_to_radiant(0)
angle_y_2 = degree_to_radiant(72)
angle_z_2 = degree_to_radiant(10)

#Focal lengths and principal components of camera 1 and camera 2

f_1 = 5
x_0_1 = 100
y_0_1 = 150

f_2 = 3
x_0_2 = 50
y_0_2 = 130

t_1 = [1, 0, 1] # in cm
t_2 = [1, 1, 0] # in cm

R_1 = total_rotation(angle_x_1,angle_y_1,angle_z_1)

R_2 = total_rotation(angle_x_2,angle_y_2,angle_z_2)



K_1 = [[f_1,0,x_0_1],[0, f_1, y_0_1],[0,0,1]]

K_2 = [[f_2,0,x_0_2],[0, f_2, y_0_2],[0,0,1]]



P_1 = np.matmul(K_1,np.column_stack((R_1,t_1)))

P_2 = np.matmul(K_2,np.column_stack((R_2,t_2)))

# 3-D world point

number_of_points = 10

camera_1_points=[]

camera_2_points=[]

X = np.random.randint(0,10,3*number_of_points).reshape(number_of_points,3,1)

for x in X:
  
  x = np.append(x,[1])
  camera_1_points.append(generate_a_point_from_camera(x,P_1))
  camera_2_points.append(generate_a_point_from_camera(x,P_2))
  

