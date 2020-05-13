import numpy as np
import matplotlib.pyplot as plt

global x_s
global y_s
global l_s

def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    pointX = data[:,0]
    pointY = data[:,1]
    label = data[:,2]

    return pointX, pointY, label

x_s, y_s, l_s = read_data("data-nonlinear.txt")

###### result 01 ######
plt.title('01 training data')
x_0 = x_s[l_s==0]
x_1 = x_s[l_s==1]
y_0 = y_s[l_s==0]
y_1 = y_s[l_s==1]
plt.scatter(x_0, y_0, c='b')
plt.scatter(x_1, y_1, c='r')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()