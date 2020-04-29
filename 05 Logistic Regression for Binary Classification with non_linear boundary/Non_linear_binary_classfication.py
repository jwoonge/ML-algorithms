import numpy as np
import matplotlib.pyplot as plt
import math

global x_d 
x_d = [0,0,1,0,2,0,1]
global y_d 
y_d = [0,0,0,1,0,2,1]

def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    pointX = data[:,0]
    pointY = data[:,1]
    label = data[:,2]

    return pointX, pointY, label

def sigmoid(z):
    return 1/(1+np.exp(np.float64(-z)+math.e**(-64)))

def func(thetas, x, y):
    ret=thetas[0]
    for i in range(1,len(thetas)):
        ret += thetas[i]*x**x_d[i]*y**y_d[i]
    return ret

def gradient_descent(thetas, x_s, y_s, labels, learning_rate):
    thetas_new = []
    m = len(labels)
    for i in range(len(thetas)):
        update = 0
        for j in range(m): # 이거 ij 순서 바꾸면?
            mult = x_s[j]**x_d[i] * y_s[j]**y_d[i]
            update += (sigmoid(func(thetas,x_s[j],y_s[j]))-labels[j])*mult/m
        thetas_new.append(thetas[i]-learning_rate*update)
    return thetas_new

x_s, y_s, l_s = read_data('data-nonlinear.txt')
x_0 = x_s[l_s==0]
x_1 = x_s[l_s==1]
y_0 = y_s[l_s==0]
y_1 = y_s[l_s==1]

plt.title('01')
plt.scatter(x_0, y_0, c='b')
plt.scatter(x_1, y_1, c='r')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()