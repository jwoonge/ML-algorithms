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

def object_func(thetas, x_s,y_s, labels):
    m = len(labels)
    ret = 0
    for i in range(m):
        ret += -labels[i] * math.log(sigmoid(func(thetas,x_s[i],y_s[i]))+np.exp(-64)) /m
        ret += -(1-labels[i])*math.log(1-(sigmoid(func(thetas,x_s[i],y_s[i])))+np.exp(-64)) /m
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

def accuracy(thetas, x_s, y_s, l_s):
    correct = 0
    for i in range(len(l_s)):
        classified = sigmoid(func(thetas, x_s[i],y_s[i]))
        if abs(classified-l_s[i]) < 0.5:
            correct += 1
    return correct/len(l_s)*100

x_s, y_s, l_s = read_data('data-nonlinear.txt')

t=0
thetas = [[1 for x in range(len(x_d))]]
error_train = [object_func(thetas[t], x_s, y_s, l_s)]
accuracy_train = [accuracy(thetas[t], x_s,y_s,l_s)]
best = 0
while True:
    thetas_new = gradient_descent(thetas[t], x_s, y_s, l_s, 3)
    thetas.append(thetas_new)
    t += 1
    error_train.append(object_func(thetas[t], x_s, y_s, l_s))
    accuracy_train.append(accuracy(thetas[t],x_s,y_s,l_s))
    ################
    if accuracy_train[-1] > accuracy_train[best]:
        best = t
    if t>30000:
        break

best_acc = accuracy_train[best]
best_thetas = thetas[best]
