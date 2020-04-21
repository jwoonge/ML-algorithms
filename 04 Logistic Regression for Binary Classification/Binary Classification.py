import numpy as np
import matplotlib.pyplot as plt
import math

def read_data(filename):
    data = np.genfromtxt("data.txt", delimiter=',')
    variables = []
    for i in range(len(data)):
        variables.append([data[i][0], data[i][1]])
    labels = data[:,2]

    return variables,labels

def sigmoid(z):
    return 1/(1+np.exp(-z))

def linear_func(thetas, variables):
    ret = thetas[0]
    for i in range(len(thetas)-1):
        ret += thetas[i+1] * variables[i]
    return ret

def object_func(thetas, variables, labels):
    m = len(variables)
    ret = 0
    for i in range(m):
        ret += -labels[i] * np.log(sigmoid(linear_func(thetas,variables[i]))) /m
        ret += -(1-labels[i])*np.log(1-sigmoid(linear_func(thetas,variables[i]))+math.e**(-64)) /m
    return ret

def gradient_descent(thetas, variables, labels, learning_rate):
    thetas_new = []
    m = len(variables)
    for i in range(len(thetas)):
        update = 0
        for j in range(m):
            if i==0:
                mult=1
            else:
                mult = variables[j][i-1]
            update += (sigmoid(linear_func(thetas,variables[j]))-labels[j])*mult/m
        thetas_new.append(thetas[i]-learning_rate*update)
    return thetas_new

variables, labels = read_data('data.txt')


plt.title("01_training_data")
x_label0 = []
y_label0 = []
x_label1 = []
y_label1 = []
for i in range(len(variables)):
    if labels[i]==0:
        x_label0.append(variables[i][0])
        y_label0.append(variables[i][1])
    else:
        x_label1.append(variables[i][0])
        y_label1.append(variables[i][1])
plt.scatter(x_label0, y_label0,alpha=0.5, c='b')
plt.scatter(x_label1, y_label1,alpha=0.5, c='r')
plt.show()