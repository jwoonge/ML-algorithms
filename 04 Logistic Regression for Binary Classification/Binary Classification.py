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

def convergence(thetas_last, thetas_new, convergence_rate = 0.00000001):
    count = 0
    for i in range(len(thetas_last)):
        rate = np.abs((thetas_new[i]+math.e**(-64) - thetas_last[i])/(thetas_last[i]+math.e**(-64)))
        print(rate)
        if rate <= convergence_rate:
            count += 1
    if count == len(thetas_last):
        return True
    else:
        return False

variables, labels = read_data('data.txt')
t=0
thetas = [[-27,0,0]]
error_train = [object_func(thetas[t],variables,labels)]

while True:
    thetas_new = gradient_descent(thetas[t], variables, labels, 0.003)
    thetas.append(thetas_new)
    t+= 1
    error_train.append(object_func(thetas[t], variables, labels))
    print(thetas_new, error_train[-1])
    if convergence(thetas[t-1], thetas[t]):
        print(convergence(thetas[t-1], thetas[t]))
        break
    if t>300:
        break
min_t = error_train.index(min(error_train))
opt_thetas = thetas[min_t]

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

plt.title("02_estimated_parameters")
thetas = np.transpose(thetas)
plt.plot(thetas[0], c='r')
plt.plot(thetas[1], c='g')
plt.plot(thetas[2], c='b')
plt.show()

plt.title("03_training_error")
plt.plot(error_train, c='b')
plt.show()

plt.title("04_classifier")
x_range = np.arange(30,100,0.5)
y_range = np.arange(30,100,0.5)
x_range,y_range = np.meshgrid(x_range,y_range)
classified = sigmoid(linear_func(opt_thetas,[x_range,y_range]))
plt.contourf(x_range,y_range,classified, 300, cmap='RdBu_r',zorder=1)
plt.scatter(x_label0, y_label0,alpha=1, c='b',zorder=2)
plt.scatter(x_label1, y_label1,alpha=1, c='r',zorder=2)
plt.show()