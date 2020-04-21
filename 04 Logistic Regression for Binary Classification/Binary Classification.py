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