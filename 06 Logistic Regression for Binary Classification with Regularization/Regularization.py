import numpy as np
import matplotlib.pyplot as plt

global x_s
global y_s
global l_s
global dims
global thetas
global lam
global learning_rate

def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    pointX = data[:,0]
    pointY = data[:,1]
    label = data[:,2]

    return pointX, pointY, label

def sigmoid(z):
    return 1/(1+np.exp(np.float64(-z)))

def func(theta, x, y):
    return np.sum(theta * (x**dims[:,0] * y**dims[:,1]))

def func_(theta, x, y):
    ret = theta[0]
    for i in range(1, len(theta)):
        ret += theta[i] * x**dims[i][0] * y**dims[i][1]
    return ret

def gradient_descent(t):
    m = len(l_s)
    data_fidelity = np.zeros(len(thetas[t]))
    regular = np.zeros(len(thetas[t]))

    for j in range(m):
        z = sigmoid(func(thetas[t], x_s[j], y_s[j]))-l_s[j]
        for i in range(len(thetas[t])):
            mult = x_s[j]**dims[i][0] * y_s[j]**dims[i][1]
            data_fidelity[i] += z*mult
    data_fidelity = data_fidelity/m
    
    for i in range(len(thetas[t])):
        regular[i] = lam * thetas[t][i]

    return thetas[-1] - learning_rate * (data_fidelity + regular)

def object_func(t):
    m = len(l_s)
    data_fidelity = 0
    for i in range(m):
        data_fidelity += -l_s[i] * np.log(sigmoid(func(thetas[t],x_s[i],y_s[i])))
        data_fidelity += -(1-l_s[i])* np.log(1-(sigmoid(func(thetas[t],x_s[i],y_s[i]))))
    data_fidelity /= m
    regular = np.sum(np.square(thetas[t])) * lam/2
    return data_fidelity + regular

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