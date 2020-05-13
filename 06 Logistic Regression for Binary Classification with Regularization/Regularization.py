import numpy as np
import matplotlib.pyplot as plt

global x_s
global y_s
global l_s
global dims
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

def gradient_descent(thetas, lam):
    m = len(l_s)
    data_fidelity = np.zeros(len(thetas[t]))
    regular = np.zeros(len(thetas[t]))

    for j in range(m):
        z = sigmoid(func(thetas, x_s[j], y_s[j]))-l_s[j]
        for i in range(len(thetas)):
            mult = x_s[j]**dims[i][0] * y_s[j]**dims[i][1]
            data_fidelity[i] += z*mult
    data_fidelity = data_fidelity/m
    
    for i in range(len(thetas)):
        regular[i] = lam * thetas[i]

    return thetas - learning_rate * (data_fidelity + regular)

def object_func(thetas):
    m = len(l_s)
    data_fidelity = 0
    for i in range(m):
        data_fidelity += -l_s[i] * np.log(sigmoid(func(thetas,x_s[i],y_s[i])))
        data_fidelity += -(1-l_s[i])* np.log(1-(sigmoid(func(thetas,x_s[i],y_s[i]))))
    data_fidelity /= m
    regular = np.sum(np.square(thetas)) * lam/2
    return data_fidelity + regular

def accuracy(thetas):
    correct = 0
    for i in range(len(l_s)):
        classified = sigmoid(func(thetas, x_s[i],y_s[i]))
        if abs(classified-l_s[i]) < 0.5:
            correct += 1
    return correct/len(l_s)*100

x_s, y_s, l_s = read_data("data-nonlinear.txt")
dims = []
for i in range(10):
    for j in range(10):
        dims.append([i,j])
dims = np.array(dims)
lambda_overfit = 0
lambda_justright = 0.5
lambda_underfit = 10
learning_rate = 0.1

t = 0
thetas_overfit = [np.zeros(100)]
thetas_justright = [np.zeros(100)]
thetas_underfit = [np.zeros(100)]
err_overfit = [object_func(thetas_overfit)]
err_justright = [object_func(thetas_justright)]
err_underfit = [object_func(thetas_underfit)]
acc_overfit = [accuracy(thetas_overfit)]
acc_justright = [accuracy(thetas_justright)]
acc_underfit = [accuracy(thetas_underfit)]

while True:
    thetas_overfit.append(gradient_descent(thetas_overfit, lambda_overfit))
    err_overfit.append(object_func(thetas_overfit[-1]))
    acc_overfit.append(accuracy(thetas_overfit[-1]))

    thetas_justright.append(gradient_descent(thetas_justright, lambda_justright))
    err_justright.append(object_func(thetas_justright[-1]))
    acc_justright.append(accuracy(thetas_justright[-1]))

    thetas_underfit.append(gradient_descent(thetas_underfit, lambda_underfit))
    err_underfit.append(object_func(thetas_underfit[-1]))
    acc_underfit.append(accuracy(thetas_underfit[-1]))

    t+= 1

    if t > 1000:
        break
    
    


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

###### result 02 ######
plt.title('02 training error')
plt.plot(err_overfit, c='r')
plt.plot(err_justright, c='g')
plt.plot(err_underfit, c='b')
plt.show()

###### result 03 ######

###### result 04 ######
plt.title('03 training accuracy')
plt.plot(acc_overfit, c='r')
plt.plot(acc_justright, c='g')
plt.plot(acc_underfit, c='b')
plt.show()

###### result 05 ######

###### result 06 ######
