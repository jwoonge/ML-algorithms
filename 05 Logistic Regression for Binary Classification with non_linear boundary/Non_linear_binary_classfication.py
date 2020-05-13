import numpy as np
import matplotlib.pyplot as plt
import math
dims = [[0,0],[1,0],[0,1],[2,0],[1,1],[0,2],[3,2],[2,3],[2,5],[5,2],[7,0],[0,7]]
global x_d 
x_d = [0,1,0,2,1,0,3,2,2,5,7,2]
global y_d 
y_d = [0,0,1,0,1,2,2,3,5,2,2,7]

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
        for j in range(m):
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

def print_test(t, thetas, error_train, acc_train, max_acc_train):
    print(t, end=', ')
    for i in range(len(thetas)):
        print(round(thetas[i],10), end=', ')
    print(round(error_train,6), round(acc_train,2), round(max_acc_train,2), end="\n")

x_s, y_s, l_s = read_data('data-nonlinear.txt')
x_0 = x_s[l_s==0]
x_1 = x_s[l_s==1]
y_0 = y_s[l_s==0]
y_1 = y_s[l_s==1]

t=0
thetas = [[-100 for x in range(len(x_d))]]
error_train = [object_func(thetas[t], x_s, y_s, l_s)]
accuracy_train = [accuracy(thetas[t], x_s,y_s,l_s)]
best = 0
while True:
    thetas_new = gradient_descent(thetas[t], x_s, y_s, l_s, 10)
    thetas.append(thetas_new)
    t += 1
    error_train.append(object_func(thetas[t], x_s, y_s, l_s))
    accuracy_train.append(accuracy(thetas[t],x_s,y_s,l_s))
    
    ################
    if accuracy_train[-1] >= accuracy_train[best]:
        best = t
    print_test(t,thetas[t],error_train[-1],accuracy_train[-1],accuracy_train[best])
    if t>50000:
        break

best_thetas = thetas[best]

###### result 01 ######
plt.title('01 training data')
plt.scatter(x_0, y_0, c='b')
plt.scatter(x_1, y_1, c='r')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

###### result 02 ######

###### result 03 ######
plt.title('03 training error')
plt.plot(error_train, c='b')
plt.show()

###### result 04 ######
plt.title('04 training accuracy')
plt.plot(accuracy_train, c='r')
plt.show()

###### result 05 ######
print('final accuracy : ',accuracy_train[-1])
print('best  accuracy : ',accuracy_train[best])

###### result 05 ######
plt.title('06 classifier')
x_range = np.arange(-1,1,0.005)
y_range = np.arange(-1,1,0.005)
x_range, y_range = np.meshgrid(x_range, y_range)
classified = sigmoid(func(thetas[best],x_range,y_range))
plt.contour(x_range,y_range,classified,levels=[0,0.5,1])
plt.colorbar()
plt.scatter(x_0, y_0, c='b')
plt.scatter(x_1, y_1, c='r')
plt.show()