import numpy as np
import matplotlib.pyplot as plt
import math
import random

'''
## 86.44
global x_d 
x_d = [0,0,1,0,2,0,1,3,2]
global y_d 
y_d = [0,0,0,1,0,2,1,2,3]
'''
'''
85.59
global x_d 
x_d = [0,0,1,0,3,2,1,0,1,2,3,4,6,0]
global y_d 
y_d = [0,1,0,3,0,1,2,5,4,3,2,1,0,6]
'''


def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    pointX = data[:,0]
    pointY = data[:,1]
    label = data[:,2]

    return pointX, pointY, label

def sigmoid(z):
    return 1/(1+np.exp(np.float64(-z)+math.e**(-64)))

def func(thetas,dims, x, y):
    ret=thetas[0]
    for i in range(1,len(thetas)):
        ret += thetas[i]*x**dims[i][0]*y**dims[i][1]
    return ret

def object_func(thetas,dims,x_s,y_s, labels):
    m = len(labels)
    ret = 0
    for i in range(m):
        ret += -labels[i] * math.log(sigmoid(func(thetas,dims,x_s[i],y_s[i]))+np.exp(-64)) /m
        ret += -(1-labels[i])*math.log(1-(sigmoid(func(thetas,dims,x_s[i],y_s[i])))+np.exp(-64)) /m
    return ret

def gradient_descent(thetas,dims, x_s, y_s, labels, learning_rate):
    thetas_new = []
    m = len(labels)
    for i in range(len(thetas)):
        update = 0
        for j in range(m): # 이거 ij 순서 바꾸면?
            mult = x_s[j]**dims[i][0] * y_s[j]**dims[i][1]
            update += (sigmoid(func(thetas,dims,x_s[j],y_s[j]))-labels[j])*mult/m
        thetas_new.append(thetas[i]-learning_rate*update)
    return thetas_new

def accuracy(thetas,dims, x_s, y_s, l_s):
    correct = 0
    for i in range(len(l_s)):
        classified = sigmoid(func(thetas,dims, x_s[i],y_s[i]))
        if abs(classified-l_s[i]) < 0.5:
            correct += 1
    return correct/len(l_s)*100
def plot06(thetas):
    plt.title('06')
    x_0 = x_s[l_s==0]
    x_1 = x_s[l_s==1]
    y_0 = y_s[l_s==0]
    y_1 = y_s[l_s==1]
    x_range = np.arange(-1,1,0.005)
    y_range = np.arange(-1,1,0.005)
    x_range, y_range = np.meshgrid(x_range, y_range)
    classified = sigmoid(func(thetas,dims,x_range,y_range))
    plt.contour(x_range,y_range,classified,levels=[0,0.5,1])
    plt.scatter(x_0, y_0, c='b')
    plt.scatter(x_1, y_1, c='r')
    plt.show()

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
x_range = np.arange(-1,1,0.005)
y_range = np.arange(-1,1,0.005)
x_range, y_range = np.meshgrid(x_range, y_range)

for i in range(10000):
    dim_num = random.randint(3,11)
    dims = [[0,0]]
    for j in range(dim_num):
        rdx = random.randint(0,12)
        rdy = random.randint(0,12)
        if not [rdx,rdy] in dims:
            dims.append([rdx,rdy])
    print(dims,'작업중')
    t=0
    thetas = [[1 for x in range(len(dims))]]
    error_train = [object_func(thetas[t],dims, x_s, y_s, l_s)]
    accuracy_train = [accuracy(thetas[t],dims, x_s,y_s,l_s)]
    best = 0
    epoch=5000
    while True:
        thetas_new = gradient_descent(thetas[t],dims, x_s, y_s, l_s, 10)
        thetas.append(thetas_new)
        t += 1
        error_train.append(object_func(thetas[t],dims, x_s, y_s, l_s))
        accuracy_train.append(accuracy(thetas[t],dims,x_s,y_s,l_s))
        ################
        if accuracy_train[-1] > accuracy_train[best]:
            best = t
        if epoch>50000:
            break
        if t>epoch:
            if accuracy_train[best]>70 and abs(best-t)<15000:
                print("추가")
                epoch+=3000
            else:
                break
        ###############

    best_acc = max(accuracy_train)
    print(best_acc)
    title = "["
    for i in range(len(dims)):
        for j in range(len(dims[i])):
            title += str(dims[i][j])
        title += ' '
    title += '] '
    title += str(round(best_acc,2))
    if best_acc > 80: 
        plt.figure()
        plt.title(title)
        classified = sigmoid(func(thetas[best],dims,x_range,y_range))
        plt.contour(x_range,y_range,classified,levels=[0,0.5,1])
        plt.scatter(x_0, y_0, c='b')
        plt.scatter(x_1, y_1, c='r')
        plt.savefig('result/'+str(i)+'.png')
        #f.write(str(i)+str(dims)+'\n')