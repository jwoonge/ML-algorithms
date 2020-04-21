import csv
import numpy as np
import matplotlib.pylab as plt

def read_csv(path):
    ret = []
    f = open(path, newline="")
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        temp = []
        for i in range(len(row)):
            temp.append(float(row[i]))
        ret.append(temp)
    return ret

def linear_model(thetas, variables):
    ret = thetas[0]
    for i in range(len(thetas)-1):
        ret += thetas[i+1] * variables[i]
    return ret

def object_function(thetas, datas):
    ret = 0
    for i in range(len(datas)):
        ret += (linear_model(thetas,datas[i][0:3]) - datas[i][-1])**2
    ret /= 2*(len(datas))
    return ret

def gradient_descent(thetas, datas, learning_rate=0.000022):
    thetas_new = []
    for i in range(len(thetas)):
        update = 0
        for j in range(len(datas)):
            if i == 0:
                mult = 1
            else:
                mult = datas[j][i-1]
            update += (linear_model(thetas,datas[j][0:3]) - datas[j][-1])/ len(datas) * mult
        thetas_new.append(thetas[i] - learning_rate*update)
    return thetas_new

def convergence(thetas_last, thetas_new, convergence_rate = 0.00001):
    count = 0
    for i in range(len(thetas_last)):
        if not thetas_last[i]==0:
            if np.abs((thetas_new[i] - thetas_last[i])/thetas_last[i]) <= convergence_rate:
                count += 1
        else:
            if np.abs(thetas_new[i]) <= convergence_rate:
                count += 1
    if count == len(thetas_last):
        return True
    else:
        return False


data_train = read_csv('data_train.csv')
data_test = read_csv('data_test.csv')

t=0
thetas = [[0,0,0,0]]
error_train = [object_function(thetas[t],data_train)]
error_test = [object_function(thetas[t], data_test)]

while True:
    thetas_new = gradient_descent(thetas[t], data_train, 0.00001)
    thetas.append(thetas_new)
    t += 1
    error_train.append(object_function(thetas[t], data_train))
    error_test.append(object_function(thetas[t], data_test))
    print(thetas_new, error_train[-1], error_test[-1])
    if convergence(thetas[t-1],thetas[t]):
        break

plt.title("1_parameters (training)")
thetas = np.transpose(thetas)
plt.plot(thetas[0], c='k')
plt.plot(thetas[1], c='r')
plt.plot(thetas[2], c='g')
plt.plot(thetas[3], c='b')
plt.show()

plt.title("2_training error")
plt.plot(error_train, c='b')
plt.show()

plt.title("3_ testing error")
plt.plot(error_test, c='r')
plt.show()