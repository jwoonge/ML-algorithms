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

data_train = read_csv('data_train.csv')
data_test = read_csv('data_test.csv')