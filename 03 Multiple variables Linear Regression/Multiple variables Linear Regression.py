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

data_train = read_csv('data_train.csv')
data_test = read_csv('data_test.csv')