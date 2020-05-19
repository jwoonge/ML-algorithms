import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name, num_row, num_col):
    f = open(file_name, "r")
    data = f.readlines()
    f.close()

    num_image = len(data)
    count = 0

    list_image = np.empty((num_row * num_col, num_image), dtype=float)
    list_label = np.empty(num_image, dtype=int)

    for line in data:
        line_data = line.split(",")
        label = line_data[0]
        im_vector = np.asfarray(line_data[1:])

        list_label[count] = label
        list_image[:,count] = im_vector
        count += 1

    return list_image, list_label

def sigmoid(z):
    return 1/(1+np.exp(np.float64(-z)))

def logistic_unit(params, input):
    return sigmoid(np.sum(params * input))

num_row = 28
num_col = 28
file_name = "mnist_test.csv"

images, labels = read_file(file_name, num_row, num_col)

avg_image = []
for num in range(10):
    avg_image.append(np.average(images[:,labels==num], axis=1).reshape((num_row, num_col)))

thetas = np.random.randn(num_row * num_col)

z = np.empty(len(labels))
for i in range(len(labels)):
    z[i] = logistic_unit(thetas,images[:,i])
avg = []
for i in range(10):
    avg.append(np.average(z[labels==i]))