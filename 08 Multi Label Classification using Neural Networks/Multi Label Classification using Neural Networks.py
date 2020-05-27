import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name, size_row, size_col, num_class):
    f = open(file_name, "r")
    data = f.readlines()
    f.close()

    num_image = len(data)
    count = 0

    list_image = np.empty((num_image, size_row * size_col), dtype=float)
    one_hot_label = np.zeros((num_image, num_class), dtype=int)

    for line in data:
        line_data = line.split(",")
        label = line_data[0]
        im_vector = np.asfarray(line_data[1:])
        im_vector = (im_vector - np.min(im_vector)) / (np.max(im_vector)-np.min(im_vector))
        list_image[count,:] = im_vector
        one_hot_label[count][int(label)] = 1
        count += 1

    return list_image, one_hot_label

def sigmoid(input):
    return 1/(1+np.exp(-input))
def sigmoid_d(input):
    return sigmoid(input)*(1-sigmoid(input))
def accuracy(h, label):
    forward_passed = np.argmax(h, axis=1)
    correct = forward_passed[forward_passed==np.argmax(label, axis=1)]
    return len(correct)/len(forward_passed)*100
def loss(h, label):
    return np.sum(np.average(-label * np.log(h) - (1-label)*np.log(1-h), axis=0))


class classifier:
    def __init__(self, shape, learning_rate=1):
        self.weights = []
        for i in range(len(shape)-1):
            self.weights.append(np.random.randn(shape[i]+1, shape[i+1]))
        self.learning_rate = learning_rate
        self.num_layer = len(self.weights)
    
    def forward_pass(self, input, weights=[]):
        if weights==[]:
            weights = self.weights
        values = []
        values.append(np.insert(input, 0, 1, axis=1))
        values.append(np.dot(values[-1], weights[0]))
        values.append(np.insert(sigmoid(values[-1]),0,1,axis=1))
        for i in range(1, len(weights)-1):
            values.append(np.dot(values[-1], weights[i]))
            values.append(np.insert(sigmoid(values[-1]),0,1,axis=1))
        values.append(np.dot(values[-1], weights[-1]))
        values.append(sigmoid(values[-1]))
        return values

    def gradient_descent(self, forward_values, label):
        update_values = []
        m = len(label)
        mult = (sigmoid_d(forward_values[-2]) * ((1-label)/(1-forward_values[-1]) - label/forward_values[-1])).T
        layer = len(self.weights)-1
        update_values.append(np.dot(mult, forward_values[2*layer]).T)
        for i in range(1, len(self.weights)):
            layer = len(self.weights) - i - 1
            mult = (np.dot(mult.T, self.weights[layer+1][1:,:].T) * (sigmoid_d(forward_values[layer*2+1]))).T
            update_values.append(np.dot(mult, forward_values[2*layer]).T)
        for i in range(len(self.weights)):
            layer = len(self.weights)-i-1
            self.weights[layer] = self.weights[layer] - self.learning_rate/m * update_values[i]

size_row = 28
size_col = 28
num_class = 10
file_name = "mnist.csv"
datas, labels = read_file(file_name, size_row, size_col, num_class)
data_train = datas[:6000,:] ; data_test = datas[6000:,:]
label_train = labels[:6000,:] ; label_test = labels[6000:,:]
classifier = classifier([784, 196, 49, 10])