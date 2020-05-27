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

    def optimize(self, data_train, label_train, data_test, label_test, epoch):
        forward_train = self.forward_pass(data_train)
        forward_test = self.forward_pass(data_test)
        h_train = forward_train[-1]
        h_test = forward_test[-1]
        loss_train = [loss(h_train, label_train)]
        loss_test = [loss(h_test, label_test)]
        accuracy_train = [accuracy(h_train, label_train)]
        accuracy_test = [accuracy(h_test, label_test)]
        self.best_train_accuracy = accuracy_train[-1]
        self.best_test_accuracy = accuracy_test[-1]
        self.best_train_loss = loss_train[-1]
        self.best_test_loss = loss_test[-1]
        self.best_weights = self.weights
        
        for i in range(epoch-1):
            self.gradient_descent(forward_train, label_train)
            forward_train = self.forward_pass(data_train)
            forward_test = self.forward_pass(data_test)
            h_train = forward_train[-1]
            h_test = forward_test[-1]
            loss_train.append(loss(h_train, label_train))
            loss_test.append(loss(h_test, label_test))
            accuracy_train.append(accuracy(h_train, label_train))
            accuracy_test.append(accuracy(h_test, label_test))
            
            if loss_test[-1] < self.best_test_loss:
                self.best_test_loss = loss_test[-1]
                self.best_train_loss = loss_train[-1]
                self.best_test_accuracy = accuracy_test[-1]
                self.best_train_accuracy = accuracy_train[-1]
                self.best_weights = self.weights

            print("LOS_T:", format(loss_train[-1],'.5f'), "\tACC_T:", format(accuracy_train[-1],'.5f'), "| LOS_V:", format(loss_test[-1],'.5f'), "ACC_V:" ,format(accuracy_test[-1],'.5f'))

        return loss_train, accuracy_train, loss_test, accuracy_test

size_row = 28
size_col = 28
num_class = 10
file_name = "mnist.csv"
datas, labels = read_file(file_name, size_row, size_col, num_class)
data_train = datas[:6000,:] ; data_test = datas[6000:,:]
label_train = labels[:6000,:] ; label_test = labels[6000:,:]
classifier = classifier([784, 196, 49, 10])
loss_train, accuracy_train, loss_test, accuracy_test = classifier.optimize(data_train, label_train, data_test, label_test, 2000)
