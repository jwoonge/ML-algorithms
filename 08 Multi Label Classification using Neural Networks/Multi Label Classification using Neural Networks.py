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


size_row = 28
size_col = 28
num_class = 10
file_name = "mnist.csv"
datas, labels = read_file(file_name, size_row, size_col, num_class)
data_train = datas[:6000,:] ; data_test = datas[6000:,:]
label_train = labels[:6000,:] ; label_test = labels[6000:,:]