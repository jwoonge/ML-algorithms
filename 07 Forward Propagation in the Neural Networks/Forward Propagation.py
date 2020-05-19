import numpy as np

def read_file(file_name, size_row, size_col):
    f = open(file_name, "r")
    data = f.readlines()
    f.close()

    num_image = len(data)
    count = 0

    list_image = np.empty((size_row * size_col, num_image), dtype=float)
    list_label = np.empty(num_image, dtype=int)

    for line in data:
        line_data = line.split(",")
        label = line_data[0]
        im_vector = np.asfarray(line_data[1:])

        list_label[count] = label
        list_image[:,count] = im_vector
        count += 1

    return list_image, list_label

size_row = 28
size_col = 28
file_name = "mnist_test.csv"

images, labels = read_file(file_name, size_row, size_col)

avg_image = []
for num in range(10):
    avg_image.append(np.average(images[:,labels==num], axis=1).reshape((size_row, size_col)))