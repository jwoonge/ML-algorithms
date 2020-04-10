import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def read_csv(path):
    data = np.genfromtxt('data.csv', delimiter=',')

    x_data = data[:, 0]
    y_data = data[:, 1]
    return x_data, y_data

x_data, y_data = read_csv('data.csv')