import numpy as np
import matplotlib as plt
import math

def original_function(input):
    a = 2
    b = 3
    ret = []
    for i in range(len(input)):
        y = a*input[i] + b
        ret.append(y)
    return ret

def random_input(x):
    a = 2
    b = 3
    ret = []
    n = np.random.randn(len(x))
    for i in range(len(x)):
        y = a* x[i] + b + n[i]
        ret.append(y)
    return ret

def object_function(theta0, theta1):
    print("TODO")


x_range = [i for i in range(10)]

datas = random_input(x_range)
