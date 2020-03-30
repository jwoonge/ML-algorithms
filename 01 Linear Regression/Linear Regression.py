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

def random_generation(x_range):
    a = 2
    b = 3
    ret = []
    n = np.random.randn(len(x_range))
    for i in range(len(x_range)):
        y = a* x_range[i] + b + n[i]
        ret.append(y)
    return ret

def linear_model(theta0, theta1, x):
    return theta0 + theta1 * x

def object_function(theta0, theta1, x_range, datas):
    ret = 0
    for i in range(len(x_range)):
        ret += (linear_model(theta0, theta1, x_range[i]) - datas[i])**2
    ret /= 2*(len(x_range))
    print(ret)
    return ret

def gradient_descent(theta0, theta1, x_s, y_s, learning_rate = 0.02):
    update_theta0 = 0
    for i in range(len(x_s)):
        update_theta0 += (linear_model(theta0, theta1, x_s[i]) - y_s[i])/len(x_s)
    #update_theta0 = round(update_theta0,20)

    update_theta1 = 0
    for i in range(len(x_s)):
        update_theta1 += ((linear_model(theta0, theta1, x_s[i]) - y_s[i]) * x_s[i])/len(x_s)
    #update_theta1 = round(update_theta1,20)

    theta0_new = theta0 - learning_rate * update_theta0
    theta1_new = theta1 - learning_rate * update_theta1

    return theta0_new, theta1_new

def optimization(theta0, theta1):
    print("TODO")

def convergence(theta0, theta1, t, convergence_rate = 0.01):
    if theta0[t]==0 or theta1[t]==0:
        return False
    if np.abs((theta0[t+1]-theta0[t])/theta0[t]) < convergence_rate:
        if np.abs((theta1[t+1]-theta1[t])/theta1[t]) < convergence_rate:
            return True
    
    return False

def convergence_stop(theta0, theta1, t):
    if theta0[t]==theta0[t+1] and theta0[t-1]==theta0[t-2] and theta1[t]==theta1[t+1] and theta1[t-1]==theta1[t-2]:
        return True
    else:
        return False

m = 1000
x_range = [0.01*i for i in range(1,m)]
datas = random_generation(x_range)
learning_rate = 0.2

t=0
theta0 = [0]
theta1 = [1]
energy = []
energy.append(object_function(theta0[t],theta1[t],x_range,datas))

while True:
    theta0_new, theta1_new = gradient_descent(theta0[t], theta1[t], x_range, datas)
    theta0.append(theta0_new)
    theta1.append(theta1_new)
    t += 1
    energy.append(object_function(theta0[t],theta1[t],x_range,datas))
    if convergence_stop(theta0, theta1,t-1):
        break

min_t = energy.index(min(energy))
print("t:",min_t,"  theta0:",theta0[min_t],"  theta1:",theta1[min_t])
