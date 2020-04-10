import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def read_csv(path):
    data = np.genfromtxt('data.csv', delimiter=',')

    x_data = data[:, 0]
    y_data = data[:, 1]
    return x_data, y_data

def linear_function(x_s, a, b):
    y_s = []
    for i in range(len(x_s)):
        y = a*x_s[i] + b
        y_s.append(y)
    return y_s

def linear_model(theta0, theta1, x):
    return theta0 + theta1 * x

def object_function(theta0, theta1, x_range, datas):
    ret = 0
    for i in range(len(x_range)):
        ret += (linear_model(theta0, theta1, x_range[i]) - datas[i])**2
    ret /= 2*(len(x_range))
    #print(ret)
    return ret

def gradient_descent(theta0, theta1, x_s, y_s, learning_rate = 0.005):
    update_theta0 = 0
    for i in range(len(x_s)):
        update_theta0 += (linear_model(theta0, theta1, x_s[i]) - y_s[i])/len(x_s)

    update_theta1 = 0
    for i in range(len(x_s)):
        update_theta1 += ((linear_model(theta0, theta1, x_s[i]) - y_s[i]) * x_s[i])/len(x_s)

    theta0_new = theta0 - learning_rate * update_theta0
    theta1_new = theta1 - learning_rate * update_theta1

    return theta0_new, theta1_new

def convergence(theta0, theta1, t, convergence_rate = 0.000001):
    if theta0[t-1]==0 or theta1[t-1]==0:
        return False
    if np.abs((theta0[t]-theta0[t-1])/theta0[t-1]) < convergence_rate:
        if np.abs((theta1[t]-theta1[t-1])/theta1[t-1]) < convergence_rate:
            return True
    return False


x_data, y_data = read_csv('data.csv')

t=0
theta0=[-30]
theta1=[-30]
energy = []
energy.append(object_function(theta0[t],theta1[t],x_data, y_data))
while True:
    theta0_new, theta1_new = gradient_descent(theta0[t], theta1[t], x_data, y_data)
    theta0.append(theta0_new)
    theta1.append(theta1_new)
    t += 1
    energy.append(object_function(theta0[t],theta1[t],x_data,y_data))
    if convergence(theta0, theta1,t):
        break
min_t = energy.index(min(energy))

plt.title("1_input_points")
plt.plot(x_data, y_data, 'k.')
plt.show()

x_range = [min(x_data), max(x_data)]
plt.title("2_linear_regression_result")
plt.plot(x_data, y_data, 'k.')
plt.plot(x_range, linear_function(x_range, theta1[min_t], theta0[min_t]), 'r')
plt.show()

theta0_range = np.arange(-30,30,0.1)
theta1_range = np.arange(-30,30,0.1)
theta0_range, theta1_range = np.meshgrid(theta0_range, theta1_range)
J = object_function(theta0_range, theta1_range, x_data, y_data)

fig = plt.figure()
plt.title("3_energy_surface")
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_range, theta1_range, J, alpha=0.7, cmap=cm.jet)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('energy')
ax.view_init(45,45)
plt.show()


fig = plt.figure()
plt.title("4_gradient_descent_path")
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_range, theta1_range, J, alpha=0.7, cmap=cm.jet)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('energy')

theta0 = theta0[0:min_t+1]
theta1 = theta1[0:min_t+1]
energy = energy[0:min_t+1]
ax.plot(theta0, theta1, energy, c='k', zorder=5)
ax.view_init(45,45)
plt.show()
print(theta0[-1], theta1[-1])