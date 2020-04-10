import numpy as np
import matplotlib.pylab as plt

def linear_function(x_s, a, b):
    y_s = []
    for i in range(len(x_s)):
        y = a*x_s[i] + b
        y_s.append(y)
    return y_s

def random_generation(x_range, a, b):
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

def gradient_descent(theta0, theta1, x_s, y_s, learning_rate = 0.03):
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

def show_result():
    plt.title("1_input_data")
    plt.plot(x_range, datas, 'k.')
    plt.plot(x_range, linear_function(x_range, a=a, b=b), 'b')
    plt.show()

    plt.title("2_output_results")
    plt.plot(x_range, datas, 'k.')
    plt.plot(x_range, linear_function(x_range, theta1[min_t], theta0[min_t]), 'r')
    plt.show()

    plt.title("3_energy values")
    plt.plot(energy, 'b')
    plt.show()

    plt.title("4_model parameters")
    plt.plot(theta0, 'r')
    plt.plot(theta1, 'b')
    plt.show()

a=3
b=-2
m = 1000
x_range = [0.01*i for i in range(m)]
datas = random_generation(x_range, a=a, b=b)

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
    if convergence(theta0, theta1,t):
        break

min_t = energy.index(min(energy))
print("t:",min_t,"  theta0:",theta0[min_t],"  theta1:",theta1[min_t])
show_result()

