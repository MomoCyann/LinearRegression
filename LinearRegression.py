import random
import numpy as np
import matplotlib.pyplot as plt

# config
# origin: y = 2x + 3
m = 200
n = 2000
x = np.zeros([m])
y = np.zeros([m])
hx = np.zeros([m])
a = 0
b = 0
alpha = 0.01 # 学习速率/梯度下降幅度
loss_min = 99999

def generate(x,y,hx):
    for i in range(m):
        tmp = random.uniform(0,20)
        x[i] = tmp
        y[i] = 2*tmp + 3 + np.random.normal() # 正态分布
        hx[i] = a*tmp + b
    return x,y,hx


def cal_loss():
    loss = (1/m) * np.sum(np.square(hx - y))
    return loss


def gradient_desent(a,b):
    desent_a = 0
    desent_b = 0
    for i in range(m):
        desent_a += x[i] * (hx[i]-y[i]) / m
        desent_b += (hx[i]-y[i]) / m
    a = a - desent_a * alpha
    b = b - desent_b * alpha
    return a,b


if __name__ == "__main__":
    generate(x,y,hx)
    for i in range(n):
        loss = cal_loss()
        if loss_min > loss:
            loss_min = loss
            a,b = gradient_desent(a,b)
            for i in range(m):
                hx[i] = a*x[i] + b
    print("final loss: " + str(loss))
    print("origin: y = 2x + 3")
    print("result is y = " + str(a) + "x + " + str(b))
    plt.scatter(x, y)  # 散点
    x2 = np.arange(0, 20, 0.1)
    y2 = a * x2 + b
    plt.plot(x2, y2, color='r')
    plt.show()
