import matplotlib.pyplot as plt 
import numpy as np
import random
#一元线性回归 y = ax + b + 误差
#损失函数为均方误差损失函数（MSE），损失函数是y关于ab的函数
#优化方法：梯度下降，ab的变化量是他们对于损失函数的偏导数
#python用sympy可以做微积分
#用一个参数去把变化量给缩小

#参数设置
a = 0
b = 0
n = 100
m = 0.01#放缩梯度
t = 5000 #迭代次数
#初始变量
x = []
y = []
y2 = []
loss_min = 9999999
best_a = 0 
best_b = 0
timer = 0

#随机生成一条线的散点
def initialize():
    for i in range(n):
        temp=random.uniform(0,10)
        x.append(temp)
        y.append(2 * temp + 3 + np.random.normal())
    return x, y

#输出模型
def model(a,b,x):
    y2.clear()
    for i in range(n):
        y2.append(a * x[i] + b)
    return y2

#计算损失函数
def loss_func(y, y2):
    sum = 0
    for i in range(n):
        sum += 0.5/n * (np.square(y[i] - y2[i]))
    return sum

#梯度下降
def optimize(a, b, x, y, y2):
    delta_a = 0
    delta_b = 0
    for i in range(n):
        delta_a += (y[i] - y2[i]) * (-x[i]) /n
        delta_b += (y2[i] - y[i]) /n
    a = a - delta_a*m
    b = b - delta_b*m
    return a, b

#主代码
x, y = initialize()
for i in range(t):
    y2 = model(a,b,x)
    loss = loss_func(y,y2)
    if loss < loss_min:
        loss_min = loss
        best_a = a
        best_b = b
        timer = t
    a, b = optimize(a,b,x,y,y2)
print(best_a, best_b, timer)

plt.scatter(x,y)#散点
x2 = np.arange(0,10,0.1)
y3 = best_a * x2+best_b
plt.plot(x2,y3)
plt.grid(True)
plt.show()

    
    
