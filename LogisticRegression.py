from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

alpha = 0.003
g = 200
hx = []
y = []
y_pred = []
loss_min = 99999
loss_save = []
def load_data():
    iris = datasets.load_iris()
    x = iris['data']
    y = iris['target']
    x = x[y!=2]
    y = y[y!=2]
    x = x[:,2:]
    # 给x第一列加一列1，常数项
    x_one = np.ones([len(x)])
    x = np.insert(x,0,values=x_one,axis=1)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
    return x_train,x_test,y_train,y_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cal_hx_y():
    hx.clear()
    y.clear()
    for i in range(len(y_train)):
        hx.append(sigmoid(np.dot(theta,x_train[i])))
        if hx[i] >= 0.5:
            y.append(1)
        else:
            y.append(0)
    return hx,y


def cal_loss():
    sum = 0
    for i in range(len(hx)):
        sum += y_train[i] * np.log(hx[i]) + (1-y_train[i]) * np.log(1-hx[i])
    return sum * (-1) / len(hx)


def  gradiant_descent(theta):
    desent_theta = np.zeros([len(theta)])
    for i in range(len(hx)):
        for j in range(len(theta)):
            desent_theta[j] += x_train[i,j] * (hx[i]-y_train[i]) / len(hx)
    theta = theta - desent_theta * alpha
    return theta


def predict():
    for i in range(len(y_test)):
        if sigmoid(np.dot(theta,x_test[i])) >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

if __name__ == "__main__":
    x_train,x_test,y_train,y_test = load_data()
    theta = np.zeros([len(x_train[0])])
    hx,y = cal_hx_y()
    for g in range(g):
        loss = cal_loss()
        loss_save.append(loss)
        if loss_min > loss:
            loss_min = loss
            theta = gradiant_descent(theta)
            hx,y = cal_hx_y()
    print(loss_min)
    print(theta)
    y_pred = predict()
    print(y_test)
    print(y_pred)
