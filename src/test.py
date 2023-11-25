from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from sampling import get_mnist, get_rcv1, get_cifar10, get_fashion_mnist
from utils import parameter
import numpy as np

# def loss(y_hat, Y):
#     # 计算损失函数
#     # loss = (-1 / len(Y)) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
#     loss =
#     return loss

def loss(y_hat, y):
    loss = 0

    for i in range(len(y_hat)):
        loss += max(0, 1 - y[i] * y_hat[i])
    return loss

dataset, X, Y, _ = get_rcv1('svm')
params = parameter(2)
# global_model = LogisticRegression(solver='newton-cg', max_iter=2000)
global_model = SVC(kernel='linear')
Y = Y.ravel()

# for i in range(params.iteration):
global_model.fit(X, Y)
y_predict = global_model.decision_function(X)

print(y_predict)

print(loss(y_predict, Y))
print(global_model)



