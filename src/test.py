from sklearn.linear_model import LogisticRegression

from sampling import get_mnist
from utils import parameter
import numpy as np

def loss(y_hat, Y):
    # 计算损失函数
    loss = (-1 / len(Y)) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
    return loss

dataset, X, Y, _ = get_mnist()
params = parameter(2)
global_model = LogisticRegression(solver='newton-cg', max_iter=2000)
Y = Y.ravel()

# for i in range(params.iteration):
global_model.fit(X, Y)
y_predict = np.minimum(1 - 1e-15, np.maximum(1e-15, global_model.predict_proba(X)))
print(loss(y_predict[:, 1], Y))
print(global_model)



