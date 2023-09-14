from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from sampling import get_mnist
from utils import parameter
import numpy as np

dataset, X, Y, _ = get_mnist()
global_model = LogisticRegression(solver='newton-cg')
parameters = parameter(1000 * dataset.length())
# weights = np.ones(dataset.length()).reshape(-1, 1)
print(X.shape, Y.shape)
Y = Y.ravel()
for global_iter in range(parameters.iteration): # parameters.iteration
    global_model.fit(X, Y)
    if (global_iter + 1) % 100 == 0:
        y_hat = global_model.predict(X)
        # 计算损失函数
        loss = (-1 / len(X)) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
        print("current iteration is {} and current_loss is {}".format(global_iter, loss))
