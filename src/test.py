from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from tensorflow import Variable

from sampling import get_mnist
from utils import parameter
from models import LRmodel_torch
import torch
import numpy as np

# def loss(y_hat, Y):
#     # 计算损失函数
#     loss = (-1 / len(Y)) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
#     return loss
#
# dataset, X, Y, _ = get_mnist()
# params = parameter(2)
# global_model = LogisticRegression(solver='newton-cg', max_iter=2000)
# Y = Y.ravel()
#
# # for i in range(params.iteration):
# global_model.fit(X, Y)
# y_predict = np.minimum(1 - 1e-15, np.maximum(1e-15, global_model.predict_proba(X)))
# print(loss(y_predict[:, 1], Y))
# print(global_model)
    # print("iteration {} and loss {}".format(i, loss))


def closure():
    pred_output = global_model(X)
    loss = criterion(pred_output, Y)
    # print("batch loss: {:.9f}".format(loss.item()))
    loss.backward()
    return loss

dataset, X, Y, _ = get_mnist()
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)
Y = Y.to(torch.float64)
params = parameter(2)
global_model = LRmodel_torch(X.shape[1])
global_model = global_model.to(torch.float64)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(global_model.parameters(), lr=1)

epoch_list = []
loss_list = []

for epoch in range(10):
    optimizer.zero_grad()
    outputs = global_model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    print("Iteration: {}. Loss: {}".format(epoch+1, loss))


