# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.calibration import CalibratedClassifierCV
#
# from sampling import get_mnist, get_rcv1, get_cifar10, get_fashion_mnist, non_iid_partition
# from utils import parameter
# import numpy as np
#
# # def loss(y_hat, Y):
# #     # 计算损失函数
# #     # loss = (-1 / len(Y)) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
# #     loss =
# #     return loss
#
# def loss(y_hat, y):
#     loss = 0
#
#     for i in range(len(y_hat)):
#         loss += max(0, 1 - y[i] * y_hat[i])
#     return loss
#
# dataset, X, Y, _ = get_rcv1('logistic')
# dataset.sort()
# partition_index = non_iid_partition(dataset.length(), 100, dataset.sort_index)
# dataset.test()
# # x_train = X[partition_index]
# # print(x_train)
# # y_train = Y[partition_index]
# # # print(partition_index[0])
# for i in range(len(partition_index)):
#     print(Y[partition_index[i]])
#     print(partition_index[i])
# # params = parameter(2)
# # global_model = LogisticRegression(solver='newton-cg', max_iter=2000)
# # # global_model = SVC(kernel='linear')
# # Y = Y.ravel()
# # y_train = y_train.ravel()
# #
# # # for i in range(params.iteration):
# # global_model.fit(x_train, y_train)
# # y_predict = global_model.decision_function(X)
# #
# # print(y_predict)
# # cnt = 0
# # for i in range(len(y_predict)):
# #     y = (y_predict[i] > 0.5)
# #     if y == Y[i]:
# #         cnt += 1
# #
# # print(cnt / len(Y))
# # print(global_model)
import os

import pandas as pd

from src.utils import get_eta, get_file_name


# info1 = get_file_name('mnist', 'logistic', 'FedAvg_GD', "non_iid")
# print("the info1 is ", info1)
# # fm = pd.read_csv('../performance/experiment/mnist/FedAvg_GD/logistic/non_iid\eta=1\(time=2024-05-28-20-57-14).csv')
# # print(fm)
# # get_file_name()

