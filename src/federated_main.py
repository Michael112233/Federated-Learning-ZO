#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import random
import time
import numpy as np
from sampling import data, iid_partition
from models import LRmodel, LRmodel_csr
from sklearn.datasets import load_svmlight_file
import scipy.io as sio
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, csr_matrix
from options import args_parser
from algorithm import dataset_name, client_number, iteration, client_rate, FedAvg, get_loss, Zeroth_grad

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()

    algorithm_name = 'zeroth_grad'

    # initialize
    if dataset_name == 'rcv':
        X, Y = load_svmlight_file('../data/rcv/rcv1_test.binary')
        Y = Y.reshape(-1, 1)
        Y = (Y + 1) / 2
        # 创建全为1的列向量，作为偏置项列
        bias_column = np.ones(X.shape[0])
        # 将偏置项列转换为稀疏矩阵格式
        bias_column_sparse = csr_matrix(bias_column).transpose()
        # 将稀疏矩阵与偏置项列拼接
        X = hstack([X, bias_column_sparse])
        # 归一化特征向量
        X = normalize(X, axis=1, norm='l2')
        dataset = data(X, Y, 0.7)
        X = dataset.X_train
        Y = dataset.Y_train
        global_model = LRmodel_csr(X.shape[1])
    else:
        mnist_data = sio.loadmat('../data/mnist/mnist.mat')
        x = mnist_data['Z']
        y = mnist_data['y']
        y = (y.astype(int) >= 5) * 1  # 将数字>=5样本设为正例，其他数字设为负例
        # 添加一列全为1的偏置项列
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        # 归一化特征向量
        x = normalize(x, axis=1, norm='l2')
        dataset = data(x, y, 0.7)
        # 提取出训练集
        X = dataset.X_train
        Y = dataset.Y_train
        global_model = LRmodel(X.shape[1])

    if algorithm_name == 'zeroth_grad':
        algorithm = Zeroth_grad(dataset, global_model)
    else:
        algorithm = FedAvg(dataset, global_model)

    client_index = []
    client_dataset = {}
    losses = []
    iter = []
    weights = np.ones(global_model.len()).reshape(-1, 1)

    # 划分客户端训练集
    partition_index = iid_partition(dataset.length(), client_number)
    for i in range(client_number):
        client_index.append(i)
    # print(client_index)

    # Training
    for i in range(iteration):
        weights_list = []
        chosen_client_num = int(max(client_rate * client_number, 1))
        chosen_client = random.sample(client_index, chosen_client_num)

        # print(i)
        # train
        for k in chosen_client:
            weight_of_client, total_grad = algorithm.update_client(weights, partition_index[k], i)
            weights_list.append(copy.deepcopy(weight_of_client))

        weights = algorithm.average(weights_list)

        if (i + 1) % 100 == 0:
            iter.append(i + 1)
            losses = get_loss(global_model, dataset, weights, i + 1, losses)

    end_time = time.time()
    print("total time is {:.3f}".format(end_time-start_time))
    print("total grad times is {:.2f}".format(total_grad))

