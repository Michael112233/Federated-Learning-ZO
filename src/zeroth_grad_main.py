#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import random
import time
import numpy as np
from sampling import data, iid_partition
from models import LRmodel, LRmodel_csr, Zeroth_grad_model
from sklearn.datasets import load_svmlight_file
import scipy.io as sio
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, csr_matrix

from options import args_parser
from utils import get_loss


def update_client(weights, chosen_index, current_round, total_grad):
    for i in range(local_iteration):
        X, Y = dataset.sample(chosen_index, batch_size)
        if current_round <= memory_length+1:
            v_matrix = np.random.randn(dataset.X_train.shape[1], 1)
        else:
            z0 = np.random.randn(dataset.X_train.shape[1], 1)
            z1 = np.random.randn(memory_length, 1)
            v_matrix = np.sqrt(1-alpha) * z0 + np.sqrt(alpha * dataset.X_train.shape[1] / memory_length) * p_matrix.dot(z1)
            # print(p_matrix)
        g = global_model.grad(weights, radius, v_matrix, X, Y)
        total_grad += 1
        weights -= eta * g
    return weights, total_grad

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    dataset_name = args.dataset
    dataset_name = 'mnist'
    if dataset_name == 'mnist':
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
        global_model = Zeroth_grad_model()
    elif dataset_name == 'rcv':
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
        global_model = LRmodel_csr()

    client_rate = 0.5
    client_number = 50
    client_index = []
    client_dataset = {}
    local_iteration = 20
    total_grad = 0
    dataset_length = dataset.length()
    n = X.shape[1]
    batch_size = 64
    # 划分客户端训练集

    partition_index = iid_partition(dataset_length, client_number)
    for i in range(client_number):
        client_index.append(i)
    # print(client_index)

    # initialize parameters
    iteration = 5000
    eta = 0.1
    radius = 1e-6
    alpha = 0.5
    memory_length = 5

    losses = []
    iter = []
    # initialize x0
    weights = np.ones(dataset.X_train.shape[1]).reshape(-1, 1)
    last_weight = np.zeros(dataset.X_train.shape[1]).reshape(-1, 1)
    q_matrix = np.random.randn(dataset.X_train.shape[1], 1)
    p_matrix = np.empty((n, memory_length))
    weight_server_list = []
    delta_weight_list = []
    for i in range(iteration):
        # draw a client set C_r
        weight_server_list.append(copy.deepcopy(weights))
        weights_list = []
        cnt = 0
        chosen_client_num = int(max(client_rate * client_number, 1))
        chosen_client = random.sample(client_index, chosen_client_num)
        # train in each client
        for k in chosen_client:
            weight_of_client, total_grad = update_client(weights, partition_index[k], i, total_grad)
            weights_list.append(copy.deepcopy(weight_of_client))

        weights = sum(weights_list) / chosen_client_num

        if i % memory_length == 1 and i > 1:
            # weight_server_array = np.array(weight_server_list).reshape((memory_length+1, n))
            # print("i={}".format(i))
            for j in range(i, i-memory_length, -1):
                # print(j, j-1)
                delta_weight_list.append(copy.deepcopy(weight_server_list[j] - weight_server_list[j-1]))
            # print(delta_weight_list)
            delta_list = np.array(delta_weight_list)
            p_matrix, _ = np.linalg.qr(delta_list.T)
            delta_weight_list = []
            p_matrix = p_matrix.reshape((n, memory_length))
            # print(p_matrix.shape)

        if (i + 1) % 50 == 0:
            iter.append(i + 1)
            losses = get_loss(global_model, dataset, weights, i+1, losses)

    end_time = time.time()
    print("total time is {:.3f}".format(end_time-start_time))
    print("total grad times is {:.2f}".format(total_grad))

