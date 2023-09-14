#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time

import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, csr_matrix
from sklearn.datasets import load_svmlight_file
import scipy.io as sio

from models import LRmodel_csr, LRmodel


class data:
    def __init__(self, x, y, ratio):
        self.X = x
        self.Y = y
        # 划分训练集、测试集
        training_samples_ratio = ratio
        N = len(self.Y)
        N_train = int(N * training_samples_ratio)

        rand_idxs = np.random.permutation(N)
        self.X_train = self.X[rand_idxs[:N_train]]
        self.Y_train = self.Y[rand_idxs[:N_train]]
        self.X_test = self.X[rand_idxs[N_train:]]
        self.Y_test = self.Y[rand_idxs[N_train:]]

    def sample(self, chosen_index, batch_size=64):
        start_time = time.time()
        # 随机选择一个小批量
        batch_indices = np.random.choice(chosen_index, batch_size, replace=True)
        X_batch = self.X_train[batch_indices]
        Y_batch = self.Y_train[batch_indices]
        end_time = time.time()
        # print(end_time - start_time)
        return X_batch, Y_batch

    def full(self):
        return self.X_train, self.Y_train
    def length(self):
        return self.X_train.shape[0]

def get_rcv1():
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
    dataset = data(X, Y, 0.8)
    X = dataset.X_train
    Y = dataset.Y_train
    global_model = LRmodel_csr(X.shape[1])
    return dataset, X, Y, global_model

def get_mnist():
    mnist_data = sio.loadmat('../data/mnist/mnist.mat')
    x = mnist_data['Z']
    y = mnist_data['y']
    y = (y.astype(int) >= 5) * 1  # 将数字>=5样本设为正例，其他数字设为负例
    # 添加一列全为1的偏置项列
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    # 归一化特征向量
    x = normalize(x, axis=1, norm='l2')
    dataset = data(x, y, 0.8)
    # 提取出训练集
    X = dataset.X_train
    Y = dataset.Y_train
    # print(X.shape)
    global_model = LRmodel(X.shape[1])
    return dataset, X, Y, global_model

def iid_partition(length, num_clients):
    N_per = int(length / num_clients)
    rand_idxs = np.random.permutation(length)
    dict_index = {}
    for i in range(num_clients):
        dict_index[i] = rand_idxs[i*N_per:(i+1)*N_per]
    return dict_index
