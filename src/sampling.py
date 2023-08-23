#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np

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
        # 随机选择一个小批量
        batch_indices = np.random.choice(chosen_index, batch_size, replace=True)
        X_batch = self.X_train[batch_indices]
        Y_batch = self.Y_train[batch_indices]
        return X_batch, Y_batch

    def full(self):
        return self.X_train, self.Y_train
    def length(self):
        return self.X_train.shape[0]


def iid_partition(length, num_clients):
    N_per = int(length / num_clients)
    rand_idxs = np.random.permutation(length)
    dict_index = {}
    for i in range(num_clients):
        dict_index[i] = rand_idxs[i*N_per:(i+1)*N_per]
    return dict_index
