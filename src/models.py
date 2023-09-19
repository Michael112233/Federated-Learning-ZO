#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import torch


#logistic regression mnist
class LRmodel:
    def __init__(self, length):
        self.length = length
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def grad(self, w, X, Y):
        y_hat = self.sigmoid(np.dot(X, w))
        # print()
        # print(w.shape)
        # print(X.shape)
        # print(Y.shape)
        # 计算梯度
        dw = (1 / len(X)) * np.dot(X.T, y_hat - Y)
        return dw
    def loss(self, w, X, Y):
        y_hat = np.minimum(1 - 1e-15, np.maximum(1e-15, self.sigmoid(np.dot(X, w))))
        # 计算损失函数
        loss = (-1 / len(X)) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
        return loss
    def acc(self, w, X, Y):
        y_hat = self.sigmoid(np.dot(X, w))
        y_hat = (y_hat >= 0.5) * 1
        corrent_array = y_hat - Y
        corrent_index = np.where(corrent_array == 0)
        accuracy = len(corrent_index[0]) / len(Y)
        return accuracy*100
    def len(self):
        return self.length

class LRmodel_csr:
    def __init__(self, length):
        self.length = length
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def grad(self, w, X, Y):
        y_hat = np.minimum(1 - 1e-15, np.maximum(1e-15, self.sigmoid(np.dot(X, w))))
        # 计算梯度
        dw = (1 / len(Y)) * X.T.dot(y_hat - Y)
        return dw
    def loss(self, w, X, Y):
        y_hat = self.sigmoid(X.dot(w))
        # 计算损失函数
        loss = (-1 / len(Y)) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
        return loss
    def len(self):
        return self.length

class LRmodel_torch(torch.nn.Module):
    def __init__(self, in_features):
        super(LRmodel_torch, self).__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
