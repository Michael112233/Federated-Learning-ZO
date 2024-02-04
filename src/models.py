#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np


# logistic regression mnist
class LRmodel:
    def __init__(self, length, isSparse=False):
        self.length = length
        self.isSparse = isSparse

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, w, X, Y):
        y_hat = np.minimum(1 - 1e-15, np.maximum(1e-15, self.sigmoid(X.dot(w))))
        return y_hat

    def grad(self, w, X, Y):
        y_hat = self.predict(w, X, Y)
        # 计算梯度
        dw = (1 / len(Y)) * X.T.dot(y_hat - Y)
        return dw

    def loss(self, w, X, Y):
        y_hat = self.predict(w, X, Y)
        y_hat_bar = np.minimum(1 - 1e-15, np.maximum(1e-15, (1 - y_hat)))
        # 计算损失函数
        loss = (-1 / len(Y)) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(y_hat_bar))
        return loss

    def acc(self, w, X, Y):
        y_hat = self.predict(w, X, Y)
        y_hat = (y_hat >= 0.5) * 1
        corrent_array = y_hat - Y
        corrent_index = np.where(corrent_array == 0)
        accuracy = len(corrent_index[0]) / len(Y)
        return accuracy * 100

    def len(self):
        return self.length


class SVM:
    def __init__(self, length, isSparse=False):
        self.length = length
        self.isSparse = isSparse

    def predict(self, weight, x):
        y_hat = x.dot(weight)
        return y_hat

    def loss(self, weight, x, y):
        # lambda_val = 1e-4
        y_hat = self.predict(weight, x)
        loss = np.mean(np.maximum(0.0, 1 - y * y_hat) ** 2) / 2
        return loss

    def acc(self, weight, x, y):
        y_hat = self.predict(weight, x)
        y_hat = np.sign(y_hat)
        corrent_array = y_hat - y
        corrent_index = np.where(corrent_array == 0)
        accuracy = len(corrent_index[0]) / len(y)
        return accuracy * 100

    def len(self):
        return self.length