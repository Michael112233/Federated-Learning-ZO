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
        np.where(z > 80, 80, z)
        np.where(z < -80, -80, z)
        return 1 / (1 + np.exp(-z))

    def predict(self, w, X, Y):
        if self.isSparse:
            y_hat = np.minimum(1 - 1e-15, np.maximum(1e-15, self.sigmoid(X.dot(w))))
        else:
            y_hat = np.minimum(1 - 1e-15, np.maximum(1e-15, self.sigmoid(np.dot(X, w))))
        return y_hat

    def grad(self, w, X, Y):
        y_hat = self.predict(w, X, Y)
        # 计算梯度
        dw = (1 / len(Y)) * np.dot(X.T, y_hat - Y)
        return dw

    def loss(self, w, X, Y):
        y_hat = self.predict(w, X, Y)
        # 计算损失函数
        loss = (-1 / len(Y)) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
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
