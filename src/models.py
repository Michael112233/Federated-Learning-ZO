#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# logistic regression mnist
class LRmodel:
    def __init__(self, length, isSparse=False):
        self.length = length
        self.isSparse = isSparse

    def modelName(self):
        return 'logistic'

    def predict(self, w, X, Y):
        y_hat = np.minimum(1 - 1e-15, np.maximum(1e-15, sigmoid(X.dot(w))))
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

    def modelName(self):
        return 'svm'

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

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input = input_dim
        self.hidden = hidden_dim
        self.output = output_dim

    def modelName(self):
        return 'NeuralNetwork'

    # transform weight vector into two matrix
    def weight_transform(self, weight):
        weight1_tmp = weight[0: self.input * self.hidden]
        weight2_tmp = weight[self.input * self.hidden:]
        weight1 = weight1_tmp.reshape((self.input, self.hidden))
        weight2 = weight2_tmp.reshape((self.hidden, self.output))
        return weight1, weight2

    def predict(self, w, x, y):
        weight1, weight2 = self.weight_transform(w)
        hidden_input = x.dot(weight1)
        hidden_output = sigmoid(hidden_input)
        output_input = hidden_output.dot(weight2)
        output_output = sigmoid(output_input)
        return output_output

    def loss(self, w, x, y):
        y_pred = self.predict(w, x, y)
        y_pred_bar = np.maximum(1e-15, (1 - y_pred))
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(y_pred_bar))

    def acc(self, w, x, y):
        y_pred = self.predict(w, x, y)
        y_hat = (y_pred > 0.5) * 1
        minus = y - y_hat
        same_label = np.where(minus == 0)
        return len(same_label[0]) / len(y) * 100

    def len(self):
        return (self.input + self.output) * self.hidden

