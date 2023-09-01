import copy
import random

import numpy as np

client_rate = 0.5
client_number = 50
local_iteration = 20
total_grad = 0
iteration = 2000
batch_size = 64


def get_loss(global_model, dataset, weights, current_round, losses):
    Xfull, Yfull = dataset.full()
    l = global_model.loss(weights, Xfull, Yfull)
    # acc = global_model.acc(weights, Xfull, Yfull)
    # print("After iteration {}: loss is {} and accuracy is {:.2f}%".format(current_round, l, acc))
    print("After iteration {}: loss is {}".format(current_round, l))

    losses.append(l)
    return losses


class FedAvg:
    def __init__(self, dataset, global_model, eta=0.1):
        self.dataset = dataset
        self.global_model = global_model
        self.total_grad = 0
        self.evaluate_time = 1
        self.chosen_client_num = int(max(client_rate * client_number, 1))
        self.eta = eta

    def update_client(self, weights, chosen_index, current_round=0):
        for i in range(local_iteration):
            X, Y = self.dataset.sample(chosen_index, batch_size)
            g = self.global_model.grad(weights, X, Y)
            self.total_grad += self.evaluate_time * batch_size
            weights -= self.eta * g
        return weights, self.total_grad

    def average(self, weights_list):
        weights = sum(weights_list) / self.chosen_client_num
        return weights


radius = 1e-6
memory_length = 5


class Zeroth_grad:
    def __init__(self, dataset, global_model, eta=0.1, alpha=0.5):
        self.dataset = dataset
        self.global_model = global_model
        self.total_grad = 0
        self.evaluate_time = 2
        self.delta_weight_list = []
        self.chosen_client_num = int(max(client_rate * client_number, 1))
        self.p_matrix = np.empty((global_model.len(), memory_length))
        self.this_weight = np.ones(global_model.len()).reshape(-1, 1)
        self.last_weight = self.this_weight
        self.eta = eta
        self.alpha = alpha

    def update_client(self, weights, chosen_index, current_round):
        for i in range(local_iteration):
            X, Y = self.dataset.sample(chosen_index, batch_size)
            # get matrix V（加一的目的是想让服务器生成完P矩阵后通过else的方法生成v矩阵）
            if current_round <= memory_length:
                v_matrix = np.random.randn(self.global_model.len(), 1)
            else:
                z0 = np.random.randn(self.global_model.len(), 1)
                z1 = np.random.randn(memory_length, 1)
                v_matrix = np.sqrt(1 - self.alpha) * z0 + np.sqrt(
                    self.alpha * self.global_model.len() / memory_length) * self.p_matrix.dot(
                    z1)
            # calculate gradient
            upper_val = self.global_model.loss((weights + radius * v_matrix), X, Y)
            lower_val = self.global_model.loss((weights - radius * v_matrix), X, Y)
            g = (upper_val - lower_val) * (1 / (2 * radius)) * v_matrix
            # 函数评估次数，需要每次乘上minibatch，在该算法中评估了两次，得出下式
            self.total_grad += self.evaluate_time * batch_size
            # gradient descent
            weights -= self.eta * g
        return weights, self.total_grad

    def average(self, weights_list):
        weights = sum(weights_list) / self.chosen_client_num
        self.this_weight = weights
        self.delta_weight_list.append(copy.deepcopy(self.this_weight - self.last_weight))
        # print(self.delta_weight_list.shape)
        self.last_weight = self.this_weight

        if len(self.delta_weight_list) % memory_length == 0:
            # generate delta_weight_list
            delta_list = np.array(self.delta_weight_list)
            delta_list = (delta_list.reshape((memory_length, self.global_model.len()))).T
            # initialize matrix P
            self.p_matrix, _ = np.linalg.qr(delta_list)
            self.delta_weight_list = []

        return weights
