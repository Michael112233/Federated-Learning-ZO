import copy
import random
import time
import numpy as np
from sampling import iid_partition
from utils import end_info, excel_solver

def get_loss(global_model, dataset, weights, current_round, losses, verbose):
    Xfull, Yfull = dataset.full()
    l = global_model.loss(weights, Xfull, Yfull)
    # acc = global_model.acc(weights, Xfull, Yfull)
    if verbose:
        # print("After iteration {}: loss is {} and accuracy is {:.2f}%".format(current_round, l, acc))
        print("After iteration {}: loss is {}".format(current_round, l))

    losses.append(l)
    return losses, l


class FedAvg:
    def __init__(self, dataset, global_model, option):
        self.client_rate = option.client_rate
        self.client_number = option.client_number
        self.local_iteration = option.local_iteration
        self.iteration = option.iteration
        self.dataset = dataset
        self.global_model = global_model
        self.total_grad = 0
        self.evaluate_time = 1
        self.grad_method = option.eta_type
        self.chosen_client_num = int(max(self.client_rate * self.client_number, 1))
        self.eta = option.eta
        self.batch_size = option.batch_size
        self.verbose = option.verbose
        self.max_grad_time = option.max_grad_time
        self.excel_solver = excel_solver()

        self.current_time = []
        self.current_grad_times = []
        self.current_loss = []
        self.current_round = []
    def update_client(self, weights, chosen_index, current_round=0):
        for i in range(self.local_iteration):
            X, Y = self.dataset.sample(chosen_index, self.batch_size)
            # print(X, Y)
            g = self.global_model.grad(weights, X, Y)
            self.total_grad += 1 * self.batch_size
            eta = self.grad_method(self.eta, current_round, i)
            # print(eta)
            weights -= eta * g
        return weights, self.total_grad

    def average(self, weights_list):
        weights = sum(weights_list) / self.chosen_client_num
        return weights

    def alg_run(self, start_time):
        client_index = []
        losses = []
        iter = []
        weights = np.ones(self.global_model.len()).reshape(-1, 1)
        # 划分客户端训练集
        partition_index = iid_partition(self.dataset.length(), self.client_number)
        for i in range(self.client_number):
            client_index.append(i)
        # print(client_index)

        # Training
        for i in range(self.iteration):
            weights_list = []
            chosen_client_num = int(max(self.client_rate * self.client_number, 1))
            chosen_client = random.sample(client_index, chosen_client_num)
            # train
            for k in chosen_client:
                weight_tmp = weights
                weight_of_client, self.total_grad = self.update_client(weight_tmp, partition_index[k], i)
                weights_list.append(copy.deepcopy(weight_of_client))

            weights = self.average(weights_list)

            current_time = time.time()

            if (i + 1) % 100 == 0:
                iter.append(i + 1)
                losses, current_loss = get_loss(self.global_model, self.dataset, weights, i + 1, losses, self.verbose)
                self.current_time.append(copy.deepcopy(current_time))
                self.current_grad_times.append(self.total_grad)
                self.current_loss.append(current_loss)
                self.current_round.append(i + 1)
                if self.total_grad >= self.max_grad_time:
                    break

        # self.excel_solver.save_excel(self.current_time, self.current_grad_times, self.current_loss, self.current_round)
        end_info(start_time, self.total_grad)
        return self.current_time, self.current_grad_times, self.current_loss, self.current_round

class Zeroth_grad:
    def __init__(self, dataset, global_model, option):
        self.dataset = dataset
        self.global_model = global_model
        self.client_rate = option.client_rate
        self.client_number = option.client_number
        self.local_iteration = option.local_iteration
        self.iteration = option.iteration
        self.radius = option.radius
        self.total_grad = 0
        self.evaluate_time = 2
        self.delta_weight_list = []
        self.chosen_client_num = int(max(self.client_rate * self.client_number, 1))
        self.p_matrix = np.empty((global_model.len(), option.memory_length))
        self.this_weight = np.ones(global_model.len()).reshape(-1, 1)
        self.last_weight = self.this_weight
        self.eta = option.eta
        self.alpha = option.alpha
        self.eta_type = option.eta_type
        self.batch_size = option.batch_size
        self.memory_length = option.memory_length
        self.verbose = option.verbose
        self.max_grad_time = option.max_grad_time
        self.excel_solver = excel_solver()
        # self.excel_solver.create_excel()
        self.current_time = []
        self.current_grad_times = []
        self.current_loss = []
        self.current_round = []

    def update_client(self, weights, chosen_index, current_round):
        for i in range(self.local_iteration):
            X, Y = self.dataset.sample(chosen_index, self.batch_size)
            # get matrix V（加一的目的是想让服务器生成完P矩阵后通过else的方法生成v矩阵）
            if current_round <= self.memory_length:
                v_matrix = np.random.randn(self.global_model.len(), 1)
            else:
                z0 = np.random.randn(self.global_model.len(), 1)
                z1 = np.random.randn(self.memory_length, 1)
                v_matrix = np.sqrt(1 - self.alpha) * z0 + np.sqrt(
                    self.alpha * self.global_model.len() / self.memory_length) * self.p_matrix.dot(
                    z1)
            # calculate gradient
            upper_val = self.global_model.loss((weights + self.radius * v_matrix), X, Y)
            lower_val = self.global_model.loss((weights - self.radius * v_matrix), X, Y)
            g = (upper_val - lower_val) * (1 / (2 * self.radius)) * v_matrix
            # 函数评估次数，需要每次乘上minibatch，在该算法中评估了两次，得出下式
            self.total_grad += 2 * self.batch_size
            eta = self.eta_type(self.eta, current_round, i)
            weights -= eta * g
        return weights, self.total_grad

    def average(self, weights_list):
        weights = sum(weights_list) / self.chosen_client_num
        self.this_weight = weights
        self.delta_weight_list.append(copy.deepcopy(self.this_weight - self.last_weight))
        # print(self.delta_weight_list.shape)
        self.last_weight = self.this_weight

        if len(self.delta_weight_list) % self.memory_length == 0:
            # generate delta_weight_list
            delta_list = np.array(self.delta_weight_list)
            delta_list = (delta_list.reshape((self.memory_length, self.global_model.len()))).T
            # initialize matrix P
            self.p_matrix, _ = np.linalg.qr(delta_list)
            self.delta_weight_list = []

        return weights

    def alg_run(self, start_time):
        client_index = []
        losses = []
        iter = []
        weights = np.ones(self.global_model.len()).reshape(-1, 1)
        # 划分客户端训练集
        partition_index = iid_partition(self.dataset.length(), self.client_number)
        for i in range(self.client_number):
            client_index.append(i)
        # print(client_index)

        # Training
        for i in range(self.iteration):
            weights_list = []
            chosen_client_num = int(max(self.client_rate * self.client_number, 1))
            chosen_client = random.sample(client_index, chosen_client_num)
            # train
            for k in chosen_client:
                weight_tmp = weights
                weight_of_client, self.total_grad = self.update_client(weight_tmp, partition_index[k], i)
                weights_list.append(copy.deepcopy(weight_of_client))

            weights = self.average(weights_list)
            current_time = time.time()

            current_time = time.time()
            if (i + 1) % 100 == 0:
                iter.append(i + 1)
                losses, current_loss = get_loss(self.global_model, self.dataset, weights, i + 1, losses, self.verbose)
                self.current_time.append(copy.deepcopy(current_time))
                self.current_grad_times.append(self.total_grad)
                self.current_loss.append(current_loss)
                self.current_round.append(i + 1)
                if self.total_grad >= self.max_grad_time:
                    break

        # self.excel_solver.save_excel(self.current_time, self.current_grad_times, self.current_loss, self.current_round)

        end_info(start_time, self.total_grad)
        return self.current_time, self.current_grad_times, self.current_loss, self.current_round

class FedNewton:
    def __init__(self, dataset, global_model, option):
        self.client_rate = option.client_rate
        self.client_number = option.client_number
        self.local_iteration = option.local_iteration
        self.iteration = option.iteration
        self.dataset = dataset
        self.global_model = global_model
        self.total_grad = 0
        self.evaluate_time = 1
        self.grad_method = option.eta_type
        self.chosen_client_num = int(max(self.client_rate * self.client_number, 1))
        self.eta = option.eta
        self.batch_size = option.batch_size
        self.verbose = option.verbose
        self.max_grad_time = option.max_grad_time
        self.excel_solver = excel_solver()

        self.current_time = []
        self.current_grad_times = []
        self.current_loss = []
        self.current_round = []
    def update_client(self, weights, chosen_index, current_round=0):
        for i in range(self.local_iteration):
            individual_model = self.global_model
            X, Y = self.dataset.sample(chosen_index, self.batch_size)
            individual_model.fit(X, Y, )
            self.total_grad += 1 * self.batch_size
        return weights, self.total_grad

    def average(self, weights_list):
        weights = sum(weights_list) / self.chosen_client_num
        return weights

    def alg_run(self, start_time):
        client_index = []
        losses = []
        iter = []
        weights = np.ones(self.global_model.len()).reshape(-1, 1)
        # 划分客户端训练集
        partition_index = iid_partition(self.dataset.length(), self.client_number)
        for i in range(self.client_number):
            client_index.append(i)
        # print(client_index)

        # Training
        for i in range(self.iteration):
            weights_list = []
            chosen_client_num = int(max(self.client_rate * self.client_number, 1))
            chosen_client = random.sample(client_index, chosen_client_num)
            # train
            for k in chosen_client:
                weight_tmp = weights
                weight_of_client, self.total_grad = self.update_client(weight_tmp, partition_index[k], i)
                weights_list.append(copy.deepcopy(weight_of_client))

            weights = self.average(weights_list)

            current_time = time.time()

            if (i + 1) % 100 == 0:
                iter.append(i + 1)
                losses, current_loss = get_loss(self.global_model, self.dataset, weights, i + 1, losses, self.verbose)
                self.current_time.append(copy.deepcopy(current_time))
                self.current_grad_times.append(self.total_grad)
                self.current_loss.append(current_loss)
                self.current_round.append(i + 1)
                if self.total_grad >= self.max_grad_time:
                    break

        self.excel_solver.save_excel(self.current_time, self.current_grad_times, self.current_loss, self.current_round)
        end_info(start_time, self.total_grad)
        return losses[-1]
