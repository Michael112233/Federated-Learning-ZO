import copy
import random
import time
import numpy as np
from sampling import iid_partition
from utils import end_info, excel_solver


def get_loss(global_model, dataset, weights, current_round, verbose):
    Xfull, Yfull = dataset.full()
    loss = global_model.loss(weights, Xfull, Yfull)
    if verbose:
        print("After iteration {}: loss is {}".format(current_round, loss))
    return loss


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
        self.radius = option.radius
        self.batch_size = option.batch_size
        self.verbose = option.verbose
        self.max_grad_time = option.max_grad_time
        self.excel_solver = excel_solver()
        self.print_iteration = option.print_iteration

        self.current_time = []
        self.current_grad_times = []
        self.current_loss = []
        self.current_round = []

    def save_info(self, start_time, current_weights, current_round):
        current_loss = get_loss(self.global_model, self.dataset, current_weights, current_round, self.verbose)
        current_time = time.time()
        self.current_time.append(copy.deepcopy(current_time - start_time))
        self.current_grad_times.append(self.total_grad)
        self.current_loss.append(current_loss)
        self.current_round.append(current_round)

    def update_client(self, current_weights, chosen_index, current_round=0):
        for i in range(self.local_iteration):
            X, Y = self.dataset.sample(chosen_index, self.batch_size)
            # calculate gradient
            # v_matrix = np.random.normal(loc=0, scale=1, size=(self.global_model.len(), 1))
            v_matrix = np.random.randn(self.global_model.len(), 1)
            upper_val = self.global_model.loss((current_weights + self.radius * v_matrix), X, Y)
            lower_val = self.global_model.loss((current_weights - self.radius * v_matrix), X, Y)
            # print(self.global_model.loss((weights), X, Y))
            g = (upper_val - lower_val) * (1 / (2 * self.radius)) * v_matrix
            # g = self.global_model.grad(weights, X, Y)
            self.total_grad += 2 * self.batch_size
            eta = self.grad_method(self.eta, current_round)
            current_weights -= eta * g
            if self.total_grad >= self.max_grad_time:
                break
        return current_weights

    def average(self, weights_list):
        sum_weight = sum(weights_list)
        length = len(weights_list)
        new_weights = sum(weights_list) / len(weights_list)
        # print(weights_list)
        return new_weights

    def alg_run(self, start_time):
        client_index = []
        weights = np.ones(self.global_model.len()).reshape(-1, 1)
        # 划分客户端训练集
        partition_index = iid_partition(self.dataset.length(), self.client_number)
        for i in range(self.client_number):
            client_index.append(i)

        self.save_info(start_time, weights, 0)
        # Training
        for i in range(self.iteration):
            # judge FEs >= maxFEs?
            weights_list = []
            # draw a client set
            chosen_client_num = int(max(self.client_rate * self.client_number, 1))
            chosen_client = random.sample(client_index, chosen_client_num)

            for k in chosen_client:
                weight_tmp = copy.deepcopy(weights)
                # weight_tmp = weights
                weight_of_client = self.update_client(weight_tmp, partition_index[k], i)
                weights_list.append(copy.deepcopy(weight_of_client))

            weights = self.average(weights_list)

            if self.total_grad >= self.max_grad_time or (i + 1) % self.print_iteration == 0:
                self.save_info(start_time, weights, i+1)
                # print("{} {}".format())
                if self.total_grad >= self.max_grad_time:
                    break

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
        self.print_iteration = option.print_iteration
        # self.excel_solver.create_excel()
        self.current_time = []
        self.current_grad_times = []
        self.current_loss = []
        self.current_round = []

    def save_info(self, start_time, current_weights, current_round):
        current_loss = get_loss(self.global_model, self.dataset, current_weights, current_round, self.verbose)
        current_time = time.time()
        self.current_time.append(copy.deepcopy(current_time - start_time))
        self.current_grad_times.append(self.total_grad)
        self.current_loss.append(current_loss)
        self.current_round.append(current_round)

    def update_client(self, current_weights, chosen_index, current_round):
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
            upper_val = self.global_model.loss((current_weights + self.radius * v_matrix), X, Y)
            lower_val = self.global_model.loss((current_weights - self.radius * v_matrix), X, Y)
            g = (upper_val - lower_val) * (1 / (2 * self.radius)) * v_matrix
            # 函数评估次数，需要每次乘上minibatch，在该算法中评估了两次，得出下式
            self.total_grad += 2 * self.batch_size
            eta = self.eta_type(self.eta, current_round, i)
            # print(eta, current_round)
            current_weights -= eta * g
            if self.total_grad >= self.max_grad_time:
                break

        return current_weights

    def average(self, weights_list):
        new_weights = sum(weights_list) / len(weights_list)
        self.this_weight = new_weights
        self.delta_weight_list.append(copy.deepcopy(self.this_weight - self.last_weight))
        self.last_weight = self.this_weight

        if len(self.delta_weight_list) % self.memory_length == 0:
            # generate delta_weight_list
            delta_list = np.array(self.delta_weight_list)
            delta_list = (delta_list.reshape((self.memory_length, self.global_model.len()))).T
            # initialize matrix P
            self.p_matrix, _ = np.linalg.qr(delta_list)
            self.delta_weight_list = []

        return new_weights

    def alg_run(self, start_time):
        client_index = []
        weights = np.ones(self.global_model.len()).reshape(-1, 1)
        self.save_info(start_time, weights, 0)
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
                weight_tmp = copy.deepcopy(weights)
                weight_of_client = self.update_client(weight_tmp, partition_index[k], i)
                weights_list.append(copy.deepcopy(weight_of_client))

            weights = self.average(weights_list)
            if self.total_grad >= self.max_grad_time or (i + 1) % self.print_iteration == 0:
                self.save_info(start_time, weights, i + 1)
                if self.total_grad >= self.max_grad_time:
                    break

        end_info(start_time, self.total_grad)
        return self.current_time, self.current_grad_times, self.current_loss, self.current_round
