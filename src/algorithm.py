import copy
import random
import time
import numpy as np
from sampling import iid_partition
from utils import end_info, excel_solver, judge_whether_print


def get_loss(global_model, dataset, weights, current_round, verbose):
    Xfull, Yfull = dataset.full()
    loss = global_model.loss(weights, Xfull, Yfull)
    accuracy = global_model.acc(weights, Xfull, Yfull)
    if verbose:
        print("After iteration {}: loss is {:.2f}, accuracy is {:.2f}%".format(current_round, loss, accuracy))
    return loss

class FedAvg_GD:
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
            X = self.dataset.X[chosen_index]
            Y = self.dataset.Y[chosen_index]
            # calculate gradient
            v_matrix = np.random.randn(self.global_model.len(), 1)
            upper_val = self.global_model.loss((current_weights + self.radius * v_matrix), X, Y)
            lower_val = self.global_model.loss((current_weights - self.radius * v_matrix), X, Y)
            # print(self.global_model.loss((weights), X, Y))
            g = (upper_val - lower_val) * (1 / (2 * self.radius)) * v_matrix
            # g = self.global_model.grad(weights, X, Y)
            self.total_grad += 2 * len(chosen_index)
            eta = self.grad_method(self.eta, current_round)
            current_weights -= eta * g
            if self.total_grad >= self.max_grad_time:
                break
        return current_weights

    def average(self, weights_list):
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

            if self.total_grad >= self.max_grad_time or judge_whether_print(i + 1) == True:
                self.save_info(start_time, weights, i+1)
                # print("{} {}".format())
                if self.total_grad >= self.max_grad_time:
                    break

        end_info(start_time, self.total_grad)
        return self.current_time, self.current_grad_times, self.current_loss, self.current_round









class FedAvg_SIGNSGD:
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
        self.client_weight = [] # 用来存放每个客户端的模型
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

    # 向所有客户端传递初始模型
    def initial_client(self, initial_weights):
        for i in range(self.client_number):
            self.client_weight.append(initial_weights)

    # 这里只是训练客户端，还没有更新客户端模型
    def train_client(self, k, chosen_index, current_round=0):
        weights = copy.deepcopy(self.client_weight[k])
        for i in range(self.local_iteration):
            X, Y = self.dataset.sample(chosen_index, self.batch_size)
            # calculate gradient
            # v_matrix = np.random.normal(loc=0, scale=1, size=(self.global_model.len(), 1))
            v_matrix = np.random.randn(self.global_model.len(), 1)
            upper_val = self.global_model.loss((weights + self.radius * v_matrix), X, Y)
            lower_val = self.global_model.loss((weights - self.radius * v_matrix), X, Y)
            g = (upper_val - lower_val) * (1 / (2 * self.radius)) * v_matrix
            self.total_grad += 2 * self.batch_size
            eta = self.grad_method(self.eta, current_round)
            weights -= eta * g
            if self.total_grad >= self.max_grad_time:
                break
        return np.sign(self.client_weight[k]-weights)

    def update_client(self, k, sum_sign, current_round):
        eta = self.grad_method(self.eta, current_round)
        self.client_weight[k] -= eta * sum_sign

    def sign(self, sign_list):
        sum_list = sum(sign_list)
        return np.sign(sum_list)

    def alg_run(self, start_time):
        client_index = []
        weights = np.ones(self.global_model.len()).reshape(-1, 1)
        # 划分客户端训练集
        partition_index = iid_partition(self.dataset.length(), self.client_number)
        for i in range(self.client_number):
            client_index.append(i)

        self.save_info(start_time, weights, 0)
        # 向所有客户端传递初始模型
        self.initial_client(weights)
        # Training
        for i in range(self.iteration):
            # 所有客户端上传符号而不是模型
            sign_list = []
            # 全部客户端都参与训练
            for k in client_index:
                sign_g = self.train_client(k, partition_index[k], i)
                sign_list.append(sign_g)

            sum_sign = self.sign(sign_list)
            eta = self.grad_method(self.eta, i)
            weights -= eta * sum_sign

            for k in client_index:
                self.update_client(k, sum_sign, i)

            if self.total_grad >= self.max_grad_time or judge_whether_print(i + 1) == True:
                self.save_info(start_time, weights, i+1)
                # print("{} {}".format())
                if self.total_grad >= self.max_grad_time:
                    break

        end_info(start_time, self.total_grad)
        return self.current_time, self.current_grad_times, self.current_loss, self.current_round










class FedZO:
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

    # 从单位球S ^ d中均匀抽样d维随机方向
    def uniform_random_direction(self,d):
        # 生成一个d维随机向量，其中每个分量都是从均匀分布[-1, 1]中随机抽样的
        random_vector = np.random.uniform(low=-1, high=1, size=d)

        # 将向量归一化为单位向量
        normalized_vector = random_vector
        # normalized_vector = random_vector / np.linalg.norm(random_vector)
        normalized_vector = normalized_vector.reshape(-1, 1)
        return normalized_vector

    def update_client(self, current_weights, chosen_index, current_round=0):
        weights = copy.deepcopy(current_weights)
        for i in range(self.local_iteration):
            X, Y = self.dataset.sample(chosen_index, self.batch_size)
            v_matrix = self.uniform_random_direction(self.global_model.len())
            upper_val = self.global_model.loss((weights + self.radius * v_matrix), X, Y)
            lower_val = self.global_model.loss(weights, X, Y)
            # print(self.global_model.loss((weights), X, Y))
            g = (upper_val - lower_val) * (1 / self.radius) * v_matrix
            # g = self.global_model.grad(weights, X, Y)
            self.total_grad += 2 * self.batch_size
            eta = self.grad_method(self.eta, current_round)
            g = g.reshape(-1, 1)
            weights -= eta * g
            if self.total_grad >= self.max_grad_time:
                break
        return weights - current_weights

    def average(self, weights_list):
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
            delta_weights_list = []
            # draw a client set
            chosen_client_num = int(max(self.client_rate * self.client_number, 1))
            chosen_client = random.sample(client_index, chosen_client_num)

            for k in chosen_client:
                weight_tmp = copy.deepcopy(weights)
                # weight_tmp = weights
                delta_weights = self.update_client(weight_tmp, partition_index[k], i)
                delta_weights_list.append(copy.deepcopy(delta_weights))

            weights += self.average(delta_weights_list)
            if self.total_grad >= self.max_grad_time or judge_whether_print(i + 1) == True:
                self.save_info(start_time, weights, i+1)
                # print("{} {}".format())
                if self.total_grad >= self.max_grad_time:
                    break

        end_info(start_time, self.total_grad)
        return self.current_time, self.current_grad_times, self.current_loss, self.current_round





class FedAvg_SGD:
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
            up = (current_weights + self.radius * v_matrix)
            down = (current_weights - self.radius * v_matrix)
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

            if self.total_grad >= self.max_grad_time or judge_whether_print(i + 1) == True:
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
            if self.total_grad >= self.max_grad_time or judge_whether_print(i + 1) == True:
                self.save_info(start_time, weights, i + 1)
                if self.total_grad >= self.max_grad_time:
                    break

        end_info(start_time, self.total_grad)
        return self.current_time, self.current_grad_times, self.current_loss, self.current_round
