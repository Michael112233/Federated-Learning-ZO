#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import random
import time
import numpy as np
from sampling import iid_partition, get_rcv1, get_mnist
from options import args_parser
from algorithm import client_number, iteration, client_rate, FedAvg, get_loss, Zeroth_grad

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()

    dataset_name = 'mnist'
    algorithm_name = 'zeroth_grad'

    # initialize
    if dataset_name == 'rcv':
        dataset, X, Y, global_model = get_rcv1()
    else:
        dataset, X, Y, global_model = get_mnist()

    if algorithm_name == 'zeroth_grad':
        algorithm = Zeroth_grad(dataset, global_model)
    else:
        algorithm = FedAvg(dataset, global_model)

    client_index = []
    client_dataset = {}
    losses = []
    iter = []
    weights = np.ones(global_model.len()).reshape(-1, 1)
    total_grad = 0
    # 划分客户端训练集
    partition_index = iid_partition(dataset.length(), client_number)
    for i in range(client_number):
        client_index.append(i)
    # print(client_index)

    # Training
    for i in range(iteration):
        weights_list = []
        chosen_client_num = int(max(client_rate * client_number, 1))
        chosen_client = random.sample(client_index, chosen_client_num)

        # print(i)
        # train
        for k in chosen_client:
            weight_of_client, total_grad = algorithm.update_client(weights, partition_index[k], i)
            weights_list.append(copy.deepcopy(weight_of_client))

        weights = algorithm.average(weights_list)

        if (i + 1) % 100 == 0:
            iter.append(i + 1)
            losses = get_loss(global_model, dataset, weights, i + 1, losses)

    end_time = time.time()
    print("total time is {:.3f}".format(end_time-start_time))
    print("total grad times is {:.2f}".format(total_grad))

