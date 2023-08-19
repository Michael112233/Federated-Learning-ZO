#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import random
import time
import numpy as np
from sampling import mnist_dataset, csr_dataset
from models import LRmodel, LRmodel_csr
from sklearn.datasets import load_svmlight_file
import scipy.io as sio

from options import args_parser


def update_client(weights):
    for i in range(local_iteration):
        X, Y = dataset.sample(batch_size=64)
        g = global_model.grad(weights, X, Y)
        weights -= eta * g
    return weights

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()

    if args.dataset == 'mnist':
        mnist_data = sio.loadmat('data/mnist/mnist.mat')
        dataset = mnist_dataset(database=mnist_data)
        global_model = LRmodel()
    elif args.dataset == 'rcv':
        X, Y = load_svmlight_file('data/rcv/rcv1_test.binary')
        Y = Y.reshape(-1, 1)
        Y = (Y + 1) / 2
        global_model = LRmodel_csr()
        dataset = csr_dataset(X, Y)

    client_rate = 0.5
    client_number = 10
    client_index = []
    local_iteration = 10
    for i in range(client_number):
        client_index.append(i)
    # print(client_index)

    # Training
    iteration = 1000
    eta = 2
    losses = []
    iter = []
    weights = np.ones(dataset.X_train.shape[1]).reshape(-1, 1)

    for i in range(iteration):
        weights_list = []
        chosen_client_num = int(max(client_rate * client_number, 1))
        chosen_client = random.sample(client_index, chosen_client_num)

        # train
        for k in chosen_client:
            weights_of_client = update_client(weights)
            weights_list.append(weights_of_client)
        weights = sum(weights_list) / chosen_client_num

        if (i + 1) % 50 == 0:
            Xfull, Yfull = dataset.full()
            l = global_model.loss(weights, Xfull, Yfull)
            # acc = global_model.acc(weights, Xfull, Yfull)
            iter.append(i + 1)
            losses.append(l)
            # print("After iteration {}: loss is {} and accuracy is {:.2f}%".format(i+1, l, acc))
            print("After iteration {}: loss is {}".format(i + 1, l))
