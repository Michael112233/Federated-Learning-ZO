#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import random
import time
import numpy as np
from sampling import mnist_dataset
from models import LRmodel
import scipy.io as sio

def update_client(weights):
    for i in range(local_iteration):
        X, Y = mnist.sample(batch_size=64)
        g = global_model.grad(weights, X, Y)
        weights -= eta * g
    return weights

if __name__ == '__main__':
    start_time = time.time()
    mnist_data = sio.loadmat('data/mnist/mnist.mat')
    mnist = mnist_dataset(database=mnist_data)
    global_model = LRmodel()
    client_rate = 0.5
    client_number = 10
    client_index = []
    local_iteration = 10
    for i in range(client_number):
        client_index.append(i)
    print(client_index)

    # Training
    iteration = 100
    eta = 2
    losses = []
    iter = []
    weights = np.ones(mnist.X_train.shape[1]).reshape(-1, 1)

    for i in range(iteration):
        weights_list = []
        for j in range(client_number):
            weights_list.append(weights)
        chosen_client_num = int(max(client_rate * client_number, 1))
        chosen_client = random.sample(client_index, chosen_client_num)

        # train
        for k in chosen_client:
            weights_list[k] = update_client(weights_list[k])
        weights = sum(weights_list) / client_number

        if (i + 1) % 1 == 0:
            Xfull, Yfull = mnist.full()
            l = global_model.loss(weights, Xfull, Yfull)
            iter.append(i + 1)
            losses.append(l)
            print(f"Loss after iteration {i + 1}: {l}")

