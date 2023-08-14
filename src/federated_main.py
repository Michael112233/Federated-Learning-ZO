#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import random
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sampling import mnist_dataset
from models import LRmodel
from tqdm import tqdm
import scipy.io as sio

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LRmodel
from utils import get_dataset, average_weights, exp_details

def update_client(weights):
    for i in range(local_iteration):
        X, Y = mnist.sample(batch_size=64)
        g = global_model.grad(weights, X, Y)
        weights -= eta * g
    return weights

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    # path_project = os.path.abspath('..')
    # logger = SummaryWriter('../logs')
    #
    # args = args_parser()
    # exp_details(args)
    #
    # if args.gpu:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'
    #
    # # load dataset and user groups
    # train_dataset, test_dataset, user_groups = get_dataset(args)
    #
    # # BUILD MODEL
    # if args.model == 'cnn':
    #     # Convolutional neural network
    #     if args.dataset == 'mnist':
    #         global_model = CNNMnist(args=args)
    #     elif args.dataset == 'fmnist':
    #         global_model = CNNFashion_Mnist(args=args)
    #     elif args.dataset == 'cifar':
    #         global_model = CNNCifar(args=args)
    #
    # elif args.model == 'mlp':
    #     # Multi-layer preceptron
    #     img_size = train_dataset[0][0].shape
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #         global_model = MLP(dim_in=len_in, dim_hidden=64,
    #                            dim_out=args.num_classes)
    # elif args.model == 'logistic': # add for logistic regression
    #     global_model = LRmodel(784, 10) # add for logistic regression
    # else:
    #     exit('Error: unrecognized model')
    mnist_data = sio.loadmat('/data/mnist/mnist.mat')
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
    iteration = 10000
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
