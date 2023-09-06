#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import math
import random
import time
import numpy as np
from sampling import iid_partition, get_rcv1, get_mnist
from options import args_parser
from algorithm import FedAvg, get_loss, Zeroth_grad
from utils import eta_class, parameter

dataset_name = 'mnist'
algorithm_name = 'zeroth_grad'

grad_option = 3
eta_list = eta_class()


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()

    eta = 1
    alpha = 0.35
    memory_length = 5
    batch_size = 64
    verbose = False
    # initialize
    eta_type = eta_list.choose(grad_option)

    if dataset_name == 'rcv':
        dataset, X, Y, global_model = get_rcv1()
    else:
        dataset, X, Y, global_model = get_mnist()

    max_grad_time = 500 * dataset.length()

    # for alpha in eta_chosen:
    para = parameter(eta_type, eta, batch_size, alpha, memory_length, verbose, max_grad_time)
    if algorithm_name == 'zeroth_grad':
        algorithm = Zeroth_grad(dataset, global_model, para)
    else:
        algorithm = FedAvg(dataset, global_model, para)

    loss = algorithm.alg_run(start_time)
    print("The loss is {} and The alpha is {}".format(loss, alpha))
    # end_time = time.time()
    # print("total time is {:.3f}".format(end_time-start_time))
    # print("total grad times is {:.2f}".format(total_grad))

