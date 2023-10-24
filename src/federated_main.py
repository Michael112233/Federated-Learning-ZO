#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import math
import random
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

from sampling import iid_partition, get_rcv1, get_mnist
from options import args_parser
from algorithm import FedAvg, Zeroth_grad
from utils import eta_class, parameter

dataset_name = 'rcv'
algorithm_name = 'zeroth_grad'


grad_option = 2
eta_list = eta_class()


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()

    # initialize
    eta = 2
    alpha = 0.5
    memory_length = 5
    batch_size = 1000
    verbose = True
    eta_type = eta_list.choose(grad_option)

    if dataset_name == 'rcv':
        dataset, X, Y, global_model = get_rcv1()
    else:
        dataset, X, Y, global_model = get_mnist()
    max_grad_time = 5000000 * dataset.length()

    para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, 100, verbose)
    if algorithm_name == 'zeroth_grad':
        algorithm = Zeroth_grad(dataset, global_model, para)
    else:
        algorithm = FedAvg(dataset, global_model, para)

    loss = algorithm.alg_run(start_time)
    print("The loss is {} and The eta is {}".format(loss, eta))
    # end_time = time.time()
    # print("total time is {:.3f}".format(end_time-start_time))
    # print("total grad times is {:.2f}".format(total_grad))
