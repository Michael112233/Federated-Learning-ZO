#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import math
import random
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

from sampling import get_cifar10, get_rcv1, get_mnist, get_fashion_mnist
from options import args_parser
from algorithm import FedAvg_SGD, Zeroth_grad, FedAvg_GD, FedAvg_SIGNSGD, FedZO
from utils import eta_class, parameter, make_dir, excel_solver, select_eta

model_name = "svm" # logistic or svm
dataset_name = 'rcv'
algorithm_name = 'FedAvg_SGD' # zeroth_grad or FedAvg_SGD or FedAvg_GD or FedAvg_SignSGD
dir_mode = 1        # means "performance/experiment"

grad_option = 2
eta_list = eta_class()


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    eta = select_eta(algorithm_name, dataset_name, model_name)
    # initialize
    alpha = 0.5
    memory_length = 5
    if dataset_name == 'rcv':
        batch_size = 1000
    else:
        batch_size = 64
    verbose = True
    eta_type = eta_list.choose(grad_option)

    if dataset_name == 'rcv':
        dataset, X, Y, global_model = get_rcv1(model_name)
    elif dataset_name == 'cifar10':
        dataset, X, Y, global_model = get_cifar10()
    elif dataset_name == 'fashion_mnist':
        dataset, X, Y, global_model = get_fashion_mnist()
    else:
        dataset, X, Y, global_model = get_mnist(model_name)
    max_grad_time = 10000 * dataset.length()

    para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, batch_size, verbose)
    make_dir(dataset_name, algorithm_name, para, dir_mode)
    if algorithm_name == 'zeroth_grad':
        algorithm = Zeroth_grad(dataset, global_model, para)
    elif algorithm_name == 'FedAvg_GD':
        algorithm = FedAvg_GD(dataset, global_model, para)
    elif algorithm_name == 'FedAvg_SignSGD':
        algorithm = FedAvg_SIGNSGD(dataset, global_model, para)
    elif algorithm_name == 'FedAvg_SGD':
        algorithm = FedAvg_SGD(dataset, global_model, para)
    elif algorithm_name == 'FedZO':
        algorithm = FedZO(dataset, global_model, para)
    else:
        print("no found this algorithm")
        exit(0)
    filename = "../performance/experiment/{}/{}/eta={}/(time={}).csv".format(dataset_name, algorithm_name, eta, str(time.strftime('%Y-%m-%d-%H-%M-%S')))

    csv_solver = excel_solver(filename)
    print(filename)
    current_time, current_grad_times, current_loss, current_round = algorithm.alg_run(start_time)
    print("The loss is {} and The eta is {}".format(current_loss[-1], eta))
    csv_solver.save_excel(current_time, current_grad_times, current_loss, current_round)
    # end_time = time.time()
    # print("total time is {:.3f}".format(end_time-start_time))
    # print("total grad times is {:.2f}".format(total_grad))
