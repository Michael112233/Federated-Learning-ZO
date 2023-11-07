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
from utils import eta_class, parameter, make_dir, excel_solver

dataset_name = 'rcv'
algorithm_name = 'FedAvg' # zeroth_grad or FedAvg
dir_mode = 1        # means "performance/experiment"

grad_option = 2
eta_list = eta_class()


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()

    # initialize
    if dataset_name == 'rcv' and algorithm_name == 'zeroth_grad':
        eta = 25
    else:
        eta = 10  # if dataset_name == 'mnist'
    alpha = 0.5
    memory_length = 5
    batch_size = 1000
    verbose = True
    eta_type = eta_list.choose(grad_option)

    if dataset_name == 'rcv':
        dataset, X, Y, global_model = get_rcv1()
        print_iteration = 100
    else:
        dataset, X, Y, global_model = get_mnist()
        print_iteration = 50
    max_grad_time = 10000 * dataset.length()

    para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, print_iteration, verbose)
    make_dir(dataset_name, algorithm_name, para, dir_mode)
    if algorithm_name == 'zeroth_grad':
        algorithm = Zeroth_grad(dataset, global_model, para)
        filename = "../performance/experiment/{}/{}/eta={}/(time={}).csv".format(
            dataset_name, algorithm_name, eta, str(time.strftime('%Y-%m-%d-%H-%M-%S')))
    else:
        algorithm = FedAvg(dataset, global_model, para)
        filename = "../performance/experiment/{}/{}/eta={}/(time={}).csv".format(
            dataset_name, algorithm_name, eta, str(time.strftime('%Y-%m-%d-%H-%M-%S')))

    csv_solver = excel_solver(filename)
    print(filename)
    current_time, current_grad_times, current_loss, current_round = algorithm.alg_run(start_time)
    print("The loss is {} and The eta is {}".format(current_loss[-1], eta))
    csv_solver.save_excel(current_time, current_grad_times, current_loss, current_round)
    # end_time = time.time()
    # print("total time is {:.3f}".format(end_time-start_time))
    # print("total grad times is {:.2f}".format(total_grad))
