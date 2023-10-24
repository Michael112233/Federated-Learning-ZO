import time
import os
import copy

import numpy as np
import pandas as pd

from multiprocessing import Pool, freeze_support
from sampling import get_rcv1, get_mnist
from algorithm import FedAvg, Zeroth_grad
from utils import excel_solver, parameter, eta_class, mkdir

current_dataset_list = []
current_algorithm_list = []
current_best_eta_list = []
current_best_loss_list = []

def make_dir(dataset_name, algorithm_name, params):
    eta = params.eta
    alpha = params.alpha
    memory_length = params.memory_length

    mkdir("../performance")
    mkdir("../performance/params")

    mkdir("../performance/params/{}/{}".format(dataset_name, algorithm_name))
    mkdir("../performance/params/{}/{}/eta={}".format(dataset_name, algorithm_name, eta))
    if algorithm_name == "zeroth":
        # print(alpha)
        mkdir("../performance/params/{}/{}/eta={}/alpha={:.2}".format(dataset_name, algorithm_name, eta, alpha))
        mkdir(
            "../performance/params/{}/{}/eta={}/alpha={:.2}/memory_length={}".format(dataset_name, algorithm_name, eta,
                                                                                     alpha, memory_length))


def get_result(filename, algorithm):
    csv_solver = excel_solver(filename)
    start_time = time.time()
    current_time, current_grad_times, current_loss, current_round = algorithm.alg_run(start_time)
    # print("{}\n".format(filename))
    csv_solver.save_excel(current_time, current_grad_times, current_loss, current_round)


eta_list = [0.001, 0.01, 0.1, 1, 10, 20]
alpha = 0.5
dataset_list = ['mnist', 'rcv']
algorithm_list = ['FedAvg', 'zeroth']
memory_length = 5
times_list = range(1, 4)
verbose = True
batch_size = 1000

eta_choose = eta_class()
eta_type = eta_choose.choose(2)


def generate_csv(dataset_name, algorithm_name, eta, times):
    if dataset_name == 'rcv':
        dataset, X, Y, global_model = get_rcv1()
    else:
        dataset, X, Y, global_model = get_mnist()
    max_grad_time = 300 * dataset.length()

    # for algorithm
    if algorithm_name == 'FedAvg':
        filename = "../performance/params/{}/{}/eta={}/({}).csv".format(
            dataset_name, algorithm_name, eta, times)
        print(filename)
        if dataset_name == "mnist":
            para = parameter(max_grad_time, eta_type, eta, 0, 0, 1000, 10, verbose)
        else:
            para = parameter(max_grad_time, eta_type, eta, 0, 0, 1000, 100, verbose)
        make_dir(dataset_name, algorithm_name, para)
        algorithm = FedAvg(dataset, global_model, para)
        get_result(filename, algorithm)
    elif algorithm_name == 'zeroth':
        filename = "../performance/params/{}/{}/eta={}/alpha={:.2}/memory_length={}/({}).csv".format(
            dataset_name,
            algorithm_name, eta, alpha,
            memory_length,
            times)
        if dataset_name == "mnist":
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, 10,
                             verbose)
        else:
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, 100,
                             verbose)
        make_dir(dataset_name, algorithm_name, para)
        print(filename)
        algorithm = Zeroth_grad(dataset, global_model, para)
        get_result(filename, algorithm)


def get_params():
    start_time = time.time()
    combine = []
    for dataset_name in dataset_list:
        for algorithm_name in algorithm_list:
            for eta in eta_list:
                for times in times_list:
                    combine.append([dataset_name, algorithm_name, eta, times])
    with Pool(10) as p:
        # ans = p.starmap(generate_csv, zip(dataset_list, algorithm_list, eta_list, times_list))
        ans = p.starmap(generate_csv, combine)
        print(ans)

    end_time = time.time()
    print("Time is {}s".format(end_time-start_time))
    # for dataset_name in dataset_list:
    #     mkdir("../performance/params/{}".format(dataset_name))
    #     for algorithm_name in algorithm_list:
    #         for eta in eta_list:
    #             # for dataset
    #             if dataset_name == 'rcv':
    #                 dataset, X, Y, global_model = get_rcv1()
    #             else:
    #                 dataset, X, Y, global_model = get_mnist()
    #             max_grad_time = 300 * dataset.length()
    #
    #             # for algorithm
    #             if algorithm_name == 'FedAvg':
    #                 for i in range(times):
    #                     filename = "../performance/params/{}/{}/eta={}/({}).csv".format(
    #                         dataset_name, algorithm_name, eta, i + 1)
    #                     print(filename)
    #                     if dataset_name == "mnist":
    #                         para = parameter(max_grad_time, eta_type, eta, 0, 0, 1000, 10, verbose)
    #                     else:
    #                         para = parameter(max_grad_time, eta_type, eta, 0, 0, 1000, 100, verbose)
    #                     make_dir(dataset_name, algorithm_name, para)
    #                     algorithm = FedAvg(dataset, global_model, para)
    #                     get_result(filename, algorithm)
    #             elif algorithm_name == 'zeroth':
    #
    #                 for i in range(times):
    #                     filename = "../performance/params/{}/{}/eta={}/alpha={:.2}/memory_length={}/({}).csv".format(
    #                         dataset_name,
    #                         algorithm_name, eta, alpha,
    #                         memory_length,
    #                         i + 1)
    #                     if dataset_name == "mnist":
    #                         para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, 10,
    #                                          verbose)
    #                     else:
    #                         para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, 100,
    #                                          verbose)
    #                     make_dir(dataset_name, algorithm_name, para)
    #                     print(filename)
    #                     algorithm = Zeroth_grad(dataset, global_model, para)
    #                     get_result(filename, algorithm)


def summary_csv():
    print("----")
    for dataset_name in dataset_list:
        for algorithm_name in algorithm_list:
            best_loss = 100
            best_eta = -1
            for eta in eta_list:
                if eta == 20:
                    if not (dataset_name == "rcv" and algorithm_name == "zeroth"):
                        continue
                g = os.walk(r"../performance/params/{}/{}/eta={}".format(dataset_name,
                                                                         algorithm_name, eta))
                current_loss_list = []
                for path, dir_list, file_list in g:
                    for file_name in file_list:
                        csv_path = (os.path.join(path, file_name))
                        csv_file = pd.read_csv(csv_path)
                        current_loss_column = np.array(csv_file["current_loss"])
                        current_loss_list.append(current_loss_column[-1])
                sorted_list = np.sort(current_loss_list)
                loss = sorted_list[max(times_list) // 2]

                if best_loss > loss:
                    best_loss = loss
                    best_eta = eta

            print("{} {} eta={} loss is {}\n".format(dataset_name, algorithm_name, best_eta, best_loss))
            current_best_eta_list.append(copy.deepcopy(best_eta))
            current_best_loss_list.append(copy.deepcopy(best_loss))
            current_dataset_name_list.append(copy.deepcopy(dataset_name))
            current_algorithm_name_list.append(copy.deepcopy(algorithm))

if __name__ == '__main__':
    freeze_support()
    get_params()
    summary_csv()
    sum_up_param()
# generate_csv()
