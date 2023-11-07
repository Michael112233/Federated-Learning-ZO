import time
import os
import copy

import numpy as np
import pandas as pd

from multiprocessing import Pool, freeze_support
from sampling import get_rcv1, get_mnist
from algorithm import FedAvg_SGD, Zeroth_grad
from utils import excel_solver, parameter, eta_class, mkdir, make_dir

current_dataset_name_list = []
current_algorithm_name_list = []
current_best_eta_list = []
current_best_loss_list = []

dir_mode = 0       # means "performance/params"


def get_result(filename, algorithm):
    csv_solver = excel_solver(filename)
    start_time = time.time()
    current_time, current_grad_times, current_loss, current_round = algorithm.alg_run(start_time)
    # print("{}\n".format(filename))
    csv_solver.save_excel(current_time, current_grad_times, current_loss, current_round)


eta_list = [0.01, 0.1, 1, 10, 20, 25, 30, 40]
# eta_list = [1]
alpha = 0.5
dataset_list = ['rcv', 'mnist']
algorithm_list = ['FedAvg', 'zeroth']
memory_length = 5
times_list = range(1, 4)
verbose = True
batch_size = 1000

max_grad_time_rcv = 162578000
max_grad_time_mnist = 14400000
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
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, 10, verbose)
        else:
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, 100, verbose)
        make_dir(dataset_name, algorithm_name, para, dir_mode)
        algorithm = FedAvg_SGD(dataset, global_model, para)
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
        make_dir(dataset_name, algorithm_name, para, dir_mode)
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
    print("Time is {}s".format(end_time - start_time))


def summary_csv():
    print("----")
    for dataset_name in dataset_list:
        for algorithm_name in algorithm_list:
            best_loss = 100
            best_eta = -1
            for eta in eta_list:
                if algorithm_name == "FedAvg":
                    g = os.walk(r"../performance/params/{}/{}/eta={}".format(dataset_name,
                                                                         algorithm_name, eta))
                else:
                    g = os.walk(r"../performance/params/{}/{}/eta={}/alpha={}/memory_length={}".format(dataset_name,
                                                                         algorithm_name, eta, alpha, memory_length))
                current_loss_list = []
                ans = 0
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
            current_algorithm_name_list.append(copy.deepcopy(algorithm_name))


def sum_up_param():
    mkdir("../performance/sum_up")
    mkdir("../performance/sum_up/eta")
    solver = excel_solver(file_path_import="../performance/sum_up/eta/eta_info.csv")
    solver.save_best_param(current_algorithm_name_list, current_dataset_name_list, current_best_eta_list,
                           current_best_loss_list)


if __name__ == '__main__':
    freeze_support()
    get_params()
    summary_csv()
    sum_up_param()
# generate_csv()
