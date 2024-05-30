import time
import os
import copy
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd

# from multiprocessing import Pool, freeze_support
from sampling import get_rcv1, get_mnist, get_cifar10, get_fashion_mnist
from algorithm import FedAvg_SGD, Zeroth_grad, FedAvg_GD, FedAvg_SIGNSGD, FedZO
from utils import excel_solver, parameter, eta_class, mkdir, make_dir

current_dataset_name_list = []
current_algorithm_name_list = []
current_best_eta_list = []
current_best_loss_list = []
current_best_sample_list = []
current_best_model_list = []

dir_mode = 0  # means "performance/params"


def get_result(filename, algorithm):
    csv_solver = excel_solver(filename)
    start_time = time.time()
    current_time, current_grad_times, current_loss, current_round = algorithm.alg_run(start_time)
    # print("{}\n".format(filename))
    csv_solver.save_excel(current_time, current_grad_times, current_loss, current_round)


eta_list = [1e-3, 1e-2, 1e-1, 1, 10, 100]
alpha_list = [0.3]
model_name_list = ['logistic', 'svm']
sample_kind_list = [0, 1]
dataset_list = ['cifar10', 'rcv', 'mnist', 'fashion_mnist']
algorithm_list = ['FedAvg_SGD', 'zeroth', 'FedAvg_GD', 'FedZO']  #'zeroth', 'FedAvg_SGD', 'FedAvg_GD', 'FedZO'
memory_length_list = [5]
times_list = range(1, 4)
verbose = True
batch_size = 64

max_grad_time_rcv = 162578000
max_grad_time_mnist = 14400000
eta_choose = eta_class()
eta_type = eta_choose.choose(2)


def generate_csv(dataset_name, algorithm_name, model_name, eta, times, alpha, memory_length, sample_kind):
    if dataset_name == 'rcv':
        dataset, X, Y, global_model = get_rcv1(model_name)
    elif dataset_name == 'cifar10':
        dataset, X, Y, global_model = get_cifar10(model_name)
    elif dataset_name == 'fashion_mnist':
        dataset, X, Y, global_model = get_fashion_mnist(model_name)
    else:
        dataset, X, Y, global_model = get_mnist(model_name)
    max_grad_time = 500 * dataset.length()
    if sample_kind == 0:
        sample_name = "iid"
    else:
        sample_name = "non_iid"

    # for algorithm
    if algorithm_name == 'FedAvg_SGD':
        filename = "../performance/params/{}/{}/{}/{}/eta={}/({}).csv".format(
            dataset_name, algorithm_name, model_name, sample_name, eta, times)
        # if(os.path.exists(filename)):
        #     return
        print(filename)
        if dataset_name == "mnist" or dataset_name == 'fashion_mnist':
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 64, verbose, sample_kind)
        else:
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, verbose, sample_kind)
        make_dir(dataset_name, algorithm_name, model_name, para, dir_mode)
        algorithm = FedAvg_SGD(dataset, global_model, para)
        get_result(filename, algorithm)
    elif algorithm_name == 'FedAvg_GD':
        filename = "../performance/params/{}/{}/{}/{}/eta={}/({}).csv".format(
            dataset_name, algorithm_name, model_name, sample_name, eta, times)
        print(filename)
        # if (os.path.exists(filename)):
        #     return
        if dataset_name == "mnist" or dataset_name == 'fashion_mnist':
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 64, verbose, sample_kind)
        else:
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, verbose, sample_kind)
        make_dir(dataset_name, algorithm_name, model_name, para, dir_mode)
        algorithm = FedAvg_GD(dataset, global_model, para)
        get_result(filename, algorithm)
    elif algorithm_name == 'zeroth':
        filename = "../performance/params/{}/{}/{}/{}/eta={}/alpha={:.2}/memory_length={}/({}).csv".format(
            dataset_name,
            algorithm_name, model_name, sample_name, eta, float(alpha),
            memory_length,
            times)
        # if (os.path.exists(filename)):
        #     return
        if dataset_name == "mnist" or dataset_name == "fashion_mnist":
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 64,
                             verbose, sample_kind)
        else:
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000,
                             verbose, sample_kind)
        make_dir(dataset_name, algorithm_name, model_name, para, dir_mode)
        print(filename)
        algorithm = Zeroth_grad(dataset, global_model, para)
        get_result(filename, algorithm)
    elif algorithm_name == 'FedAvg_SignSGD':
        filename = "../performance/params/{}/{}/{}/eta={}/({}).csv".format(
            dataset_name,
            algorithm_name, model_name, eta,
            times)
        if dataset_name == "mnist" or dataset_name == "fashion_mnist":
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 64,
                             verbose)
        else:
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000,
                             verbose)
        # make_dir(dataset_name, algorithm_name, model_name, para, dir_mode)
        print(filename)
        algorithm = FedAvg_SIGNSGD(dataset, global_model, para)
        get_result(filename, algorithm)
    elif algorithm_name == 'FedZO':
        filename = "../performance/params/{}/{}/{}/{}/eta={}/({}).csv".format(
            dataset_name, algorithm_name, model_name, sample_name, eta, times)
        if (os.path.exists(filename)):
            return
        if dataset_name == "mnist" or dataset_name == "fashion_mnist":
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 64,
                             verbose, sample_kind)
        else:
            para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000,
                             verbose, sample_kind)
        make_dir(dataset_name, algorithm_name, model_name, para, dir_mode)
        print(filename)
        algorithm = FedZO(dataset, global_model, para)
        get_result(filename, algorithm)


def get_eta_params():
    start_time = time.time()
    combine = []
    for sample_kind in sample_kind_list:
        for dataset_name in dataset_list:
            for algorithm_name in algorithm_list:
                for model_name in model_name_list:
                    for eta in eta_list:
                        if algorithm_name == 'zeroth':
                            for memory_length in memory_length_list:
                                for alpha in alpha_list:
                                    for times in times_list:
                                        print(dataset_name, algorithm_name, model_name, eta, times, alpha, memory_length, sample_kind)
                                        # generate_csv(dataset_name, algorithm_name, model_name, eta, times, alpha, memory_length)
                                        combine.append([dataset_name, algorithm_name, model_name, eta, times, alpha, memory_length, sample_kind])
                        else:
                            for times in times_list:
                                print(dataset_name, algorithm_name, model_name, eta, times, 0, 0, sample_kind)
                                # generate_csv(dataset_name, algorithm_name, model_name, eta, times, alpha, memory_length)
                                combine.append([dataset_name, algorithm_name, model_name, eta, times, 0, 0, sample_kind])

    with Pool(10) as p:
        # ans = p.starmap(generate_csv, zip(dataset_list, algorithm_list, eta_list, times_list))
        ans = p.starmap(generate_csv, combine)
        print(ans)

    end_time = time.time()
    print("Time is {}s".format(end_time - start_time))


def get_zeroth_params():
    combine = []
    algorithm_name = 'zeroth'
    for sample_kind in sample_kind_list:
        for dataset_name in dataset_list:
            for model_name in model_name_list:
                for eta in eta_list:
                    for memory_length in memory_length_list:
                        for alpha in alpha_list:
                            for times in times_list:
                                print(dataset_name, algorithm_name, model_name, eta, times, alpha,
                                      memory_length, sample_kind)
                                combine.append(
                                    [dataset_name, algorithm_name, model_name, eta, times, alpha, memory_length,
                                     sample_kind])
    with Pool(10) as p:
        # ans = p.starmap(generate_csv, zip(dataset_list, algorithm_list, eta_list, times_list))
        ans = p.starmap(generate_csv, combine)
        print(ans)

def summary_csv():
    print("----")
    for sample_kind in sample_kind_list:
        for dataset_name in dataset_list:
            for algorithm_name in algorithm_list:
                for model_name in model_name_list:
                    best_loss = 1000000
                    best_eta = -1
                    g = 0
                    current_loss_list = []
                    for eta in eta_list:
                        if sample_kind == 1:
                            sample_name = 'non_iid'
                        else:
                            sample_name = 'iid'
                        if algorithm_name == "zeroth":
                            for alpha in alpha_list:
                                for memory_length in memory_length_list:
                                    g = os.walk(r"../performance/params/{}/{}/{}/{}/eta={}/alpha={:.2}/memory_length={}".format(
                                                dataset_name,
                                                algorithm_name,
                                                model_name, sample_name,
                                                eta, float(alpha),
                                                memory_length))
                        else:
                            g = os.walk(r"../performance/params/{}/{}/{}/{}/eta={}".format(dataset_name, algorithm_name, model_name, sample_name, eta))
                        for path, dir_list, file_list in g:
                            # print(path, dir_list, file_list)
                            for file_name in file_list:
                                csv_path = (os.path.join(path, file_name))
                                csv_file = pd.read_csv(csv_path)
                                current_loss_column = np.array(csv_file["current_loss"])
                                current_loss_list.append(current_loss_column[-1])
                        if len(current_loss_list) == 0:
                            continue
                        sorted_list = np.sort(current_loss_list)
                        # print(sorted_list)
                        loss = sorted_list[max(times_list) // 2]

                        if best_loss > loss:
                            best_loss = loss
                            best_eta = eta

                        # print("{} {} eta={} memory_length={} alpha={} loss is {}\n".format(dataset_name, algorithm_name,
                        #                                                            best_eta, memory_length,
                        #                                                            alpha, best_loss))
                    current_best_eta_list.append(copy.deepcopy(best_eta))
                    current_best_loss_list.append(copy.deepcopy(best_loss))
                    current_best_model_list.append(copy.deepcopy(model_name))
                    current_best_sample_list.append(copy.deepcopy(sample_kind))
                    current_dataset_name_list.append(copy.deepcopy(dataset_name))
                    current_algorithm_name_list.append(copy.deepcopy(algorithm_name))


def sum_up_param():
    mkdir("../performance/sum_up")
    mkdir("../performance/sum_up/eta")
    solver = excel_solver(file_path_import="../performance/sum_up/eta/eta_info.csv")
    solver.save_best_param(current_algorithm_name_list, current_dataset_name_list, current_best_eta_list,
                           current_best_loss_list, current_best_sample_list, current_best_model_list)


if __name__ == '__main__':
    freeze_support()
    # get_eta_params()
    # get_zeroth_params()
    summary_csv()
    sum_up_param()
# generate_csv()
