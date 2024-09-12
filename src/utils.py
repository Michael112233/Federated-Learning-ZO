#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
import csv
import os
import time

import numpy as np
import pandas as pd


def judge_whether_print(current_round):
    if current_round <= 10:
        return True
    elif current_round <= 200:
        return current_round % 10 == 0
    elif current_round <= 500:
        return current_round % 50 == 0
    elif current_round <= 2000:
        return current_round % 100 == 0
    elif current_round <= 4000:
        return current_round % 200 == 0
    else:
        return current_round % 500 == 0


# 参数类，用于传递超参数以及超参数集中化
hidden_layer = 50
class parameter:
    def __init__(self, max_grad_time, eta_type=1, eta=0.1, alpha=0.5, memory_length=5, batch_size=1000,
                 verbose=True, sample_kind=0):
        self.eta_type = eta_type
        self.eta = eta
        self.alpha = alpha
        self.memory_length = memory_length
        self.verbose = verbose
        self.batch_size = batch_size
        self.max_grad_time = max_grad_time
        self.client_rate = 0.1
        self.client_number = 100
        self.local_iteration = 50  # origin 50, 10
        self.total_grad = 0
        self.iteration = 30000
        self.radius = 1e-4
        self.sample_kind = sample_kind  # 0 for iid, 1 for non_iid


# 规定了eta的计算方式
class eta_class:
    def divide_eta(self, eta, iter, local_iter):
        return eta / ((iter + 1) * (local_iter + 1))

    def sqrt_eta(self, eta, iter, local_iter=0):
        return eta / math.sqrt((iter + 1))

    def same_eta(self, eta, iter=0, local_iter=0):
        return eta

    def choose(self, option):
        if option > 3 or option < 0:
            option = 1
        if option == 1:
            return self.same_eta
        if option == 2:
            return self.sqrt_eta
        if option == 3:
            return self.divide_eta


def get_eta(dataset_name, model_name, algorithm_name, sample_kind):
    eta_file = pd.read_csv('../performance/sum_up/eta/eta_info.csv')
    eta_infos = np.array(eta_file)
    for eta_info in eta_infos:
        cur_dataset = eta_info[2]
        cur_algorithm = eta_info[1]
        cur_model = eta_info[3]
        cur_eta = eta_info[4]
        cur_sample_kind = num2kind(eta_info[6])
        # print(eta_info)
        if dataset_name == cur_dataset and cur_algorithm == algorithm_name and cur_model == model_name and cur_sample_kind == sample_kind:
            return cur_eta
    return -1

def get_file_name(dataset_name, model_name, algorithm_name, sample_kind):
    csv_path = "../performance/experiment/{}/{}/{}/{}".format(dataset_name, algorithm_name, model_name, sample_kind)
    print(os.walk(csv_path))
    min_eta = 1000000
    min_path = ""
    for subdir, dirs, files in os.walk(csv_path):
        # print(subdir, dirs, files)
        for file in files:
            if file.endswith('.csv'):
                # 构建文件的完整路径
                file_path = os.path.join(subdir, file)
                print(file_path)
                # 读取CSV文件
                data = pd.read_csv(file_path)
                # 获取最后一列的数据
                last_column = data.iloc[:, -1]
                print(last_column[len(last_column)-1])
                if last_column[len(last_column)-1] < min_eta:
                    min_eta = last_column[len(last_column)-1]
                    min_path = file_path
                print(file_path, min_eta)
    # print(min_path)
    print(csv_path)

    return min_path

# 单次实验最终结果的输出
def end_info(start_time, total_grad):
    end_time = time.time()
    print("total time is {:.3f}".format(end_time - start_time))
    print("total grad times is {:.2f}".format(total_grad))


# 放置一些csv存储相关的代码
class excel_solver:
    def __init__(self, file_path_import=""):
        file_path = "../performance/excel/"
        file_name = str(time.strftime('%Y-%m-%d-%H-%M-%S')) + ".csv"
        if file_path_import == "":
            self.file_path = file_path + file_name
        else:
            self.file_path = file_path_import

    def save_excel(self, current_time, current_grad_times, current_loss, current_round):
        # print(current_round)
        dataframe = pd.DataFrame(
            {'current_round': current_round, 'current_grad_times': current_grad_times, 'current_time': current_time,
             'current_loss': current_loss})
        dataframe.to_csv(self.file_path, index=True)

    def save_best_param(self, algorithm, dataset, best_eta, best_loss, sample_kind, model_name, filename):
        dataframe = pd.DataFrame(
            {'algorithm': algorithm, 'dataset': dataset, 'model_name': model_name, 'best_eta': best_eta,
             'best_loss': best_loss, 'sample_kind': sample_kind, 'file_name': filename})
        dataframe.to_csv(self.file_path, index=True)


# 工作流调参步骤，用于新建文件夹
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


# mode = 1 -> experiment
# mode = 0 -> get params
def make_dir(dataset_name, algorithm_name, model_name, params, mode):
    eta = params.eta
    alpha = params.alpha
    memory_length = params.memory_length
    print(params.sample_kind)
    if params.sample_kind == 0:
        sample_name = "iid"
    else:
        sample_name = "non_iid"
    dir_name = ""
    if mode == 0:
        dir_name = "params"
    elif mode == 1:
        dir_name = "experiment"
    else:
        return
    print('-----mkdir')
    mkdir("../performance")
    mkdir("../performance/{}".format(dir_name))
    mkdir("../performance/{}/{}".format(dir_name, dataset_name))
    mkdir("../performance/{}/{}/{}".format(dir_name, dataset_name, algorithm_name))
    mkdir("../performance/{}/{}/{}/{}".format(dir_name, dataset_name, algorithm_name, model_name))
    mkdir("../performance/{}/{}/{}/{}/{}".format(dir_name, dataset_name, algorithm_name, model_name, sample_name))
    mkdir("../performance/{}/{}/{}/{}/{}/eta={}".format(dir_name, dataset_name, algorithm_name, model_name, sample_name,
                                                        eta))
    if mode == 0 and (algorithm_name == "zeroth" or algorithm_name == "zeroth_grad"):
        # print(alpha)
        mkdir("../performance/{}/{}/{}/{}/{}/eta={}/alpha={:.2}".format(dir_name, dataset_name, algorithm_name,
                                                                        model_name, sample_name,
                                                                        eta, float(alpha)))
        mkdir(
            "../performance/{}/{}/{}/{}/{}/eta={}/alpha={:.2}/memory_length={}".format(dir_name, dataset_name,
                                                                                       algorithm_name, model_name,
                                                                                       sample_name,
                                                                                       eta, float(alpha), memory_length))


import os


def find_latest_path(folder):
    latest_file = None
    latest_time = 0
    print(folder)
    for root, dirs, files in os.walk(folder):
        print(files)
        for file in files:
            file_path = os.path.join(root, file)
            # 获取文件的创建时间
            file_creation_time = os.path.getctime(file_path)
            # 检查是否是最晚创建的文件
            if file_creation_time > latest_time:
                latest_time = file_creation_time
                latest_file = file_path

    return latest_file


def num2kind(number):
    if number == 1:
        return "non_iid"
    else:
        return "iid"
