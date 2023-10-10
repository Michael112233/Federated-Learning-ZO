#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
import csv
import time
import pandas as pd

class parameter:
    def __init__(self, max_grad_time, eta_type=1, eta=0.1, alpha=0.5, memory_length=5, batch_size=1000, verbose=True):
        self.eta_type = eta_type
        self.eta = eta
        self.alpha = alpha
        self.memory_length = memory_length
        self.verbose = verbose
        self.batch_size = 1000
        self.max_grad_time = max_grad_time
        self.client_rate = 0.1
        self.client_number = 100
        self.local_iteration = 20 # origin 500
        self.total_grad = 0
        self.iteration = 4000
        self.radius = 1e-6


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


def end_info(start_time, total_grad):
    end_time = time.time()
    print("total time is {:.3f}".format(end_time - start_time))
    print("total grad times is {:.2f}".format(total_grad))


class excel_solver:
    def __init__(self, file_path_import):
        file_path = "../performance/excel/"
        file_name = str(time.strftime('%Y-%m-%d-%H-%M-%S')) + ".csv"
        if file_path_import == "":
            self.file_path = file_path + file_name
        else:
            self.file_path = file_path_import

    def save_excel(self, current_time, current_grad_times, current_loss, current_round):
        # print(current_round)
        dataframe = pd.DataFrame(
            {'current_round': current_round, 'current_grad_times': current_grad_times, 'current_time': current_time, 'current_loss': current_loss})
        dataframe.to_csv(self.file_path, index=True)