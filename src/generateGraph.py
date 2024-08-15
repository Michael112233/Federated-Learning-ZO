from datetime import datetime
import math
import time

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from src.utils import find_latest_path, get_eta, get_file_name, mkdir

matplotlib.use('TkAgg')

dataset_name_list = ['rcv']
model_name_list = ["neural_network"]
iid_info_list = ['non_iid']

# dataset_name = "mnist"
# model_name = "neural_network"
# iid_info = "non_iid"  # non-iid -> 1, iid -> 0
for dataset_name in dataset_name_list:
    for model_name in model_name_list:
        for iid_info in iid_info_list:
            plt.clf()

            now = datetime.now()
            save_path = '../performance/graphs/{}/{}/{}/{}.jpg'.format(dataset_name, model_name, iid_info, now.strftime('%Y-%m-%d-%H-%M-%S'))

            mkdir("../performance/graphs")
            mkdir("../performance/graphs/{}".format(dataset_name))
            mkdir("../performance/graphs/{}/{}".format(dataset_name, model_name))
            mkdir("../performance/graphs/{}/{}/{}".format(dataset_name, model_name, iid_info))

            # FedAvg_GD
            fedavg_GD = pd.read_csv(get_file_name(dataset_name, model_name, 'FedAvg_GD', iid_info))
            fes1 = np.array(fedavg_GD['current_grad_times'])
            loss1 = np.array(fedavg_GD['current_loss'])
            plt.semilogy(fes1, loss1, label="fedavg_GD")

            # FedAvg_SGD
            fedavg_SGD = pd.read_csv(get_file_name(dataset_name, model_name, 'FedAvg_SGD', iid_info))
            # fedavg_SGD = pd.read_csv('../performance/experiment/fashion_mnist/FedAvg_SGD/neural_network/iid/eta=1/(time=2024-07-20-21-37-33).csv')
            fes5 = np.array(fedavg_SGD['current_grad_times'])
            loss5 = np.array(fedavg_SGD['current_loss'])
            plt.semilogy(fes5, loss5, label="fedavg_SGD")

            # Zeroth-order
            subspace = pd.read_csv(get_file_name(dataset_name, model_name, 'zeroth', iid_info))
            fes2 = np.array(subspace['current_grad_times'])
            loss2 = np.array(subspace['current_loss'])
            plt.semilogy(fes2, loss2, label="FedSGES")

            # fed
            fedzo = pd.read_csv(get_file_name(dataset_name, model_name, 'FedZO', iid_info))
            fes3 = np.array(fedzo['current_grad_times'])
            loss3 = np.array(fedzo['current_loss'])
            # fes3[0] += 1
            plt.semilogy(fes3, loss3, color="grey", label="fedzo")


            plt.xlabel("FES")
            plt.ylabel("loss")
            plt.title(model_name + " model, " + dataset_name + " dataset, " + iid_info)

            # plt.xlim(0, 0.2*1e8)

            plt.legend()
            plt.savefig(save_path)
