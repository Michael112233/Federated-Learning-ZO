from datetime import datetime
import math
import time

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
import pandas as pd
import numpy as np

from src.utils import find_latest_path, get_eta, get_file_name, mkdir

matplotlib.use('TkAgg')
font = FontProperties(size=20)

# dataset_name_list = ['rcv']
dataset_name_list = ['rcv']
model_name_list = ['svm']
# model_name_list = ['neural_network']
iid_info_list = ['non_iid']
width = 3
labelsize = 14
minn = 0.29
maxn = 1.6

# dataset_name = "mnist"
# model_name = "neural_network"
# iid_info = "non_iid"  # non-iid -> 1, iid -> 0
for dataset_name in dataset_name_list:
    for model_name in model_name_list:
        for iid_info in iid_info_list:
            plt.clf()

            # plt.figure(figsize=(8, 7))
            # plt.ylim(0.2, 0.71)
            plt.tight_layout(pad=1.0)

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
            plt.semilogy(fes1, loss1, label="FedAvg-GD", linewidth=width)
            # plt.plot(fes1, loss1, label="fedavg_GD")


            # FedAvg_SGD
            fedavg_SGD = pd.read_csv(get_file_name(dataset_name, model_name, 'FedAvg_SGD', iid_info))
            # fedavg_SGD = pd.read_csv('../performance/experiment/fashion_mnist/FedAvg_SGD/neural_network/iid/eta=1/(time=2024-07-20-21-37-33).csv')
            fes5 = np.array(fedavg_SGD['current_grad_times'])
            loss5 = np.array(fedavg_SGD['current_loss'])
            plt.semilogy(fes5, loss5, label="FedAvg-SGD", linewidth=width)
            # plt.plot(fes5, loss5, label="fedavg_SGD")

            # Zeroth-order
            subspace = pd.read_csv(get_file_name(dataset_name, model_name, 'zeroth', iid_info))
            print("Zeroth-order", subspace)
            fes2 = np.array(subspace['current_grad_times'])
            loss2 = np.array(subspace['current_loss'])
            plt.semilogy(fes2, loss2, label="ZoFedHT", linewidth=width)
            # plt.plot(fes2, loss2, label="fed***")

            # fed
            # fedzo = pd.read_csv(get_file_name(dataset_name, model_name, 'FedZO', iid_info))
            # print("Zeroth-order", fedzo)
            # fes3 = np.array(fedzo['current_grad_times'])
            # loss3 = np.array(fedzo['current_loss'])
            # # fes3[0] += 1
            # plt.semilogy(fes3, loss3, color="grey", label="FedZO", linewidth=width)
            # # plt.plot(fes3, loss3, color="grey", label="fedzo")

            plt.tick_params(axis='both', which='both', labelsize=labelsize)
            # plt.xlabel("Function evaluation times", font=font)
            # plt.ylabel("Loss", font=font)
            # plt.title(model_name + " model, " + dataset_name + " dataset, " + iid_info)

            # if dataset_name == 'fashion_mnist':
            plt.ylim([minn, maxn])
            # plt.yscale('log')

            plt.subplots_adjust(left=0.2,right=1)
            if model_name == 'logistic':
                plt.legend(fontsize=16)


            plt.savefig(save_path, bbox_inches='tight')
            # plt.show()
#