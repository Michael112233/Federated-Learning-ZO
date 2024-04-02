import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils import find_latest_path

dataset_name = "fashion_mnist"
model_name = "svm"

# ./performance/experiment/rcv/FedAvg_GD/logistic/eta=10/(time=2023-11-27-23-32-30).csv

fedavg_GD = pd.read_csv(find_latest_path("../performance/experiment/" + dataset_name + "/FedAvg_GD/" + model_name + "/"))
fes1 = np.array(fedavg_GD['current_grad_times'])
loss1 = np.array(fedavg_GD['current_loss'])
# fes1[0] += 1
plt.semilogy(fes1, loss1, label="fedavg_GD")

# ./performance/experiment/rcv/FedAvg_SGD/logistic/eta=10/(time=2023-11-27-23-32-04).csv
fedavg_SGD = pd.read_csv(find_latest_path("../performance/experiment/" + dataset_name + "/FedAvg_SGD/" + model_name + "/"))

fes5 = np.array(fedavg_SGD['current_grad_times'])
loss5 = np.array(fedavg_SGD['current_loss'])
# fes5[0] += 1
plt.semilogy(fes5, loss5, label="fedavg_SGD")

# ./performance/experiment/rcv/zeroth_grad/logistic/eta=25/(time=2023-11-27-23-31-38).csv
subspace = pd.read_csv(find_latest_path("../performance/experiment/" + dataset_name + "/zeroth_grad/" + model_name + "/"))
fes2 = np.array(subspace['current_grad_times'])
loss2 = np.array(subspace['current_loss'])
# fes2[0] += 1
plt.semilogy(fes2, loss2, label="FedSGES")

# ./performance/experiment/rcv/FedZO/logistic/eta=50/(time=2023-11-27-23-32-52).csv
fedzo = pd.read_csv(find_latest_path("../performance/experiment/" + dataset_name + "/FedZO/" + model_name + "/"))
fes3 = np.array(fedzo['current_grad_times'])
loss3 = np.array(fedzo['current_loss'])
# fes3[0] += 1
plt.semilogy(fes3, loss3, color="grey", label="fedzo")


plt.xlabel("FES")
plt.ylabel("loss")
plt.title(model_name + " model, " + dataset_name + " dataset")

# plt.xlim(0, 0.2*1e8)

plt.legend()
plt.show()

